//! CPU mesh processing for sparse-volume exports.
//!
//! Edge costs, normal-cone clustering and narrow-band dual contouring follow
//! CuMesh (12289e1062f0603f2f0d0771b02e1395d247f26f). Execution and atlas packing
//! are portable Rust; no CUDA, OpenCV or native atlas library is required.

use super::{cross, dot, length, normalized, sub, GeometryError, TriangleSurface};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Normal-cone clustering controls used before conformal parameterization.
#[derive(Clone, Debug)]
pub struct ChartOptions {
    /// Maximum merge cost, in radians, including area and compactness penalties.
    pub threshold_cone_half_angle_rad: f32,
    /// Boundary refinement passes per global iteration.
    pub refine_iterations: usize,
    /// Cone recomputation / clustering iterations.
    pub global_iterations: usize,
    /// Neighbor edge-length weight during boundary refinement.
    pub smooth_strength: f32,
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self {
            threshold_cone_half_angle_rad: std::f32::consts::FRAC_PI_2,
            refine_iterations: 100,
            global_iterations: 3,
            smooth_strength: 1.0,
        }
    }
}

/// Domain and unsigned-distance isosurface controls for dual contouring.
#[derive(Clone, Copy, Debug)]
pub struct RemeshOptions {
    /// Number of cells along each domain axis.
    pub resolution: usize,
    /// World-space center of the cubic domain.
    pub center: [f32; 3],
    /// World-space side length of the cubic domain.
    pub scale: f32,
    /// Isosurface distance in voxel units.
    pub band: f32,
    /// Blend extracted positions toward their closest original surface points.
    pub project_back: f32,
}

type Edges = BTreeMap<(u32, u32), Vec<(usize, usize)>>;

fn edges(surface: &TriangleSurface) -> Edges {
    let mut result = Edges::new();
    for (i, face) in surface.faces.iter().enumerate() {
        for corner in 0..3 {
            let (a, b) = (face[corner], face[(corner + 1) % 3]);
            result
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push((i, corner));
        }
    }
    result
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    parent[a.max(b)] = a.min(b);
}

pub(super) fn compact(surface: &mut TriangleSurface) -> Result<(), GeometryError> {
    let mut map = vec![None; surface.vertices.len()];
    for face in &surface.faces {
        for &v in face {
            map[v as usize] = Some(0);
        }
    }
    let mut vertices = Vec::new();
    for (v, index) in map.iter_mut().enumerate() {
        if index.is_some() {
            *index = Some(
                u32::try_from(vertices.len())
                    .map_err(|_| GeometryError::Invalid("surface index capacity"))?,
            );
            vertices.push(surface.vertices[v]);
        }
    }
    for face in &mut surface.faces {
        for v in face {
            *v = map[*v as usize].ok_or(GeometryError::Invalid("unreferenced index"))?;
        }
    }
    surface.vertices = vertices;
    Ok(())
}

fn face_normal(surface: &TriangleSurface, face: [u32; 3]) -> [f32; 3] {
    let p = face.map(|v| surface.vertices[v as usize]);
    cross(sub(p[1], p[0]), sub(p[2], p[0]))
}

pub(super) fn remove_degenerate(surface: &mut TriangleSurface) {
    let vertices = &surface.vertices;
    surface.faces.retain(|f| {
        let p = f.map(|v| vertices[v as usize]);
        let a = sub(p[1], p[0]);
        let b = sub(p[2], p[0]);
        let c = sub(p[2], p[1]);
        let area = 0.5 * length(cross(a, b));
        area > 1e-24 && area > 1e-12 * dot(a, a).max(dot(b, b)).max(dot(c, c))
    });
}

// Reconstruct corner equivalence through manifold edges, rather than deleting
// non-manifold incident faces. This is CuMesh's repair_non_manifold_edges contract.
/// Remove duplicate faces, split non-manifold vertex fans, remove
/// small components by world-space area, compact indices and cap short holes.
/// # Errors
/// Reports invalid surface buffers, thresholds or index capacity.
pub fn cleanup_mesh(
    surface: &mut TriangleSurface,
    min_component_area: f32,
    max_hole_perimeter: f32,
) -> Result<(), GeometryError> {
    surface.validate()?;
    if !min_component_area.is_finite()
        || min_component_area < 0.0
        || !max_hole_perimeter.is_finite()
        || max_hole_perimeter < 0.0
    {
        return Err(GeometryError::Invalid("mesh cleanup thresholds"));
    }
    let mut seen = BTreeSet::new();
    surface.faces.retain(|f| {
        let mut key = *f;
        key.sort_unstable();
        seen.insert(key)
    });
    let corners = surface
        .faces
        .len()
        .checked_mul(3)
        .ok_or(GeometryError::Invalid("corner capacity"))?;
    let mut parent = (0..corners).collect::<Vec<_>>();
    for ((a, b), incident) in edges(surface) {
        if let [(f, _), (g, _)] = incident.as_slice() {
            for vertex in [a, b] {
                let i = surface.faces[*f]
                    .iter()
                    .position(|&v| v == vertex)
                    .ok_or(GeometryError::Invalid("edge corner"))?;
                let j = surface.faces[*g]
                    .iter()
                    .position(|&v| v == vertex)
                    .ok_or(GeometryError::Invalid("edge corner"))?;
                join(&mut parent, 3 * f + i, 3 * g + j);
            }
        }
    }
    let mut ids = BTreeMap::new();
    let mut vertices = Vec::new();
    for (i, face) in surface.faces.iter_mut().enumerate() {
        for (j, vertex) in face.iter_mut().enumerate() {
            let key = root(&mut parent, i * 3 + j);
            let id = if let Some(&id) = ids.get(&key) {
                id
            } else {
                let id = u32::try_from(vertices.len())
                    .map_err(|_| GeometryError::Invalid("repaired vertex capacity"))?;
                vertices.push(surface.vertices[*vertex as usize]);
                ids.insert(key, id);
                id
            };
            *vertex = id;
        }
    }
    surface.vertices = vertices;
    let mut parent = (0..surface.faces.len()).collect::<Vec<_>>();
    for incident in edges(surface).values() {
        if let [(a, _), (b, _)] = incident.as_slice() {
            join(&mut parent, *a, *b);
        }
    }
    let mut areas = vec![0.0; surface.faces.len()];
    for (i, &face) in surface.faces.iter().enumerate() {
        areas[root(&mut parent, i)] += 0.5 * length(face_normal(surface, face));
    }
    let keep = (0..surface.faces.len())
        .map(|i| areas[root(&mut parent, i)] >= min_component_area)
        .collect::<Vec<_>>();
    let mut i = 0;
    surface.faces.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
    compact(surface)?;
    surface.fill_holes(max_hole_perimeter)
}

/// Orient each manifold-connected face component consistently with its seed face.
/// # Errors
/// Reports invalid surface buffers.
pub fn unify_face_orientations(surface: &mut TriangleSurface) -> Result<(), GeometryError> {
    surface.validate()?;
    let mut adjacency = vec![Vec::new(); surface.faces.len()];
    for incident in edges(surface).values() {
        if let [(a, i), (b, j)] = incident.as_slice() {
            let same = surface.faces[*a][*i] == surface.faces[*b][*j];
            adjacency[*a].push((*b, same));
            adjacency[*b].push((*a, same));
        }
    }
    let mut flips = vec![None; surface.faces.len()];
    for seed in 0..surface.faces.len() {
        if flips[seed].is_some() {
            continue;
        }
        flips[seed] = Some(false);
        let mut queue = VecDeque::from([seed]);
        while let Some(i) = queue.pop_front() {
            for &(j, same) in &adjacency[i] {
                if flips[j].is_none() {
                    flips[j] = Some(flips[i].unwrap_or(false) ^ same);
                    queue.push_back(j);
                }
            }
        }
    }
    for (face, flip) in surface.faces.iter_mut().zip(flips) {
        if flip == Some(true) {
            face.swap(1, 2);
        }
    }
    Ok(())
}

// CuMesh midpoint/boundary-preserving QEM + edge-length + skinny-triangle cost.
// A batch collapses only edges that minimize the cost throughout their face stars.
/// Simplify toward a triangle-count target using boundary-preserving QEM edge costs.
/// # Errors
/// Reports invalid buffers, a zero target or exhaustion of nonflipping collapses.
pub fn simplify_mesh(surface: &mut TriangleSurface, target: usize) -> Result<(), GeometryError> {
    surface.validate()?;
    if target == 0 {
        return Err(GeometryError::Invalid("zero simplification target"));
    }
    let mut threshold = 1e-8_f32;
    while surface.faces.len() > target {
        let edge_map = edges(surface);
        let mut incident = vec![Vec::new(); surface.vertices.len()];
        let mut quadrics = vec![[[0.0_f32; 4]; 4]; surface.vertices.len()];
        for (i, &face) in surface.faces.iter().enumerate() {
            let n = normalized(face_normal(surface, face));
            let plane = [
                n[0],
                n[1],
                n[2],
                -dot(n, surface.vertices[face[0] as usize]),
            ];
            for &v in &face {
                incident[v as usize].push(i);
                for a in 0..4 {
                    for b in 0..4 {
                        quadrics[v as usize][a][b] += plane[a] * plane[b];
                    }
                }
            }
        }
        let mut boundary = vec![false; surface.vertices.len()];
        for (&(a, b), faces) in &edge_map {
            if faces.len() == 1 {
                boundary[a as usize] = true;
                boundary[b as usize] = true;
            }
        }
        let mut candidates = Vec::new();
        let mut minimum = vec![(f32::INFINITY, usize::MAX); surface.faces.len()];
        for &(a, b) in edge_map.keys() {
            let (a, b) = (a as usize, b as usize);
            let p = if boundary[a] && !boundary[b] {
                surface.vertices[a]
            } else if boundary[b] && !boundary[a] {
                surface.vertices[b]
            } else {
                std::array::from_fn(|i| 0.5 * surface.vertices[a][i] + 0.5 * surface.vertices[b][i])
            };
            let h = [p[0], p[1], p[2], 1.0];
            let mut cost = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    cost += h[i] * (quadrics[a][i][j] + quadrics[b][i][j]) * h[j];
                }
            }
            let delta = sub(surface.vertices[a], surface.vertices[b]);
            let edge_length2 = dot(delta, delta);
            cost += 1e-2 * edge_length2;
            let mut skinny = 0.0;
            let mut count = 0;
            for &(v, other) in &[(a, b), (b, a)] {
                for &fid in &incident[v] {
                    let face = surface.faces[fid];
                    if face.contains(&(other as u32)) {
                        continue;
                    }
                    let old_normal = face_normal(surface, face);
                    let points = face.map(|id| {
                        if id as usize == v {
                            p
                        } else {
                            surface.vertices[id as usize]
                        }
                    });
                    let n = cross(sub(points[1], points[0]), sub(points[2], points[0]));
                    if dot(old_normal, n) < 0.0 {
                        cost = f32::INFINITY;
                        break;
                    }
                    let e = [
                        sub(points[1], points[0]),
                        sub(points[2], points[0]),
                        sub(points[2], points[1]),
                    ];
                    let denom = e.iter().map(|&e| dot(e, e)).sum::<f32>().max(1e-12);
                    skinny += 1.0 - (2.0 * 3.0_f32.sqrt() * length(n) / denom).clamp(0.0, 1.0);
                    count += 1;
                }
            }
            if count > 0 {
                cost += 1e-3 * skinny / count as f32 * edge_length2;
            }
            // A quadric can acquire tiny negative round-off; packed CuMesh costs
            // are positive float keys. Clamp round-off at the same zero boundary.
            cost = cost.max(0.0);
            let id = candidates.len();
            for &f in incident[a].iter().chain(&incident[b]) {
                if cost
                    .total_cmp(&minimum[f].0)
                    .then(id.cmp(&minimum[f].1))
                    .is_lt()
                {
                    minimum[f] = (cost, id);
                }
            }
            candidates.push((a, b, p, cost));
        }
        let before = surface.faces.len();
        let mut map = (0..surface.vertices.len()).collect::<Vec<_>>();
        let mut collapsed = 0;
        for (id, &(a, b, p, cost)) in candidates.iter().enumerate() {
            if cost.is_finite()
                && cost <= threshold
                && incident[a]
                    .iter()
                    .chain(&incident[b])
                    .all(|&f| minimum[f].1 == id)
            {
                surface.vertices[a] = p;
                map[b] = a;
                collapsed += 1;
            }
        }
        for face in &mut surface.faces {
            for v in face {
                *v = map[*v as usize] as u32;
            }
        }
        surface
            .faces
            .retain(|f| f[0] != f[1] && f[1] != f[2] && f[2] != f[0]);
        if collapsed == 0 && !candidates.iter().any(|c| c.3.is_finite()) {
            return Err(GeometryError::Invalid(
                "no nonflipping simplification remains",
            ));
        }
        if before - surface.faces.len() < before.div_ceil(100) {
            threshold *= 10.0;
        }
        compact(surface)?;
    }
    Ok(())
}

struct Node {
    low: [f32; 3],
    high: [f32; 3],
    children: Option<(usize, usize)>,
    start: usize,
    end: usize,
}

/// Closest-point acceleration over an immutable original surface.
pub(super) struct SurfaceBvh<'a> {
    surface: &'a TriangleSurface,
    faces: Vec<usize>,
    nodes: Vec<Node>,
}

impl<'a> SurfaceBvh<'a> {
    pub(super) fn new(surface: &'a TriangleSurface) -> Result<Self, GeometryError> {
        surface.validate()?;
        if surface.faces.is_empty() {
            return Err(GeometryError::Invalid("empty projection surface"));
        }
        let mut result = Self {
            surface,
            faces: (0..surface.faces.len()).collect(),
            nodes: Vec::new(),
        };
        result.build(0, result.faces.len());
        Ok(result)
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for &f in &self.faces[start..end] {
            for &v in &self.surface.faces[f] {
                for axis in 0..3 {
                    low[axis] = low[axis].min(self.surface.vertices[v as usize][axis]);
                    high[axis] = high[axis].max(self.surface.vertices[v as usize][axis]);
                }
            }
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            low,
            high,
            children: None,
            start,
            end,
        });
        if end - start > 8 {
            let axis = (0..3)
                .max_by(|&a, &b| (high[a] - low[a]).total_cmp(&(high[b] - low[b])))
                .unwrap_or(0);
            let middle = (start + end) / 2;
            let surface = self.surface;
            self.faces[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
                let center = |f: usize| {
                    surface.faces[f]
                        .iter()
                        .map(|&v| surface.vertices[v as usize][axis] / 3.0)
                        .sum::<f32>()
                };
                center(a).total_cmp(&center(b)).then(a.cmp(&b))
            });
            let left = self.build(start, middle);
            let right = self.build(middle, end);
            self.nodes[id].children = Some((left, right));
        }
        id
    }

    pub(super) fn closest(&self, point: [f32; 3]) -> ([f32; 3], f32) {
        let mut best = (point, f32::INFINITY);
        let mut stack = vec![0];
        let bound = |node: &Node| {
            (0..3)
                .map(|i| {
                    let d = (node.low[i] - point[i])
                        .max(0.0)
                        .max(point[i] - node.high[i]);
                    d * d
                })
                .sum::<f32>()
        };
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if bound(node) > best.1 {
                continue;
            }
            if let Some((a, b)) = node.children {
                if bound(&self.nodes[a]) < bound(&self.nodes[b]) {
                    stack.extend([b, a]);
                } else {
                    stack.extend([a, b]);
                }
            } else {
                for &f in &self.faces[node.start..node.end] {
                    let triangle = self.surface.faces[f].map(|v| self.surface.vertices[v as usize]);
                    let p = closest_triangle(point, triangle);
                    let d = sub(p, point);
                    let distance = dot(d, d);
                    if distance < best.1 {
                        best = (p, distance);
                    }
                }
            }
        }
        (best.0, best.1.sqrt())
    }
}

fn closest_triangle(p: [f32; 3], t: [[f32; 3]; 3]) -> [f32; 3] {
    // Voronoi regions of a triangle (including degenerate edges).
    let ab = sub(t[1], t[0]);
    let ac = sub(t[2], t[0]);
    let ap = sub(p, t[0]);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return t[0];
    }
    let bp = sub(p, t[1]);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return t[1];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(f32::MIN_POSITIVE);
        return std::array::from_fn(|i| t[0][i] + v * ab[i]);
    }
    let cp = sub(p, t[2]);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return t[2];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(f32::MIN_POSITIVE);
        return std::array::from_fn(|i| t[0][i] + w * ac[i]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(f32::MIN_POSITIVE);
        return std::array::from_fn(|i| t[1][i] + w * (t[2][i] - t[1][i]));
    }
    let denom = (va + vb + vc).max(f32::MIN_POSITIVE);
    std::array::from_fn(|i| t[0][i] + (vb * ab[i] + vc * ac[i]) / denom)
}

/// Extract the unsigned-distance band isosurface with sparse mean-intersection
/// dual contouring, then optionally project vertices toward the input surface.
/// # Errors
/// Reports invalid domain/options, incompatible subdivision resolution or capacity.
pub fn remesh_narrow_band_dc(
    surface: &TriangleSurface,
    options: &RemeshOptions,
) -> Result<TriangleSurface, GeometryError> {
    let bvh = SurfaceBvh::new(surface)?;
    let RemeshOptions {
        resolution,
        center,
        scale,
        band,
        project_back: project,
    } = *options;
    if resolution == 0
        || resolution > i32::MAX as usize
        || center.iter().any(|v| !v.is_finite())
        || !scale.is_finite()
        || scale <= 0.0
        || !band.is_finite()
        || band <= 0.0
        || !project.is_finite()
    {
        return Err(GeometryError::Invalid("remeshing domain or options"));
    }
    let mut base = resolution;
    while base > 32 {
        if base % 2 != 0 {
            return Err(GeometryError::Invalid(
                "remesh resolution must halve to at most 32",
            ));
        }
        base /= 2;
    }
    let eps = band * scale / resolution as f32;
    if !eps.is_finite() {
        return Err(GeometryError::Invalid("remesh distance overflow"));
    }
    let mut cells = Vec::new();
    for x in 0..base {
        for y in 0..base {
            for z in 0..base {
                cells.push([x, y, z]);
            }
        }
    }
    loop {
        let cell_size = scale / base as f32;
        cells.retain(|c| {
            let p = std::array::from_fn(|i| {
                ((c[i] as f32 + 0.5) / base as f32 - 0.5) * scale + center[i]
            });
            (bvh.closest(p).1 - eps).abs() < 0.87 * cell_size
        });
        if base == resolution {
            break;
        }
        base *= 2;
        let mut children = Vec::with_capacity(
            cells
                .len()
                .checked_mul(8)
                .ok_or(GeometryError::Invalid("remesh cell capacity"))?,
        );
        for c in cells {
            for child in 0..8 {
                children.push(std::array::from_fn(|i| c[i] * 2 + ((child >> i) & 1)));
            }
        }
        cells = children;
    }
    let mut distances = BTreeMap::new();
    for c in &cells {
        for corner in 0..8 {
            let key: [usize; 3] = std::array::from_fn(|i| c[i] + ((corner >> i) & 1));
            distances.entry(key).or_insert_with(|| {
                let p = std::array::from_fn(|i| {
                    (key[i] as f32 / resolution as f32 - 0.5) * scale + center[i]
                });
                bvh.closest(p).1 - eps
            });
        }
    }
    let mut vertices = Vec::with_capacity(cells.len());
    let mut signs = Vec::with_capacity(cells.len());
    for c in &cells {
        let mut sum = [0.0; 3];
        let mut count = 0;
        let mut sign = [0_i8; 3];
        for axis in 0..3 {
            for u in 0..2 {
                for v in 0..2 {
                    let mut a = *c;
                    a[(axis + 1) % 3] += u;
                    a[(axis + 2) % 3] += v;
                    let mut b = a;
                    b[axis] += 1;
                    let (da, db) = (distances[&a], distances[&b]);
                    if (da < 0.0) != (db < 0.0) {
                        let t = -da / (db - da);
                        for i in 0..3 {
                            sum[i] += a[i] as f32 + if i == axis { t } else { 0.0 };
                        }
                        count += 1;
                        if u == 1 && v == 1 {
                            sign[axis] = if da < 0.0 { 1 } else { -1 };
                        }
                    }
                }
            }
        }
        let p = std::array::from_fn(|i| {
            let q = if count > 0 {
                sum[i] / count as f32
            } else {
                c[i] as f32 + 0.5
            };
            (q / resolution as f32 - 0.5) * scale + center[i]
        });
        vertices.push(p);
        signs.push(sign);
    }
    let lookup = cells
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect::<BTreeMap<_, _>>();
    let offsets = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let mut faces = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        for axis in 0..3 {
            if signs[i][axis] == 0 {
                continue;
            }
            let q =
                offsets[axis].map(|o| lookup.get(&std::array::from_fn(|i| c[i] + o[i])).copied());
            if let [Some(a), Some(b), Some(c), Some(d)] = q {
                let q = [a, b, c, d];
                let mut choices = [[0, 1, 2, 0, 2, 3], [0, 1, 3, 3, 1, 2]];
                if signs[i][axis] == 1 {
                    choices = [[0, 2, 1, 0, 3, 2], [0, 3, 1, 3, 2, 1]];
                }
                // Preserve the source's alignment score (its second normal uses
                // slots 1,2,3 of the six-index candidate, rather than its last tri).
                let alignment = |s: [usize; 6]| {
                    let p = s.map(|v| vertices[q[v]]);
                    dot(
                        cross(sub(p[1], p[0]), sub(p[2], p[0])),
                        cross(sub(p[2], p[1]), sub(p[3], p[1])),
                    )
                    .abs()
                };
                let split = if alignment(choices[0]) > alignment(choices[1]) {
                    choices[0]
                } else {
                    choices[1]
                };
                for t in split.as_chunks::<3>().0 {
                    let mut f = [0; 3];
                    for j in 0..3 {
                        f[j] = u32::try_from(q[t[j]])
                            .map_err(|_| GeometryError::Invalid("remesh index capacity"))?;
                    }
                    faces.push(f);
                }
            }
        }
    }
    let mut result = TriangleSurface { vertices, faces };
    compact(&mut result)?;
    if project > 0.0 {
        for p in &mut result.vertices {
            let projected = bvh.closest(*p).0;
            for i in 0..3 {
                p[i] -= project * (p[i] - projected[i]);
            }
        }
    }
    result.validate()?;
    Ok(result)
}

fn validate_charts(options: &ChartOptions) -> Result<(), GeometryError> {
    if !options.threshold_cone_half_angle_rad.is_finite()
        || options.threshold_cone_half_angle_rad < 0.0
        || !options.smooth_strength.is_finite()
        || options.smooth_strength < 0.0
    {
        return Err(GeometryError::Invalid("chart options"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Cone {
    axis: [f32; 3],
    half_angle: f32,
}

fn merge_cones(a: Cone, b: Cone) -> Cone {
    let cosine = dot(a.axis, b.axis).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let low = (-a.half_angle).min(angle - b.half_angle);
    let high = a.half_angle.max(angle + b.half_angle);
    let rotation = (high + low) * 0.5;
    let axis = if angle < 1e-3 {
        a.axis
    } else {
        let tangent = normalized(std::array::from_fn(|i| b.axis[i] - a.axis[i] * cosine));
        normalized(std::array::from_fn(|i| {
            a.axis[i] * rotation.cos() + tangent[i] * rotation.sin()
        }))
    };
    Cone {
        axis,
        half_angle: (high - low) * 0.5,
    }
}

fn recompute_cones(ids: &[usize], normals: &[[f32; 3]]) -> Vec<Cone> {
    let mut axes = vec![[0.0; 3]; ids.len()];
    for (&id, n) in ids.iter().zip(normals) {
        for i in 0..3 {
            axes[id][i] += n[i];
        }
    }
    let mut cones = axes
        .into_iter()
        .map(|axis| Cone {
            axis: normalized(axis),
            half_angle: 0.0,
        })
        .collect::<Vec<_>>();
    for (&id, &n) in ids.iter().zip(normals) {
        cones[id].half_angle = cones[id]
            .half_angle
            .max(dot(cones[id].axis, n).clamp(-1.0, 1.0).acos());
    }
    cones
}

fn clusters(surface: &TriangleSurface, options: &ChartOptions) -> Vec<Vec<usize>> {
    let edge_map = edges(surface);
    let normals = surface
        .faces
        .iter()
        .map(|&f| normalized(face_normal(surface, f)))
        .collect::<Vec<_>>();
    let areas = surface
        .faces
        .iter()
        .map(|&f| 0.5 * length(face_normal(surface, f)))
        .collect::<Vec<_>>();
    let mut ids = (0..surface.faces.len()).collect::<Vec<_>>();
    for _ in 0..options.global_iterations {
        loop {
            let cones = recompute_cones(&ids, &normals);
            let mut area = vec![0.0; ids.len()];
            for (&id, &a) in ids.iter().zip(&areas) {
                area[id] += a;
            }
            let mut adjacency = BTreeMap::<(usize, usize), f32>::new();
            let mut perimeter = vec![0.0; ids.len()];
            for (&(a, b), faces) in &edge_map {
                if let [(f, _), (g, _)] = faces.as_slice() {
                    let (f, g) = (ids[*f], ids[*g]);
                    if f == g {
                        continue;
                    }
                    let l = length(sub(
                        surface.vertices[a as usize],
                        surface.vertices[b as usize],
                    ));
                    *adjacency.entry((f.min(g), f.max(g))).or_default() += l;
                    perimeter[f] += l;
                    perimeter[g] += l;
                }
            }
            let mut costs = Vec::with_capacity(adjacency.len());
            let mut minimum = vec![(f32::INFINITY, usize::MAX); ids.len()];
            for (&(a, b), &l) in &adjacency {
                let cone = merge_cones(cones[a], cones[b]);
                let new_area = area[a] + area[b];
                let p = perimeter[a] + perimeter[b] - 2.0 * l;
                let cost = cone.half_angle
                    + 0.1 * new_area
                    + 0.0001 * p * p / new_area.max(f32::MIN_POSITIVE);
                let i = costs.len();
                for c in [a, b] {
                    if cost
                        .total_cmp(&minimum[c].0)
                        .then(i.cmp(&minimum[c].1))
                        .is_lt()
                    {
                        minimum[c] = (cost, i);
                    }
                }
                costs.push((a, b, cost, cone));
            }
            let mut map = (0..ids.len()).collect::<Vec<_>>();
            let mut changed = false;
            for (i, &(a, b, cost, _)) in costs.iter().enumerate() {
                if cost <= options.threshold_cone_half_angle_rad
                    && minimum[a].1 == i
                    && minimum[b].1 == i
                {
                    map[b] = a;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            for id in &mut ids {
                *id = map[*id];
            }
        }
        let mut cones = recompute_cones(&ids, &normals);
        for _ in 0..options.refine_iterations {
            let mut next = ids.clone();
            for (f, face) in surface.faces.iter().enumerate() {
                let mut scores = BTreeMap::from([(ids[f], 0.0)]);
                for i in 0..3 {
                    let (a, b) = (face[i], face[(i + 1) % 3]);
                    let l = length(sub(
                        surface.vertices[a as usize],
                        surface.vertices[b as usize],
                    ));
                    for &(g, _) in &edge_map[&(a.min(b), a.max(b))] {
                        if f != g {
                            *scores.entry(ids[g]).or_default() += l;
                        }
                    }
                }
                let mut best = ids[f];
                let mut score = -1e9_f32;
                for (id, smooth) in scores {
                    let geometry = dot(cones[id].axis, normals[f]);
                    if geometry <= 0.0 {
                        continue;
                    }
                    let candidate = geometry + options.smooth_strength * smooth;
                    if candidate - score > 1e-5 || ((candidate - score).abs() <= 1e-5 && id < best)
                    {
                        best = id;
                        score = candidate;
                    }
                }
                next[f] = best;
            }
            ids = next;
            cones = recompute_cones(&ids, &normals);
        }
        let mut parent = (0..ids.len()).collect::<Vec<_>>();
        for incident in edge_map.values() {
            if let [(a, _), (b, _)] = incident.as_slice() {
                if ids[*a] == ids[*b] {
                    join(&mut parent, *a, *b);
                }
            }
        }
        ids = (0..ids.len()).map(|i| root(&mut parent, i)).collect();
    }
    // Refinement can disconnect a label. Split its manifold-connected islands
    // before solving a chart, matching the source's chart-mesh construction.
    let mut adjacency = vec![Vec::new(); ids.len()];
    for incident in edge_map.values() {
        if let [(a, _), (b, _)] = incident.as_slice() {
            if ids[*a] == ids[*b] {
                adjacency[*a].push(*b);
                adjacency[*b].push(*a);
            }
        }
    }
    let mut visited = vec![false; ids.len()];
    let mut result = Vec::new();
    for seed in 0..ids.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut faces = vec![seed];
        let mut cursor = 0;
        while cursor < faces.len() {
            for &f in &adjacency[faces[cursor]] {
                if !visited[f] {
                    visited[f] = true;
                    faces.push(f);
                }
            }
            cursor += 1;
        }
        result.push(faces);
    }
    result
}

struct Chart {
    vertices: Vec<u32>,
    faces: Vec<[u32; 3]>,
    uv: Vec<[f32; 2]>,
    extent: [f32; 2],
}

// Least-squares conformal map: two Cauchy-Riemann residuals per triangle,
// area-weighted, with two farthest boundary vertices pinned. Solve the sparse
// normal equations by diagonally preconditioned conjugate gradients in FP64.
// An invalid/folded chart is cut and solved again; no distorted map is accepted.
fn parameterize(surface: &TriangleSurface, faces: &[usize]) -> Option<Chart> {
    let mut map = BTreeMap::new();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for &f in faces {
        let mut t = [0; 3];
        for (i, &v) in surface.faces[f].iter().enumerate() {
            let id = if let Some(&id) = map.get(&v) {
                id
            } else {
                let id = u32::try_from(vertices.len()).ok()?;
                vertices.push(v);
                map.insert(v, id);
                id
            };
            t[i] = id;
        }
        triangles.push(t);
    }
    let points = vertices
        .iter()
        .map(|&v| surface.vertices[v as usize])
        .collect::<Vec<_>>();
    let local = TriangleSurface {
        vertices: points.clone(),
        faces: triangles.clone(),
    };
    let boundary = edges(&local)
        .into_iter()
        .filter(|(_, f)| f.len() == 1)
        .flat_map(|((a, b), _)| [a as usize, b as usize])
        .collect::<BTreeSet<_>>();
    let &first = boundary.first()?;
    let farthest = |a: usize| {
        boundary.iter().copied().max_by(|&b, &c| {
            dot(sub(points[b], points[a]), sub(points[b], points[a]))
                .total_cmp(&dot(sub(points[c], points[a]), sub(points[c], points[a])))
        })
    };
    let a = farthest(first)?;
    let b = farthest(a)?;
    let separation = length(sub(points[a], points[b])) as f64;
    if separation <= 0.0 {
        return None;
    }
    let n = vertices.len() * 2;
    let mut pinned = vec![false; n];
    let mut fixed = vec![0.0; n];
    for i in [a * 2, a * 2 + 1, b * 2, b * 2 + 1] {
        pinned[i] = true;
    }
    fixed[b * 2] = separation;
    let mut rows = Vec::<[(usize, f64); 6]>::new();
    for t in &triangles {
        let p = t.map(|v| points[v as usize]);
        let ab = sub(p[1], p[0]);
        let ac = sub(p[2], p[0]);
        let x1 = length(ab) as f64;
        let x2 = dot(ab, ac) as f64 / x1;
        let y2 = length(cross(ab, ac)) as f64 / x1;
        let norm = (x1 * y2).sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return None;
        }
        let gx = [-y2 / norm, y2 / norm, 0.0];
        let gy = [(x2 - x1) / norm, -x2 / norm, x1 / norm];
        let r1 = std::array::from_fn(|i| {
            let c = i / 2;
            (
                t[c] as usize * 2 + i % 2,
                if i % 2 == 0 { gx[c] } else { -gy[c] },
            )
        });
        let r2 = std::array::from_fn(|i| {
            let c = i / 2;
            (
                t[c] as usize * 2 + i % 2,
                if i % 2 == 0 { gy[c] } else { gx[c] },
            )
        });
        rows.extend([r1, r2]);
    }
    let mut rhs = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    for row in &rows {
        let constant = row
            .iter()
            .filter(|(i, _)| pinned[*i])
            .map(|&(i, v)| v * fixed[i])
            .sum::<f64>();
        for &(i, v) in row {
            if !pinned[i] {
                rhs[i] -= v * constant;
                diagonal[i] += v * v;
            }
        }
    }
    if (0..n).any(|i| !pinned[i] && diagonal[i] <= 0.0) {
        return None;
    }
    let multiply = |x: &[f64]| {
        let mut y = vec![0.0; n];
        for row in &rows {
            let value = row
                .iter()
                .filter(|(i, _)| !pinned[*i])
                .map(|&(i, v)| v * x[i])
                .sum::<f64>();
            for &(i, v) in row {
                if !pinned[i] {
                    y[i] += v * value;
                }
            }
        }
        y
    };
    let inner = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let mut x = vec![0.0; n];
    let mut residual = rhs.clone();
    let mut z = (0..n)
        .map(|i| {
            if pinned[i] {
                0.0
            } else {
                residual[i] / diagonal[i]
            }
        })
        .collect::<Vec<_>>();
    let mut direction = z.clone();
    let mut rz = inner(&residual, &z);
    let tolerance = inner(&rhs, &rhs).max(1e-30) * 1e-18;
    for _ in 0..n.saturating_mul(4).max(64) {
        if inner(&residual, &residual) <= tolerance {
            break;
        }
        let ad = multiply(&direction);
        let denominator = inner(&direction, &ad);
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let alpha = rz / denominator;
        for i in 0..n {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ad[i];
        }
        for i in 0..n {
            z[i] = if pinned[i] {
                0.0
            } else {
                residual[i] / diagonal[i]
            };
        }
        let next = inner(&residual, &z);
        let beta = next / rz;
        for i in 0..n {
            direction[i] = z[i] + beta * direction[i];
        }
        rz = next;
    }
    if inner(&residual, &residual) > tolerance * 100.0 {
        return None;
    }
    for i in 0..n {
        if pinned[i] {
            x[i] = fixed[i];
        }
    }
    let mut uv = x
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32])
        .collect::<Vec<_>>();
    if uv.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let area2 = |t: [u32; 3]| {
        let p = t.map(|v| uv[v as usize]);
        (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])
    };
    let sign = area2(triangles[0]).signum();
    if triangles.iter().any(|&t| area2(t) * sign <= 1e-20) {
        return None;
    }
    if sign < 0.0 {
        for p in &mut uv {
            p[1] = -p[1];
        }
    }
    if overlaps(&uv, &triangles) {
        return None;
    }
    // Equal texel density across charts: scale UV area to world-space area.
    let uv_area = triangles
        .iter()
        .map(|t| {
            let p = t.map(|v| uv[v as usize]);
            0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
                .abs()
        })
        .sum::<f32>();
    let world_area = faces
        .iter()
        .map(|&f| 0.5 * length(face_normal(surface, surface.faces[f])))
        .sum::<f32>();
    let scale = (world_area / uv_area).sqrt();
    if !scale.is_finite() {
        return None;
    }
    let low: [f32; 2] =
        std::array::from_fn(|i| uv.iter().map(|p| p[i]).fold(f32::INFINITY, f32::min));
    for p in &mut uv {
        for i in 0..2 {
            p[i] = (p[i] - low[i]) * scale;
        }
    }
    let extent = std::array::from_fn(|i| uv.iter().map(|p| p[i]).fold(0.0, f32::max));
    Some(Chart {
        vertices,
        faces: triangles,
        uv,
        extent,
    })
}

fn overlaps(uv: &[[f32; 2]], faces: &[[u32; 3]]) -> bool {
    let side = (faces.len() as f64).sqrt().ceil().max(1.0) as usize;
    let low: [f32; 2] =
        std::array::from_fn(|i| uv.iter().map(|p| p[i]).fold(f32::INFINITY, f32::min));
    let extent: [f32; 2] = std::array::from_fn(|i| {
        uv.iter()
            .map(|p| p[i] - low[i])
            .fold(0.0, f32::max)
            .max(1e-20)
    });
    let mut bins = BTreeMap::<(usize, usize), Vec<usize>>::new();
    let cross2 = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    for (f, face) in faces.iter().enumerate() {
        let p = face.map(|v| uv[v as usize]);
        let lo: [usize; 2] = std::array::from_fn(|i| {
            ((p.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min) - low[i]) / extent[i]
                * side as f32)
                .floor()
                .max(0.0) as usize
        });
        let hi: [usize; 2] = std::array::from_fn(|i| {
            (((p.iter().map(|v| v[i]).fold(f32::NEG_INFINITY, f32::max) - low[i]) / extent[i]
                * side as f32)
                .floor()
                .max(0.0) as usize)
                .min(side)
        });
        let mut tested = BTreeSet::new();
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                let bin = bins.entry((x, y)).or_default();
                for &g in bin.iter() {
                    if !tested.insert(g) {
                        continue;
                    }
                    let q = faces[g].map(|v| uv[v as usize]);
                    for (a, b) in [(p, q), (q, p)] {
                        for v in a {
                            if (0..3).all(|i| cross2(b[i], b[(i + 1) % 3], v) > 1e-10) {
                                return true;
                            }
                        }
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            let a = cross2(p[i], p[(i + 1) % 3], q[j]);
                            let b = cross2(p[i], p[(i + 1) % 3], q[(j + 1) % 3]);
                            let c = cross2(q[j], q[(j + 1) % 3], p[i]);
                            let d = cross2(q[j], q[(j + 1) % 3], p[(i + 1) % 3]);
                            if a * b < -1e-20 && c * d < -1e-20 {
                                return true;
                            }
                        }
                    }
                }
                bin.push(f);
            }
        }
    }
    false
}

pub(super) struct Atlas {
    pub surface: TriangleSurface,
    pub uv: Vec<[f32; 2]>,
    pub vertex_map: Vec<usize>,
}

pub(super) fn unwrap(
    surface: &TriangleSurface,
    options: &ChartOptions,
    texture_size: usize,
) -> Result<Atlas, GeometryError> {
    validate_charts(options)?;
    let mut pending = clusters(surface, options);
    let mut charts = Vec::new();
    while let Some(mut faces) = pending.pop() {
        if let Some(chart) = parameterize(surface, &faces) {
            charts.push(chart);
            continue;
        }
        if faces.len() == 1 {
            return Err(GeometryError::Invalid("degenerate UV chart"));
        }
        // Cut failed parameterizations at the median along their longest
        // geometric axis. Re-clustering each side splits disconnected pieces.
        let center = |f: usize, axis: usize| {
            surface.faces[f]
                .iter()
                .map(|&v| surface.vertices[v as usize][axis] / 3.0)
                .sum::<f32>()
        };
        let spans: [f32; 3] = std::array::from_fn(|i| {
            faces
                .iter()
                .map(|&f| center(f, i))
                .fold(f32::NEG_INFINITY, f32::max)
                - faces
                    .iter()
                    .map(|&f| center(f, i))
                    .fold(f32::INFINITY, f32::min)
        });
        let axis = (0..3)
            .max_by(|&a, &b| spans[a].total_cmp(&spans[b]))
            .unwrap_or(0);
        faces.sort_by(|&a, &b| center(a, axis).total_cmp(&center(b, axis)).then(a.cmp(&b)));
        let right = faces.split_off(faces.len() / 2);
        pending.extend([faces, right]);
    }
    charts.sort_by(|a, b| {
        b.extent[1]
            .total_cmp(&a.extent[1])
            .then(b.extent[0].total_cmp(&a.extent[0]))
    });
    let pack = |scale: f32| -> Option<Vec<[usize; 2]>> {
        let mut cursor = [0_usize; 2];
        let mut row = 0;
        let mut origins = Vec::new();
        for chart in &charts {
            let size = chart.extent.map(|v| (v * scale).ceil() as usize + 4);
            if size[0] > texture_size || size[1] > texture_size {
                return None;
            }
            if cursor[0] + size[0] > texture_size {
                cursor[0] = 0;
                cursor[1] += row;
                row = 0;
            }
            if cursor[1] + size[1] > texture_size {
                return None;
            }
            origins.push([cursor[0] + 2, cursor[1] + 2]);
            cursor[0] += size[0];
            row = row.max(size[1]);
        }
        Some(origins)
    };
    if pack(0.0).is_none() {
        return Err(GeometryError::Invalid("texture capacity for chart gutters"));
    }
    let max_extent = charts.iter().flat_map(|c| c.extent).fold(0.0, f32::max);
    let mut low = 0.0;
    let mut high = texture_size as f32 / max_extent.max(f32::MIN_POSITIVE);
    for _ in 0..32 {
        let mid = (low + high) * 0.5;
        if pack(mid).is_some() {
            low = mid;
        } else {
            high = mid;
        }
    }
    if low <= 0.0 {
        return Err(GeometryError::Invalid(
            "texture capacity for chart interiors",
        ));
    }
    let origins = pack(low).ok_or(GeometryError::Invalid("chart packing"))?;
    let mut output = TriangleSurface {
        vertices: Vec::new(),
        faces: Vec::new(),
    };
    let mut uv = Vec::new();
    let mut vertex_map = Vec::new();
    for (chart, origin) in charts.into_iter().zip(origins) {
        let base = u32::try_from(output.vertices.len())
            .map_err(|_| GeometryError::Invalid("atlas index capacity"))?;
        for &v in &chart.vertices {
            output.vertices.push(surface.vertices[v as usize]);
            vertex_map.push(v as usize);
        }
        for p in chart.uv {
            uv.push(std::array::from_fn(|i| {
                (origin[i] as f32 + p[i] * low) / texture_size as f32
            }));
        }
        for f in chart.faces {
            let mut face = [0; 3];
            for i in 0..3 {
                face[i] = base
                    .checked_add(f[i])
                    .ok_or(GeometryError::Invalid("atlas index capacity"))?;
            }
            output.faces.push(face);
        }
    }
    Ok(Atlas {
        surface: output,
        uv,
        vertex_map,
    })
}
