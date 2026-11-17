//! Sparse dual-grid geometry, texture baking and portable PBR assets.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::io::Cursor;

use thiserror::Error;

mod processing;
pub use processing::{
    cleanup_mesh, remesh_narrow_band_dc, simplify_mesh, unify_face_orientations, ChartOptions,
    RemeshOptions,
};

/// Geometry and texture contract failures.
#[derive(Debug, Error)]
pub enum GeometryError {
    /// Invalid dimensions, indices or attribute buffers.
    #[error("invalid geometry: {0}")]
    Invalid(&'static str),
    /// Image encoding failed.
    #[error(transparent)]
    Image(#[from] image::ImageError),
    /// JSON serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Asset output failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// An indexed triangle surface in world coordinates.
#[derive(Clone, Debug)]
pub struct TriangleSurface {
    /// XYZ positions.
    pub vertices: Vec<[f32; 3]>,
    /// Oriented triangle indices.
    pub faces: Vec<[u32; 3]>,
}

impl TriangleSurface {
    /// Validate finite positions and triangle indices.
    /// # Errors
    /// Reports nonfinite vertices or indices outside the vertex buffer.
    pub fn validate(&self) -> Result<(), GeometryError> {
        if self.vertices.iter().flatten().any(|x| !x.is_finite())
            || self
                .faces
                .iter()
                .flatten()
                .any(|&i| i as usize >= self.vertices.len())
        {
            return Err(GeometryError::Invalid("surface vertices or indices"));
        }
        Ok(())
    }

    /// Fill manifold boundary loops whose world-space perimeter is below a bound.
    /// Inserts a boundary-midpoint mean and a fan reversing the boundary orientation.
    /// # Errors
    /// Reports invalid surface buffers or a nonfinite perimeter bound.
    pub fn fill_holes(&mut self, max_perimeter: f32) -> Result<(), GeometryError> {
        self.validate()?;
        if !max_perimeter.is_finite() || max_perimeter < 0.0 {
            return Err(GeometryError::Invalid("hole perimeter"));
        }
        let mut edges = BTreeMap::<(u32, u32), Vec<(u32, u32)>>::new();
        for face in &self.faces {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                edges.entry((a.min(b), a.max(b))).or_default().push((a, b));
            }
        }
        let mut next = BTreeMap::<u32, Vec<u32>>::new();
        for directed in edges.values().filter(|edge| edge.len() == 1) {
            next.entry(directed[0].0).or_default().push(directed[0].1);
            next.entry(directed[0].1).or_default().push(directed[0].0);
        }
        let mut used = BTreeSet::new();
        for (&start, outgoing) in &next {
            if outgoing.len() != 2 || used.contains(&start) {
                continue;
            }
            let mut polygon = Vec::new();
            let mut current = start;
            let mut previous = None;
            let mut perimeter = 0.0;
            loop {
                if used.contains(&current) {
                    break;
                }
                used.insert(current);
                polygon.push(current);
                let Some(successors) = next.get(&current).filter(|value| value.len() == 2) else {
                    break;
                };
                let successor = if Some(successors[0]) == previous {
                    successors[1]
                } else {
                    successors[0]
                };
                perimeter += length(sub(
                    self.vertices[current as usize],
                    self.vertices[successor as usize],
                ));
                previous = Some(current);
                current = successor;
                if current == start {
                    break;
                }
            }
            if current != start || polygon.len() < 3 || perimeter >= max_perimeter {
                continue;
            }
            // CuMesh inserts the mean of boundary-edge midpoints, then fans
            // every oriented boundary edge to that center.
            let center = u32::try_from(self.vertices.len())
                .map_err(|_| GeometryError::Invalid("hole vertex capacity"))?;
            let mut position = [0.0; 3];
            for (i, &a) in polygon.iter().enumerate() {
                let b = polygon[(i + 1) % polygon.len()];
                for (axis, coordinate) in position.iter_mut().enumerate() {
                    *coordinate +=
                        (self.vertices[a as usize][axis] + self.vertices[b as usize][axis]) * 0.5;
                }
            }
            for value in &mut position {
                *value /= polygon.len() as f32;
            }
            self.vertices.push(position);
            for (i, &a) in polygon.iter().enumerate() {
                let b = polygon[(i + 1) % polygon.len()];
                let direction = edges[&(a.min(b), a.max(b))][0];
                self.faces.push([direction.1, direction.0, center]);
            }
        }
        self.validate()
    }
}

/// Occupied cells with local dual vertices and axis-edge intersection flags.
#[derive(Debug)]
pub struct FlexibleDualGrid {
    /// XYZ integer cell coordinates, in original insertion order.
    pub coordinates: Vec<[i64; 3]>,
    /// Local vertex offsets in cell units.
    pub vertices: Vec<[f32; 3]>,
    /// Intersections along each positive axis edge.
    pub intersected: Vec<[bool; 3]>,
}

/// Recover decoder child occupancy from an encoder's saved coordinate level.
/// Child bits use X-fast XYZ order, matching spatial-to-channel packing.
/// # Errors
/// Reports coordinate arithmetic overflow.
pub fn dual_grid_subdivision(
    parents: &[[i64; 3]],
    children: &[[i64; 3]],
) -> Result<Vec<[f32; 8]>, GeometryError> {
    let children = children.iter().copied().collect::<BTreeSet<_>>();
    parents
        .iter()
        .map(|parent| {
            let mut occupied = [0.0; 8];
            for (child, value) in occupied.iter_mut().enumerate() {
                let mut coordinate = [0; 3];
                for axis in 0..3 {
                    coordinate[axis] = parent[axis]
                        .checked_mul(2)
                        .and_then(|v| v.checked_add(((child >> axis) & 1) as i64))
                        .ok_or(GeometryError::Invalid("subdivision coordinate overflow"))?;
                }
                *value = if children.contains(&coordinate) {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(occupied)
        })
        .collect()
}

/// Extract a surface with the positive per-vertex quad-split weights.
/// # Errors
/// Reports invalid or duplicate rows, nonpositive weights, coordinate or index overflow.
pub fn flexible_dual_grid_to_mesh(
    grid: &FlexibleDualGrid,
    split_weights: &[f32],
    resolution: usize,
) -> Result<TriangleSurface, GeometryError> {
    let count = grid.coordinates.len();
    if resolution == 0
        || grid.vertices.len() != count
        || grid.intersected.len() != count
        || split_weights.len() != count
        || count > u32::MAX as usize
    {
        return Err(GeometryError::Invalid("dual-grid rows or resolution"));
    }
    if grid.vertices.iter().flatten().any(|v| !v.is_finite())
        || split_weights.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(GeometryError::Invalid(
            "dual-grid local vertices or split weights",
        ));
    }
    let bound = i64::try_from(resolution)
        .map_err(|_| GeometryError::Invalid("dual-grid resolution capacity"))?;
    if grid
        .coordinates
        .iter()
        .flatten()
        .any(|&v| v < 0 || v >= bound)
    {
        return Err(GeometryError::Invalid(
            "dual-grid coordinate outside resolution",
        ));
    }
    let scale = 1.0 / resolution as f32;
    let vertices = grid
        .coordinates
        .iter()
        .zip(&grid.vertices)
        .map(|(coordinate, local)| {
            std::array::from_fn(|axis| (coordinate[axis] as f32 + local[axis]) * scale - 0.5)
        })
        .collect();
    let index = grid
        .coordinates
        .iter()
        .enumerate()
        .map(|(row, &coordinate)| (coordinate, row as u32))
        .collect::<BTreeMap<_, _>>();
    if index.len() != count {
        return Err(GeometryError::Invalid("duplicate dual-grid coordinates"));
    }
    let offsets = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let mut faces = Vec::new();
    for (row, coordinate) in grid.coordinates.iter().enumerate() {
        for (axis, axis_offsets) in offsets.iter().enumerate() {
            if !grid.intersected[row][axis] {
                continue;
            }
            let mut quad = Vec::with_capacity(4);
            for offset in axis_offsets {
                let mut neighbor = [0; 3];
                for dimension in 0..3 {
                    neighbor[dimension] = coordinate[dimension]
                        .checked_add(offset[dimension])
                        .ok_or(GeometryError::Invalid("dual-grid coordinate overflow"))?;
                }
                quad.push(index.get(&neighbor).copied());
            }
            let quad = quad.into_iter().collect::<Option<Vec<_>>>();
            if let Some(quad) = quad {
                let first = split_weights[quad[0] as usize] * split_weights[quad[2] as usize];
                let second = split_weights[quad[1] as usize] * split_weights[quad[3] as usize];
                if !first.is_finite() || !second.is_finite() {
                    return Err(GeometryError::Invalid("dual-grid split product overflow"));
                }
                let split = if first > second {
                    [0, 1, 2, 0, 2, 3]
                } else {
                    [0, 1, 3, 3, 1, 2]
                };
                faces.extend(
                    split
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|triangle| [quad[triangle[0]], quad[triangle[1]], quad[triangle[2]]]),
                );
            }
        }
    }
    let result = TriangleSurface { vertices, faces };
    result.validate()?;
    Ok(result)
}

struct Cell {
    coordinate: [i64; 3],
    mean: [f32; 3],
    count: f32,
    intersected: [bool; 3],
    qef: [[f32; 4]; 4],
}

/// Voxelize a surface with scan-line intersections, face/boundary QEFs and box constraints.
///
/// Arithmetic follows the original o-voxel CPU contract: FP64 scan lines, FP32
/// accumulated planes and a regularized, constrained three-dimensional solve.
/// Positions must already use the [-0.5, 0.5] model coordinate frame.
/// # Errors
/// Reports invalid surfaces, resolution or nonfinite QEF weights.
pub fn mesh_to_flexible_dual_grid(
    surface: &TriangleSurface,
    resolution: usize,
    face_weight: f32,
    boundary_weight: f32,
    regularization_weight: f32,
) -> Result<FlexibleDualGrid, GeometryError> {
    surface.validate()?;
    if resolution == 0
        || resolution > i32::MAX as usize
        || [face_weight, boundary_weight, regularization_weight]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(GeometryError::Invalid("voxelization controls"));
    }
    let size = 1.0 / resolution as f32;
    let vertices = surface
        .vertices
        .iter()
        .map(|vertex| vertex.map(|value| value + 0.5))
        .collect::<Vec<_>>();
    let mut cells = Vec::<Cell>::new();
    let mut indices = BTreeMap::new();
    for face in &surface.faces {
        let triangle = face.map(|index| vertices[index as usize]);
        let normal = normalized(cross(
            sub(triangle[1], triangle[0]),
            sub(triangle[2], triangle[1]),
        ));
        if length(normal) == 0.0 {
            continue;
        }
        let plane = [normal[0], normal[1], normal[2], -dot(normal, triangle[0])];
        let qef: [[f32; 4]; 4] =
            std::array::from_fn(|i| std::array::from_fn(|j| plane[i] * plane[j]));
        for axis2 in 0..3 {
            let axis0 = (axis2 + 1) % 3;
            let axis1 = (axis2 + 2) % 3;
            let mut points = triangle.map(|v| {
                [
                    f64::from(v[axis0]),
                    f64::from(v[axis1]),
                    f64::from(v[axis2]),
                ]
            });
            points.sort_by(|a, b| a[1].total_cmp(&b[1]));
            let row =
                |value: f64| ((value / f64::from(size)) as i64).clamp(0, resolution as i64 - 1);
            let start = row(points[0][1]);
            let mid = row(points[1][1]);
            let end = row(points[2][1]);
            for (begin, end, a, b, c) in [
                (start, mid, points[0], points[1], points[2]),
                (mid, end, points[2], points[1], points[0]),
            ] {
                for yi in begin..end {
                    let y = (yi + 1) as f64 * f64::from(size);
                    let edge = |end: [f64; 3]| {
                        let alpha = if a[1] == end[1] {
                            0.0
                        } else {
                            (y - a[1]) / (end[1] - a[1])
                        };
                        [
                            (1.0 - alpha) * a[0] + alpha * end[0],
                            (1.0 - alpha) * a[2] + alpha * end[2],
                        ]
                    };
                    let mut left = edge(b);
                    let mut right = edge(c);
                    if left[0] > right[0] {
                        std::mem::swap(&mut left, &mut right);
                    }
                    for xi in row(left[0])..row(right[0]) {
                        let x = (xi + 1) as f64 * f64::from(size);
                        let alpha = if left[0] == right[0] {
                            0.0
                        } else {
                            (x - left[0]) / (right[0] - left[0])
                        };
                        let z = (1.0 - alpha) * left[1] + alpha * right[1];
                        let zi = (z / f64::from(size)) as i64;
                        if zi < 0 || zi >= resolution as i64 {
                            continue;
                        }
                        let mut position = [0.0; 3];
                        position[axis0] = x as f32;
                        position[axis1] = y as f32;
                        position[axis2] = z as f32;
                        for dx in 0..2 {
                            for dy in 0..2 {
                                let mut coordinate = [0; 3];
                                coordinate[axis0] = xi + dx;
                                coordinate[axis1] = yi + dy;
                                coordinate[axis2] = zi;
                                let index = *indices.entry(coordinate).or_insert_with(|| {
                                    cells.push(Cell {
                                        coordinate,
                                        mean: [0.0; 3],
                                        count: 0.0,
                                        intersected: [false; 3],
                                        qef: [[0.0; 4]; 4],
                                    });
                                    cells.len() - 1
                                });
                                let cell = &mut cells[index];
                                for (value, addition) in cell.mean.iter_mut().zip(position) {
                                    *value += addition;
                                }
                                cell.count += 1.0;
                                if dx == 0 && dy == 0 {
                                    cell.intersected[axis2] = true;
                                }
                                add_qef(&mut cell.qef, &qef, 1.0);
                            }
                        }
                    }
                }
            }
        }
    }
    if face_weight > 0.0 {
        for face in &surface.faces {
            let triangle = face.map(|index| vertices[index as usize]);
            let edges = [
                sub(triangle[1], triangle[0]),
                sub(triangle[2], triangle[1]),
                sub(triangle[0], triangle[2]),
            ];
            let normal = normalized(cross(edges[0], edges[1]));
            let plane = [normal[0], normal[1], normal[2], -dot(normal, triangle[0])];
            let qef = std::array::from_fn(|i| std::array::from_fn(|j| plane[i] * plane[j]));
            let min: [i64; 3] = std::array::from_fn(|axis| {
                ((triangle
                    .iter()
                    .map(|v| v[axis])
                    .fold(f32::INFINITY, f32::min)
                    / size) as i64)
                    .max(0)
            });
            let max: [i64; 3] = std::array::from_fn(|axis| {
                ((triangle
                    .iter()
                    .map(|v| v[axis])
                    .fold(f32::NEG_INFINITY, f32::max)
                    / size
                    + 1.0) as i64)
                    .min(resolution as i64)
            });
            let corner = normal.map(|value| if value > 0.0 { size } else { 0.0 });
            let d1 = dot(normal, sub(corner, triangle[0]));
            let d2 = dot(normal, sub(sub([size; 3], corner), triangle[0]));
            for z in min[2]..max[2] {
                for y in min[1]..max[1] {
                    for x in min[0]..max[0] {
                        let coordinate = [x, y, z];
                        let Some(&index) = indices.get(&coordinate) else {
                            continue;
                        };
                        let p = coordinate.map(|value| value as f32 * size);
                        let distance = dot(normal, p);
                        if (distance + d1) * (distance + d2) > 0.0 {
                            continue;
                        }
                        let mut inside = true;
                        for axis in 0..3 {
                            let a = axis;
                            let b = (axis + 1) % 3;
                            let perpendicular = (axis + 2) % 3;
                            let sign = if normal[perpendicular] < 0.0 {
                                -1.0
                            } else {
                                1.0
                            };
                            for edge in 0..3 {
                                let n = [-sign * edges[edge][b], sign * edges[edge][a]];
                                let offset = -n[0] * triangle[edge][a] - n[1] * triangle[edge][b]
                                    + n[0].max(0.0) * size
                                    + n[1].max(0.0) * size;
                                if n[0] * p[a] + n[1] * p[b] + offset < 0.0 {
                                    inside = false;
                                }
                            }
                        }
                        if inside {
                            add_qef(&mut cells[index].qef, &qef, 1.0);
                        }
                    }
                }
            }
        }
    }
    if boundary_weight > 0.0 {
        let mut edges = BTreeMap::<(u32, u32), usize>::new();
        for face in &surface.faces {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        for ((a, b), count) in edges {
            if count != 1 {
                continue;
            }
            let a = vertices[a as usize];
            let b = vertices[b as usize];
            let direction = sub(b, a).map(f64::from);
            let distance = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            if distance < 1e-6 {
                continue;
            }
            let direction = direction.map(|value| value / distance);
            let matrix: [[f32; 3]; 3] = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (if i == j { 1.0 } else { 0.0 }) - (direction[i] * direction[j]) as f32
                })
            });
            let rhs: [f32; 3] = std::array::from_fn(|i| -dot(matrix[i], a));
            let mut qef = [[0.0; 4]; 4];
            for i in 0..3 {
                for j in 0..3 {
                    qef[i][j] = matrix[i][j];
                }
                qef[i][3] = rhs[i];
                qef[3][i] = rhs[i];
            }
            qef[3][3] = -dot(a, rhs);
            let mut coordinate = a.map(|v| (v / size).floor() as i64);
            let step = direction.map(|v| if v > 0.0 { 1 } else { -1 });
            let mut next: [f64; 3] = std::array::from_fn(|axis| {
                if direction[axis] == 0.0 {
                    f64::INFINITY
                } else {
                    let border = size * (coordinate[axis] + i64::from(step[axis] > 0)) as f32;
                    f64::from(border - a[axis]) / direction[axis]
                }
            });
            let delta = direction.map(|v| f64::from(size) / v.abs());
            loop {
                if coordinate.iter().all(|&v| v >= 0 && v < resolution as i64) {
                    if let Some(&index) = indices.get(&coordinate) {
                        add_qef(&mut cells[index].qef, &qef, boundary_weight);
                    }
                }
                let axis = if next[0] < next[1] {
                    if next[0] < next[2] {
                        0
                    } else {
                        2
                    }
                } else if next[1] < next[2] {
                    1
                } else {
                    2
                };
                if next[axis] > distance {
                    break;
                }
                coordinate[axis] += step[axis];
                next[axis] += delta[axis];
            }
        }
    }
    let mut grid = FlexibleDualGrid {
        coordinates: Vec::with_capacity(cells.len()),
        vertices: Vec::with_capacity(cells.len()),
        intersected: Vec::with_capacity(cells.len()),
    };
    for mut cell in cells {
        let center = cell.mean.map(|value| value / cell.count);
        if regularization_weight > 0.0 {
            let weight = regularization_weight * cell.count;
            for (axis, coordinate) in center.iter().enumerate() {
                cell.qef[axis][axis] += weight;
                cell.qef[axis][3] -= weight * coordinate;
                cell.qef[3][axis] -= weight * coordinate;
            }
            cell.qef[3][3] += weight * dot(center, center);
        }
        let min = cell.coordinate.map(|value| value as f32 * size);
        let max = cell.coordinate.map(|value| (value + 1) as f32 * size);
        let vertex = constrained_qef(&cell.qef, min, max);
        grid.coordinates.push(cell.coordinate);
        grid.vertices.push(std::array::from_fn(|axis| {
            vertex[axis] * resolution as f32 - cell.coordinate[axis] as f32
        }));
        grid.intersected.push(cell.intersected);
    }
    Ok(grid)
}

fn add_qef(target: &mut [[f32; 4]; 4], source: &[[f32; 4]; 4], weight: f32) {
    for i in 0..4 {
        for j in 0..4 {
            target[i][j] += weight * source[i][j];
        }
    }
}

fn constrained_qef(q: &[[f32; 4]; 4], min: [f32; 3], max: [f32; 3]) -> [f32; 3] {
    let solve = |fixed: [Option<f32>; 3]| {
        let free = (0..3)
            .filter(|&axis| fixed[axis].is_none())
            .collect::<Vec<_>>();
        let mut matrix = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for (i, &axis) in free.iter().enumerate() {
            rhs[i] = -q[axis][3];
            for j in 0..3 {
                if let Some(value) = fixed[j] {
                    rhs[i] -= q[axis][j] * value;
                }
            }
            for (j, &other) in free.iter().enumerate() {
                matrix[i][j] = q[axis][other];
            }
        }
        let answer = qr_solve(matrix, rhs, free.len());
        let mut point = [0.0; 3];
        for axis in 0..3 {
            point[axis] = fixed[axis].unwrap_or(0.0);
        }
        for (i, &axis) in free.iter().enumerate() {
            point[axis] = answer[i];
        }
        point
    };
    let unconstrained = solve([None; 3]);
    let inside = |p: [f32; 3]| (0..3).all(|axis| p[axis] >= min[axis] && p[axis] <= max[axis]);
    if inside(unconstrained) {
        return unconstrained;
    }
    let mut best = f32::INFINITY;
    let mut result = min;
    let mut candidate = |point: [f32; 3]| {
        if !inside(point) {
            return;
        }
        let p = [point[0], point[1], point[2], 1.0];
        let error = (0..4)
            .map(|j| (0..4).map(|i| p[i] * q[i][j]).sum::<f32>() * p[j])
            .sum::<f32>();
        if error < best {
            best = error;
            result = point;
        }
    };
    for axis in 0..3 {
        for bound in [min[axis], max[axis]] {
            let mut fixed = [None; 3];
            fixed[axis] = Some(bound);
            candidate(solve(fixed));
        }
    }
    for free in 0..3 {
        let a = (free + 1) % 3;
        let b = (free + 2) % 3;
        for va in [min[a], max[a]] {
            for vb in [min[b], max[b]] {
                let mut fixed = [None; 3];
                fixed[a] = Some(va);
                fixed[b] = Some(vb);
                candidate(solve(fixed));
            }
        }
    }
    for x in [max[0], min[0]] {
        for y in [max[1], min[1]] {
            for z in [max[2], min[2]] {
                candidate([x, y, z]);
            }
        }
    }
    result
}

#[expect(
    clippy::needless_range_loop,
    reason = "QR updates address matrix rows and columns independently; explicit indices preserve the source arithmetic order"
)]
fn qr_solve(mut a: [[f32; 3]; 3], mut b: [f32; 3], n: usize) -> [f32; 3] {
    let mut permutation = [0, 1, 2];
    for k in 0..n {
        let mut pivot = k;
        let mut largest = -1.0;
        for col in k..n {
            let norm = (k..n).map(|row| a[row][col] * a[row][col]).sum::<f32>();
            if norm > largest {
                largest = norm;
                pivot = col;
            }
        }
        for row in 0..n {
            a[row].swap(k, pivot);
        }
        permutation.swap(k, pivot);
        let norm = (k..n).map(|row| a[row][k] * a[row][k]).sum::<f32>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let beta = if a[k][k] >= 0.0 { -norm } else { norm };
        let mut v = [0.0; 3];
        for row in k..n {
            v[row] = a[row][k];
        }
        v[k] -= beta;
        let denominator = (k..n).map(|row| v[row] * v[row]).sum::<f32>();
        for col in k..n {
            let factor = 2.0 * (k..n).map(|row| v[row] * a[row][col]).sum::<f32>() / denominator;
            for row in k..n {
                a[row][col] -= factor * v[row];
            }
        }
        let factor = 2.0 * (k..n).map(|row| v[row] * b[row]).sum::<f32>() / denominator;
        for row in k..n {
            b[row] -= factor * v[row];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..n).rev() {
        if a[row][row].abs() > f32::EPSILON * a[0][0].abs() {
            x[row] =
                (b[row] - (row + 1..n).map(|col| a[row][col] * x[col]).sum::<f32>()) / a[row][row];
        }
    }
    let mut result = [0.0; 3];
    for i in 0..n {
        result[permutation[i]] = x[i];
    }
    result
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}
fn normalized(a: [f32; 3]) -> [f32; 3] {
    let n = length(a);
    if n == 0.0 {
        [0.0; 3]
    } else {
        a.map(|v| v / n)
    }
}
fn barycentric(p: [f32; 2], t: [[f32; 2]; 3]) -> Option<[f32; 3]> {
    let edge = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let area = edge(t[0], t[1], t[2]);
    if area == 0.0 {
        return None;
    }
    let a = edge(t[1], t[2], p) / area;
    let b = edge(t[2], t[0], p) / area;
    Some([a, b, 1.0 - a - b])
}

/// Sparse FP32 PBR voxels with explicit world-to-grid geometry.
#[derive(Debug)]
pub struct SparsePbrVolume<'a> {
    /// XYZ occupied cells.
    pub coordinates: &'a [[i64; 3]],
    /// RGB, metallic, roughness and alpha, one row per cell.
    pub attributes: &'a [[f32; 6]],
    /// Grid side length.
    pub resolution: usize,
}

/// A surface with independent UV vertices and baked PBR images.
#[derive(Debug)]
pub struct TexturedSurface {
    /// Surface after seam expansion.
    pub surface: TriangleSurface,
    /// UVs, one for every surface vertex.
    pub uvs: Vec<[f32; 2]>,
    /// Unit, area-weighted vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Square image side length.
    pub texture_size: usize,
    /// RGBA base-color pixels, in image row order.
    pub base_color: Vec<u8>,
    /// RGB pixels with roughness in green and metallic in blue.
    pub metallic_roughness: Vec<u8>,
    /// Preserve both sides for open standard exports; remeshed exports are closed.
    pub double_sided: bool,
}

/// Bake sparse PBR attributes to a conformal chart atlas, or supplied UVs.
///
/// Supplying UVs preserves an existing mesh's mapping. Without UVs, normal-cone
/// charts are parameterized with LSCM and packed with texel gutters.
/// Attribute samples use FP32 trilinear interpolation, normalized over occupied
/// neighbors. A query with no occupied neighbor produces zero attributes.
/// Byte conversion clips and truncates before fast-marching seam filling.
/// # Errors
/// Reports incompatible geometry, UVs, sparse buffers or texture capacity.
pub fn bake_pbr_texture(
    surface: &TriangleSurface,
    uvs: Option<&[[f32; 2]]>,
    volume: &SparsePbrVolume<'_>,
    texture_size: usize,
) -> Result<TexturedSurface, GeometryError> {
    bake_pbr_texture_with_charts(surface, uvs, volume, texture_size, &ChartOptions::default())
}

/// Bake an existing mesh without changing its topology, with explicit chart controls.
/// # Errors
/// Reports invalid geometry, sparse buffers, chart controls or texture capacity.
pub fn bake_pbr_texture_with_charts(
    surface: &TriangleSurface,
    uvs: Option<&[[f32; 2]]>,
    volume: &SparsePbrVolume<'_>,
    texture_size: usize,
    charts: &ChartOptions,
) -> Result<TexturedSurface, GeometryError> {
    bake_pbr_texture_impl(surface, uvs, volume, texture_size, charts, None)
}

/// Unwrap and bake after projecting texel positions onto a reference surface.
/// # Errors
/// Reports invalid geometry, parameterization, sparse buffer or baking failures.
pub fn bake_pbr_texture_projected(
    surface: &TriangleSurface,
    projection_surface: &TriangleSurface,
    volume: &SparsePbrVolume<'_>,
    texture_size: usize,
    charts: &ChartOptions,
) -> Result<TexturedSurface, GeometryError> {
    let bvh = processing::SurfaceBvh::new(projection_surface)?;
    bake_pbr_texture_impl(surface, None, volume, texture_size, charts, Some(&bvh))
}

fn bake_pbr_texture_impl(
    surface: &TriangleSurface,
    uvs: Option<&[[f32; 2]]>,
    volume: &SparsePbrVolume<'_>,
    texture_size: usize,
    charts: &ChartOptions,
    projection: Option<&processing::SurfaceBvh<'_>>,
) -> Result<TexturedSurface, GeometryError> {
    surface.validate()?;
    let mut cleaned;
    let surface = if uvs.is_none() {
        cleaned = surface.clone();
        processing::remove_degenerate(&mut cleaned);
        processing::compact(&mut cleaned)?;
        &cleaned
    } else {
        surface
    };
    if texture_size < 2
        || volume.resolution == 0
        || volume.resolution > i32::MAX as usize
        || volume.coordinates.len() != volume.attributes.len()
        || volume.attributes.iter().flatten().any(|v| !v.is_finite())
        || surface.faces.is_empty()
    {
        return Err(GeometryError::Invalid(
            "texture geometry or sparse attribute rows",
        ));
    }
    let pixels = texture_size
        .checked_mul(texture_size)
        .ok_or(GeometryError::Invalid("texture size overflow"))?;
    pixels
        .checked_mul(4)
        .ok_or(GeometryError::Invalid("texture byte capacity"))?;
    let mut normals = vec![[0.0; 3]; surface.vertices.len()];
    for face in &surface.faces {
        let p = face.map(|index| surface.vertices[index as usize]);
        let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        for &index in face {
            for axis in 0..3 {
                normals[index as usize][axis] += normal[axis];
            }
        }
    }
    for normal in &mut normals {
        *normal = normalized(*normal);
    }
    let (output, uv, output_normals) = if let Some(uvs) = uvs {
        if uvs.len() != surface.vertices.len()
            || uvs.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(GeometryError::Invalid("UV buffer"));
        }
        (surface.clone(), uvs.to_vec(), normals)
    } else {
        let atlas = processing::unwrap(surface, charts, texture_size)?;
        let output_normals = atlas.vertex_map.iter().map(|&v| normals[v]).collect();
        (atlas.surface, atlas.uv, output_normals)
    };
    let lookup = volume
        .coordinates
        .iter()
        .enumerate()
        .map(|(row, &coord)| (coord, row))
        .collect::<BTreeMap<_, _>>();
    if lookup.len() != volume.coordinates.len()
        || volume
            .coordinates
            .iter()
            .flatten()
            .any(|&v| v < 0 || v >= volume.resolution as i64)
    {
        return Err(GeometryError::Invalid("sparse volume coordinates"));
    }
    let mut attributes = vec![[0.0; 6]; pixels];
    let mut mask = vec![true; pixels];
    for face in &output.faces {
        let triangle =
            face.map(|index| uv[index as usize].map(|value| value * texture_size as f32));
        let low: [usize; 2] = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.0) as usize
        });
        let high: [usize; 2] = std::array::from_fn(|axis| {
            (triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .max(0.0) as usize)
                .min(texture_size)
        });
        for y in low[1]..high[1] {
            for x in low[0]..high[0] {
                let Some(weights) = barycentric([x as f32 + 0.5, y as f32 + 0.5], triangle) else {
                    continue;
                };
                if weights.iter().any(|&value| value < 0.0) {
                    continue;
                }
                let position: [f32; 3] = std::array::from_fn(|axis| {
                    (0..3)
                        .map(|corner| {
                            output.vertices[face[corner] as usize][axis] * weights[corner]
                        })
                        .sum::<f32>()
                });
                let position = projection.map_or(position, |bvh| bvh.closest(position).0);
                let grid = position.map(|value| (value + 0.5) * volume.resolution as f32);
                // Far-outside queries have no volume support. Check before the
                // float-to-integer conversion so saturated i64s cannot overflow.
                if grid
                    .iter()
                    .any(|&v| !v.is_finite() || v < -1.0 || v > volume.resolution as f32 + 1.0)
                {
                    mask[y * texture_size + x] = false;
                    continue;
                }
                let base = grid.map(|value| (value - 0.5).floor() as i64);
                let mut value = [0.0; 6];
                let mut neighbors = [None; 8];
                let mut weights = [0.0; 8];
                let mut sum = 0.0;
                for child in 0..8 {
                    let coord: [i64; 3] =
                        std::array::from_fn(|axis| base[axis] + ((child >> axis) & 1) as i64);
                    if coord
                        .iter()
                        .any(|&v| v < 0 || v >= volume.resolution as i64)
                    {
                        continue;
                    }
                    let Some(&row) = lookup.get(&coord) else {
                        continue;
                    };
                    let per_axis: [f32; 3] = std::array::from_fn(|axis| {
                        1.0 - (grid[axis] - coord[axis] as f32 - 0.5).abs()
                    });
                    weights[child] = per_axis[0] * per_axis[1] * per_axis[2];
                    sum += weights[child];
                    neighbors[child] = Some(row);
                }
                for child in 0..8 {
                    if let Some(row) = neighbors[child] {
                        let weight = weights[child] / sum.max(1e-12);
                        for (channel, output) in value.iter_mut().enumerate() {
                            *output += volume.attributes[row][channel] * weight;
                        }
                    }
                }
                attributes[y * texture_size + x] = value;
                mask[y * texture_size + x] = false;
            }
        }
    }
    let mut color = Vec::with_capacity(pixels * 3);
    let mut metallic = Vec::with_capacity(pixels);
    let mut roughness = Vec::with_capacity(pixels);
    let mut alpha = Vec::with_capacity(pixels);
    for value in attributes {
        color.extend(
            value[..3]
                .iter()
                .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8),
        );
        metallic.push((value[3] * 255.0).clamp(0.0, 255.0) as u8);
        roughness.push((value[4] * 255.0).clamp(0.0, 255.0) as u8);
        alpha.push((value[5] * 255.0).clamp(0.0, 255.0) as u8);
    }
    inpaint_telea_u8(&mut color, &mask, texture_size, texture_size, 3, 3)?;
    inpaint_telea_u8(&mut metallic, &mask, texture_size, texture_size, 1, 1)?;
    inpaint_telea_u8(&mut roughness, &mask, texture_size, texture_size, 1, 1)?;
    inpaint_telea_u8(&mut alpha, &mask, texture_size, texture_size, 1, 1)?;
    let mut base_color = Vec::with_capacity(pixels * 4);
    let mut metallic_roughness = Vec::with_capacity(pixels * 3);
    for index in 0..pixels {
        base_color.extend_from_slice(&color[index * 3..index * 3 + 3]);
        base_color.push(alpha[index]);
        metallic_roughness.extend_from_slice(&[0, roughness[index], metallic[index]]);
    }
    Ok(TexturedSurface {
        surface: output,
        uvs: uv,
        normals: output_normals,
        texture_size,
        base_color,
        metallic_roughness,
        double_sided: true,
    })
}

impl TexturedSurface {
    /// Serialize a self-contained GLB with embedded PNG PBR textures.
    /// Positions/normals use the upstream Y/Z exchange; V is flipped for GLB.
    /// # Errors
    /// Reports invalid attribute buffers, size overflow, image or JSON encoding errors.
    pub fn to_glb(&self) -> Result<Vec<u8>, GeometryError> {
        self.surface.validate()?;
        let count = self.surface.vertices.len();
        if self.normals.len() != count || self.uvs.len() != count {
            return Err(GeometryError::Invalid("GLB vertex attributes"));
        }
        let width = u32::try_from(self.texture_size)
            .map_err(|_| GeometryError::Invalid("GLB texture width"))?;
        let color = image::RgbaImage::from_raw(width, width, self.base_color.clone())
            .ok_or(GeometryError::Invalid("base-color image"))?;
        let material = image::RgbImage::from_raw(width, width, self.metallic_roughness.clone())
            .ok_or(GeometryError::Invalid("metallic-roughness image"))?;
        let mut color_png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(color).write_to(&mut color_png, image::ImageFormat::Png)?;
        let mut material_png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(material)
            .write_to(&mut material_png, image::ImageFormat::Png)?;
        let mut binary = Vec::new();
        let mut views = Vec::new();
        let mut append = |data: &[u8], target: Option<u32>| {
            while binary.len() % 4 != 0 {
                binary.push(0);
            }
            let offset = binary.len();
            binary.extend_from_slice(data);
            let mut view =
                serde_json::json!({"buffer":0,"byteOffset":offset,"byteLength":data.len()});
            if let Some(target) = target {
                view["target"] = target.into();
            }
            views.push(view);
            views.len() - 1
        };
        let converted = self
            .surface
            .vertices
            .iter()
            .map(|v| [v[0], v[2], -v[1]])
            .collect::<Vec<_>>();
        let floats = |values: Vec<f32>| {
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let position = append(
            &floats(converted.iter().flatten().copied().collect()),
            Some(34962),
        );
        let normal = append(
            &floats(
                self.normals
                    .iter()
                    .flat_map(|v| [v[0], v[2], -v[1]])
                    .collect(),
            ),
            Some(34962),
        );
        let uv = append(
            &floats(self.uvs.iter().flat_map(|v| [v[0], 1.0 - v[1]]).collect()),
            Some(34962),
        );
        let faces = append(
            &self
                .surface
                .faces
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            Some(34963),
        );
        let color = append(color_png.get_ref(), None);
        let material = append(material_png.get_ref(), None);
        let min: [f32; 3] = std::array::from_fn(|axis| {
            converted
                .iter()
                .map(|v| v[axis])
                .fold(f32::INFINITY, f32::min)
        });
        let max: [f32; 3] = std::array::from_fn(|axis| {
            converted
                .iter()
                .map(|v| v[axis])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let json = serde_json::json!({
            "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
            "buffers":[{"byteLength":binary.len()}],"bufferViews":views,
            "accessors":[{"bufferView":position,"componentType":5126,"count":count,"type":"VEC3","min":min,"max":max},{"bufferView":normal,"componentType":5126,"count":count,"type":"VEC3"},{"bufferView":uv,"componentType":5126,"count":count,"type":"VEC2"},{"bufferView":faces,"componentType":5125,"count":self.surface.faces.len()*3,"type":"SCALAR"}],
            "images":[{"bufferView":color,"mimeType":"image/png"},{"bufferView":material,"mimeType":"image/png"}],"textures":[{"source":0},{"source":1}],
            "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"baseColorFactor":[1,1,1,1],"metallicRoughnessTexture":{"index":1},"metallicFactor":1,"roughnessFactor":1},"doubleSided":self.double_sided,"alphaMode":"OPAQUE"}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]}]
        });
        let mut json = serde_json::to_vec(&json)?;
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        while binary.len() % 4 != 0 {
            binary.push(0);
        }
        let total = u32::try_from(12usize + 8 + json.len() + 8 + binary.len())
            .map_err(|_| GeometryError::Invalid("GLB exceeds u32 capacity"))?;
        let mut output = Vec::with_capacity(total as usize);
        output.extend_from_slice(b"glTF");
        output.extend_from_slice(&2u32.to_le_bytes());
        output.extend_from_slice(&total.to_le_bytes());
        output.extend_from_slice(&(json.len() as u32).to_le_bytes());
        output.extend_from_slice(b"JSON");
        output.extend_from_slice(&json);
        output.extend_from_slice(&(binary.len() as u32).to_le_bytes());
        output.extend_from_slice(b"BIN\0");
        output.extend_from_slice(&binary);
        Ok(output)
    }
}

#[derive(Clone, Copy)]
struct Front {
    distance: f32,
    order: usize,
    row: usize,
    column: usize,
}
impl PartialEq for Front {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.order == other.order
    }
}
impl Eq for Front {}
impl PartialOrd for Front {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Front {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.order.cmp(&self.order))
    }
}

fn marching_distance(row: usize, column: usize, width: usize, flags: &[u8], time: &[f32]) -> f32 {
    let index = row * width + column;
    let solve = |a: usize, b: usize| {
        let x = f64::from(time[a]);
        let y = f64::from(time[b]);
        (if flags[a] != 2 && flags[b] != 2 {
            if (x - y).abs() >= 1.0 {
                1.0 + x.min(y)
            } else {
                (x + y + (2.0 - (x - y) * (x - y)).sqrt()) * 0.5
            }
        } else if flags[a] != 2 {
            1.0 + x
        } else if flags[b] != 2 {
            1.0 + y
        } else {
            1.0 + x.min(y)
        }) as f32
    };
    solve(index - width, index - 1)
        .min(solve(index + width, index - 1))
        .min(solve(index - width, index + 1))
        .min(solve(index + width, index + 1))
}

// The byte-gradient stencils and fast-marching arithmetic below follow the
// independent reference in OpenCV 4.12.0 modules/photo/src/inpaint.cpp.
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The name of Intel Corporation may not be used to endorse or promote
//   products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL CORPORATION OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/// Fill masked byte pixels by Telea's signed-distance fast-marching method.
///
/// This Rust implementation preserves raster-order heap ties, FP32 accumulation,
/// the source byte-image gradient stencils and the final `round(value + 0.5)`.
/// No native image-processing library is loaded.
/// # Errors
/// Reports invalid image/mask buffers, channel counts, or absent known pixels.
pub fn inpaint_telea_u8(
    values: &mut [u8],
    mask: &[bool],
    width: usize,
    height: usize,
    channels: usize,
    radius: usize,
) -> Result<(), GeometryError> {
    let pixels = width
        .checked_mul(height)
        .ok_or(GeometryError::Invalid("inpaint extent overflow"))?;
    let elements = pixels
        .checked_mul(channels)
        .ok_or(GeometryError::Invalid("inpaint buffer overflow"))?;
    if width == 0
        || height == 0
        || mask.len() != pixels
        || values.len() != elements
        || ![1, 3].contains(&channels)
    {
        return Err(GeometryError::Invalid("inpaint buffers"));
    }
    if !mask.iter().any(|&value| value) {
        return Ok(());
    }
    if mask.iter().all(|&value| value) {
        return Err(GeometryError::Invalid("inpaint has no known pixels"));
    }
    let stride = width
        .checked_add(2)
        .ok_or(GeometryError::Invalid("inpaint border overflow"))?;
    let rows = height
        .checked_add(2)
        .ok_or(GeometryError::Invalid("inpaint border overflow"))?;
    let count = stride
        .checked_mul(rows)
        .ok_or(GeometryError::Invalid("inpaint extent overflow"))?;
    let radius = radius.clamp(1, 100);
    let mut flags = vec![0u8; count];
    let mut time = vec![1e6f32; count];
    let mut initial = Vec::new();
    for y in 1..=height {
        for x in 1..=width {
            if mask[(y - 1) * width + x - 1] {
                flags[y * stride + x] = 2;
            }
        }
    }
    for y in 1..=height {
        for x in 1..=width {
            let index = y * stride + x;
            if flags[index] == 0
                && [index - stride, index - 1, index + stride, index + 1]
                    .iter()
                    .any(|&n| flags[n] == 2)
            {
                time[index] = 0.0;
                initial.push(Front {
                    distance: 0.0,
                    order: initial.len(),
                    row: y,
                    column: x,
                });
            }
        }
    }
    let mut exterior = vec![0u8; count];
    for y in 1..=height {
        for x in 1..=width {
            if flags[y * stride + x] == 2 || time[y * stride + x] == 0.0 {
                continue;
            }
            let found = (y.saturating_sub(radius).max(1)..=(y + radius).min(height)).any(|j| {
                (x.saturating_sub(radius).max(1)..=(x + radius).min(width))
                    .any(|i| flags[j * stride + i] == 2)
            });
            if found {
                exterior[y * stride + x] = 2;
            }
        }
    }
    let mut heap = BinaryHeap::from(initial.clone());
    let mut order = initial.len();
    while let Some(front) = heap.pop() {
        let index = front.row * stride + front.column;
        exterior[index] = 3;
        for (y, x) in [
            (front.row - 1, front.column),
            (front.row, front.column - 1),
            (front.row + 1, front.column),
            (front.row, front.column + 1),
        ] {
            if y == 0 || x == 0 || y > height || x > width || exterior[y * stride + x] != 2 {
                continue;
            }
            let distance = marching_distance(y, x, stride, &exterior, &time);
            time[y * stride + x] = distance;
            exterior[y * stride + x] = 1;
            heap.push(Front {
                distance,
                order,
                row: y,
                column: x,
            });
            order += 1;
        }
    }
    for index in 0..count {
        if exterior[index] == 3 {
            time[index] = -time[index];
        }
    }
    let mut heap = BinaryHeap::from(initial);
    let mut order = heap.len();
    while let Some(front) = heap.pop() {
        flags[front.row * stride + front.column] = 0;
        for (y, x) in [
            (front.row - 1, front.column),
            (front.row, front.column - 1),
            (front.row + 1, front.column),
            (front.row, front.column + 1),
        ] {
            let index = y * stride + x;
            if y == 0 || x == 0 || y > height || x > width || flags[index] != 2 {
                continue;
            }
            let distance = marching_distance(y, x, stride, &flags, &time);
            time[index] = distance;
            let gradient = |minus: usize, plus: usize| {
                if flags[plus] != 2 {
                    if flags[minus] != 2 {
                        (time[plus] - time[minus]) * 0.5
                    } else {
                        time[plus] - time[index]
                    }
                } else if flags[minus] != 2 {
                    time[index] - time[minus]
                } else {
                    0.0
                }
            };
            let grad_t = [
                gradient(index - 1, index + 1),
                gradient(index - stride, index + stride),
            ];
            for channel in 0..channels {
                let mut value = 0.0f32;
                let mut jx = 0.0f32;
                let mut jy = 0.0f32;
                let mut sum = 1e-20f32;
                for k in y.saturating_sub(radius).max(1)..=(y + radius).min(height) {
                    for l in x.saturating_sub(radius).max(1)..=(x + radius).min(width) {
                        let dy = y as isize - k as isize;
                        let dx = x as isize - l as isize;
                        let squared = dx * dx + dy * dy;
                        if flags[k * stride + l] == 2
                            || squared > (radius * radius) as isize
                            || squared == 0
                        {
                            continue;
                        }
                        let rx = dx as f32;
                        let ry = dy as f32;
                        let squared = rx * rx + ry * ry;
                        let dst = if channels == 3 {
                            (1.0 / (f64::from(squared) * f64::from(squared).sqrt())) as f32
                        } else {
                            1.0 / (squared * squared.sqrt())
                        };
                        let level = (1.0
                            / (1.0 + f64::from((time[k * stride + l] - time[index]).abs())))
                            as f32;
                        let mut direction = rx * grad_t[0] + ry * grad_t[1];
                        if direction.abs() <= 0.01 {
                            direction = 0.000001;
                        }
                        let weight = (dst * level * direction).abs();
                        let km = k as isize - 1 + isize::from(k == 1);
                        let kp = k as isize - 1 - isize::from(k == height);
                        let lm = l as isize - 1 + isize::from(l == 1);
                        let lp = l as isize - 1 - isize::from(l == width);
                        let pixel = |row: isize, column: isize| {
                            f32::from(
                                values[(row.clamp(0, height as isize - 1) as usize * width
                                    + column.clamp(0, width as isize - 1) as usize)
                                    * channels
                                    + channel],
                            )
                        };
                        let gx = if flags[k * stride + l + 1] != 2 {
                            if flags[k * stride + l - 1] != 2 {
                                (pixel(km, lp + 1) - pixel(km, lm - 1)) * 2.0
                            } else {
                                pixel(km, lp + 1) - pixel(km, lm)
                            }
                        } else if flags[k * stride + l - 1] != 2 {
                            pixel(km, lp) - pixel(km, lm - 1)
                        } else {
                            0.0
                        };
                        let gy = if flags[(k + 1) * stride + l] != 2 {
                            if flags[(k - 1) * stride + l] != 2 {
                                (pixel(kp + 1, lm) - pixel(km - 1, lm)) * 2.0
                            } else {
                                pixel(kp + 1, lm) - pixel(km, lm)
                            }
                        } else if flags[(k - 1) * stride + l] != 2 {
                            pixel(kp, lm) - pixel(km - 1, lm)
                        } else {
                            0.0
                        };
                        value += weight * pixel(k as isize - 1, l as isize - 1);
                        jx -= weight * (gx * rx);
                        jy -= weight * (gy * ry);
                        sum += weight;
                    }
                }
                let predicted = value / sum + (jx + jy) / ((jx * jx + jy * jy).sqrt() + 1e-20);
                values[((y - 1) * width + x - 1) * channels + channel] =
                    (f64::from(predicted) + 0.5)
                        .round_ties_even()
                        .clamp(0.0, 255.0) as u8;
            }
            flags[index] = 1;
            heap.push(Front {
                distance,
                order,
                row: y,
                column: x,
            });
            order += 1;
        }
    }
    Ok(())
}
