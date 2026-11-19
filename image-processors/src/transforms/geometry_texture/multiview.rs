//! Camera rendering and view-to-UV projection, independent of model execution.

use super::{
    cross, dot, length, normalized, processing, sub, vertex_normals, ChartOptions, GeometryError,
    NormalSource, TexturedSurface, TriangleSurface,
};

/// Inverse of mean-centering, maximum-absolute scaling and source axis conversion.
#[derive(Debug)]
pub struct MeshTransform {
    /// Original-space mean, or zero when centering is disabled.
    pub offset: [f64; 3],
    /// Canonical-to-original multiplicative scale.
    pub scale: f64,
    /// Whether the source's optional front-X to front-Y rotation was applied.
    pub front_x_to_y: bool,
}

impl MeshTransform {
    /// Restore an exported surface to its original orientation, size and location.
    pub fn restore(&self, surface: &mut TriangleSurface) {
        for point in &mut surface.vertices {
            let mut p = point.map(f64::from);
            if self.front_x_to_y {
                p = [-p[1], p[0], p[2]];
            }
            p = [p[0], p[2], -p[1]];
            *point = std::array::from_fn(|i| (p[i] * self.scale + self.offset[i]) as f32);
        }
    }

    /// Restore an exported asset, rotating normals along with its geometry.
    pub fn restore_textured(&self, asset: &mut TexturedSurface) {
        self.restore(&mut asset.surface);
        for normal in &mut asset.normals {
            let p = if self.front_x_to_y {
                [-normal[1], normal[0], normal[2]]
            } else {
                *normal
            };
            *normal = [p[0], p[2], -p[1]];
        }
    }
}

/// Apply the source +Y-up/+X-front mesh convention with NumPy-style FP64 preparation.
/// `half_extent` is the source maximum absolute coordinate after scaling (0.5).
/// # Errors
/// Rejects empty/nonfinite vertices, degenerate extent, invalid scale and face indices.
pub fn canonical_mesh(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    center: bool,
    half_extent: f64,
    front_x_to_y: bool,
) -> Result<(TriangleSurface, MeshTransform), GeometryError> {
    if vertices.is_empty()
        || vertices.iter().flatten().any(|v| !v.is_finite())
        || !half_extent.is_finite()
        || half_extent <= 0.0
    {
        return Err(GeometryError::Invalid("mesh normalization input"));
    }
    let offset: [f64; 3] = if center {
        std::array::from_fn(|i| vertices.iter().map(|v| v[i]).sum::<f64>() / vertices.len() as f64)
    } else {
        [0.0; 3]
    };
    let extent = vertices
        .iter()
        .flat_map(|v| (0..3).map(move |i| (v[i] - offset[i]).abs()))
        .fold(0.0, f64::max);
    if extent == 0.0 {
        return Err(GeometryError::Invalid("zero mesh extent"));
    }
    let scale = extent / half_extent;
    let vertices = vertices
        .iter()
        .map(|v| {
            let p: [f64; 3] = std::array::from_fn(|i| (v[i] - offset[i]) / extent * half_extent);
            let p = [p[0], -p[2], p[1]];
            let p = if front_x_to_y { [p[1], -p[0], p[2]] } else { p };
            p.map(|v| v as f32)
        })
        .collect();
    let surface = TriangleSurface {
        vertices,
        faces: faces.to_vec(),
    };
    surface.validate()?;
    Ok((
        surface,
        MeshTransform {
            offset,
            scale,
            front_x_to_y,
        },
    ))
}

/// Camera basis, matching the right/up/back columns of a camera-to-world matrix.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World-space camera center.
    pub position: [f32; 3],
    /// World-space local X direction.
    pub right: [f32; 3],
    /// World-space local Y direction.
    pub up: [f32; 3],
    /// World-space local Z direction, away from the scene.
    pub back: [f32; 3],
    /// Full orthographic view extent. Ignored for perspective views.
    pub scale: f32,
    /// Perspective vertical field of view, or orthographic when absent.
    pub fov_degrees: Option<f32>,
}

impl Camera {
    /// Construct an orbit camera looking at the origin with world Z upward.
    /// # Errors
    /// Rejects nonfinite, singular or nonpositive camera parameters.
    pub fn orbit(
        elevation: f32,
        azimuth: f32,
        distance: f32,
        scale: f32,
        fov_degrees: Option<f32>,
    ) -> Result<Self, GeometryError> {
        if ![elevation, azimuth, distance, scale]
            .iter()
            .all(|x| x.is_finite())
            || distance <= 0.0
            || scale <= 0.0
            || fov_degrees.is_some_and(|v| !v.is_finite() || v <= 0.0 || v >= 180.0)
        {
            return Err(GeometryError::Invalid("camera parameters"));
        }
        let (e, a) = (elevation.to_radians(), azimuth.to_radians());
        let position = [
            distance * e.cos() * a.cos(),
            distance * e.cos() * a.sin(),
            distance * e.sin(),
        ];
        let back = normalized(position);
        let right = normalized(cross(back.map(|v| -v), [0.0, 0.0, 1.0]));
        if length(right) == 0.0 {
            return Err(GeometryError::Invalid("singular orbit camera"));
        }
        let up = normalized(cross(right, back.map(|v| -v)));
        Ok(Self {
            position,
            right,
            up,
            back,
            scale,
            fov_degrees,
        })
    }

    fn project(&self, point: [f32; 3], width: usize, height: usize) -> [f32; 3] {
        let offset = sub(point, self.position);
        let z = -dot(offset, self.back);
        let extent = self
            .fov_degrees
            .map_or(self.scale, |f| 2.0 * z * (f.to_radians() * 0.5).tan());
        [
            ((dot(offset, self.right) / extent) + 0.5) * width as f32 - 0.5,
            (0.5 - dot(offset, self.up) / extent) * height as f32 - 0.5,
            z,
        ]
    }

    /// Upstream orthographic direction/normalized-position map in NCHW order.
    /// # Errors
    /// Rejects perspective cameras and empty or overflowing dimensions.
    pub fn plucker(&self, size: usize) -> Result<Vec<f32>, GeometryError> {
        let pixels = size
            .checked_mul(size)
            .filter(|&n| n > 0)
            .ok_or(GeometryError::Invalid("view size"))?;
        if self.fov_degrees.is_some() {
            return Err(GeometryError::Invalid(
                "orthographic Plucker camera required",
            ));
        }
        let mut output = vec![
            0.0;
            pixels
                .checked_mul(6)
                .ok_or(GeometryError::Invalid("view allocation"))?
        ];
        let direction = self.back.map(|v| -v);
        let position = normalized(self.position);
        for y in 0..size {
            for x in 0..size {
                // Despite the upstream function's name, its orthographic branch
                // repeats direction and normalized camera position, not ray moments.
                for c in 0..3 {
                    output[c * pixels + y * size + x] =
                        ((direction[c] + 1.0) * 0.5).clamp(0.0, 1.0);
                    output[(c + 3) * pixels + y * size + x] =
                        ((position[c] + 1.0) * 0.5).clamp(0.0, 1.0);
                }
            }
        }
        Ok(output)
    }
}

/// Interpolated image-space geometry in row-major order.
#[derive(Debug)]
pub struct ViewGeometry {
    /// Square image extent.
    pub size: usize,
    /// Foreground coverage.
    pub mask: Vec<bool>,
    /// World-space positions; zero on background.
    pub positions: Vec<[f32; 3]>,
    /// Smooth unit vertex normals; zero on background.
    pub normals: Vec<[f32; 3]>,
    /// Positive camera-space distance; infinity on background.
    pub depth: Vec<f32>,
    /// Nearest triangle, or background.
    pub triangles: Vec<Option<usize>>,
    /// Perspective-correct barycentric interpolation weights.
    pub barycentrics: Vec<[f32; 3]>,
}

impl ViewGeometry {
    /// Six-channel position/normal control tensor in NCHW order.
    pub fn conditioning(&self) -> Vec<f32> {
        let pixels = self.size * self.size;
        let mut output = vec![0.0; 6 * pixels];
        for i in 0..pixels {
            for c in 0..3 {
                output[c * pixels + i] = (self.positions[i][c] + 0.5).clamp(0.0, 1.0);
                output[(c + 3) * pixels + i] = (self.normals[i][c] * 0.5 + 0.5).clamp(0.0, 1.0);
            }
        }
        output
    }
}

fn barycentric(p: [f32; 2], triangle: [[f32; 2]; 3]) -> Option<[f32; 3]> {
    let weights = super::barycentric_with_minimum_area(p, triangle, f32::EPSILON)?;
    weights.iter().all(|&v| v >= 0.0).then_some(weights)
}

/// Render world positions and smooth normals without an external rasterizer.
/// # Errors
/// Rejects invalid surface data and empty or overflowing view sizes.
pub fn render(
    surface: &TriangleSurface,
    camera: &Camera,
    size: usize,
    normal_source: NormalSource<'_>,
) -> Result<ViewGeometry, GeometryError> {
    surface.validate()?;
    let count = size
        .checked_mul(size)
        .filter(|&n| n > 0)
        .ok_or(GeometryError::Invalid("render size"))?;
    let normals = vertex_normals(surface, normal_source)?;
    let mut output = ViewGeometry {
        size,
        mask: vec![false; count],
        positions: vec![[0.0; 3]; count],
        normals: vec![[0.0; 3]; count],
        depth: vec![f32::INFINITY; count],
        triangles: vec![None; count],
        barycentrics: vec![[0.0; 3]; count],
    };
    for (triangle_index, face) in surface.faces.iter().enumerate() {
        let vertices = face.map(|i| surface.vertices[i as usize]);
        let projected = vertices.map(|p| camera.project(p, size, size));
        if projected
            .iter()
            .any(|p| p[2] <= 0.0 || p.iter().any(|v| !v.is_finite()))
        {
            continue;
        }
        let lo: [usize; 2] = std::array::from_fn(|i| {
            projected
                .iter()
                .map(|p| p[i])
                .fold(f32::INFINITY, f32::min)
                .ceil()
                .max(0.0) as usize
        });
        let hi: [usize; 2] = std::array::from_fn(|i| {
            projected
                .iter()
                .map(|p| p[i])
                .fold(f32::NEG_INFINITY, f32::max)
                .floor()
                .max(0.0)
                .min((size - 1) as f32) as usize
        });
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let Some(mut weights) =
                    barycentric([x as f32, y as f32], projected.map(|p| [p[0], p[1]]))
                else {
                    continue;
                };
                if camera.fov_degrees.is_some() {
                    for i in 0..3 {
                        weights[i] /= projected[i][2];
                    }
                    let sum = weights.iter().sum::<f32>();
                    weights = weights.map(|v| v / sum);
                }
                let depth = (0..3).map(|i| weights[i] * projected[i][2]).sum::<f32>();
                let pixel = y * size + x;
                if depth >= output.depth[pixel] {
                    continue;
                }
                output.depth[pixel] = depth;
                output.mask[pixel] = true;
                output.triangles[pixel] = Some(triangle_index);
                output.barycentrics[pixel] = weights;
                output.positions[pixel] =
                    std::array::from_fn(|c| (0..3).map(|i| weights[i] * vertices[i][c]).sum());
                output.normals[pixel] = normalized(std::array::from_fn(|c| {
                    (0..3)
                        .map(|i| weights[i] * normals[face[i] as usize][c])
                        .sum()
                }));
            }
        }
    }
    Ok(output)
}

/// UV surface and its per-texel interpolated positions, normals and coverage.
#[derive(Debug)]
pub struct Atlas {
    /// Seam-expanded geometry and UV coordinates supplied by the existing chart owner.
    pub texture: TexturedSurface,
    /// World positions in image row order.
    pub positions: Vec<[f32; 3]>,
    /// World normals in image row order.
    pub normals: Vec<[f32; 3]>,
    /// Texels lying inside a UV triangle.
    pub mask: Vec<bool>,
}

/// Chart a mesh using the existing unwrap implementation, then interpolate UV geometry.
/// Existing UV coordinates may be supplied to preserve an application's atlas.
/// # Errors
/// Rejects invalid geometry, UV lengths, texture sizes and charting failures.
pub fn atlas(
    surface: &TriangleSurface,
    uvs: Option<&[[f32; 2]]>,
    size: usize,
    charts: &ChartOptions,
) -> Result<Atlas, GeometryError> {
    atlas_with_normals(surface, uvs, size, charts, NormalSource::AreaWeighted)
}

/// Chart a mesh while retaining selected vertex normals through UV seam expansion.
/// # Errors
/// Rejects invalid geometry, supplied normals, UV storage or charting failures.
pub fn atlas_with_normals(
    surface: &TriangleSurface,
    uvs: Option<&[[f32; 2]]>,
    size: usize,
    charts: &ChartOptions,
    normal_source: NormalSource<'_>,
) -> Result<Atlas, GeometryError> {
    surface.validate()?;
    let count = size
        .checked_mul(size)
        .filter(|&n| n >= 4)
        .ok_or(GeometryError::Invalid("atlas size"))?;
    let source_normals = vertex_normals(surface, normal_source)?;
    let (surface, uvs, normals) = if let Some(uvs) = uvs {
        if uvs.len() != surface.vertices.len() || uvs.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GeometryError::Invalid("UV coordinates"));
        }
        (surface.clone(), uvs.to_vec(), source_normals)
    } else {
        let result = processing::unwrap(surface, charts, size)?;
        let normals = result
            .vertex_map
            .iter()
            .map(|&i| source_normals[i])
            .collect();
        (result.surface, result.uv, normals)
    };
    let mut positions = vec![[0.0; 3]; count];
    let mut normal_map = vec![[0.0; 3]; count];
    let mut mask = vec![false; count];
    for face in &surface.faces {
        let triangle = face.map(|i| {
            let uv = uvs[i as usize];
            [uv[0] * size as f32 - 0.5, (1.0 - uv[1]) * size as f32 - 0.5]
        });
        let lo: [usize; 2] = std::array::from_fn(|i| {
            triangle
                .iter()
                .map(|p| p[i])
                .fold(f32::INFINITY, f32::min)
                .ceil()
                .max(0.0) as usize
        });
        let hi: [usize; 2] = std::array::from_fn(|i| {
            triangle
                .iter()
                .map(|p| p[i])
                .fold(f32::NEG_INFINITY, f32::max)
                .floor()
                .max(0.0)
                .min((size - 1) as f32) as usize
        });
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let Some(weights) = barycentric([x as f32, y as f32], triangle) else {
                    continue;
                };
                let pixel = y * size + x;
                mask[pixel] = true;
                positions[pixel] = std::array::from_fn(|c| {
                    (0..3)
                        .map(|i| weights[i] * surface.vertices[face[i] as usize][c])
                        .sum()
                });
                normal_map[pixel] = normalized(std::array::from_fn(|c| {
                    (0..3)
                        .map(|i| weights[i] * normals[face[i] as usize][c])
                        .sum()
                }));
            }
        }
    }
    Ok(Atlas {
        texture: TexturedSurface {
            surface,
            uvs,
            normals,
            texture_size: size,
            base_color: vec![0; count * 4],
            metallic_roughness: [0, 255, 0].repeat(count),
            double_sided: true,
        },
        positions,
        normals: normal_map,
        mask,
    })
}

/// Bilinear zero-padded sampling with align_corners=false pixel coordinates.
pub fn sample<const N: usize>(image: &[[f32; N]], size: usize, x: f32, y: f32) -> [f32; N] {
    sample_filtered(image, size, x, y, false, false)
}

/// Bilinearly sample a repeating UV texture at pixel-center coordinates.
/// Invalid coordinates or storage return zero, as in [`sample`].
pub fn sample_repeat<const N: usize>(image: &[[f32; N]], size: usize, x: f32, y: f32) -> [f32; N] {
    sample_filtered(image, size, x, y, true, false)
}

/// Sample the nearest texel with repeating UV boundaries for source score maps.
/// Invalid coordinates or storage return zero, as in [`sample`].
pub fn sample_repeat_nearest<const N: usize>(
    image: &[[f32; N]],
    size: usize,
    x: f32,
    y: f32,
) -> [f32; N] {
    sample_filtered(image, size, x, y, true, true)
}

fn sample_filtered<const N: usize>(
    image: &[[f32; N]],
    size: usize,
    x: f32,
    y: f32,
    repeat: bool,
    nearest: bool,
) -> [f32; N] {
    let mut result = [0.0; N];
    if size == 0 || !x.is_finite() || !y.is_finite() || size.checked_mul(size) != Some(image.len())
    {
        return result;
    }
    let side = size as f32;
    let (x, y) = if repeat {
        (x.rem_euclid(side), y.rem_euclid(side))
    } else {
        if x < -1.0 || y < -1.0 || x >= side || y >= side {
            return result;
        }
        (x, y)
    };
    if nearest {
        let ix = ((x + 0.5).floor() as usize) % size;
        let iy = ((y + 0.5).floor() as usize) % size;
        return image[iy * size + ix];
    }
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    let (dx, dy) = (x - x.floor(), y - y.floor());
    for (oy, wy) in [(0, 1.0 - dy), (1, dy)] {
        for (ox, wx) in [(0, 1.0 - dx), (1, dx)] {
            let (ix, iy) = if repeat {
                (
                    (x0 + ox).rem_euclid(size as isize),
                    (y0 + oy).rem_euclid(size as isize),
                )
            } else {
                let (ix, iy) = (x0 + ox, y0 + oy);
                if ix < 0 || iy < 0 || ix >= size as isize || iy >= size as isize {
                    continue;
                }
                (ix, iy)
            };
            let pixel = image[iy as usize * size + ix as usize];
            for (value, &channel) in result.iter_mut().zip(&pixel) {
                *value += channel * wx * wy;
            }
        }
    }
    result
}

/// One view projected into an atlas with source visibility and incidence tests.
#[derive(Debug)]
pub struct Projection {
    /// Bilinearly projected RGB.
    pub colors: Vec<[f32; 3]>,
    /// Source validity tests, including the UV triangle mask.
    pub valid: Vec<bool>,
    /// Camera-facing normal cosine.
    pub incidence: Vec<f32>,
}

/// Source visibility and depth-discontinuity rejection controls.
#[derive(Debug)]
pub struct ProjectionOptions {
    /// Maximum bilinear world-position reprojection error.
    pub position_tolerance: f32,
    /// Minimum camera-facing normal cosine.
    pub incidence_threshold: f32,
    /// Maximum bilinear depth-gradient magnitude, when selected.
    pub depth_gradient_threshold: Option<f32>,
    /// Odd max-pool footprint for depth-gradient rejection.
    pub depth_gradient_dilation: usize,
}

fn depth_gradients(depth: &[f32], size: usize) -> Vec<f32> {
    let mut result = vec![0.0; depth.len()];
    for y in 0..size {
        for x in 0..size {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for ky in 0_isize..3 {
                for kx in 0_isize..3 {
                    let (ix, iy) = (x as isize + kx - 1, y as isize + ky - 1);
                    if ix < 0 || iy < 0 || ix >= size as isize || iy >= size as isize {
                        continue;
                    }
                    let d = depth[iy as usize * size + ix as usize];
                    gx += d * (1 - kx) as f32 * if ky == 1 { 2.0 } else { 1.0 };
                    gy += d * (1 - ky) as f32 * if kx == 1 { 2.0 } else { 1.0 };
                }
            }
            result[y * size + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    result
}

/// Project a view using explicit source visibility and depth-discontinuity controls.
/// # Errors
/// Rejects image/geometry size mismatches and invalid atlas buffers.
pub fn project(
    atlas: &Atlas,
    camera: &Camera,
    geometry: &ViewGeometry,
    colors: &[[f32; 3]],
    options: &ProjectionOptions,
) -> Result<Projection, GeometryError> {
    if colors.len() != geometry.mask.len() {
        return Err(GeometryError::Invalid("projection image size"));
    }
    let normal_cos = geometry
        .normals
        .iter()
        .map(|n| [dot(*n, camera.back).clamp(0.0, 1.0)])
        .collect::<Vec<_>>();
    let gradients = if options.depth_gradient_threshold.is_some() {
        let depth = geometry
            .depth
            .iter()
            .map(|&v| if v.is_finite() { v } else { 100.0 })
            .collect::<Vec<_>>();
        let gradients = depth_gradients(&depth, geometry.size);
        let side = options.depth_gradient_dilation;
        let pooled = super::super::dilate_f32(
            &gradients,
            super::super::ImageSize {
                width: geometry.size,
                height: geometry.size,
            },
            &vec![
                1;
                side.checked_mul(side)
                    .ok_or(GeometryError::Invalid("depth footprint"))?
            ],
            super::super::ImageSize {
                width: side,
                height: side,
            },
        )
        .map_err(|e| GeometryError::Completion(e.to_string()))?;
        Some(pooled.into_iter().map(|v| [v]).collect::<Vec<_>>())
    } else {
        None
    };
    let mut projection = Projection {
        colors: Vec::with_capacity(atlas.mask.len()),
        valid: Vec::with_capacity(atlas.mask.len()),
        incidence: Vec::with_capacity(atlas.mask.len()),
    };
    for (i, point) in atlas.positions.iter().enumerate() {
        let p = camera.project(*point, geometry.size, geometry.size);
        let world = sample(&geometry.positions, geometry.size, p[0], p[1]);
        let cosine = sample(&normal_cos, geometry.size, p[0], p[1])[0];
        projection
            .colors
            .push(sample(colors, geometry.size, p[0], p[1]));
        let depth_valid = match (&gradients, options.depth_gradient_threshold) {
            (Some(gradient), Some(threshold)) => {
                sample(gradient, geometry.size, p[0], p[1])[0] < threshold
            }
            _ => true,
        };
        projection.valid.push(
            atlas.mask[i]
                && p[2] > 0.0
                && length(sub(world, *point)) < options.position_tolerance
                && cosine > options.incidence_threshold
                && depth_valid,
        );
        projection.incidence.push(cosine);
    }
    Ok(projection)
}

/// Blend valid view colors with cosine.powf(alpha) weights and first-view priority.
/// # Errors
/// Rejects inconsistent view buffers or nonfinite alpha.
pub fn blend(
    views: &[Projection],
    alpha: f32,
    first_view_dominate: bool,
) -> Result<(Vec<[f32; 3]>, Vec<bool>), GeometryError> {
    let count = views.first().map_or(0, |v| v.colors.len());
    if !alpha.is_finite()
        || views.iter().any(|v| {
            v.colors.len() != count || v.valid.len() != count || v.incidence.len() != count
        })
    {
        return Err(GeometryError::Invalid("view blend buffers"));
    }
    let mut colors = vec![[0.0; 3]; count];
    let mut valid = vec![false; count];
    for i in 0..count {
        let mut denominator = 0.0;
        for (view_index, view) in views.iter().enumerate() {
            if !view.valid[i] || (first_view_dominate && view_index > 0 && views[0].valid[i]) {
                continue;
            }
            let weight = view.incidence[i].powf(alpha);
            denominator += weight;
            for (channel, &value) in colors[i].iter_mut().zip(&view.colors[i]) {
                *channel += value * weight;
            }
        }
        if denominator > 0.0 {
            colors[i] = colors[i].map(|v| v / denominator.max(1e-5));
            valid[i] = true;
        }
    }
    Ok((colors, valid))
}

/// Pad unknown texture pixels using the existing byte Telea owner, radius three.
/// # Errors
/// Rejects inconsistent buffers or an image with no known content.
pub fn pad(colors: &mut [[f32; 3]], known: &[bool], size: usize) -> Result<(), GeometryError> {
    let mut bytes = colors
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8))
        .collect::<Vec<_>>();
    let missing = known.iter().map(|v| !v).collect::<Vec<_>>();
    super::inpaint_telea_u8(&mut bytes, &missing, size, size, 3, 3)?;
    for (color, bytes) in colors.iter_mut().zip(bytes.as_chunks::<3>().0) {
        *color = std::array::from_fn(|c| f32::from(bytes[c]) / 255.0);
    }
    Ok(())
}

/// Source-gradient Poisson blending with fixed borders and simultaneous Jacobi updates.
/// # Errors
/// Rejects incompatible images or masks.
pub fn poisson_blend(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
    mask: &[bool],
    size: usize,
    iterations: usize,
) -> Result<Vec<[f32; 3]>, GeometryError> {
    if size < 3
        || source.len() != size * size
        || target.len() != source.len()
        || mask.len() != source.len()
    {
        return Err(GeometryError::Invalid("Poisson image buffers"));
    }
    let mut result = target.to_vec();
    let mut next = result.clone();
    for _ in 0..iterations {
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                let i = y * size + x;
                if !mask[i] {
                    continue;
                }
                let neighbors = [i - size, i + size, i - 1, i + 1];
                for c in 0..3 {
                    let laplacian =
                        4.0 * source[i][c] - neighbors.iter().map(|&j| source[j][c]).sum::<f32>();
                    next[i][c] =
                        (neighbors.iter().map(|&j| result[j][c]).sum::<f32>() + laplacian) / 4.0;
                }
            }
        }
        std::mem::swap(&mut result, &mut next);
    }
    // The upstream solver clamps only solved interior pixels when copying its
    // result back; retain unconstrained boundary and unselected target values.
    for y in 1..size - 1 {
        for x in 1..size - 1 {
            let i = y * size + x;
            if mask[i] {
                result[i] = result[i].map(|v| v.clamp(0.0, 1.0));
            }
        }
    }
    Ok(result)
}

fn texture_coordinate(atlas: &Atlas, view: &ViewGeometry, pixel: usize) -> Option<[f32; 2]> {
    let face = atlas.texture.surface.faces[view.triangles[pixel]?];
    let weights = view.barycentrics[pixel];
    let uv: [f32; 2] = std::array::from_fn(|c| {
        (0..3)
            .map(|i| weights[i] * atlas.texture.uvs[face[i] as usize][c])
            .sum()
    });
    let size = atlas.texture.texture_size as f32;
    Some([uv[0] * size - 0.5, (1.0 - uv[1]) * size - 0.5])
}

/// Source view-selection and iterative learned view completion controls.
#[derive(Debug)]
pub struct ViewCompletion {
    /// Minimum source completion rounds.
    pub min_rounds: usize,
    /// Maximum source completion rounds.
    pub max_rounds: usize,
    /// Stop when the best view score falls below this threshold after minimum rounds.
    pub score_threshold: f32,
    /// Learned completion view resolution.
    pub size: usize,
}

/// Complete UV holes by choosing and inpainting the most useful perspective view.
/// The callback owns the learned network; this module owns geometry and masks.
/// # Errors
/// Reports invalid controls, rendering, projection, padding or callback errors.
pub fn complete_views(
    atlas: &Atlas,
    colors: &mut [[f32; 3]],
    known: &mut [bool],
    options: &ViewCompletion,
    mut inpaint: impl FnMut(&[[f32; 3]], &[bool], usize) -> Result<Vec<[f32; 3]>, GeometryError>,
) -> Result<(), GeometryError> {
    if options.min_rounds > options.max_rounds
        || options.size == 0
        || !options.score_threshold.is_finite()
    {
        return Err(GeometryError::Invalid("view completion controls"));
    }
    let mut cameras = Vec::new();
    for elevation in (-60..=60).step_by(15) {
        for azimuth in (0..360).step_by(30) {
            cameras.push(Camera::orbit(
                elevation as f32,
                azimuth as f32,
                1.2,
                1.1,
                Some(40.0),
            )?);
        }
    }
    let atlas_size = atlas.texture.texture_size;
    let mut scores = known
        .iter()
        .map(|&v| [if v { 1.0 } else { 0.0 }])
        .collect::<Vec<_>>();
    let mut previous_score = 1.0;
    for round in 0..options.max_rounds {
        if round >= options.min_rounds && previous_score <= options.score_threshold {
            break;
        }
        let mut best = (0, f32::NEG_INFINITY);
        for (index, camera) in cameras.iter().enumerate() {
            let view = render(
                &atlas.texture.surface,
                camera,
                256,
                NormalSource::Supplied(&atlas.texture.normals),
            )?;
            let mut score = 0.0;
            for pixel in 0..view.mask.len() {
                let Some(uv) = texture_coordinate(atlas, &view, pixel) else {
                    continue;
                };
                let attr = sample_repeat_nearest(&scores, atlas_size, uv[0], uv[1])[0];
                let cosine = dot(view.normals[pixel], camera.back).clamp(0.0, 1.0);
                if cosine > 0.1 {
                    score += if attr < 1e-3 {
                        1.0
                    } else if attr > 1e-3 {
                        (cosine - attr - 0.3).max(0.0)
                    } else {
                        0.0
                    };
                }
            }
            score /= 256.0 * 256.0;
            if score > best.1 {
                best = (index, score);
            }
        }
        previous_score = best.1;
        let camera = &cameras[best.0];
        let size = options.size;
        let view = render(
            &atlas.texture.surface,
            camera,
            size,
            NormalSource::Supplied(&atlas.texture.normals),
        )?;
        let mut image = vec![[0.5; 3]; size * size];
        let mut mask = vec![false; size * size];
        for pixel in 0..mask.len() {
            let Some(uv) = texture_coordinate(atlas, &view, pixel) else {
                continue;
            };
            image[pixel] = sample_repeat(colors, atlas_size, uv[0], uv[1]);
            let score = sample_repeat_nearest(&scores, atlas_size, uv[0], uv[1])[0];
            let cosine = dot(view.normals[pixel], camera.back).clamp(0.0, 1.0);
            mask[pixel] = score < 1e-3 || cosine - score > 0.3;
        }
        // Source max-pool erosion radius 3 followed by dilation radius 5.
        let dimensions = super::super::ImageSize {
            width: size,
            height: size,
        };
        let inverted = mask
            .iter()
            .map(|&v| if v { -1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let eroded = super::super::dilate_f32(
            &inverted,
            dimensions,
            &[1; 49],
            super::super::ImageSize {
                width: 7,
                height: 7,
            },
        )
        .map_err(|e| GeometryError::Completion(e.to_string()))?;
        let eroded = eroded.iter().map(|v| -v).collect::<Vec<_>>();
        let enlarged = super::super::dilate_f32(
            &eroded,
            dimensions,
            &[1; 121],
            super::super::ImageSize {
                width: 11,
                height: 11,
            },
        )
        .map_err(|e| GeometryError::Completion(e.to_string()))?;
        mask = enlarged.iter().map(|&v| v > 0.0).collect();
        // The completion render uses DepthControlNetNormalization (0.25..1).
        let minimum = view
            .depth
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(length(camera.position), f32::min);
        let maximum = view
            .depth
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(minimum, f32::max);
        let depth = view
            .depth
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    (1.0 - ((v - minimum) / (maximum - minimum + 1e-5)).clamp(0.0, 1.0)) * 0.75
                        + 0.25
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let gradients = depth_gradients(&depth, size);
        for (missing, gradient) in mask.iter_mut().zip(gradients) {
            if gradient > 0.1 {
                *missing = false;
            }
        }
        let filled = inpaint(&image, &mask, size)?;
        if filled.len() != image.len() {
            return Err(GeometryError::Invalid("completed view size"));
        }
        let projection = project(
            atlas,
            camera,
            &view,
            &filled,
            &ProjectionOptions {
                position_tolerance: 1e-3,
                incidence_threshold: 0.1,
                depth_gradient_threshold: Some(0.1),
                depth_gradient_dilation: 3,
            },
        )?;
        let mask_values = mask
            .iter()
            .map(|&v| [if v { 1.0 } else { 0.0 }])
            .collect::<Vec<_>>();
        for i in 0..known.len() {
            let point = camera.project(atlas.positions[i], size, size);
            if projection.valid[i] && sample(&mask_values, size, point[0], point[1])[0] > 0.9 {
                colors[i] = projection.colors[i];
                known[i] = true;
                scores[i][0] = scores[i][0].max(projection.incidence[i]);
            }
        }
        pad(colors, &atlas.mask, atlas_size)?;
    }
    Ok(())
}

/// Pad completed texture colors and store the shared export owner's RGBA buffer.
/// # Errors
/// Rejects incompatible atlas buffers or padding failures.
pub fn finish_texture(
    atlas: &mut Atlas,
    colors: &[[f32; 3]],
    known: &[bool],
) -> Result<(), GeometryError> {
    if colors.len() != atlas.mask.len() || known.len() != colors.len() {
        return Err(GeometryError::Invalid("texture buffers"));
    }
    let mut colors = colors.to_vec();
    pad(&mut colors, known, atlas.texture.texture_size)?;
    atlas.texture.base_color = colors
        .iter()
        .flat_map(|p| {
            [
                (p[0].clamp(0.0, 1.0) * 255.0) as u8,
                (p[1].clamp(0.0, 1.0) * 255.0) as u8,
                (p[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ]
        })
        .collect();
    Ok(())
}

#[cfg(test)]
mod coverage_tests {
    #[test]
    fn central_weights_retain_bake_and_multiview_policies() {
        let triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(
            super::barycentric([0.25, 0.25], triangle),
            Some([0.5, 0.25, 0.25])
        );
        assert!(super::super::barycentric([-0.01, 0.25], triangle).is_some());
        assert!(super::barycentric([-0.01, 0.25], triangle).is_none());
        for area in [f32::EPSILON * 0.5, f32::EPSILON, f32::EPSILON * 2.0] {
            let triangle = [[0.0, 0.0], [area, 0.0], [0.0, 1.0]];
            let point = [area * 0.25, 0.25];
            assert!(super::super::barycentric(point, triangle).is_some());
            assert_eq!(
                super::barycentric(point, triangle).is_some(),
                area > f32::EPSILON
            );
        }
        assert!(super::super::barycentric([0.0, 0.0], [[0.0; 2]; 3]).is_none());
        assert!(super::barycentric([0.0, 0.0], [[0.0; 2]; 3]).is_none());
    }
}
