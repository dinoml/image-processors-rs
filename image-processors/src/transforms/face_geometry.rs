//! Similarity estimation and byte RGB affine resampling in Rust.

use super::ImageSize;

/// Invalid landmark or affine-image input.
#[derive(Debug, Clone, Copy)]
pub enum FaceGeometryError {
    /// Invalid, insufficient or degenerate corresponding landmarks.
    Landmarks,
    /// Invalid image storage or dimensions.
    Geometry,
    /// Noninvertible affine transform.
    Singular,
}
impl std::fmt::Display for FaceGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Landmarks => "invalid face landmarks",
            Self::Geometry => "invalid affine image geometry",
            Self::Singular => "singular affine transform",
        })
    }
}
impl std::error::Error for FaceGeometryError {}

/// Fit a two-dimensional, orientation-preserving similarity transform to all pairs.
/// The returned matrix maps source pixel coordinates to destination coordinates.
///
/// # Errors
/// Rejects mismatched, nonfinite, insufficient or degenerate landmark pairs.
pub fn estimate_similarity(
    from: &[[f32; 2]],
    to: &[[f32; 2]],
) -> Result<[[f64; 3]; 2], FaceGeometryError> {
    if from.len() != to.len()
        || from.len() < 2
        || from.iter().chain(to).flatten().any(|x| !x.is_finite())
    {
        return Err(FaceGeometryError::Landmarks);
    }
    let n = from.len() as f64;
    let center = |points: &[[f32; 2]]| {
        [
            points.iter().map(|p| f64::from(p[0])).sum::<f64>() / n,
            points.iter().map(|p| f64::from(p[1])).sum::<f64>() / n,
        ]
    };
    let source = center(from);
    let target = center(to);
    let (mut real, mut imaginary, mut energy) = (0.0, 0.0, 0.0);
    for (p, q) in from.iter().zip(to) {
        let x = f64::from(p[0]) - source[0];
        let y = f64::from(p[1]) - source[1];
        let u = f64::from(q[0]) - target[0];
        let v = f64::from(q[1]) - target[1];
        real += x * u + y * v;
        imaginary += x * v - y * u;
        energy += x * x + y * y;
    }
    if energy <= f64::EPSILON {
        return Err(FaceGeometryError::Landmarks);
    }
    let a = real / energy;
    let b = imaginary / energy;
    Ok([
        [a, -b, target[0] - a * source[0] + b * source[1]],
        [b, a, target[1] - b * source[0] - a * source[1]],
    ])
}

/// Robust least-median similarity estimation for corresponding face landmarks.
///
/// Uses two-point hypotheses, deterministic multiply-with-carry sampling,
/// squared FP32 residuals, the least-median scale estimate and inlier refinement.
///
/// # Errors
/// Rejects nonfinite, mismatched, insufficient or degenerate landmarks/controls.
pub fn estimate_similarity_lmeds(
    from: &[[f32; 2]],
    to: &[[f32; 2]],
    max_iterations: usize,
    confidence: f64,
    refine_iterations: usize,
) -> Result<[[f64; 3]; 2], FaceGeometryError> {
    let fallback = estimate_similarity(from, to)?;
    if !(0.0..1.0).contains(&confidence) || max_iterations == 0 {
        return Err(FaceGeometryError::Landmarks);
    }
    if from.len() == 2 {
        return Ok(fallback);
    }
    let iterations = ((1.0 - confidence).ln() / (1.0 - 0.55_f64.powi(2)).ln())
        .round_ties_even()
        .max(3.0) as usize;
    let mut rng = u64::MAX;
    let mut next = || {
        rng = u64::from(rng as u32)
            .wrapping_mul(4_164_903_690)
            .wrapping_add(rng >> 32);
        rng as u32 as usize
    };
    let mut best = None;
    let mut median = f32::INFINITY;
    for _ in 0..iterations.min(max_iterations) {
        let first = next() % from.len();
        let mut second = next() % from.len();
        for _ in 0..10000 {
            if first != second {
                break;
            }
            second = next() % from.len();
        }
        if first == second {
            continue;
        }
        let Ok(candidate) =
            estimate_similarity(&[from[first], from[second]], &[to[first], to[second]])
        else {
            continue;
        };
        let m = candidate.map(|row| row.map(|v| v as f32));
        let mut errors = from
            .iter()
            .zip(to)
            .map(|(p, q)| {
                let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] - q[0];
                let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] - q[1];
                x * x + y * y
            })
            .collect::<Vec<_>>();
        errors.sort_by(f32::total_cmp);
        let value = errors[errors.len() / 2];
        if value < median {
            median = value;
            best = Some(candidate);
        }
    }
    let matrix = best.ok_or(FaceGeometryError::Landmarks)?;
    if refine_iterations == 0 {
        return Ok(matrix);
    }
    let sigma = (2.5 * 1.4826 * (1.0 + 5.0 / (from.len() - 2) as f64) * f64::from(median).sqrt())
        .max(0.001);
    let m = matrix.map(|row| row.map(|v| v as f32));
    let mut source = Vec::new();
    let mut target = Vec::new();
    for (p, q) in from.iter().zip(to) {
        let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] - q[0];
        let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] - q[1];
        if x * x + y * y <= (sigma * sigma) as f32 {
            source.push(*p);
            target.push(*q);
        }
    }
    if source.len() >= 2 {
        estimate_similarity(&source, &target)
    } else {
        Ok(matrix)
    }
}

/// Warp interleaved RGB bytes with inverse affine mapping and constant borders.
///
/// Coordinates use a ten-bit affine accumulator and a five-bit interpolation
/// lattice; RGB values round once after integer bilinear interpolation.
///
/// # Errors
/// Rejects invalid storage, overflow, nonfinite or singular transforms.
pub fn warp_affine_rgb(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: [u8; 3],
) -> Result<Vec<u8>, FaceGeometryError> {
    warp_affine_u8(values, source, target, forward, &border)
}

/// Warp interleaved byte pixels, retaining the caller's channel count.
///
/// `border` supplies one constant value per channel (one to four channels).
/// Coordinates and interpolation use the same lattice as [`warp_affine_rgb`].
/// # Errors
/// Rejects invalid channels, dimensions, storage or affine transforms.
pub fn warp_affine_u8(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: &[u8],
) -> Result<Vec<u8>, FaceGeometryError> {
    let channels = border.len();
    let len = source
        .width
        .checked_mul(source.height)
        .and_then(|v| v.checked_mul(channels));
    let output_len = target
        .width
        .checked_mul(target.height)
        .and_then(|v| v.checked_mul(channels));
    if !(1..=4).contains(&channels)
        || source.width == 0
        || source.height == 0
        || target.width == 0
        || target.height == 0
        || len != Some(values.len())
        || output_len.is_none()
        || forward.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(FaceGeometryError::Geometry);
    }
    let [[a, b, c], [d, e, f]] = invert_affine(forward)?;
    if [a, b, c, d, e, f].iter().any(|v| !v.is_finite())
        || [source.width, source.height, target.width, target.height]
            .iter()
            .any(|&v| v > i32::MAX as usize)
        || (a.abs() + d.abs()) * target.width as f64 * 1024.0
            + (b.abs() + e.abs()) * target.height as f64 * 1024.0
            + (c.abs() + f.abs()) * 1024.0
            > (i64::MAX / 4) as f64
    {
        return Err(FaceGeometryError::Geometry);
    }
    let mut output = vec![0; output_len.ok_or(FaceGeometryError::Geometry)?];
    for y in 0..target.height {
        let x0 = ((b * y as f64 + c) * 1024.0).round_ties_even() as i64 + 16;
        let y0 = ((e * y as f64 + f) * 1024.0).round_ties_even() as i64 + 16;
        for x in 0..target.width {
            let xx = (x0 + (a * x as f64 * 1024.0).round_ties_even() as i64) >> 5;
            let yy = (y0 + (d * x as f64 * 1024.0).round_ties_even() as i64) >> 5;
            let sx = xx >> 5;
            let sy = yy >> 5;
            let fx = (xx & 31) as u32;
            let fy = (yy & 31) as u32;
            let weights = [
                (32 - fx) * (32 - fy),
                fx * (32 - fy),
                (32 - fx) * fy,
                fx * fy,
            ];
            for channel in 0..channels {
                let mut sum = 0_u32;
                for (tap, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let px = sx + dx;
                    let py = sy + dy;
                    let sample = if px >= 0
                        && py >= 0
                        && px < source.width as i64
                        && py < source.height as i64
                    {
                        values[(py as usize * source.width + px as usize) * channels + channel]
                    } else {
                        border[channel]
                    };
                    sum += u32::from(sample) * weights[tap];
                }
                output[(y * target.width + x) * channels + channel] = ((sum + 512) >> 10) as u8;
            }
        }
    }
    Ok(output)
}

trait AffineFloat: Copy {
    fn finite(self) -> bool;
    fn zero() -> Self;
    fn add_product(self, other: Self, weight: f64) -> Self;
}

/// Invert a finite two-dimensional affine transform in source scalar order.
/// # Errors
/// Rejects nonfinite or singular transforms.
pub fn invert_affine(forward: [[f64; 3]; 2]) -> Result<[[f64; 3]; 2], FaceGeometryError> {
    if forward.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FaceGeometryError::Geometry);
    }
    let determinant = forward[0][0] * forward[1][1] - forward[0][1] * forward[1][0];
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(FaceGeometryError::Singular);
    }
    let inverse = 1.0 / determinant;
    let a = forward[1][1] * inverse;
    let b = -forward[0][1] * inverse;
    let d = -forward[1][0] * inverse;
    let e = forward[0][0] * inverse;
    let result = [
        [a, b, -a * forward[0][2] - b * forward[1][2]],
        [d, e, -d * forward[0][2] - e * forward[1][2]],
    ];
    if result.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FaceGeometryError::Geometry);
    }
    Ok(result)
}
impl AffineFloat for f64 {
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn zero() -> Self {
        0.0
    }
    fn add_product(self, other: Self, weight: f64) -> Self {
        self + other * weight
    }
}
impl AffineFloat for f32 {
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn zero() -> Self {
        0.0
    }
    fn add_product(self, other: Self, weight: f64) -> Self {
        self + other * weight as f32
    }
}

/// Affine FP64 image/mask resampling on the source INTER_LINEAR coordinate lattice.
/// INTER_AREA in warpAffine selects this same bilinear interpolation path.
/// # Errors
/// Rejects invalid geometry, transforms or nonfinite values.
pub fn warp_affine_f64(
    values: &[f64],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: &[f64],
) -> Result<Vec<f64>, FaceGeometryError> {
    warp_affine_float(values, source, target, forward, border)
}

/// Affine uint16 resampling with FP32 interpolation and final tie-even saturation.
/// # Errors
/// Rejects invalid geometry or affine transforms.
pub fn warp_affine_u16(
    values: &[u16],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: &[u16],
) -> Result<Vec<u16>, FaceGeometryError> {
    let values: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let border: Vec<f32> = border.iter().map(|value| f32::from(*value)).collect();
    Ok(
        warp_affine_float(&values, source, target, forward, &border)?
            .into_iter()
            .map(|value| value.round_ties_even().clamp(0.0, f32::from(u16::MAX)) as u16)
            .collect(),
    )
}

fn warp_affine_float<T: AffineFloat>(
    values: &[T],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: &[T],
) -> Result<Vec<T>, FaceGeometryError> {
    let channels = border.len();
    let input_len = source
        .height
        .checked_mul(source.width)
        .and_then(|n| n.checked_mul(channels));
    let output_len = target
        .height
        .checked_mul(target.width)
        .and_then(|n| n.checked_mul(channels));
    if !(1..=4).contains(&channels)
        || input_len != Some(values.len())
        || output_len.is_none()
        || [source.width, source.height, target.width, target.height]
            .iter()
            .any(|value| *value == 0 || *value > i32::MAX as usize)
        || values.iter().chain(border).any(|value| !value.finite())
        || forward.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(FaceGeometryError::Geometry);
    }
    let [[a, b, c], [d, e, f]] = invert_affine(forward)?;
    if [a, b, c, d, e, f].iter().any(|value| !value.is_finite())
        || (a.abs() + d.abs()) * target.width as f64 * 1024.0
            + (b.abs() + e.abs()) * target.height as f64 * 1024.0
            + (c.abs() + f.abs()) * 1024.0
            > (i64::MAX / 4) as f64
    {
        return Err(FaceGeometryError::Geometry);
    }
    let mut output = vec![T::zero(); output_len.ok_or(FaceGeometryError::Geometry)?];
    for y in 0..target.height {
        let x0 = ((b * y as f64 + c) * 1024.0).round_ties_even() as i64 + 16;
        let y0 = ((e * y as f64 + f) * 1024.0).round_ties_even() as i64 + 16;
        for x in 0..target.width {
            let xx = (x0 + (a * x as f64 * 1024.0).round_ties_even() as i64) >> 5;
            let yy = (y0 + (d * x as f64 * 1024.0).round_ties_even() as i64) >> 5;
            let fx = (xx & 31) as f64 / 32.0;
            let fy = (yy & 31) as f64 / 32.0;
            let weights = [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ];
            for channel in 0..channels {
                let mut value = T::zero();
                for (tap, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let sx = (xx >> 5) + dx;
                    let sy = (yy >> 5) + dy;
                    let sample = if sx >= 0
                        && sy >= 0
                        && sx < source.width as i64
                        && sy < source.height as i64
                    {
                        values[(sy as usize * source.width + sx as usize) * channels + channel]
                    } else {
                        border[channel]
                    };
                    value = value.add_product(sample, weights[tap]);
                }
                output[(y * target.width + x) * channels + channel] = value;
            }
        }
    }
    Ok(output)
}
