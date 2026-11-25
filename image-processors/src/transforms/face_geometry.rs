//! Similarity estimation and affine/perspective image resampling in Rust.

use super::ImageSize;

/// Invalid landmark or transformed-image input.
#[derive(Debug, Clone, Copy)]
pub enum FaceGeometryError {
    /// Invalid, insufficient or degenerate corresponding landmarks.
    Landmarks,
    /// Invalid image storage or dimensions.
    Geometry,
    /// Noninvertible image transform.
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

/// Estimate a full affine map from corresponding FP32 points with RANSAC.
///
/// Uses the source three-point affine model, deterministic multiply-with-carry
/// sampling, FP32 reprojection errors and adaptive iteration count. The final
/// inlier fit retains FP64 coefficients. This is independent Rust geometry;
/// no native computer-vision library is linked.
/// # Errors
/// Rejects invalid controls/points, fewer than three correspondences or a
/// degenerate set without a noncollinear affine model.
pub fn estimate_affine_2d_ransac(
    from: &[[f32; 2]],
    to: &[[f32; 2]],
    threshold: f64,
    confidence: f64,
    max_iterations: usize,
    refine_iterations: usize,
) -> Result<([[f64; 3]; 2], Vec<bool>), FaceGeometryError> {
    if from.len() != to.len()
        || from.len() < 3
        || !threshold.is_finite()
        || threshold <= 0.0
        || !(0.0..1.0).contains(&confidence)
        || confidence == 0.0
        || max_iterations == 0
        || from
            .iter()
            .chain(to)
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(FaceGeometryError::Landmarks);
    }
    let mut state = u64::MAX;
    let mut best = None;
    let mut best_mask = vec![false; from.len()];
    let mut best_count = 0;
    let mut limit = max_iterations;
    let mut iteration = 0;
    while iteration < limit {
        let mut sample = [0; 3];
        let mut found = false;
        for _ in 0..10000 {
            for index in 0..3 {
                loop {
                    state = u64::from(state as u32).wrapping_mul(4164903690) + (state >> 32);
                    let value = state as u32 as usize % from.len();
                    if !sample[..index].contains(&value) {
                        sample[index] = value;
                        break;
                    }
                }
            }
            if from.len() == 3 {
                sample = [0, 1, 2];
            }
            let valid = |points: &[[f32; 2]]| {
                let a = points[sample[0]];
                let b = points[sample[1]];
                let c = points[sample[2]];
                let dx1 = b[0] - a[0];
                let dy1 = b[1] - a[1];
                let dx2 = c[0] - a[0];
                let dy2 = c[1] - a[1];
                (dx2 * dy1 - dy2 * dx1).abs()
                    > f32::EPSILON * (dx1.abs() + dy1.abs() + dx2.abs() + dy2.abs())
            };
            if valid(from) && valid(to) {
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
        let [x1, y1] = from[sample[0]].map(f64::from);
        let [x2, y2] = from[sample[1]].map(f64::from);
        let [x3, y3] = from[sample[2]].map(f64::from);
        let inverse = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        let matrix = std::array::from_fn(|axis| {
            let a = f64::from(to[sample[0]][axis]);
            let b = f64::from(to[sample[1]][axis]);
            let c = f64::from(to[sample[2]][axis]);
            [
                inverse * (a * (y2 - y3) + b * (y3 - y1) + c * (y1 - y2)),
                inverse * (a * (x3 - x2) + b * (x1 - x3) + c * (x2 - x1)),
                inverse
                    * (a * (x2 * y3 - x3 * y2) + b * (x3 * y1 - x1 * y3) + c * (x1 * y2 - x2 * y1)),
            ]
        });
        let floats = matrix.map(|row| row.map(|value| value as f32));
        let mask: Vec<bool> = from
            .iter()
            .zip(to)
            .map(|(p, q)| {
                let x = floats[0][0] * p[0] + floats[0][1] * p[1] + floats[0][2] - q[0];
                let y = floats[1][0] * p[0] + floats[1][1] * p[1] + floats[1][2] - q[1];
                x * x + y * y <= (threshold * threshold) as f32
            })
            .collect();
        let count = mask.iter().filter(|&&inside| inside).count();
        if count > best_count.max(2) {
            best = Some(matrix);
            best_mask = mask;
            best_count = count;
            let denominator = (1.0 - (count as f64 / from.len() as f64).powi(3)).ln();
            if denominator == f64::NEG_INFINITY {
                limit = 0;
            } else if denominator.is_finite() && denominator < 0.0 {
                limit = limit.min(
                    ((1.0 - confidence).ln() / denominator)
                        .round_ties_even()
                        .max(0.0) as usize,
                );
            }
        }
        iteration += 1;
        if from.len() == 3 {
            break;
        }
    }
    let mut matrix = best.ok_or(FaceGeometryError::Landmarks)?;
    // Retain the source Levenberg-Marquardt iteration and damping policy,
    // rather than replacing refinement with a single least-squares fit.
    if best_count > 3 && refine_iterations > 0 {
        let mut normal = [[0.0; 3]; 3];
        for (point, &keep) in from.iter().zip(&best_mask) {
            if !keep {
                continue;
            }
            let row = [f64::from(point[0]), f64::from(point[1]), 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
        let residual = |model: [[f64; 3]; 2]| {
            let mut error = 0.0;
            let mut maximum: f64 = 0.0;
            let mut gradient = [[0.0; 3]; 2];
            for ((point, target), &keep) in from.iter().zip(to).zip(&best_mask) {
                if !keep {
                    continue;
                }
                let row = [f64::from(point[0]), f64::from(point[1]), 1.0];
                for axis in 0..2 {
                    let value = model[axis][0] * row[0] + model[axis][1] * row[1] + model[axis][2]
                        - f64::from(target[axis]);
                    error += value * value;
                    maximum = maximum.max(value.abs());
                    for column in 0..3 {
                        gradient[axis][column] += row[column] * value;
                    }
                }
            }
            (error, maximum, gradient)
        };
        let mut lambda = 1.0;
        let mut lc = 0.75;
        for _ in 0..refine_iterations {
            let (error, _, gradient) = residual(matrix);
            let mut damped = normal;
            for index in 0..3 {
                damped[index][index] += lambda * normal[index][index];
            }
            let delta = [
                affine_solve3(damped, gradient[0])?,
                affine_solve3(damped, gradient[1])?,
            ];
            let candidate = std::array::from_fn(|axis| {
                std::array::from_fn(|column| matrix[axis][column] - delta[axis][column])
            });
            let (candidate_error, candidate_maximum, _) = residual(candidate);
            let mut reduction = 0.0;
            let mut dot = 0.0;
            for axis in 0..2 {
                for row in 0..3 {
                    let product = (0..3)
                        .map(|column| normal[row][column] * delta[axis][column])
                        .sum::<f64>();
                    reduction += delta[axis][row] * (2.0 * gradient[axis][row] - product);
                    dot += delta[axis][row] * gradient[axis][row];
                }
            }
            let ratio = (error - candidate_error)
                / if reduction.abs() > f64::EPSILON {
                    reduction
                } else {
                    1.0
                };
            if ratio > 0.75 {
                lambda *= 0.5;
                if lambda < lc {
                    lambda = 0.0;
                }
            } else if ratio < 0.25 {
                let mut nu = ((candidate_error - error)
                    / if dot.abs() > f64::EPSILON { dot } else { 1.0 }
                    + 2.0)
                    .clamp(2.0, 10.0);
                if lambda == 0.0 {
                    let mut maximum = f64::EPSILON;
                    for column in 0..3 {
                        let mut unit = [0.0; 3];
                        unit[column] = 1.0;
                        maximum = maximum.max(affine_solve3(normal, unit)?[column].abs());
                    }
                    lc = 1.0 / maximum;
                    lambda = lc;
                    nu *= 0.5;
                }
                lambda *= nu;
            }
            if candidate_error < error {
                matrix = candidate;
            }
            let delta_maximum = delta
                .iter()
                .flatten()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            let maximum = if candidate_error < error {
                candidate_maximum
            } else {
                residual(matrix).1
            };
            if delta_maximum < f64::from(f32::EPSILON) || maximum < f64::from(f32::EPSILON) {
                break;
            }
        }
    }
    Ok((matrix, best_mask))
}

fn affine_solve3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Result<[f64; 3], FaceGeometryError> {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&matrix[row]);
        augmented[row][3] = rhs[row];
    }
    for column in 0..3 {
        let mut pivot = column;
        for row in column + 1..3 {
            if augmented[row][column].abs() > augmented[pivot][column].abs() {
                pivot = row;
            }
        }
        augmented.swap(column, pivot);
        let divisor = augmented[column][column];
        if divisor == 0.0 {
            return Err(FaceGeometryError::Singular);
        }
        for value in &mut augmented[column][column..] {
            *value /= divisor;
        }
        let pivot_values = augmented[column];
        for (row, values) in augmented.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = values[column];
            for (value, &pivot_value) in values[column..].iter_mut().zip(&pivot_values[column..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    Ok(std::array::from_fn(|row| augmented[row][3]))
}

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

/// Warp interleaved FP64 pixels with scikit-image's order-one sampling.
///
/// `forward` maps source to destination pixel coordinates. Unlike the
/// INTER_LINEAR APIs, inverse coordinates and bilinear weights remain FP64,
/// without a coordinate lattice. The constant border is zero. Output clipping
/// follows `warp(..., mode="constant", cval=0, clip=True, preserve_range=True)`:
/// zero expands the input range only when it actually occurs in the output.
///
/// # Errors
/// Rejects empty/overflowing geometry, inconsistent interleaved storage,
/// nonfinite pixels/transforms or a singular affine transform.
pub fn warp_affine_skimage_f64(
    values: &[f64],
    source: ImageSize,
    target: ImageSize,
    channels: usize,
    forward: [[f64; 3]; 2],
) -> Result<Vec<f64>, FaceGeometryError> {
    warp_affine_skimage_inverse_f64(values, source, target, channels, invert_affine(forward)?)
}

/// Apply scikit-image order-one sampling using an explicit destination-to-source map.
///
/// This entrypoint retains an already computed `tform.inverse` matrix exactly;
/// it does not invert and reconstruct it. Clipping and borders are identical to
/// [`warp_affine_skimage_f64`].
/// # Errors
/// Rejects invalid geometry, pixels or a nonfinite/singular inverse map.
pub fn warp_affine_skimage_inverse_f64(
    values: &[f64],
    source: ImageSize,
    target: ImageSize,
    channels: usize,
    inverse: [[f64; 3]; 2],
) -> Result<Vec<f64>, FaceGeometryError> {
    let input_len = source
        .width
        .checked_mul(source.height)
        .and_then(|n| n.checked_mul(channels));
    let output_len = target
        .width
        .checked_mul(target.height)
        .and_then(|n| n.checked_mul(channels));
    if channels == 0
        || [source.width, source.height, target.width, target.height].contains(&0)
        || input_len != Some(values.len())
        || output_len.is_none()
        || values.iter().any(|value| !value.is_finite())
        || inverse.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(FaceGeometryError::Geometry);
    }
    invert_affine(inverse)?;
    let [[a, b, c], [d, e, f]] = inverse;
    let mut output = vec![0.0; output_len.ok_or(FaceGeometryError::Geometry)?];
    for y in 0..target.height {
        for x in 0..target.width {
            let sx = a * x as f64 + b * y as f64 + c;
            let sy = d * x as f64 + e * y as f64 + f;
            if !sx.is_finite() || !sy.is_finite() {
                return Err(FaceGeometryError::Geometry);
            }
            // Outside the one-pixel interpolation halo every tap is cval.
            if sx <= -1.0 || sy <= -1.0 || sx >= source.width as f64 || sy >= source.height as f64 {
                continue;
            }
            let left = sx.floor();
            let top = sy.floor();
            let column_fraction = sx - left;
            let row_fraction = sy - top;
            for channel in 0..channels {
                let mut taps = [0.0; 4];
                for (index, (dx, dy)) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
                    .into_iter()
                    .enumerate()
                {
                    let column = left + dx;
                    let row = top + dy;
                    if column >= 0.0
                        && row >= 0.0
                        && column < source.width as f64
                        && row < source.height as f64
                    {
                        taps[index] = values
                            [(row as usize * source.width + column as usize) * channels + channel];
                    }
                }
                let upper = (1.0 - column_fraction) * taps[0] + column_fraction * taps[1];
                let lower = (1.0 - column_fraction) * taps[2] + column_fraction * taps[3];
                output[(y * target.width + x) * channels + channel] =
                    (1.0 - row_fraction) * upper + row_fraction * lower;
            }
        }
    }
    let mut minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let output_minimum = output.iter().copied().fold(f64::INFINITY, f64::min);
    let output_maximum = output.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(minimum..=maximum).contains(&0.0) && output_minimum <= 0.0 && output_maximum >= 0.0 {
        minimum = minimum.min(0.0);
        maximum = maximum.max(0.0);
    }
    for value in &mut output {
        *value = value.clamp(minimum, maximum);
    }
    Ok(output)
}

/// Apply the scikit-image affine warp to byte pixels, then truncate to bytes.
///
/// This preserves the order of `warp(..., preserve_range=True).astype(uint8)`;
/// it does not round the FP64 result to the nearest integer.
/// # Errors
/// Reports invalid geometry, storage or transforms as in [`warp_affine_skimage_f64`].
pub fn warp_affine_skimage_u8(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    channels: usize,
    forward: [[f64; 3]; 2],
) -> Result<Vec<u8>, FaceGeometryError> {
    let pixels: Vec<f64> = values.iter().map(|&value| f64::from(value)).collect();
    Ok(
        warp_affine_skimage_f64(&pixels, source, target, channels, forward)?
            .into_iter()
            .map(|value| value as u8)
            .collect(),
    )
}

/// Byte counterpart of [`warp_affine_skimage_inverse_f64`], retaining truncation.
/// # Errors
/// Reports invalid image storage, geometry or inverse transforms.
pub fn warp_affine_skimage_inverse_u8(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    channels: usize,
    inverse: [[f64; 3]; 2],
) -> Result<Vec<u8>, FaceGeometryError> {
    let pixels: Vec<f64> = values.iter().map(|&value| f64::from(value)).collect();
    Ok(
        warp_affine_skimage_inverse_f64(&pixels, source, target, channels, inverse)?
            .into_iter()
            .map(|value| value as u8)
            .collect(),
    )
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
            let start = (y * target.width + x) * channels;
            bilinear_u8(
                values,
                source,
                border,
                [xx, yy],
                &mut output[start..start + channels],
            );
        }
    }
    Ok(output)
}

/// Sampling used by a perspective image warp.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PerspectiveInterpolation {
    /// Round source coordinates to the nearest pixel, with ties to even.
    Nearest,
    /// Five-bit bilinear interpolation with one final byte rounding.
    #[default]
    Bilinear,
    /// Sixteen-neighbor -0.75 cubic interpolation with normalized Q15 weights.
    Bicubic,
}

/// Pixel values used outside a perspective warp's source image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerspectiveBorder {
    /// One constant value for each RGB channel.
    Constant([u8; 3]),
    /// Extend the closest source edge pixel.
    Replicate,
}

/// Explicit sampling and coordinate conventions for a perspective warp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerspectiveWarpOptions {
    /// Source-pixel interpolation.
    pub interpolation: PerspectiveInterpolation,
    /// Sampling beyond the source image.
    pub border: PerspectiveBorder,
    /// Interpret corners as canvas edges and sample destination pixel centers.
    /// This applies `T(-0.5) * forward * T(0.5)` before matrix inversion.
    pub pixel_centers: bool,
}

impl Default for PerspectiveWarpOptions {
    fn default() -> Self {
        Self {
            interpolation: PerspectiveInterpolation::Bilinear,
            border: PerspectiveBorder::Constant([0; 3]),
            pixel_centers: false,
        }
    }
}

/// Sample byte pixels at an arbitrary inverse coordinate map.
///
/// Coordinates are FP32 source pixel `(x, y)` pairs in output row-major order.
/// They round directly to the five-bit interpolation lattice. Interpolation
/// and constant borders share the existing affine warp sampler.
///
/// # Errors
/// Rejects invalid storage, nonfinite coordinates, dimensions outside the
/// signed-short remapping range, or other than one to four channels.
pub fn remap_bilinear_u8(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    coordinates: &[[f32; 2]],
    border: &[u8],
) -> Result<Vec<u8>, FaceGeometryError> {
    let channels = border.len();
    let input_len = source
        .width
        .checked_mul(source.height)
        .and_then(|pixels| pixels.checked_mul(channels));
    let pixels = target
        .width
        .checked_mul(target.height)
        .ok_or(FaceGeometryError::Geometry)?;
    let output_len = pixels
        .checked_mul(channels)
        .ok_or(FaceGeometryError::Geometry)?;
    if !(1..=4).contains(&channels)
        || [source.width, source.height, target.width, target.height]
            .iter()
            .any(|&size| size == 0 || size >= i16::MAX as usize)
        || input_len != Some(values.len())
        || coordinates.len() != pixels
        || coordinates.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(FaceGeometryError::Geometry);
    }
    let mut output = vec![0; output_len];
    for (coordinate, pixel) in coordinates.iter().zip(output.chunks_exact_mut(channels)) {
        let fixed = coordinate.map(|value| {
            let rounded = (value * 32.0).round_ties_even() as i32;
            let base = (rounded >> 5).clamp(i16::MIN as i32, i16::MAX as i32);
            i64::from(base) * 32 + i64::from(rounded & 31)
        });
        bilinear_u8(values, source, border, fixed, pixel);
    }
    Ok(output)
}

/// Bilinearly remap interleaved FP32 pixels with replicated image borders.
///
/// Coordinates round to the same five-bit lattice as [`remap_bilinear_u8`].
/// Pixel values remain floating point, including NaN and infinity.
/// # Errors
/// Rejects invalid dimensions, storage, channel counts or nonfinite coordinates.
pub fn remap_bilinear_f32(
    values: &[f32],
    source: ImageSize,
    target: ImageSize,
    coordinates: &[[f32; 2]],
    channels: usize,
) -> Result<Vec<f32>, FaceGeometryError> {
    let pixels = target
        .width
        .checked_mul(target.height)
        .ok_or(FaceGeometryError::Geometry)?;
    let length = pixels
        .checked_mul(channels)
        .ok_or(FaceGeometryError::Geometry)?;
    if channels == 0
        || source.width == 0
        || source.height == 0
        || target.width == 0
        || target.height == 0
        || source
            .width
            .checked_mul(source.height)
            .and_then(|n| n.checked_mul(channels))
            != Some(values.len())
        || coordinates.len() != pixels
        || coordinates.iter().flatten().any(|v| !v.is_finite())
        || [source.width, source.height, target.width, target.height]
            .iter()
            .any(|&n| n >= i16::MAX as usize)
    {
        return Err(FaceGeometryError::Geometry);
    }
    let mut output = vec![0.0; length];
    for (coordinate, pixel) in coordinates.iter().zip(output.chunks_exact_mut(channels)) {
        let fixed = coordinate.map(|v| (v * 32.0).round_ties_even() as i32);
        let x = (fixed[0] >> 5).clamp(i16::MIN as i32, i16::MAX as i32);
        let y = (fixed[1] >> 5).clamp(i16::MIN as i32, i16::MAX as i32);
        let fx = (fixed[0] & 31) as f32 / 32.0;
        let fy = (fixed[1] & 31) as f32 / 32.0;
        for (channel, value) in pixel.iter_mut().enumerate() {
            for (dx, dy, weight) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let sx = (x + dx).clamp(0, source.width as i32 - 1) as usize;
                let sy = (y + dy).clamp(0, source.height as i32 - 1) as usize;
                *value += values[(sy * source.width + sx) * channels + channel] * weight;
            }
        }
    }
    Ok(output)
}

/// Nearest-neighbor byte remapping with replicated borders.
///
/// Floating source coordinates round to even before clamping to the image.
/// # Errors
/// Rejects invalid geometry, storage or nonfinite coordinates.
pub fn remap_nearest_u8(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    coordinates: &[[f32; 2]],
    channels: usize,
) -> Result<Vec<u8>, FaceGeometryError> {
    let pixels = target
        .width
        .checked_mul(target.height)
        .ok_or(FaceGeometryError::Geometry)?;
    let length = pixels
        .checked_mul(channels)
        .ok_or(FaceGeometryError::Geometry)?;
    if channels == 0
        || source.width == 0
        || source.height == 0
        || target.width == 0
        || target.height == 0
        || source
            .width
            .checked_mul(source.height)
            .and_then(|n| n.checked_mul(channels))
            != Some(values.len())
        || coordinates.len() != pixels
        || coordinates.iter().flatten().any(|v| !v.is_finite())
        || [source.width, source.height, target.width, target.height]
            .iter()
            .any(|&n| n >= i16::MAX as usize)
    {
        return Err(FaceGeometryError::Geometry);
    }
    let mut output = vec![0; length];
    for (coordinate, pixel) in coordinates.iter().zip(output.chunks_exact_mut(channels)) {
        let x = (coordinate[0].round_ties_even() as i32).clamp(0, source.width as i32 - 1) as usize;
        let y =
            (coordinate[1].round_ties_even() as i32).clamp(0, source.height as i32 - 1) as usize;
        let start = (y * source.width + x) * channels;
        pixel.copy_from_slice(&values[start..start + channels]);
    }
    Ok(output)
}

/// Warp a quadrilateral in an interleaved RGB image to a rectangular canvas.
///
/// `corners` are FP32 source pixel coordinates corresponding to destination
/// `(0, height)`, `(0, 0)`, `(width, 0)`, `(width, height)`, respectively.
/// Coordinates describe canvas edges, without a half-pixel offset. The FP64
/// homography is solved with partial-pivot elimination, then inverted for
/// sampling. Perspective division rounds directly to a five-bit interpolation
/// lattice; unlike affine sampling it has no ten-bit intermediate accumulator.
/// Byte bilinear interpolation rounds once, with a constant RGB border.
///
/// This is the sampling contract used by MediaPipe's CPU ImageToTensor path.
/// It is implemented entirely in Rust and does not call OpenCV.
///
/// # Errors
/// Rejects invalid RGB storage, nonfinite corners, zero/overflowing dimensions,
/// dimensions outside the signed-short remapping range, or singular transforms.
pub fn warp_perspective_quad_rgb(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    corners: [[f32; 2]; 4],
    border: [u8; 3],
) -> Result<Vec<u8>, FaceGeometryError> {
    warp_perspective_quad_rgb_with_options(
        values,
        source,
        target,
        corners,
        PerspectiveWarpOptions {
            border: PerspectiveBorder::Constant(border),
            ..Default::default()
        },
    )
}

/// Warp RGB bytes using explicit interpolation, border and pixel-center options.
///
/// Corner order and the homography solve match [`warp_perspective_quad_rgb`].
/// Nearest sampling rounds directly to integer source coordinates. Bilinear
/// sampling retains the existing five-bit interpolation lattice. Replicated
/// borders clamp remapping coordinates, without allocating a padded image.
/// `pixel_centers` implements the half-pixel correction used by MediaPipe's CPU
/// ImageCroppingCalculator; its nearest/replicate path is entirely Rust.
///
/// # Errors
/// Rejects invalid storage, dimensions, nonfinite corners or singular transforms.
pub fn warp_perspective_quad_rgb_with_options(
    values: &[u8],
    source: ImageSize,
    target: ImageSize,
    corners: [[f32; 2]; 4],
    options: PerspectiveWarpOptions,
) -> Result<Vec<u8>, FaceGeometryError> {
    let source_len = source
        .width
        .checked_mul(source.height)
        .and_then(|len| len.checked_mul(3));
    let output_len = target
        .width
        .checked_mul(target.height)
        .and_then(|len| len.checked_mul(3));
    if [source.width, source.height, target.width, target.height]
        .into_iter()
        .any(|size| size == 0 || size >= i16::MAX as usize)
        || source_len != Some(values.len())
        || output_len.is_none()
        || corners.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(FaceGeometryError::Geometry);
    }
    let inverse = quad_inverse(corners, target, options.pixel_centers)?;
    let border = match options.border {
        PerspectiveBorder::Constant(rgb) => rgb,
        PerspectiveBorder::Replicate => [0; 3],
    };
    let lattice = match options.interpolation {
        PerspectiveInterpolation::Nearest => 1.0,
        PerspectiveInterpolation::Bilinear | PerspectiveInterpolation::Bicubic => 32.0,
    };
    let mut output = vec![0; output_len.ok_or(FaceGeometryError::Geometry)?];
    for y in 0..target.height {
        let row_x = inverse[1] * y as f64 + inverse[2];
        let row_y = inverse[4] * y as f64 + inverse[5];
        let row_w = inverse[7] * y as f64 + inverse[8];
        for x in 0..target.width {
            let denominator = row_w + inverse[6] * x as f64;
            let scale = if denominator == 0.0 {
                0.0
            } else {
                lattice / denominator
            };
            let mut fixed_x = ((row_x + inverse[0] * x as f64) * scale)
                .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
                .round_ties_even() as i64;
            let mut fixed_y = ((row_y + inverse[3] * x as f64) * scale)
                .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
                .round_ties_even() as i64;
            let start = (y * target.width + x) * 3;
            if options.interpolation == PerspectiveInterpolation::Nearest {
                fixed_x = fixed_x.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
                fixed_y = fixed_y.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
                if options.border == PerspectiveBorder::Replicate {
                    fixed_x = fixed_x.clamp(0, source.width as i64 - 1);
                    fixed_y = fixed_y.clamp(0, source.height as i64 - 1);
                }
                let pixel = if fixed_x >= 0
                    && fixed_y >= 0
                    && fixed_x < source.width as i64
                    && fixed_y < source.height as i64
                {
                    let offset = (fixed_y as usize * source.width + fixed_x as usize) * 3;
                    &values[offset..offset + 3]
                } else {
                    &border
                };
                output[start..start + 3].copy_from_slice(pixel);
            } else if options.interpolation == PerspectiveInterpolation::Bicubic {
                bicubic_perspective_u8(
                    values,
                    source,
                    options.border,
                    [fixed_x, fixed_y],
                    &mut output[start..start + 3],
                );
            } else {
                if options.border == PerspectiveBorder::Replicate {
                    fixed_x = fixed_x.clamp(0, (source.width as i64 - 1) * 32);
                    fixed_y = fixed_y.clamp(0, (source.height as i64 - 1) * 32);
                }
                bilinear_u8(
                    values,
                    source,
                    &border,
                    [fixed_x, fixed_y],
                    &mut output[start..start + 3],
                );
            }
        }
    }
    Ok(output)
}

fn bicubic_perspective_u8(
    values: &[u8],
    source: ImageSize,
    border: PerspectiveBorder,
    fixed: [i64; 2],
    output: &mut [u8],
) {
    // Reuse the resize owner's FP32 coefficient polynomial. Warp sampling has
    // a distinct Q5-coordinate/Q15-two-dimensional-table rounding contract.
    static TABLE: std::sync::LazyLock<[[i16; 16]; 1024]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|index| {
            let x = image_resize_kernels::opencv_cubic_coefficients((index % 32) as f32 / 32.0);
            let y = image_resize_kernels::opencv_cubic_coefficients((index / 32) as f32 / 32.0);
            let mut weights: [i16; 16] = std::array::from_fn(|tap| {
                (y[tap / 4] * x[tap % 4] * 32768.0)
                    .round_ties_even()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16
            });
            let difference = weights.iter().map(|&weight| i32::from(weight)).sum::<i32>() - 32768;
            if difference != 0 {
                let (mut minimum, mut maximum) = (10, 10);
                for row in 2..4 {
                    for column in 2..4 {
                        let tap = row * 4 + column;
                        if weights[tap] < weights[minimum] {
                            minimum = tap;
                        } else if weights[tap] > weights[maximum] {
                            maximum = tap;
                        }
                    }
                }
                let tap = if difference < 0 { maximum } else { minimum };
                weights[tap] = (i32::from(weights[tap]) - difference) as i16;
            }
            weights
        })
    });
    let x = (fixed[0] >> 5).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) - 1;
    let y = (fixed[1] >> 5).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) - 1;
    let coefficients = &TABLE[((fixed[1] & 31) * 32 + (fixed[0] & 31)) as usize];
    for (channel, value) in output.iter_mut().enumerate() {
        let mut sum = 0_i32;
        for (tap, &weight) in coefficients.iter().enumerate() {
            let mut px = x + (tap % 4) as i64;
            let mut py = y + (tap / 4) as i64;
            if border == PerspectiveBorder::Replicate {
                px = px.clamp(0, source.width as i64 - 1);
                py = py.clamp(0, source.height as i64 - 1);
            }
            let pixel =
                if px >= 0 && py >= 0 && px < source.width as i64 && py < source.height as i64 {
                    values[(py as usize * source.width + px as usize) * 3 + channel]
                } else if let PerspectiveBorder::Constant(color) = border {
                    color[channel]
                } else {
                    0
                };
            sum += i32::from(pixel) * i32::from(weight);
        }
        *value = ((sum + (1 << 14)) >> 15).clamp(0, 255) as u8;
    }
}

fn quad_inverse(
    corners: [[f32; 2]; 4],
    target: ImageSize,
    pixel_centers: bool,
) -> Result<[f64; 9], FaceGeometryError> {
    let destinations = [
        [0.0, target.height as f32],
        [0.0, 0.0],
        [target.width as f32, 0.0],
        [target.width as f32, target.height as f32],
    ];
    let mut equations = [[0.0_f64; 9]; 8];
    for (index, ([x, y], [u, v])) in corners.into_iter().zip(destinations).enumerate() {
        // Source point products are evaluated in FP32 before entering the FP64 solve.
        equations[index] = [
            f64::from(x),
            f64::from(y),
            1.0,
            0.0,
            0.0,
            0.0,
            f64::from(-x * u),
            f64::from(-y * u),
            f64::from(u),
        ];
        equations[index + 4] = [
            0.0,
            0.0,
            0.0,
            f64::from(x),
            f64::from(y),
            1.0,
            f64::from(-x * v),
            f64::from(-y * v),
            f64::from(v),
        ];
    }
    if equations.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FaceGeometryError::Geometry);
    }
    for column in 0..8 {
        let mut pivot = column;
        for (row, equation) in equations.iter().enumerate().skip(column + 1) {
            if equation[column].abs() > equations[pivot][column].abs() {
                pivot = row;
            }
        }
        if equations[pivot][column].abs() < f64::EPSILON * 100.0 {
            return Err(FaceGeometryError::Singular);
        }
        equations.swap(column, pivot);
        let pivot_row = equations[column];
        let reciprocal = -1.0 / pivot_row[column];
        for equation in equations.iter_mut().skip(column + 1) {
            let multiplier = equation[column] * reciprocal;
            for (coefficient, &pivot_value) in equation.iter_mut().zip(&pivot_row).skip(column + 1)
            {
                *coefficient += multiplier * pivot_value;
            }
        }
    }
    let mut forward = [0.0; 9];
    forward[8] = 1.0;
    for row in (0..8).rev() {
        let equation = equations[row];
        let mut value = equation[8];
        for (&coefficient, &known) in equation[..8].iter().zip(&forward[..8]).skip(row + 1) {
            value -= coefficient * known;
        }
        forward[row] = value / equation[row];
    }
    if pixel_centers {
        // Match ImageCroppingCalculator's FP64 multiply order: shift the
        // destination first, then the source, and invert the corrected matrix.
        for column in 0..3 {
            forward[column] -= 0.5 * forward[6 + column];
            forward[3 + column] -= 0.5 * forward[6 + column];
        }
        forward[2] += 0.5 * (forward[0] + forward[1]);
        forward[5] += 0.5 * (forward[3] + forward[4]);
        forward[8] += 0.5 * (forward[6] + forward[7]);
    }
    let [a, b, c, d, e, f, g, h, i] = forward;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(FaceGeometryError::Singular);
    }
    let reciprocal = 1.0 / determinant;
    let inverse = [
        e * i - f * h,
        c * h - b * i,
        b * f - c * e,
        f * g - d * i,
        a * i - c * g,
        c * d - a * f,
        d * h - e * g,
        b * g - a * h,
        a * e - b * d,
    ]
    .map(|value| value * reciprocal);
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(FaceGeometryError::Singular);
    }
    Ok(inverse)
}

fn bilinear_u8(
    values: &[u8],
    source: ImageSize,
    border: &[u8],
    fixed: [i64; 2],
    output: &mut [u8],
) {
    let channels = border.len();
    let [xx, yy] = fixed;
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
    for (channel, pixel) in output.iter_mut().enumerate() {
        let mut sum = 0_u32;
        for (tap, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let px = sx + dx;
            let py = sy + dy;
            let sample =
                if px >= 0 && py >= 0 && px < source.width as i64 && py < source.height as i64 {
                    values[(py as usize * source.width + px as usize) * channels + channel]
                } else {
                    border[channel]
                };
            sum += u32::from(sample) * weights[tap];
        }
        *pixel = ((sum + 512) >> 10) as u8;
    }
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

/// Affine FP32 image/mask resampling on the source INTER_LINEAR coordinate lattice.
///
/// Pixels are interleaved, with one constant border value per channel (one to
/// four channels). The forward transform maps source to destination pixel
/// coordinates. Sampling retains FP32 products and accumulation without byte
/// quantization, using the same coordinate lattice as [`warp_affine_f64`].
///
/// # Errors
/// Rejects invalid geometry, storage, transforms or nonfinite values.
pub fn warp_affine_f32(
    values: &[f32],
    source: ImageSize,
    target: ImageSize,
    forward: [[f64; 3]; 2],
    border: &[f32],
) -> Result<Vec<f32>, FaceGeometryError> {
    warp_affine_float(values, source, target, forward, border)
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
