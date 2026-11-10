//! Shared host spatial filters; no native image-processing library is linked.

use super::ImageSize;

/// Invalid spatial-filter input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialFilterError {
    /// Dimensions, channels or storage length are inconsistent or overflow.
    Geometry,
    /// Sigma must be finite and positive.
    Sigma,
    /// Kernel dimensions must be positive and odd.
    Kernel,
    /// Floating-point input contains a nonfinite value.
    NonFinite,
}
impl std::fmt::Display for SpatialFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Geometry => "invalid spatial-filter geometry or storage",
            Self::Sigma => "Gaussian sigma must be finite and positive",
            Self::Kernel => "invalid spatial-filter kernel",
            Self::NonFinite => "spatial-filter input must be finite",
        })
    }
}
impl std::error::Error for SpatialFilterError {}

fn geometry(len: usize, size: ImageSize, channels: usize) -> Result<(), SpatialFilterError> {
    if size.width == 0
        || size.height == 0
        || channels == 0
        || size
            .width
            .checked_mul(size.height)
            .and_then(|n| n.checked_mul(channels))
            != Some(len)
    {
        return Err(SpatialFilterError::Geometry);
    }
    Ok(())
}

/// Square minimum filtering of interleaved byte channels with replicated borders.
///
/// This preserves Pillow's `MinFilter` byte values, including at image edges.
/// Each channel is filtered independently; pass one channel to erode an alpha
/// matte without changing its RGB image. A kernel size of one is the identity.
///
/// # Errors
/// Returns an error for inconsistent storage, empty or overflowing geometry,
/// or a kernel size that is zero or even.
pub fn minimum_filter_u8(
    values: &[u8],
    size: ImageSize,
    channels: usize,
    kernel_size: usize,
) -> Result<Vec<u8>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(SpatialFilterError::Kernel);
    }
    let radius = kernel_size / 2;
    let mut horizontal = vec![0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            // Repeating an edge value cannot change a minimum, so clipping
            // the window is exactly equivalent to replicating the border.
            let start = x.saturating_sub(radius);
            let end = x.saturating_add(radius).min(size.width - 1);
            for channel in 0..channels {
                let mut minimum = u8::MAX;
                for source_x in start..=end {
                    minimum = minimum.min(values[(y * size.width + source_x) * channels + channel]);
                }
                horizontal[(y * size.width + x) * channels + channel] = minimum;
            }
        }
    }
    let mut output = vec![0; values.len()];
    for y in 0..size.height {
        let start = y.saturating_sub(radius);
        let end = y.saturating_add(radius).min(size.height - 1);
        for x in 0..size.width {
            for channel in 0..channels {
                let mut minimum = u8::MAX;
                for source_y in start..=end {
                    minimum =
                        minimum.min(horizontal[(source_y * size.width + x) * channels + channel]);
                }
                output[(y * size.width + x) * channels + channel] = minimum;
            }
        }
    }
    Ok(output)
}

fn reflect101(mut index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let last = (length - 1) as isize;
    while index < 0 || index > last {
        index = if index < 0 { -index } else { 2 * last - index };
    }
    index as usize
}

/// SciPy-style Gaussian filtering with half-sample-symmetric (`reflect`) borders.
///
/// Samples are interleaved; channels are filtered independently. `sigma` is in
/// height/width order. Each axis uses radius `floor(truncate * sigma + 0.5)`;
/// coefficients, accumulation and the intermediate image remain FP64. As in
/// `scipy.ndimage.gaussian_filter`, the vertical pass precedes the horizontal
/// pass and an axis with sigma at most 1e-15 is left unchanged. Reflection
/// repeats the edge sample, including for singleton axes and large kernels.
/// Existing reflect-101 filters retain their behavior.
///
/// # Errors
/// Rejects inconsistent geometry, nonfinite samples, negative/nonfinite sigma,
/// negative/nonfinite truncation, or support that cannot be represented.
pub fn gaussian_filter_reflect_f64(
    values: &[f64],
    size: ImageSize,
    channels: usize,
    sigma: [f64; 2],
    truncate: f64,
) -> Result<Vec<f64>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    if sigma.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(SpatialFilterError::Sigma);
    }
    if !truncate.is_finite() || truncate < 0.0 {
        return Err(SpatialFilterError::Kernel);
    }
    let mut output = values.to_vec();
    for (axis, sigma) in sigma.into_iter().enumerate() {
        if sigma <= 1e-15 {
            continue;
        }
        let radius = (truncate * sigma + 0.5).floor();
        if !radius.is_finite() || radius > f64::from(i32::MAX / 2) {
            return Err(SpatialFilterError::Kernel);
        }
        let radius = radius as usize;
        let weights = coefficients(sigma, radius * 2 + 1)?;
        let input = output;
        output = vec![0.0; input.len()];
        let extent = if axis == 0 { size.height } else { size.width };
        let step = if axis == 0 {
            size.width * channels
        } else {
            channels
        };
        // i128 arithmetic prevents overflow when reflecting long supports.
        let period = 2 * extent as i128;
        for y in 0..size.height {
            for x in 0..size.width {
                let position = if axis == 0 { y } else { x };
                let origin = (y * size.width + x) * channels;
                for channel in 0..channels {
                    let mut sum = input[origin + channel] * weights[radius];
                    // ndimage's symmetric kernel evaluates the outer pairs first.
                    for distance in (1..=radius).rev() {
                        let mut samples = [0.0; 2];
                        for (sample, offset) in samples
                            .iter_mut()
                            .zip([-(distance as i128), distance as i128])
                        {
                            let folded = (position as i128 + offset).rem_euclid(period);
                            let reflected = if folded < extent as i128 {
                                folded
                            } else {
                                period - 1 - folded
                            } as usize;
                            let index = origin - position * step + reflected * step + channel;
                            *sample = input[index];
                        }
                        sum += (samples[0] + samples[1]) * weights[radius - distance];
                    }
                    output[origin + channel] = sum;
                }
            }
        }
    }
    Ok(output)
}

fn coefficients(sigma: f64, side: usize) -> Result<Vec<f64>, SpatialFilterError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SpatialFilterError::Sigma);
    }
    if side == 0 || side.is_multiple_of(2) || side > i32::MAX as usize {
        return Err(SpatialFilterError::Kernel);
    }
    let radius = side / 2;
    let mut weights = (0..side)
        .map(|i| {
            let distance = i as f64 - radius as f64;
            (-0.5 * distance * distance / (sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let sum: f64 = weights.iter().sum();
    for value in &mut weights {
        *value /= sum;
    }
    Ok(weights)
}

fn kernel(
    sigma: f64,
    requested: Option<ImageSize>,
    byte: bool,
) -> Result<ImageSize, SpatialFilterError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SpatialFilterError::Sigma);
    }
    let automatic = (sigma * if byte { 6.0 } else { 8.0 } + 1.0).round_ties_even() as usize | 1;
    let size = requested.unwrap_or(ImageSize {
        height: automatic,
        width: automatic,
    });
    if size.width == 0
        || size.height == 0
        || size.width.is_multiple_of(2)
        || size.height.is_multiple_of(2)
    {
        return Err(SpatialFilterError::Kernel);
    }
    Ok(size)
}

/// Separable Gaussian filtering with FP32 coefficients and reflect-101 borders.
///
/// Each axis has an explicit odd support and sigma. Coefficients, their
/// normalization and both convolution passes remain FP32, matching tensor
/// Gaussian preprocessing. The horizontal pass precedes the vertical pass.
/// Existing OpenCV-compatible filters retain their coefficient precision.
///
/// # Errors
/// Rejects invalid geometry, nonfinite samples, invalid sigma/support, or a
/// reflection radius at least as large as its input dimension.
pub fn gaussian_blur_tensor_f32(
    values: &[f32],
    size: ImageSize,
    channels: usize,
    sigma: [f32; 2],
    support: ImageSize,
) -> Result<Vec<f32>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let mut kernels = Vec::with_capacity(2);
    for (side, extent, sigma) in [
        (support.height, size.height, sigma[0]),
        (support.width, size.width, sigma[1]),
    ] {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(SpatialFilterError::Sigma);
        }
        if side == 0 || side.is_multiple_of(2) || side / 2 >= extent {
            return Err(SpatialFilterError::Kernel);
        }
        let mut weights = (0..side)
            .map(|index| {
                let distance = index as f32 - (side / 2) as f32;
                (-(distance * distance) / (2.0 * (sigma * sigma))).exp()
            })
            .collect::<Vec<_>>();
        let sum: f32 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
        }
        kernels.push(weights);
    }
    let mut intermediate = vec![0.0; values.len()];
    let mut output = vec![0.0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..channels {
                let mut sum = 0.0_f32;
                for (tap, weight) in kernels[1].iter().enumerate() {
                    let column = reflect101(
                        x as isize + tap as isize - (support.width / 2) as isize,
                        size.width,
                    );
                    sum += values[(y * size.width + column) * channels + channel] * weight;
                }
                intermediate[(y * size.width + x) * channels + channel] = sum;
            }
        }
    }
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..channels {
                let mut sum = 0.0_f32;
                for (tap, weight) in kernels[0].iter().enumerate() {
                    let row = reflect101(
                        y as isize + tap as isize - (support.height / 2) as isize,
                        size.height,
                    );
                    sum += intermediate[(row * size.width + x) * channels + channel] * weight;
                }
                output[(y * size.width + x) * channels + channel] = sum;
            }
        }
    }
    Ok(output)
}

/// Gaussian filtering of interleaved FP32 values with reflect-101 borders.
///
/// Automatic support is `round(8*sigma+1)|1`; explicit support preserves the
/// caller's kernel dimensions. Coefficients and intermediates remain FP32.
///
/// # Errors
/// Rejects invalid geometry, nonfinite values, sigma or kernel dimensions.
pub fn gaussian_blur_f32(
    values: &[f32],
    size: ImageSize,
    channels: usize,
    sigma: f64,
    requested: Option<ImageSize>,
) -> Result<Vec<f32>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let support = kernel(sigma, requested, false)?;
    let horizontal = coefficients(sigma, support.width)?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let vertical = coefficients(sigma, support.height)?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let mut intermediate = vec![0.0_f32; values.len()];
    let mut output = vec![0.0_f32; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            for c in 0..channels {
                let mut sum = 0.0;
                for (tap, &weight) in horizontal.iter().enumerate() {
                    let sx = reflect101(
                        x as isize + tap as isize - (support.width / 2) as isize,
                        size.width,
                    );
                    sum += values[(y * size.width + sx) * channels + c] * weight;
                }
                intermediate[(y * size.width + x) * channels + c] = sum;
            }
        }
    }
    for y in 0..size.height {
        for x in 0..size.width {
            for c in 0..channels {
                let mut sum = intermediate[(y * size.width + x) * channels + c]
                    * vertical[support.height / 2];
                for tap in 1..=support.height / 2 {
                    let above = reflect101(y as isize - tap as isize, size.height);
                    let below = reflect101(y as isize + tap as isize, size.height);
                    sum += (intermediate[(above * size.width + x) * channels + c]
                        + intermediate[(below * size.width + x) * channels + c])
                        * vertical[support.height / 2 + tap];
                }
                output[(y * size.width + x) * channels + c] = sum;
            }
        }
    }
    Ok(output)
}

/// Gaussian filtering with FP64 coefficients, intermediates and reflect-101 borders.
/// # Errors
/// Rejects invalid geometry, nonfinite samples or invalid support/sigma.
pub fn gaussian_blur_f64(
    values: &[f64],
    size: ImageSize,
    channels: usize,
    sigma: f64,
    requested: Option<ImageSize>,
) -> Result<Vec<f64>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let support = kernel(sigma, requested, false)?;
    let horizontal = coefficients(sigma, support.width)?;
    let vertical = coefficients(sigma, support.height)?;
    let mut intermediate = vec![0.0; values.len()];
    let mut output = vec![0.0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..channels {
                let mut value = values[(y * size.width + x) * channels + channel]
                    * horizontal[support.width / 2];
                for tap in 1..=support.width / 2 {
                    let left = reflect101(x as isize - tap as isize, size.width);
                    let right = reflect101(x as isize + tap as isize, size.width);
                    value += (values[(y * size.width + left) * channels + channel]
                        + values[(y * size.width + right) * channels + channel])
                        * horizontal[support.width / 2 + tap];
                }
                intermediate[(y * size.width + x) * channels + channel] = value;
            }
        }
    }
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..channels {
                let mut value = intermediate[(y * size.width + x) * channels + channel]
                    * vertical[support.height / 2];
                for tap in 1..=support.height / 2 {
                    let above = reflect101(y as isize - tap as isize, size.height);
                    let below = reflect101(y as isize + tap as isize, size.height);
                    value += (intermediate[(above * size.width + x) * channels + channel]
                        + intermediate[(below * size.width + x) * channels + channel])
                        * vertical[support.height / 2 + tap];
                }
                output[(y * size.width + x) * channels + channel] = value;
            }
        }
    }
    Ok(output)
}

fn fixed_coefficients(sigma: f64, side: usize) -> Result<Vec<u16>, SpatialFilterError> {
    let weights = coefficients(sigma, side)?;
    let mut output = vec![0_u16; side];
    let mut error = 0.0;
    let mut sum = 0_u16;
    for i in 0..side / 2 {
        let adjusted = weights[i] * 256.0 + error;
        let rounded = adjusted.round_ties_even();
        error = adjusted - rounded;
        let value = rounded as u16;
        output[i] = value;
        output[side - 1 - i] = value;
        sum += value;
    }
    output[side / 2] = 256_u16
        .checked_sub(sum * 2)
        .ok_or(SpatialFilterError::Kernel)?;
    Ok(output)
}

/// Compute three-tap Sobel derivatives with FP32 samples and intermediates.
///
/// Reflect-101 borders and separable horizontal-then-vertical evaluation match
/// the CV_32F Sobel path with grouped symmetric taps. The vertical derivative smooths
/// each row before subtracting rows; FP64 filtering followed by a cast does not
/// preserve these intermediate rounding boundaries.
///
/// # Errors
/// Rejects invalid geometry or nonfinite samples.
pub fn sobel_xy_f32(
    values: &[f32],
    size: ImageSize,
) -> Result<(Vec<f32>, Vec<f32>), SpatialFilterError> {
    geometry(values.len(), size, 1)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let mut horizontal_derivative = vec![0.0_f32; values.len()];
    let mut horizontal_smoothing = vec![0.0_f32; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            let left = reflect101(x as isize - 1, size.width);
            let right = reflect101(x as isize + 1, size.width);
            let offset = y * size.width;
            horizontal_derivative[offset + x] = values[offset + right] - values[offset + left];
            horizontal_smoothing[offset + x] =
                (values[offset + left] + values[offset + right]) + values[offset + x] * 2.0;
        }
    }
    let mut dx = vec![0.0_f32; values.len()];
    let mut dy = vec![0.0_f32; values.len()];
    for y in 0..size.height {
        let top = reflect101(y as isize - 1, size.height) * size.width;
        let bottom = reflect101(y as isize + 1, size.height) * size.width;
        for x in 0..size.width {
            let offset = y * size.width + x;
            dx[offset] = (horizontal_derivative[top + x] + horizontal_derivative[bottom + x])
                + horizontal_derivative[offset] * 2.0;
            dy[offset] = horizontal_smoothing[bottom + x] - horizontal_smoothing[top + x];
        }
    }
    Ok((dx, dy))
}

/// Compute the two three-tap Sobel derivatives of a scalar FP64 image.
/// Reflect-101 borders and FP64 intermediate arithmetic are retained.
/// # Errors
/// Rejects invalid geometry or nonfinite samples.
pub fn sobel_xy_f64(
    values: &[f64],
    size: ImageSize,
) -> Result<(Vec<f64>, Vec<f64>), SpatialFilterError> {
    geometry(values.len(), size, 1)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let mut dx = vec![0.0; values.len()];
    let mut dy = vec![0.0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            let left = reflect101(x as isize - 1, size.width);
            let right = reflect101(x as isize + 1, size.width);
            let top = reflect101(y as isize - 1, size.height);
            let bottom = reflect101(y as isize + 1, size.height);
            let at = |yy: usize, xx: usize| values[yy * size.width + xx];
            dx[y * size.width + x] = (at(top, right) - at(top, left))
                + (at(bottom, right) - at(bottom, left))
                + 2.0 * (at(y, right) - at(y, left));
            dy[y * size.width + x] = (at(bottom, left) + at(bottom, right) + 2.0 * at(bottom, x))
                - (at(top, left) + at(top, right) + 2.0 * at(top, x));
        }
    }
    Ok((dx, dy))
}

/// Rectangular dilation of scalar FP64 pixels with the default center anchor.
/// Even support sizes retain the asymmetric floor(size/2) anchor. Outside
/// samples have neutral negative-infinite values.
/// # Errors
/// Rejects invalid geometry, empty support or nonfinite samples.
pub fn dilate_rect_f64(
    values: &[f64],
    size: ImageSize,
    support: ImageSize,
) -> Result<Vec<f64>, SpatialFilterError> {
    geometry(values.len(), size, 1)?;
    if support.width == 0 || support.height == 0 {
        return Err(SpatialFilterError::Kernel);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let mut horizontal = vec![f64::NEG_INFINITY; values.len()];
    let mut output = vec![f64::NEG_INFINITY; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            let start = x.saturating_sub(support.width / 2);
            let end = x
                .saturating_add(support.width - support.width / 2)
                .min(size.width);
            horizontal[y * size.width + x] = values[y * size.width + start..y * size.width + end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
        }
    }
    for y in 0..size.height {
        for x in 0..size.width {
            let start = y.saturating_sub(support.height / 2);
            let end = y
                .saturating_add(support.height - support.height / 2)
                .min(size.height);
            for yy in start..end {
                output[y * size.width + x] =
                    output[y * size.width + x].max(horizontal[yy * size.width + x]);
            }
        }
    }
    Ok(output)
}

/// Gaussian filtering of bytes with reflect-101 borders and fixed-point rounding.
///
/// Uses automatic `round(6*sigma+1)|1` support for byte inputs and symmetric
/// error-diffused eight-fractional-bit coefficients. Conversion back to bytes
/// occurs after both separable passes.
///
/// # Errors
/// Rejects invalid geometry, sigma or kernel dimensions.
pub fn gaussian_blur_u8(
    values: &[u8],
    size: ImageSize,
    channels: usize,
    sigma: f64,
    requested: Option<ImageSize>,
) -> Result<Vec<u8>, SpatialFilterError> {
    geometry(values.len(), size, channels)?;
    let support = kernel(sigma, requested, true)?;
    let horizontal = fixed_coefficients(sigma, support.width)?;
    let vertical = fixed_coefficients(sigma, support.height)?;
    let mut intermediate = vec![0_u32; values.len()];
    let mut output = vec![0_u8; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            for c in 0..channels {
                let mut sum = 0_u32;
                for (tap, &weight) in horizontal.iter().enumerate() {
                    let sx = reflect101(
                        x as isize + tap as isize - (support.width / 2) as isize,
                        size.width,
                    );
                    sum +=
                        u32::from(values[(y * size.width + sx) * channels + c]) * u32::from(weight);
                }
                intermediate[(y * size.width + x) * channels + c] = sum;
            }
        }
    }
    for y in 0..size.height {
        for x in 0..size.width {
            for c in 0..channels {
                let mut sum = 0_u32;
                for (tap, &weight) in vertical.iter().enumerate() {
                    let sy = reflect101(
                        y as isize + tap as isize - (support.height / 2) as isize,
                        size.height,
                    );
                    sum += intermediate[(sy * size.width + x) * channels + c] * u32::from(weight);
                }
                output[(y * size.width + x) * channels + c] = ((sum + 32768) >> 16) as u8;
            }
        }
    }
    Ok(output)
}

/// Dilate a scalar FP32 plane with a nonempty binary footprint and neutral borders.
///
/// # Errors
/// Rejects inconsistent dimensions, nonfinite samples or invalid footprints.
pub fn dilate_f32(
    values: &[f32],
    size: ImageSize,
    footprint: &[u8],
    support: ImageSize,
) -> Result<Vec<f32>, SpatialFilterError> {
    geometry(values.len(), size, 1)?;
    geometry(footprint.len(), support, 1)?;
    if support.width.is_multiple_of(2)
        || support.height.is_multiple_of(2)
        || !footprint.iter().any(|&v| v != 0)
    {
        return Err(SpatialFilterError::Kernel);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SpatialFilterError::NonFinite);
    }
    let mut output = vec![0.0_f32; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            let mut maximum = f32::NEG_INFINITY;
            for ky in 0..support.height {
                for kx in 0..support.width {
                    if footprint[ky * support.width + kx] == 0 {
                        continue;
                    }
                    let sy = y as isize + ky as isize - (support.height / 2) as isize;
                    let sx = x as isize + kx as isize - (support.width / 2) as isize;
                    if sy >= 0 && sx >= 0 && sy < size.height as isize && sx < size.width as isize {
                        maximum = maximum.max(values[sy as usize * size.width + sx as usize]);
                    }
                }
            }
            output[y * size.width + x] = maximum;
        }
    }
    Ok(output)
}
