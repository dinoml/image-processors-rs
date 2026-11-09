//! KAIR utils_image.imresize_np: antialiased cubic, symmetric edges, FP32 passes.

use crate::{ImageSize, ResizeError};

/// Integer downsampling of planar FP32 images using KAIR's MATLAB-style cubic.
///
/// The cubic coefficient is -0.5, with widened antialias support, half-pixel
/// coordinates, edge-inclusive symmetric extension, and height before width.
/// Coefficients, normalization, dot products and intermediate pixels are FP32.
/// No range clipping or integer quantization is performed. This differs from
/// both OpenCV cubic and Torchvision's boundary-renormalized antialias profile.
///
/// # Errors
/// Rejects zero/overflowing geometry, a zero divisor, non-divisible input sizes,
/// incompatible buffers or nonfinite pixels.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "KAIR explicitly calculates indices and coefficients in FP32; geometry is bounded before signed indexing"
)]
pub fn downsample_f32_kair_bicubic_planes(
    values: &[f32],
    source: ImageSize,
    planes: usize,
    divisor: usize,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    if divisor == 0
        || planes == 0
        || !source.height.is_multiple_of(divisor)
        || !source.width.is_multiple_of(divisor)
    {
        return Err(ResizeError::InvalidScaleFactor(divisor as f32));
    }
    let taps = divisor
        .checked_mul(4)
        .and_then(|value| value.checked_add(2))
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if source.height > (isize::MAX as usize) / 4
        || source.width > (isize::MAX as usize) / 4
        || taps > (isize::MAX as usize) / 4
    {
        return Err(ResizeError::ImageSizeOverflow);
    }
    let pixels = source
        .height
        .checked_mul(source.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let expected = pixels
        .checked_mul(planes)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if values.len() != expected {
        return Err(ResizeError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ResizeError::NonFiniteValue { index });
    }
    let target = ImageSize::new(source.height / divisor, source.width / divisor)?;
    let vertical = coefficients(target.height, divisor, taps);
    let horizontal = coefficients(target.width, divisor, taps);
    let intermediate_pixels = target
        .height
        .checked_mul(source.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let output_pixels = target
        .height
        .checked_mul(target.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let mut intermediate = vec![
        0.0_f32;
        intermediate_pixels
            .checked_mul(planes)
            .ok_or(ResizeError::ImageSizeOverflow)?
    ];
    let mut output = vec![
        0.0_f32;
        output_pixels
            .checked_mul(planes)
            .ok_or(ResizeError::ImageSizeOverflow)?
    ];
    for channel in 0..planes {
        for (y, (left, weights)) in vertical.iter().enumerate() {
            for x in 0..source.width {
                let mut sum = 0.0_f32;
                for (tap, weight) in weights.iter().enumerate() {
                    let sy = symmetric(*left + tap as isize, source.height);
                    sum += values[channel * pixels + sy * source.width + x] * weight;
                }
                intermediate[channel * intermediate_pixels + y * source.width + x] = sum;
            }
        }
        for y in 0..target.height {
            for (x, (left, weights)) in horizontal.iter().enumerate() {
                let mut sum = 0.0_f32;
                for (tap, weight) in weights.iter().enumerate() {
                    let sx = symmetric(*left + tap as isize, source.width);
                    sum += intermediate[channel * intermediate_pixels + y * source.width + sx]
                        * weight;
                }
                output[channel * output_pixels + y * target.width + x] = sum;
            }
        }
    }
    Ok(output)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "source FP32 coordinate arithmetic with validated extents"
)]
fn coefficients(length: usize, divisor: usize, taps: usize) -> Vec<(isize, Vec<f32>)> {
    let scale = (1.0_f64 / divisor as f64) as f32;
    let offset = (0.5 * (1.0 - divisor as f64)) as f32;
    (0..length)
        .map(|index| {
            let center = (index + 1) as f32 / scale + offset;
            let left = (center - (2 * divisor) as f32).floor() as isize;
            let mut weights = (0..taps)
                .map(|tap| {
                    let distance = ((center - (left + tap as isize) as f32) * scale).abs();
                    let squared = distance * distance;
                    let cubed = squared * distance;
                    let cubic = if distance <= 1.0 {
                        1.5 * cubed - 2.5 * squared + 1.0
                    } else if distance <= 2.0 {
                        -0.5 * cubed + 2.5 * squared - 4.0 * distance + 2.0
                    } else {
                        0.0
                    };
                    scale * cubic
                })
                .collect::<Vec<_>>();
            let total: f32 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= total;
            }
            // KAIR starts its source indices at one. Zero-weight border columns
            // can remain here without changing the extension or dot product.
            (left - 1, weights)
        })
        .collect()
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "validated geometry and Euclidean remainder give an in-bounds index"
)]
fn symmetric(index: isize, length: usize) -> usize {
    let period = 2 * length as isize;
    let position = index.rem_euclid(period) as usize;
    if position < length {
        position
    } else {
        2 * length - position - 1
    }
}
