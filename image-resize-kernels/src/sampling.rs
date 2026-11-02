//! FP32 planar sampling with normalized coordinates and zero border values.

use crate::{ImageSize, ResizeError};

/// Tensor grid-sampling interpolation, without antialiasing or integer rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridInterpolation {
    /// Four-neighbor bilinear interpolation.
    Bilinear,
    /// Sixteen-neighbor cubic interpolation with coefficient -0.75.
    Bicubic,
}

/// Sample planar channels with an interleaved normalized XY grid.
///
/// Coordinates follow PyTorch grid_sample with align_corners=true and zero
/// padding. Out-of-bounds neighbors contribute zero even at fractional borders.
/// All coordinate and interpolation arithmetic is FP32.
///
/// # Errors
/// Returns an error for invalid dimensions, buffer lengths or nonfinite grid values.
pub fn grid_sample_f32_planes(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    grid: &[[f32; 2]],
    target: ImageSize,
    interpolation: GridInterpolation,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    let source_pixels = source
        .height
        .checked_mul(source.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let target_pixels = target
        .height
        .checked_mul(target.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let expected = source_pixels
        .checked_mul(channels)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if channels == 0 || values.len() != expected {
        return Err(ResizeError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    if grid.len() != target_pixels {
        return Err(ResizeError::InvalidBufferLength {
            expected: target_pixels,
            actual: grid.len(),
        });
    }
    let length = target_pixels
        .checked_mul(channels)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let mut output = vec![0.0; length];
    for (pixel, coordinate) in grid.iter().enumerate() {
        if coordinate.iter().any(|value| !value.is_finite()) {
            return Err(ResizeError::NonFiniteValue { index: pixel });
        }
        let x = (coordinate[0] + 1.0) / 2.0 * (source.width - 1) as f32;
        let y = (coordinate[1] + 1.0) / 2.0 * (source.height - 1) as f32;
        // Entire interpolation support lies outside the source at these coordinates.
        if x < -2.0 || y < -2.0 || x > source.width as f32 + 1.0 || y > source.height as f32 + 1.0 {
            continue;
        }
        let ix = x.floor() as isize;
        let iy = y.floor() as isize;
        let dx = x - ix as f32;
        let dy = y - iy as f32;
        for channel in 0..channels {
            let at = |row: isize, column: isize| -> f32 {
                if row < 0
                    || column < 0
                    || row >= source.height as isize
                    || column >= source.width as isize
                {
                    0.0
                } else {
                    values[channel * source_pixels + row as usize * source.width + column as usize]
                }
            };
            output[channel * target_pixels + pixel] = match interpolation {
                GridInterpolation::Bilinear => {
                    let northwest = (1.0 - dx) * (1.0 - dy);
                    let northeast = dx * (1.0 - dy);
                    let southwest = (1.0 - dx) * dy;
                    let southeast = dx * dy;
                    at(iy, ix) * northwest
                        + at(iy, ix + 1) * northeast
                        + at(iy + 1, ix) * southwest
                        + at(iy + 1, ix + 1) * southeast
                }
                GridInterpolation::Bicubic => {
                    let cx = cubic_coefficients(dx);
                    let cy = cubic_coefficients(dy);
                    let mut rows = [0.0; 4];
                    for (row, value) in rows.iter_mut().enumerate() {
                        for (column, weight) in cx.iter().enumerate() {
                            *value += at(iy + row as isize - 1, ix + column as isize - 1) * weight;
                        }
                    }
                    rows.iter()
                        .zip(cy)
                        .map(|(value, weight)| value * weight)
                        .sum()
                }
            };
        }
    }
    Ok(output)
}

fn cubic_coefficients(fraction: f32) -> [f32; 4] {
    // Tensor grid_sample evaluates these polynomials in FP32; the resize
    // owner's existing f64 coefficient builder has different rounding semantics.
    let cubic1 = |x: f32| ((1.25 * x - 2.25) * x) * x + 1.0;
    let cubic2 = |x: f32| ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0;
    [
        cubic2(fraction + 1.0),
        cubic1(fraction),
        cubic1(1.0 - fraction),
        cubic2(2.0 - fraction),
    ]
}
