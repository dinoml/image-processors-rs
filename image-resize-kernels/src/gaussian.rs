//! Gaussian filtering with explicit accumulation and border policies.

use crate::{ImageSize, ResizeError};

/// Separable Gaussian filtering with SciPy's half-sample symmetric boundary.
///
/// The height axis runs before width. Weights and sums use FP64, with FP32
/// rounding after each axis, matching `scipy.ndimage.gaussian_filter` on FP32
/// planes. `kernel=1` is an identity filter.
///
/// # Errors
/// Reports invalid geometry, buffer length, kernel, sigma or size overflow.
pub fn gaussian_blur_reflect_f32_planes(
    values: &[f32],
    size: ImageSize,
    planes: usize,
    kernel: usize,
    sigma: f64,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(size.height, size.width)?;
    if kernel == 0
        || kernel.is_multiple_of(2)
        || kernel > isize::MAX as usize
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(ResizeError::InvalidScaleFactor(sigma as f32));
    }
    let pixels = size
        .height
        .checked_mul(size.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let length = pixels
        .checked_mul(planes)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if values.len() != length {
        return Err(ResizeError::InvalidBufferLength {
            expected: length,
            actual: values.len(),
        });
    }
    if planes == 0 {
        return Ok(Vec::new());
    }
    let radius = kernel / 2;
    let mut weights = (0..=radius)
        .map(|distance| (-0.5 / (sigma * sigma) * (distance as f64).powi(2)).exp())
        .collect::<Vec<_>>();
    let sum = weights[0] + 2.0 * weights[1..].iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= sum;
    }
    let reflect = |index: i128, extent: usize| {
        let extent = extent as i128;
        let index = index.rem_euclid(2 * extent);
        if index < extent {
            index as usize
        } else {
            (2 * extent - 1 - index) as usize
        }
    };
    let mut intermediate = vec![0.0; length];
    let mut output = vec![0.0; length];
    for plane in 0..planes {
        let offset = plane * pixels;
        for y in 0..size.height {
            for x in 0..size.width {
                let mut value = f64::from(values[offset + y * size.width + x]) * weights[0];
                // SciPy's symmetric correlate1d visits the outer pair first.
                for distance in (1..=radius).rev() {
                    let above = reflect(y as i128 - distance as i128, size.height);
                    let below = reflect(y as i128 + distance as i128, size.height);
                    value += (f64::from(values[offset + above * size.width + x])
                        + f64::from(values[offset + below * size.width + x]))
                        * weights[distance];
                }
                intermediate[offset + y * size.width + x] = value as f32;
            }
        }
        for y in 0..size.height {
            for x in 0..size.width {
                let row = offset + y * size.width;
                let mut value = f64::from(intermediate[row + x]) * weights[0];
                for distance in (1..=radius).rev() {
                    let left = reflect(x as i128 - distance as i128, size.width);
                    let right = reflect(x as i128 + distance as i128, size.width);
                    value += (f64::from(intermediate[row + left])
                        + f64::from(intermediate[row + right]))
                        * weights[distance];
                }
                output[row + x] = value as f32;
            }
        }
    }
    Ok(output)
}

/// Blur independent planar channels with a normalized Gaussian outer product.
///
/// Weights and accumulation are FP32, matching Torchvision tensor Gaussian
/// filtering of a zero-padded image followed by removal of that padding.
///
/// # Errors
/// Reports invalid sizes, buffer lengths, kernel size, sigma or size overflow.
pub fn gaussian_blur_zero_f32_planes(
    values: &[f32],
    size: ImageSize,
    planes: usize,
    kernel: usize,
    sigma: f32,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(size.height, size.width)?;
    if kernel <= 1
        || kernel.is_multiple_of(2)
        || kernel > isize::MAX as usize
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(ResizeError::InvalidScaleFactor(sigma));
    }
    let pixels = size
        .height
        .checked_mul(size.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let length = pixels
        .checked_mul(planes)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if values.len() != length {
        return Err(ResizeError::InvalidBufferLength {
            expected: length,
            actual: values.len(),
        });
    }
    let radius = (kernel - 1) / 2;
    let mut gaussian = (0..kernel)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect::<Vec<_>>();
    let sum: f32 = gaussian.iter().sum();
    for value in &mut gaussian {
        *value /= sum;
    }
    let mut output = vec![0.0; length];
    for (source, destination) in values
        .chunks_exact(pixels)
        .zip(output.chunks_exact_mut(pixels))
    {
        for y in 0..size.height {
            for x in 0..size.width {
                let mut value = 0.0;
                for (ky, wy) in gaussian.iter().enumerate() {
                    let iy = y as isize + ky as isize - radius as isize;
                    if iy < 0 || iy >= size.height as isize {
                        continue;
                    }
                    for (kx, wx) in gaussian.iter().enumerate() {
                        let ix = x as isize + kx as isize - radius as isize;
                        if ix >= 0 && ix < size.width as isize {
                            value += source[iy as usize * size.width + ix as usize] * (wy * wx);
                        }
                    }
                }
                destination[y * size.width + x] = value;
            }
        }
    }
    Ok(output)
}
