//! FP32 Gaussian filtering with zero border values.

use crate::{ImageSize, ResizeError};

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
