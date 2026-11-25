//! Local percentile normalization and optional circular surround filtering.

use super::{elements, invalid, percentile, starts, CellposeNormalization, Result};
use crate::transforms::{gaussian_filter_reflect_f64, ImageSize};
use image_resize_kernels::{resize_f64_opencv, OpenCvInterpolation};

pub(super) fn smooth_sharpen(
    pixels: &mut [f32],
    shape: &[usize],
    smooth: f32,
    sharpen: f32,
) -> Result<()> {
    if !smooth.is_finite() || !sharpen.is_finite() || smooth < 0.0 || sharpen < 0.0 {
        return Err(invalid("invalid smoothing/sharpening radius"));
    }
    if smooth == 0.0 && sharpen == 0.0 {
        return Ok(());
    }
    let height = shape[shape.len() - 2];
    let width = shape[shape.len() - 1];
    let area = elements(&[height, width])?;
    // The upstream FFT implements circular correlation followed by fftshift.
    // The separable Gaussian permits the same boundary/centering mathematics
    // directly in the preprocessing owner, without a new FFT dependency.
    let filter = |input: &[f32], sigma: f32| {
        let kernel = |length: usize| {
            let mut values = (0..length)
                .map(|i| {
                    let coordinate = if length == 1 {
                        -(height as f32) / 2.0
                    } else {
                        -(height as f32) / 2.0
                            + i as f32 * (height as f32 + 1.0) / (length - 1) as f32
                    };
                    (-(coordinate * coordinate) / (2.0 * sigma * sigma)).exp()
                })
                .collect::<Vec<_>>();
            let sum = values.iter().sum::<f32>();
            for value in &mut values {
                *value /= sum;
            }
            values
        };
        let ky = kernel(height);
        let kx = kernel(width);
        let mut horizontal = vec![0.0; area];
        let mut output = vec![0.0; area];
        for y in 0..height {
            for x in 0..width {
                for (k, &weight) in kx.iter().enumerate() {
                    horizontal[y * width + x] +=
                        input[y * width + (x + width.div_ceil(2) + k) % width] * weight;
                }
            }
        }
        for y in 0..height {
            for x in 0..width {
                for (k, &weight) in ky.iter().enumerate() {
                    output[y * width + x] +=
                        horizontal[((y + height.div_ceil(2) + k) % height) * width + x] * weight;
                }
            }
        }
        output
    };
    for plane in pixels.chunks_exact_mut(area) {
        let source = plane.to_vec();
        let mut output = if smooth > 0.0 {
            filter(&source, smooth)
        } else {
            source.clone()
        };
        if sharpen > 0.0 {
            let surround = filter(&source, sharpen);
            for (out, value) in output.iter_mut().zip(surround) {
                *out -= value;
            }
            if smooth == 0.0 {
                // Upstream replaces, rather than adds to, the central coefficient.
                let center_y = height / 2;
                let center_x = width / 2;
                let gaussian = |length: usize, index: usize| {
                    let weights = (0..length)
                        .map(|i| {
                            let x = if length == 1 {
                                -(height as f32) / 2.0
                            } else {
                                -(height as f32) / 2.0
                                    + i as f32 * (height as f32 + 1.0) / (length - 1) as f32
                            };
                            (-(x * x) / (2.0 * sharpen * sharpen)).exp()
                        })
                        .collect::<Vec<_>>();
                    weights[index] / weights.iter().sum::<f32>()
                };
                let center = gaussian(height, center_y) * gaussian(width, center_x);
                for (out, &value) in output.iter_mut().zip(&source) {
                    *out += value * center;
                }
            }
        }
        plane.copy_from_slice(&output);
    }
    Ok(())
}

pub(super) fn local_percentiles(
    pixels: &mut [f32],
    shape: &[usize],
    options: &CellposeNormalization,
) -> Result<()> {
    let (depth, height, width) = if shape.len() == 3 {
        (shape[0], shape[1], shape[2])
    } else {
        (1, shape[0], shape[1])
    };
    let area = elements(&[height, width])?;
    let block = options.tile_norm_blocksize;
    let bh = block.min(height);
    let bw = block.min(width);
    let ys = starts(height, bh, 0.1, false);
    let xs = starts(width, bw, 0.1, false);
    let grid = ys.len() * xs.len();
    if !options.tile_norm_smooth_3d.is_finite() || options.tile_norm_smooth_3d < 0.0 {
        return Err(invalid("invalid normalization grid smoothing"));
    }
    for c in 0..3 {
        let mut low = vec![0.0f64; depth * grid];
        let mut high = low.clone();
        for z in 0..depth {
            for (j, &y) in ys.iter().enumerate() {
                for (i, &x) in xs.iter().enumerate() {
                    let mut sample = Vec::with_capacity(bh * bw);
                    for yy in y..y + bh {
                        sample.extend_from_slice(
                            &pixels[(c * depth + z) * area + yy * width + x
                                ..(c * depth + z) * area + yy * width + x + bw],
                        );
                    }
                    sample.sort_by(f32::total_cmp);
                    low[z * grid + j * xs.len() + i] = percentile(&sample, options.percentile[0]);
                    high[z * grid + j * xs.len() + i] = percentile(&sample, options.percentile[1]);
                }
            }
            let good = (0..grid)
                .filter(|&i| high[z * grid + i] - low[z * grid + i] >= 1e-3)
                .collect::<Vec<_>>();
            for i in 0..grid {
                if high[z * grid + i] - low[z * grid + i] >= 1e-3 {
                    continue;
                }
                if let Some(&nearest) = good.iter().min_by_key(|&&j| {
                    (i / xs.len()).abs_diff(j / xs.len()).pow(2)
                        + (i % xs.len()).abs_diff(j % xs.len()).pow(2)
                }) {
                    low[z * grid + i] = low[z * grid + nearest];
                    high[z * grid + i] = high[z * grid + nearest];
                } else {
                    low[z * grid + i] = 0.0;
                    high[z * grid + i] = 1.0;
                }
            }
        }
        // Preserve the original axes: grids have Z,Ytile,Xtile order, and the
        // reference smooths axes zero and one, then optionally axis one again.
        let grid_size = ImageSize {
            height: depth,
            width: ys.len(),
        };
        for sigma in [[1.0, 0.0], [0.0, 1.0]] {
            low = gaussian_filter_reflect_f64(&low, grid_size, xs.len(), sigma, 4.0)
                .map_err(|e| invalid(e.to_string()))?;
            high = gaussian_filter_reflect_f64(&high, grid_size, xs.len(), sigma, 4.0)
                .map_err(|e| invalid(e.to_string()))?;
        }
        if options.norm_3d {
            let sigma = if options.tile_norm_smooth_3d == 0.0 {
                1.0
            } else {
                options.tile_norm_smooth_3d
            };
            low = gaussian_filter_reflect_f64(&low, grid_size, xs.len(), [0.0, sigma], 4.0)
                .map_err(|e| invalid(e.to_string()))?;
            high = gaussian_filter_reflect_f64(&high, grid_size, xs.len(), [0.0, sigma], 4.0)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let common = options.norm_3d || depth == 1;
        let mean = |values: &[f64]| {
            (0..grid)
                .map(|i| (0..depth).map(|z| values[z * grid + i]).sum::<f64>() / depth as f64)
                .collect::<Vec<_>>()
        };
        let common_low = common.then(|| mean(&low));
        let common_high = common.then(|| mean(&high));
        for z in 0..depth {
            let source = image_resize_kernels::ImageSize {
                height: ys.len(),
                width: xs.len(),
            };
            let target = image_resize_kernels::ImageSize { height, width };
            let lower = resize_f64_opencv(
                common_low
                    .as_deref()
                    .unwrap_or(&low[z * grid..(z + 1) * grid]),
                source,
                1,
                target,
                OpenCvInterpolation::Linear,
            )
            .map_err(|e| invalid(e.to_string()))?;
            let upper = resize_f64_opencv(
                common_high
                    .as_deref()
                    .unwrap_or(&high[z * grid..(z + 1) * grid]),
                source,
                1,
                target,
                OpenCvInterpolation::Linear,
            )
            .map_err(|e| invalid(e.to_string()))?;
            // normalize99_tile assigns x01_rsz to BOTH multichannel output
            // bounds in its per-Z branch. Our canonical input has three
            // channels, so preserve that assignment and its range rejection.
            let upper = if common { upper } else { lower.clone() };
            for p in 0..area {
                // In the original non-volume multichannel branch, x01/x99
                // already have YXC shape when they are indexed with [0].
                // This selects row zero, which then broadcasts across Y.
                let bound = if shape.len() == 2 { p % width } else { p };
                if !common && upper[bound] - lower[bound] < 1e-3 {
                    return Err(invalid(
                        "cannot use norm3D=False with tile_norm, sample is too sparse; set norm3D=True or tile_norm=0",
                    ));
                }
                let value = &mut pixels[(c * depth + z) * area + p];
                let (low, high) = if common {
                    (lower[bound], upper[bound])
                } else {
                    (
                        f64::from(lower[bound] as f32),
                        f64::from(upper[bound] as f32),
                    )
                };
                *value = (f64::from(*value) - low) as f32;
                let denominator = if common {
                    high - low
                } else {
                    f64::from((high - low) as f32)
                };
                *value = (f64::from(*value) / denominator) as f32;
                if options.invert {
                    *value = 1.0 - *value;
                }
            }
        }
    }
    Ok(())
}
