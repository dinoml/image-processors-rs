//! PyTorch adaptive-average area interpolation for floating image boundaries.

use rayon::prelude::*;

use crate::{image_len, validate_image_data, ImageSize, ResizeError};

/// Resize interleaved FP32 pixels using PyTorch's contiguous area-mode semantics.
///
/// Each output bin covers `[floor(i * input / output), ceil((i + 1) * input /
/// output))` on each axis, including enlargement and mixed-axis resizing. Bins
/// overlap at fractional boundaries; this is not fractional pixel-area weighting.
/// Values accumulate in FP32 in source row/column order, then divide by bin
/// height and bin width separately, as in ATen's contiguous adaptive-average
/// pooling CPU/CUDA kernels. Channel interleaving does not change that order.
///
/// # Errors
/// Rejects zero dimensions, unsupported channels, inconsistent storage, size
/// overflow and nonfinite input values. Input and output use the same channel count.
pub fn resize_f32_area_torch(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    validate_image_data(values.len(), source, channels)?;
    let output_len = image_len(target, channels)?;
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ResizeError::NonFiniteValue { index });
    }
    // Widen before multiplication so large, otherwise valid dimensions cannot
    // overflow while deriving the exact integer bin boundaries.
    let columns: Vec<(usize, usize)> = (0..target.width)
        .map(|column| {
            let start = column as u128 * source.width as u128 / target.width as u128;
            let end =
                1 + (((column + 1) as u128 * source.width as u128 - 1) / target.width as u128);
            (start as usize, end as usize)
        })
        .collect();
    let mut output = vec![0.0_f32; output_len];
    output
        .par_chunks_exact_mut(target.width * channels)
        .enumerate()
        .for_each(|(row, output_row)| {
            let start_row = (row as u128 * source.height as u128 / target.height as u128) as usize;
            let end_row = (1
                + (((row + 1) as u128 * source.height as u128 - 1) / target.height as u128))
                as usize;
            for (column, &(start_column, end_column)) in columns.iter().enumerate() {
                for channel in 0..channels {
                    let mut sum = 0.0_f32;
                    for input_row in start_row..end_row {
                        for input_column in start_column..end_column {
                            sum += values
                                [(input_row * source.width + input_column) * channels + channel];
                        }
                    }
                    output_row[column * channels + channel] =
                        sum / (end_row - start_row) as f32 / (end_column - start_column) as f32;
                }
            }
        });
    Ok(output)
}
