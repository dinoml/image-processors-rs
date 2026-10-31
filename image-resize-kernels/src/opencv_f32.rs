//! Optional native OpenCV floating-point interpolation without byte quantization.

use ::opencv::{
    core::{Mat, Size},
    imgproc,
    prelude::*,
};

use super::{image_len, validate_image_data, ImageSize, ResizeError};

impl From<::opencv::Error> for ResizeError {
    fn from(error: ::opencv::Error) -> Self {
        Self::OpenCv {
            code: error.code,
            message: error.message,
        }
    }
}

/// Resize interleaved FP32 values using OpenCV `INTER_LINEAR`.
///
/// Coordinates, FP32 coefficients, border behavior and accumulation are supplied
/// by OpenCV itself. No intermediate byte conversion or antialiasing is applied.
///
/// # Errors
/// Rejects invalid dimensions, channel counts, nonfinite values, storage-length
/// mismatches and dimensions outside OpenCV's signed-integer limits. Propagates
/// native resize errors.
pub fn resize_f32_opencv_linear(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    validate_image_data(values.len(), source, channels)?;
    let target_elements = image_len(target, channels)?;
    for (dimension, value) in [
        ("source height", source.height),
        ("source width", source.width),
        ("target height", target.height),
        ("target width", target.width),
        ("source elements", values.len()),
        ("target elements", target_elements),
        ("channels", channels),
    ] {
        if value > i32::MAX as usize {
            return Err(ResizeError::DimensionTooLarge {
                dimension,
                value,
                max: i32::MAX as usize,
            });
        }
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ResizeError::NonFiniteValue { index });
    }
    if source == target {
        return Ok(values.to_vec());
    }
    let data = Mat::from_slice(values)?;
    let source_mat = data.reshape(channels as i32, source.height as i32)?;
    let mut resized = Mat::default();
    imgproc::resize(
        &source_mat,
        &mut resized,
        Size::new(target.width as i32, target.height as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let flat = resized.reshape(1, 1)?;
    Ok(flat.data_typed::<f32>()?.to_vec())
}

/// Divide byte values in FP64, cast each result to FP32, then resize in FP32.
///
/// The division precedes interpolation and does not use an FP32 reciprocal.
/// This preserves NumPy's `(uint8_image / divisor).astype(float32)` rounding.
///
/// # Errors
/// Rejects nonpositive/nonfinite divisors and all invalid inputs reported by
/// [`resize_f32_opencv_linear`], including division results that overflow FP32.
pub fn resize_u8_to_f32_opencv_linear(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    divisor: f64,
) -> Result<Vec<f32>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    if !divisor.is_finite() || divisor <= 0.0 {
        return Err(ResizeError::InvalidRescaleDivisor(divisor));
    }
    let values = values
        .iter()
        .map(|value| (f64::from(*value) / divisor) as f32)
        .collect::<Vec<_>>();
    resize_f32_opencv_linear(&values, source, channels, target)
}
