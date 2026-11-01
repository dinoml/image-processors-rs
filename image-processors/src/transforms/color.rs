//! Shared color transforms retaining source-specific precision and rounding.

use super::{TransformError, new_frame};
use crate::media::{ImageFrame, PixelFormat};

/// Convert planar FP32 RGB predictions to grayscale before integer quantization.
/// Uses the upstream BGR scalar evaluation order
/// `(B * 0.114 + G * 0.587) + R * 0.299`, with FP32 intermediates and no
/// clipping or integer rounding. This fixes a scalar policy rather than assuming
/// platform-dependent SIMD fusion. The existing NTSC and byte contracts are unchanged.
///
/// # Errors
/// Rejects data that cannot contain three equally sized channel planes.
pub fn rgb_planar_f32_to_grayscale_opencv(values: &[f32]) -> Result<Vec<f32>, TransformError> {
    if !values.len().is_multiple_of(3) {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 3,
            actual: values.len(),
        });
    }
    let plane = values.len() / 3;
    Ok((0..plane)
        .map(|index| {
            (values[2 * plane + index] * 0.114_f32 + values[plane + index] * 0.587_f32)
                + values[index] * 0.299_f32
        })
        .collect())
}

/// Convert decoded RGB/RGBA bytes to grayscale using the OpenCV CPU contract.
///
/// Uses the 15-bit BT.601 coefficients from OpenCV's `RGB2Gray<uchar>`;
/// alpha is ignored. The existing general luminance conversion is unchanged.
/// Frame timing is preserved. No native image-processing library is called.
///
/// # Errors
/// Returns an error when the output frame cannot be represented.
pub fn grayscale_opencv(image: &ImageFrame) -> Result<ImageFrame, TransformError> {
    if image.pixel_format() == PixelFormat::Luma8 {
        return Ok(image.clone());
    }
    let data = image
        .data()
        .chunks_exact(image.channels())
        .map(|pixel| {
            ((u32::from(pixel[0]) * 9798
                + u32::from(pixel[1]) * 19235
                + u32::from(pixel[2]) * 3735
                + 16384)
                >> 15) as u8
        })
        .collect();
    new_frame(
        image.width(),
        image.height(),
        PixelFormat::Luma8,
        data,
        image.timing(),
    )
}

/// Convert one unit-range RGB pixel to MATLAB-style YCbCr in FP32.
///
/// The input is copied and remains unchanged. Multiplication by 255 is performed
/// in FP32 before promotion to the FP64 matrix calculation. Each matrix result
/// is divided by 255, offset, divided by 255 again, and finally converted back
/// to FP32. These precision boundaries are part of the conversion contract;
/// combining the scale factors changes rounding. Output values are not clipped.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the conversion contract returns FP32 after the FP64 matrix calculation"
)]
pub fn matlab_rgb_to_ycbcr_f32(rgb: [f32; 3]) -> [f32; 3] {
    let [red, green, blue] = rgb.map(|value| f64::from(value * 255.0_f32));
    let converted = [
        (red * 65.481 + green * 128.553 + blue * 24.966) / 255.0 + 16.0,
        (red * -37.797 + green * -74.203 + blue * 112.0) / 255.0 + 128.0,
        (red * 112.0 + green * -93.786 + blue * -18.214) / 255.0 + 128.0,
    ];
    converted.map(|value| (value / 255.0) as f32)
}

/// Convert one unit-range MATLAB-style YCbCr pixel to unit-range RGB in FP32.
///
/// The input is copied and remains unchanged. Multiplication by 255 is performed
/// in FP32 before promotion to the FP64 matrix calculation. Matrix results are
/// multiplied by 255, offset, clipped to `[0, 255]`, divided by 255, and finally
/// converted back to FP32. The decimal inverse coefficients and operation order
/// are preserved explicitly, including clipping after the offset rather than
/// clipping the input luminance or chroma.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the conversion contract returns FP32 after the FP64 matrix calculation"
)]
pub fn matlab_ycbcr_to_rgb_f32(ycbcr: [f32; 3]) -> [f32; 3] {
    let [luminance, cb, cr] = ycbcr.map(|value| f64::from(value * 255.0_f32));
    let converted = [
        (luminance * 0.00456621 + cb * 0.0 + cr * 0.00625893) * 255.0 - 222.921,
        (luminance * 0.00456621 + cb * -0.00153632 + cr * -0.00318811) * 255.0 + 135.576,
        (luminance * 0.00456621 + cb * 0.00791071 + cr * 0.0) * 255.0 - 276.836,
    ];
    converted.map(|value| (value.clamp(0.0, 255.0) / 255.0) as f32)
}
