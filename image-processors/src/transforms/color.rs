//! Floating-point MATLAB-style RGB and YCbCr color transforms.

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
