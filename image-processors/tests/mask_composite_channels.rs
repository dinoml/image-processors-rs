use image_processors::transforms::{
    composite_mask_f32_frame_with_precision, MaskCompositePrecision,
};
use image_processors::{ImageFrame, PixelFormat};

fn rgb(values: &[u8]) -> ImageFrame {
    ImageFrame::new(values.len() / 3, 1, PixelFormat::Rgb8, values.to_vec())
        .expect("fixture has complete RGB pixels")
}

#[test]
fn per_channel_opacity_preserves_independent_rgb_masks() {
    let background = rgb(&[10, 20, 30, 100, 110, 120]);
    let foreground = rgb(&[110, 120, 130, 200, 210, 220]);
    for precision in [
        MaskCompositePrecision::Float32,
        MaskCompositePrecision::Float32ForegroundFloat64Background,
    ] {
        let output = composite_mask_f32_frame_with_precision(
            &background,
            &foreground,
            &[0.0, 0.5, 1.0, 1.0, 0.0, 0.25],
            precision,
        )
        .expect("one valid opacity per RGB channel");
        assert_eq!(output.data(), &[10, 70, 130, 200, 110, 145]);
    }
}

#[test]
fn per_pixel_opacity_still_broadcasts_over_channels() {
    let output = composite_mask_f32_frame_with_precision(
        &rgb(&[10, 20, 30, 100, 110, 120]),
        &rgb(&[110, 120, 130, 200, 210, 220]),
        &[0.25, 0.75],
        MaskCompositePrecision::Float32,
    )
    .expect("legacy per-pixel mask remains valid");
    assert_eq!(output.data(), &[35, 45, 55, 175, 185, 195]);
}

#[test]
fn mixed_precision_retains_the_background_promotion_boundary() {
    let background = rgb(&[5, 5, 5]);
    let foreground = rgb(&[255, 255, 255]);
    for (precision, expected) in [
        (MaskCompositePrecision::Float32, [30, 30, 30]),
        (
            MaskCompositePrecision::Float32ForegroundFloat64Background,
            [29, 29, 29],
        ),
    ] {
        let output = composite_mask_f32_frame_with_precision(
            &background,
            &foreground,
            &[0.1, 0.1, 0.1],
            precision,
        )
        .expect("valid RGB mask with fractional opacity");
        assert_eq!(output.data(), &expected);
    }
}

#[test]
fn malformed_mask_lengths_are_rejected() {
    let frame = rgb(&[1, 2, 3, 4, 5, 6]);
    assert!(composite_mask_f32_frame_with_precision(
        &frame,
        &frame,
        &[0.5; 3],
        MaskCompositePrecision::Float32,
    )
    .is_err());
}

#[test]
fn nonfinite_channel_opacity_is_rejected() {
    let frame = rgb(&[1, 2, 3]);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(composite_mask_f32_frame_with_precision(
            &frame,
            &frame,
            &[0.0, invalid, 1.0],
            MaskCompositePrecision::Float32,
        )
        .is_err());
    }
}
