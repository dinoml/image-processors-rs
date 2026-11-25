use image_processors::media::{jpeg_reference_roundtrip, ImageFrame, MediaError, PixelFormat};
use image_processors::transforms::{
    composite_alpha_to_rgb_with_rounding, pad_frame_with_canvas_fill, CanvasFill, Padding,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    jpeg: Vec<JpegCase>,
    alpha: Vec<AlphaCase>,
    padding: Vec<PaddingCase>,
}
#[derive(Deserialize)]
struct JpegCase {
    name: String,
    width: usize,
    height: usize,
    input: Vec<u8>,
    expected: Vec<u8>,
}
#[derive(Deserialize)]
struct AlphaCase {
    background: [u8; 3],
    input: Vec<u8>,
    expected: Vec<u8>,
}
#[derive(Deserialize)]
struct PaddingCase {
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    input: Vec<u8>,
    expected: Vec<u8>,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/flux3_preprocess_original.json"))
        .expect("valid original CPU reference fixtures")
}

#[test]
fn reference_jpeg_matches_original_pillow_policy_or_reports_missing_feature() {
    for case in fixture().jpeg {
        let frame = ImageFrame::new(case.width, case.height, PixelFormat::Rgb8, case.input)
            .expect("valid RGB fixture");
        assert_eq!(case.expected.len(), frame.data().len());
        let result = jpeg_reference_roundtrip(&frame, 92);
        #[cfg(feature = "turbojpeg")]
        {
            let output = result.expect("enabled JPEG codec");
            assert_eq!((output.width(), output.height()), (case.width, case.height));
            assert_eq!(output.data(), case.expected, "{}", case.name);
            assert_eq!(output.timing(), frame.timing());
        }
        #[cfg(not(feature = "turbojpeg"))]
        assert!(
            matches!(result, Err(MediaError::TurboJpegUnavailable)),
            "{}",
            case.name
        );
    }
}

#[test]
fn jpeg_rejects_invalid_quality_and_non_rgb_before_codec_access() {
    let rgb = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).expect("RGB");
    for quality in [0, 101, 255] {
        assert!(matches!(
            jpeg_reference_roundtrip(&rgb, quality),
            Err(MediaError::InvalidJpegReference(_))
        ));
    }
    let rgba = ImageFrame::new(1, 1, PixelFormat::Rgba8, vec![1, 2, 3, 4]).expect("RGBA");
    assert!(matches!(
        jpeg_reference_roundtrip(&rgba, 92),
        Err(MediaError::InvalidJpegReference(_))
    ));
}

#[test]
fn rounded_alpha_matches_pillow_for_every_alpha_and_preserves_default_policy() {
    for case in fixture().alpha {
        let rounded = composite_alpha_to_rgb_with_rounding(
            &case.input,
            PixelFormat::Rgba8,
            case.background,
            true,
        )
        .expect("valid rounded alpha");
        assert_eq!(rounded, case.expected);
        let legacy = image_processors::transforms::composite_alpha_to_rgb(
            &case.input,
            PixelFormat::Rgba8,
            case.background,
        )
        .expect("valid legacy alpha");
        let explicit = composite_alpha_to_rgb_with_rounding(
            &case.input,
            PixelFormat::Rgba8,
            case.background,
            false,
        )
        .expect("valid explicit default");
        assert_eq!(legacy, explicit);
    }
}

#[test]
fn symmetric_canvas_matches_numpy_including_singleton_and_repeated_edges() {
    for case in fixture().padding {
        let frame = ImageFrame::new(case.width, case.height, PixelFormat::Rgb8, case.input)
            .expect("valid padding fixture");
        let output = pad_frame_with_canvas_fill(
            &frame,
            Padding {
                left: case.left,
                top: case.top,
                right: case.right,
                bottom: case.bottom,
            },
            CanvasFill::SymmetricImage,
        )
        .expect("valid symmetric padding");
        assert_eq!(output.data(), case.expected);
    }
}
