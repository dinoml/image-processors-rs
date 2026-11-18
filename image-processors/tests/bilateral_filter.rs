use image_processors::transforms::{bilateral_filter_rgb_u8, ImageSize};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    width: usize,
    height: usize,
    diameter: i32,
    sigma_color: f64,
    sigma_space: f64,
    iterations: usize,
    input: Vec<u8>,
    expected: Vec<u8>,
}

#[test]
fn bilateral_rgb_matches_independent_opencv_reference_pixels() {
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/cosmos_bilateral_opencv.json"))
            .expect("valid independently generated OpenCV fixture");
    for case in reference.cases {
        let size = ImageSize::new(case.height, case.width).expect("valid fixture dimensions");
        let actual = bilateral_filter_rgb_u8(
            &case.input,
            size,
            case.diameter,
            case.sigma_color,
            case.sigma_space,
            case.iterations,
        )
        .expect("valid OpenCV bilateral fixture");
        assert_eq!(actual, case.expected, "OpenCV fixture {}", case.name);
    }
}

#[test]
fn bilateral_zero_passes_preserves_pixels_and_bad_inputs_return_errors() {
    let size = ImageSize::new(1, 2).expect("valid dimensions");
    let pixels = [0, 17, 255, 200, 100, 0];
    assert_eq!(
        bilateral_filter_rgb_u8(&pixels, size, 9, 75.0, 75.0, 0).expect("zero passes is identity"),
        pixels
    );
    assert!(bilateral_filter_rgb_u8(&pixels[..5], size, 9, 75.0, 75.0, 1).is_err());
    assert!(bilateral_filter_rgb_u8(&pixels, size, 9, f64::NAN, 75.0, 1).is_err());
    assert!(bilateral_filter_rgb_u8(&pixels, size, 9, 75.0, f64::INFINITY, 1).is_err());
}
