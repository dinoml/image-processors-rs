use image_processors::{media::PixelFormat, text::CaptionRenderer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    width: usize,
    height: usize,
    png: String,
}

#[test]
fn original_basic_captions_match_exact_reference_pixels() {
    let renderer = CaptionRenderer::from_bytes(
        include_bytes!("fixtures/native_caption/pillow-default-aileron.ttf").to_vec(),
        Vec::new(),
    )
    .expect("the original default caption font must parse");
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/native_caption/reference.json"))
            .expect("the original caption reference manifest must parse");
    for case in reference.cases {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native_caption")
            .join(case.png);
        let expected = image::open(path)
            .expect("the lossless original RGB fixture must decode")
            .to_rgb8();
        let actual = renderer
            .render(&case.text)
            .expect("the original caption must render natively");
        assert_eq!((actual.width(), actual.height()), (case.width, case.height));
        assert_eq!(actual.pixel_format(), PixelFormat::Rgb8);
        assert_eq!(actual.data(), expected.as_raw(), "caption {:?}", case.text);
    }
}

#[test]
fn malformed_default_or_fallback_font_returns_an_error() {
    assert!(CaptionRenderer::from_bytes(Vec::new(), Vec::new()).is_err());
    assert!(CaptionRenderer::from_bytes(vec![0; 64], Vec::new()).is_err());
    let valid = include_bytes!("fixtures/native_caption/pillow-default-aileron.ttf");
    assert!(CaptionRenderer::from_bytes(valid.to_vec(), vec![vec![0; 64]]).is_err());
}
