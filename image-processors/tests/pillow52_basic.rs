#![cfg(feature = "freetype")]
use image_processors::text::TrueTypeFont;
use serde_json::Value;
fn font() -> Option<TrueTypeFont> {
    let Some(path) = std::env::var_os("DINOML_ARIAL_REFERENCE") else {
        assert!(
            std::env::var_os("DINOML_REQUIRE_FONT_REFERENCE").is_none(),
            "required pinned Arial reference not configured"
        );
        eprintln!("font reference skipped: set DINOML_ARIAL_REFERENCE to pinned Arial.ttf");
        return None;
    };
    let data = std::fs::read(path).unwrap();
    assert_eq!(data.len(), 1_036_584);
    Some(TrueTypeFont::from_bytes(data).unwrap())
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/pillow52_basic.json")).unwrap()
}
#[test]
fn original_pillow52_basic_metrics_include_offsets_and_empty_glyphs() {
    let Some(font) = font() else { return };
    let data = fixture();
    for c in data["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let m = font
            .measure_pillow52_basic(text, c["pixel_size"].as_u64().unwrap() as u32)
            .unwrap();
        assert_eq!(serde_json::json!(m.size), c["size"], "{text:?}");
        assert_eq!(serde_json::json!(m.mask_size), c["mask_size"], "{text:?}");
        assert_eq!(serde_json::json!(m.offset), c["offset"], "{text:?}");
    }
}
#[test]
fn pillow_basic_cpu_rasters_preserve_clipping_and_rgb_blending() {
    let Some(font) = font() else { return };
    let data = fixture();
    let reference = include_bytes!("fixtures/pillow_basic_smoke.rgb");
    let (width, height) = (128, 64);
    let bytes = width * height * 3;
    for (index, c) in data["cases"].as_array().unwrap().iter().enumerate() {
        let color: Vec<u8> = c["background"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect();
        let mut pixels: Vec<u8> = (0..width * height)
            .flat_map(|_| color.iter().copied())
            .collect();
        let origin = c["origin"].as_array().unwrap();
        font.draw_rgb_pillow52_basic(
            &mut pixels,
            width,
            height,
            c["text"].as_str().unwrap(),
            [
                origin[0].as_i64().unwrap() as i32,
                origin[1].as_i64().unwrap() as i32,
            ],
            c["pixel_size"].as_u64().unwrap() as u32,
        )
        .unwrap();
        assert!(
            pixels == reference[index * bytes..(index + 1) * bytes],
            "raster case{index}"
        );
    }
}
#[test]
fn pillow_basic_rejects_zero_size_and_invalid_target_storage() {
    let Some(font) = font() else { return };
    assert!(font.measure_pillow52_basic("A", 0).is_err());
    assert!(font
        .draw_rgb_pillow52_basic(&mut [0; 2], 1, 1, "A", [0, 0], 24)
        .is_err());
}
