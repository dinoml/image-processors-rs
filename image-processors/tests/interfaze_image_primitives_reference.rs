// Independent OpenCV 4.10.0 / Pillow 10.4.0 CPU fixtures; versions and contracts are in the fixture.
use image_processors::{
    transforms::{
        autocontrast_frame, warp_perspective_quad_rgb_with_options, PerspectiveBorder,
        PerspectiveInterpolation, PerspectiveWarpOptions,
    },
    ImageFrame, ImageSize, PixelFormat,
};
use serde_json::{json, Value};

#[test]
fn image_primitives_match_independent_cpu_references() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/interfaze_image_primitives_reference.json"
    ))?;
    let mut failures = Vec::new();
    let mut comparisons = 0;
    let mut check = |case: &Value, actual: Vec<u8>| {
        comparisons += 1;
        let expected: Vec<u8> = serde_json::from_value(case["expected"].clone()).unwrap();
        if actual != expected {
            let differing = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| a.abs_diff(*b))
                .max();
            failures.push(json!({"case":case["name"],"actualLength":actual.len(),
                "expectedLength":expected.len(),"differingBytes":differing,"maxError":max_error}));
        }
    };
    for case in fixture["warpCases"].as_array().expect("warp cases") {
        let corners: [[f32; 2]; 4] = serde_json::from_value(case["corners"].clone())?;
        let values: Vec<u8> = serde_json::from_value(case["source"].clone())?;
        let color: [u8; 3] = serde_json::from_value(case["color"].clone())?;
        let options = PerspectiveWarpOptions {
            interpolation: PerspectiveInterpolation::Bicubic,
            border: if case["border"] == "replicate" {
                PerspectiveBorder::Replicate
            } else {
                PerspectiveBorder::Constant(color)
            },
            pixel_centers: false,
        };
        check(
            case,
            warp_perspective_quad_rgb_with_options(
                &values,
                ImageSize::new(
                    case["source_height"].as_u64().unwrap() as usize,
                    case["source_width"].as_u64().unwrap() as usize,
                )?,
                ImageSize::new(
                    case["height"].as_u64().unwrap() as usize,
                    case["width"].as_u64().unwrap() as usize,
                )?,
                corners,
                options,
            )?,
        );
    }
    for case in fixture["autocontrastCases"]
        .as_array()
        .expect("contrast cases")
    {
        let values: Vec<u8> = serde_json::from_value(case["source"].clone())?;
        let frame = ImageFrame::new(
            case["width"].as_u64().unwrap() as usize,
            case["height"].as_u64().unwrap() as usize,
            if case["format"] == "L" {
                PixelFormat::Luma8
            } else {
                PixelFormat::Rgb8
            },
            values,
        )?;
        check(
            case,
            autocontrast_frame(&frame, case["cutoff"].as_f64().unwrap())?
                .data()
                .to_vec(),
        );
    }
    println!("{}", json!({"comparisons":comparisons,"failures":failures}));
    assert!(
        failures.is_empty(),
        "CPU reference mismatches: {failures:?}"
    );
    Ok(())
}
