use image_processors::transforms::{
    generate_layered_crop_boxes_with_policy, CropGenerationPolicy, ImageSize,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    width: usize,
    height: usize,
    overlap: f64,
    expected: Vec<[usize; 4]>,
}

#[test]
fn crop_grid_matches_unmodified_original_tiles() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/interfaze_crop_original.json"))
            .expect("valid original Interfaze tile fixtures");
    for case in fixture.cases {
        let boxes = generate_layered_crop_boxes_with_policy(
            ImageSize::new(case.height, case.width).expect("positive original image dimensions"),
            CropGenerationPolicy::AxisRelativeTiesToEven {
                crop_layers: 1,
                overlap_ratio: case.overlap,
            },
        )
        .expect("valid positive original image dimensions and overlap");
        let actual: Vec<_> = boxes
            .into_iter()
            .filter(|crop| crop.layer == 1)
            .map(|crop| [crop.x_min, crop.y_min, crop.x_max, crop.y_max])
            .collect();
        assert_eq!(
            actual, case.expected,
            "image {}x{}, overlap {}",
            case.width, case.height, case.overlap
        );
    }
}
