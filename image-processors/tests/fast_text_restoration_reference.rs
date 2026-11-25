use image_processors::postprocess::{
    post_process_fast_text_detection, FastTextBoxType, FastTextDetectionOptions,
};
use image_processors::transforms::ImageSize;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    image_size: [usize; 2],
    original_size: [usize; 2],
    kernel_labels: Vec<i64>,
    labels: Vec<i64>,
    scores: Vec<f32>,
    bbox_type: FastTextBoxType,
    min_score: f64,
    min_area: usize,
    expected_boxes: Vec<Vec<i32>>,
    expected_scores: Vec<f64>,
    expected_error: Option<String>,
}

#[test]
fn fast_restoration_matches_pinned_original_cpu_geometry_and_filtering() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/fast_restoration_original.json"))
            .expect("original FAST fixture must parse");
    assert_eq!(fixture.cases.len(), 66);
    for case in fixture.cases {
        let result = post_process_fast_text_detection(
            &case.kernel_labels,
            &case.labels,
            &case.scores,
            ImageSize::new(case.image_size[0], case.image_size[1]).expect("valid fixture size"),
            ImageSize::new(case.original_size[0], case.original_size[1])
                .expect("valid original size"),
            FastTextDetectionOptions {
                min_score: case.min_score,
                min_area: case.min_area,
                bbox_type: case.bbox_type,
            },
        );
        if case.expected_error.is_some() {
            assert!(
                result.is_err(),
                "{} must reject degenerate geometry",
                case.name
            );
            continue;
        }
        let actual = result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let boxes: Vec<Vec<i32>> = actual
            .iter()
            .map(|d| d.points.iter().flatten().copied().collect())
            .collect();
        assert_eq!(
            boxes, case.expected_boxes,
            "{} geometry/filtering",
            case.name
        );
        assert_eq!(
            actual.len(),
            case.expected_scores.len(),
            "{} count",
            case.name
        );
        for (detection, expected) in actual.iter().zip(&case.expected_scores) {
            assert!(
                (detection.score - expected).abs() <= 1.0e-6,
                "{} mean: {} vs {expected}",
                case.name,
                detection.score
            );
        }
    }
}
