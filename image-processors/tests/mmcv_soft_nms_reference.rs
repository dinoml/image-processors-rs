use image_processors::tensor::DType;
use image_processors::transforms::{
    soft_non_max_suppression, soft_non_max_suppression_by_class, DetectionBoundingBox,
    ObjectDetectionPrediction, SoftNmsMethod, SoftNmsOptions,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    boxes: Vec<[f32; 4]>,
    scores: Vec<f32>,
    classes: Vec<i64>,
    method: u8,
    offset: bool,
    threshold: f32,
    sigma: f32,
    cutoff: f32,
    coordinate_offsets: bool,
    expected: Vec<(usize, u32)>,
}

#[test]
fn mmcv_soft_nms_matches_original_cpu_selection_decay_and_class_offsets() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/mmcv_soft_nms_original.json"))
            .expect("original MMCV fixtures");
    assert_eq!(fixture.cases.len(), 78);
    for case in fixture.cases {
        let boxes = case
            .boxes
            .iter()
            .map(|p| DetectionBoundingBox::new(p[0], p[1], p[2], p[3]).expect("valid fixture box"))
            .collect::<Vec<_>>();
        let method = match case.method {
            0 => SoftNmsMethod::Hard,
            1 => SoftNmsMethod::Linear,
            2 => SoftNmsMethod::Gaussian,
            _ => panic!("invalid fixture method"),
        };
        let options = SoftNmsOptions {
            iou_threshold: case.threshold,
            sigma: case.sigma,
            min_score: case.cutoff,
            method,
            inclusive_pixels: case.offset,
        };
        let result = if case.kind == "single" {
            soft_non_max_suppression(&boxes, &case.scores, options)
        } else {
            let predictions = boxes
                .iter()
                .zip(&case.scores)
                .zip(&case.classes)
                .map(|((bbox, score), class)| ObjectDetectionPrediction {
                    bbox: *bbox,
                    score: *score,
                    class_label: *class,
                })
                .collect::<Vec<_>>();
            soft_non_max_suppression_by_class(
                &predictions,
                options,
                DType::F32,
                case.coordinate_offsets,
            )
        }
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(
            result.iter().map(|x| x.0).collect::<Vec<_>>(),
            case.expected.iter().map(|x| x.0).collect::<Vec<_>>(),
            "{} indices",
            case.name
        );
        for ((_, actual), (_, bits)) in result.iter().zip(&case.expected) {
            let expected = f32::from_bits(*bits);
            assert!(
                (expected.is_nan() && actual.is_nan()) || (*actual - expected).abs() <= 1.0e-6,
                "{} score: {actual} vs {expected}",
                case.name
            );
        }
    }
}
