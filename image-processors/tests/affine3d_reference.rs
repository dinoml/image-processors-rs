use image_processors::transforms::{
    estimate_affine_3d, invert_transform_3d, non_max_suppression_inclusive,
    non_max_suppression_minimum, project_points_3d, transform_points_3d, DetectionBoundingBox,
    ImageSize,
};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/affine3d_cpu.json")).unwrap()
}
fn triples(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| std::array::from_fn(|i| p[i].as_f64().unwrap()))
        .collect()
}
fn matrix(v: &Value) -> [[f64; 4]; 4] {
    std::array::from_fn(|r| std::array::from_fn(|c| v[r][c].as_f64().unwrap()))
}
fn close(actual: f64, expected: f64, absolute: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= absolute + relative * expected.abs(),
        "actual={actual}, reference={expected}"
    );
}
#[test]
fn affine_fit_matches_original_skimage_normalized_tls() {
    for case in fixture()["fits"].as_array().unwrap() {
        let source = triples(&case["source"]);
        let target = triples(&case["target"]);
        let actual = estimate_affine_3d(&source, &target).unwrap();
        let expected = matrix(&case["expected"]);
        for (row, e) in actual.iter().zip(expected) {
            for (&a, b) in row.iter().zip(e) {
                close(
                    a,
                    b,
                    case["absoluteTolerance"].as_f64().unwrap(),
                    case["relativeTolerance"].as_f64().unwrap(),
                );
            }
        }
    }
}
#[test]
fn homogeneous_transform_inverse_and_original_camera_projection_match_numpy() {
    let f = fixture();
    let c = &f["geometry"];
    let points = triples(&c["points"]);
    let m = matrix(&c["matrix"]);
    let inverse = invert_transform_3d(m).unwrap();
    let expected_inverse = matrix(&c["inverse"]);
    for (row, e) in inverse.iter().zip(expected_inverse) {
        for (&a, b) in row.iter().zip(e) {
            close(a, b, 1e-12, 1e-12);
        }
    }
    let actual = transform_points_3d(&points, m).unwrap();
    let expected = triples(&c["transformed"]);
    for (p, e) in actual.iter().zip(expected) {
        for (&a, b) in p.iter().zip(e) {
            close(a, b, 1e-12, 1e-12);
        }
    }
    let restored = transform_points_3d(&actual, inverse).unwrap();
    for (p, e) in restored.iter().zip(points) {
        for (&a, b) in p.iter().zip(e) {
            close(a, b, 1e-12, 1e-12);
        }
    }
    for c in f["projection"].as_array().unwrap() {
        let size = ImageSize {
            height: c["size"][0].as_u64().unwrap() as usize,
            width: c["size"][1].as_u64().unwrap() as usize,
        };
        let actual = project_points_3d(&triples(&c["points"]), matrix(&c["matrix"]), size).unwrap();
        for (p, e) in actual.iter().zip(c["expected"].as_array().unwrap()) {
            close(p[0], e[0].as_f64().unwrap(), 1e-8, 1e-11);
            close(p[1], e[1].as_f64().unwrap(), 1e-8, 1e-11);
        }
    }
}
#[test]
fn minimum_area_and_existing_union_suppression_match_pinned_facenet() {
    for c in fixture()["nms"].as_array().unwrap() {
        let boxes: Vec<_> = c["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                DetectionBoundingBox::new(
                    b[0].as_f64().unwrap() as f32,
                    b[1].as_f64().unwrap() as f32,
                    b[2].as_f64().unwrap() as f32,
                    b[3].as_f64().unwrap() as f32,
                )
                .unwrap()
            })
            .collect();
        let scores: Vec<_> = c["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected = |name: &str| -> Vec<usize> {
            c[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let threshold = c["threshold"].as_f64().unwrap() as f32;
        assert_eq!(
            non_max_suppression_minimum(&boxes, &scores, threshold).unwrap(),
            expected("expectedMinimum")
        );
        assert_eq!(
            non_max_suppression_inclusive(&boxes, &scores, threshold).unwrap(),
            expected("expectedUnion")
        );
    }
}
#[test]
fn invalid_and_degenerate_geometry_is_rejected() {
    assert!(estimate_affine_3d(&[[0.; 3]; 3], &[[0.; 3]; 3]).is_err());
    assert!(estimate_affine_3d(&[[0.; 3]; 5], &[[1.; 3]; 5]).is_err());
    assert!(invert_transform_3d([[0.; 4]; 4]).is_err());
    let identity = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    assert!(transform_points_3d(&[[f64::NAN, 0., 0.]], identity).is_err());
    assert!(project_points_3d(
        &[[0.; 3]],
        identity,
        ImageSize {
            width: 0,
            height: 1
        }
    )
    .is_err());
    assert!(non_max_suppression_minimum(&[], &[], f32::NAN).is_err());
}
