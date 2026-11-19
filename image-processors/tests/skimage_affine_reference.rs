use image_processors::transforms::{
    warp_affine_skimage_f64, warp_affine_skimage_inverse_f64, warp_affine_skimage_inverse_u8,
    warp_affine_skimage_u8, ImageSize,
};
use serde_json::Value;
fn matrix(v: &Value) -> [[f64; 3]; 2] {
    std::array::from_fn(|r| std::array::from_fn(|c| v[r][c].as_f64().unwrap()))
}
fn size(v: &Value) -> ImageSize {
    ImageSize {
        width: v[1].as_u64().unwrap() as usize,
        height: v[0].as_u64().unwrap() as usize,
    }
}
#[test]
fn affine_rgb_matches_original_skimage_cpu_float_and_byte_paths() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/skimage_affine.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let input: Vec<u8> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let pixels: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
        let expected: Vec<f64> = case["sampled_float64"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let bytes: Vec<u8> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let source = size(&case["source_size"]);
        let target = size(&case["target_size"]);
        let forward = matrix(&case["forward_matrix"]);
        let inverse = matrix(&case["inverse_matrix"]);
        for actual in [
            warp_affine_skimage_f64(&pixels, source, target, 3, forward).unwrap(),
            warp_affine_skimage_inverse_f64(&pixels, source, target, 3, inverse).unwrap(),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() < 1e-9,
                    "{} index{i}: actual{a}, expected{e}",
                    case["name"]
                );
            }
        }
        assert_eq!(
            warp_affine_skimage_inverse_u8(&input, source, target, 3, inverse).unwrap(),
            bytes,
            "inverse {}",
            case["name"]
        );
        assert_eq!(
            warp_affine_skimage_u8(&input, source, target, 3, forward).unwrap(),
            bytes,
            "forward {}",
            case["name"]
        );
    }
}
#[test]
fn invalid_affine_inputs_return_errors() {
    let size = ImageSize {
        width: 1,
        height: 1,
    };
    assert!(warp_affine_skimage_u8(&[0, 1, 2], size, size, 3, [[0.; 3]; 2]).is_err());
    assert!(warp_affine_skimage_u8(&[], size, size, 3, [[1., 0., 0.], [0., 1., 0.]]).is_err());
    assert!(warp_affine_skimage_inverse_f64(
        &[f64::NAN; 3],
        size,
        size,
        3,
        [[1., 0., 0.], [0., 1., 0.]]
    )
    .is_err());
}

#[test]
fn affine_estimation_matches_opencv_ransac_inliers_and_refinement() {
    use image_processors::transforms::estimate_affine_2d_ransac;
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/affine_ransac_opencv.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let points = |v: &Value| -> Vec<[f32; 2]> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|p| [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32])
                .collect()
        };
        let source = points(&case["source"]);
        let target = points(&case["target"]);
        let (actual, inliers) = estimate_affine_2d_ransac(
            &source,
            &target,
            case["threshold"].as_f64().unwrap(),
            case["confidence"].as_f64().unwrap(),
            case["max_iterations"].as_u64().unwrap() as usize,
            case["refine_iterations"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let expected = matrix(&case["expected_matrix"]);
        let expected_inliers: Vec<bool> = case["expected_inliers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        assert_eq!(inliers, expected_inliers, "{}", case["name"]);
        for (r, row) in actual.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                assert!(
                    (value - expected[r][c]).abs() < 3e-6,
                    "{} coefficient{r},{c}: actual{value}, expected{}",
                    case["name"],
                    expected[r][c]
                );
            }
        }
    }
    let line = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]];
    assert!(estimate_affine_2d_ransac(&line, &line, 3., 0.99, 2000, 10).is_err());
    assert!(estimate_affine_2d_ransac(&line[..2], &line[..2], 3., 0.99, 2000, 10).is_err());
    assert!(estimate_affine_2d_ransac(&line, &line, 0., 0.99, 2000, 10).is_err());
}
