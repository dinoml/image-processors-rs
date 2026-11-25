use image_processors::{
    postprocess::{
        approximate_closed_polygon, bitmap_contours, closed_polygon_area, ContourRetrieval,
    },
    ImageSize,
};
use serde_json::Value;

fn points(value: &Value) -> Vec<[f32; 2]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            [
                point[0].as_f64().unwrap() as f32,
                point[1].as_f64().unwrap() as f32,
            ]
        })
        .collect()
}

#[test]
fn contours_and_closed_polygons_match_pinned_opencv_cpu_references() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/mask_contour_reference.json")).unwrap();
    let mut comparisons = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let height = usize::try_from(case["size"][0].as_u64().unwrap()).unwrap();
        let width = usize::try_from(case["size"][1].as_u64().unwrap()).unwrap();
        let mut values = vec![0.0; height * width];
        for run in case["foreground_runs"].as_array().unwrap() {
            let start = usize::try_from(run[0].as_u64().unwrap()).unwrap();
            let length = usize::try_from(run[1].as_u64().unwrap()).unwrap();
            values[start..start + length].fill(1.0);
        }
        for (key, retrieval) in [
            ("external", ContourRetrieval::External),
            ("all", ContourRetrieval::List),
        ] {
            let mut actual = bitmap_contours(
                &values,
                ImageSize::new(height, width).unwrap(),
                0.5,
                retrieval,
            )
            .unwrap();
            actual.reverse(); // The existing public owner returns discovery order.
            let expected: Vec<_> = case[key].as_array().unwrap().iter().map(points).collect();
            assert_eq!(actual, expected, "{name}: {key} retrieval and vertex order");
            comparisons += 1;
            if key == "external" {
                for (contour, area) in expected.iter().zip(case["areas"].as_array().unwrap()) {
                    assert_eq!(
                        closed_polygon_area(contour),
                        area.as_f64().unwrap(),
                        "{name}: area"
                    );
                    comparisons += 1;
                }
            }
        }
        for reference in case["approximations"].as_array().unwrap() {
            let epsilon = reference["epsilon"].as_f64().unwrap();
            let actual =
                approximate_closed_polygon(&points(&reference["points"]), epsilon).unwrap();
            assert_eq!(
                actual,
                points(&reference["expected"]),
                "{name}: approximation at {epsilon}"
            );
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 96);
}

#[test]
fn contour_and_polygon_inputs_reject_invalid_geometry_and_nonfinite_values() {
    let size = ImageSize::new(2, 2).unwrap();
    assert!(bitmap_contours(&[0.0; 3], size, 0.5, ContourRetrieval::External).is_err());
    assert!(bitmap_contours(&[0.0; 4], size, f32::NAN, ContourRetrieval::External).is_err());
    assert!(bitmap_contours(
        &[0.0, 0.0, f32::INFINITY, 1.0],
        size,
        0.5,
        ContourRetrieval::List
    )
    .is_err());
    assert!(approximate_closed_polygon(&[[f32::NAN, 0.0]], 0.0).is_err());
    assert!(approximate_closed_polygon(&[[0.0, 0.0]], -1.0).is_err());
    assert!(approximate_closed_polygon(&[[0.0, 0.0]], f64::INFINITY).is_err());
    assert!(approximate_closed_polygon(&[], 0.0).unwrap().is_empty());
    assert_eq!(closed_polygon_area(&[[0.0, 0.0], [1.0, 1.0]]), 0.0);
}
