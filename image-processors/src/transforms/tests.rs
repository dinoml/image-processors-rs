use proptest::prelude::*;

use super::*;

fn assert_center_box_close(actual: DetectionCenterBox, expected: DetectionCenterBox) {
    assert!(
        (actual.center_x - expected.center_x).abs() < 1e-6,
        "center_x mismatch: actual={} expected={}",
        actual.center_x,
        expected.center_x
    );
    assert!(
        (actual.center_y - expected.center_y).abs() < 1e-6,
        "center_y mismatch: actual={} expected={}",
        actual.center_y,
        expected.center_y
    );
    assert!(
        (actual.width - expected.width).abs() < 1e-6,
        "width mismatch: actual={} expected={}",
        actual.width,
        expected.width
    );
    assert!(
        (actual.height - expected.height).abs() < 1e-6,
        "height mismatch: actual={} expected={}",
        actual.height,
        expected.height
    );
}

fn assert_point_close(actual: ImagePoint, expected: ImagePoint) {
    assert!(
        (actual.x - expected.x).abs() < 1e-6,
        "x mismatch: actual={} expected={}",
        actual.x,
        expected.x
    );
    assert!(
        (actual.y - expected.y).abs() < 1e-6,
        "y mismatch: actual={} expected={}",
        actual.y,
        expected.y
    );
}

fn assert_detection_box_close(actual: DetectionBoundingBox, expected: DetectionBoundingBox) {
    assert!(
        (actual.x_min - expected.x_min).abs() < 1e-5,
        "x_min mismatch: actual={} expected={}",
        actual.x_min,
        expected.x_min
    );
    assert!(
        (actual.y_min - expected.y_min).abs() < 1e-5,
        "y_min mismatch: actual={} expected={}",
        actual.y_min,
        expected.y_min
    );
    assert!(
        (actual.x_max - expected.x_max).abs() < 1e-5,
        "x_max mismatch: actual={} expected={}",
        actual.x_max,
        expected.x_max
    );
    assert!(
        (actual.y_max - expected.y_max).abs() < 1e-5,
        "y_max mismatch: actual={} expected={}",
        actual.y_max,
        expected.y_max
    );
}

fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} mismatch: actual={actual} expected={expected}"
        );
    }
}

mod detection_masks;
mod frame_ops;
mod layout_geometry;
mod vlm_layouts;
