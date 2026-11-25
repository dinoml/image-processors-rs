use image_processors::processors::cellpose::{normalize_planes, CellposeNormalization};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    shape: Vec<usize>,
    options: CellposeNormalization,
    input: Vec<f32>,
    expected: Vec<f32>,
}

#[test]
fn normalization_matches_unmodified_original_cpu_functions() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "fixtures/cellpose_normalization_original.json"
    ))
    .expect("valid original Cellpose fixtures");
    let mut failures = Vec::new();
    for case in fixture.cases {
        let mut actual = case.input;
        if let Err(error) = normalize_planes(&mut actual, &case.shape, &case.options) {
            failures.push(format!("{}: unexpected error {error}", case.name));
            continue;
        }
        assert_eq!(actual.len(), case.expected.len());
        let mismatch = actual.iter().zip(&case.expected).enumerate().find(|(_, (a, b))| {
            !a.is_finite() || (*a - *b).abs() > 2e-6 * b.abs().max(1.0)
        });
        if let Some((index, (a, b))) = mismatch {
            failures.push(format!("{}: index {index}, actual {a}, original {b}", case.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
