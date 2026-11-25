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
    #[serde(default)]
    expected_bits: Vec<u32>,
    expected_error: Option<String>,
    tolerance: f32,
}

#[test]
fn local_normalization_matches_unmodified_original_cpu_functions() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "fixtures/cellpose_local_normalization_original.json"
    ))
    .expect("valid original Cellpose smoothing and local normalization fixtures");
    let mut failures = Vec::new();
    for case in fixture.cases {
        let mut actual = case.input;
        let result = normalize_planes(&mut actual, &case.shape, &case.options);
        if let Some(original_error) = case.expected_error {
            if result.is_ok() {
                failures.push(format!(
                    "{}: expected original {original_error} rejection",
                    case.name
                ));
            }
            continue;
        }
        if let Err(error) = result {
            failures.push(format!("{}: unexpected error {error}", case.name));
            continue;
        }
        assert_eq!(actual.len(), case.expected_bits.len(), "{}", case.name);
        let mismatch =
            actual
                .iter()
                .zip(case.expected_bits)
                .enumerate()
                .find_map(|(index, (&a, bits))| {
                    let expected = f32::from_bits(bits);
                    let matches = if expected.is_nan() {
                        a.is_nan()
                    } else if expected.is_infinite() {
                        a == expected
                    } else {
                        a.is_finite()
                            && (a - expected).abs() <= case.tolerance * expected.abs().max(1.0)
                    };
                    (!matches).then_some((index, a, expected))
                });
        if let Some((index, a, expected)) = mismatch {
            failures.push(format!(
                "{}: index {index}, actual {a}, original {expected}",
                case.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
