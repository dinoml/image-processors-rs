use image_processors::processors::cellpose::stitch_masks;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: Vec<usize>,
    threshold: f64,
    input: Vec<Vec<u32>>,
    expected: Vec<Vec<u32>>,
}

#[test]
fn stitching_matches_unmodified_original_cpu_functions() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/cellpose_stitch_original.json"))
            .expect("valid original Cellpose stitch fixtures");
    let mut failures = Vec::new();
    for case in fixture.cases {
        let mut actual = case.input;
        assert_eq!(actual.len(), case.shape[0], "{}", case.name);
        if let Err(error) = stitch_masks(&mut actual, case.threshold) {
            failures.push(format!("{}: unexpected error {error}", case.name));
            continue;
        }
        if let Some((index, (a, expected))) = actual
            .iter()
            .flatten()
            .zip(case.expected.iter().flatten())
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            failures.push(format!(
                "{}: index {index}, actual {a}, original {expected}",
                case.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
