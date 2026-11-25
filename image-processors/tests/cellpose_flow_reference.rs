use image_processors::processors::cellpose::{
    reconstruct_masks, CellposeField, CellposeMaskOptions,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    fields: Vec<Field>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Field {
    name: String,
    shape: Vec<usize>,
    flows: Vec<f32>,
    cell_probability: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    field: usize,
    options: CellposeMaskOptions,
    expected: Vec<u32>,
}

#[test]
fn flow_reconstruction_matches_unmodified_original_cpu_pipeline() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/cellpose_flow_original.json"))
            .expect("valid unmodified original Cellpose CPU flow fixtures");
    let fields: Vec<_> = fixture
        .fields
        .into_iter()
        .map(|field| {
            (
                field.name,
                CellposeField {
                    shape: field.shape,
                    flows: field.flows,
                    cell_probability: field.cell_probability,
                },
            )
        })
        .collect();
    let mut failures = Vec::new();
    for case in fixture.cases {
        let (name, field) = &fields[case.field];
        let actual = match reconstruct_masks(field, &case.options) {
            Ok(mask) => mask,
            Err(error) => {
                failures.push(format!("{} ({name}): unexpected error {error}", case.name));
                continue;
            }
        };
        assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
        if let Some((index, (a, expected))) = actual
            .iter()
            .zip(&case.expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            let differing = actual
                .iter()
                .zip(&case.expected)
                .filter(|(a, b)| a != b)
                .count();
            failures.push(format!(
                "{}: index {index}, actual {a}, original {expected}, differing pixels {differing}",
                case.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
