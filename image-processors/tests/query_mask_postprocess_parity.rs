use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use image_processors::{
    ImageSize, Layout, ProcessorOutput, ProcessorTensorName, RecipePostprocessContext,
    RecipePostprocessError, RecipePostprocessOutput, RecipeSegmentationTask,
    TaskVisionImageProcessor, TaskVisionImageProcessorConfig, TaskVisionProcessorPreset,
    TaskVisionResize, TaskVisionSegmentationRequest, Tensor, TensorData,
};
use serde::Deserialize;

#[path = "common/fixture_contract.rs"]
mod fixture_contract;

use fixture_contract::AcceptanceContract;

const FIXTURE: &str = include_str!("fixtures/transformers/query_mask_postprocess.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-query-mask-postprocess.v1";
const AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const GENERATOR: &str = "scripts/parity/transformers_query_mask_postprocess.py";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    upstream: Upstream,
    generator: String,
    acceptance: AcceptanceContract,
    families: BTreeMap<String, serde_json::Value>,
    scenarios: BTreeMap<String, Scenario>,
    cases: Vec<ParityCase>,
    unsupported: Vec<UnsupportedCase>,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    library: String,
    version: String,
    source: String,
    commit: String,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    class_logits_shape: Vec<usize>,
    class_logits: Vec<f32>,
    mask_logits_shape: Vec<usize>,
    mask_logits: Vec<f32>,
    target_size: [usize; 2],
    #[serde(default)]
    label_ids_to_fuse: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    family: String,
    task: String,
    scenario: String,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    size: [usize; 2],
    #[serde(default)]
    num_labels: Option<usize>,
    #[serde(default)]
    class_ids: Vec<i64>,
    #[serde(default)]
    scores: Vec<f32>,
    #[serde(default)]
    segmentation: Vec<i64>,
    #[serde(default)]
    segments: Vec<ExpectedSegment>,
}

#[derive(Debug, Deserialize)]
struct ExpectedSegment {
    id: i64,
    label_id: i64,
    was_fused: bool,
    score: f32,
}

#[derive(Debug, Deserialize)]
struct UnsupportedCase {
    family: String,
    task: String,
    reason: String,
    upstream_without_metadata: String,
}

#[test]
fn query_mask_fixture_is_pinned_to_clean_audited_source_metadata() {
    let fixture = fixture();
    fixture.acceptance.validate();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
    assert_eq!(fixture.upstream.library, "transformers");
    assert!(!fixture.upstream.version.is_empty());
    assert_eq!(fixture.upstream.commit, AUDIT_COMMIT);
    assert!(fixture.upstream.source.ends_with(AUDIT_COMMIT));
    assert_eq!(fixture.generator, GENERATOR);
    assert_eq!(fixture.families.len(), 4);
    let actual_cases = fixture
        .cases
        .iter()
        .map(|case| {
            (
                case.family.as_str(),
                case.task.as_str(),
                case.scenario.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected_cases = [
        ("mask_former", "semantic", "primary"),
        ("mask_former", "instance", "primary"),
        ("mask_former", "instance", "no_mask"),
        ("mask_former", "panoptic", "fusion"),
        ("mask_former", "panoptic", "no_mask"),
        ("mask2_former", "semantic", "primary"),
        ("mask2_former", "instance", "primary"),
        ("mask2_former", "instance", "no_mask"),
        ("mask2_former", "panoptic", "fusion"),
        ("mask2_former", "panoptic", "no_mask"),
        ("one_former", "semantic", "primary"),
        ("one_former", "semantic", "oneformer_downsample"),
        ("one_former", "panoptic", "fusion"),
        ("one_former", "panoptic", "no_mask"),
        ("one_former", "panoptic", "oneformer_downsample"),
        ("eomt", "semantic", "primary"),
        ("eomt", "instance", "primary"),
        ("eomt", "instance", "no_mask"),
        ("eomt", "panoptic", "fusion"),
        ("eomt", "panoptic", "no_mask"),
        ("eomt", "panoptic", "assigned_overlap"),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    assert_eq!(actual_cases, expected_cases);
    assert_eq!(fixture.cases.len(), expected_cases.len());
    assert!(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate should be inside the workspace")
        .join(&fixture.generator)
        .is_file());
}

#[test]
fn query_mask_family_outputs_match_transformers_full_payloads() {
    let fixture = fixture();
    let tolerance = fixture.acceptance.comparison.full_payload;

    for case in &fixture.cases {
        let scenario = fixture
            .scenarios
            .get(&case.scenario)
            .unwrap_or_else(|| panic!("missing scenario {}", case.scenario));
        let processor = processor(&case.family);
        let outputs = processor
            .post_process_segmentation_outputs(
                request(case, scenario),
                &model_outputs(scenario),
                &ProcessorOutput::new(),
                RecipePostprocessContext::new()
                    .with_target_sizes(&[image_size(scenario.target_size)]),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{} {} scenario {} failed: {error}",
                    case.family, case.task, case.scenario
                )
            });

        match (case.task.as_str(), outputs.as_slice()) {
            ("semantic", [RecipePostprocessOutput::SemanticSegmentation { predictions, .. }]) => {
                let [actual] = predictions.as_slice() else {
                    panic!("{} semantic output was not one image", case.family);
                };
                assert_eq!(actual.size, image_size(case.expected.size));
                assert_eq!(Some(actual.num_labels), case.expected.num_labels);
                assert_eq!(actual.class_ids, case.expected.class_ids);
                assert_float_payload(&actual.scores, &case.expected.scores, tolerance, case);
            }
            ("instance", [RecipePostprocessOutput::InstanceSegmentation { predictions, .. }])
            | ("panoptic", [RecipePostprocessOutput::PanopticSegmentation { predictions, .. }]) => {
                let [actual] = predictions.as_slice() else {
                    panic!("{} {} output was not one image", case.family, case.task);
                };
                assert_eq!(actual.size, image_size(case.expected.size));
                assert_eq!(actual.segmentation, case.expected.segmentation);
                assert_eq!(actual.segments.len(), case.expected.segments.len());
                for (index, (actual, expected)) in actual
                    .segments
                    .iter()
                    .zip(&case.expected.segments)
                    .enumerate()
                {
                    assert_eq!(actual.id, expected.id, "segment {index} in {case:?}");
                    assert_eq!(
                        actual.label_id, expected.label_id,
                        "segment {index} in {case:?}"
                    );
                    assert_eq!(
                        actual.was_fused, expected.was_fused,
                        "segment {index} in {case:?}"
                    );
                    assert!(
                        tolerance.matches(actual.score.into(), expected.score.into()),
                        "segment {index} score in {case:?}: actual={}, expected={}",
                        actual.score,
                        expected.score
                    );
                }
            }
            (_, unexpected) => panic!("unexpected output for {case:?}: {unexpected:?}"),
        }
    }
}

#[test]
fn oneformer_instance_requires_external_dataset_metadata() {
    let fixture = fixture();
    let [unsupported] = fixture.unsupported.as_slice() else {
        panic!("fixture should record one unsupported external-metadata boundary");
    };
    assert_eq!(unsupported.family, "one_former");
    assert_eq!(unsupported.task, "instance");
    assert!(unsupported.reason.contains("class_info_file"));
    assert!(unsupported
        .upstream_without_metadata
        .starts_with("TypeError:"));

    let scenario = fixture
        .scenarios
        .get("primary")
        .expect("primary scenario should exist");
    let error = processor("one_former")
        .post_process_segmentation_outputs(
            RecipeSegmentationTask::Instance,
            &model_outputs(scenario),
            &ProcessorOutput::new(),
            RecipePostprocessContext::new().with_target_sizes(&[image_size(scenario.target_size)]),
        )
        .expect_err("OneFormer instance processing must not invent dataset metadata");
    assert!(matches!(
        error,
        RecipePostprocessError::ExternalMetadataRequired {
            strategy: "one_former",
            task: "instance_segmentation",
            ..
        }
    ));
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("query-mask fixture should deserialize")
}

fn processor(family: &str) -> TaskVisionImageProcessor {
    let preset = match family {
        "mask_former" => TaskVisionProcessorPreset::MaskFormer,
        "mask2_former" => TaskVisionProcessorPreset::Mask2Former,
        "one_former" => TaskVisionProcessorPreset::OneFormer,
        "eomt" => TaskVisionProcessorPreset::Eomt,
        other => panic!("unknown query-mask family {other}"),
    };
    let mut config = TaskVisionImageProcessorConfig::for_preset(preset);
    if preset == TaskVisionProcessorPreset::Eomt {
        config.resize = TaskVisionResize::ShortestEdge {
            shortest_edge: 4,
            longest_edge: Some(6),
            multiple: None,
        };
    }
    TaskVisionImageProcessor::new(config)
        .unwrap_or_else(|error| panic!("{family} processor should build: {error}"))
}

fn request(case: &ParityCase, scenario: &Scenario) -> TaskVisionSegmentationRequest {
    match case.task.as_str() {
        "semantic" => TaskVisionSegmentationRequest::new(RecipeSegmentationTask::Semantic),
        "instance" => TaskVisionSegmentationRequest::new(RecipeSegmentationTask::Instance),
        "panoptic" => {
            TaskVisionSegmentationRequest::panoptic(scenario.label_ids_to_fuse.iter().copied())
        }
        other => panic!("unknown segmentation task {other}"),
    }
}

fn model_outputs(scenario: &Scenario) -> ProcessorOutput {
    let mut output = ProcessorOutput::new();
    output.insert_tensor(
        ProcessorTensorName::other("class_queries_logits"),
        Tensor::new(
            TensorData::F32(scenario.class_logits.clone()),
            scenario.class_logits_shape.clone(),
            Layout::CHW,
        )
        .expect("class logits should form a tensor"),
    );
    output.insert_tensor(
        ProcessorTensorName::other("masks_queries_logits"),
        Tensor::new(
            TensorData::F32(scenario.mask_logits.clone()),
            scenario.mask_logits_shape.clone(),
            Layout::NCHW,
        )
        .expect("mask logits should form a tensor"),
    );
    output
}

fn image_size([height, width]: [usize; 2]) -> ImageSize {
    ImageSize::new(height, width).expect("fixture image size should be valid")
}

fn assert_float_payload(
    actual: &[f32],
    expected: &[f32],
    tolerance: fixture_contract::FloatTolerance,
    case: &ParityCase,
) {
    assert_eq!(actual.len(), expected.len(), "score length in {case:?}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            tolerance.matches(actual.into(), expected.into()),
            "score {index} in {case:?}: actual={actual}, expected={expected}"
        );
    }
}
