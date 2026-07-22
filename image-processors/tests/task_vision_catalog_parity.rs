use std::collections::BTreeMap;
use std::sync::OnceLock;

use image_processors::{
    post_process_depth_map, post_process_recipe_outputs, post_process_semantic_segmentation,
    CompatibilityStatus, DetectionBoundingBox, ImageFrame, ImageProcessorError, ImageSize, Layout,
    PixelFormat, ProcessorCatalog, ProcessorMetadataName, ProcessorMetadataValue, ProcessorOutput,
    ProcessorRecipe, ProcessorRecipeInput, ProcessorRecipeOutput, ProcessorRecipePostprocess,
    ProcessorRecipeStage, ProcessorTensorName, RecipeDepthUnit, RecipeDetectionScoreMode,
    RecipeError, RecipeImageSizeSource, RecipeObjectDetectionPostprocess, RecipePostprocessContext,
    RecipePostprocessOutput, ResizeParity, TaskVisionImageProcessor,
    TaskVisionImageProcessorConfig, TaskVisionPadding, TaskVisionPoseBox,
    TaskVisionProcessorPreset, TaskVisionResize, Tensor, TensorData, TransformError,
    UpstreamLibrary,
};
use serde::Deserialize;

#[path = "common/fixture_contract.rs"]
mod fixture_contract;

use fixture_contract::{AcceptanceContract, ComparisonPolicy};

const FIXTURE: &str = include_str!("fixtures/transformers/catalog_task_vision.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-task-vision-parity.v3";
const AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const GENERATOR: &str = "scripts/parity/transformers_task_vision_parity.py";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    upstream: Upstream,
    generator: String,
    acceptance: AcceptanceContract,
    cases: Vec<ParityCase>,
    alias_cases: Vec<AliasParityCase>,
    postprocess: PostprocessFixture,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    library: String,
    version: String,
    source: String,
    commit: String,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    class_name: String,
    recipe_id: String,
    config: TaskVisionImageProcessorConfig,
    input: FixtureInput,
    outputs: BTreeMap<String, TensorFixture>,
    #[serde(default)]
    metadata: BTreeMap<String, Vec<Vec<usize>>>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FixtureInputKind {
    Single,
    Pair,
    Matte,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    kind: FixtureInputKind,
    width: usize,
    height: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
struct TensorFixture {
    shape: Vec<usize>,
    layout: String,
    dtype: String,
    data: Vec<f64>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct BackendPayloadFixture {
    outputs: BTreeMap<String, TensorFixture>,
    #[serde(default)]
    metadata: BTreeMap<String, Vec<Vec<usize>>>,
    #[serde(default)]
    auxiliary: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AliasParityCase {
    class_name: String,
    alias_class_name: String,
    config: TaskVisionImageProcessorConfig,
    input: FixtureInput,
    #[serde(default)]
    pose_boxes: Vec<[f32; 4]>,
    geometry: String,
    canonical_resize_parity: String,
    pil_resize_parity: String,
    max_float_output_difference: f64,
    backend_byte_identical: bool,
    canonical: BackendPayloadFixture,
    pil: BackendPayloadFixture,
}

#[derive(Debug, Deserialize)]
struct PostprocessFixture {
    detection: DetectionPostprocessFixture,
    depth: DepthPostprocessFixture,
    segmentation: SegmentationPostprocessFixture,
    sam2_masks: SamMaskPostprocessFixture,
    matching: MatchingPostprocessFixture,
}

#[derive(Debug, Deserialize)]
struct DetectionPostprocessFixture {
    target_size: [usize; 2],
    logits: Vec<f32>,
    boxes: Vec<f32>,
    softmax_background: DetectionExpected,
    sigmoid_best_per_query: DetectionExpected,
    sigmoid_top_k_100: DetectionExpected,
    sigmoid_top_queries: DetectionExpected,
}

#[derive(Debug, Deserialize)]
struct DetectionExpected {
    scores: Vec<f32>,
    labels: Vec<i64>,
    output_boxes: Vec<[f32; 4]>,
}

#[derive(Debug, Deserialize)]
struct DepthPostprocessFixture {
    target_size: [usize; 2],
    input_shape: Vec<usize>,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SegmentationPostprocessFixture {
    target_size: [usize; 2],
    class_logits_shape: Vec<usize>,
    class_logits: Vec<f32>,
    mask_logits_shape: Vec<usize>,
    mask_logits: Vec<f32>,
    class_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct SamMaskPostprocessFixture {
    input_shape: Vec<usize>,
    input: Vec<f32>,
    target_size: [usize; 2],
    output_shape: Vec<usize>,
    output: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct MatchingPostprocessFixture {
    target_sizes: [[usize; 2]; 2],
    normalized_keypoints: Vec<Vec<[f32; 2]>>,
    match_indices: Vec<i64>,
    input_scores: Vec<f32>,
    keypoints0: Vec<[f32; 2]>,
    keypoints1: Vec<[f32; 2]>,
    scores: Vec<f32>,
}

#[test]
fn task_vision_fixture_is_pinned_to_the_audited_transformers_source() {
    let fixture = load_fixture();

    fixture.acceptance.validate();

    assert_eq!(
        (
            fixture.schema.as_str(),
            fixture.upstream.commit.as_str(),
            fixture.generator.as_str(),
        ),
        (FIXTURE_SCHEMA, AUDIT_COMMIT, GENERATOR),
        "fixture source: {}",
        fixture.upstream.source
    );
    assert_eq!(fixture.upstream.library, "transformers");
    assert!(!fixture.upstream.version.is_empty());
    assert_eq!(
        fixture.acceptance.deterministic_input.locations,
        ["cases[].input", "alias_cases[].input", "postprocess"]
    );
    assert_eq!(
        fixture.acceptance.processor_config.locations,
        ["cases[].config", "alias_cases[].config"]
    );
}

#[test]
fn every_task_vision_preset_has_full_value_fixture_parity() {
    let fixture = load_fixture();
    assert_eq!(fixture.cases.len(), TaskVisionProcessorPreset::ALL.len());

    for case in fixture.cases {
        let preset = TaskVisionProcessorPreset::from_class_name(&case.class_name)
            .unwrap_or_else(|| panic!("fixture uses unknown class {}", case.class_name));
        assert_eq!(case.config.preset, preset, "{}", case.class_name);
        assert_eq!(case.recipe_id, preset.recipe_id(), "{}", case.class_name);

        let processor = TaskVisionImageProcessor::new(case.config)
            .unwrap_or_else(|error| panic!("{} config failed: {error}", case.class_name));
        let first = deterministic_rgb(case.input.width, case.input.height, 53, 29, 71, 11);
        let output = match case.input.kind {
            FixtureInputKind::Single => processor.preprocess_image_output(&first),
            FixtureInputKind::Pair => {
                let second = deterministic_rgb(case.input.width, case.input.height, 31, 47, 17, 23);
                processor.preprocess_pair_output(&first, &second)
            }
            FixtureInputKind::Matte => {
                let trimap = deterministic_trimap(case.input.width, case.input.height);
                processor.preprocess_matte_output(&first, &trimap)
            }
        }
        .unwrap_or_else(|error| panic!("{} preprocessing failed: {error}", case.class_name));

        assert_eq!(
            output.tensors().len(),
            case.outputs.len(),
            "{}",
            case.class_name
        );
        for (name, expected) in &case.outputs {
            let tensor_name = match name.as_str() {
                "pixel_values" => image_processors::ProcessorTensorName::PixelValues,
                "pixel_mask" => image_processors::ProcessorTensorName::PixelMask,
                other => image_processors::ProcessorTensorName::other(other),
            };
            let actual = output
                .tensor(&tensor_name)
                .unwrap_or_else(|| panic!("{} missing output {name}", case.class_name));
            assert_tensor_matches(actual, expected, &case.class_name, name);
        }
        assert_metadata_matches(&output, &case.metadata, &case.class_name);
    }
}

#[test]
fn every_task_vision_pil_alias_has_explicit_backend_full_payload_parity() {
    let fixture = load_fixture();
    let catalog = ProcessorCatalog::new();
    let mut catalog_aliases = Vec::new();
    for preset in TaskVisionProcessorPreset::ALL {
        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
            .unwrap_or_else(|| panic!("catalog missing {}", preset.class_name()));
        catalog_aliases.extend(
            entry
                .class_aliases()
                .iter()
                .copied()
                .filter(|alias| alias.ends_with("ImageProcessorPil")),
        );
    }
    catalog_aliases.sort_unstable();
    let mut fixture_aliases: Vec<&str> = fixture
        .alias_cases
        .iter()
        .map(|case| case.alias_class_name.as_str())
        .collect();
    fixture_aliases.sort_unstable();
    assert_eq!(fixture_aliases, catalog_aliases);

    for case in fixture.alias_cases {
        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, &case.alias_class_name)
            .unwrap_or_else(|| panic!("catalog missing {}", case.alias_class_name));
        assert_eq!(entry.class_name(), case.class_name);
        assert_eq!(case.canonical_resize_parity, "Torchvision");
        assert_eq!(case.pil_resize_parity, "Compatibility");
        assert_eq!(
            TaskVisionImageProcessorConfig::for_class_name(&case.class_name)
                .expect("canonical class config")
                .resize_parity,
            ResizeParity::Torchvision,
        );
        assert_eq!(
            TaskVisionImageProcessorConfig::for_class_name(&case.alias_class_name)
                .expect("PIL alias config")
                .resize_parity,
            ResizeParity::Compatibility,
        );
        assert_ne!(case.input.width, 6, "{} input must resize", case.class_name);
        assert_ne!(
            case.input.height, 4,
            "{} input must resize",
            case.class_name
        );
        assert!(!case.geometry.is_empty(), "{} geometry", case.class_name);
        if case.class_name == "VitPoseImageProcessor" {
            assert_eq!(
                case.pose_boxes,
                vec![[0.0, 0.0, 7.0, 5.0], [1.25, 0.75, 3.5, 2.25]]
            );
            assert_eq!(
                case.canonical.outputs["pixel_values"].shape[0], 2,
                "VitPose must preserve both full-frame and off-center float boxes"
            );
        }
        assert_eq!(
            case.canonical.metadata, case.pil.metadata,
            "{}",
            case.class_name
        );
        assert_eq!(
            case.canonical.auxiliary, case.pil.auxiliary,
            "{} auxiliary payload",
            case.class_name
        );
        assert_eq!(
            case.canonical.outputs.keys().collect::<Vec<_>>(),
            case.pil.outputs.keys().collect::<Vec<_>>(),
            "{} output keys",
            case.class_name
        );
        let observed_difference = max_backend_fixture_difference(&case.canonical, &case.pil);
        let tolerance = comparison_policy().statistics;
        assert!(
            tolerance.matches(observed_difference, case.max_float_output_difference),
            "{} recorded backend difference: observed={observed_difference}, fixture={}, allowed={}",
            case.class_name,
            case.max_float_output_difference,
            tolerance.allowed_delta(case.max_float_output_difference),
        );
        assert_eq!(
            case.canonical == case.pil,
            case.backend_byte_identical,
            "{} recorded byte identity",
            case.class_name
        );

        for (parity, expected, backend_name) in [
            (
                ResizeParity::Torchvision,
                &case.canonical,
                case.class_name.as_str(),
            ),
            (
                ResizeParity::Compatibility,
                &case.pil,
                case.alias_class_name.as_str(),
            ),
        ] {
            let mut config = case.config.clone();
            config.resize_parity = parity;
            let output = preprocess_alias_fixture_case(config, &case)
                .unwrap_or_else(|error| panic!("{backend_name} preprocessing failed: {error}"));
            assert_eq!(
                output.tensors().len(),
                expected.outputs.len(),
                "{backend_name} output count"
            );
            for (name, tensor_fixture) in &expected.outputs {
                let tensor_name = match name.as_str() {
                    "pixel_values" => ProcessorTensorName::PixelValues,
                    "pixel_mask" => ProcessorTensorName::PixelMask,
                    other => ProcessorTensorName::other(other),
                };
                let tensor = output
                    .tensor(&tensor_name)
                    .unwrap_or_else(|| panic!("{backend_name} missing {name}"));
                assert_tensor_matches(tensor, tensor_fixture, backend_name, name);
            }
            assert_metadata_matches(&output, &expected.metadata, backend_name);
        }
    }
}

#[test]
fn every_task_vision_preset_exposes_valid_task_postprocessing() {
    for preset in TaskVisionProcessorPreset::ALL {
        let config = TaskVisionImageProcessorConfig::for_preset(preset);
        let recipe = config
            .processor_recipe()
            .unwrap_or_else(|error| panic!("{} recipe failed: {error}", preset.class_name()));

        assert_eq!(recipe.id(), preset.recipe_id(), "{}", preset.class_name());
        assert!(
            !recipe.postprocess().is_empty(),
            "{} must describe task postprocessing",
            preset.class_name()
        );
    }
}

#[test]
fn task_vision_catalog_entries_resolve_fixture_backed_recipe_ids() {
    let catalog = ProcessorCatalog::new();
    for preset in TaskVisionProcessorPreset::ALL {
        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
            .unwrap_or_else(|| panic!("catalog missing {}", preset.class_name()));
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
        assert_eq!(entry.recipe_id(), Some(preset.recipe_id()));
    }
}

#[test]
fn class_resolution_rejects_fabricated_pil_aliases() {
    for class_name in [
        "Sam2ImageProcessorPil",
        "PPDocLayoutV2ImageProcessorPil",
        "DepthProImageProcessorPil",
        "VitPoseImageProcessorPilPil",
    ] {
        assert_eq!(TaskVisionProcessorPreset::from_class_name(class_name), None);
        assert_eq!(
            TaskVisionImageProcessorConfig::for_class_name(class_name),
            None
        );
    }
}

#[test]
fn fixed_resize_configs_reject_zero_dimensions() {
    for preset in [
        TaskVisionProcessorPreset::Owlv2,
        TaskVisionProcessorPreset::VitPose,
    ] {
        for size in [
            ImageSize {
                height: 0,
                width: 6,
            },
            ImageSize {
                height: 4,
                width: 0,
            },
        ] {
            let mut config = TaskVisionImageProcessorConfig::for_preset(preset);
            config.resize = TaskVisionResize::Fixed { size };
            assert!(matches!(
                TaskVisionImageProcessor::new(config),
                Err(ImageProcessorError::Transform(
                    TransformError::InvalidSize { .. }
                ))
            ));
        }
    }
}

#[test]
fn recipes_preserve_padding_order_and_custom_resize_semantics() {
    let mut padded = TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::Dpt);
    padded.resize = TaskVisionResize::Fixed {
        size: ImageSize::new(4, 6).expect("fixed size"),
    };
    padded.pre_resize_padding = TaskVisionPadding::BottomRightToMultiple { multiple: 2 };
    let padded_recipe = padded
        .processor_recipe()
        .expect("padded recipe should build");
    assert!(matches!(
        padded_recipe.stages(),
        [
            ProcessorRecipeStage::ConvertPixelFormat { .. },
            ProcessorRecipeStage::PadToMultiple { .. },
            ProcessorRecipeStage::Resize { .. },
            ..
        ]
    ));

    let owlv2 = TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::Owlv2)
        .processor_recipe()
        .expect("Owlv2 recipe should build");
    assert!(owlv2.stages().iter().any(|stage| matches!(
        stage,
        ProcessorRecipeStage::Owlv2AntialiasedResize {
            pad_to_square: true,
            do_rescale: true,
            ..
        }
    )));
    assert!(!owlv2
        .stages()
        .iter()
        .any(|stage| matches!(stage, ProcessorRecipeStage::Resize { .. })));

    let vitpose = TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::VitPose)
        .processor_recipe()
        .expect("VitPose recipe should build");
    assert!(vitpose.stages().iter().any(|stage| matches!(
        stage,
        ProcessorRecipeStage::VitPoseAffine {
            normalize_factor: 200.0,
            padding_factor: 1.25,
            ..
        }
    )));
    assert!(!vitpose
        .stages()
        .iter()
        .any(|stage| matches!(stage, ProcessorRecipeStage::Resize { .. })));

    for invalid_stage in [
        ProcessorRecipeStage::Owlv2AntialiasedResize {
            size: ImageSize {
                height: 0,
                width: 6,
            },
            pad_to_square: false,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
        },
        ProcessorRecipeStage::VitPoseAffine {
            size: ImageSize {
                height: 4,
                width: 6,
            },
            normalize_factor: 0.0,
            padding_factor: 1.25,
        },
    ] {
        let error = ProcessorRecipe::new(
            "transformers.invalid_custom_task_vision_stage",
            ProcessorRecipeInput::default(),
            vec![
                ProcessorRecipeStage::ConvertPixelFormat {
                    format: PixelFormat::Rgb8,
                },
                invalid_stage,
            ],
            ProcessorRecipeOutput::batch(Layout::NCHW),
        )
        .expect_err("invalid custom stage must fail recipe validation");
        assert!(matches!(error, RecipeError::InvalidTransformStage { .. }));
    }
}

#[test]
fn detection_score_modes_match_full_transformers_postprocess_payloads() {
    let fixture = load_fixture().postprocess.detection;
    let cases = [
        (
            RecipeDetectionScoreMode::SoftmaxWithBackground,
            None,
            0.5,
            &fixture.softmax_background,
        ),
        (
            RecipeDetectionScoreMode::SigmoidBestPerQuery,
            None,
            0.1,
            &fixture.sigmoid_best_per_query,
        ),
        (
            RecipeDetectionScoreMode::SigmoidTopK,
            Some(100),
            0.5,
            &fixture.sigmoid_top_k_100,
        ),
        (
            RecipeDetectionScoreMode::SigmoidTopK,
            None,
            0.5,
            &fixture.sigmoid_top_queries,
        ),
    ];

    for (mode, top_k, threshold, expected) in cases {
        let actual = execute_detection_descriptor(&fixture, mode, top_k, threshold);
        assert_detection_predictions(&actual, expected, mode);
    }
}

#[test]
fn legacy_detection_descriptor_json_defaults_to_softmax_background() {
    let descriptor: RecipeObjectDetectionPostprocess = serde_json::from_str(
        r#"{
            "logits_output":"logits",
            "boxes_output":"pred_boxes",
            "num_labels_with_background":null,
            "score_threshold":0.5,
            "target_size_source":{"source":"caller_provided"}
        }"#,
    )
    .expect("legacy descriptor should deserialize");

    assert_eq!(
        (descriptor.score_mode, descriptor.top_k),
        (RecipeDetectionScoreMode::SoftmaxWithBackground, None)
    );
}

#[test]
fn semantic_segmentation_postprocess_matches_full_transformers_payload() {
    let fixture = load_fixture().postprocess.segmentation;
    let mask_size = ImageSize::new(fixture.mask_logits_shape[2], fixture.mask_logits_shape[3])
        .expect("mask size should be valid");
    let target_size = fixture_size(fixture.target_size);
    let output = post_process_semantic_segmentation(
        &fixture.class_logits,
        &fixture.mask_logits,
        fixture.class_logits_shape[1],
        fixture.class_logits_shape[2],
        mask_size,
        Some(target_size),
    )
    .expect("semantic segmentation postprocess should succeed");

    assert_eq!(output.size, target_size);
    assert_eq!(output.class_ids, fixture.class_ids);
}

#[test]
fn depth_postprocess_matches_full_transformers_payload() {
    let fixture = load_fixture().postprocess.depth;
    let source_size = ImageSize::new(
        fixture.input_shape[fixture.input_shape.len() - 2],
        fixture.input_shape[fixture.input_shape.len() - 1],
    )
    .expect("depth source size should be valid");
    let output = post_process_depth_map(
        &fixture.input,
        source_size,
        RecipeDepthUnit::Relative,
        Some(fixture_size(fixture.target_size)),
    )
    .expect("depth postprocess should succeed");

    assert_float_payload(output.values(), &fixture.output, "depth values");
}

#[test]
fn sam2_direct_mask_postprocess_matches_full_transformers_payload() {
    let fixture = load_fixture().postprocess.sam2_masks;
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::Sam2,
    ))
    .expect("SAM2 preset should build");
    let mask_size = ImageSize::new(
        fixture.input_shape[fixture.input_shape.len() - 2],
        fixture.input_shape[fixture.input_shape.len() - 1],
    )
    .expect("mask size should be valid");
    let output = processor
        .post_process_direct_masks(
            &fixture.input,
            mask_size,
            fixture.input_shape[1] * fixture.input_shape[2],
            &[fixture_size(fixture.target_size)],
            0.0,
        )
        .expect("SAM2 direct mask postprocess should succeed");
    let actual: Vec<i64> = output[0][0].iter().map(|&value| i64::from(value)).collect();

    assert_eq!(fixture.output_shape, vec![1, 1, 3, 4]);
    assert_eq!(actual, fixture.output);
}

#[test]
fn matching_coordinate_postprocess_matches_full_transformers_payload() {
    let fixture = load_fixture().postprocess.matching;
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::LightGlue,
    ))
    .expect("LightGlue preset should build");
    let output = processor
        .post_process_indexed_keypoint_matching(
            &fixture.normalized_keypoints[0],
            &fixture.normalized_keypoints[1],
            &fixture.match_indices,
            &fixture.input_scores,
            [
                fixture_size(fixture.target_sizes[0]),
                fixture_size(fixture.target_sizes[1]),
            ],
            0.1,
        )
        .expect("indexed matching coordinates should restore");

    assert_coordinate_payload(output.keypoints0(), &fixture.keypoints0, "keypoints0");
    assert_coordinate_payload(output.keypoints1(), &fixture.keypoints1, "keypoints1");
    assert_float_payload(output.matching_scores(), &fixture.scores, "matching scores");
}

#[test]
fn indexed_matching_rejects_non_finite_values_and_invalid_target_sizes() {
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::LightGlue,
    ))
    .expect("LightGlue preset should build");
    let valid_points = [[0.25, 0.5]];
    let valid_sizes = [
        ImageSize::new(10, 20).expect("size should be valid"),
        ImageSize::new(8, 12).expect("size should be valid"),
    ];

    let coordinate_error = processor
        .post_process_indexed_keypoint_matching(
            &[[f32::NAN, 0.5]],
            &valid_points,
            &[0],
            &[0.8],
            valid_sizes,
            0.1,
        )
        .expect_err("non-finite coordinates must fail");
    assert!(matches!(
        coordinate_error,
        ImageProcessorError::Transform(TransformError::InvalidPoint { .. })
    ));

    for (scores, threshold) in [([f32::NAN], 0.1), ([0.8], f32::INFINITY)] {
        let error = processor
            .post_process_indexed_keypoint_matching(
                &valid_points,
                &valid_points,
                &[0],
                &scores,
                valid_sizes,
                threshold,
            )
            .expect_err("non-finite scores and thresholds must fail");
        assert!(matches!(
            error,
            ImageProcessorError::Transform(TransformError::InvalidScoreValue(_))
        ));
    }

    let invalid_size_error = processor
        .post_process_indexed_keypoint_matching(
            &valid_points,
            &valid_points,
            &[0],
            &[0.8],
            [
                ImageSize {
                    height: 0,
                    width: 20,
                },
                valid_sizes[1],
            ],
            0.1,
        )
        .expect_err("zero target dimensions must fail");
    assert!(matches!(
        invalid_size_error,
        ImageProcessorError::Transform(TransformError::InvalidSize { .. })
    ));
}

#[test]
fn vitpose_affine_preprocessing_rejects_invalid_boxes_and_contracts() {
    for invalid in [
        [f32::NAN, 0.0, 4.0, 4.0],
        [0.0, 0.0, 0.0, 4.0],
        [0.0, 0.0, 4.0, -1.0],
    ] {
        let error = TaskVisionPoseBox::new(invalid[0], invalid[1], invalid[2], invalid[3])
            .expect_err("invalid VitPose boxes must fail");
        assert!(matches!(error, TransformError::InvalidBoundingBox { .. }));
    }

    let image = deterministic_rgb(7, 5, 53, 29, 71, 11);
    let bbox = TaskVisionPoseBox::new(0.0, 0.0, 7.0, 5.0).expect("box should be valid");
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::VitPose,
    ))
    .expect("VitPose preset should build");
    assert!(matches!(
        processor
            .preprocess_pose_output(&image, &[])
            .expect_err("empty box batches must fail"),
        ImageProcessorError::EmptyBatch
    ));

    let wrong_processor = TaskVisionImageProcessor::new(
        TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::Dpt),
    )
    .expect("DPT preset should build");
    assert!(matches!(
        wrong_processor
            .preprocess_pose_output(&image, &[bbox])
            .expect_err("non-VitPose processors must reject pose boxes"),
        ImageProcessorError::UnsupportedProcessorOption { .. }
    ));

    let mut unsupported_config =
        TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::VitPose);
    unsupported_config.resize = image_processors::TaskVisionResize::None;
    let unsupported_processor = TaskVisionImageProcessor::new(unsupported_config)
        .expect("no-resize config should validate");
    assert!(matches!(
        unsupported_processor
            .preprocess_pose_output(&image, &[bbox])
            .expect_err("affine preprocessing requires fixed output geometry"),
        ImageProcessorError::UnsupportedProcessorOption { .. }
    ));
}

fn execute_detection_descriptor(
    fixture: &DetectionPostprocessFixture,
    mode: RecipeDetectionScoreMode,
    top_k: Option<usize>,
    threshold: f32,
) -> Vec<image_processors::ObjectDetectionPrediction> {
    let mut descriptor = RecipeObjectDetectionPostprocess::new(
        "logits",
        "pred_boxes",
        None,
        threshold,
        RecipeImageSizeSource::CallerProvided,
    )
    .with_score_mode(mode);
    if let Some(top_k) = top_k {
        descriptor = descriptor.with_top_k(top_k);
    }
    let recipe = ProcessorRecipe::new_postprocess_only(
        "transformers.fixture_detection_postprocess",
        ProcessorRecipeInput::default(),
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![ProcessorRecipePostprocess::ObjectDetection(descriptor)],
    )
    .expect("detection fixture recipe should be valid");
    let num_queries = fixture.boxes.len() / 4;
    let num_labels = fixture.logits.len() / num_queries;
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("logits"),
        Tensor::new(
            TensorData::F32(fixture.logits.clone()),
            vec![1, num_queries, num_labels],
            Layout::CHW,
        )
        .expect("fixture logits should be valid"),
    );
    model_outputs.insert_tensor(
        ProcessorTensorName::other("pred_boxes"),
        Tensor::new(
            TensorData::F32(fixture.boxes.clone()),
            vec![1, num_queries, 4],
            Layout::CHW,
        )
        .expect("fixture boxes should be valid"),
    );
    let target_size = [fixture_size(fixture.target_size)];
    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_size),
    )
    .expect("detection fixture descriptor should execute");
    let [RecipePostprocessOutput::ObjectDetection { predictions, .. }] = outputs.as_slice() else {
        panic!("expected object detection output, got {outputs:?}")
    };
    predictions[0].clone()
}

fn assert_detection_predictions(
    actual: &[image_processors::ObjectDetectionPrediction],
    expected: &DetectionExpected,
    mode: RecipeDetectionScoreMode,
) {
    assert_eq!(actual.len(), expected.scores.len(), "{mode:?}");
    let actual_scores: Vec<f32> = actual.iter().map(|prediction| prediction.score).collect();
    assert_float_payload(&actual_scores, &expected.scores, "detection scores");
    let actual_labels: Vec<i64> = actual
        .iter()
        .map(|prediction| prediction.class_label)
        .collect();
    assert_eq!(actual_labels, expected.labels, "{mode:?}");
    let actual_boxes: Vec<DetectionBoundingBox> =
        actual.iter().map(|prediction| prediction.bbox).collect();
    let expected_boxes: Vec<DetectionBoundingBox> = expected
        .output_boxes
        .iter()
        .map(|coords| {
            DetectionBoundingBox::new(coords[0], coords[1], coords[2], coords[3])
                .expect("fixture box should be valid")
        })
        .collect();
    assert_eq!(actual_boxes, expected_boxes, "{mode:?}");
}

fn fixture_size(size: [usize; 2]) -> ImageSize {
    ImageSize::new(size[0], size[1]).expect("fixture size should be valid")
}

fn flatten_points(points: &[[f32; 2]]) -> Vec<f32> {
    points.iter().flat_map(|point| *point).collect()
}

fn assert_coordinate_payload(
    actual: &[image_processors::CoordinatePoint],
    expected: &[[f32; 2]],
    label: &str,
) {
    let actual: Vec<f32> = actual.iter().flat_map(|point| [point.x, point.y]).collect();
    let expected: Vec<f32> = flatten_points(expected);
    assert_float_payload(&actual, &expected, label);
}

fn assert_float_payload(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let tolerance = comparison_policy().full_payload;
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| f64::from((actual - expected).abs()))
        .fold(0.0, f64::max);
    let first_difference =
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (&actual, &expected))| {
                !tolerance.matches(f64::from(actual), f64::from(expected))
            });
    assert!(
        first_difference.is_none(),
        "{label} full payload max_abs={max_abs}, first_difference={first_difference:?}"
    );
}

fn max_backend_fixture_difference(
    canonical: &BackendPayloadFixture,
    pil: &BackendPayloadFixture,
) -> f64 {
    canonical
        .outputs
        .iter()
        .filter(|(_, tensor)| tensor.dtype == "float32")
        .map(|(name, canonical_tensor)| {
            let pil_tensor = pil
                .outputs
                .get(name)
                .unwrap_or_else(|| panic!("PIL payload missing {name}"));
            assert_eq!(canonical_tensor.shape, pil_tensor.shape, "{name} shape");
            assert_eq!(canonical_tensor.layout, pil_tensor.layout, "{name} layout");
            assert_eq!(canonical_tensor.dtype, pil_tensor.dtype, "{name} dtype");
            canonical_tensor
                .data
                .iter()
                .zip(&pil_tensor.data)
                .map(|(&canonical, &pil)| (canonical - pil).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn preprocess_fixture_case(
    config: TaskVisionImageProcessorConfig,
    input: &FixtureInput,
) -> Result<ProcessorOutput, ImageProcessorError> {
    let processor = TaskVisionImageProcessor::new(config)?;
    let first = deterministic_rgb(input.width, input.height, 53, 29, 71, 11);
    match input.kind {
        FixtureInputKind::Single => processor.preprocess_image_output(&first),
        FixtureInputKind::Pair => {
            let second = deterministic_rgb(input.width, input.height, 31, 47, 17, 23);
            processor.preprocess_pair_output(&first, &second)
        }
        FixtureInputKind::Matte => {
            let trimap = deterministic_trimap(input.width, input.height);
            processor.preprocess_matte_output(&first, &trimap)
        }
    }
}

fn preprocess_alias_fixture_case(
    config: TaskVisionImageProcessorConfig,
    case: &AliasParityCase,
) -> Result<ProcessorOutput, ImageProcessorError> {
    if config.preset != TaskVisionProcessorPreset::VitPose {
        return preprocess_fixture_case(config, &case.input);
    }
    let processor = TaskVisionImageProcessor::new(config)?;
    let image = deterministic_rgb(case.input.width, case.input.height, 53, 29, 71, 11);
    let boxes = case
        .pose_boxes
        .iter()
        .map(|bbox| TaskVisionPoseBox::new(bbox[0], bbox[1], bbox[2], bbox[3]))
        .collect::<Result<Vec<_>, _>>()?;
    processor.preprocess_pose_output(&image, &boxes)
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("task-vision fixture JSON should parse")
}

fn comparison_policy() -> &'static ComparisonPolicy {
    static POLICY: OnceLock<ComparisonPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        let fixture = load_fixture();
        fixture.acceptance.validate();
        fixture.acceptance.comparison
    })
}

fn deterministic_rgb(
    width: usize,
    height: usize,
    row_factor: u32,
    column_factor: u32,
    channel_factor: u32,
    offset: u32,
) -> ImageFrame {
    let mut data = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        for column in 0..width {
            for channel in 0..3 {
                data.push(
                    ((row as u32 * row_factor
                        + column as u32 * column_factor
                        + channel as u32 * channel_factor
                        + offset)
                        % 256) as u8,
                );
            }
        }
    }
    ImageFrame::new(width, height, PixelFormat::Rgb8, data)
        .expect("deterministic RGB fixture should be valid")
}

fn deterministic_trimap(width: usize, height: usize) -> ImageFrame {
    let data = (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| ((row as u32 * 85 + column as u32 * 51) % 256) as u8)
        })
        .collect();
    ImageFrame::new(width, height, PixelFormat::Luma8, data)
        .expect("deterministic trimap should be valid")
}

fn assert_tensor_matches(
    actual: &Tensor,
    expected: &TensorFixture,
    class_name: &str,
    output_name: &str,
) {
    let expected_shape = if output_name == "pixel_mask" && expected.shape.len() == 3 {
        vec![expected.shape[0], 1, expected.shape[1], expected.shape[2]]
    } else {
        expected.shape.clone()
    };
    assert_eq!(actual.shape(), expected_shape, "{class_name} {output_name}");
    assert_eq!(
        layout_name(actual.layout()),
        expected.layout,
        "{class_name} {output_name}"
    );

    match expected.dtype.as_str() {
        "float32" => {
            let TensorData::F32(actual) = actual.data() else {
                panic!(
                    "{class_name} {output_name} expected float32, got {:?}",
                    actual.data().dtype()
                );
            };
            let tolerance = comparison_policy().full_payload;
            assert_eq!(
                actual.len(),
                expected.data.len(),
                "{class_name} {output_name}"
            );
            let max_abs = actual
                .iter()
                .zip(&expected.data)
                .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
                .fold(0.0, f64::max);
            let first_difference = actual
                .iter()
                .zip(&expected.data)
                .enumerate()
                .find(|(_, (&actual, &expected))| !tolerance.matches(f64::from(actual), expected));
            assert!(
                first_difference.is_none(),
                "{class_name} {output_name} full payload max_abs={max_abs}, first_difference={first_difference:?}"
            );
        }
        "int64" => {
            let actual: Vec<f64> = match actual.data() {
                TensorData::Bool(values) if output_name == "pixel_mask" => {
                    values.iter().map(|&value| f64::from(value)).collect()
                }
                TensorData::I64(values) => {
                    values.iter().copied().map(|value| value as f64).collect()
                }
                data => panic!(
                    "{class_name} {output_name} expected int64, got {:?}",
                    data.dtype()
                ),
            };
            assert_eq!(actual, expected.data, "{class_name} {output_name}");
        }
        dtype => panic!("{class_name} {output_name} fixture has unsupported dtype {dtype}"),
    }
}

fn assert_metadata_matches(
    output: &image_processors::ProcessorOutput,
    expected: &BTreeMap<String, Vec<Vec<usize>>>,
    class_name: &str,
) {
    for (name, expected_sizes) in expected {
        let metadata_name = match name.as_str() {
            "original_sizes" => ProcessorMetadataName::OriginalSizes,
            "reshaped_input_sizes" => ProcessorMetadataName::ReshapedInputSizes,
            other => ProcessorMetadataName::other(other),
        };
        let value = output
            .metadata_value(&metadata_name)
            .unwrap_or_else(|| panic!("{class_name} missing metadata {name}"));
        let ProcessorMetadataValue::ImageSizes(actual_sizes) = value else {
            panic!("{class_name} metadata {name} has wrong type")
        };
        let expected_sizes: Vec<ImageSize> = expected_sizes
            .iter()
            .map(|size| {
                ImageSize::new(size[0], size[1]).expect("fixture metadata size should be valid")
            })
            .collect();
        assert_eq!(actual_sizes, &expected_sizes, "{class_name} {name}");
    }
}

fn layout_name(layout: Layout) -> &'static str {
    match layout {
        Layout::NCHW => "nchw",
        Layout::NHWC => "nhwc",
        Layout::NPCHW => "npchw",
        Layout::NPHWC => "nphwc",
        Layout::NIPCHW => "nipchw",
        Layout::NIPHWC => "niphwc",
        other => panic!("unexpected task-vision fixture layout {other:?}"),
    }
}
