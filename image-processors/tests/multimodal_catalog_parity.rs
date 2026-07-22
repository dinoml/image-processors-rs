use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::postprocess::{
    post_process_recipe_outputs, RecipePostprocessContext, RecipePostprocessOutput,
};
use image_processors::processors::{
    multimodal_preset, normalize_ocr_word, post_process_document_rectification,
    post_process_table_recognition, post_process_text_recognition, MultimodalArray,
    MultimodalArrayData, MultimodalCategory, MultimodalOutput, MultimodalProcessor,
    MultimodalProcessorError, MultimodalProfile, MultimodalResizeBackend, MultimodalValue,
    MULTIMODAL_CLASS_ALIASES, MULTIMODAL_PRESETS,
};
use image_processors::{
    CompatibilityStatus, ImageFrame, ImageSize, Layout, PixelFormat, ProcessorCatalog,
    ProcessorOutput, ProcessorRecipePostprocess, ProcessorTensorName, RecipePatchStage,
    RecipeTokenSequenceTask, Tensor, TensorData, TokenSequencePostprocessOutput, UpstreamLibrary,
};
use serde::Deserialize;
use serde_json::Value;

#[path = "common/fixture_contract.rs"]
mod fixture_contract;

use fixture_contract::{AcceptanceContract, ComparisonPolicy};

const FIXTURE: &str = include_str!("fixtures/transformers/catalog_multimodal.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-multimodal-parity.v2";
const AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const GENERATOR: &str = "scripts/parity/transformers_multimodal_parity.py";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    generator: String,
    upstream: Upstream,
    acceptance: AcceptanceContract,
    cases: Vec<ParityCase>,
    aliases: Vec<AliasParityCase>,
    postprocess: Value,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    library: String,
    source: String,
    version: String,
    commit: String,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    category: String,
    class_name: String,
    model_type: String,
    profile: String,
    recipe_id: String,
    source: FixtureSource,
    input: FixtureInput,
    processor_config: Value,
    outputs: BTreeMap<String, Value>,
    resize_probe: ResizeProbe,
}

#[derive(Debug, Deserialize)]
struct ResizeProbe {
    input: FixtureInput,
    outputs: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct AliasParityCase {
    alias: String,
    canonical: String,
    source: FixtureSource,
    input: FixtureInput,
    processor_config: Value,
    outputs: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct FixtureSource {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    kind: String,
    mode: String,
    width: usize,
    height: usize,
    frames: usize,
}

#[test]
fn every_multimodal_preset_matches_full_transformers_payload() {
    let fixture = fixture();

    for case in &fixture.cases {
        let frames = (0..case.input.frames)
            .map(|offset| deterministic_frame(case.input.width, case.input.height, offset))
            .collect::<Vec<_>>();
        let processor = MultimodalProcessor::for_class(&case.class_name)
            .unwrap_or_else(|error| panic!("{} should resolve: {error}", case.class_name));
        let actual = if case.input.kind == "image" {
            processor.preprocess_image(&frames[0])
        } else {
            processor.preprocess_frames(&frames)
        }
        .unwrap_or_else(|error| panic!("{} should preprocess: {error}", case.class_name));

        assert_eq!(
            actual.fields().len(),
            case.outputs.len(),
            "{} output field count",
            case.class_name
        );
        for (name, expected) in &case.outputs {
            let actual = actual
                .get(name)
                .unwrap_or_else(|| panic!("{} missing output {name}", case.class_name));
            assert_value_matches(&case.class_name, name, actual, expected);
        }
    }
}

#[test]
fn video_llama3_recipe_exposes_fixture_backed_temporal_patch_stage() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.class_name == "VideoLlama3ImageProcessor")
        .expect("fixture should contain VideoLlama3");
    let patch_size = case.processor_config["patch_size"]
        .as_u64()
        .expect("VideoLlama3 fixture should record patch_size") as usize;
    let temporal_patch_size = case.processor_config["temporal_patch_size"]
        .as_u64()
        .expect("VideoLlama3 fixture should record temporal_patch_size")
        as usize;
    let merge_size = case.processor_config["merge_size"]
        .as_u64()
        .expect("VideoLlama3 fixture should record merge_size") as usize;
    let preset = multimodal_preset(&case.class_name).expect("VideoLlama3 preset should exist");
    let recipe = preset
        .processor_recipe()
        .expect("VideoLlama3 recipe should validate");

    let patch = recipe
        .patch_stage()
        .expect("VideoLlama3 recipe should expose patch flattening");

    assert_eq!(
        patch,
        RecipePatchStage::flatten(Layout::CHW, patch_size, temporal_patch_size, merge_size,)
    );
    assert_eq!(
        patch
            .temporal_grid_thw(
                case.input.frames,
                ImageSize::new(recipe.output_edge(), recipe.output_edge())
                    .expect("fixture output edge should be valid"),
            )
            .expect("fixture temporal grid should be valid"),
        [case.input.frames, 4, 4]
    );

    let actual = preprocess_fixture_input(&case.class_name, &case.input, 0);
    assert_output_matches(&case.class_name, &actual, &case.outputs);
}

#[test]
fn every_multimodal_preset_matches_nontrivial_resize_payload() {
    let fixture = fixture();

    for case in &fixture.cases {
        let preset = multimodal_preset(&case.class_name)
            .unwrap_or_else(|| panic!("missing preset for {}", case.class_name));
        assert!(
            case.resize_probe.input.width > preset.output_edge(),
            "{} resize probe must be larger than its {}px output",
            case.class_name,
            preset.output_edge()
        );
        assert_eq!(
            case.resize_probe.input.width, case.resize_probe.input.height,
            "{}",
            case.class_name
        );
        let actual = preprocess_fixture_input(&case.class_name, &case.resize_probe.input, 2);
        assert_output_matches(&case.class_name, &actual, &case.resize_probe.outputs);
    }
}

#[test]
fn every_catalog_alias_matches_its_nontrivial_transformers_resize_payload() {
    let fixture = fixture();

    assert_eq!(fixture.aliases.len(), MULTIMODAL_CLASS_ALIASES.len());
    let expected_aliases = MULTIMODAL_CLASS_ALIASES
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    for case in &fixture.aliases {
        assert_eq!(
            expected_aliases.get(case.alias.as_str()),
            Some(&case.canonical.as_str()),
            "{}",
            case.alias
        );
        assert_eq!(case.source.sha256.len(), 64, "{}", case.alias);
        assert!(case.processor_config.is_object(), "{}", case.alias);
        let preset = multimodal_preset(&case.alias)
            .unwrap_or_else(|| panic!("missing preset for alias {}", case.alias));
        assert!(
            case.input.width > preset.output_edge(),
            "{} resize probe must be larger than its {}px output",
            case.alias,
            preset.output_edge()
        );
        let actual = preprocess_fixture_input(&case.alias, &case.input, 2);
        let processor = MultimodalProcessor::for_class(&case.alias)
            .unwrap_or_else(|error| panic!("{} should resolve: {error}", case.alias));
        assert_eq!(
            processor.requested_class_name(),
            case.alias,
            "{}",
            case.alias
        );
        assert_eq!(
            processor.resize_backend(),
            if case.alias.ends_with("Pil") {
                MultimodalResizeBackend::Pillow
            } else {
                MultimodalResizeBackend::Torchvision
            },
            "{}",
            case.alias
        );
        assert_output_matches(&case.alias, &actual, &case.outputs);
    }
}

#[test]
fn multimodal_fixture_records_exact_sources_and_compact_contracts() {
    let fixture = fixture();

    fixture.acceptance.validate();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
    assert_eq!(fixture.generator, GENERATOR);
    assert_eq!(fixture.upstream.library, "transformers");
    assert_eq!(fixture.upstream.source, "huggingface/transformers");
    assert_eq!(fixture.upstream.commit, AUDIT_COMMIT);
    assert!(!fixture.upstream.version.trim().is_empty());
    assert_eq!(fixture.cases.len(), 52);
    assert_eq!(MULTIMODAL_PRESETS.len(), 52);
    assert_eq!(
        fixture.acceptance.deterministic_input.locations,
        [
            "fixture",
            "cases[].input",
            "cases[].resize_probe.input",
            "aliases[].input",
            "postprocess",
        ]
    );
    assert_eq!(
        fixture.acceptance.processor_config.locations,
        ["cases[].processor_config", "aliases[].processor_config"]
    );

    let mut names = HashSet::with_capacity(fixture.cases.len());
    let mut category_counts = [0usize; 3];
    for case in &fixture.cases {
        assert!(
            names.insert(case.class_name.as_str()),
            "duplicate {}",
            case.class_name
        );
        assert_eq!(case.input.mode, "RGB", "{}", case.class_name);
        assert_eq!(case.input.width, case.input.height, "{}", case.class_name);
        assert!(case.processor_config.is_object(), "{}", case.class_name);
        assert_eq!(case.source.sha256.len(), 64, "{}", case.class_name);

        let preset = multimodal_preset(&case.class_name)
            .unwrap_or_else(|| panic!("missing preset for {}", case.class_name));
        assert_eq!(preset.model_type(), case.model_type, "{}", case.class_name);
        assert_eq!(preset.recipe_id(), case.recipe_id, "{}", case.class_name);
        let recipe = preset
            .processor_recipe()
            .unwrap_or_else(|error| panic!("{} recipe should validate: {error}", case.class_name));
        assert_eq!(recipe.id(), case.recipe_id, "{}", case.class_name);
        assert_eq!(recipe.preset(), *preset, "{}", case.class_name);
        assert_eq!(recipe.class_name(), case.class_name, "{}", case.class_name);
        assert_eq!(recipe.model_type(), case.model_type, "{}", case.class_name);
        assert_eq!(recipe.category(), preset.category(), "{}", case.class_name);
        assert_eq!(recipe.profile(), preset.profile(), "{}", case.class_name);
        assert_eq!(
            recipe.output_edge(),
            preset.output_edge(),
            "{}",
            case.class_name
        );
        assert_eq!(
            recipe.minimum_input_edge(),
            preset.minimum_input_edge(),
            "{}",
            case.class_name
        );
        assert_eq!(
            recipe.audited_input_edge(),
            case.input.width,
            "{}",
            case.class_name
        );
        assert_eq!(
            recipe.patch_size(),
            preset.patch_size(),
            "{}",
            case.class_name
        );
        assert_eq!(
            recipe
                .build()
                .unwrap_or_else(|error| panic!("{} recipe should build: {error}", case.class_name))
                .preset(),
            preset,
            "{}",
            case.class_name
        );
        assert_eq!(
            recipe
                .build()
                .unwrap_or_else(|error| panic!("{} recipe should build: {error}", case.class_name))
                .resize_backend(),
            MultimodalResizeBackend::Torchvision,
            "{}",
            case.class_name
        );
        assert_eq!(
            preset.source_path(),
            case.source.path,
            "{}",
            case.class_name
        );
        assert_eq!(preset.audit_commit(), AUDIT_COMMIT, "{}", case.class_name);
        assert_eq!(
            preset.profile(),
            expected_profile(&case.profile),
            "{}",
            case.class_name
        );
        assert_eq!(
            preset.category(),
            expected_category(&case.category),
            "{}",
            case.class_name
        );
        assert!(preset.output_edge() >= 4, "{}", case.class_name);
        assert_eq!(
            preset.audited_input_edge(),
            case.input.width,
            "{}",
            case.class_name
        );

        match preset.category() {
            MultimodalCategory::VisionLanguage => category_counts[0] += 1,
            MultimodalCategory::DocumentUnderstanding => category_counts[1] += 1,
            MultimodalCategory::Video => category_counts[2] += 1,
            _ => panic!("unexpected multimodal category for {}", case.class_name),
        }
    }
    assert_eq!(category_counts, [34, 15, 3]);
}

#[test]
fn every_catalog_alias_resolves_to_its_canonical_multimodal_preset() {
    for &(alias, canonical) in MULTIMODAL_CLASS_ALIASES {
        let alias_preset =
            multimodal_preset(alias).unwrap_or_else(|| panic!("missing multimodal alias {alias}"));
        let canonical_preset = multimodal_preset(canonical)
            .unwrap_or_else(|| panic!("missing canonical multimodal preset {canonical}"));

        assert_eq!(alias_preset, canonical_preset, "{alias}");
        assert_eq!(
            MultimodalProcessor::for_class(alias)
                .expect("catalog alias should build")
                .preset(),
            canonical_preset,
            "{alias}"
        );
    }
}

#[test]
fn multimodal_presets_and_aliases_match_catalog_contracts() {
    let catalog = ProcessorCatalog::new();
    for preset in MULTIMODAL_PRESETS {
        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
            .unwrap_or_else(|| panic!("catalog missing {}", preset.class_name()));
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
        assert_eq!(entry.recipe_id(), Some(preset.recipe_id()));

        let mut expected_aliases = MULTIMODAL_CLASS_ALIASES
            .iter()
            .filter_map(|&(alias, canonical)| (canonical == preset.class_name()).then_some(alias))
            .collect::<Vec<_>>();
        let mut catalog_aliases = entry.class_aliases().to_vec();
        expected_aliases.sort_unstable();
        catalog_aliases.sort_unstable();
        assert_eq!(catalog_aliases, expected_aliases, "{}", preset.class_name());
    }
}

#[test]
fn multimodal_public_inputs_reject_invalid_contracts() {
    assert!(matches!(
        MultimodalProcessor::for_class("UnknownImageProcessor"),
        Err(MultimodalProcessorError::UnknownClass(_))
    ));

    let processor = MultimodalProcessor::for_class("BlipImageProcessor")
        .expect("BLIP compact preset should build");
    assert!(matches!(
        processor.preprocess_frames(&[]),
        Err(MultimodalProcessorError::EmptyInput)
    ));
    let grayscale = ImageFrame::new(8, 8, PixelFormat::Luma8, vec![0; 64])
        .expect("grayscale frame should be valid");
    assert!(matches!(
        processor.preprocess_image(&grayscale),
        Err(MultimodalProcessorError::UnsupportedPixelFormat(
            PixelFormat::Luma8
        ))
    ));
    let non_square = deterministic_frame(8, 9, 0);
    assert!(matches!(
        processor.preprocess_image(&non_square),
        Err(MultimodalProcessorError::NonSquareInput {
            width: 8,
            height: 9,
        })
    ));
    let too_small = deterministic_frame(7, 7, 0);
    assert!(matches!(
        processor.preprocess_image(&too_small),
        Err(MultimodalProcessorError::InputTooSmall {
            minimum: 8,
            actual: 7,
        })
    ));

    assert!(matches!(
        MultimodalArray::new([0], MultimodalArrayData::F32(Vec::new())),
        Err(MultimodalProcessorError::ZeroArrayDimension)
    ));
    assert!(matches!(
        MultimodalArray::new([usize::MAX, 2], MultimodalArrayData::F32(Vec::new())),
        Err(MultimodalProcessorError::ShapeOverflow)
    ));
    assert!(matches!(
        MultimodalArray::new([2], MultimodalArrayData::F32(vec![0.0])),
        Err(MultimodalProcessorError::InvalidArrayLength {
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn frame_multiplicity_is_processed_or_rejected_explicitly() {
    for preset in MULTIMODAL_PRESETS {
        let edge = preset.audited_input_edge();
        let frames = [
            deterministic_frame(edge, edge, 0),
            deterministic_frame(edge, edge, 1),
        ];
        let processor = MultimodalProcessor::for_class(preset.class_name())
            .expect("canonical multimodal preset should build");
        let result = processor.preprocess_frames(&frames);

        if matches!(
            preset.profile(),
            MultimodalProfile::PatchGrid | MultimodalProfile::Video
        ) {
            let output = result.unwrap_or_else(|error| {
                panic!(
                    "{} should process multiple frames: {error}",
                    preset.class_name()
                )
            });
            let field = if preset.profile() == MultimodalProfile::PatchGrid {
                "image_grid_thw"
            } else if preset.class_name() == "TvpImageProcessor" {
                "pixel_values"
            } else {
                "pixel_values_images"
            };
            let MultimodalValue::Array(array) = output
                .get(field)
                .unwrap_or_else(|| panic!("{} missing {field}", preset.class_name()))
            else {
                panic!("{} {field} should be an array", preset.class_name());
            };
            let frame_axis = if preset.class_name() == "TvpImageProcessor" {
                1
            } else {
                0
            };
            assert_eq!(array.shape()[frame_axis], 2, "{}", preset.class_name());
        } else {
            assert!(
                matches!(
                    result,
                    Err(MultimodalProcessorError::UnsupportedFrameCount {
                        class_name,
                        actual: 2,
                    }) if class_name == preset.class_name()
                ),
                "{} must reject extra frames instead of ignoring them",
                preset.class_name()
            );
        }
    }
}

#[test]
fn ppocr_recognition_postprocess_matches_full_fixture() {
    let fixture = fixture();
    let recognition = &fixture.postprocess["recognition"];
    let characters = ["", "a", "b", "c"];

    for class_name in [
        "PPOCRV5ServerRecImageProcessor",
        "PPOCRV6SmallRecImageProcessor",
    ] {
        let case = &recognition[class_name];
        let logits = decode_f32_array(&case["logits"]);
        let shape = array_shape(&case["logits"]);
        let result =
            post_process_text_recognition(&logits, shape[0], shape[1], shape[2], &characters)
                .expect("recognition logits should decode");
        let token_sequences =
            decode_fixture_token_sequences(case, class_name, RecipeTokenSequenceTask::Ocr);
        let expected = &case["result"][0];
        let tolerance = comparison_policy().full_payload;

        assert_token_sequences_match(&token_sequences, &case["token_sequences"], class_name);
        assert_eq!(result.len(), 1, "{class_name}");
        assert_eq!(result[0].text, expected["text"], "{class_name}");
        assert!(
            tolerance.matches(
                f64::from(result[0].score),
                expected["score"].as_f64().expect("score")
            ),
            "{class_name}: {:?} != {expected}",
            result[0]
        );
    }
}

#[test]
fn slanext_table_postprocess_matches_full_fixture() {
    let fixture = fixture();
    let case = &fixture.postprocess["table"];
    let logits = decode_f32_array(&case["logits"]);
    let shape = array_shape(&case["logits"]);
    let vocabulary = slanext_vocabulary();
    let vocabulary_refs = vocabulary.iter().map(String::as_str).collect::<Vec<_>>();
    let actual = post_process_table_recognition(
        &logits,
        shape[1],
        &vocabulary_refs,
        0,
        vocabulary.len() - 1,
    )
    .expect("table logits should decode");
    let token_sequences = decode_fixture_token_sequences(
        case,
        "SLANeXtImageProcessor",
        RecipeTokenSequenceTask::TableStructure,
    );
    let expected = &case["result"];
    let expected_structure = expected["structure"]
        .as_array()
        .expect("structure array")
        .iter()
        .map(|value| value.as_str().expect("structure token").to_owned())
        .collect::<Vec<_>>();
    let tolerance = comparison_policy().full_payload;

    assert_token_sequences_match(&token_sequences, &case["token_sequences"], "SLANeXt");
    assert_eq!(actual.structure, expected_structure);
    assert!(tolerance.matches(
        f64::from(actual.score),
        expected["structure_score"].as_f64().expect("score")
    ));
}

fn decode_fixture_token_sequences(
    case: &Value,
    class_name: &str,
    task: RecipeTokenSequenceTask,
) -> Vec<TokenSequencePostprocessOutput> {
    let logits = decode_f32_array(&case["logits"]);
    let shape = array_shape(&case["logits"]);
    let recipe = multimodal_preset(class_name)
        .unwrap_or_else(|| panic!("missing preset for {class_name}"))
        .processor_recipe()
        .expect("multimodal recipe should be valid")
        .structured_postprocess_recipe()
        .expect("structured postprocess recipe should be valid")
        .unwrap_or_else(|| panic!("{class_name} should define structured postprocessing"));
    let [ProcessorRecipePostprocess::TokenSequence(descriptor)] = recipe.postprocess() else {
        panic!("{class_name} should expose one token-sequence descriptor");
    };
    assert_eq!(descriptor.task, task, "{class_name}");
    assert_eq!(
        descriptor.logits_output, "last_hidden_state",
        "{class_name}"
    );
    match descriptor.decoder {
        image_processors::RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id } => {
            assert_eq!(case["decoder"]["strategy"], "ctc_greedy");
            assert_eq!(
                serde_json::json!(blank_token_id),
                case["decoder"]["blank_token_id"]
            );
        }
        image_processors::RecipeTokenSequenceDecoder::Greedy {
            begin_token_id,
            end_token_id,
        } => {
            assert_eq!(case["decoder"]["strategy"], "greedy");
            assert_eq!(
                serde_json::json!(begin_token_id),
                case["decoder"]["begin_token_id"]
            );
            assert_eq!(
                serde_json::json!(end_token_id),
                case["decoder"]["end_token_id"]
            );
        }
        decoder => panic!("unsupported fixture decoder {decoder:?}"),
    }
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("last_hidden_state"),
        Tensor::new(TensorData::F32(logits), shape, Layout::CHW)
            .expect("fixture logits tensor should be valid"),
    );
    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("fixture token sequence should decode");
    let [RecipePostprocessOutput::TokenSequences { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one token-sequence output, got {outputs:?}");
    };
    outputs.clone()
}

fn assert_token_sequences_match(
    actual: &[TokenSequencePostprocessOutput],
    expected: &Value,
    context: &str,
) {
    let expected = expected.as_array().expect("token sequence array");
    assert_eq!(actual.len(), expected.len(), "{context}");
    let tolerance = comparison_policy().full_payload;
    for (sequence_index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected_ids = expected["token_ids"]
            .as_array()
            .expect("token id array")
            .iter()
            .map(|value| value.as_u64().expect("token id") as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            actual.token_ids(),
            expected_ids.as_slice(),
            "{context}:{sequence_index}"
        );
        let expected_scores = expected["scores"].as_array().expect("score array");
        assert_eq!(
            actual.scores().len(),
            expected_scores.len(),
            "{context}:{sequence_index}"
        );
        for (score_index, (&actual, expected)) in
            actual.scores().iter().zip(expected_scores).enumerate()
        {
            assert!(
                tolerance.matches(f64::from(actual), expected.as_f64().expect("token score")),
                "{context}:{sequence_index}:{score_index}"
            );
        }
    }
}

#[test]
fn uvdoc_rectification_postprocess_matches_every_output_byte() {
    let fixture = fixture();
    let case = &fixture.postprocess["rectification"];
    let prediction = decode_f32_array(&case["prediction"]);
    let prediction_shape = array_shape(&case["prediction"]);
    let original_value = &case["original_images"]["items"][0];
    let original = decode_f32_array(original_value);
    let original_shape = array_shape(original_value);
    let grid_elements = prediction_shape[2] * prediction_shape[3];
    let grid = (0..grid_elements)
        .map(|index| [prediction[index], prediction[grid_elements + index]])
        .collect::<Vec<_>>();
    let actual = post_process_document_rectification(
        &original,
        original_shape[2],
        original_shape[1],
        &grid,
        prediction_shape[3],
        prediction_shape[2],
        255.0,
    )
    .expect("identity rectification should succeed");
    let expected_value = &case["result"]["items"][0]["items"]["images"];
    let expected = decode_u8_array(expected_value);

    assert_eq!(actual.width, prediction_shape[3]);
    assert_eq!(actual.height, prediction_shape[2]);
    assert_eq!(actual.bgr, expected);
}

#[test]
fn uvdoc_rectification_rejects_unsafe_dimensions_and_non_finite_values() {
    let image = [0.0, 0.25, 0.5];
    let grid = [[0.0, 0.0]];

    assert!(matches!(
        post_process_document_rectification(&[], 0, 1, &grid, 1, 1, 255.0),
        Err(MultimodalProcessorError::InvalidRectificationDimensions {
            field: "image",
            width: 0,
            height: 1,
        })
    ));
    assert!(matches!(
        post_process_document_rectification(&image, 1, 1, &[], 0, 1, 255.0),
        Err(MultimodalProcessorError::InvalidRectificationDimensions {
            field: "output",
            width: 0,
            height: 1,
        })
    ));
    assert!(matches!(
        post_process_document_rectification(&image, 1, 1, &grid, 1, 1, f32::NAN),
        Err(MultimodalProcessorError::NonFiniteRectificationScale(value)) if value.is_nan()
    ));

    let non_finite_image = [0.0, f32::INFINITY, 0.5];
    assert!(matches!(
        post_process_document_rectification(&non_finite_image, 1, 1, &grid, 1, 1, 255.0),
        Err(MultimodalProcessorError::NonFiniteRectificationImageValue {
            index: 1,
            value,
        }) if value == f32::INFINITY
    ));
    let non_finite_grid = [[0.0, f32::NEG_INFINITY]];
    assert!(matches!(
        post_process_document_rectification(&image, 1, 1, &non_finite_grid, 1, 1, 255.0),
        Err(MultimodalProcessorError::NonFiniteRectificationGridCoordinate {
            index: 0,
            axis: "y",
            value,
        }) if value == f32::NEG_INFINITY
    ));
}

#[test]
fn layoutlm_word_boxes_use_exact_page_coordinates() {
    let word = normalize_ocr_word("total", [10, 20, 90, 80], 100, 100)
        .expect("non-empty page should normalize");

    assert_eq!(word.text, "total");
    assert_eq!(word.box_1000, [100, 200, 900, 800]);
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("multimodal fixture should parse")
}

fn comparison_policy() -> &'static ComparisonPolicy {
    static POLICY: OnceLock<ComparisonPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        let fixture = fixture();
        fixture.acceptance.validate();
        fixture.acceptance.comparison
    })
}

fn preprocess_fixture_input(
    class_name: &str,
    input: &FixtureInput,
    offset: usize,
) -> MultimodalOutput {
    let frames = (0..input.frames)
        .map(|frame| deterministic_frame(input.width, input.height, offset + frame))
        .collect::<Vec<_>>();
    let processor = MultimodalProcessor::for_class(class_name)
        .unwrap_or_else(|error| panic!("{class_name} should resolve: {error}"));
    let result = if input.kind == "image" {
        processor.preprocess_image(&frames[0])
    } else {
        processor.preprocess_frames(&frames)
    };
    result.unwrap_or_else(|error| panic!("{class_name} should preprocess: {error}"))
}

fn assert_output_matches(
    class_name: &str,
    actual: &MultimodalOutput,
    expected: &BTreeMap<String, Value>,
) {
    assert_eq!(
        actual.fields().len(),
        expected.len(),
        "{class_name} output field count"
    );
    for (name, expected) in expected {
        let actual = actual
            .get(name)
            .unwrap_or_else(|| panic!("{class_name} missing output {name}"));
        assert_value_matches(class_name, name, actual, expected);
    }
}

fn deterministic_frame(width: usize, height: usize, offset: usize) -> ImageFrame {
    let data = (0..width * height * 3)
        .map(|index| ((index * 37 + 17 + offset * 53) % 256) as u8)
        .collect::<Vec<_>>();
    ImageFrame::new(width, height, PixelFormat::Rgb8, data)
        .expect("deterministic image should be valid")
}

fn expected_category(value: &str) -> MultimodalCategory {
    match value {
        "vision_language" => MultimodalCategory::VisionLanguage,
        "document_understanding" => MultimodalCategory::DocumentUnderstanding,
        "video" => MultimodalCategory::Video,
        other => panic!("unexpected category {other}"),
    }
}

fn expected_profile(value: &str) -> MultimodalProfile {
    match value {
        "fixed" => MultimodalProfile::Fixed,
        "fixed_mask" => MultimodalProfile::FixedMask,
        "tiled" => MultimodalProfile::Tiled,
        "patch_grid" => MultimodalProfile::PatchGrid,
        "patch_frames" => MultimodalProfile::PatchFrames,
        "adaptive_patches" => MultimodalProfile::AdaptivePatches,
        "fuyu" => MultimodalProfile::Fuyu,
        "gemma4" => MultimodalProfile::Gemma4,
        "gemma4_unified" => MultimodalProfile::Gemma4Unified,
        "minicpm" => MultimodalProfile::MiniCpm,
        "phi4" => MultimodalProfile::Phi4,
        "video" => MultimodalProfile::Video,
        "uvdoc" => MultimodalProfile::UvDoc,
        other => panic!("unexpected profile {other}"),
    }
}

fn assert_value_matches(class_name: &str, field: &str, actual: &MultimodalValue, expected: &Value) {
    match expected["kind"].as_str() {
        Some("array") => {
            let MultimodalValue::Array(actual) = actual else {
                panic!("{class_name}:{field} should be an array, got {actual:?}");
            };
            assert_array_matches(class_name, field, actual, expected);
        }
        Some("sequence") => {
            let MultimodalValue::Sequence(actual) = actual else {
                panic!("{class_name}:{field} should be a sequence, got {actual:?}");
            };
            let expected = expected["items"].as_array().expect("sequence items");
            assert_eq!(actual.len(), expected.len(), "{class_name}:{field}");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_value_matches(class_name, &format!("{field}[{index}]"), actual, expected);
            }
        }
        Some("mapping") => {
            let MultimodalValue::Mapping(actual) = actual else {
                panic!("{class_name}:{field} should be a mapping, got {actual:?}");
            };
            let expected = expected["items"].as_object().expect("mapping items");
            assert_eq!(actual.len(), expected.len(), "{class_name}:{field}");
            for (name, expected) in expected {
                assert_value_matches(
                    class_name,
                    &format!("{field}.{name}"),
                    actual.get(name).expect("mapping field"),
                    expected,
                );
            }
        }
        None if expected.is_i64() || expected.is_u64() => {
            let MultimodalValue::Integer(actual) = actual else {
                panic!("{class_name}:{field} should be an integer, got {actual:?}");
            };
            assert_eq!(
                *actual,
                expected.as_i64().expect("signed integer"),
                "{class_name}:{field}"
            );
        }
        None if expected.is_f64() => {
            let MultimodalValue::Float(actual) = actual else {
                panic!("{class_name}:{field} should be a float, got {actual:?}");
            };
            let expected = expected.as_f64().expect("float");
            assert!(
                comparison_policy().full_payload.matches(*actual, expected),
                "{class_name}:{field} {actual} != {expected}"
            );
        }
        None if expected.is_string() => {
            let MultimodalValue::Text(actual) = actual else {
                panic!("{class_name}:{field} should be text, got {actual:?}");
            };
            assert_eq!(
                actual,
                expected.as_str().expect("text"),
                "{class_name}:{field}"
            );
        }
        kind => panic!("{class_name}:{field} has unsupported fixture kind {kind:?}"),
    }
}

fn assert_array_matches(class_name: &str, field: &str, actual: &MultimodalArray, expected: &Value) {
    assert_eq!(
        actual.shape(),
        array_shape(expected),
        "{class_name}:{field} shape"
    );
    match (
        actual.data(),
        expected["dtype"].as_str().expect("array dtype"),
    ) {
        (MultimodalArrayData::F32(actual), "float32") => {
            let expected = decode_f32_array(expected);
            let tolerance = comparison_policy().full_payload;
            assert_eq!(actual.len(), expected.len(), "{class_name}:{field}");
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    tolerance.matches(f64::from(actual), f64::from(expected)),
                    "{class_name}:{field}[{index}] {actual} != {expected}"
                );
            }
        }
        (MultimodalArrayData::U8(actual), "uint8") => {
            assert_eq!(actual, &decode_u8_array(expected), "{class_name}:{field}");
        }
        (MultimodalArrayData::I32(actual), "int32") => {
            assert_eq!(actual, &decode_i32_array(expected), "{class_name}:{field}");
        }
        (MultimodalArrayData::I64(actual), "int64") => {
            assert_eq!(actual, &decode_i64_array(expected), "{class_name}:{field}");
        }
        (MultimodalArrayData::Bool(actual), "bool") => {
            assert_eq!(actual, &decode_u8_array(expected), "{class_name}:{field}");
        }
        (actual, dtype) => panic!("{class_name}:{field} dtype mismatch: {actual:?} != {dtype}"),
    }
}

fn array_shape(value: &Value) -> Vec<usize> {
    value["shape"]
        .as_array()
        .expect("array shape")
        .iter()
        .map(|dimension| dimension.as_u64().expect("shape dimension") as usize)
        .collect()
}

fn array_bytes(value: &Value) -> Vec<u8> {
    BASE64
        .decode(value["data"].as_str().expect("base64 array data"))
        .expect("valid base64 array data")
}

fn decode_f32_array(value: &Value) -> Vec<f32> {
    array_bytes(value)
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 chunk")))
        .collect()
}

fn decode_i32_array(value: &Value) -> Vec<i32> {
    array_bytes(value)
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("i32 chunk")))
        .collect()
}

fn decode_i64_array(value: &Value) -> Vec<i64> {
    array_bytes(value)
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("i64 chunk")))
        .collect()
}

fn decode_u8_array(value: &Value) -> Vec<u8> {
    array_bytes(value)
}

fn slanext_vocabulary() -> Vec<String> {
    let mut vocabulary = vec![
        "sos".to_owned(),
        "<thead>".to_owned(),
        "</thead>".to_owned(),
        "<tbody>".to_owned(),
        "</tbody>".to_owned(),
        "<tr>".to_owned(),
        "</tr>".to_owned(),
        "<td".to_owned(),
        ">".to_owned(),
        "</td>".to_owned(),
    ];
    vocabulary.extend((2..=20).map(|span| format!(" colspan=\"{span}\"")));
    vocabulary.extend((2..=20).map(|span| format!(" rowspan=\"{span}\"")));
    vocabulary.push("<td></td>".to_owned());
    vocabulary.push("eos".to_owned());
    vocabulary
}
