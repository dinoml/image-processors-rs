use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    EncoderGeometry, EncoderImageProcessor, EncoderImageProcessorConfig,
    EncoderImageProcessorPreset, ImageFrame, ImageSize, Layout, PixelFormat, ProcessorRecipeStage,
    ProcessorTensorName, Swin2SrImageProcessor, Swin2SrImageProcessorConfig, Tensor, TensorData,
    TensorLeadingAxis,
};
use serde::Deserialize;

#[path = "common/fixture_contract.rs"]
mod fixture_contract;

use fixture_contract::{AcceptanceContract, FloatTolerance};

const FIXTURE: &str = include_str!("fixtures/transformers/catalog_encoder_restoration.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-encoder-restoration-parity.v1";
const AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const SOURCE_MANIFEST_SHA256: &str =
    "8a994581e9fa2d8e134dddafcb6f884afdcd8d198b48831db8ea5624bd8acccc";
const GENERATOR: &str = "scripts/parity/transformers_encoder_restoration_parity.py";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    generator: String,
    upstream: Upstream,
    acceptance: AcceptanceContract,
    image: FixtureImage,
    cases: Vec<ParityCase>,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    commit: String,
    source_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct FixtureImage {
    mode: String,
    width: usize,
    height: usize,
    data: DataBlob,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    class_name: String,
    canonical_class: String,
    recipe_id: String,
    processor_config: CompactConfig,
    stages: BTreeMap<String, TensorReference>,
    outputs: BTreeMap<String, TensorReference>,
}

#[derive(Debug, Default, Deserialize)]
struct CompactConfig {
    size: Option<SizeSpec>,
    crop_size: Option<SizeSpec>,
    crop_pct: Option<f64>,
    include_top: Option<bool>,
    do_color_quantize: Option<bool>,
    clusters: Option<Vec<[f32; 3]>>,
    resize_short: Option<usize>,
    size_divisor: Option<usize>,
    pretrained_cfg: Option<TimmConfig>,
}

#[derive(Debug, Deserialize)]
struct SizeSpec {
    height: Option<usize>,
    width: Option<usize>,
    shortest_edge: Option<usize>,
}

impl SizeSpec {
    fn image_size(&self, field: &str) -> ImageSize {
        ImageSize::new(
            self.height
                .unwrap_or_else(|| panic!("{field} missing height")),
            self.width
                .unwrap_or_else(|| panic!("{field} missing width")),
        )
        .unwrap_or_else(|error| panic!("invalid {field}: {error}"))
    }

    fn shortest_edge(&self, field: &str) -> usize {
        self.shortest_edge
            .unwrap_or_else(|| panic!("{field} missing shortest_edge"))
    }
}

#[derive(Debug, Deserialize)]
struct TimmConfig {
    input_size: [usize; 3],
    crop_pct: f64,
}

#[derive(Debug, Deserialize)]
struct TensorReference {
    shape: Vec<usize>,
    dtype: String,
    data: DataBlob,
}

#[derive(Debug, Deserialize)]
struct DataBlob {
    encoding: String,
    #[serde(default)]
    byte_order: Option<String>,
    value: String,
}

#[test]
fn fixture_records_exact_audited_transformers_source() {
    let fixture = fixture();

    fixture.acceptance.validate();

    assert_eq!(
        (
            fixture.schema.as_str(),
            fixture.generator.as_str(),
            fixture.upstream.commit.as_str(),
            fixture.upstream.source_manifest_sha256.as_str(),
        ),
        (
            FIXTURE_SCHEMA,
            GENERATOR,
            AUDIT_COMMIT,
            SOURCE_MANIFEST_SHA256
        )
    );
    assert_eq!(fixture.acceptance.deterministic_input.locations, ["image"]);
    assert_eq!(
        fixture.acceptance.processor_config.locations,
        ["cases[].processor_config"]
    );
}

#[test]
fn fixture_covers_every_encoder_class_alias_and_swin2sr_alias() {
    let fixture = fixture();
    let actual: BTreeSet<_> = fixture
        .cases
        .iter()
        .map(|case| case.class_name.as_str())
        .collect();
    let mut expected = BTreeSet::new();
    for preset in EncoderImageProcessorPreset::ALL {
        expected.insert(preset.class_name());
        expected.extend(preset.class_aliases().iter().copied());
    }
    expected.insert("Swin2SRImageProcessor");
    expected.insert("Swin2SRImageProcessorPil");

    assert_eq!(actual, expected);
}

#[test]
fn every_catalog_recipe_id_is_constructible() {
    let fixture = fixture();
    let fixture_ids: BTreeSet<_> = fixture
        .cases
        .iter()
        .map(|case| case.recipe_id.as_str())
        .collect();
    let mut recipe_ids = BTreeSet::new();
    for preset in EncoderImageProcessorPreset::ALL {
        let recipe = EncoderImageProcessorConfig::for_preset(preset)
            .processor_recipe()
            .unwrap_or_else(|error| panic!("{} recipe failed: {error}", preset.class_name()));
        recipe_ids.insert(recipe.id().to_owned());
    }
    let swin = Swin2SrImageProcessorConfig::default()
        .processor_recipe()
        .expect("default Swin2SR recipe should build");
    recipe_ids.insert(swin.id().to_owned());

    assert_eq!(
        recipe_ids,
        fixture_ids.into_iter().map(str::to_owned).collect()
    );
}

#[test]
fn compact_processed_and_output_payloads_match_transformers() {
    let fixture = fixture();
    let image = fixture_image(&fixture.image);
    let tolerance = fixture.acceptance.comparison.full_payload;

    for case in &fixture.cases {
        let processed_reference = case
            .stages
            .get("processed_image")
            .unwrap_or_else(|| panic!("{} missing processed_image", case.class_name));
        let (expected_output_name, output_reference) = case
            .outputs
            .iter()
            .next()
            .unwrap_or_else(|| panic!("{} missing output", case.class_name));
        assert_eq!(
            case.outputs.len(),
            1,
            "{} fixture must contain one complete main output",
            case.class_name
        );

        let (processed, output_name, output) = if case.canonical_class == "Swin2SRImageProcessor" {
            let output_config = swin_config(case);
            let output_processor = Swin2SrImageProcessor::new(output_config)
                .unwrap_or_else(|error| panic!("{} output config: {error}", case.class_name));
            let processed = output_processor
                .preprocess_image_stage(&image)
                .unwrap_or_else(|error| panic!("{} processed stage: {error}", case.class_name));
            let typed_output = output_processor
                .preprocess_image_output(&image)
                .unwrap_or_else(|error| panic!("{} output: {error}", case.class_name));
            let named = typed_output
                .tensors()
                .first()
                .unwrap_or_else(|| panic!("{} missing Rust output", case.class_name));
            (
                processed,
                named.name().as_str().to_owned(),
                named.tensor().clone(),
            )
        } else {
            let output_config = encoder_config(case);
            let output_processor = EncoderImageProcessor::new(output_config)
                .unwrap_or_else(|error| panic!("{} output config: {error}", case.class_name));
            let processed = output_processor
                .preprocess_image_stage(&image)
                .unwrap_or_else(|error| panic!("{} processed stage: {error}", case.class_name));
            let typed_output = output_processor
                .preprocess_image_output(&image)
                .unwrap_or_else(|error| panic!("{} output: {error}", case.class_name));
            assert_eq!(
                typed_output.tensors().len(),
                1,
                "{} should emit only its main tensor",
                case.class_name
            );
            let named = &typed_output.tensors()[0];
            (
                processed,
                named.name().as_str().to_owned(),
                named.tensor().clone(),
            )
        };

        assert_eq!(
            output_name, *expected_output_name,
            "{} output name",
            case.class_name
        );
        assert_tensor_matches(
            &case.class_name,
            "processed_image",
            &processed,
            processed_reference,
            tolerance,
        );
        assert_tensor_matches(
            &case.class_name,
            expected_output_name,
            &output,
            output_reference,
            tolerance,
        );
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("encoder/restoration fixture should parse")
}

fn fixture_image(reference: &FixtureImage) -> ImageFrame {
    assert_eq!(reference.mode, "RGB");
    ImageFrame::new(
        reference.width,
        reference.height,
        PixelFormat::Rgb8,
        decode_blob(&reference.data),
    )
    .expect("fixture image should be valid")
}

fn encoder_config(case: &ParityCase) -> EncoderImageProcessorConfig {
    let preset = EncoderImageProcessorPreset::from_class_name(&case.class_name)
        .unwrap_or_else(|| panic!("unsupported fixture class {}", case.class_name));
    let mut config = EncoderImageProcessorConfig::for_class_name(&case.class_name)
        .unwrap_or_else(|| panic!("unsupported fixture class {}", case.class_name));
    let size = case.processor_config.size.as_ref();
    let crop_size = case.processor_config.crop_size.as_ref();
    config.geometry = match config.geometry {
        EncoderGeometry::Fixed { .. } => EncoderGeometry::Fixed {
            size: size
                .unwrap_or_else(|| panic!("{} missing size", case.class_name))
                .image_size("size"),
        },
        EncoderGeometry::FixedThenCenterCrop { .. } => EncoderGeometry::FixedThenCenterCrop {
            size: size
                .unwrap_or_else(|| panic!("{} missing size", case.class_name))
                .image_size("size"),
            crop_size: crop_size
                .unwrap_or_else(|| panic!("{} missing crop_size", case.class_name))
                .image_size("crop_size"),
        },
        EncoderGeometry::ShortestEdgeThenCenterCrop { .. } => {
            EncoderGeometry::ShortestEdgeThenCenterCrop {
                shortest_edge: size
                    .unwrap_or_else(|| panic!("{} missing size", case.class_name))
                    .shortest_edge("size"),
                crop_size: crop_size
                    .unwrap_or_else(|| panic!("{} missing crop_size", case.class_name))
                    .image_size("crop_size"),
            }
        }
        EncoderGeometry::ScaledShortestEdgeThenCenterCrop {
            scale_numerator,
            scale_denominator,
            ..
        } => EncoderGeometry::ScaledShortestEdgeThenCenterCrop {
            shortest_edge: size
                .unwrap_or_else(|| panic!("{} missing size", case.class_name))
                .shortest_edge("size"),
            scale_numerator,
            scale_denominator,
            crop_size: crop_size
                .unwrap_or_else(|| panic!("{} missing crop_size", case.class_name))
                .image_size("crop_size"),
        },
        EncoderGeometry::CropPercentage {
            warp_at_or_above, ..
        } => {
            let (output_edge, crop_percentage) = match &case.processor_config.pretrained_cfg {
                Some(timm) => (timm.input_size[1], timm.crop_pct),
                None => (
                    size.unwrap_or_else(|| panic!("{} missing size", case.class_name))
                        .shortest_edge("size"),
                    case.processor_config
                        .crop_pct
                        .unwrap_or_else(|| panic!("{} missing crop_pct", case.class_name)),
                ),
            };
            EncoderGeometry::CropPercentage {
                output_edge,
                crop_percentage,
                warp_at_or_above,
            }
        }
        EncoderGeometry::PerceiverCropThenResize { .. } => {
            EncoderGeometry::PerceiverCropThenResize {
                size: size
                    .unwrap_or_else(|| panic!("{} missing size", case.class_name))
                    .image_size("size"),
                crop_reference: crop_size
                    .unwrap_or_else(|| panic!("{} missing crop_size", case.class_name))
                    .image_size("crop_size"),
            }
        }
        EncoderGeometry::PaddleShortestEdgeThenCenterCrop { .. } => {
            EncoderGeometry::PaddleShortestEdgeThenCenterCrop {
                shortest_edge: case
                    .processor_config
                    .resize_short
                    .unwrap_or_else(|| panic!("{} missing resize_short", case.class_name)),
                size_divisor: case
                    .processor_config
                    .size_divisor
                    .unwrap_or_else(|| panic!("{} missing size_divisor", case.class_name)),
                crop_size: crop_size
                    .unwrap_or_else(|| panic!("{} missing crop_size", case.class_name))
                    .image_size("crop_size"),
            }
        }
        _ => panic!("{} uses an unsupported encoder geometry", case.class_name),
    };

    if preset == EncoderImageProcessorPreset::ImageGpt {
        config.color_clusters = case.processor_config.clusters.clone();
        config.do_color_quantize = case.processor_config.do_color_quantize.unwrap_or(true);
    }
    if preset == EncoderImageProcessorPreset::EfficientNet
        && !case.processor_config.include_top.unwrap_or(true)
    {
        config.additional_std = None;
    }

    config
}

fn swin_config(case: &ParityCase) -> Swin2SrImageProcessorConfig {
    Swin2SrImageProcessorConfig {
        size_divisor: case
            .processor_config
            .size_divisor
            .unwrap_or_else(|| panic!("{} missing size_divisor", case.class_name)),
        ..Default::default()
    }
}

fn assert_tensor_matches(
    class_name: &str,
    field: &str,
    actual: &Tensor,
    expected: &TensorReference,
    tolerance: FloatTolerance,
) {
    assert_eq!(actual.shape(), expected.shape, "{class_name} {field} shape");
    let (expected_layout, expected_leading_axis) = match expected.shape.len() {
        2 => (Layout::NC, None),
        4 => (Layout::NCHW, Some(TensorLeadingAxis::Batch)),
        rank => panic!("{class_name} {field} unsupported fixture rank {rank}"),
    };
    assert_eq!(
        actual.layout(),
        expected_layout,
        "{class_name} {field} layout"
    );
    assert_eq!(
        actual.leading_axis(),
        expected_leading_axis,
        "{class_name} {field} leading axis"
    );
    match expected.dtype.as_str() {
        "uint8" => {
            let actual_values = match actual.data() {
                TensorData::U8(values) => values,
                data => panic!(
                    "{class_name} {field} expected uint8, got {:?}",
                    data.dtype()
                ),
            };
            let expected_values = decode_blob(&expected.data);
            assert_byte_values_close(class_name, field, actual_values, &expected_values);
        }
        "float32" => {
            let actual_values = match actual.data() {
                TensorData::F32(values) => values,
                data => panic!(
                    "{class_name} {field} expected float32, got {:?}",
                    data.dtype()
                ),
            };
            let expected_values = reference_f32(expected);
            assert_float_values_close(
                class_name,
                field,
                actual_values,
                &expected_values,
                tolerance,
            );
        }
        "int64" => {
            let actual_values = match actual.data() {
                TensorData::I64(values) => values,
                data => panic!(
                    "{class_name} {field} expected int64, got {:?}",
                    data.dtype()
                ),
            };
            assert_eq!(
                actual_values,
                &reference_i64(expected),
                "{class_name} {field} complete int64 payload"
            );
        }
        dtype => panic!("unsupported fixture dtype {dtype}"),
    }
}

fn assert_byte_values_close(class_name: &str, field: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(
        actual, expected,
        "{class_name} {field} complete byte payload"
    );
}

fn assert_float_values_close(
    class_name: &str,
    field: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: FloatTolerance,
) {
    assert_eq!(actual.len(), expected.len(), "{class_name} {field} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        assert!(
            tolerance.matches(f64::from(actual), f64::from(expected)),
            "{class_name} {field} value {index}: actual={actual}, expected={expected}, abs_diff={difference}, allowed={}",
            tolerance.allowed_delta(f64::from(expected))
        );
    }
}

fn decode_blob(blob: &DataBlob) -> Vec<u8> {
    assert_eq!(blob.encoding, "base64");
    if let Some(byte_order) = &blob.byte_order {
        assert_eq!(byte_order, "little");
    }
    BASE64
        .decode(&blob.value)
        .expect("base64 payload should decode")
}

fn reference_f32(reference: &TensorReference) -> Vec<f32> {
    let bytes = decode_blob(&reference.data);
    assert_eq!(bytes.len() % size_of::<f32>(), 0, "unaligned float fixture");
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("float chunk is four bytes")))
        .collect()
}

fn reference_i64(reference: &TensorReference) -> Vec<i64> {
    let bytes = decode_blob(&reference.data);
    assert_eq!(bytes.len() % size_of::<i64>(), 0, "unaligned int64 fixture");
    bytes
        .chunks_exact(size_of::<i64>())
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("int64 chunk is eight bytes")))
        .collect()
}

#[test]
fn output_name_types_match_public_conventions() {
    assert_eq!(
        (
            ProcessorTensorName::PixelValues.as_str(),
            ProcessorTensorName::other("input_ids").as_str().to_owned(),
        ),
        ("pixel_values", "input_ids".to_owned())
    );
}

#[test]
fn dinov3_stage_and_batch_preserve_rescale_before_resize_order() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.class_name == "DINOv3ViTImageProcessor")
        .expect("DINOv3 fixture case should exist");
    let image = fixture_image(&fixture.image);
    let processor = EncoderImageProcessor::new(encoder_config(case))
        .expect("DINOv3 fixture config should build");

    let stage = processor
        .preprocess_image_stage(&image)
        .expect("DINOv3 stage should preprocess");
    assert!(
        matches!(stage.data(), TensorData::F32(_)),
        "DINOv3 stage must retain the upstream rescale-before-resize float domain"
    );
    assert_tensor_matches(
        &case.class_name,
        "processed_image",
        &stage,
        case.stages
            .get("processed_image")
            .expect("DINOv3 stage fixture should exist"),
        fixture.acceptance.comparison.full_payload,
    );

    let single = processor
        .preprocess_image(&image)
        .expect("DINOv3 image should preprocess");
    let batch = processor
        .preprocess_images(&[image.clone(), image])
        .expect("DINOv3 batch should preprocess");
    assert_eq!(batch.shape()[0], 2);
    assert_eq!(&batch.shape()[1..], &single.shape()[1..]);
    assert_eq!(batch.layout(), single.layout());
    assert_eq!(batch.leading_axis(), Some(TensorLeadingAxis::Batch));
    let TensorData::F32(single_values) = single.data() else {
        panic!("DINOv3 single output should be float32");
    };
    let TensorData::F32(batch_values) = batch.data() else {
        panic!("DINOv3 batch output should be float32");
    };
    assert_eq!(&batch_values[..single_values.len()], single_values);
    assert_eq!(&batch_values[single_values.len()..], single_values);
}

#[test]
fn swin2sr_advances_dimensions_already_divisible_by_the_window() {
    let image = ImageFrame::new(8, 4, PixelFormat::Rgb8, vec![17; 8 * 4 * 3])
        .expect("test frame should be valid");
    let processor = Swin2SrImageProcessor::new(Swin2SrImageProcessorConfig {
        size_divisor: 4,
        ..Default::default()
    })
    .expect("Swin2SR config should build");

    let prepared = processor
        .prepare_image(&image)
        .expect("symmetric padding should succeed");

    assert_eq!((prepared.height(), prepared.width()), (8, 12));

    let recipe = processor
        .config()
        .processor_recipe()
        .expect("Swin2SR recipe should build");
    assert!(matches!(
        recipe.stages().first(),
        Some(ProcessorRecipeStage::PadSymmetricToNextMultiple { multiples })
            if *multiples == ImageSize::new(4, 4).expect("valid multiples")
    ));
}

#[test]
fn swin2sr_postprocess_clamps_restoration_output_to_unit_range() {
    let processor = Swin2SrImageProcessor::new(Swin2SrImageProcessorConfig::default())
        .expect("Swin2SR config should build");
    let output = Tensor::new(
        TensorData::F32(vec![-0.25, 0.5, 1.25]),
        [1, 3, 1, 1],
        Layout::NCHW,
    )
    .expect("restoration tensor should be valid");

    let frames = processor
        .postprocess_image_tensor(&output)
        .expect("restoration output should postprocess");

    assert_eq!(frames[0].data(), [0, 128, 255]);
}
