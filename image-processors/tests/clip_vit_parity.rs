use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, ImageFrame, ImageProcessor, Layout, PixelFormat,
    ProcessorOutput, ProcessorTensorName, ResizeParity, Tensor, TensorData, VitImageProcessor,
    VitImageProcessorConfig,
};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/transformers_clip_vit.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const PIXEL_VALUES: &str = "pixel_values";
const PROCESSED_IMAGE: &str = "processed_image";
const PROCESSED_IMAGE_MAX_ABS_TOLERANCE: u8 = 0;
const PROCESSED_IMAGE_MEAN_ABS_TOLERANCE: f64 = 0.0;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    cases: Vec<ParityCase>,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    family: String,
    class_name: String,
    backend: String,
    model_id: String,
    image: FixtureImage,
    #[serde(default)]
    stages: BTreeMap<String, TensorSummary>,
    outputs: BTreeMap<String, TensorSummary>,
    comparison: ComparisonPolicy,
}

#[derive(Debug, Deserialize)]
struct ComparisonPolicy {
    float_full_values: FloatTolerance,
    float_statistics: FloatTolerance,
}

#[derive(Debug, Deserialize)]
struct FloatTolerance {
    absolute_tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureImage {
    mode: String,
    width: usize,
    height: usize,
}

#[derive(Debug, Deserialize)]
struct TensorSummary {
    shape: Vec<usize>,
    dtype: String,
    min: f64,
    mean: f64,
    max: f64,
    sample: Vec<f64>,
    data: Option<TensorDataBlob>,
}

#[derive(Debug, Deserialize)]
struct TensorDataBlob {
    encoding: String,
    dtype: String,
    value: String,
}

#[test]
fn fixture_schema_matches_expected_version() {
    let fixture = load_fixture();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn clip_pixel_values_match_transformers_fixture() {
    for class_name in ["CLIPImageProcessorPil", "CLIPImageProcessor"] {
        let case = fixture_case(class_name);
        let frame = deterministic_frame(case.image.width, case.image.height);
        let config = ClipImageProcessorConfig {
            resize_parity: fixture_resize_parity(&case),
            ..Default::default()
        };
        let processor = ClipImageProcessor::new(config).expect("fixture CLIP config should build");

        let output = processor
            .preprocess_image_output(&frame)
            .expect("CLIP preprocessing should succeed");

        assert_output_has_pixel_values_only(&case, &output);
        assert_pixel_values_match(&case, output.pixel_values().expect("missing pixel_values"));
    }
}

#[test]
fn vit_pixel_values_match_transformers_fixture() {
    for class_name in ["ViTImageProcessorPil", "ViTImageProcessor"] {
        let case = fixture_case(class_name);
        let frame = deterministic_frame(case.image.width, case.image.height);
        let config = VitImageProcessorConfig {
            resize_parity: fixture_resize_parity(&case),
            ..Default::default()
        };
        let processor = VitImageProcessor::new(config).expect("fixture ViT config should build");

        let output = processor
            .preprocess_image_output(&frame)
            .expect("ViT preprocessing should succeed");

        assert_output_has_pixel_values_only(&case, &output);
        assert_pixel_values_match(&case, output.pixel_values().expect("missing pixel_values"));
    }
}

#[test]
fn clip_processed_image_stage_matches_transformers_fixture() {
    for class_name in ["CLIPImageProcessorPil", "CLIPImageProcessor"] {
        let case = fixture_case(class_name);
        let frame = deterministic_frame(case.image.width, case.image.height);
        let processor = unnormalized_image_processor(&case);

        let tensor = processor
            .preprocess_image(&frame)
            .expect("CLIP processed-image stage should succeed");

        assert_processed_image_stage_matches(&case, &tensor);
    }
}

#[test]
fn vit_processed_image_stage_matches_transformers_fixture() {
    for class_name in ["ViTImageProcessorPil", "ViTImageProcessor"] {
        let case = fixture_case(class_name);
        let frame = deterministic_frame(case.image.width, case.image.height);
        let processor = unnormalized_image_processor(&case);

        let tensor = processor
            .preprocess_image(&frame)
            .expect("ViT processed-image stage should succeed");

        assert_processed_image_stage_matches(&case, &tensor);
    }
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("fixture JSON should parse")
}

fn fixture_case(class_name: &str) -> ParityCase {
    let mut matches = load_fixture()
        .cases
        .into_iter()
        .filter(|case| case.class_name == class_name);
    let case = matches
        .next()
        .unwrap_or_else(|| panic!("missing {class_name} fixture case"));
    assert!(
        matches.next().is_none(),
        "fixture should contain one {class_name} case"
    );
    assert_eq!(case.image.mode, "RGB");
    case
}

fn deterministic_frame(width: usize, height: usize) -> ImageFrame {
    let values = (0..width * height * 3)
        .map(|index| ((index as u32 * 37 + 17) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Rgb8, values)
        .expect("deterministic fixture image should be valid")
}

fn unnormalized_image_processor(case: &ParityCase) -> ImageProcessor {
    let mut config = match case.family.as_str() {
        "clip" => ClipImageProcessorConfig::default().image_processor_config(),
        "vit" => VitImageProcessorConfig::default().image_processor_config(),
        family => panic!("unsupported parity family: {family}"),
    };
    config.resize_parity = fixture_resize_parity(case);
    config.do_rescale = false;
    config.do_normalize = false;
    ImageProcessor::new(config).expect("unnormalized image processor config should build")
}

fn fixture_resize_parity(case: &ParityCase) -> ResizeParity {
    match case.backend.as_str() {
        "pil" => ResizeParity::Compatibility,
        "torchvision" => ResizeParity::Torchvision,
        backend => panic!("unsupported fixture backend: {backend}"),
    }
}

fn assert_pixel_values_match(case: &ParityCase, tensor: &Tensor) {
    let reference = case
        .outputs
        .get(PIXEL_VALUES)
        .unwrap_or_else(|| panic!("{} fixture missing pixel_values", case.family));
    let values = tensor.data().to_vec::<f32>();
    let observed = summarize(&values);

    assert_eq!(
        tensor.shape(),
        reference.shape.as_slice(),
        "{}",
        case.model_id
    );
    assert_eq!(tensor.layout(), Layout::NCHW, "{}", case.model_id);
    assert_eq!(reference.dtype, "float32");
    assert!(
        matches!(tensor.data(), TensorData::F32(_)),
        "{} produced {:?}, expected float32",
        case.model_id,
        tensor.data().dtype()
    );
    assert_close(
        observed.min,
        reference.min,
        case.comparison.float_statistics.absolute_tolerance,
        &case.family,
        "min",
    );
    assert_close(
        observed.mean,
        reference.mean,
        case.comparison.float_statistics.absolute_tolerance,
        &case.family,
        "mean",
    );
    assert_close(
        observed.max,
        reference.max,
        case.comparison.float_statistics.absolute_tolerance,
        &case.family,
        "max",
    );
    let expected = reference_f32(reference);
    assert_eq!(
        values.len(),
        expected.len(),
        "{} pixel_values length",
        case.model_id
    );
    assert_float_values_close(case, &values, &expected, &reference.shape);
    assert_sample_close(case, &values, &reference.sample);
}

fn assert_processed_image_stage_matches(case: &ParityCase, tensor: &Tensor) {
    let reference = case
        .stages
        .get(PROCESSED_IMAGE)
        .unwrap_or_else(|| panic!("{} fixture missing processed_image stage", case.family));
    let expected = reference_bytes(reference);
    let actual = tensor.data().to_vec::<u8>();
    let diff = byte_diff(&actual, &expected);

    assert_eq!(
        tensor.shape(),
        reference.shape.as_slice(),
        "{}",
        case.model_id
    );
    assert_eq!(tensor.layout(), Layout::NCHW, "{}", case.model_id);
    assert_eq!(reference.dtype, "uint8");
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} processed_image byte length",
        case.model_id
    );
    assert_eq!(
        diff.max_abs,
        PROCESSED_IMAGE_MAX_ABS_TOLERANCE,
        "{} processed_image byte drift: {}",
        case.model_id,
        diff.describe(&reference.shape)
    );
    assert!(
        diff.mean_abs <= PROCESSED_IMAGE_MEAN_ABS_TOLERANCE,
        "{} processed_image mean byte drift too high: {}",
        case.model_id,
        diff.describe(&reference.shape)
    );
    assert_byte_sample_close(&case.family, &actual, &reference.sample);
}

fn assert_output_has_pixel_values_only(case: &ParityCase, output: &ProcessorOutput) {
    let reference_names: Vec<_> = case.outputs.keys().map(String::as_str).collect();
    assert_eq!(reference_names, [PIXEL_VALUES], "{}", case.model_id);
    assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
    assert_eq!(
        output.tensors()[0].name(),
        &ProcessorTensorName::PixelValues,
        "{}",
        case.model_id
    );
    assert!(
        output.metadata().is_empty(),
        "{} should not emit metadata",
        case.model_id
    );
    assert_eq!(ProcessorTensorName::PixelValues.as_str(), PIXEL_VALUES);
}

fn reference_bytes(reference: &TensorSummary) -> Vec<u8> {
    let data = reference
        .data
        .as_ref()
        .expect("processed_image stage should include full base64 data");
    assert_eq!(data.encoding, "base64");
    assert_eq!(data.dtype, "uint8");
    BASE64
        .decode(&data.value)
        .expect("processed_image base64 should decode")
}

fn reference_f32(reference: &TensorSummary) -> Vec<f32> {
    let data = reference
        .data
        .as_ref()
        .expect("pixel_values output should include full base64 data");
    assert_eq!(data.encoding, "base64");
    assert_eq!(data.dtype, "float32");
    let bytes = BASE64
        .decode(&data.value)
        .expect("pixel_values base64 should decode");
    assert_eq!(
        bytes.len() % size_of::<f32>(),
        0,
        "float32 payload should be aligned"
    );
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: f64,
    mean: f64,
    max: f64,
}

fn summarize(values: &[f32]) -> Stats {
    assert!(!values.is_empty(), "tensor should not be empty");

    let mut min = f64::from(values[0]);
    let mut max = min;
    let mut sum = 0.0;

    for &value in values {
        let value = f64::from(value);
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }

    Stats {
        min,
        mean: sum / values.len() as f64,
        max,
    }
}

fn assert_sample_close(case: &ParityCase, values: &[f32], reference: &[f64]) {
    let family = &case.family;
    assert!(
        values.len() >= reference.len(),
        "{family} tensor shorter than fixture sample: got {}, need {}",
        values.len(),
        reference.len()
    );

    for (index, expected) in reference.iter().copied().enumerate() {
        assert_close(
            f64::from(values[index]),
            expected,
            case.comparison.float_full_values.absolute_tolerance,
            family,
            &format!("sample[{index}]"),
        );
    }
}

fn assert_byte_sample_close(family: &str, values: &[u8], reference: &[f64]) {
    assert!(
        values.len() >= reference.len(),
        "{family} processed_image shorter than fixture sample: got {}, need {}",
        values.len(),
        reference.len()
    );

    for (index, expected) in reference.iter().copied().enumerate() {
        let actual = f64::from(values[index]);
        assert_close(
            actual,
            expected,
            f64::from(PROCESSED_IMAGE_MAX_ABS_TOLERANCE),
            family,
            &format!("processed_image.sample[{index}]"),
        );
    }
}

fn assert_float_values_close(case: &ParityCase, actual: &[f32], expected: &[f32], shape: &[usize]) {
    let tolerance = case.comparison.float_full_values.absolute_tolerance;
    let diff = float_diff(actual, expected, tolerance);

    assert!(
        diff.max_abs <= tolerance,
        "{} pixel_values drift too high: {}",
        case.family,
        diff.describe(shape)
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, family: &str, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "{family} {label} differs from Transformers fixture: actual={actual:.8}, expected={expected:.8}, delta={delta:.8}, tolerance={tolerance:.8}"
    );
}

#[derive(Debug)]
struct ByteDiff {
    max_abs: u8,
    mean_abs: f64,
    first_diff: Option<ByteDifference>,
}

impl ByteDiff {
    fn describe(&self, shape: &[usize]) -> String {
        match self.first_diff {
            Some(first) => format!(
                "max_abs={}, mean_abs={:.6}, first_diff={} actual={} expected={}",
                self.max_abs,
                self.mean_abs,
                format_tensor_index(shape, first.index),
                first.actual,
                first.expected
            ),
            None => format!("max_abs={}, mean_abs={:.6}", self.max_abs, self.mean_abs),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ByteDifference {
    index: usize,
    actual: u8,
    expected: u8,
}

#[derive(Debug)]
struct FloatDiff {
    max_abs: f64,
    mean_abs: f64,
    first_diff: Option<FloatDifference>,
}

impl FloatDiff {
    fn describe(&self, shape: &[usize]) -> String {
        match self.first_diff {
            Some(first) => format!(
                "max_abs={:.8}, mean_abs={:.8}, first_diff={} actual={:.8} expected={:.8}",
                self.max_abs,
                self.mean_abs,
                format_tensor_index(shape, first.index),
                first.actual,
                first.expected
            ),
            None => format!("max_abs={:.8}, mean_abs={:.8}", self.max_abs, self.mean_abs),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FloatDifference {
    index: usize,
    actual: f64,
    expected: f64,
}

fn float_diff(actual: &[f32], expected: &[f32], tolerance: f64) -> FloatDiff {
    assert_eq!(
        actual.len(),
        expected.len(),
        "float diff requires equal-length tensors"
    );
    assert!(!actual.is_empty(), "float diff requires non-empty tensors");

    let mut max_abs = 0.0;
    let mut sum_abs = 0.0;
    let mut first_diff = None;

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let delta = (actual - expected).abs();
        max_abs = f64::max(max_abs, delta);
        sum_abs += delta;
        if delta > tolerance && first_diff.is_none() {
            first_diff = Some(FloatDifference {
                index,
                actual,
                expected,
            });
        }
    }

    FloatDiff {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        first_diff,
    }
}

fn byte_diff(actual: &[u8], expected: &[u8]) -> ByteDiff {
    assert_eq!(
        actual.len(),
        expected.len(),
        "byte diff requires equal-length tensors"
    );

    let mut max_abs = 0u8;
    let mut sum_abs = 0u64;
    let mut first_diff = None;

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let delta = actual.abs_diff(expected);
        max_abs = max_abs.max(delta);
        sum_abs += u64::from(delta);
        if delta != 0 && first_diff.is_none() {
            first_diff = Some(ByteDifference {
                index,
                actual,
                expected,
            });
        }
    }

    ByteDiff {
        max_abs,
        mean_abs: sum_abs as f64 / actual.len() as f64,
        first_diff,
    }
}

fn format_tensor_index(shape: &[usize], index: usize) -> String {
    if shape.len() != 4 {
        return format!("flat[{index}]");
    }

    let width = shape[3];
    let height = shape[2];
    let channels = shape[1];
    let per_image = channels * height * width;
    let n = index / per_image;
    let image_offset = index % per_image;
    let c = image_offset / (height * width);
    let channel_offset = image_offset % (height * width);
    let y = channel_offset / width;
    let x = channel_offset % width;
    format!("n={n}, c={c}, y={y}, x={x}")
}
