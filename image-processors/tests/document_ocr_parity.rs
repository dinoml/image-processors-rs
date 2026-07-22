use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    DocumentOcrImageProcessor, DocumentOcrImageProcessorConfig, ImageFrame, ImageSize, Layout,
    PixelFormat, ResizeParity, Tensor, TensorData,
};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/transformers_donut.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const PIXEL_VALUES: &str = "pixel_values";

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
    images: Vec<FixtureImage>,
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

#[derive(Clone, Debug, Deserialize)]
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
    data: TensorDataBlob,
}

#[derive(Debug, Deserialize)]
struct TensorDataBlob {
    encoding: String,
    dtype: String,
    value: String,
}

#[test]
fn document_ocr_fixture_schema_matches_expected_version() {
    let fixture = load_fixture();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn donut_output_matches_transformers_fixture() {
    for case in fixture_cases() {
        let frames = case
            .images
            .iter()
            .enumerate()
            .map(|(index, image)| deterministic_frame(image.width, image.height, index))
            .collect::<Vec<_>>();
        let processor = DocumentOcrImageProcessor::new(DocumentOcrImageProcessorConfig {
            image_size: ImageSize::new(12, 8).unwrap(),
            resize_parity: fixture_resize_parity(&case),
            ..Default::default()
        })
        .expect("Donut document/OCR config should build");

        let tensor = processor
            .preprocess_images(&frames)
            .expect("Donut preprocessing should succeed");

        assert_pixel_values_match(&case, &tensor);
    }
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("fixture JSON should parse")
}

fn fixture_cases() -> Vec<ParityCase> {
    let cases = load_fixture()
        .cases
        .into_iter()
        .filter(|case| case.family == "donut")
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 2, "fixture should cover both Donut backends");
    assert!(cases.iter().all(|case| case.image.mode == "RGB"));
    cases
}

fn fixture_resize_parity(case: &ParityCase) -> ResizeParity {
    match case.backend.as_str() {
        "pil" => ResizeParity::Compatibility,
        "torchvision" => ResizeParity::Torchvision,
        backend => panic!("unsupported {} backend: {backend}", case.class_name),
    }
}

fn deterministic_frame(width: usize, height: usize, offset: usize) -> ImageFrame {
    let values = (0..width * height * 3)
        .map(|index| ((index as u32 * 37 + 17 + offset as u32 * 53) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Rgb8, values)
        .expect("deterministic fixture image should be valid")
}

fn assert_pixel_values_match(case: &ParityCase, tensor: &Tensor) {
    let reference = case
        .outputs
        .get(PIXEL_VALUES)
        .expect("fixture missing pixel_values");
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
    let stats_tolerance = case.comparison.float_statistics.absolute_tolerance;
    assert_close(observed.min, reference.min, stats_tolerance, "min");
    assert_close(observed.mean, reference.mean, stats_tolerance, "mean");
    assert_close(observed.max, reference.max, stats_tolerance, "max");

    let expected = reference_f32(reference);
    assert_eq!(
        values.len(),
        expected.len(),
        "{} pixel_values",
        case.model_id
    );
    assert_float_values_close(case, &values, &expected, &reference.shape);
    assert_sample_close(case, &values, &reference.sample);
}

fn reference_f32(reference: &TensorSummary) -> Vec<f32> {
    assert_eq!(reference.data.encoding, "base64");
    assert_eq!(reference.data.dtype, "float32");
    let bytes = BASE64
        .decode(&reference.data.value)
        .expect("float32 base64 should decode");
    assert_eq!(bytes.len() % size_of::<f32>(), 0);
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
        .collect()
}

fn summarize(values: &[f32]) -> TensorStats {
    assert!(!values.is_empty());
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values {
        let value = f64::from(*value);
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    TensorStats {
        min,
        mean: sum / values.len() as f64,
        max,
    }
}

fn assert_float_values_close(case: &ParityCase, actual: &[f32], expected: &[f32], shape: &[usize]) {
    let tolerance = case.comparison.float_full_values.absolute_tolerance;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (f64::from(*actual) - f64::from(*expected)).abs();
        assert!(
            diff <= tolerance,
            "value mismatch at flat index {index} shape {shape:?}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

fn assert_sample_close(case: &ParityCase, actual: &[f32], expected: &[f64]) {
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(
            f64::from(*actual),
            *expected,
            case.comparison.float_full_values.absolute_tolerance,
            &format!("sample[{index}]"),
        );
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{label}: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
    );
}

#[derive(Debug)]
struct TensorStats {
    min: f64,
    mean: f64,
    max: f64,
}
