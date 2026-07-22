use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    Gemma3ImageProcessor, Gemma3ImageProcessorConfig, ImageFrame, ImageSize, Layout, PixelFormat,
    ProcessorMetadataName, ProcessorMetadataValue, ProcessorTensorName, ResizeFilter, ResizeParity,
    Tensor, TensorData,
};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/transformers_gemma3.json");
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
    #[serde(default)]
    images: Vec<FixtureImage>,
    metadata: Gemma3Metadata,
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
struct Gemma3Metadata {
    original_sizes: Vec<[usize; 2]>,
    reshaped_input_sizes: Vec<[usize; 2]>,
    num_crops: Vec<usize>,
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
fn gemma3_fixture_schema_matches_expected_version() {
    let fixture = load_fixture();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn gemma3_pan_and_scan_output_matches_transformers_fixture() {
    for case in fixture_cases() {
        let images = fixture_images(&case)
            .iter()
            .enumerate()
            .map(|(index, image)| deterministic_frame(image.width, image.height, index))
            .collect::<Vec<_>>();
        let processor = Gemma3ImageProcessor::new(Gemma3ImageProcessorConfig {
            size: ImageSize::new(5, 5).unwrap(),
            do_pan_and_scan: true,
            pan_and_scan_min_crop_size: 5,
            pan_and_scan_max_num_crops: 4,
            pan_and_scan_min_ratio_to_activate: 1.2,
            resample: ResizeFilter::Nearest,
            resize_parity: ResizeParity::PixelExact,
            ..Default::default()
        })
        .expect("small Gemma3 config should build");

        let output = processor
            .preprocess_images_output(&images)
            .expect("Gemma3 preprocessing should succeed");

        assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
        assert_eq!(
            output.tensors()[0].name(),
            &ProcessorTensorName::PixelValues,
            "{}",
            case.model_id
        );
        assert_pixel_values_match(
            &case,
            output.pixel_values().expect("missing pixel_values tensor"),
        );
        assert_metadata_matches(&case, &output);
    }
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("fixture JSON should parse")
}

fn fixture_cases() -> Vec<ParityCase> {
    let cases = load_fixture()
        .cases
        .into_iter()
        .filter(|case| case.family == "gemma3")
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 2, "fixture should cover both Gemma3 backends");
    for case in &cases {
        assert!(matches!(case.backend.as_str(), "pil" | "torchvision"));
        assert!(!case.class_name.is_empty());
        assert_eq!(case.image.mode, "RGB");
        for image in &case.images {
            assert_eq!(image.mode, "RGB");
        }
    }
    cases
}

fn fixture_images(case: &ParityCase) -> Vec<FixtureImage> {
    if case.images.is_empty() {
        vec![case.image.clone()]
    } else {
        case.images.clone()
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

fn assert_metadata_matches(case: &ParityCase, output: &image_processors::ProcessorOutput) {
    let metadata_names = output
        .metadata()
        .iter()
        .map(|metadata| metadata.name().as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        metadata_names,
        ["original_sizes", "reshaped_input_sizes", "num_crops"],
        "{}",
        case.model_id
    );

    assert_eq!(
        output.original_sizes().expect("missing original_sizes"),
        image_sizes_from_pairs(&case.metadata.original_sizes).as_slice(),
        "{}",
        case.model_id
    );
    assert_eq!(
        output
            .reshaped_input_sizes()
            .expect("missing reshaped_input_sizes"),
        image_sizes_from_pairs(&case.metadata.reshaped_input_sizes).as_slice(),
        "{}",
        case.model_id
    );

    let Some(ProcessorMetadataValue::Counts(num_crops)) =
        output.metadata_value(&ProcessorMetadataName::other("num_crops"))
    else {
        panic!("missing num_crops metadata");
    };
    assert_eq!(num_crops, &case.metadata.num_crops, "{}", case.model_id);
}

fn image_sizes_from_pairs(pairs: &[[usize; 2]]) -> Vec<ImageSize> {
    pairs
        .iter()
        .map(|[height, width]| {
            ImageSize::new(*height, *width).expect("fixture image size should be valid")
        })
        .collect()
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
    assert!(values.len() >= reference.len());
    for (index, expected) in reference.iter().copied().enumerate() {
        assert_close(
            f64::from(values[index]),
            expected,
            case.comparison.float_full_values.absolute_tolerance,
            &format!("sample[{index}]"),
        );
    }
}

fn assert_float_values_close(case: &ParityCase, actual: &[f32], expected: &[f32], shape: &[usize]) {
    assert_eq!(actual.len(), expected.len());
    let diff = float_diff(actual, expected);

    assert!(
        diff.max_abs <= case.comparison.float_full_values.absolute_tolerance,
        "Gemma3 pixel_values drift too high: {}",
        diff.describe(shape)
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "Gemma3 {label} differs from Transformers fixture: actual={actual:.8}, expected={expected:.8}, delta={delta:.8}, tolerance={tolerance:.8}"
    );
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
                "shape={shape:?}, max_abs={:.8}, mean_abs={:.8}, first_diff={} actual={:.8} expected={:.8}",
                self.max_abs, self.mean_abs, first.index, first.actual, first.expected
            ),
            None => format!(
                "shape={shape:?}, max_abs={:.8}, mean_abs={:.8}",
                self.max_abs, self.mean_abs
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FloatDifference {
    index: usize,
    actual: f32,
    expected: f32,
}

fn float_diff(actual: &[f32], expected: &[f32]) -> FloatDiff {
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0;
    let mut first_diff = None;

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let delta = f64::from((actual - expected).abs());
        if delta > 0.0 && first_diff.is_none() {
            first_diff = Some(FloatDifference {
                index,
                actual,
                expected,
            });
        }
        max_abs = max_abs.max(delta);
        sum_abs += delta;
    }

    FloatDiff {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        first_diff,
    }
}
