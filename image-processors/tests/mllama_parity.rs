use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    ImageFrame, ImageSize, Layout, MllamaImageProcessor, MllamaImageProcessorConfig, PixelFormat,
    ProcessorMetadataName, ProcessorMetadataValue, ProcessorTensorName, ResizeFilter, ResizeParity,
    Tensor, TensorData,
};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/transformers_mllama.json");
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
    metadata: MllamaMetadata,
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
struct MllamaMetadata {
    original_sizes: Vec<[usize; 2]>,
    reshaped_input_sizes: Vec<[usize; 2]>,
    canvas_sizes: Vec<[usize; 2]>,
    image_patch_counts: Vec<usize>,
    num_tiles: Vec<Vec<usize>>,
    aspect_ratio_ids: Vec<Vec<usize>>,
    aspect_ratio_mask: Vec<Vec<Vec<bool>>>,
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
fn mllama_fixture_schema_matches_expected_version() {
    let fixture = load_fixture();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn mllama_nested_batch_output_matches_transformers_fixture() {
    for case in fixture_cases() {
        let images = fixture_images(&case);
        let samples = images
            .iter()
            .enumerate()
            .map(|(index, image)| vec![deterministic_frame(image.width, image.height, index)])
            .collect::<Vec<_>>();
        let processor = MllamaImageProcessor::new(MllamaImageProcessorConfig {
            tile_size: 5,
            max_image_tiles: 4,
            resample: ResizeFilter::Nearest,
            resize_parity: ResizeParity::PixelExact,
            ..Default::default()
        })
        .expect("small Mllama config should build");

        let output = processor
            .preprocess_image_samples_output(&samples)
            .expect("Mllama preprocessing should succeed");

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
        .filter(|case| case.family == "mllama")
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 2, "fixture should cover both Mllama backends");
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
    assert_eq!(tensor.layout(), Layout::NIPCHW, "{}", case.model_id);
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
    assert_float_values_close(case, &values, &expected);
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
        [
            "original_sizes",
            "reshaped_input_sizes",
            "image_patch_counts",
            "canvas_sizes",
            "num_tiles",
            "aspect_ratio_ids",
            "aspect_ratio_mask"
        ],
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
    assert_eq!(
        output
            .image_patch_counts()
            .expect("missing image_patch_counts"),
        case.metadata.image_patch_counts.as_slice(),
        "{}",
        case.model_id
    );

    let Some(ProcessorMetadataValue::ImageSizes(canvas_sizes)) =
        output.metadata_value(&ProcessorMetadataName::other("canvas_sizes"))
    else {
        panic!("missing canvas_sizes metadata");
    };
    assert_eq!(
        canvas_sizes,
        &image_sizes_from_pairs(&case.metadata.canvas_sizes),
        "{}",
        case.model_id
    );

    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("num_tiles")),
        Some(case.metadata.num_tiles.as_slice()),
        "{}",
        case.model_id
    );
    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("aspect_ratio_ids")),
        Some(case.metadata.aspect_ratio_ids.as_slice()),
        "{}",
        case.model_id
    );
    assert_eq!(
        output.nested_bool_mask(&ProcessorMetadataName::other("aspect_ratio_mask")),
        Some(case.metadata.aspect_ratio_mask.as_slice()),
        "{}",
        case.model_id
    );
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
            "sample",
        );
    }
}

fn assert_float_values_close(case: &ParityCase, values: &[f32], expected: &[f32]) {
    assert_eq!(values.len(), expected.len());
    for (index, (observed, expected)) in values.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(*observed) - f64::from(*expected)).abs()
                <= case.comparison.float_full_values.absolute_tolerance,
            "value mismatch at flat index {index}: observed {observed}, expected {expected}"
        );
    }
}

fn assert_close(observed: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "{label} mismatch: observed {observed}, expected {expected}, tolerance {tolerance}"
    );
}
