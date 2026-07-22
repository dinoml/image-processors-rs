use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    ImageFrame, ImageLayout, ImageSize, Layout, LlavaNextImageProcessor,
    LlavaNextImageProcessorConfig, PixelFormat, ProcessorMetadataName, ProcessorMetadataValue,
    ProcessorTensorName, ResizeFilter, ResizeParity, Tensor, TensorData,
};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/transformers_llava_next.json");
const NHWC_FIXTURE: &str = include_str!("fixtures/transformers_llava_next_nhwc.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const PIXEL_VALUES: &str = "pixel_values";
const IMAGE_SIZES: &str = "image_sizes";
const SELECTED_SIZES: &str = "selected_sizes";

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
    metadata: LlavaNextMetadata,
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
struct LlavaNextMetadata {
    image_sizes: Vec<[usize; 2]>,
    reshaped_input_sizes: Vec<[usize; 2]>,
    selected_sizes: Vec<[usize; 2]>,
    image_patch_counts: Vec<usize>,
    patch_grids: Vec<[usize; 2]>,
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
fn llava_next_fixture_schema_matches_expected_version() {
    let fixture = load_fixture(FIXTURE);
    let nhwc_fixture = load_fixture(NHWC_FIXTURE);

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
    assert_eq!(nhwc_fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn llava_next_batch_output_matches_transformers_fixture() {
    for case in fixture_cases(FIXTURE, "llava_next") {
        let images = fixture_images(&case);
        let frames = images
            .iter()
            .enumerate()
            .map(|(index, image)| deterministic_frame(image.width, image.height, index))
            .collect::<Vec<_>>();
        let config = LlavaNextImageProcessorConfig {
            resize_parity: fixture_resize_parity(&case),
            ..Default::default()
        };
        let processor =
            LlavaNextImageProcessor::new(config).expect("fixture LLaVA-NeXT config should build");

        let output = processor
            .preprocess_images_output(&frames)
            .expect("LLaVA-NeXT preprocessing should succeed");

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
            Layout::NPCHW,
        );
        assert_metadata_matches(&case, &output);
    }
}

#[test]
fn llava_next_custom_grid_output_matches_transformers_fixture() {
    for case in fixture_cases(NHWC_FIXTURE, "llava_next_nhwc") {
        let images = fixture_images(&case);
        let frames = images
            .iter()
            .enumerate()
            .map(|(index, image)| deterministic_frame(image.width, image.height, index))
            .collect::<Vec<_>>();
        let processor = LlavaNextImageProcessor::new(LlavaNextImageProcessorConfig {
            size: ImageSize::new(4, 4).unwrap(),
            crop_size: ImageSize::new(4, 4).unwrap(),
            image_grid_pinpoints: vec![
                ImageSize::new(4, 4).unwrap(),
                ImageSize::new(4, 8).unwrap(),
                ImageSize::new(8, 4).unwrap(),
                ImageSize::new(8, 8).unwrap(),
            ],
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Nearest,
            resize_parity: ResizeParity::PixelExact,
            ..Default::default()
        })
        .expect("custom LLaVA-NeXT config should build");

        let output = processor
            .preprocess_images_output(&frames)
            .expect("LLaVA-NeXT preprocessing should succeed");

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
            Layout::NPCHW,
        );
        assert_metadata_matches(&case, &output);
    }
}

fn load_fixture(input: &str) -> Fixture {
    serde_json::from_str(input).expect("fixture JSON should parse")
}

fn fixture_cases(input: &str, family: &str) -> Vec<ParityCase> {
    let cases = load_fixture(input)
        .cases
        .into_iter()
        .filter(|case| case.family == family)
        .collect::<Vec<_>>();
    assert_eq!(
        cases.len(),
        2,
        "fixture should cover both LLaVA-NeXT backends"
    );
    for case in &cases {
        assert_eq!(case.image.mode, "RGB");
        for image in &case.images {
            assert_eq!(image.mode, "RGB");
        }
    }
    cases
}

fn fixture_resize_parity(case: &ParityCase) -> ResizeParity {
    match case.backend.as_str() {
        "pil" => ResizeParity::Compatibility,
        "torchvision" => ResizeParity::Torchvision,
        backend => panic!("unsupported {} backend: {backend}", case.class_name),
    }
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

fn assert_pixel_values_match(case: &ParityCase, tensor: &Tensor, expected_layout: Layout) {
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
    assert_eq!(tensor.layout(), expected_layout, "{}", case.model_id);
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
        [
            "original_sizes",
            "reshaped_input_sizes",
            "image_patch_counts",
            SELECTED_SIZES
        ],
        "{}",
        case.model_id
    );

    let upstream_image_sizes = reference_image_sizes(case, IMAGE_SIZES);
    let original_sizes = image_sizes_from_pairs(&case.metadata.image_sizes);
    assert_eq!(upstream_image_sizes, original_sizes, "{}", case.model_id);
    assert_eq!(
        output.original_sizes().expect("missing original_sizes"),
        original_sizes.as_slice(),
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

    let Some(ProcessorMetadataValue::ImageSizes(selected_sizes)) =
        output.metadata_value(&ProcessorMetadataName::other(SELECTED_SIZES))
    else {
        panic!("missing selected_sizes metadata");
    };
    assert_eq!(
        selected_sizes,
        &image_sizes_from_pairs(&case.metadata.selected_sizes),
        "{}",
        case.model_id
    );

    let patch_counts = case
        .metadata
        .patch_grids
        .iter()
        .map(|grid| 1 + grid[0] * grid[1])
        .collect::<Vec<_>>();
    assert_eq!(
        patch_counts, case.metadata.image_patch_counts,
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

fn reference_image_sizes(case: &ParityCase, output_name: &str) -> Vec<ImageSize> {
    let reference = case
        .outputs
        .get(output_name)
        .unwrap_or_else(|| panic!("fixture missing {output_name}"));
    assert_eq!(reference.dtype, "int64");
    assert_eq!(reference.shape.len(), 2);
    assert_eq!(reference.shape[1], 2);
    reference_i64(reference)
        .chunks_exact(2)
        .map(|chunk| {
            ImageSize::new(
                usize::try_from(chunk[0]).expect("height should fit usize"),
                usize::try_from(chunk[1]).expect("width should fit usize"),
            )
            .expect("fixture image size should be valid")
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

fn reference_i64(reference: &TensorSummary) -> Vec<i64> {
    assert_eq!(reference.data.encoding, "base64");
    assert_eq!(reference.data.dtype, "int64");
    let bytes = BASE64
        .decode(&reference.data.value)
        .expect("int64 base64 should decode");
    assert_eq!(bytes.len() % size_of::<i64>(), 0);
    bytes
        .chunks_exact(size_of::<i64>())
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunk length is 8")))
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
    let tolerance = case.comparison.float_full_values.absolute_tolerance;
    let diff = float_diff(actual, expected, tolerance);

    assert!(
        diff.max_abs <= tolerance,
        "LLaVA-NeXT pixel_values drift too high: {}",
        diff.describe(shape)
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "LLaVA-NeXT {label} differs from Transformers fixture: actual={actual:.8}, expected={expected:.8}, delta={delta:.8}, tolerance={tolerance:.8}"
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

fn float_diff(actual: &[f32], expected: &[f32], tolerance: f64) -> FloatDiff {
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0;
    let mut sum_abs = 0.0;
    let mut first_diff = None;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let abs = f64::from((actual - expected).abs());
        if abs > max_abs {
            max_abs = abs;
        }
        if first_diff.is_none() && abs > tolerance {
            first_diff = Some(FloatDifference {
                index,
                actual,
                expected,
            });
        }
        sum_abs += abs;
    }

    FloatDiff {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        first_diff,
    }
}
