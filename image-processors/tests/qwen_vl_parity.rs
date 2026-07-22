use std::{collections::BTreeMap, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    ImageFrame, ImageSize, Layout, PixelFormat, QwenVlImageProcessor, QwenVlImageProcessorConfig,
    ResizeParity, Tensor, TensorData, VideoClip, VideoFrame,
};
use serde::Deserialize;

const SINGLE_IMAGE_FIXTURE: &str = include_str!("fixtures/transformers_qwen_vl.json");
const MULTI_IMAGE_FIXTURE: &str = include_str!("fixtures/transformers_qwen_vl_multi_image.json");
const VIDEO_FIXTURE: &str = include_str!("fixtures/transformers_qwen_vl_video.json");
const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const PIXEL_VALUES: &str = "pixel_values";
const PIXEL_VALUES_VIDEOS: &str = "pixel_values_videos";
const IMAGE_GRID_THW: &str = "image_grid_thw";
const VIDEO_GRID_THW: &str = "video_grid_thw";

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
    #[serde(default)]
    video_frames: Vec<FixtureImage>,
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
fn qwen_vl_fixture_schema_matches_expected_version() {
    let fixture = load_fixture(SINGLE_IMAGE_FIXTURE);

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn qwen_vl_multi_image_fixture_schema_matches_expected_version() {
    let fixture = load_fixture(MULTI_IMAGE_FIXTURE);

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn qwen_vl_video_fixture_schema_matches_expected_version() {
    let fixture = load_fixture(VIDEO_FIXTURE);

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
}

#[test]
fn qwen_vl_single_image_output_matches_transformers_fixture() {
    for case in fixture_cases(SINGLE_IMAGE_FIXTURE) {
        let frame = deterministic_frame(case.image.width, case.image.height, 0);
        let processor = qwen_processor(&case);

        let output = processor
            .preprocess_image_output(&frame)
            .expect("Qwen/VLM preprocessing should succeed");

        assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
        assert_pixel_values_match(
            &case,
            output.pixel_values().expect("missing pixel_values tensor"),
        );
        assert_metadata_matches(
            &case,
            &output,
            processor.config().patch_size,
            std::slice::from_ref(&case.image),
        );
    }
}

#[test]
fn qwen_vl_multi_image_output_matches_transformers_fixture() {
    for case in fixture_cases(MULTI_IMAGE_FIXTURE) {
        let images = fixture_images(&case);
        let frames = images
            .iter()
            .enumerate()
            .map(|(index, image)| deterministic_frame(image.width, image.height, index))
            .collect::<Vec<_>>();
        let processor = qwen_processor(&case);

        let output = processor
            .preprocess_images_output(&frames)
            .expect("Qwen/VLM batch preprocessing should succeed");

        assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
        assert_pixel_values_match(
            &case,
            output.pixel_values().expect("missing pixel_values tensor"),
        );
        assert_metadata_matches(&case, &output, processor.config().patch_size, &images);
    }
}

#[test]
fn qwen_vl_video_output_matches_transformers_fixture() {
    let case = fixture_cases(VIDEO_FIXTURE)
        .pop()
        .expect("missing Qwen/VLM video fixture case");
    let video_frames = fixture_video_frames(&case);
    let frames = video_frames
        .iter()
        .enumerate()
        .map(|(index, image)| {
            VideoFrame::new(deterministic_frame(image.width, image.height, index))
        })
        .collect::<Vec<_>>();
    let video = VideoClip::new(frames, Some(24.0)).expect("fixture video should be valid");
    let processor = qwen_processor(&case);

    let output = processor
        .preprocess_video_output(&video)
        .expect("Qwen/VLM video preprocessing should succeed");

    assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
    assert_tensor_values_match(
        &case,
        output.pixel_values().expect("missing pixel_values tensor"),
        PIXEL_VALUES_VIDEOS,
    );
    assert_video_metadata_matches(&case, &output, processor.config().patch_size, &video_frames);
}

fn assert_metadata_matches(
    case: &ParityCase,
    output: &image_processors::ProcessorOutput,
    patch_size: usize,
    images: &[FixtureImage],
) {
    let expected_grid = reference_grid_thw(case, IMAGE_GRID_THW);
    assert_eq!(
        output.image_grid_thw().expect("missing image_grid_thw"),
        expected_grid.as_slice(),
        "{}",
        case.model_id
    );
    let original_sizes = images
        .iter()
        .map(|image| ImageSize {
            height: image.height,
            width: image.width,
        })
        .collect::<Vec<_>>();
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
        reshaped_sizes_for_grid(&expected_grid, patch_size).as_slice(),
        "{}",
        case.model_id
    );
}

fn assert_video_metadata_matches(
    case: &ParityCase,
    output: &image_processors::ProcessorOutput,
    patch_size: usize,
    frames: &[FixtureImage],
) {
    let expected_grid = reference_grid_thw(case, VIDEO_GRID_THW);
    assert_eq!(
        output
            .image_grid_thw()
            .expect("missing video grid metadata"),
        expected_grid.as_slice(),
        "{}",
        case.model_id
    );
    let original_sizes = frames
        .iter()
        .map(|frame| ImageSize {
            height: frame.height,
            width: frame.width,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        output.original_sizes().expect("missing original_sizes"),
        original_sizes.as_slice(),
        "{}",
        case.model_id
    );
    let target = ImageSize::new(
        expected_grid[0][1] * patch_size,
        expected_grid[0][2] * patch_size,
    )
    .expect("fixture video size should be valid");
    assert_eq!(
        output
            .reshaped_input_sizes()
            .expect("missing reshaped_input_sizes"),
        vec![target; frames.len()].as_slice(),
        "{}",
        case.model_id
    );
}

fn load_fixture(contents: &str) -> Fixture {
    serde_json::from_str(contents).expect("fixture JSON should parse")
}

fn fixture_cases(contents: &str) -> Vec<ParityCase> {
    let cases = load_fixture(contents)
        .cases
        .into_iter()
        .filter(|case| case.family == "qwen_vl")
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "missing Qwen/VLM fixture case");
    for case in &cases {
        assert_eq!(case.image.mode, "RGB");
        for image in &case.images {
            assert_eq!(image.mode, "RGB");
        }
        for frame in &case.video_frames {
            assert_eq!(frame.mode, "RGB");
        }
    }
    cases
}

fn qwen_processor(case: &ParityCase) -> QwenVlImageProcessor {
    let config = QwenVlImageProcessorConfig {
        resize_parity: match case.backend.as_str() {
            "pil" => ResizeParity::Compatibility,
            "torchvision" => ResizeParity::Torchvision,
            backend => panic!("unsupported {} backend: {backend}", case.class_name),
        },
        ..Default::default()
    };
    QwenVlImageProcessor::new(config).expect("fixture Qwen/VLM config should build")
}

fn fixture_images(case: &ParityCase) -> Vec<FixtureImage> {
    if case.images.is_empty() {
        vec![case.image.clone()]
    } else {
        case.images.clone()
    }
}

fn fixture_video_frames(case: &ParityCase) -> Vec<FixtureImage> {
    assert!(
        !case.video_frames.is_empty(),
        "video fixture should contain video_frames"
    );
    case.video_frames.clone()
}

fn deterministic_frame(width: usize, height: usize, offset: usize) -> ImageFrame {
    let values = (0..width * height * 3)
        .map(|index| ((index as u32 * 37 + 17 + offset as u32 * 53) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Rgb8, values)
        .expect("deterministic fixture image should be valid")
}

fn assert_pixel_values_match(case: &ParityCase, tensor: &Tensor) {
    assert_tensor_values_match(case, tensor, PIXEL_VALUES);
}

fn assert_tensor_values_match(case: &ParityCase, tensor: &Tensor, output_name: &str) {
    let reference = case
        .outputs
        .get(output_name)
        .unwrap_or_else(|| panic!("fixture missing {output_name}"));
    let values = tensor.data().to_vec::<f32>();
    let observed = summarize(&values);

    assert_eq!(
        tensor.shape(),
        reference.shape.as_slice(),
        "{}",
        case.model_id
    );
    assert_eq!(tensor.layout(), Layout::NC, "{}", case.model_id);
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

fn reference_grid_thw(case: &ParityCase, output_name: &str) -> Vec<[usize; 3]> {
    let reference = case
        .outputs
        .get(output_name)
        .unwrap_or_else(|| panic!("fixture missing {output_name}"));
    assert_eq!(reference.dtype, "int64");
    assert_eq!(reference.shape.len(), 2);
    assert_eq!(reference.shape[1], 3);
    reference_i64(reference)
        .chunks_exact(3)
        .map(|chunk| {
            [
                usize::try_from(chunk[0]).expect("temporal grid should fit usize"),
                usize::try_from(chunk[1]).expect("height grid should fit usize"),
                usize::try_from(chunk[2]).expect("width grid should fit usize"),
            ]
        })
        .collect()
}

fn reshaped_sizes_for_grid(grid: &[[usize; 3]], patch_size: usize) -> Vec<ImageSize> {
    grid.iter()
        .map(|entry| {
            ImageSize::new(entry[1] * patch_size, entry[2] * patch_size)
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
        "Qwen/VLM pixel_values drift too high: {}",
        diff.describe(shape)
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "Qwen/VLM {label} differs from Transformers fixture: actual={actual:.8}, expected={expected:.8}, delta={delta:.8}, tolerance={tolerance:.8}"
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
    actual: f64,
    expected: f64,
}

fn float_diff(actual: &[f32], expected: &[f32], tolerance: f64) -> FloatDiff {
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
