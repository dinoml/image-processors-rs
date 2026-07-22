use std::{collections::BTreeMap, fmt::Display, mem::size_of};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image_processors::{
    ImageFrame, ImageSequence, Layout, LoopBehavior, PixelFormat, ProcessorOutput, Tensor,
    TensorData, TensorLeadingAxis, VideoClip, VideoFrame,
};
use serde::Deserialize;

const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const PIXEL_VALUES: &str = "pixel_values";

pub(super) struct VideoParityHarness {
    family: &'static str,
    display_name: &'static str,
    fixture: &'static str,
    batch_fixture: &'static str,
}

impl VideoParityHarness {
    pub(super) const fn new(
        family: &'static str,
        display_name: &'static str,
        fixture: &'static str,
        batch_fixture: &'static str,
    ) -> Self {
        Self {
            family,
            display_name,
            fixture,
            batch_fixture,
        }
    }

    pub(super) fn assert_fixture_schema(&self) {
        assert_eq!(load_fixture(self.fixture).schema, FIXTURE_SCHEMA);
    }

    pub(super) fn assert_video_output<P, E>(
        &self,
        processor: impl Fn(&str) -> P,
        preprocess: impl Fn(&P, &VideoClip) -> Result<ProcessorOutput, E>,
    ) where
        E: Display,
    {
        for case in self.fixture_cases(self.fixture) {
            let video = deterministic_video_clip(&case.video_frames, 0);
            let processor = processor(&case.backend);
            let output = preprocess(&processor, &video).unwrap_or_else(|error| {
                panic!(
                    "{} preprocessing should succeed: {error}",
                    self.display_name
                )
            });

            assert_output_matches(&case, &output);
        }
    }

    pub(super) fn assert_image_sequence_output<P, E>(
        &self,
        processor: impl Fn(&str) -> P,
        preprocess: impl Fn(&P, &ImageSequence) -> Result<ProcessorOutput, E>,
    ) where
        E: Display,
    {
        for case in self.fixture_cases(self.fixture) {
            let sequence = deterministic_image_sequence(&case.video_frames, 0);
            let processor = processor(&case.backend);
            let output = preprocess(&processor, &sequence).unwrap_or_else(|error| {
                panic!(
                    "{} image sequence preprocessing should succeed: {error}",
                    self.display_name
                )
            });

            assert_output_matches(&case, &output);
        }
    }

    pub(super) fn assert_batched_video_output<P, E>(
        &self,
        processor: impl Fn(&str) -> P,
        preprocess: impl Fn(&P, &[VideoClip]) -> Result<ProcessorOutput, E>,
    ) where
        E: Display,
    {
        for case in self.fixture_cases(self.batch_fixture) {
            assert_eq!(case.video_batch_size, Some(case.videos.len()));
            assert!(!case.videos.is_empty());
            for frames in &case.videos {
                assert_rgb_frames(frames);
            }

            let videos = case
                .videos
                .iter()
                .enumerate()
                .map(|(batch_index, frames)| deterministic_video_clip(frames, batch_index))
                .collect::<Vec<_>>();
            let processor = processor(&case.backend);
            let output = preprocess(&processor, &videos).unwrap_or_else(|error| {
                panic!(
                    "{} batch preprocessing should succeed: {error}",
                    self.display_name
                )
            });

            assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
            assert_batched_pixel_values_match(
                &case,
                output.pixel_values().expect("missing pixel_values tensor"),
            );
        }
    }

    fn fixture_cases(&self, contents: &str) -> Vec<ParityCase> {
        let fixture = load_fixture(contents);
        assert_eq!(fixture.schema, FIXTURE_SCHEMA);
        let cases = fixture
            .cases
            .into_iter()
            .filter(|case| case.family == self.family)
            .collect::<Vec<_>>();
        assert!(
            !cases.is_empty(),
            "missing {} fixture case",
            self.display_name
        );
        for case in &cases {
            assert_rgb_frames(&case.video_frames);
        }
        cases
    }
}

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    cases: Vec<ParityCase>,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    family: String,
    backend: String,
    model_id: String,
    video_frames: Vec<FixtureImage>,
    #[serde(default)]
    videos: Vec<Vec<FixtureImage>>,
    video_batch_size: Option<usize>,
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

fn load_fixture(contents: &str) -> Fixture {
    serde_json::from_str(contents).expect("fixture JSON should parse")
}

fn assert_rgb_frames(frames: &[FixtureImage]) {
    assert!(!frames.is_empty());
    for frame in frames {
        assert_eq!(frame.mode, "RGB");
    }
}

fn deterministic_video_clip(frames: &[FixtureImage], batch_index: usize) -> VideoClip {
    let frames = frames
        .iter()
        .enumerate()
        .map(|(index, image)| {
            let offset = batch_index * frames.len() + index;
            VideoFrame::new(deterministic_frame(image.width, image.height, offset))
        })
        .collect::<Vec<_>>();
    VideoClip::new(frames, Some(24.0)).expect("fixture video should be valid")
}

fn deterministic_image_sequence(frames: &[FixtureImage], batch_index: usize) -> ImageSequence {
    let frames = frames
        .iter()
        .enumerate()
        .map(|(index, image)| {
            let offset = batch_index * frames.len() + index;
            deterministic_frame(image.width, image.height, offset)
        })
        .collect::<Vec<_>>();
    ImageSequence::new(frames, LoopBehavior::Once).expect("fixture sequence should be valid")
}

fn deterministic_frame(width: usize, height: usize, offset: usize) -> ImageFrame {
    let values = (0..width * height * 3)
        .map(|index| ((index as u32 * 37 + 17 + offset as u32 * 53) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Rgb8, values)
        .expect("deterministic fixture image should be valid")
}

fn assert_output_matches(case: &ParityCase, output: &ProcessorOutput) {
    assert_eq!(output.tensors().len(), 1, "{}", case.model_id);
    assert_pixel_values_match(
        case,
        output.pixel_values().expect("missing pixel_values tensor"),
    );
}

fn assert_pixel_values_match(case: &ParityCase, tensor: &Tensor) {
    let reference = case
        .outputs
        .get(PIXEL_VALUES)
        .expect("fixture missing pixel_values");
    assert_pixel_values_match_with(
        case,
        tensor,
        reference,
        &reference.shape[1..],
        Layout::NCHW,
        Some(TensorLeadingAxis::Frames),
    );
}

fn assert_batched_pixel_values_match(case: &ParityCase, tensor: &Tensor) {
    let reference = case
        .outputs
        .get(PIXEL_VALUES)
        .expect("fixture missing pixel_values");
    assert_pixel_values_match_with(
        case,
        tensor,
        reference,
        &reference.shape,
        Layout::BFCHW,
        Some(TensorLeadingAxis::Batch),
    );
    assert_eq!(tensor.batch(), Some(case.videos.len()), "{}", case.model_id);
    assert_eq!(
        tensor.frames(),
        Some(case.videos[0].len()),
        "{}",
        case.model_id
    );
}

fn assert_pixel_values_match_with(
    case: &ParityCase,
    tensor: &Tensor,
    reference: &TensorSummary,
    expected_shape: &[usize],
    expected_layout: Layout,
    expected_leading_axis: Option<TensorLeadingAxis>,
) {
    let values = tensor.data().to_vec::<f32>();
    let observed = summarize(&values);

    assert_eq!(tensor.shape(), expected_shape, "{}", case.model_id);
    assert_eq!(tensor.layout(), expected_layout, "{}", case.model_id);
    assert_eq!(
        tensor.leading_axis(),
        expected_leading_axis,
        "{}",
        case.model_id
    );
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
    assert_float_values_close(case, &values, &expected, expected_shape);
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
