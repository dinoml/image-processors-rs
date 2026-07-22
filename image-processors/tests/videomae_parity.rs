#[path = "common/video_parity.rs"]
mod video_parity;

use image_processors::{
    ImageSize, ResizeFilter, ResizeParity, VideoMaeImageProcessor, VideoMaeImageProcessorConfig,
};
use video_parity::VideoParityHarness;

const PARITY: VideoParityHarness = VideoParityHarness::new(
    "videomae",
    "VideoMAE",
    include_str!("fixtures/transformers_videomae.json"),
    include_str!("fixtures/transformers_videomae_batch.json"),
);

#[test]
fn videomae_fixture_schema_matches_expected_version() {
    PARITY.assert_fixture_schema();
}

#[test]
fn videomae_video_output_matches_transformers_fixture() {
    PARITY.assert_video_output(processor, VideoMaeImageProcessor::preprocess_video_output);
}

#[test]
fn videomae_image_sequence_output_matches_transformers_fixture() {
    PARITY.assert_image_sequence_output(
        processor,
        VideoMaeImageProcessor::preprocess_image_sequence_output,
    );
}

#[test]
fn videomae_batched_video_output_matches_transformers_fixture() {
    PARITY.assert_batched_video_output(processor, VideoMaeImageProcessor::preprocess_videos_output);
}

fn processor(backend: &str) -> VideoMaeImageProcessor {
    VideoMaeImageProcessor::new(VideoMaeImageProcessorConfig {
        size: 4,
        crop_size: ImageSize::new(4, 4).expect("fixture crop size should be valid"),
        resample: ResizeFilter::Nearest,
        resize_parity: resize_parity(backend),
        ..Default::default()
    })
    .expect("VideoMAE config should build")
}

fn resize_parity(backend: &str) -> ResizeParity {
    match backend {
        // The fixture pins nearest-neighbor sampling, which is byte-identical
        // across both upstream backends and uses the explicit exact path.
        "pil" | "torchvision" => ResizeParity::PixelExact,
        backend => panic!("unsupported VideoMAE backend: {backend}"),
    }
}
