#[path = "common/video_parity.rs"]
mod video_parity;

use image_processors::{
    ImageSize, ResizeFilter, ResizeParity, VivitImageProcessor, VivitImageProcessorConfig,
};
use video_parity::VideoParityHarness;

const PARITY: VideoParityHarness = VideoParityHarness::new(
    "vivit",
    "ViViT",
    include_str!("fixtures/transformers_vivit.json"),
    include_str!("fixtures/transformers_vivit_batch.json"),
);

#[test]
fn vivit_fixture_schema_matches_expected_version() {
    PARITY.assert_fixture_schema();
}

#[test]
fn vivit_video_output_matches_transformers_fixture() {
    PARITY.assert_video_output(processor, VivitImageProcessor::preprocess_video_output);
}

#[test]
fn vivit_image_sequence_output_matches_transformers_fixture() {
    PARITY.assert_image_sequence_output(
        processor,
        VivitImageProcessor::preprocess_image_sequence_output,
    );
}

#[test]
fn vivit_batched_video_output_matches_transformers_fixture() {
    PARITY.assert_batched_video_output(processor, VivitImageProcessor::preprocess_videos_output);
}

fn processor(backend: &str) -> VivitImageProcessor {
    assert_eq!(backend, "pil", "ViViT has only a PIL fixture backend");
    VivitImageProcessor::new(VivitImageProcessorConfig {
        size: 4,
        crop_size: ImageSize::new(4, 4).expect("fixture crop size should be valid"),
        resample: ResizeFilter::Nearest,
        resize_parity: ResizeParity::Compatibility,
        ..Default::default()
    })
    .expect("ViViT config should build")
}
