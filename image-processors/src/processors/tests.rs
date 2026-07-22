use ::image::codecs::png::PngEncoder;
use ::image::{ColorType, ImageEncoder};

use super::*;
use crate::transforms::ResizeKernel;

fn expected_pixel_mask(sizes: &[ImageSize], target: ImageSize) -> Vec<bool> {
    let mut values = Vec::with_capacity(sizes.len() * target.height * target.width);
    for size in sizes {
        for row in 0..target.height {
            for column in 0..target.width {
                values.push(row < size.height && column < size.width);
            }
        }
    }
    values
}

fn assert_bbox_close(
    actual: crate::transforms::DetectionBoundingBox,
    expected: crate::transforms::DetectionBoundingBox,
) {
    assert!(
        (actual.x_min - expected.x_min).abs() < 1e-6,
        "x_min mismatch: actual={} expected={}",
        actual.x_min,
        expected.x_min
    );
    assert!(
        (actual.y_min - expected.y_min).abs() < 1e-6,
        "y_min mismatch: actual={} expected={}",
        actual.y_min,
        expected.y_min
    );
    assert!(
        (actual.x_max - expected.x_max).abs() < 1e-6,
        "x_max mismatch: actual={} expected={}",
        actual.x_max,
        expected.x_max
    );
    assert!(
        (actual.y_max - expected.y_max).abs() < 1e-6,
        "y_max mismatch: actual={} expected={}",
        actual.y_max,
        expected.y_max
    );
}

fn qwen_expected_patch(r: [f32; 4], g: [f32; 4], b: [f32; 4]) -> Vec<f32> {
    let mut values = Vec::with_capacity(24);
    for channel in [r, g, b] {
        values.extend_from_slice(&channel);
        values.extend_from_slice(&channel);
    }
    values
}

fn qwen_expected_temporal_patch(first: [[f32; 4]; 3], second: [[f32; 4]; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(24);
    for (first_channel, second_channel) in first.into_iter().zip(second) {
        values.extend_from_slice(&first_channel);
        values.extend_from_slice(&second_channel);
    }
    values
}

fn qwen_small_patch_processor() -> QwenVlImageProcessor {
    QwenVlImageProcessor::new(QwenVlImageProcessorConfig {
        resize_limits: ResizeLimits {
            factor: 2,
            min_pixels: 4,
            max_pixels: 8,
        },
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 1,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

fn llava_next_small_patch_processor() -> LlavaNextImageProcessor {
    LlavaNextImageProcessor::new(LlavaNextImageProcessorConfig {
        size: square_size(2),
        crop_size: square_size(2),
        image_grid_pinpoints: vec![
            ImageSize {
                height: 2,
                width: 2,
            },
            ImageSize {
                height: 2,
                width: 4,
            },
        ],
        resample: ResizeFilter::Nearest,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

fn pixtral_small_patch_processor() -> PixtralImageProcessor {
    PixtralImageProcessor::new(PixtralImageProcessorConfig {
        max_size: square_size(4),
        patch_size: square_size(2),
        resample: ResizeFilter::Nearest,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

fn idefics3_small_split_processor() -> Idefics3ImageProcessor {
    Idefics3ImageProcessor::new(Idefics3ImageProcessorConfig {
        longest_edge: 10,
        max_image_size: 5,
        resample: ResizeFilter::Nearest,
        resize_parity: ResizeParity::PixelExact,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

fn gemma3_small_pan_and_scan_processor() -> Gemma3ImageProcessor {
    Gemma3ImageProcessor::new(Gemma3ImageProcessorConfig {
        size: square_size(2),
        do_pan_and_scan: true,
        pan_and_scan_min_crop_size: 2,
        pan_and_scan_max_num_crops: 4,
        pan_and_scan_min_ratio_to_activate: 1.1,
        resample: ResizeFilter::Nearest,
        resize_parity: ResizeParity::PixelExact,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

fn mllama_small_tile_processor() -> MllamaImageProcessor {
    MllamaImageProcessor::new(MllamaImageProcessorConfig {
        tile_size: 2,
        max_image_tiles: 4,
        resample: ResizeFilter::Nearest,
        resize_parity: ResizeParity::PixelExact,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap()
}

mod diffusion_video;
mod family_configs;
mod hf_config;
mod outputs;
mod recipes;
mod vlm;

fn write_test_png(name: &str, pixel: &[u8; 3]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(pixel, 1, 1, ColorType::Rgb8.into())
        .unwrap();
    std::fs::write(&path, png).unwrap();
    path
}

fn video_clip_from_rgb_pixels(pixels: &[[u8; 3]]) -> VideoClip {
    let frames = pixels
        .iter()
        .map(|pixel| {
            let image = ImageFrame::new(1, 1, PixelFormat::Rgb8, pixel.to_vec()).unwrap();
            crate::media::VideoFrame::new(image)
        })
        .collect();
    VideoClip::new(frames, Some(24.0)).unwrap()
}

fn assert_f32_values_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "value mismatch at {index}: actual={actual} expected={expected}"
        );
    }
}
