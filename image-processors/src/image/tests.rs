use ::image::codecs::png::PngEncoder;
use ::image::{ColorType, ImageEncoder};

use super::*;

#[test]
fn open_loads_file_as_main_entrypoint() {
    let path = std::env::temp_dir().join("image_processors_open_loads_file.png");
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&[0, 127, 255], 1, 1, ColorType::Rgb8.into())
        .unwrap();
    std::fs::write(&path, png).unwrap();

    let processor = ImageProcessor::default();
    let tensor = processor.open(&path).unwrap();

    assert_eq!(tensor.shape(), [1, 3, 1, 1]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    let values = tensor.data().to_vec::<f32>();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 127.0_f32 / 255.0).abs() < 1e-6);
    assert_eq!(values[2], 1.0);

    let _ = std::fs::remove_file(path);
}

#[test]
fn preprocess_resizes_converts_and_normalizes() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![128]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(2),
        width: Some(2),
        pixel_format: Some(PixelFormat::Rgb8),
        do_normalize: true,
        image_mean: vec![0.5],
        image_std: vec![0.5],
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();

    assert_eq!(tensor.shape(), [1, 3, 2, 2]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert!(tensor
        .data()
        .to_vec::<f32>()
        .iter()
        .all(|value| (*value - 0.003921628).abs() < 1e-6));
}

#[test]
fn preprocess_batches_images() {
    let image = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();
    let processor = ImageProcessor::default();

    let tensor = processor
        .preprocess_images(&[image.clone(), image])
        .unwrap();

    assert_eq!(tensor.shape(), [2, 3, 1, 1]);
}

#[test]
fn preprocess_image_output_wraps_pixel_values() {
    let image = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();
    let processor = ImageProcessor::default();

    let output = processor.preprocess_image_output(&image).unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [1, 3, 1, 1]);
}

#[test]
fn open_batch_into_uses_recycled_workspace_buffer() {
    let first = write_test_png("image_processors_workspace_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_workspace_second.png", &[0, 255, 0]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        batch_execution: BatchExecution::Serial,
        ..Default::default()
    })
    .unwrap();
    let mut workspace = ImageProcessorWorkspace::new();

    let tensor = processor
        .open_batch_into(&[&first, &second], &mut workspace)
        .unwrap();
    workspace.recycle_tensor(tensor).unwrap();
    let capacity_after_recycle = workspace.f32_capacity();
    let tensor = processor
        .open_batch_into(&[&first, &second], &mut workspace)
        .unwrap();

    assert_eq!(tensor.shape(), [2, 3, 1, 1]);
    assert!(capacity_after_recycle >= tensor.data().len());

    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[test]
fn open_batch_into_reuses_resize_coefficients() {
    let first = write_test_png("image_processors_resize_workspace_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_resize_workspace_second.png", &[0, 255, 0]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(2),
        width: Some(2),
        resize_mode: ResizeMode::Crop,
        resize_parity: ResizeParity::Compatibility,
        batch_execution: BatchExecution::Serial,
        ..Default::default()
    })
    .unwrap();
    let mut workspace = ImageProcessorWorkspace::new();

    let tensor = processor
        .open_batch_into(&[&first, &second], &mut workspace)
        .unwrap();

    assert_eq!(tensor.shape(), [2, 3, 2, 2]);
    assert_eq!(workspace.resize_coefficient_cache_len(), 2);

    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[test]
fn open_batch_into_direct_resize_matches_staged_path() {
    let first = write_test_rgb_png(
        "image_processors_direct_first.png",
        3,
        2,
        &[
            255, 0, 0, 0, 255, 0, 0, 0, 255, 32, 64, 96, 128, 160, 192, 224, 240, 16,
        ],
    );
    let second = write_test_rgb_png(
        "image_processors_direct_second.png",
        3,
        2,
        &[
            0, 32, 64, 96, 128, 160, 192, 224, 240, 255, 192, 128, 64, 32, 0, 16, 80, 144,
        ],
    );
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(2),
        width: Some(2),
        resize_mode: ResizeMode::Crop,
        resize_parity: ResizeParity::Compatibility,
        resample: ResizeFilter::Bicubic,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: true,
        rescale_factor: 1.0 / 255.0,
        do_normalize: true,
        image_mean: vec![0.48145466, 0.4578275, 0.40821073],
        image_std: vec![0.26862954, 0.261_302_6, 0.275_777_1],
        batch_execution: BatchExecution::Serial,
        ..Default::default()
    })
    .unwrap();
    let staged = processor.open_batch(&[&first, &second]).unwrap();
    let mut workspace = ImageProcessorWorkspace::new();
    let direct = processor
        .open_batch_into(&[&first, &second], &mut workspace)
        .unwrap();

    assert_eq!(direct.shape(), staged.shape());
    assert_eq!(direct.layout(), staged.layout());
    assert_eq!(direct.data().to_vec::<f32>(), staged.data().to_vec::<f32>());

    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[test]
fn open_batch_into_direct_resize_converts_mixed_formats() {
    let first = write_test_gray_png(
        "image_processors_direct_mixed_gray.png",
        3,
        2,
        &[0, 64, 128, 192, 224, 255],
    );
    let second = write_test_rgb_png(
        "image_processors_direct_mixed_rgb.png",
        3,
        2,
        &[
            0, 32, 64, 96, 128, 160, 192, 224, 240, 255, 192, 128, 64, 32, 0, 16, 80, 144,
        ],
    );
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(2),
        width: Some(2),
        resize_mode: ResizeMode::Crop,
        resize_parity: ResizeParity::Compatibility,
        resample: ResizeFilter::Bicubic,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: true,
        rescale_factor: 1.0 / 255.0,
        batch_execution: BatchExecution::Serial,
        ..Default::default()
    })
    .unwrap();
    let staged = processor.open_batch(&[&first, &second]).unwrap();
    let mut workspace = ImageProcessorWorkspace::new();
    let direct = processor
        .open_batch_into(&[&first, &second], &mut workspace)
        .unwrap();

    assert_eq!(direct.shape(), staged.shape());
    assert_eq!(direct.layout(), staged.layout());
    assert_eq!(direct.data().to_vec::<f32>(), staged.data().to_vec::<f32>());

    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[cfg(not(feature = "parallel"))]
#[test]
fn open_batch_parallel_reports_missing_feature_when_disabled() {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        batch_execution: BatchExecution::Parallel,
        ..Default::default()
    })
    .unwrap();

    let err = processor
        .open_batch(&[std::env::temp_dir().join("image_processors_missing_parallel.png")])
        .unwrap_err();

    assert!(matches!(err, ImageProcessorError::ParallelBatchUnavailable));
}

#[cfg(feature = "parallel")]
#[test]
fn open_batch_parallel_loads_paths_and_stacks_tensor() {
    let first = write_test_png("image_processors_parallel_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_parallel_second.png", &[0, 255, 0]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        batch_execution: BatchExecution::Parallel,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.open_batch(&[&first, &second]).unwrap();

    assert_eq!(tensor.shape(), [2, 3, 1, 1]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.batch(), Some(2));

    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[test]
fn preprocess_images_writes_nchw_batches_directly() {
    let first = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0, 10, 20, 30, 40, 50]).unwrap();
    let second = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![60, 70, 80, 90, 100, 110]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        output_layout: Layout::NCHW,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_images(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 3, 1, 2]);
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![0.0, 30.0, 10.0, 40.0, 20.0, 50.0, 60.0, 90.0, 70.0, 100.0, 80.0, 110.0,]
    );
}

#[test]
fn preprocess_images_writes_nhwc_batches_directly() {
    let first = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0, 10, 20, 30, 40, 50]).unwrap();
    let second = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![60, 70, 80, 90, 100, 110]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_images(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 1, 2, 3]);
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,]
    );
}

#[test]
fn preprocess_image_view_borrows_when_config_is_byte_preserving() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let view = processor.preprocess_image_view(&frame).unwrap();
    let TensorDataView::U8(values) = view.data() else {
        panic!("expected borrowed u8 tensor data");
    };

    assert!(std::ptr::eq(values.as_ptr(), frame.data().as_ptr()));
    assert_eq!(view.shape(), [1, 1, 2, 3]);
    assert_eq!(view.layout(), Layout::NHWC);
    assert_eq!(view.batch(), Some(1));
}

#[test]
fn preprocess_image_view_allows_noop_resize() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(1),
        width: Some(2),
        do_rescale: false,
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let view = processor.preprocess_image_view(&frame).unwrap();
    let TensorDataView::U8(values) = view.data() else {
        panic!("expected borrowed u8 tensor data");
    };

    assert!(std::ptr::eq(values.as_ptr(), frame.data().as_ptr()));
}

#[test]
fn preprocess_image_view_rejects_value_changing_config() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let err = processor.preprocess_image_view(&frame).unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::ZeroCopyUnavailable {
            reason: "rescale converts pixel bytes"
        }
    ));
}

#[test]
fn preprocess_image_sequence_views_borrow_each_frame_when_config_is_byte_preserving() {
    let first = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
    let second = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![4, 5, 6]).unwrap();
    let sequence =
        ImageSequence::new(vec![first, second], crate::media::LoopBehavior::Once).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let views = processor
        .preprocess_image_sequence_views(&sequence)
        .unwrap();
    let TensorDataView::U8(first_values) = views[0].data() else {
        panic!("expected borrowed first frame data");
    };
    let TensorDataView::U8(second_values) = views[1].data() else {
        panic!("expected borrowed second frame data");
    };

    assert_eq!(views.len(), 2);
    assert!(std::ptr::eq(
        first_values.as_ptr(),
        sequence.frames()[0].data().as_ptr()
    ));
    assert!(std::ptr::eq(
        second_values.as_ptr(),
        sequence.frames()[1].data().as_ptr()
    ));
    assert_eq!(views[0].shape(), [1, 1, 1, 3]);
    assert_eq!(views[0].frames(), Some(1));
    assert_eq!(views[0].batch(), None);
}

#[test]
fn preprocess_image_sequence_views_reject_value_changing_config() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();
    let sequence = ImageSequence::new(vec![frame], crate::media::LoopBehavior::Once).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let err = processor
        .preprocess_image_sequence_views(&sequence)
        .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::ZeroCopyUnavailable {
            reason: "rescale converts pixel bytes"
        }
    ));
}

#[test]
fn preprocess_video_views_borrow_each_frame_when_config_is_byte_preserving() {
    let first = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![1, 2, 3]).unwrap();
    let second = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![4, 5, 6]).unwrap();
    let video = VideoClip::new(
        vec![
            crate::media::VideoFrame::new(first),
            crate::media::VideoFrame::new(second),
        ],
        Some(30.0),
    )
    .unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let views = processor.preprocess_video_views(&video).unwrap();
    let TensorDataView::U8(first_values) = views[0].data() else {
        panic!("expected borrowed first video frame data");
    };
    let TensorDataView::U8(second_values) = views[1].data() else {
        panic!("expected borrowed second video frame data");
    };

    assert_eq!(views.len(), 2);
    assert!(std::ptr::eq(
        first_values.as_ptr(),
        video.frames()[0].image().data().as_ptr()
    ));
    assert!(std::ptr::eq(
        second_values.as_ptr(),
        video.frames()[1].image().data().as_ptr()
    ));
    assert_eq!(views[1].shape(), [1, 1, 1, 3]);
    assert_eq!(views[1].frames(), Some(1));
    assert_eq!(views[1].batch(), None);
}

#[test]
fn preprocess_videos_stacks_clips_as_batched_video_tensor() {
    let first = video_clip_from_rgb_pixels(&[[0, 10, 20], [30, 40, 50]]);
    let second = video_clip_from_rgb_pixels(&[[60, 70, 80], [90, 100, 110]]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        output_layout: Layout::NCHW,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_videos(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 2, 3, 1, 1]);
    assert_eq!(tensor.layout(), Layout::BFCHW);
    assert_eq!(tensor.batch(), Some(2));
    assert_eq!(tensor.frames(), Some(2));
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,]
    );
}

#[test]
fn preprocess_videos_uses_channel_last_batched_video_layout() {
    let first = video_clip_from_rgb_pixels(&[[1, 2, 3], [4, 5, 6]]);
    let second = video_clip_from_rgb_pixels(&[[7, 8, 9], [10, 11, 12]]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_videos(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 2, 1, 1, 3]);
    assert_eq!(tensor.layout(), Layout::BFHWC);
    assert_eq!(tensor.batch(), Some(2));
    assert_eq!(tensor.frames(), Some(2));
}

#[test]
fn preprocess_videos_rejects_mixed_frame_counts() {
    let first = video_clip_from_rgb_pixels(&[[0, 10, 20], [30, 40, 50]]);
    let second = video_clip_from_rgb_pixels(&[[60, 70, 80]]);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_rescale: false,
        ..Default::default()
    })
    .unwrap();

    let err = processor.preprocess_videos(&[first, second]).unwrap_err();

    assert!(matches!(err, ImageProcessorError::IncompatibleBatchShapes));
}

#[test]
fn processor_config_exposes_semantic_output_layouts() {
    let config = ImageProcessorConfig {
        output_layout: Layout::NHWC,
        ..Default::default()
    };

    assert_eq!(
        config.output_image_layout(),
        ImageLayout::HeightWidthChannels
    );
    assert_eq!(
        config.output_video_layout(),
        VideoLayout::FramesHeightWidthChannels
    );
}

#[test]
fn processor_config_defaults_to_decode_and_batch_execution() {
    let config = ImageProcessorConfig::default();

    assert_eq!(config.batch_execution, BatchExecution::default());
    assert_eq!(config.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(config.resize_parity, ResizeParity::Resampling);
}

#[cfg(feature = "parallel")]
#[test]
fn batch_execution_auto_switches_to_parallel_at_threshold() {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        batch_execution: BatchExecution::Auto,
        ..Default::default()
    })
    .unwrap();

    assert!(!processor
        .should_run_parallel(DEFAULT_PARALLEL_BATCH_THRESHOLD - 1)
        .unwrap());
    assert!(processor
        .should_run_parallel(DEFAULT_PARALLEL_BATCH_THRESHOLD)
        .unwrap());
}

#[cfg(not(feature = "parallel"))]
#[test]
fn batch_execution_auto_remains_serial_without_parallel_feature() {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        batch_execution: BatchExecution::Auto,
        ..Default::default()
    })
    .unwrap();

    assert!(!processor
        .should_run_parallel(DEFAULT_PARALLEL_BATCH_THRESHOLD)
        .unwrap());
}

#[cfg(feature = "parallel")]
#[test]
fn batch_execution_default_uses_auto_when_parallel_enabled() {
    assert_eq!(BatchExecution::default(), BatchExecution::Auto);
    assert_eq!(default_batch_execution(), BatchExecution::Auto);
}

#[cfg(not(feature = "parallel"))]
#[test]
fn batch_execution_default_uses_auto_when_parallel_disabled() {
    assert_eq!(BatchExecution::default(), BatchExecution::Auto);
    assert_eq!(default_batch_execution(), BatchExecution::Auto);
}

#[test]
fn new_rejects_non_finite_rescale_factor() {
    let err = ImageProcessor::new(ImageProcessorConfig {
        rescale_factor: f32::NAN,
        ..Default::default()
    })
    .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::NonFiniteRescaleFactor(_)
    ));
}

#[test]
fn new_rejects_invalid_normalization_std() {
    let err = ImageProcessor::new(ImageProcessorConfig {
        do_normalize: true,
        image_mean: vec![0.0],
        image_std: vec![-1.0],
        ..Default::default()
    })
    .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::NonPositiveNormalizationStd { index: 0, .. }
    ));
}

#[test]
fn preprocess_image_sequence_marks_first_axis_as_frames() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();
    let sequence =
        ImageSequence::new(vec![frame.clone(), frame], crate::media::LoopBehavior::Once).unwrap();
    let processor = ImageProcessor::default();

    let tensor = processor.preprocess_image_sequence(&sequence).unwrap();

    assert_eq!(tensor.frames(), Some(2));
    assert_eq!(tensor.batch(), None);
}

fn video_clip_from_rgb_pixels(pixels: &[[u8; 3]]) -> VideoClip {
    let frames = pixels
        .iter()
        .map(|pixel| {
            let image = ImageFrame::new(1, 1, PixelFormat::Rgb8, pixel.to_vec()).unwrap();
            crate::media::VideoFrame::new(image)
        })
        .collect();
    VideoClip::new(frames, Some(30.0)).unwrap()
}

#[test]
fn preprocess_matches_clip_style_two_pixel_golden_values() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0, 0, 0, 255, 127, 64]).unwrap();
    let processor = ImageProcessor::new(ImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        do_normalize: true,
        image_mean: vec![0.48145466, 0.4578275, 0.40821073],
        image_std: vec![0.26862954, 0.261_302_6, 0.275_777_1],
        output_layout: Layout::NHWC,
        ..Default::default()
    })
    .unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();
    let values = tensor.data().to_vec::<f32>();
    let expected = [
        -1.7922626,
        -1.7520971,
        -1.4802198,
        1.9303361,
        0.15388948,
        -0.57013553,
    ];

    assert_eq!(tensor.shape(), [1, 1, 2, 3]);
    assert!(values
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (*actual - expected).abs() < 1e-5));
}

fn write_test_png(name: &str, pixel: &[u8; 3]) -> std::path::PathBuf {
    write_test_rgb_png(name, 1, 1, pixel)
}

fn write_test_rgb_png(name: &str, width: u32, height: u32, pixels: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(pixels, width, height, ColorType::Rgb8.into())
        .unwrap();
    std::fs::write(&path, png).unwrap();
    path
}

fn write_test_gray_png(name: &str, width: u32, height: u32, pixels: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(pixels, width, height, ColorType::L8.into())
        .unwrap();
    std::fs::write(&path, png).unwrap();
    path
}
