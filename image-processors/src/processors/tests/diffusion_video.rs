use super::*;

#[test]
fn vae_processor_resizes_to_vae_scale_factor_multiple() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let frame = ImageFrame::new(17, 9, PixelFormat::Rgb8, vec![128; 17 * 9 * 3]).unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();

    assert_eq!(tensor.shape(), [1, 3, 8, 16]);
    assert_eq!(tensor.layout(), Layout::NCHW);
}

#[test]
fn flux2_processor_validates_concatenates_and_preprocesses_reference_images() {
    let processor = Flux2ImageProcessor::new(Flux2ImageProcessorConfig::default()).unwrap();
    let left = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![64; 4 * 2 * 3]).unwrap();
    let right = ImageFrame::new(2, 4, PixelFormat::Luma8, vec![192; 2 * 4]).unwrap();

    let err = processor.check_image_input(&left).unwrap_err();
    assert!(matches!(
        err,
        ImageProcessorError::Transform(TransformError::ImageTooSmall { .. })
    ));

    let concatenated = processor.concatenate_images(&[left, right]).unwrap();
    assert_eq!(concatenated.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(concatenated.width(), 6);
    assert_eq!(concatenated.height(), 4);

    let limited = processor.resize_if_exceeds_area(&concatenated, 12).unwrap();
    assert_eq!(limited.width() * limited.height(), 8);

    let tensor = processor
        .preprocess_image_with_options(
            &concatenated,
            ImageProcessorOptions {
                height: Some(32),
                width: Some(48),
                resize_mode: None,
            },
        )
        .unwrap();
    assert_eq!(tensor.shape(), [1, 3, 32, 48]);
    assert_eq!(tensor.layout(), Layout::NCHW);
}

#[test]
fn hunyuan_video_15_processor_selects_directional_aspect_bucket() {
    let processor =
        HunyuanVideo15ImageProcessor::new(HunyuanVideo15ImageProcessorConfig::default()).unwrap();

    let selected = processor
        .calculate_default_height_width(720, 1280, 256)
        .unwrap();

    assert_eq!(selected, ImageSize::new(192, 336).unwrap());
    assert!(processor
        .bucket_candidates(256)
        .unwrap()
        .contains(&selected));
}

#[test]
fn joy_image_edit_processor_selects_bucket_and_preprocesses_reference_image() {
    let config = JoyImageEditImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    assert_eq!(
        config
            .target_size_for_size(ImageSize::new(600, 1000).unwrap())
            .unwrap(),
        ImageSize::new(768, 1280).unwrap()
    );
    let processor = JoyImageEditImageProcessor::new(config).unwrap();
    let frame = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![128; 4 * 2 * 3]).unwrap();

    let tensor = processor
        .preprocess_image_with_options(
            &frame,
            ImageProcessorOptions {
                height: Some(600),
                width: Some(1000),
                resize_mode: None,
            },
        )
        .unwrap();

    assert_eq!(tensor.shape(), [1, 3, 768, 1280]);
    assert_eq!(tensor.layout(), Layout::NCHW);
}

#[test]
fn wan_animate_processor_rounds_target_and_preprocesses_reference_image() {
    let config = WanAnimateImageProcessorConfig::default();
    assert_eq!(
        config
            .target_size_for_size(ImageSize::new(20, 33).unwrap())
            .unwrap(),
        ImageSize::new(16, 32).unwrap()
    );
    let processor = WanAnimateImageProcessor::new(config).unwrap();
    let frame = ImageFrame::new(33, 20, PixelFormat::Rgb8, vec![128; 33 * 20 * 3]).unwrap();

    let tensor = processor
        .preprocess_image_with_options(
            &frame,
            ImageProcessorOptions {
                height: Some(20),
                width: Some(33),
                resize_mode: None,
            },
        )
        .unwrap();

    assert_eq!(tensor.shape(), [1, 3, 16, 32]);
    assert_eq!(tensor.layout(), Layout::NCHW);
}

#[test]
fn document_ocr_processor_aligns_long_axis_with_target() {
    let processor = DocumentOcrImageProcessor::new(DocumentOcrImageProcessorConfig {
        image_size: ImageSize::new(4, 2).unwrap(),
        do_resize: false,
        do_thumbnail: false,
        do_align_long_axis: true,
        do_pad: false,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap();
    let values = (1u8..=6)
        .flat_map(|value| [value, value, value])
        .collect::<Vec<_>>();
    let frame = ImageFrame::new(3, 2, PixelFormat::Rgb8, values).unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();

    assert_eq!(tensor.shape(), [1, 3, 3, 2]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![
            4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 4.0, 1.0, 5.0, 2.0, 6.0,
            3.0,
        ]
    );
}

#[test]
fn videomae_processor_preprocesses_decoded_clip_as_frame_tensor() {
    let processor = VideoMaeImageProcessor::new(VideoMaeImageProcessorConfig {
        size: 2,
        crop_size: ImageSize::new(2, 2).unwrap(),
        resample: ResizeFilter::Nearest,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap();
    let first = ImageFrame::new(
        3,
        3,
        PixelFormat::Rgb8,
        vec![
            1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24, 5, 15, 25, 6, 16, 26, 7, 17, 27, 8, 18, 28,
            9, 19, 29,
        ],
    )
    .unwrap();
    let second = ImageFrame::new(
        3,
        3,
        PixelFormat::Rgb8,
        vec![
            101, 111, 121, 102, 112, 122, 103, 113, 123, 104, 114, 124, 105, 115, 125, 106, 116,
            126, 107, 117, 127, 108, 118, 128, 109, 119, 129,
        ],
    )
    .unwrap();
    let video = VideoClip::new(
        vec![
            crate::media::VideoFrame::new(first),
            crate::media::VideoFrame::new(second),
        ],
        Some(24.0),
    )
    .unwrap();

    let tensor = processor.preprocess_video(&video).unwrap();

    assert_eq!(tensor.shape(), [2, 3, 2, 2]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.leading_axis(), Some(TensorLeadingAxis::Frames));
}

#[test]
fn videomae_processor_preprocesses_video_batch_as_batched_video_tensor() {
    let processor = VideoMaeImageProcessor::new(VideoMaeImageProcessorConfig {
        do_resize: false,
        do_center_crop: false,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap();
    let first = video_clip_from_rgb_pixels(&[[0, 10, 20], [30, 40, 50]]);
    let second = video_clip_from_rgb_pixels(&[[60, 70, 80], [90, 100, 110]]);

    let tensor = processor.preprocess_videos(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 2, 3, 1, 1]);
    assert_eq!(tensor.layout(), Layout::BFCHW);
    assert_eq!(tensor.batch(), Some(2));
    assert_eq!(tensor.frames(), Some(2));
}

#[test]
fn vivit_processor_preprocesses_video_batch_as_channel_last_batched_video_tensor() {
    let processor = VivitImageProcessor::new(VivitImageProcessorConfig {
        do_resize: false,
        do_center_crop: false,
        do_rescale: false,
        offset: false,
        do_normalize: false,
        output_layout: VideoLayout::FramesHeightWidthChannels,
        ..Default::default()
    })
    .unwrap();
    let first = video_clip_from_rgb_pixels(&[[1, 2, 3], [4, 5, 6]]);
    let second = video_clip_from_rgb_pixels(&[[7, 8, 9], [10, 11, 12]]);

    let tensor = processor.preprocess_videos(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 2, 1, 1, 3]);
    assert_eq!(tensor.layout(), Layout::BFHWC);
    assert_eq!(tensor.batch(), Some(2));
    assert_eq!(tensor.frames(), Some(2));
}

#[test]
fn ltx2_processor_stacks_channel_last_reference_videos() {
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig {
        do_resize: false,
        output_layout: VideoLayout::FramesHeightWidthChannels,
        ..Default::default()
    })
    .unwrap();
    let videos = [
        video_clip_from_rgb_pixels(&[[0, 127, 255], [255, 127, 0]]),
        video_clip_from_rgb_pixels(&[[64, 128, 192], [32, 96, 160]]),
    ];
    let target_size = ImageSize::new(1, 1).unwrap();

    let clip = processor
        .preprocess_reference_video_hdr(&videos[0], target_size)
        .unwrap();
    assert_eq!(clip.shape(), [2, 1, 1, 3]);
    assert_eq!(clip.layout(), Layout::NHWC);
    assert_eq!(clip.leading_axis(), Some(TensorLeadingAxis::Frames));

    let batch = processor
        .preprocess_reference_videos_hdr(&videos, target_size)
        .unwrap();
    assert_eq!(batch.shape(), [2, 2, 1, 1, 3]);
    assert_eq!(batch.layout(), Layout::BFHWC);
    assert_eq!(batch.leading_axis(), Some(TensorLeadingAxis::Batch));
    assert_eq!(batch.batch(), Some(2));
    assert_eq!(batch.frames(), Some(2));
    assert_f32_values_close(
        &batch.data().to_vec::<f32>(),
        &[
            -1.0,
            -0.003_921_568_4,
            1.0,
            1.0,
            -0.003_921_568_4,
            -1.0,
            -0.498_039_2,
            0.003_921_569,
            0.505_882_4,
            -0.749_019_6,
            -0.247_058_81,
            0.254_901_98,
        ],
        1e-6,
    );
}

#[test]
fn ltx2_processor_uses_global_replicate_for_bottom_and_right_padding() {
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig {
        do_resize: false,
        ..Default::default()
    })
    .unwrap();
    let first = ImageFrame::new(
        2,
        2,
        PixelFormat::Rgb8,
        vec![0, 0, 0, 127, 127, 127, 255, 255, 255, 64, 64, 64],
    )
    .unwrap();
    let second = ImageFrame::new(
        2,
        2,
        PixelFormat::Rgb8,
        vec![10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
    )
    .unwrap();
    let video = VideoClip::new(
        vec![
            crate::media::VideoFrame::new(first),
            crate::media::VideoFrame::new(second),
        ],
        Some(24.0),
    )
    .unwrap();

    let tensor = processor
        .preprocess_reference_video_hdr(&video, ImageSize::new(3, 4).unwrap())
        .unwrap();

    assert_eq!(tensor.shape(), [2, 3, 3, 4]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.leading_axis(), Some(TensorLeadingAxis::Frames));
    let values = tensor.data().to_vec::<f32>();
    let first_red_plane = &values[..12];
    assert_f32_values_close(
        first_red_plane,
        &[
            -1.0,
            -0.003_921_568_4,
            -0.003_921_568_4,
            -0.003_921_568_4,
            1.0,
            -0.498_039_2,
            -0.498_039_2,
            -0.498_039_2,
            1.0,
            -0.498_039_2,
            -0.498_039_2,
            -0.498_039_2,
        ],
        1e-6,
    );
    assert_eq!(
        first_red_plane[8], first_red_plane[4],
        "bottom padding must replicate after right padding makes reflection invalid"
    );
    assert_eq!(
        first_red_plane[2], first_red_plane[1],
        "right padding must replicate the last source column"
    );
}

#[test]
fn ltx2_processor_defaults_to_vae_rounding_then_f32_reflect_padding() {
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig::default()).unwrap();
    let first = ImageFrame::new(
        50,
        70,
        PixelFormat::Rgb8,
        (0..50 * 70 * 3)
            .map(|index| ((index as u32 * 37 + 17) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let second = first.clone();

    let vae = processor
        .vae_processor()
        .preprocess_images(&[first.clone(), second.clone()])
        .unwrap();
    assert_eq!(vae.shape(), [2, 3, 64, 32]);
    assert_eq!(vae.layout(), Layout::NCHW);

    let video = VideoClip::new(
        vec![
            crate::media::VideoFrame::new(first),
            crate::media::VideoFrame::new(second),
        ],
        Some(24.0),
    )
    .unwrap();
    let tensor = processor
        .preprocess_reference_video_hdr(&video, ImageSize::new(40, 32).unwrap())
        .unwrap();

    assert_eq!(tensor.shape(), [2, 3, 40, 32]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.leading_axis(), Some(TensorLeadingAxis::Frames));
    let values = tensor.data().to_vec::<f32>();
    assert_eq!(values[20], values[18]);
    assert_eq!(values[31], values[7]);
}

#[test]
fn ltx2_processor_rejects_frames_smaller_than_the_default_vae_scale() {
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig::default()).unwrap();
    let frame = ImageFrame::new(31, 31, PixelFormat::Rgb8, vec![0; 31 * 31 * 3]).unwrap();

    let error = processor
        .preprocess_reference_image_hdr(&frame, ImageSize::new(40, 40).unwrap())
        .unwrap_err();

    assert!(matches!(
        error,
        ImageProcessorError::Transform(TransformError::InvalidSize {
            height: 0,
            width: 0
        })
    ));
}

#[test]
fn ltx2_processor_postprocesses_logc3_hdr_video_tensor() {
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-0.814_382, 0.0, 1.0, -1.0, -0.700_684_3, -0.814_382]),
        vec![1, 2, 3, 1, 1],
        Layout::BFCHW,
    )
    .unwrap();

    let output = processor.postprocess_hdr_video(&tensor).unwrap();

    assert_eq!(output.layout(), Layout::BFHWC);
    assert_eq!(output.shape(), [1, 2, 1, 1, 3]);
    assert_f32_values_close(
        &output.data().to_vec::<f32>(),
        &[
            0.0,
            0.513_383_4,
            55.079_56,
            -0.017_290_42,
            0.010_591_047,
            0.0,
        ],
        1e-5,
    );
}

#[test]
fn vivit_processor_rejects_offset_without_rescale() {
    let err = VivitImageProcessor::new(VivitImageProcessorConfig {
        do_rescale: false,
        offset: true,
        ..Default::default()
    })
    .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::UnsupportedProcessorOption { field: "offset" }
    ));
}

#[test]
fn vae_processor_normalizes_unit_range_to_signed_range() {
    let config = VaeImageProcessorConfig {
        do_resize: false,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessor::new(config).unwrap();
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 127, 255]).unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();
    let values = tensor.data().to_vec::<f32>();

    assert_eq!(tensor.shape(), [1, 3, 1, 1]);
    assert!((values[0] + 1.0).abs() < 1e-6);
    assert!((values[1] - ((127.0 / 255.0) * 2.0 - 1.0)).abs() < 1e-6);
    assert!((values[2] - 1.0).abs() < 1e-6);
}

#[test]
fn ldm3d_depth_map_rejects_mismatched_value_count() {
    let err = DepthMapU16::new(2, 2, vec![0; 3]).unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::Transform(TransformError::InvalidBufferLength {
            expected: 4,
            actual: 3
        })
    ));
}

#[test]
fn ldm3d_preprocess_returns_rgb_and_depth_tensors() {
    let config = VaeImageProcessorConfig {
        do_resize: false,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessorLdm3d::new(config).unwrap();
    let rgb = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0, 0, 0, 255, 127, 64]).unwrap();
    let depth = DepthMapU16::new(2, 1, vec![0, u16::MAX]).unwrap();

    let output = processor.preprocess(&rgb, &depth).unwrap();

    assert_eq!(output.rgb().shape(), [1, 3, 1, 2]);
    assert_eq!(output.depth().shape(), [1, 1, 1, 2]);
    assert_eq!(output.depth().data().to_vec::<f32>(), vec![-1.0, 1.0]);
}

#[test]
fn ldm3d_preprocess_resizes_depth_to_rgb_vae_target() {
    let config = VaeImageProcessorConfig {
        vae_scale_factor: 1,
        resample: ResizeFilter::Nearest,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessorLdm3d::new(config).unwrap();
    let rgb = ImageFrame::new(4, 4, PixelFormat::Rgb8, vec![128; 4 * 4 * 3]).unwrap();
    let depth = DepthMapU16::new(4, 4, (0..16).map(|value| value * 257).collect()).unwrap();

    let output = processor
        .preprocess_with_options(
            &rgb,
            &depth,
            ImageProcessorOptions {
                height: Some(2),
                width: Some(2),
                resize_mode: None,
            },
        )
        .unwrap();

    assert_eq!(output.rgb().shape(), [1, 3, 2, 2]);
    assert_eq!(output.depth().shape(), [1, 1, 2, 2]);
}

#[test]
fn ldm3d_postprocess_scalar_depth_channel_returns_rgb_and_u16_depth() {
    let processor = VaeImageProcessorLdm3d::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-1.0, 0.0, 1.0, 1.0]),
        vec![1, 4, 1, 1],
        Layout::NCHW,
    )
    .unwrap();

    let output = processor.postprocess(&tensor, None).unwrap();

    assert_eq!(output.rgb()[0].data(), &[0, 128, 255]);
    assert_eq!(output.depth()[0].data(), &[u16::MAX]);
}

#[test]
fn ldm3d_postprocess_rgb_like_depth_uses_green_and_blue_bytes() {
    let config = VaeImageProcessorConfig {
        do_normalize: false,
        ..Default::default()
    };
    let processor = VaeImageProcessorLdm3d::new(config).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0, 0.0, 0.0, 0.0, 1.0, 128.0 / 255.0]),
        vec![1, 1, 1, 6],
        Layout::NHWC,
    )
    .unwrap();

    let output = processor.postprocess(&tensor, None).unwrap();

    assert_eq!(output.depth()[0].data(), &[u16::from(u8::MAX) * 256 + 128]);
}

#[test]
fn vae_preprocess_inpaint_crops_image_and_mask_from_foreground_bounds() {
    let config = VaeImageProcessorConfig {
        vae_scale_factor: 1,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessor::new(config).unwrap();
    let image = ImageFrame::new(4, 4, PixelFormat::Rgb8, vec![128; 4 * 4 * 3]).unwrap();
    let mut mask_data = vec![0; 4 * 4];
    mask_data[5] = 255;
    mask_data[6] = 255;
    mask_data[9] = 255;
    mask_data[10] = 255;
    let mask = ImageFrame::new(4, 4, PixelFormat::Luma8, mask_data).unwrap();

    let output = processor
        .preprocess_inpaint_with_options(
            &image,
            &mask,
            ImageProcessorOptions {
                height: Some(2),
                width: Some(2),
                resize_mode: None,
            },
            Some(0),
        )
        .unwrap();

    assert_eq!(output.pixel_values().shape(), [1, 3, 2, 2]);
    assert_eq!(output.mask().shape(), [1, 1, 2, 2]);
    assert_eq!(
        output.overlay_context().unwrap().crop_box,
        ImageCropBox::new(1, 1, 3, 3)
    );
}

#[test]
fn vae_preprocess_inpaint_binarizes_grayscale_mask_without_normalization() {
    let config = VaeImageProcessorConfig {
        do_resize: false,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessor::new(config).unwrap();
    let image = ImageFrame::new(3, 1, PixelFormat::Rgb8, vec![0; 3 * 3]).unwrap();
    let mask = ImageFrame::new(3, 1, PixelFormat::Luma8, vec![0, 127, 128]).unwrap();

    let output = processor.preprocess_inpaint(&image, &mask).unwrap();

    assert_eq!(output.mask().shape(), [1, 1, 1, 3]);
    assert_eq!(output.mask().data().to_vec::<f32>(), vec![0.0, 0.0, 1.0]);
    assert!(output.overlay_context().is_none());
}

#[test]
fn vae_postprocess_inpaint_applies_saved_crop_overlay_context() {
    let config = VaeImageProcessorConfig {
        vae_scale_factor: 1,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessor::new(config).unwrap();
    let image = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![10, 20, 30, 40, 50, 60]).unwrap();
    let mask = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 255]).unwrap();
    let preprocess = processor
        .preprocess_inpaint_with_options(
            &image,
            &mask,
            ImageProcessorOptions {
                height: Some(1),
                width: Some(1),
                resize_mode: None,
            },
            Some(0),
        )
        .unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![1.0, -1.0, -1.0]),
        vec![1, 3, 1, 1],
        Layout::NCHW,
    )
    .unwrap();

    let frames = processor
        .postprocess_inpaint(&tensor, &preprocess, None)
        .unwrap();

    assert_eq!(frames[0].data(), &[10, 20, 30, 255, 0, 0]);
}

#[test]
fn vae_postprocess_latent_returns_tensor_without_denormalizing() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(TensorData::F32(vec![-1.0]), vec![1, 1, 1, 1], Layout::NCHW).unwrap();

    let output = processor
        .postprocess(&tensor, VaeOutputType::Latent, None)
        .unwrap();

    assert_eq!(
        output.as_tensor().unwrap().data().to_vec::<f32>(),
        vec![-1.0]
    );
}

#[test]
fn vae_postprocess_array_returns_nhwc_denormalized_tensor() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-1.0, 1.0, 0.0, 0.5, 1.0, -1.0]),
        vec![1, 3, 1, 2],
        Layout::NCHW,
    )
    .unwrap();

    let output = processor
        .postprocess(&tensor, VaeOutputType::Array, None)
        .unwrap();
    let tensor = output.as_tensor().unwrap();

    assert_eq!(tensor.layout(), Layout::NHWC);
    assert_eq!(tensor.shape(), [1, 1, 2, 3]);
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![0.0, 0.5, 1.0, 1.0, 0.75, 0.0]
    );
}

#[test]
fn vae_postprocess_array_returns_bfhwc_denormalized_video_tensor() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-1.0, 0.0, 1.0, 1.0, -1.0, 0.0]),
        vec![1, 2, 3, 1, 1],
        Layout::BFCHW,
    )
    .unwrap();

    let output = processor
        .postprocess(&tensor, VaeOutputType::Array, None)
        .unwrap();
    let tensor = output.as_tensor().unwrap();

    assert_eq!(tensor.layout(), Layout::BFHWC);
    assert_eq!(tensor.shape(), [1, 2, 1, 1, 3]);
    assert_eq!(
        tensor.data().to_vec::<f32>(),
        vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.5]
    );
}

#[test]
fn vae_postprocess_frames_rounds_denormalized_rgb_values() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-1.0, 0.0, 1.0]),
        vec![1, 3, 1, 1],
        Layout::NCHW,
    )
    .unwrap();

    let output = processor
        .postprocess(&tensor, VaeOutputType::Frames, None)
        .unwrap();
    let frames = output.as_frames().unwrap();

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].pixel_format(), PixelFormat::Rgb8);
    assert_eq!(frames[0].data(), &[0, 128, 255]);
}

#[test]
fn vae_postprocess_frames_flattens_batched_video_tensor() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![-1.0, 0.0, 1.0, 1.0, -1.0, 0.0]),
        vec![1, 2, 1, 1, 3],
        Layout::BFHWC,
    )
    .unwrap();

    let output = processor
        .postprocess(&tensor, VaeOutputType::Frames, None)
        .unwrap();
    let frames = output.as_frames().unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].data(), &[0, 128, 255]);
    assert_eq!(frames[1].data(), &[255, 0, 128]);
}

#[test]
fn vae_postprocess_rejects_denormalize_flag_count_mismatch() {
    let processor = VaeImageProcessor::new(VaeImageProcessorConfig::default()).unwrap();
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 6]),
        vec![2, 3, 1, 1],
        Layout::NCHW,
    )
    .unwrap();

    let err = processor
        .postprocess(&tensor, VaeOutputType::Tensor, Some(&[true]))
        .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::InvalidDenormalizeFlags {
            expected: 2,
            actual: 1
        }
    ));
}
