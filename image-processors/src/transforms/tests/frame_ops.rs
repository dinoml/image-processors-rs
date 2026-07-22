use super::*;

#[test]
fn resize_frame_resizes_rgb_data() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();

    let resized = resize_frame(
        &frame,
        ImageSize {
            height: 2,
            width: 2,
        },
        ResizeFilter::Nearest,
        ResizeMode::Default,
    )
    .unwrap();

    assert_eq!(resized.width(), 2);
    assert_eq!(resized.height(), 2);
    assert_eq!(resized.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(
        resized.data(),
        vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
    );
}

#[test]
fn resize_filter_exposes_explicit_kernel() {
    assert_eq!(ResizeFilter::Nearest.kernel(), ResizeKernel::Nearest);
    assert_eq!(ResizeFilter::Bilinear.kernel(), ResizeKernel::Triangle);
    assert_eq!(ResizeFilter::Bicubic.kernel(), ResizeKernel::CatmullRom);
    assert_eq!(ResizeFilter::Lanczos.kernel(), ResizeKernel::Lanczos3);
}

#[test]
fn resize_decision_accepts_resampling_parity() {
    let decision = ResizeDecision::new(ResizeFilter::Bicubic, ResizeParity::Resampling)
        .expect("resampling parity should accept blending filters");

    assert_eq!(decision.filter(), ResizeFilter::Bicubic);
    assert_eq!(decision.kernel(), ResizeKernel::CatmullRom);
    assert_eq!(decision.parity(), ResizeParity::Resampling);
}

#[test]
fn resize_decision_accepts_compatibility_parity() {
    let decision = ResizeDecision::new(ResizeFilter::Bicubic, ResizeParity::Compatibility)
        .expect("compatibility parity should accept blending filters");

    assert_eq!(decision.filter(), ResizeFilter::Bicubic);
    assert_eq!(decision.kernel(), ResizeKernel::CatmullRom);
    assert_eq!(decision.parity(), ResizeParity::Compatibility);
}

#[test]
fn resize_decision_rejects_blending_filter_for_pixel_exact_parity() {
    let err = ResizeDecision::new(ResizeFilter::Bicubic, ResizeParity::PixelExact).unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidResizeParity {
            filter: ResizeFilter::Bicubic,
            parity: ResizeParity::PixelExact,
        }
    );
}

#[test]
fn resize_frame_with_pixel_exact_decision_uses_nearest_kernel() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 255]).unwrap();

    let resized = resize_frame_with_decision(
        &frame,
        ImageSize::new(1, 4).unwrap(),
        ResizeDecision::pixel_exact(),
        ResizeMode::Default,
    )
    .unwrap();

    assert_eq!(resized.data(), &[0, 0, 255, 255]);
}

#[test]
fn resampling_decision_resizes_rgb_data() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();

    let resized = resize_frame_with_decision(
        &frame,
        ImageSize::new(2, 2).unwrap(),
        ResizeFilter::Nearest.decision(),
        ResizeMode::Default,
    )
    .unwrap();

    assert_eq!(resized.width(), 2);
    assert_eq!(resized.height(), 2);
    assert_eq!(
        resized.data(),
        vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
    );
}

#[test]
fn converts_frame_pixel_format() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();

    let gray = convert_frame_pixel_format(&frame, PixelFormat::Luma8).unwrap();
    let rgba = convert_frame_pixel_format(&frame, PixelFormat::Rgba8).unwrap();

    assert_eq!(gray.data(), vec![18]);
    assert_eq!(rgba.data(), vec![10, 20, 30, 255]);
}

#[test]
fn convert_frame_pixel_format_rejects_trailing_channel_bytes() {
    let err = convert_to_rgb(&[1, 2, 3, 4, 5], 4).unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidChannelDataLength {
            channels: 4,
            actual: 5,
        }
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn resize_frame_rejects_u32_dimension_overflow() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![255]).unwrap();
    let err = resize_frame(
        &frame,
        ImageSize::new(1, u32::MAX as usize + 1).unwrap(),
        ResizeFilter::Nearest,
        ResizeMode::Default,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TransformError::DimensionTooLarge {
            dimension: "target width",
            ..
        }
    ));
}

#[test]
fn crop_region_finds_non_zero_mask_bounds() {
    let mask = ImageFrame::new(
        4,
        4,
        PixelFormat::Luma8,
        vec![0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0],
    )
    .unwrap();

    let region = crop_region(
        &mask,
        ImageSize {
            height: 4,
            width: 4,
        },
        0,
    )
    .unwrap();

    assert_eq!(region, (1, 1, 3, 3));
}

#[test]
fn pad_frame_adds_constant_rgb_border() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![10, 20, 30, 40, 50, 60]).unwrap();

    let padded = pad_frame(&frame, Padding::new(1, 1, 0, 1), &[1, 2, 3]).unwrap();

    assert_eq!(padded.width(), 4);
    assert_eq!(padded.height(), 2);
    assert_eq!(
        padded.data(),
        &[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 10, 20, 30, 40, 50, 60, 1, 2, 3,]
    );
}

#[test]
fn pad_frame_accepts_scalar_fill() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![9]).unwrap();

    let padded = pad_frame(&frame, Padding::symmetric(1, 0), &[4]).unwrap();

    assert_eq!(padded.width(), 1);
    assert_eq!(padded.height(), 3);
    assert_eq!(padded.data(), &[4, 9, 4]);
}

#[test]
fn pad_frame_with_canvas_fill_extends_image_edges() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let padded =
        pad_frame_with_canvas_fill(&frame, Padding::all(1), CanvasFill::ImageEdges).unwrap();

    assert_eq!(padded.width(), 4);
    assert_eq!(padded.height(), 4);
    assert_eq!(
        padded.data(),
        &[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
    );
}

#[test]
fn pad_frame_with_canvas_fill_reflects_image_edges() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let padded =
        pad_frame_with_canvas_fill(&frame, Padding::all(1), CanvasFill::ReflectImage).unwrap();

    assert_eq!(padded.width(), 4);
    assert_eq!(padded.height(), 4);
    assert_eq!(
        padded.data(),
        &[4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1]
    );
}

#[test]
fn symmetric_next_multiple_padding_repeats_edges_and_always_advances() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let padded = pad_frame_symmetric_to_next_multiple(
        &frame,
        ImageSize::new(2, 2).expect("valid multiples"),
    )
    .expect("symmetric padding should succeed");

    assert_eq!((padded.height(), padded.width()), (4, 4));
    assert_eq!(
        padded.data(),
        &[1, 2, 2, 1, 3, 4, 4, 3, 3, 4, 4, 3, 1, 2, 2, 1]
    );
}

#[test]
fn temporal_repeat_last_plan_maps_padded_indices_to_final_frame() {
    let plan = temporal_repeat_last_plan(5, 4).unwrap();

    assert_eq!(
        plan,
        TemporalRepeatLastPlan {
            source_frame_count: 5,
            multiple: 4,
            target_frame_count: 8,
            padding_frames: 3,
        }
    );
    assert!(!plan.is_identity());
    assert_eq!(plan.group_count(), 2);
    assert_eq!(
        plan.source_indices().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 4, 4, 4]
    );
    assert_eq!(plan.source_index(8), None);
}

#[test]
fn temporal_repeat_last_plan_rejects_invalid_inputs() {
    assert_eq!(
        temporal_repeat_last_plan(0, 4).unwrap_err(),
        TransformError::EmptyFrameBatch
    );
    assert_eq!(
        temporal_repeat_last_plan(4, 0).unwrap_err(),
        TransformError::InvalidTemporalMultiple(0)
    );
}

#[test]
fn repeat_last_frame_batch_clones_final_item_to_multiple() {
    let padded = repeat_last_frame_batch_to_multiple(&[1, 2, 3], 2).unwrap();

    assert_eq!(padded, vec![1, 2, 3, 3]);
}

#[test]
fn repeat_last_video_clip_preserves_fps_and_reuses_final_frame() {
    let first = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![1]).unwrap();
    let second = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![2]).unwrap();
    let clip = VideoClip::new(
        vec![VideoFrame::new(first), VideoFrame::new(second)],
        Some(12.0),
    )
    .unwrap();

    let padded = repeat_last_video_clip_to_multiple(&clip, 4).unwrap();

    assert_eq!(padded.len(), 4);
    assert_eq!(padded.fps(), Some(12.0));
    assert_eq!(padded.frames()[2].image().data(), &[2]);
    assert_eq!(padded.frames()[3].image().data(), &[2]);
}

#[test]
fn pad_video_clip_frames_with_canvas_fill_extends_each_frame_edges() {
    let first = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![1, 2]).unwrap();
    let second = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![10, 20]).unwrap();
    let clip = VideoClip::new(
        vec![VideoFrame::new(first), VideoFrame::new(second)],
        Some(24.0),
    )
    .unwrap();

    let padded = pad_video_clip_frames_with_canvas_fill(
        &clip,
        Padding::symmetric(0, 1),
        CanvasFill::ImageEdges,
    )
    .unwrap();

    assert_eq!(padded.len(), 2);
    assert_eq!(padded.fps(), Some(24.0));
    assert_eq!(padded.frames()[0].image().data(), &[1, 1, 2, 2]);
    assert_eq!(padded.frames()[1].image().data(), &[10, 10, 20, 20]);
}

#[test]
fn pad_frame_rejects_invalid_fill_length() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();

    let err = pad_frame(&frame, Padding::all(1), &[1, 2]).unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidPaddingFill {
            channels: 3,
            actual: 2,
        }
    );
}

#[test]
fn patch_grid_selects_resolution_by_effective_area_then_waste() {
    let selected = select_patch_grid_resolution(
        ImageSize::new(300, 500).unwrap(),
        &[
            ImageSize::new(672, 336).unwrap(),
            ImageSize::new(672, 672).unwrap(),
            ImageSize::new(336, 672).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(selected, ImageSize::new(336, 672).unwrap());
}

#[test]
fn patch_grid_plan_computes_resize_padding_and_counts() {
    let plan = patch_grid_plan(
        ImageSize::new(300, 500).unwrap(),
        &[
            ImageSize::new(336, 672).unwrap(),
            ImageSize::new(672, 336).unwrap(),
        ],
        224,
    )
    .unwrap();

    assert_eq!(plan.selected_size, ImageSize::new(336, 672).unwrap());
    assert_eq!(plan.resized_size, ImageSize::new(336, 560).unwrap());
    assert_eq!(plan.padding, Padding::new(0, 56, 0, 56));
    assert_eq!((plan.patches_height, plan.patches_width), (2, 3));
    assert_eq!(plan.tiled_patch_count().unwrap(), 6);
    assert_eq!(plan.image_patch_count().unwrap(), 7);
}

#[test]
fn patch_grid_image_patches_return_base_image_then_row_major_tiles() {
    let frame = ImageFrame::new(4, 2, PixelFormat::Luma8, (1..=8).collect()).unwrap();

    let patches = patch_grid_image_patches(
        &frame,
        &[ImageSize::new(2, 4).unwrap()],
        ImageSize::new(2, 4).unwrap(),
        2,
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(patches.plan.image_patch_count().unwrap(), 3);
    assert_eq!(patches.frames.len(), 3);
    assert_eq!(patches.frames[0].data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(patches.frames[1].data(), &[1, 2, 5, 6]);
    assert_eq!(patches.frames[2].data(), &[3, 4, 7, 8]);
}

#[test]
fn patch_grid_batch_plan_pads_to_largest_patch_count() {
    let first = patch_grid_plan(
        ImageSize::new(300, 500).unwrap(),
        &[ImageSize::new(336, 672).unwrap()],
        224,
    )
    .unwrap();
    let second = patch_grid_plan(
        ImageSize::new(224, 224).unwrap(),
        &[ImageSize::new(224, 224).unwrap()],
        224,
    )
    .unwrap();

    let batch = patch_grid_batch_plan(&[first, second]).unwrap();

    assert_eq!(batch.max_patches, 7);
    assert_eq!(batch.patch_counts, vec![7, 2]);
    assert_eq!(batch.padding_patches, vec![0, 5]);
}

#[test]
fn patch_grid_resolution_selection_rejects_empty_candidates() {
    let err = select_patch_grid_resolution(ImageSize::new(1, 1).unwrap(), &[]).unwrap_err();

    assert_eq!(err, TransformError::EmptyResolutionCandidates);
}

#[test]
fn center_crop_frame_extracts_middle_pixels() {
    let frame = ImageFrame::new(
        4,
        4,
        PixelFormat::Luma8,
        (0..16).map(|value| value as u8).collect(),
    )
    .unwrap();

    let cropped = center_crop_frame(&frame, ImageSize::new(2, 2).unwrap()).unwrap();

    assert_eq!(cropped.width(), 2);
    assert_eq!(cropped.height(), 2);
    assert_eq!(cropped.data(), &[5, 6, 9, 10]);
}

#[test]
fn center_crop_frame_rejects_larger_target() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 1]).unwrap();
    let target = ImageSize::new(2, 2).unwrap();

    let err = center_crop_frame(&frame, target).unwrap_err();

    assert_eq!(
        err,
        TransformError::CropTooLarge {
            source_size: ImageSize {
                height: 1,
                width: 2,
            },
            target_size: target,
        }
    );
}

#[test]
fn crop_frame_extracts_absolute_image_crop_box() {
    let frame = ImageFrame::new(
        4,
        3,
        PixelFormat::Luma8,
        (0..12).map(|value| value as u8).collect(),
    )
    .unwrap();

    let cropped = crop_frame(&frame, ImageCropBox::new(1, 1, 4, 3)).unwrap();

    assert_eq!(cropped.width(), 3);
    assert_eq!(cropped.height(), 2);
    assert_eq!(cropped.data(), &[5, 6, 7, 9, 10, 11]);
}

#[test]
fn crop_frame_rejects_out_of_bounds_image_crop_box() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 1]).unwrap();
    let crop_box = ImageCropBox::new(1, 0, 3, 1);

    let err = crop_frame(&frame, crop_box).unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidImageCropBox {
            crop_box,
            image_size: ImageSize {
                height: 1,
                width: 2,
            },
        }
    );
}

#[test]
fn overlay_frame_copies_rgb_foreground_at_position() {
    let background = ImageFrame::new(3, 2, PixelFormat::Rgb8, vec![0; 18]).unwrap();
    let foreground = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![255, 0, 0, 0, 255, 0]).unwrap();

    let overlaid = overlay_frame(&background, &foreground, OverlayPosition::new(1, 1)).unwrap();

    assert_eq!(
        overlaid.data(),
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0,]
    );
}

#[test]
fn overlay_frame_clips_negative_position() {
    let background = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();
    let foreground = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![9, 8, 7, 6]).unwrap();

    let overlaid = overlay_frame(&background, &foreground, OverlayPosition::new(-1, 0)).unwrap();

    assert_eq!(overlaid.data(), &[8, 2, 6, 4]);
}

#[test]
fn overlay_frame_alpha_blends_rgba_foreground() {
    let background = ImageFrame::new(1, 1, PixelFormat::Rgba8, vec![10, 20, 30, 255]).unwrap();
    let foreground = ImageFrame::new(1, 1, PixelFormat::Rgba8, vec![110, 220, 30, 128]).unwrap();

    let overlaid = overlay_frame(&background, &foreground, OverlayPosition::origin()).unwrap();

    assert_eq!(overlaid.data(), &[60, 120, 30, 255]);
}

#[test]
fn overlay_frame_rejects_pixel_format_mismatch() {
    let background = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![0, 0, 0]).unwrap();
    let foreground = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![255]).unwrap();

    let err = overlay_frame(&background, &foreground, OverlayPosition::origin()).unwrap_err();

    assert_eq!(
        err,
        TransformError::IncompatiblePixelFormat {
            frame: "foreground",
            expected: PixelFormat::Rgb8,
            actual: PixelFormat::Luma8,
        }
    );
}

#[test]
fn composite_mask_frame_blends_with_luma_mask() {
    let background = ImageFrame::new(
        3,
        1,
        PixelFormat::Rgb8,
        vec![0, 0, 0, 10, 20, 30, 100, 100, 100],
    )
    .unwrap();
    let foreground = ImageFrame::new(
        3,
        1,
        PixelFormat::Rgb8,
        vec![100, 50, 0, 110, 220, 30, 200, 0, 100],
    )
    .unwrap();
    let mask = ImageFrame::new(3, 1, PixelFormat::Luma8, vec![0, 128, 255]).unwrap();

    let composited = composite_mask_frame(&background, &foreground, &mask).unwrap();

    assert_eq!(composited.data(), &[0, 0, 0, 60, 120, 30, 200, 0, 100]);
}

#[test]
fn composite_mask_frame_rejects_mask_size_mismatch() {
    let background = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![0]).unwrap();
    let foreground = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![255]).unwrap();
    let mask = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 255]).unwrap();

    let err = composite_mask_frame(&background, &foreground, &mask).unwrap_err();

    assert_eq!(
        err,
        TransformError::IncompatibleFrameSize {
            frame: "mask",
            expected: ImageSize {
                height: 1,
                width: 1,
            },
            actual: ImageSize {
                height: 1,
                width: 2,
            },
        }
    );
}

proptest! {
    #[test]
    fn smart_resize_returns_positive_factor_multiples(
        height in 1usize..2048,
        width in 1usize..2048,
        factor in 1usize..64,
    ) {
        prop_assume!(height.max(width) as f64 / height.min(width) as f64 <= 200.0);
        let size = ImageSize::new(height, width).unwrap();
        let resized = size
            .smart_resize(ResizeLimits {
                factor,
                min_pixels: factor * factor,
                max_pixels: factor * factor * 4096,
            })
            .unwrap();

        prop_assert!(resized.height > 0);
        prop_assert!(resized.width > 0);
        prop_assert_eq!(resized.height % factor, 0);
        prop_assert_eq!(resized.width % factor, 0);
    }

    #[test]
    fn crop_region_returns_in_bounds_region_for_zero_masks(
        height in 1usize..32,
        width in 1usize..32,
        pad in 0usize..8,
    ) {
        let frame = ImageFrame::new(
            width,
            height,
            PixelFormat::Luma8,
            vec![0; width * height],
        )
        .unwrap();

        let region = crop_region(
            &frame,
            ImageSize::new(height, width).unwrap(),
            pad,
        )
        .unwrap();

        prop_assert_eq!(region, (0, 0, width, height));
    }
}
