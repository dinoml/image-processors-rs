use super::*;

#[test]
fn scale_factor_resize_plan_rounds_down_to_scale_factor() {
    let plan = scale_factor_resize_plan(ImageSize::new(513, 777).unwrap(), None, 8).unwrap();

    assert_eq!(plan.requested_size, ImageSize::new(513, 777).unwrap());
    assert_eq!(plan.resized_size, ImageSize::new(512, 776).unwrap());
    assert_eq!(plan.latent_size, ImageSize::new(64, 97).unwrap());
}

#[test]
fn scale_factor_resize_plan_uses_explicit_requested_size() {
    let plan = scale_factor_resize_plan(
        ImageSize::new(513, 777).unwrap(),
        Some(ImageSize::new(768, 1025).unwrap()),
        8,
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(768, 1024).unwrap());
    assert_eq!(plan.latent_size, ImageSize::new(96, 128).unwrap());
}

#[test]
fn shortest_edge_resize_size_applies_optional_longest_edge_cap() {
    let tiny = ImageSize::new(17, 13).unwrap();
    let wide = ImageSize::new(800, 2000).unwrap();

    assert_eq!(
        shortest_edge_resize_size(tiny, 800, Some(1333)).unwrap(),
        ImageSize::new(1046, 800).unwrap()
    );
    assert_eq!(
        shortest_edge_resize_size(wide, 800, Some(1333)).unwrap(),
        ImageSize::new(533, 1332).unwrap()
    );
}

#[test]
fn longest_edge_resize_size_preserves_aspect_ratio() {
    assert_eq!(
        longest_edge_resize_size(ImageSize::new(300, 500).unwrap(), 1024).unwrap(),
        ImageSize::new(614, 1024).unwrap()
    );
}

#[test]
fn should_rotate_to_match_orientation_matches_target_orientation() {
    assert!(should_rotate_to_match_orientation(
        ImageSize::new(100, 300).unwrap(),
        ImageSize::new(2560, 1920).unwrap(),
    )
    .unwrap());
    assert!(!should_rotate_to_match_orientation(
        ImageSize::new(300, 100).unwrap(),
        ImageSize::new(2560, 1920).unwrap(),
    )
    .unwrap());
}

#[test]
fn shortest_edge_resize_size_uses_target_canvas_shortest_edge() {
    let target = ImageSize::new(2560, 1920).unwrap();

    assert_eq!(
        shortest_edge_resize_size(
            ImageSize::new(300, 100).unwrap(),
            target.height.min(target.width),
            None,
        )
        .unwrap(),
        ImageSize::new(5760, 1920).unwrap()
    );
}

#[test]
fn fit_inside_size_fits_inside_target() {
    assert_eq!(
        fit_inside_size(
            ImageSize::new(5760, 1920).unwrap(),
            ImageSize::new(2560, 1920).unwrap(),
        )
        .unwrap(),
        ImageSize::new(2560, 853).unwrap()
    );
}

#[test]
fn fit_inside_size_preserves_square_aspect_ratio() {
    assert_eq!(
        fit_inside_size(
            ImageSize::new(3000, 3000).unwrap(),
            ImageSize::new(2560, 1920).unwrap(),
        )
        .unwrap(),
        ImageSize::new(1920, 1920).unwrap()
    );
}

#[test]
fn centered_padding_places_image_on_canvas() {
    assert_eq!(
        centered_padding(
            ImageSize::new(2000, 1000).unwrap(),
            ImageSize::new(2560, 1920).unwrap(),
        )
        .unwrap(),
        Padding::new(280, 460, 280, 460)
    );
}

#[test]
fn centered_padding_rejects_oversized_images() {
    let err = centered_padding(
        ImageSize::new(3000, 1000).unwrap(),
        ImageSize::new(2560, 1920).unwrap(),
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::CropTooLarge {
            source_size: ImageSize::new(2560, 1920).unwrap(),
            target_size: ImageSize::new(3000, 1000).unwrap(),
        }
    );
}

#[test]
fn select_aspect_ratio_bucket_matches_closest_ratio() {
    let bucket = select_aspect_ratio_bucket(
        ImageSize::new(700, 1400).unwrap(),
        &[
            ImageSize::new(704, 1408).unwrap(),
            ImageSize::new(1024, 1024).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(bucket, ImageSize::new(704, 1408).unwrap());
}

#[test]
fn select_aspect_ratio_bucket_rejects_empty_candidates() {
    let err = select_aspect_ratio_bucket(ImageSize::new(1, 1).unwrap(), &[]).unwrap_err();

    assert_eq!(err, TransformError::EmptyResolutionCandidates);
}

#[test]
fn video_size_bucket_plan_prefers_closest_frame_count() {
    let plan = video_size_bucket_plan(
        VideoSizeBucket::new(9, ImageSize::new(480, 640).unwrap()).unwrap(),
        &[
            VideoSizeBucket::new(16, ImageSize::new(480, 640).unwrap()).unwrap(),
            VideoSizeBucket::new(10, ImageSize::new(720, 1280).unwrap()).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(
        plan.selected,
        VideoSizeBucket::new(10, ImageSize::new(720, 1280).unwrap()).unwrap()
    );
    assert_eq!(plan.frame_delta(), 1);
}

#[test]
fn video_size_bucket_plan_uses_aspect_ratio_with_equal_frame_delta() {
    let plan = video_size_bucket_plan(
        VideoSizeBucket::new(9, ImageSize::new(480, 640).unwrap()).unwrap(),
        &[
            VideoSizeBucket::new(8, ImageSize::new(720, 1280).unwrap()).unwrap(),
            VideoSizeBucket::new(10, ImageSize::new(480, 640).unwrap()).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(
        plan.selected,
        VideoSizeBucket::new(10, ImageSize::new(480, 640).unwrap()).unwrap()
    );
}

#[test]
fn video_size_bucket_plan_rejects_empty_candidates() {
    let err = video_size_bucket_plan(
        VideoSizeBucket::new(1, ImageSize::new(1, 1).unwrap()).unwrap(),
        &[],
    )
    .unwrap_err();

    assert_eq!(err, TransformError::EmptyResolutionCandidates);
}

#[test]
fn video_clip_size_bucket_plan_rejects_mixed_frame_sizes() {
    let first = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![1, 2]).unwrap();
    let second = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![3]).unwrap();
    let clip = VideoClip::new(
        vec![VideoFrame::new(first), VideoFrame::new(second)],
        Some(24.0),
    )
    .unwrap();

    let err = video_clip_size_bucket_plan(
        &clip,
        &[VideoSizeBucket::new(2, ImageSize::new(1, 2).unwrap()).unwrap()],
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::IncompatibleFrameSize {
            frame: "video",
            expected: ImageSize::new(1, 2).unwrap(),
            actual: ImageSize::new(1, 1).unwrap(),
        }
    );
}

#[test]
fn pad_to_multiple_plan_rounds_up_with_bottom_right_padding() {
    let plan =
        pad_to_multiple_plan(ImageSize::new(5, 7).unwrap(), ImageSize::new(4, 6).unwrap()).unwrap();

    assert_eq!(
        plan,
        PadToMultiplePlan {
            source_size: ImageSize::new(5, 7).unwrap(),
            multiples: ImageSize::new(4, 6).unwrap(),
            padded_size: ImageSize::new(8, 12).unwrap(),
            padding: Padding::new(0, 5, 3, 0),
        }
    );
    assert!(!plan.is_identity());
    assert_eq!(plan.content_crop_box(), ImageCropBox::new(0, 0, 7, 5));
}

#[test]
fn pad_to_multiple_plan_rejects_zero_multiple() {
    let err = pad_to_multiple_plan(
        ImageSize::new(5, 7).unwrap(),
        ImageSize {
            height: 0,
            width: 4,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidSize {
            height: 0,
            width: 4,
        }
    );
}

#[test]
fn pad_frame_to_multiple_with_canvas_fill_extends_image_edges() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let padded = pad_frame_to_multiple_with_canvas_fill(
        &frame,
        ImageSize::new(3, 4).unwrap(),
        CanvasFill::ImageEdges,
    )
    .unwrap();

    assert_eq!(padded.width(), 4);
    assert_eq!(padded.height(), 3);
    assert_eq!(padded.data(), &[1, 2, 2, 2, 3, 4, 4, 4, 3, 4, 4, 4]);
}

#[test]
fn unpad_frame_to_size_crops_top_left_content() {
    let frame = ImageFrame::new(
        4,
        3,
        PixelFormat::Luma8,
        vec![1, 2, 2, 2, 3, 4, 4, 4, 3, 4, 4, 4],
    )
    .unwrap();

    let unpadded = unpad_frame_to_size(&frame, ImageSize::new(2, 2).unwrap()).unwrap();

    assert_eq!(unpadded.width(), 2);
    assert_eq!(unpadded.height(), 2);
    assert_eq!(unpadded.data(), &[1, 2, 3, 4]);
}

#[test]
fn resize_center_crop_plan_uses_ceil_cover_resize() {
    let plan = resize_center_crop_plan(
        ImageSize::new(3, 5).unwrap(),
        ImageSize::new(4, 4).unwrap(),
        ResizeRounding::Ceil,
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(4, 7).unwrap());
    assert_eq!(plan.crop_box, ImageCropBox::new(1, 0, 5, 4));
}

#[test]
fn resize_center_crop_plan_uses_floor_cover_resize() {
    let plan = resize_center_crop_plan(
        ImageSize::new(400, 800).unwrap(),
        ImageSize::new(512, 512).unwrap(),
        ResizeRounding::Floor,
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(512, 1024).unwrap());
    assert_eq!(plan.crop_box, ImageCropBox::new(256, 0, 768, 512));
}

#[test]
fn resize_center_crop_frame_preserves_constant_pixels() {
    let frame = ImageFrame::new(5, 3, PixelFormat::Rgb8, vec![7; 5 * 3 * 3]).unwrap();

    let output = resize_center_crop_frame(
        &frame,
        ImageSize::new(4, 4).unwrap(),
        ResizeFilter::Bilinear,
        ResizeRounding::Ceil,
    )
    .unwrap();

    assert_eq!(output.width(), 4);
    assert_eq!(output.height(), 4);
    assert_eq!(output.data(), &[7; 4 * 4 * 3]);
}

#[test]
fn multiple_of_resize_plan_rounds_to_per_axis_multiples() {
    let plan = multiple_of_resize_plan(
        ImageSize::new(513, 777).unwrap(),
        None,
        ImageSize::new(16, 32).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(512, 768).unwrap());
}

#[test]
fn multiple_of_resize_plan_uses_requested_size() {
    let plan = multiple_of_resize_plan(
        ImageSize::new(100, 100).unwrap(),
        Some(ImageSize::new(65, 97).unwrap()),
        ImageSize::new(8, 16).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(64, 96).unwrap());
}

#[test]
fn resize_fill_plan_centers_contained_image() {
    let plan = resize_fill_plan(
        ImageSize::new(100, 300).unwrap(),
        ImageSize::new(160, 160).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(53, 160).unwrap());
    assert_eq!(plan.padding, Padding::new(53, 0, 54, 0));
}

#[test]
fn bottom_right_resize_pad_plan_keeps_top_left_anchor() {
    let plan = bottom_right_resize_pad_plan(
        ImageSize::new(100, 300).unwrap(),
        ImageSize::new(160, 160).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(53, 160).unwrap());
    assert_eq!(plan.padding, Padding::new(0, 0, 107, 0));
}

#[test]
fn resize_frame_to_bottom_right_padded_frame_extends_edges() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let padded = resize_frame_to_bottom_right_padded_frame(
        &frame,
        ImageSize::new(3, 4).unwrap(),
        CanvasFill::ImageEdges,
        ResizeFilter::Nearest.decision(),
    )
    .unwrap();

    assert_eq!(padded.width(), 4);
    assert_eq!(padded.height(), 3);
    assert_eq!(padded.data(), &[1, 2, 2, 2, 3, 4, 4, 4, 3, 4, 4, 4]);
}

#[test]
fn resize_rgb_to_fill_frame_uses_constant_rgb_fill() {
    let frame = ImageFrame::new(3, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

    let output = resize_rgb_to_fill_frame(
        &frame,
        ImageSize::new(3, 3).unwrap(),
        RgbCanvasFill::ConstantRgb([10, 20, 30]),
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(output.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(
        output.data(),
        &[
            10, 20, 30, 10, 20, 30, 10, 20, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 10, 20, 30,
            10, 20, 30,
        ]
    );
}

#[test]
fn resize_rgb_to_fill_frame_can_extend_image_edges() {
    let frame = ImageFrame::new(3, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

    let output = resize_rgb_to_fill_frame(
        &frame,
        ImageSize::new(3, 3).unwrap(),
        RgbCanvasFill::ImageEdges,
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(
        output.data(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,]
    );
}

#[test]
fn resize_frame_to_fill_frame_uses_constant_luma_fill() {
    let frame = ImageFrame::new(3, 1, PixelFormat::Luma8, vec![1, 2, 3]).unwrap();

    let output = resize_frame_to_fill_frame(
        &frame,
        ImageSize::new(3, 3).unwrap(),
        CanvasFill::constant(vec![9]),
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(output.pixel_format(), PixelFormat::Luma8);
    assert_eq!(output.data(), &[9, 9, 9, 1, 2, 3, 9, 9, 9]);
}

#[test]
fn resize_frame_to_fill_frame_can_extend_non_rgb_edges() {
    let frame = ImageFrame::new(3, 1, PixelFormat::Luma8, vec![1, 2, 3]).unwrap();

    let output = resize_frame_to_fill_frame(
        &frame,
        ImageSize::new(3, 3).unwrap(),
        CanvasFill::ImageEdges,
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(output.pixel_format(), PixelFormat::Luma8);
    assert_eq!(output.data(), &[1, 2, 3, 1, 2, 3, 1, 2, 3]);
}

#[test]
fn resize_frame_to_fill_frame_can_reflect_non_rgb_edges() {
    let frame = ImageFrame::new(2, 2, PixelFormat::Luma8, vec![1, 2, 3, 4]).unwrap();

    let output = resize_frame_to_fill_frame(
        &frame,
        ImageSize::new(2, 4).unwrap(),
        CanvasFill::ReflectImage,
        ResizeFilter::Nearest,
    )
    .unwrap();

    assert_eq!(output.pixel_format(), PixelFormat::Luma8);
    assert_eq!(output.data(), &[2, 1, 2, 1, 4, 3, 4, 3]);
}

#[test]
fn resize_frame_to_fill_frame_rejects_invalid_fill_length() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();

    let err = resize_frame_to_fill_frame(
        &frame,
        ImageSize::new(2, 2).unwrap(),
        CanvasFill::constant(vec![1, 2]),
        ResizeFilter::Nearest,
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidPaddingFill {
            channels: 3,
            actual: 2,
        }
    );
}

#[test]
fn validate_image_size_constraints_accepts_limits() {
    let size = ImageSize::new(128, 256).unwrap();
    let constraints = ImageSizeConstraints {
        max_aspect_ratio: 8.0,
        min_side_length: 64,
    };

    assert!(validate_image_size_constraints(size, constraints).is_ok());
}

#[test]
fn validate_image_size_constraints_rejects_small_side() {
    let size = ImageSize::new(63, 128).unwrap();
    let constraints = ImageSizeConstraints {
        max_aspect_ratio: 8.0,
        min_side_length: 64,
    };

    let err = validate_image_size_constraints(size, constraints).unwrap_err();

    assert_eq!(
        err,
        TransformError::ImageTooSmall {
            size,
            min_side_length: constraints.min_side_length,
        }
    );
}

#[test]
fn validate_image_size_constraints_rejects_extreme_ratio() {
    let size = ImageSize::new(64, 1024).unwrap();
    let constraints = ImageSizeConstraints {
        max_aspect_ratio: 8.0,
        min_side_length: 64,
    };

    let err = validate_image_size_constraints(size, constraints).unwrap_err();

    assert_eq!(
        err,
        TransformError::AspectRatioTooLarge {
            aspect_ratio: 16.0,
            max_aspect_ratio: constraints.max_aspect_ratio,
        }
    );
}

#[test]
fn area_resize_plan_limits_area_with_truncation() {
    let plan = area_resize_plan(ImageSize::new(1000, 2000).unwrap(), 1024 * 1024).unwrap();

    assert!(plan.should_resize());
    assert_eq!(plan.resized_size, ImageSize::new(724, 1448).unwrap());
}

#[test]
fn area_resize_plan_keeps_small_images() {
    let source = ImageSize::new(256, 512).unwrap();

    let plan = area_resize_plan(source, 1024 * 1024).unwrap();

    assert!(!plan.should_resize());
    assert_eq!(plan.resized_size, source);
}

#[test]
fn resize_frame_to_area_limit_returns_clone_under_area_limit() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();

    let resized = resize_frame_to_area_limit(&frame, 1024 * 1024, ResizeFilter::Bilinear).unwrap();

    assert_eq!(resized, frame);
}

#[test]
fn center_crop_box_selects_center_window() {
    let crop_box =
        center_crop_box(ImageSize::new(5, 7).unwrap(), ImageSize::new(3, 4).unwrap()).unwrap();

    assert_eq!(crop_box, ImageCropBox::new(1, 1, 5, 4));
}

#[test]
fn crop_frame_extracts_center_crop_box() {
    let frame = ImageFrame::new(
        4,
        3,
        PixelFormat::Luma8,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    .unwrap();

    let crop_box = center_crop_box(frame_size(&frame), ImageSize::new(1, 2).unwrap()).unwrap();
    let cropped = crop_frame(&frame, crop_box).unwrap();

    assert_eq!(cropped.width(), 2);
    assert_eq!(cropped.height(), 1);
    assert_eq!(cropped.data(), &[6, 7]);
}

#[test]
fn concatenate_frames_horizontally_rgb_rejects_empty_batch() {
    let err = concatenate_frames_horizontally_rgb(&[], [255, 255, 255]).unwrap_err();

    assert_eq!(err, TransformError::EmptyImageBatch);
}

#[test]
fn concatenate_frames_horizontally_rgb_converts_single_frame() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![9, 8]).unwrap();

    let output =
        concatenate_frames_horizontally_rgb(std::slice::from_ref(&frame), [255, 255, 255]).unwrap();

    assert_eq!(output.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(output.data(), &[9, 9, 9, 8, 8, 8]);
}

#[test]
fn concatenate_frames_horizontally_rgb_centers_on_fill_canvas() {
    let wide = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![255, 0, 0, 255, 0, 0]).unwrap();
    let tall = ImageFrame::new(
        1,
        3,
        PixelFormat::Rgb8,
        vec![0, 255, 0, 0, 255, 0, 0, 255, 0],
    )
    .unwrap();

    let output = concatenate_frames_horizontally_rgb(&[wide, tall], [255, 255, 255]).unwrap();

    assert_eq!(output.width(), 3);
    assert_eq!(output.height(), 3);
    assert_eq!(output.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(
        output.data(),
        &[
            255, 255, 255, 255, 255, 255, 0, 255, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255,
            255, 255, 255, 255, 0, 255, 0,
        ]
    );
}
