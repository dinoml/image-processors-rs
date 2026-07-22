use super::*;

#[test]
fn latent_channel_check_matches_configured_latent_channels() {
    assert!(is_latent_channel_count(4, 4).unwrap());
    assert!(!is_latent_channel_count(3, 4).unwrap());
}

#[test]
fn unit_signed_range_normalize_and_denormalize_are_inverse_mappings() {
    let normalized = normalize_unit_to_signed(&[0.0, 0.5, 1.0]).unwrap();
    let denormalized = denormalize_signed_to_unit(&[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();

    assert_eq!(normalized, vec![-1.0, 0.0, 1.0]);
    assert_eq!(denormalized, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn logc3_to_linear_matches_ltx2_anchor_values() {
    let values = logc3_to_linear(&[LOGC3_F, LOGC3_E * LOGC3_CUT + LOGC3_F, 0.5, 1.0]).unwrap();

    assert_f32_slice_close(&values, &[0.0, 0.010_591_047, 0.513_383_4, 55.079_56], 1e-5);
}

#[test]
fn binarize_mask_to_unit_f32_uses_greater_than_or_equal_half() {
    let mask = binarize_mask_to_unit_f32(&[0.49, 0.5, 0.51]).unwrap();

    assert_eq!(mask, vec![0.0, 1.0, 1.0]);
}

#[test]
fn inpaint_overlay_blends_full_frame_with_luma_mask() {
    let original =
        ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![10, 20, 30, 100, 110, 120]).unwrap();
    let generated = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![200, 0, 0, 0, 200, 0]).unwrap();
    let mask = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![0, 255]).unwrap();

    let overlaid = inpaint_overlay(&original, &generated, &mask, None).unwrap();

    assert_eq!(overlaid.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(overlaid.data(), &[10, 20, 30, 0, 200, 0]);
}

#[test]
fn inpaint_overlay_uses_soft_mask_alpha() {
    let original = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![10, 20, 30]).unwrap();
    let generated = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![110, 120, 130]).unwrap();
    let mask = ImageFrame::new(1, 1, PixelFormat::Luma8, vec![128]).unwrap();

    let overlaid = inpaint_overlay(&original, &generated, &mask, None).unwrap();

    assert_eq!(overlaid.data(), &[60, 70, 80]);
}

#[test]
fn inpaint_overlay_places_generated_image_in_crop_box() {
    let original = ImageFrame::new(
        4,
        1,
        PixelFormat::Rgb8,
        vec![10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0],
    )
    .unwrap();
    let generated = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![200, 0, 0]).unwrap();
    let mask = ImageFrame::new(4, 1, PixelFormat::Luma8, vec![0, 255, 255, 0]).unwrap();

    let overlaid = inpaint_overlay(
        &original,
        &generated,
        &mask,
        Some(ImageCropBox::new(1, 0, 3, 1)),
    )
    .unwrap();

    assert_eq!(overlaid.data(), &[10, 0, 0, 200, 0, 0, 200, 0, 0, 40, 0, 0]);
}

#[test]
fn inpaint_overlay_rejects_out_of_bounds_crop_box() {
    let original = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0; 6]).unwrap();
    let generated = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
    let mask = ImageFrame::new(2, 1, PixelFormat::Luma8, vec![255, 255]).unwrap();
    let crop_box = ImageCropBox::new(0, 0, 3, 1);

    let err = inpaint_overlay(&original, &generated, &mask, Some(crop_box)).unwrap_err();

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
fn downsample_attention_mask_repeats_single_mask_for_batch_and_embed_dim() {
    let downsampled =
        downsample_attention_mask(&[0.25], ImageSize::new(1, 1).unwrap(), 2, 1, 3).unwrap();

    assert_eq!(downsampled.downsample_size, ImageSize::new(1, 1).unwrap());
    assert_eq!(downsampled.shape(), [2, 1, 3]);
    assert_eq!(downsampled.values, vec![0.25, 0.25, 0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn downsample_attention_mask_pads_when_downsample_grid_has_fewer_queries() {
    let downsampled =
        downsample_attention_mask(&[1.0], ImageSize::new(1, 1).unwrap(), 1, 10, 1).unwrap();

    assert_eq!(downsampled.downsample_size, ImageSize::new(4, 2).unwrap());
    assert_eq!(downsampled.shape(), [1, 10, 1]);
    assert!(downsampled.values[..8]
        .iter()
        .all(|value| (*value - 1.0).abs() < 1e-6));
    assert_eq!(&downsampled.values[8..], &[0.0, 0.0]);
}

#[test]
fn downsample_attention_mask_rejects_zero_attention_dimension() {
    let err =
        downsample_attention_mask(&[1.0], ImageSize::new(1, 1).unwrap(), 1, 0, 1).unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidAttentionShape {
            batch_size: 1,
            num_queries: 0,
            value_embed_dim: 1,
        }
    );
}

#[test]
fn downsample_attention_mask_rejects_incomplete_mask_plane() {
    let err = downsample_attention_mask(&[1.0, 0.0, 1.0], ImageSize::new(2, 2).unwrap(), 1, 1, 1)
        .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidBufferLength {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn patch_aligned_resize_plan_aligns_small_images_to_patch_grid() {
    let plan = patch_aligned_resize_plan(
        ImageSize::new(17, 33).unwrap(),
        ImageSize::new(1024, 1024).unwrap(),
        ImageSize::new(16, 16).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(32, 48).unwrap());
    assert_eq!((plan.patch_rows, plan.patch_columns), (2, 3));
    assert_eq!(plan.patch_count().unwrap(), 6);
}

#[test]
fn patch_aligned_resize_size_floors_downscale_then_aligns_up() {
    let resized = patch_aligned_resize_size(
        ImageSize::new(3000, 2000).unwrap(),
        ImageSize::new(1024, 1024).unwrap(),
        ImageSize::new(16, 16).unwrap(),
    )
    .unwrap();

    assert_eq!(resized, ImageSize::new(1024, 688).unwrap());
}

#[test]
fn patch_aligned_resize_plan_supports_rectangular_limits_and_patches() {
    let plan = patch_aligned_resize_plan(
        ImageSize::new(800, 1200).unwrap(),
        ImageSize::new(512, 1024).unwrap(),
        ImageSize::new(32, 16).unwrap(),
    )
    .unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(512, 768).unwrap());
    assert_eq!((plan.patch_rows, plan.patch_columns), (16, 48));
}

#[test]
fn spatial_batch_padding_plan_pads_to_largest_resized_size() {
    let plan = spatial_batch_padding_plan(&[
        ImageSize::new(32, 48).unwrap(),
        ImageSize::new(16, 64).unwrap(),
    ])
    .unwrap();

    assert_eq!(plan.target_size, ImageSize::new(32, 64).unwrap());
    assert_eq!(
        plan.padding,
        vec![Padding::new(0, 16, 0, 0), Padding::new(0, 0, 16, 0)]
    );
}

#[test]
fn spatial_batch_padding_plan_rejects_empty_batches() {
    let err = spatial_batch_padding_plan(&[]).unwrap_err();

    assert_eq!(err, TransformError::EmptyImageBatch);
}

#[test]
fn supported_tiled_canvas_grids_match_upstream_id_order() {
    let grids = supported_tiled_canvas_grids(4).unwrap();

    assert_eq!(
        grids,
        vec![
            TiledCanvasGrid::new(1, 1).unwrap(),
            TiledCanvasGrid::new(1, 2).unwrap(),
            TiledCanvasGrid::new(1, 3).unwrap(),
            TiledCanvasGrid::new(1, 4).unwrap(),
            TiledCanvasGrid::new(2, 1).unwrap(),
            TiledCanvasGrid::new(2, 2).unwrap(),
            TiledCanvasGrid::new(3, 1).unwrap(),
            TiledCanvasGrid::new(4, 1).unwrap(),
        ]
    );
}

#[test]
fn tiled_canvas_plan_chooses_best_downscale_canvas() {
    let plan = tiled_canvas_plan(ImageSize::new(300, 500).unwrap(), 224, 4).unwrap();

    assert_eq!(plan.grid, TiledCanvasGrid::new(2, 2).unwrap());
    assert_eq!(plan.canvas_size, ImageSize::new(448, 448).unwrap());
    assert_eq!(plan.resized_size, ImageSize::new(268, 448).unwrap());
    assert_eq!(plan.padding, Padding::new(0, 0, 180, 0));
    assert_eq!(plan.aspect_ratio_id, 6);
}

#[test]
fn tiled_canvas_plan_prefers_smallest_upscale_canvas_then_area() {
    let plan = tiled_canvas_plan(ImageSize::new(100, 300).unwrap(), 224, 4).unwrap();

    assert_eq!(plan.grid, TiledCanvasGrid::new(1, 2).unwrap());
    assert_eq!(plan.canvas_size, ImageSize::new(224, 448).unwrap());
    assert_eq!(plan.resized_size, ImageSize::new(100, 300).unwrap());
    assert_eq!(plan.padding, Padding::new(0, 148, 124, 0));
    assert_eq!(plan.aspect_ratio_id, 2);
}

#[test]
fn tiled_canvas_aspect_ratio_mask_marks_valid_tiles() {
    let mask = tiled_canvas_aspect_ratio_mask(TiledCanvasGrid::new(1, 3).unwrap(), 4).unwrap();

    assert_eq!(mask, vec![true, true, true, false]);
}

#[test]
fn tiled_canvas_batch_metadata_pads_ids_and_masks() {
    let metadata = tiled_canvas_batch_metadata(
        &[
            vec![
                TiledCanvasGrid::new(1, 2).unwrap(),
                TiledCanvasGrid::new(2, 2).unwrap(),
            ],
            vec![TiledCanvasGrid::new(3, 1).unwrap()],
        ],
        4,
    )
    .unwrap();

    assert_eq!(metadata.max_images_per_sample, 2);
    assert_eq!(metadata.num_tiles, vec![vec![2, 4], vec![3]]);
    assert_eq!(metadata.aspect_ratio_ids, vec![vec![2, 6], vec![7, 0]]);
    assert_eq!(
        metadata.aspect_ratio_mask,
        vec![
            vec![vec![true, true, false, false], vec![true, true, true, true],],
            vec![
                vec![true, true, true, false],
                vec![true, false, false, false],
            ],
        ]
    );
}

#[test]
fn tiled_canvas_batch_metadata_rejects_empty_image_batches() {
    let err = tiled_canvas_batch_metadata(&[Vec::new()], 4).unwrap_err();

    assert_eq!(err, TransformError::EmptyImageBatch);
}

#[test]
fn nested_image_grid_metadata_derives_row_major_target_positions() {
    let metadata = nested_image_grid_metadata(
        vec![
            vec![
                ImageSize::new(64, 128).unwrap(),
                ImageSize::new(64, 128).unwrap(),
                ImageSize::new(64, 128).unwrap(),
            ],
            vec![
                ImageSize::new(64, 128).unwrap(),
                ImageSize::new(64, 128).unwrap(),
                ImageSize::new(64, 128).unwrap(),
            ],
        ],
        vec![vec![false, false, false], vec![true, false, true]],
    )
    .unwrap();

    assert_eq!(metadata.rows, 2);
    assert_eq!(metadata.columns, 3);
    assert_eq!(
        metadata.target_positions,
        vec![
            NestedImageGridTarget::new(1, 0),
            NestedImageGridTarget::new(1, 2),
        ]
    );
    assert_eq!(metadata.target_mask[1], vec![true, false, true]);
}

#[test]
fn nested_image_grid_metadata_rejects_inconsistent_mask_shape() {
    let err = nested_image_grid_metadata(
        vec![vec![
            ImageSize::new(64, 128).unwrap(),
            ImageSize::new(64, 128).unwrap(),
        ]],
        vec![vec![false]],
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InconsistentNestedImageGridColumns {
            field: "target_mask",
            row: 0,
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn split_image_resize_size_rounds_short_side_to_even() {
    let resized = split_image_resize_size(ImageSize::new(301, 500).unwrap(), 364).unwrap();

    assert_eq!(resized, ImageSize::new(220, 364).unwrap());
}

#[test]
fn split_image_resize_size_caps_large_outputs() {
    let resized = split_image_resize_size(ImageSize::new(1000, 3000).unwrap(), 5000).unwrap();

    assert_eq!(resized, ImageSize::new(1364, 4096).unwrap());
}

#[test]
fn split_image_encoder_size_rounds_to_tile_multiple() {
    let size = split_image_encoder_size(ImageSize::new(300, 700).unwrap(), 364).unwrap();

    assert_eq!(size, ImageSize::new(364, 728).unwrap());
}

#[test]
fn split_image_plan_reports_rows_columns_and_global_frame() {
    let plan = split_image_plan(ImageSize::new(300, 700).unwrap(), 700, 364).unwrap();

    assert_eq!(plan.resized_size, ImageSize::new(300, 700).unwrap());
    assert_eq!(plan.vision_encoder_size, ImageSize::new(364, 728).unwrap());
    assert_eq!((plan.rows, plan.columns), (1, 2));
    assert_eq!(plan.frame_count, 3);
}

#[test]
fn split_image_plan_reports_zero_rows_columns_without_split_crops() {
    let plan = split_image_plan(ImageSize::new(2, 2).unwrap(), 4, 4).unwrap();

    assert_eq!(plan.vision_encoder_size, ImageSize::new(4, 4).unwrap());
    assert_eq!((plan.rows, plan.columns), (0, 0));
    assert_eq!(plan.frame_count, 1);
}

#[test]
fn split_image_batch_metadata_sums_flattened_frame_counts() {
    let first = split_image_plan(ImageSize::new(300, 700).unwrap(), 700, 364).unwrap();
    let second = split_image_plan(ImageSize::new(2, 2).unwrap(), 4, 4).unwrap();
    let third = split_image_plan(ImageSize::new(700, 300).unwrap(), 700, 364).unwrap();

    let metadata = split_image_batch_metadata(&[vec![first, second], vec![third]]).unwrap();

    assert_eq!(metadata.max_frames_per_sample, 4);
    assert_eq!(metadata.sample_frame_counts, vec![4, 3]);
    assert_eq!(metadata.image_frame_counts, vec![vec![3, 1], vec![3]]);
    assert_eq!(metadata.rows, vec![vec![1, 0], vec![2]]);
    assert_eq!(metadata.columns, vec![vec![2, 0], vec![1]]);
}

#[test]
fn nested_frame_batch_padding_plan_builds_pixel_attention_masks() {
    let plan = nested_frame_batch_padding_plan(&[
        vec![ImageSize::new(2, 3).unwrap(), ImageSize::new(4, 4).unwrap()],
        vec![ImageSize::new(1, 2).unwrap()],
    ])
    .unwrap();

    assert_eq!(plan.max_frames_per_sample, 2);
    assert_eq!(plan.target_size, ImageSize::new(4, 4).unwrap());
    assert_eq!(
        plan.padding,
        vec![
            vec![Padding::new(0, 1, 2, 0), Padding::new(0, 0, 0, 0)],
            vec![Padding::new(0, 2, 3, 0)],
        ]
    );
    assert_eq!(
        plan.pixel_attention_mask,
        vec![
            vec![
                vec![
                    true, true, true, false, true, true, true, false, false, false, false, false,
                    false, false, false, false,
                ],
                vec![
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true,
                ],
            ],
            vec![
                vec![
                    true, true, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false,
                ],
                vec![false; 16],
            ],
        ]
    );
}

#[test]
fn nested_frame_batch_padding_plan_rejects_empty_frame_batches() {
    let err = nested_frame_batch_padding_plan(&[Vec::new()]).unwrap_err();

    assert_eq!(err, TransformError::EmptyImageBatch);
}

#[test]
fn aspect_ratio_crop_plan_returns_no_crops_below_activation_ratio() {
    let plan = aspect_ratio_crop_plan(
        ImageSize::new(100, 149).unwrap(),
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap();

    assert_eq!(plan.crop_count(), 0);
    assert_eq!((plan.crop_rows, plan.crop_columns), (0, 0));
    assert_eq!(plan.crop_size, None);
}

#[test]
fn aspect_ratio_crop_plan_splits_landscape_images_into_columns() {
    let plan = aspect_ratio_crop_plan(
        ImageSize::new(100, 350).unwrap(),
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap();

    assert_eq!((plan.crop_rows, plan.crop_columns), (1, 4));
    assert_eq!(plan.crop_size, Some(ImageSize::new(100, 88).unwrap()));
    assert_eq!(
        plan.crops,
        vec![
            AspectRatioCrop {
                origin_x: 0,
                origin_y: 0,
                size: ImageSize::new(100, 88).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 88,
                origin_y: 0,
                size: ImageSize::new(100, 88).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 176,
                origin_y: 0,
                size: ImageSize::new(100, 88).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 264,
                origin_y: 0,
                size: ImageSize::new(100, 86).unwrap(),
            },
        ]
    );
}

#[test]
fn aspect_ratio_crop_plan_splits_portrait_images_into_rows() {
    let plan = aspect_ratio_crop_plan(
        ImageSize::new(350, 100).unwrap(),
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap();

    assert_eq!((plan.crop_rows, plan.crop_columns), (4, 1));
    assert_eq!(plan.crop_size, Some(ImageSize::new(88, 100).unwrap()));
    assert_eq!(
        plan.crops,
        vec![
            AspectRatioCrop {
                origin_x: 0,
                origin_y: 0,
                size: ImageSize::new(88, 100).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 0,
                origin_y: 88,
                size: ImageSize::new(88, 100).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 0,
                origin_y: 176,
                size: ImageSize::new(88, 100).unwrap(),
            },
            AspectRatioCrop {
                origin_x: 0,
                origin_y: 264,
                size: ImageSize::new(86, 100).unwrap(),
            },
        ]
    );
}

#[test]
fn aspect_ratio_crop_plan_skips_crops_when_nominal_size_is_too_small() {
    let plan = aspect_ratio_crop_plan(
        ImageSize::new(100, 350).unwrap(),
        AspectRatioCropOptions {
            min_crop_size: 120,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap();

    assert_eq!(plan.crop_count(), 0);
}

#[test]
fn aspect_ratio_crop_batch_metadata_reports_num_crops_per_image() {
    let metadata = aspect_ratio_crop_batch_metadata(
        &[
            ImageSize::new(100, 350).unwrap(),
            ImageSize::new(100, 149).unwrap(),
        ],
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap();

    assert_eq!(metadata.num_crops, vec![4, 0]);
}

#[test]
fn aspect_ratio_crop_batch_metadata_rejects_empty_batches() {
    let err = aspect_ratio_crop_batch_metadata(
        &[],
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: 1.5,
        },
    )
    .unwrap_err();

    assert_eq!(err, TransformError::EmptyImageBatch);
}

#[test]
fn aspect_ratio_crop_plan_rejects_invalid_activation_ratio() {
    let err = aspect_ratio_crop_plan(
        ImageSize::new(100, 350).unwrap(),
        AspectRatioCropOptions {
            min_crop_size: 80,
            max_num_crops: 4,
            min_ratio_to_activate: -1.0,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidAspectRatioCropActivationRatio(-1.0)
    );
}
