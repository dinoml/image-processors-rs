use super::*;

const PROBE_SIZE: ImageSize = ImageSize {
    height: 1,
    width: 1,
};

pub(super) fn validate_id(id: &str) -> Result<(), RecipeError> {
    if id.trim().is_empty() {
        Err(RecipeError::EmptyId)
    } else {
        Ok(())
    }
}

fn validate_transform<T>(
    stage_index: usize,
    stage: &'static str,
    result: Result<T, TransformError>,
) -> Result<(), RecipeError> {
    result
        .map(drop)
        .map_err(|source| RecipeError::InvalidTransformStage {
            stage_index,
            stage,
            source,
        })
}

pub(super) fn validate_stages(stages: &[ProcessorRecipeStage]) -> Result<(), RecipeError> {
    if stages.is_empty() {
        return Err(RecipeError::EmptyStages);
    }

    let mut pixel_format = None;
    let mut smart_resize_factor = None;
    let mut patch_flatten = None;
    for (index, stage) in stages.iter().enumerate() {
        match stage {
            ProcessorRecipeStage::ConvertPixelFormat { format } => {
                pixel_format = Some(*format);
            }
            ProcessorRecipeStage::Resize { resize } => {
                validate_resize_stage(index, *resize)?;
                if let RecipeResizeTarget::SmartResize { limits } = resize.target {
                    smart_resize_factor = Some(limits.factor);
                }
            }
            ProcessorRecipeStage::Owlv2AntialiasedResize {
                size,
                do_rescale,
                rescale_factor,
                ..
            } => {
                validate_transform(index, stage.kind(), ImageSize::new(size.height, size.width))?;
                if *do_rescale && !rescale_factor.is_finite() {
                    return Err(RecipeError::NonFiniteRescaleFactor {
                        stage_index: index,
                        value: *rescale_factor,
                    });
                }
            }
            ProcessorRecipeStage::VitPoseAffine {
                size,
                normalize_factor,
                padding_factor,
            } => {
                validate_transform(index, stage.kind(), ImageSize::new(size.height, size.width))?;
                for value in [*normalize_factor, *padding_factor] {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(RecipeError::InvalidTransformStage {
                            stage_index: index,
                            stage: stage.kind(),
                            source: TransformError::InvalidFloatScaleFactor(value),
                        });
                    }
                }
            }
            ProcessorRecipeStage::Rescale { factor } if !factor.is_finite() => {
                return Err(RecipeError::NonFiniteRescaleFactor {
                    stage_index: index,
                    value: *factor,
                });
            }
            ProcessorRecipeStage::Rescale { .. }
            | ProcessorRecipeStage::Binarize
            | ProcessorRecipeStage::Overlay { .. }
            | ProcessorRecipeStage::MaskComposite
            | ProcessorRecipeStage::ConcatenateHorizontallyRgb { .. } => {}
            ProcessorRecipeStage::Normalize { mean, std } => {
                validate_normalization_stats(index, "mean", mean, pixel_format)?;
                validate_normalization_stats(index, "std", std, pixel_format)?;
                validate_positive_std(index, std)?;
            }
            ProcessorRecipeStage::PatchFlatten { patch } => {
                validate_patch_stage(index, *patch)?;
                patch_flatten = Some((index, *patch));
            }
            ProcessorRecipeStage::SampleFrames { sampling } => {
                validate_frame_sampling_stage(index, *sampling)?;
            }
            ProcessorRecipeStage::TemporalRepeatLast { multiple } => {
                validate_transform(index, stage.kind(), temporal_repeat_last_plan(1, *multiple))?;
            }
            ProcessorRecipeStage::PatchGrid { patch_grid } => {
                validate_patch_grid_stage(index, patch_grid)?;
            }
            ProcessorRecipeStage::ImageSplit { split } => {
                validate_image_split_stage(index, *split)?;
            }
            ProcessorRecipeStage::AspectRatioCrops { aspect_ratio_crops } => {
                validate_aspect_ratio_crops_stage(index, *aspect_ratio_crops)?;
            }
            ProcessorRecipeStage::TiledCanvas { tile } => {
                validate_tiled_canvas_stage(index, *tile)?;
            }
            ProcessorRecipeStage::DocumentGeometry { document } => {
                validate_document_geometry_stage(index, *document)?;
            }
            ProcessorRecipeStage::ValidateImageSize { constraints } => {
                validate_image_size_constraint_stage(index, *constraints)?;
            }
            ProcessorRecipeStage::SelectAspectRatioBucket { candidates } => {
                validate_transform(
                    index,
                    stage.kind(),
                    select_aspect_ratio_bucket(PROBE_SIZE, candidates),
                )?;
            }
            ProcessorRecipeStage::SelectVideoSizeBucket { candidates } => {
                validate_transform(
                    index,
                    stage.kind(),
                    video_size_bucket_plan(
                        VideoSizeBucket {
                            frame_count: 1,
                            size: PROBE_SIZE,
                        },
                        candidates,
                    ),
                )?;
            }
            ProcessorRecipeStage::AreaResize { target_area, .. } => {
                validate_transform(
                    index,
                    stage.kind(),
                    area_resize_plan(PROBE_SIZE, *target_area),
                )?;
            }
            ProcessorRecipeStage::Crop { crop } => {
                validate_crop_stage(index, *crop)?;
            }
            ProcessorRecipeStage::ResizeCenterCrop { size, rounding, .. } => {
                validate_transform(
                    index,
                    stage.kind(),
                    resize_center_crop_plan(PROBE_SIZE, *size, *rounding),
                )?;
            }
            ProcessorRecipeStage::Pad { pad } => {
                validate_pad_stage(index, pad, pixel_format)?;
            }
            ProcessorRecipeStage::PadCanvas { padding, fill } => {
                validate_pad_canvas_stage(index, *padding, fill, pixel_format)?;
            }
            ProcessorRecipeStage::PadToMultiple { multiples, fill } => {
                validate_pad_to_multiple_stage(index, *multiples, fill, pixel_format)?;
            }
            ProcessorRecipeStage::PadSymmetricToNextMultiple { multiples } => {
                validate_symmetric_next_multiple_stage(index, *multiples)?;
            }
            ProcessorRecipeStage::RoundToMultiple {
                requested_size,
                multiples,
            } => {
                validate_round_to_multiple_stage(index, *requested_size, *multiples)?;
            }
            ProcessorRecipeStage::ResizeFill { size, fill, .. } => {
                validate_resize_fill_stage(index, *size, fill, pixel_format)?;
            }
            ProcessorRecipeStage::ResizeFillRgb { size, .. } => {
                validate_transform(index, stage.kind(), resize_fill_plan(PROBE_SIZE, *size))?;
            }
        }
    }
    validate_patch_resize_geometry(smart_resize_factor, patch_flatten)?;

    Ok(())
}

pub(super) fn validate_postprocess(
    postprocess: &[ProcessorRecipePostprocess],
) -> Result<(), RecipeError> {
    for (index, descriptor) in postprocess.iter().enumerate() {
        match descriptor {
            ProcessorRecipePostprocess::ObjectDetection(object_detection) => {
                validate_output_name(index, "logits_output", &object_detection.logits_output)?;
                validate_output_name(index, "boxes_output", &object_detection.boxes_output)?;
                if let Some(num_labels_with_background) =
                    object_detection.num_labels_with_background
                {
                    validate_class_count(index, num_labels_with_background)?;
                }
                if object_detection.top_k == Some(0) {
                    return Err(RecipeError::InvalidPostprocessQueryCount {
                        postprocess_index: index,
                        value: 0,
                    });
                }
                validate_score_threshold(
                    index,
                    "score_threshold",
                    object_detection.score_threshold,
                )?;
                validate_image_size_source(
                    index,
                    "target_size_source",
                    &object_detection.target_size_source,
                )?;
            }
            ProcessorRecipePostprocess::Segmentation(segmentation) => {
                validate_output_name(
                    index,
                    "class_logits_output",
                    &segmentation.class_logits_output,
                )?;
                validate_output_name(
                    index,
                    "mask_logits_output",
                    &segmentation.mask_logits_output,
                )?;
                if let Some(num_queries) = segmentation.num_queries {
                    validate_query_count(index, num_queries)?;
                }
                if let Some(num_labels_with_background) = segmentation.num_labels_with_background {
                    validate_class_count(index, num_labels_with_background)?;
                }
                if let Some(mask_size) = segmentation.mask_size {
                    validate_postprocess_image_size(index, "mask_size", mask_size)?;
                }
                validate_score_threshold(
                    index,
                    "score_threshold",
                    segmentation.options.score_threshold,
                )?;
                validate_score_threshold(
                    index,
                    "mask_threshold",
                    segmentation.options.mask_threshold,
                )?;
                validate_iou_threshold(
                    index,
                    "overlap_mask_area_threshold",
                    segmentation.options.overlap_mask_area_threshold,
                )?;
                if let Some(source) = &segmentation.options.target_size_source {
                    validate_image_size_source(index, "target_size_source", source)?;
                }
            }
            ProcessorRecipePostprocess::BinaryMasks(binary_masks) => {
                validate_output_name(
                    index,
                    "mask_logits_output",
                    &binary_masks.mask_logits_output,
                )?;
                if let Some(mask_size) = binary_masks.mask_size {
                    validate_postprocess_image_size(index, "mask_size", mask_size)?;
                }
                if let Some(masks_per_image) = binary_masks.masks_per_image {
                    validate_mask_count(index, masks_per_image)?;
                }
                validate_image_size_source(
                    index,
                    "original_size_source",
                    &binary_masks.original_size_source,
                )?;
                validate_image_size_source(
                    index,
                    "reshaped_input_size_source",
                    &binary_masks.reshaped_input_size_source,
                )?;
                validate_score_threshold(index, "mask_threshold", binary_masks.mask_threshold)?;
            }
            ProcessorRecipePostprocess::Depth(depth) => {
                validate_output_name(index, "depth_output", &depth.depth_output)?;
                if let Some(source) = &depth.target_size_source {
                    validate_image_size_source(index, "target_size_source", source)?;
                }
            }
            ProcessorRecipePostprocess::DenseMap(dense_map) => {
                validate_output_name(index, "output", &dense_map.output)?;
                if let Some(channels) = dense_map.channels {
                    validate_channel_count(index, channels)?;
                }
                validate_target_names(index, &dense_map.target_names)?;
                if let Some(source) = &dense_map.target_size_source {
                    validate_image_size_source(index, "target_size_source", source)?;
                }
            }
            ProcessorRecipePostprocess::VideoTensor(video_tensor) => {
                validate_output_name(index, "output", &video_tensor.output)?;
            }
            ProcessorRecipePostprocess::Coordinates(coordinates) => {
                validate_output_name(index, "coordinates_output", &coordinates.coordinates_output)?;
                if let Some(scores_output) = &coordinates.scores_output {
                    validate_output_name(index, "scores_output", scores_output)?;
                }
                validate_image_size_source(
                    index,
                    "target_size_source",
                    &coordinates.target_size_source,
                )?;
            }
            ProcessorRecipePostprocess::TokenSequence(token_sequence) => {
                validate_output_name(index, "logits_output", &token_sequence.logits_output)?;
            }
            ProcessorRecipePostprocess::OutputHook(output_hook) => {
                validate_output_name(index, "output", &output_hook.output)?;
                if let Some(source) = &output_hook.target_size_source {
                    validate_image_size_source(index, "target_size_source", source)?;
                }
            }
        }
    }

    Ok(())
}

fn validate_output_name(
    postprocess_index: usize,
    field: &'static str,
    name: &str,
) -> Result<(), RecipeError> {
    if name.trim().is_empty() {
        Err(RecipeError::EmptyPostprocessOutputName {
            postprocess_index,
            field,
        })
    } else {
        Ok(())
    }
}

fn validate_image_size_source(
    postprocess_index: usize,
    field: &'static str,
    source: &RecipeImageSizeSource,
) -> Result<(), RecipeError> {
    match source {
        RecipeImageSizeSource::Other(name) if name.trim().is_empty() => {
            Err(RecipeError::EmptyPostprocessSourceName {
                postprocess_index,
                field,
            })
        }
        RecipeImageSizeSource::OriginalSizes
        | RecipeImageSizeSource::ReshapedInputSizes
        | RecipeImageSizeSource::CallerProvided
        | RecipeImageSizeSource::Other(_) => Ok(()),
    }
}

fn validate_class_count(postprocess_index: usize, value: usize) -> Result<(), RecipeError> {
    if value < 2 {
        Err(RecipeError::InvalidPostprocessClassCount {
            postprocess_index,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_query_count(postprocess_index: usize, value: usize) -> Result<(), RecipeError> {
    if value == 0 {
        Err(RecipeError::InvalidPostprocessQueryCount {
            postprocess_index,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_mask_count(postprocess_index: usize, value: usize) -> Result<(), RecipeError> {
    if value == 0 {
        Err(RecipeError::InvalidPostprocessMaskCount {
            postprocess_index,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_channel_count(postprocess_index: usize, value: usize) -> Result<(), RecipeError> {
    if value == 0 {
        Err(RecipeError::InvalidPostprocessChannelCount {
            postprocess_index,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_target_names(
    postprocess_index: usize,
    target_names: &[String],
) -> Result<(), RecipeError> {
    for (target_index, target_name) in target_names.iter().enumerate() {
        if target_name.trim().is_empty() {
            return Err(RecipeError::EmptyPostprocessTargetName {
                postprocess_index,
                target_index,
            });
        }
    }
    Ok(())
}

fn validate_score_threshold(
    postprocess_index: usize,
    field: &'static str,
    value: f32,
) -> Result<(), RecipeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RecipeError::InvalidPostprocessThreshold {
            postprocess_index,
            field,
            value,
        })
    }
}

fn validate_iou_threshold(
    postprocess_index: usize,
    field: &'static str,
    value: f32,
) -> Result<(), RecipeError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(RecipeError::InvalidPostprocessThreshold {
            postprocess_index,
            field,
            value,
        })
    }
}

pub(super) fn validate_postprocess_image_size(
    postprocess_index: usize,
    field: &'static str,
    size: ImageSize,
) -> Result<(), RecipeError> {
    ImageSize::new(size.height, size.width)
        .map(|_| ())
        .map_err(|source| RecipeError::InvalidPostprocessImageSize {
            postprocess_index,
            field,
            source,
        })
}

fn validate_image_size_constraint_stage(
    stage_index: usize,
    constraints: ImageSizeConstraints,
) -> Result<(), RecipeError> {
    let min_side = constraints.min_side_length.max(1);
    validate_transform(
        stage_index,
        "validate_image_size",
        validate_image_size_constraints(
            ImageSize {
                height: min_side,
                width: min_side,
            },
            constraints,
        ),
    )
}

fn validate_crop_stage(stage_index: usize, crop: RecipeCropStage) -> Result<(), RecipeError> {
    match crop {
        RecipeCropStage::Center { size } => validate_transform(
            stage_index,
            crop.kind(),
            ImageSize::new(size.height, size.width),
        ),
        RecipeCropStage::Absolute { crop_box } => {
            let image_size = ImageSize {
                height: crop_box.y_max.max(1),
                width: crop_box.x_max.max(1),
            };
            validate_transform(
                stage_index,
                crop.kind(),
                validate_image_crop_box(crop_box, image_size),
            )
        }
    }
}

fn validate_pad_stage(
    stage_index: usize,
    pad: &RecipePadStage,
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "pad",
        padded_image_size(PROBE_SIZE, pad.padding),
    )?;
    validate_constant_fill(stage_index, "pad", &pad.fill, pixel_format)
}

fn validate_pad_canvas_stage(
    stage_index: usize,
    padding: Padding,
    fill: &CanvasFill,
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "pad_canvas",
        padded_image_size(PROBE_SIZE, padding),
    )?;

    if let CanvasFill::Constant(fill) = fill {
        validate_constant_fill(stage_index, "pad_canvas", fill, pixel_format)?;
    }

    Ok(())
}

fn validate_pad_to_multiple_stage(
    stage_index: usize,
    multiples: ImageSize,
    fill: &CanvasFill,
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "pad_to_multiple",
        pad_to_multiple_plan(PROBE_SIZE, multiples),
    )?;

    if let CanvasFill::Constant(fill) = fill {
        validate_constant_fill(stage_index, "pad_to_multiple", fill, pixel_format)?;
    }

    Ok(())
}

fn validate_symmetric_next_multiple_stage(
    stage_index: usize,
    multiples: ImageSize,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "pad_symmetric_to_next_multiple",
        ImageSize::new(multiples.height, multiples.width),
    )
}

fn validate_round_to_multiple_stage(
    stage_index: usize,
    requested_size: Option<ImageSize>,
    multiples: ImageSize,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "round_to_multiple",
        ImageSize::new(multiples.height, multiples.width),
    )?;

    if let Some(requested_size) = requested_size {
        validate_transform(
            stage_index,
            "round_to_multiple",
            multiple_of_resize_plan(requested_size, Some(requested_size), multiples),
        )?;
    }

    Ok(())
}

fn validate_resize_fill_stage(
    stage_index: usize,
    size: ImageSize,
    fill: &CanvasFill,
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "resize_fill",
        resize_fill_plan(PROBE_SIZE, size),
    )?;

    if let CanvasFill::Constant(fill) = fill {
        validate_constant_fill(stage_index, "resize_fill", fill, pixel_format)?;
    }

    Ok(())
}

fn validate_constant_fill(
    stage_index: usize,
    stage: &'static str,
    fill: &[u8],
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    match pixel_format {
        Some(pixel_format) => validate_transform(
            stage_index,
            stage,
            validate_padding_fill(fill, pixel_format),
        ),
        None if fill.is_empty() => validate_transform(
            stage_index,
            stage,
            validate_padding_fill(fill, PixelFormat::Luma8),
        ),
        None => Ok(()),
    }
}

fn validate_resize_stage(index: usize, resize: RecipeResizeStage) -> Result<(), RecipeError> {
    match resize.target {
        RecipeResizeTarget::Fixed { size } => {
            if size.height == 0 || size.width == 0 {
                return Err(RecipeError::InvalidResizeTarget { size });
            }
        }
        RecipeResizeTarget::Dynamic => {}
        RecipeResizeTarget::SmartResize { limits } => validate_smart_resize_limits(index, limits)?,
        RecipeResizeTarget::ShortestEdge {
            shortest_edge,
            longest_edge,
        } => {
            validate_transform(
                index,
                resize.target.kind(),
                shortest_edge_resize_size(PROBE_SIZE, shortest_edge, longest_edge),
            )?;
        }
        RecipeResizeTarget::LongestEdge { longest_edge } => {
            validate_transform(
                index,
                resize.target.kind(),
                longest_edge_resize_size(PROBE_SIZE, longest_edge),
            )?;
        }
    }
    ResizeDecision::new(resize.filter, resize.parity).map_err(|source| {
        RecipeError::InvalidResizeParity {
            stage_index: index,
            source,
        }
    })?;
    Ok(())
}

fn validate_smart_resize_limits(index: usize, limits: ResizeLimits) -> Result<(), RecipeError> {
    PROBE_SIZE
        .smart_resize(limits)
        .map(|_| ())
        .map_err(|source| RecipeError::InvalidSmartResizeLimits {
            stage_index: index,
            source,
        })
}

fn validate_patch_stage(index: usize, patch: RecipePatchStage) -> Result<(), RecipeError> {
    match patch.source_layout {
        Layout::NCHW | Layout::NHWC => {}
        Layout::NC
        | Layout::CHW
        | Layout::HWC
        | Layout::NPCHW
        | Layout::NPHWC
        | Layout::BFCHW
        | Layout::BFHWC
        | Layout::NIPCHW
        | Layout::NIPHWC => {
            return Err(RecipeError::UnsupportedPatchSourceLayout {
                stage_index: index,
                layout: patch.source_layout,
            });
        }
    }
    validate_patch_dimension(index, "patch_size", patch.patch_size)?;
    validate_patch_dimension(index, "temporal_patch_size", patch.temporal_patch_size)?;
    validate_patch_dimension(index, "merge_size", patch.merge_size)
}

fn validate_patch_dimension(
    stage_index: usize,
    field: &'static str,
    value: usize,
) -> Result<(), RecipeError> {
    if value == 0 {
        Err(RecipeError::InvalidPatchDimension {
            stage_index,
            field,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_frame_sampling_stage(
    stage_index: usize,
    sampling: FrameSampling,
) -> Result<(), RecipeError> {
    if sampling.stride == 0 {
        return Err(RecipeError::InvalidFrameSamplingStride {
            stage_index,
            stride: sampling.stride,
        });
    }
    if let Some(max_frames) = sampling.max_frames {
        if max_frames == 0 {
            return Err(RecipeError::InvalidFrameSamplingLimit {
                stage_index,
                max_frames,
            });
        }
    }
    Ok(())
}

fn validate_patch_grid_stage(
    stage_index: usize,
    patch_grid: &RecipePatchGridStage,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "patch_grid",
        ImageSize::new(patch_grid.base_size.height, patch_grid.base_size.width),
    )?;
    validate_transform(
        stage_index,
        "patch_grid",
        patch_grid_plan(
            PROBE_SIZE,
            &patch_grid.candidate_sizes,
            patch_grid.patch_size,
        ),
    )
}

fn validate_image_split_stage(
    stage_index: usize,
    split: RecipeImageSplitStage,
) -> Result<(), RecipeError> {
    if split.do_resize {
        validate_transform(
            stage_index,
            "image_split",
            split_image_plan(PROBE_SIZE, split.longest_edge, split.max_image_size),
        )
    } else {
        validate_transform(
            stage_index,
            "image_split",
            split_image_encoder_size(PROBE_SIZE, split.max_image_size),
        )
    }
}

fn validate_aspect_ratio_crops_stage(
    stage_index: usize,
    aspect_ratio_crops: RecipeAspectRatioCropStage,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "aspect_ratio_crops",
        aspect_ratio_crop_plan(
            ImageSize {
                height: 2,
                width: 4,
            },
            aspect_ratio_crops.options,
        ),
    )
}

fn validate_tiled_canvas_stage(
    stage_index: usize,
    tile: RecipeTiledCanvasStage,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "tiled_canvas",
        tiled_canvas_plan(PROBE_SIZE, tile.tile_size, tile.max_image_tiles),
    )
}

fn validate_document_geometry_stage(
    stage_index: usize,
    document: RecipeDocumentGeometryStage,
) -> Result<(), RecipeError> {
    validate_transform(
        stage_index,
        "document_geometry",
        ImageSize::new(document.target_size.height, document.target_size.width),
    )?;
    if document.do_align_long_axis {
        validate_transform(
            stage_index,
            "document_geometry",
            should_rotate_to_match_orientation(PROBE_SIZE, document.target_size),
        )?;
    }
    if document.do_resize {
        validate_transform(
            stage_index,
            "document_geometry",
            shortest_edge_resize_size(
                PROBE_SIZE,
                document.target_size.height.min(document.target_size.width),
                None,
            ),
        )?;
    }
    if document.do_thumbnail {
        validate_transform(
            stage_index,
            "document_geometry",
            fit_inside_size(PROBE_SIZE, document.target_size),
        )?;
    }
    if document.do_pad {
        validate_transform(
            stage_index,
            "document_geometry",
            centered_padding(PROBE_SIZE, document.target_size),
        )?;
    }
    Ok(())
}

fn validate_patch_resize_geometry(
    smart_resize_factor: Option<usize>,
    patch_flatten: Option<(usize, RecipePatchStage)>,
) -> Result<(), RecipeError> {
    let (Some(resize_factor), Some((stage_index, patch))) = (smart_resize_factor, patch_flatten)
    else {
        return Ok(());
    };
    let expected = patch
        .patch_size
        .checked_mul(patch.merge_size)
        .ok_or(RecipeError::PatchGeometryOverflow { stage_index })?;
    if resize_factor != expected {
        return Err(RecipeError::InvalidPatchGeometry {
            stage_index,
            resize_factor,
            patch_size: patch.patch_size,
            merge_size: patch.merge_size,
        });
    }
    Ok(())
}

fn validate_normalization_stats(
    stage_index: usize,
    field: &'static str,
    values: &[f32],
    pixel_format: Option<PixelFormat>,
) -> Result<(), RecipeError> {
    if values.is_empty() {
        return Err(RecipeError::EmptyNormalizationStats { stage_index, field });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(RecipeError::NonFiniteNormalizationStat {
                stage_index,
                field,
                index,
                value,
            });
        }
    }
    if let Some(pixel_format) = pixel_format {
        let channels = pixel_format.channels();
        if values.len() != 1 && values.len() != channels {
            return Err(RecipeError::InvalidNormalizationStats {
                stage_index,
                channels,
                actual: values.len(),
            });
        }
    }
    Ok(())
}

fn validate_positive_std(stage_index: usize, values: &[f32]) -> Result<(), RecipeError> {
    for (index, value) in values.iter().copied().enumerate() {
        if value <= 0.0 {
            return Err(RecipeError::NonPositiveNormalizationStd {
                stage_index,
                index,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_output(
    output: ProcessorRecipeOutput,
    stages: &[ProcessorRecipeStage],
) -> Result<(), RecipeError> {
    match output.layout {
        Layout::NCHW | Layout::NHWC => Ok(()),
        Layout::NC
            if stages
                .iter()
                .any(|stage| matches!(stage, ProcessorRecipeStage::PatchFlatten { .. })) =>
        {
            Ok(())
        }
        Layout::NC
        | Layout::CHW
        | Layout::HWC
        | Layout::NPCHW
        | Layout::NPHWC
        | Layout::BFCHW
        | Layout::BFHWC
        | Layout::NIPCHW
        | Layout::NIPHWC => Err(RecipeError::UnsupportedOutputLayout(output.layout)),
    }
}

pub(super) fn generic_output_layout(
    output: ProcessorRecipeOutput,
    stages: &[ProcessorRecipeStage],
) -> Layout {
    stages
        .iter()
        .find_map(|stage| match stage {
            ProcessorRecipeStage::PatchFlatten { patch } => Some(patch.source_layout),
            _ => None,
        })
        .unwrap_or(output.layout)
}
