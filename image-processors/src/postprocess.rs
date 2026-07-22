//! Task-specific output restoration helpers.
//!
//! This module groups postprocessing APIs that restore model outputs back to
//! image coordinates or user-facing mask and segmentation structures. The
//! underlying implementations remain shared with transform primitives, so
//! existing `transforms` paths continue to work.

use crate::media::{ImageFrame, MediaError, PixelFormat};
use crate::output::{
    ProcessorMetadataName, ProcessorMetadataValue, ProcessorOutput, ProcessorTensorName,
};
use crate::recipe::{
    ProcessorRecipe, ProcessorRecipePostprocess, RecipeBinaryMaskPostprocess,
    RecipeCoordinatePostprocess, RecipeCoordinateTask, RecipeDenseMapPostprocess,
    RecipeDenseMapTask, RecipeDepthPostprocess, RecipeDepthUnit, RecipeDetectionScoreMode,
    RecipeImageSizeSource, RecipeObjectDetectionPostprocess, RecipeOutputHookPostprocess,
    RecipeOutputHookTask, RecipeSegmentationPostprocess, RecipeSegmentationStrategy,
    RecipeSegmentationTask, RecipeTokenSequenceDecoder, RecipeTokenSequencePostprocess,
    RecipeTokenSequenceTask, RecipeTokenVocabularySource, RecipeVideoTensorPostprocess,
    RecipeVideoTransferFunction,
};
use crate::tensor::{Layout, Tensor, TensorData, TensorError, TensorLeadingAxis};

#[doc(inline)]
pub use crate::transforms::{
    binarize_mask, binary_mask_to_box, binary_mask_to_rle, binary_rle_to_mask, box_near_crop_edge,
    detection_box_iou, filter_generated_masks, generate_layered_crop_boxes, logc3_to_linear,
    mask_stability_score, non_max_suppression, normalized_point_grid, pad_crop_mask,
    post_process_binary_mask, post_process_generated_masks, post_process_instance_segmentation,
    post_process_object_detection, post_process_panoptic_segmentation,
    post_process_semantic_segmentation, post_process_sigmoid_best_object_detection,
    post_process_sigmoid_top_k_object_detection, resize_padded_mask_logits, scale_detection_box,
    scale_image_point, BinaryRleMask, CropGenerationOptions, DetectionBoundingBox,
    DetectionCenterBox, FilteredMask, GeneratedMaskPrediction, ImagePoint, ImageSize,
    LayeredCropBox, MaskFilterOptions, ObjectDetectionPrediction, SegmentationPostProcessOptions,
    SegmentationPostProcessPrediction, SegmentationSegmentInfo, SemanticSegmentationPrediction,
    TransformError,
};

mod query_mask;
mod types;

pub use types::*;

/// Restores an image tensor into owned image frames.
///
/// The input tensor values are interpreted as unit-range image values. Values
/// outside `[0, 1]` are clamped before conversion to `u8`. Channel counts of
/// 1, 3, and 4 produce luma, RGB, and RGBA frames respectively. Batched image
/// layouts return one frame per sample in tensor order.
///
/// # Errors
///
/// Returns an error when the tensor layout is not `CHW`, `HWC`, `NCHW`, or
/// `NHWC`, the channel count is unsupported, size arithmetic overflows, or the
/// generated frame violates image-frame invariants.
pub fn post_process_image_tensor(
    tensor: &Tensor,
) -> Result<Vec<ImageFrame>, TensorPostprocessError> {
    let values = tensor.data().to_vec::<f32>();
    match tensor.layout() {
        Layout::CHW => {
            let [channels, height, width] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            Ok(vec![image_frame_from_chw_unit_values(
                &values, size, *channels,
            )?])
        }
        Layout::HWC => {
            let [height, width, channels] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            Ok(vec![image_frame_from_hwc_unit_values(
                &values, size, *channels,
            )?])
        }
        Layout::NCHW => {
            let [batch, channels, height, width] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            post_process_nchw_images(&values, *batch, *channels, size)
        }
        Layout::NHWC => {
            let [batch, height, width, channels] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            post_process_nhwc_images(&values, *batch, *channels, size)
        }
        layout => Err(TensorPostprocessError::UnsupportedLayout(layout)),
    }
}

/// Restores an image or video tensor into owned video clips.
///
/// Batched video layouts `BFCHW` and `BFHWC` return one clip per batch item,
/// with frames kept in temporal order. For `NCHW` and `NHWC` tensors whose
/// leading axis is marked as [`TensorLeadingAxis::Frames`], the samples are
/// returned as one unbatched clip; otherwise each sample is returned as a
/// single-frame clip. `CHW` and `HWC` tensors return one single-frame clip.
///
/// # Errors
///
/// Returns an error when the tensor layout has no image axes, the channel count
/// is unsupported, size arithmetic overflows, or generated frames violate
/// image-frame invariants.
pub fn post_process_video_tensor(
    tensor: &Tensor,
) -> Result<Vec<Vec<ImageFrame>>, TensorPostprocessError> {
    let values = tensor.data().to_vec::<f32>();
    match tensor.layout() {
        Layout::BFCHW => {
            let [batch, frames, channels, height, width] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            post_process_bfchw_videos(&values, *batch, *frames, *channels, size)
        }
        Layout::BFHWC => {
            let [batch, frames, height, width, channels] = tensor.shape() else {
                return Err(TensorPostprocessError::UnsupportedLayout(tensor.layout()));
            };
            let size = ImageSize::new(*height, *width)?;
            post_process_bfhwc_videos(&values, *batch, *frames, *channels, size)
        }
        Layout::CHW | Layout::HWC => post_process_image_tensor(tensor).map(|frames| vec![frames]),
        Layout::NCHW | Layout::NHWC => {
            let frames = post_process_image_tensor(tensor)?;
            if tensor.leading_axis() == Some(TensorLeadingAxis::Frames) {
                Ok(vec![frames])
            } else {
                Ok(frames.into_iter().map(|frame| vec![frame]).collect())
            }
        }
        layout => Err(TensorPostprocessError::UnsupportedLayout(layout)),
    }
}

/// Restores a decoded LogC3 video tensor to linear HDR values.
///
/// The input tensor is interpreted as VAE-decoded values in `[-1, 1]`, which
/// are first denormalized to clamped `[0, 1]` LogC3 code values. The returned
/// tensor stores linear HDR `f32` values in `BFHWC` layout, matching this
/// crate's channel-last video array convention.
///
/// # Errors
///
/// Returns an error when the tensor layout is not `BFCHW` or `BFHWC`, the
/// tensor contains non-finite values, layout conversion fails, or output tensor
/// construction fails.
pub fn post_process_logc3_hdr_video_tensor(
    tensor: &Tensor,
) -> Result<Tensor, TensorPostprocessError> {
    let channel_last = match tensor.layout() {
        Layout::BFCHW => tensor.to_layout(Layout::BFHWC)?,
        Layout::BFHWC => tensor.clone(),
        layout => return Err(TensorPostprocessError::UnsupportedLayout(layout)),
    };
    let denormalized =
        crate::transforms::denormalize_signed_to_unit(&channel_last.data().to_vec::<f32>())?;
    let values = crate::transforms::logc3_to_linear(&denormalized)?;
    Ok(Tensor::new(
        TensorData::F32(values),
        channel_last.shape().to_vec(),
        Layout::BFHWC,
    )?)
}

/// Executes the postprocessing descriptors attached to a processor recipe.
///
/// `model_outputs` should contain named tensors produced by the model head
/// (for example `logits`, `pred_boxes`, or `pred_masks`). `preprocessing_output`
/// should be the [`ProcessorOutput`] returned by the corresponding processor
/// preprocessing call, so descriptor size sources such as `original_sizes` and
/// `reshaped_input_sizes` can be resolved.
///
/// # Errors
///
/// Returns an error when a referenced tensor or metadata item is missing, a
/// tensor shape is incompatible with the descriptor, a descriptor kind is not
/// executable yet, or the underlying task postprocess helper rejects the
/// supplied values.
pub fn post_process_recipe_outputs(
    recipe: &ProcessorRecipe,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<Vec<RecipePostprocessOutput>, RecipePostprocessError> {
    recipe
        .postprocess()
        .iter()
        .enumerate()
        .map(|(descriptor_index, descriptor)| match descriptor {
            ProcessorRecipePostprocess::ObjectDetection(object_detection) => {
                execute_object_detection_descriptor(
                    descriptor_index,
                    object_detection,
                    model_outputs,
                    preprocessing_output,
                    context,
                )
            }
            ProcessorRecipePostprocess::Segmentation(segmentation) => {
                execute_segmentation_descriptor(
                    descriptor_index,
                    segmentation,
                    model_outputs,
                    preprocessing_output,
                    context,
                )
            }
            ProcessorRecipePostprocess::BinaryMasks(binary_masks) => {
                execute_binary_masks_descriptor(
                    descriptor_index,
                    binary_masks,
                    model_outputs,
                    preprocessing_output,
                    context,
                )
            }
            ProcessorRecipePostprocess::Depth(depth) => execute_depth_descriptor(
                descriptor_index,
                depth,
                model_outputs,
                preprocessing_output,
                context,
            ),
            ProcessorRecipePostprocess::DenseMap(dense_map) => execute_dense_map_descriptor(
                descriptor_index,
                dense_map,
                model_outputs,
                preprocessing_output,
                context,
            ),
            ProcessorRecipePostprocess::VideoTensor(video_tensor) => {
                execute_video_tensor_descriptor(descriptor_index, video_tensor, model_outputs)
            }
            ProcessorRecipePostprocess::Coordinates(coordinates) => execute_coordinate_descriptor(
                descriptor_index,
                coordinates,
                model_outputs,
                preprocessing_output,
                context,
            ),
            ProcessorRecipePostprocess::TokenSequence(token_sequence) => {
                execute_token_sequence_descriptor(descriptor_index, token_sequence, model_outputs)
            }
            ProcessorRecipePostprocess::OutputHook(output_hook) => execute_output_hook_descriptor(
                descriptor_index,
                output_hook,
                model_outputs,
                preprocessing_output,
                context,
            ),
        })
        .collect()
}

fn execute_object_detection_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeObjectDetectionPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let logits_tensor = tensor_by_output_name(model_outputs, &descriptor.logits_output)?;
    let boxes_tensor = tensor_by_output_name(model_outputs, &descriptor.boxes_output)?;
    let (batch, num_queries, inferred_labels) =
        object_detection_logits_shape(&descriptor.logits_output, logits_tensor)?;
    let (box_batch, box_queries) =
        object_detection_boxes_shape(&descriptor.boxes_output, boxes_tensor)?;

    if batch != box_batch || num_queries != box_queries {
        return Err(RecipePostprocessError::IncompatibleTensorShapes {
            left: descriptor.logits_output.clone(),
            left_shape: logits_tensor.shape().to_vec(),
            right: descriptor.boxes_output.clone(),
            right_shape: boxes_tensor.shape().to_vec(),
        });
    }

    let num_labels = descriptor
        .num_labels_with_background
        .unwrap_or(inferred_labels);
    if num_labels != inferred_labels {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.logits_output.clone(),
            expected: "last dimension to match num_labels_with_background",
            actual: logits_tensor.shape().to_vec(),
        });
    }

    let target_sizes = resolve_image_sizes(
        &descriptor.target_size_source,
        preprocessing_output,
        context,
    )?;
    validate_size_batch(&descriptor.target_size_source, target_sizes, batch)?;

    let logits = logits_tensor.data().to_vec::<f32>();
    let pred_boxes = detection_center_boxes_from_tensor(&descriptor.boxes_output, boxes_tensor)?;
    let logits_per_sample = checked_mul(num_queries, num_labels)?;

    let mut predictions = Vec::with_capacity(batch);
    for (sample_index, &target_size) in target_sizes.iter().enumerate() {
        let logits_start = checked_mul(sample_index, logits_per_sample)?;
        let boxes_start = checked_mul(sample_index, num_queries)?;
        let sample_logits = &logits[logits_start..logits_start + logits_per_sample];
        let sample_boxes = &pred_boxes[boxes_start..boxes_start + num_queries];
        let sample_predictions = match descriptor.score_mode {
            RecipeDetectionScoreMode::SoftmaxWithBackground => {
                crate::transforms::post_process_object_detection(
                    sample_logits,
                    sample_boxes,
                    num_labels,
                    target_size,
                    descriptor.score_threshold,
                )
            }
            RecipeDetectionScoreMode::SigmoidBestPerQuery => {
                crate::transforms::post_process_sigmoid_best_object_detection(
                    sample_logits,
                    sample_boxes,
                    num_labels,
                    target_size,
                    descriptor.score_threshold,
                )
            }
            RecipeDetectionScoreMode::SigmoidTopK => {
                crate::transforms::post_process_sigmoid_top_k_object_detection(
                    sample_logits,
                    sample_boxes,
                    num_labels,
                    target_size,
                    descriptor.score_threshold,
                    descriptor.top_k.unwrap_or(num_queries),
                )
            }
        }?;
        predictions.push(sample_predictions);
    }

    Ok(RecipePostprocessOutput::ObjectDetection {
        descriptor_index,
        predictions,
    })
}

fn execute_segmentation_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeSegmentationPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let class_tensor = tensor_by_output_name(model_outputs, &descriptor.class_logits_output)?;
    let mask_tensor = tensor_by_output_name(model_outputs, &descriptor.mask_logits_output)?;
    let (batch, inferred_queries, inferred_labels) =
        query_class_logits_shape(&descriptor.class_logits_output, class_tensor)?;
    let (mask_batch, mask_queries, inferred_mask_size) =
        query_mask_logits_shape(&descriptor.mask_logits_output, mask_tensor)?;

    if batch != mask_batch || inferred_queries != mask_queries {
        return Err(RecipePostprocessError::IncompatibleTensorShapes {
            left: descriptor.class_logits_output.clone(),
            left_shape: class_tensor.shape().to_vec(),
            right: descriptor.mask_logits_output.clone(),
            right_shape: mask_tensor.shape().to_vec(),
        });
    }

    let num_queries = descriptor.num_queries.unwrap_or(inferred_queries);
    if num_queries != inferred_queries {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.class_logits_output.clone(),
            expected: "query dimension to match num_queries",
            actual: class_tensor.shape().to_vec(),
        });
    }
    let num_labels_with_background = descriptor
        .num_labels_with_background
        .unwrap_or(inferred_labels);
    if num_labels_with_background != inferred_labels {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.class_logits_output.clone(),
            expected: "last dimension to match num_labels_with_background",
            actual: class_tensor.shape().to_vec(),
        });
    }

    let mask_size = descriptor.mask_size.unwrap_or(inferred_mask_size);
    if mask_size != inferred_mask_size {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.mask_logits_output.clone(),
            expected: "mask spatial dimensions to match mask_size",
            actual: mask_tensor.shape().to_vec(),
        });
    }

    let target_sizes = descriptor
        .options
        .target_size_source
        .as_ref()
        .map(|source| {
            let sizes = resolve_image_sizes(source, preprocessing_output, context)?;
            validate_size_batch(source, sizes, batch)?;
            Ok::<&[ImageSize], RecipePostprocessError>(sizes)
        })
        .transpose()?;
    let class_logits = class_tensor.data().to_vec::<f32>();
    let mask_logits = mask_tensor.data().to_vec::<f32>();
    let class_logits_per_sample = checked_mul(num_queries, num_labels_with_background)?;
    let mask_pixels = checked_pixels(mask_size)?;
    let mask_logits_per_sample = checked_mul(num_queries, mask_pixels)?;

    match descriptor.task {
        RecipeSegmentationTask::Semantic => {
            let mut predictions = Vec::with_capacity(batch);
            for sample_index in 0..batch {
                let class_start = checked_mul(sample_index, class_logits_per_sample)?;
                let mask_start = checked_mul(sample_index, mask_logits_per_sample)?;
                let sample_classes =
                    &class_logits[class_start..class_start + class_logits_per_sample];
                let sample_masks = &mask_logits[mask_start..mask_start + mask_logits_per_sample];
                let target_size = target_sizes.map(|sizes| sizes[sample_index]);
                let prediction = match descriptor.strategy {
                    RecipeSegmentationStrategy::OneFormer => {
                        query_mask::post_process_oneformer_semantic(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            target_size,
                        )?
                    }
                    RecipeSegmentationStrategy::Mask2Former
                    | RecipeSegmentationStrategy::Eomt { .. } => {
                        let intermediate_size = match descriptor.strategy {
                            RecipeSegmentationStrategy::Mask2Former => ImageSize::new(384, 384)?,
                            RecipeSegmentationStrategy::Eomt { size } => size,
                            _ => mask_size,
                        };
                        let resized_masks = query_mask::resize_mask_planes_bilinear(
                            sample_masks,
                            num_queries,
                            mask_size,
                            intermediate_size,
                        )?;
                        crate::transforms::post_process_semantic_segmentation(
                            sample_classes,
                            &resized_masks,
                            num_queries,
                            num_labels_with_background,
                            intermediate_size,
                            target_size,
                        )?
                    }
                    RecipeSegmentationStrategy::Standard
                    | RecipeSegmentationStrategy::MaskFormer => {
                        crate::transforms::post_process_semantic_segmentation(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            target_size,
                        )?
                    }
                };
                predictions.push(prediction);
            }
            Ok(RecipePostprocessOutput::SemanticSegmentation {
                descriptor_index,
                predictions,
            })
        }
        RecipeSegmentationTask::Instance => {
            let mut predictions = Vec::with_capacity(batch);
            for sample_index in 0..batch {
                let class_start = checked_mul(sample_index, class_logits_per_sample)?;
                let mask_start = checked_mul(sample_index, mask_logits_per_sample)?;
                let sample_classes =
                    &class_logits[class_start..class_start + class_logits_per_sample];
                let sample_masks = &mask_logits[mask_start..mask_start + mask_logits_per_sample];
                let options = segmentation_options(descriptor, target_sizes, sample_index);
                let prediction = match descriptor.strategy {
                    RecipeSegmentationStrategy::Standard => {
                        crate::transforms::post_process_instance_segmentation(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::MaskFormer => {
                        query_mask::post_process_maskformer_instance(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::Mask2Former => {
                        let intermediate_size = ImageSize::new(384, 384)?;
                        let resized_masks = query_mask::resize_mask_planes_bilinear(
                            sample_masks,
                            num_queries,
                            mask_size,
                            intermediate_size,
                        )?;
                        query_mask::post_process_maskformer_instance(
                            sample_classes,
                            &resized_masks,
                            num_queries,
                            num_labels_with_background,
                            intermediate_size,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::Eomt { size } => {
                        let target_size = options.target_size.ok_or(
                            RecipePostprocessError::UnsupportedDescriptor {
                                task: "eomt_instance_requires_target_sizes",
                            },
                        )?;
                        let restored_masks = query_mask::restore_eomt_mask_logits(
                            sample_masks,
                            num_queries,
                            mask_size,
                            size,
                            target_size,
                        )?;
                        query_mask::post_process_eomt_instance(
                            sample_classes,
                            &restored_masks,
                            num_queries,
                            num_labels_with_background,
                            target_size,
                            options.score_threshold,
                        )?
                    }
                    RecipeSegmentationStrategy::OneFormer => {
                        return Err(RecipePostprocessError::ExternalMetadataRequired {
                            strategy: "one_former",
                            task: "instance_segmentation",
                            metadata: "class_info_file and thing class metadata",
                        });
                    }
                };
                predictions.push(prediction);
            }
            Ok(RecipePostprocessOutput::InstanceSegmentation {
                descriptor_index,
                predictions,
            })
        }
        RecipeSegmentationTask::Panoptic => {
            let mut predictions = Vec::with_capacity(batch);
            for sample_index in 0..batch {
                let class_start = checked_mul(sample_index, class_logits_per_sample)?;
                let mask_start = checked_mul(sample_index, mask_logits_per_sample)?;
                let sample_classes =
                    &class_logits[class_start..class_start + class_logits_per_sample];
                let sample_masks = &mask_logits[mask_start..mask_start + mask_logits_per_sample];
                let options = segmentation_options(descriptor, target_sizes, sample_index);
                let prediction = match descriptor.strategy {
                    RecipeSegmentationStrategy::Standard
                    | RecipeSegmentationStrategy::MaskFormer => {
                        crate::transforms::post_process_panoptic_segmentation(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            &descriptor.label_ids_to_fuse,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::OneFormer => {
                        crate::transforms::post_process_oneformer_panoptic_segmentation(
                            sample_classes,
                            sample_masks,
                            num_queries,
                            num_labels_with_background,
                            mask_size,
                            &descriptor.label_ids_to_fuse,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::Mask2Former => {
                        let intermediate_size = ImageSize::new(384, 384)?;
                        let resized_masks = query_mask::resize_mask_planes_bilinear(
                            sample_masks,
                            num_queries,
                            mask_size,
                            intermediate_size,
                        )?;
                        crate::transforms::post_process_panoptic_segmentation(
                            sample_classes,
                            &resized_masks,
                            num_queries,
                            num_labels_with_background,
                            intermediate_size,
                            &descriptor.label_ids_to_fuse,
                            options,
                        )?
                    }
                    RecipeSegmentationStrategy::Eomt { size } => {
                        let target_size = options.target_size.ok_or(
                            RecipePostprocessError::UnsupportedDescriptor {
                                task: "eomt_panoptic_requires_target_sizes",
                            },
                        )?;
                        let restored_masks = query_mask::restore_eomt_mask_logits(
                            sample_masks,
                            num_queries,
                            mask_size,
                            size,
                            target_size,
                        )?;
                        query_mask::post_process_eomt_panoptic(
                            sample_classes,
                            &restored_masks,
                            num_queries,
                            num_labels_with_background,
                            target_size,
                            &descriptor.label_ids_to_fuse,
                            options,
                        )?
                    }
                };
                predictions.push(prediction);
            }
            Ok(RecipePostprocessOutput::PanopticSegmentation {
                descriptor_index,
                predictions,
            })
        }
    }
}

fn execute_depth_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeDepthPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let depth_tensor = tensor_by_output_name(model_outputs, &descriptor.depth_output)?;
    let (batch, depth_size) = depth_tensor_shape(&descriptor.depth_output, depth_tensor)?;
    let target_sizes = if descriptor.resize_back {
        descriptor
            .target_size_source
            .as_ref()
            .map(|source| {
                let sizes = resolve_image_sizes(source, preprocessing_output, context)?;
                validate_size_batch(source, sizes, batch)?;
                Ok::<&[ImageSize], RecipePostprocessError>(sizes)
            })
            .transpose()?
    } else {
        None
    };
    let values = depth_tensor.data().to_vec::<f32>();
    let pixels_per_sample = checked_pixels(depth_size)?;

    let mut outputs = Vec::with_capacity(batch);
    for sample_index in 0..batch {
        let start = checked_mul(sample_index, pixels_per_sample)?;
        outputs.push(post_process_depth_map(
            &values[start..start + pixels_per_sample],
            depth_size,
            descriptor.unit,
            target_sizes.map(|sizes| sizes[sample_index]),
        )?);
    }

    Ok(RecipePostprocessOutput::Depth {
        descriptor_index,
        outputs,
    })
}

/// Restores one depth-like map to an optional target size.
///
/// # Errors
///
/// Returns an error when dimensions, value counts, or depth values are invalid.
pub fn post_process_depth_map(
    values: &[f32],
    source_size: ImageSize,
    unit: RecipeDepthUnit,
    target_size: Option<ImageSize>,
) -> Result<DepthMapPostprocessOutput, TransformError> {
    let output_size = target_size.unwrap_or(source_size);
    let values = crate::transforms::resize_f32_image_bilinear(values, source_size, output_size)?;
    Ok(DepthMapPostprocessOutput::new(output_size, unit, values))
}

fn execute_dense_map_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeDenseMapPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let tensor = tensor_by_output_name(model_outputs, &descriptor.output)?;
    let tensor_shape = dense_map_tensor_shape(&descriptor.output, tensor)?;
    if let Some(channels) = descriptor.channels {
        if channels != tensor_shape.channels {
            return Err(RecipePostprocessError::InvalidTensorShape {
                output: descriptor.output.clone(),
                expected: "channel dimension to match descriptor channels",
                actual: tensor.shape().to_vec(),
            });
        }
    }
    let target_count = dense_map_target_count(
        &descriptor.output,
        tensor.shape(),
        tensor_shape.batch,
        &descriptor.target_names,
    )?;
    let (target_sizes, target_size_mode) = if descriptor.resize_back {
        descriptor
            .target_size_source
            .as_ref()
            .map(|source| {
                let sizes = resolve_image_sizes(source, preprocessing_output, context)?;
                let mode = validate_dense_map_size_batch(
                    source,
                    sizes,
                    tensor_shape.batch,
                    target_count,
                    !descriptor.target_names.is_empty(),
                )?;
                Ok::<(&[ImageSize], DenseMapTargetSizeMode), RecipePostprocessError>((sizes, mode))
            })
            .transpose()?
            .map_or((None, None), |(sizes, mode)| (Some(sizes), Some(mode)))
    } else {
        (None, None)
    };
    let values = tensor.data().to_vec::<f32>();

    let mut outputs = Vec::with_capacity(tensor_shape.batch);
    for sample_index in 0..tensor_shape.batch {
        let sample_values = dense_map_sample_values(&values, tensor_shape, sample_index)?;
        let target_size = match (target_sizes, target_size_mode) {
            (Some(sizes), Some(mode)) => Some(sizes[mode.index(sample_index)]),
            _ => None,
        };
        outputs.push(post_process_dense_map_with_target_name(
            &sample_values,
            tensor_shape.size,
            tensor_shape.channels,
            descriptor.task,
            dense_map_target_name(&descriptor.target_names, sample_index),
            target_size,
        )?);
    }

    Ok(RecipePostprocessOutput::DenseMap {
        descriptor_index,
        outputs,
    })
}

fn execute_video_tensor_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeVideoTensorPostprocess,
    model_outputs: &ProcessorOutput,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let tensor = tensor_by_output_name(model_outputs, &descriptor.output)?;
    if !matches!(tensor.layout(), Layout::BFCHW | Layout::BFHWC) {
        return Err(RecipePostprocessError::UnsupportedTensorLayout {
            output: descriptor.output.clone(),
            expected: "BFCHW or BFHWC",
            actual: tensor.layout(),
        });
    }

    let tensor = match descriptor.transfer {
        RecipeVideoTransferFunction::LogC3 => post_process_logc3_hdr_video_tensor(tensor)
            .map_err(|error| tensor_postprocess_recipe_error(&descriptor.output, error))?,
    };

    Ok(RecipePostprocessOutput::VideoTensor {
        descriptor_index,
        output_name: descriptor.output.clone(),
        transfer: descriptor.transfer,
        tensor,
    })
}

/// Restores one dense image-like map to an optional target size.
///
/// `values` must be row-major pixels with channels interleaved per pixel. The
/// returned output uses the same value order after optional bilinear resizing.
///
/// # Errors
///
/// Returns an error when dimensions, channel count, value count, or map values
/// are invalid.
pub fn post_process_dense_map(
    values: &[f32],
    source_size: ImageSize,
    channels: usize,
    task: RecipeDenseMapTask,
    target_size: Option<ImageSize>,
) -> Result<DenseMapPostprocessOutput, TransformError> {
    post_process_dense_map_with_target_name(values, source_size, channels, task, None, target_size)
}

fn post_process_dense_map_with_target_name(
    values: &[f32],
    source_size: ImageSize,
    channels: usize,
    task: RecipeDenseMapTask,
    target_name: Option<String>,
    target_size: Option<ImageSize>,
) -> Result<DenseMapPostprocessOutput, TransformError> {
    if channels == 0 {
        return Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: values.len(),
        });
    }

    let expected = checked_dense_value_count(source_size, channels)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    for value in values.iter().copied() {
        if !value.is_finite() {
            return Err(TransformError::InvalidImageValue(value));
        }
    }

    let output_size = target_size.unwrap_or(source_size);
    ImageSize::new(output_size.height, output_size.width)?;
    if output_size == source_size {
        return Ok(DenseMapPostprocessOutput::new(
            task,
            target_name,
            output_size,
            channels,
            values.to_vec(),
        ));
    }

    let source_pixels = checked_transform_pixels(source_size)?;
    let output_pixels = checked_transform_pixels(output_size)?;
    let mut resized_planes = Vec::with_capacity(channels);
    for channel in 0..channels {
        let mut plane = Vec::with_capacity(source_pixels);
        for pixel_index in 0..source_pixels {
            plane.push(values[pixel_index * channels + channel]);
        }
        resized_planes.push(crate::transforms::resize_f32_image_bilinear(
            &plane,
            source_size,
            output_size,
        )?);
    }

    let output_values = interleave_channel_planes(&resized_planes, output_pixels)?;
    Ok(DenseMapPostprocessOutput::new(
        task,
        target_name,
        output_size,
        channels,
        output_values,
    ))
}

fn execute_coordinate_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeCoordinatePostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let coordinates_tensor = tensor_by_output_name(model_outputs, &descriptor.coordinates_output)?;
    let (batch, points_per_sample) =
        coordinate_tensor_shape(&descriptor.coordinates_output, coordinates_tensor)?;
    let target_sizes = resolve_image_sizes(
        &descriptor.target_size_source,
        preprocessing_output,
        context,
    )?;
    validate_size_batch(&descriptor.target_size_source, target_sizes, batch)?;

    let coordinates = coordinates_tensor.data().to_vec::<f32>();
    let scores =
        coordinate_scores_from_tensor(descriptor, model_outputs, batch, points_per_sample)?;
    let values_per_sample = checked_mul(points_per_sample, 2)?;

    let mut outputs = Vec::with_capacity(batch);
    for (sample_index, &target_size) in target_sizes.iter().enumerate() {
        let coordinate_start = checked_mul(sample_index, values_per_sample)?;
        let score_start = checked_mul(sample_index, points_per_sample)?;
        let sample_scores = scores
            .as_ref()
            .map(|scores| &scores[score_start..score_start + points_per_sample]);
        outputs.push(post_process_coordinates(
            &coordinates[coordinate_start..coordinate_start + values_per_sample],
            sample_scores,
            descriptor.task,
            target_size,
        )?);
    }

    Ok(RecipePostprocessOutput::Coordinates {
        descriptor_index,
        outputs,
    })
}

/// Restores normalized coordinate pairs to one target image size.
///
/// `coordinates` must contain row-major `[x, y]` pairs normalized to the unit
/// image extent. The restored `x` values are multiplied by `target_size.width`
/// and `y` values by `target_size.height`.
///
/// # Errors
///
/// Returns an error when coordinate or score lengths are inconsistent, or when
/// any coordinate or score value is non-finite.
pub fn post_process_coordinates(
    coordinates: &[f32],
    scores: Option<&[f32]>,
    task: RecipeCoordinateTask,
    target_size: ImageSize,
) -> Result<CoordinatePostprocessOutput, TransformError> {
    let point_count = coordinate_point_count(coordinates.len())?;
    if let Some(scores) = scores {
        validate_coordinate_scores_len(scores.len(), point_count)?;
    }

    let mut points = Vec::with_capacity(point_count);
    for point in coordinates.chunks_exact(2) {
        let normalized = [point[0], point[1]];
        validate_coordinate_point(normalized)?;
        points.push(CoordinatePoint::new(
            normalized[0] * target_size.width as f32,
            normalized[1] * target_size.height as f32,
        )?);
    }

    let scores = scores
        .map(|scores| {
            scores
                .iter()
                .copied()
                .map(|score| {
                    validate_coordinate_score(score)?;
                    Ok(score)
                })
                .collect::<Result<Vec<_>, TransformError>>()
        })
        .transpose()?;

    Ok(CoordinatePostprocessOutput::new(
        task,
        target_size,
        points,
        scores,
    ))
}

fn execute_token_sequence_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeTokenSequencePostprocess,
    model_outputs: &ProcessorOutput,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let tensor = tensor_by_output_name(model_outputs, &descriptor.logits_output)?;
    let [batch, steps, vocabulary] = tensor.shape() else {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.logits_output.clone(),
            expected: "[batch, steps, vocabulary]",
            actual: tensor.shape().to_vec(),
        });
    };
    if *batch == 0 || *steps == 0 || *vocabulary == 0 {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.logits_output.clone(),
            expected: "positive [batch, steps, vocabulary] dimensions",
            actual: tensor.shape().to_vec(),
        });
    }
    let values = tensor.data().to_vec::<f32>();
    let outputs = decode_token_sequences(descriptor, &values, *batch, *steps, *vocabulary)?;

    Ok(RecipePostprocessOutput::TokenSequences {
        descriptor_index,
        outputs,
    })
}

/// Decodes row-major `[batch, steps, vocabulary]` logits for recipe execution
/// and processor-specific structured-output adapters.
pub(crate) fn decode_token_sequences(
    descriptor: &RecipeTokenSequencePostprocess,
    values: &[f32],
    batch: usize,
    steps: usize,
    vocabulary: usize,
) -> Result<Vec<TokenSequencePostprocessOutput>, RecipePostprocessError> {
    if batch == 0 || steps == 0 || vocabulary == 0 {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.logits_output.clone(),
            expected: "positive [batch, steps, vocabulary] dimensions",
            actual: vec![batch, steps, vocabulary],
        });
    }
    let expected = checked_mul(checked_mul(batch, steps)?, vocabulary)?;
    if values.len() != expected {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.logits_output.clone(),
            expected: "[batch, steps, vocabulary] values",
            actual: vec![values.len()],
        });
    }
    validate_decoder_token_ids(descriptor, vocabulary)?;

    let mut outputs = Vec::with_capacity(batch);
    for sample_index in 0..batch {
        let mut token_ids = Vec::with_capacity(steps);
        let mut scores = Vec::with_capacity(steps);
        let mut previous_token_id = None;
        for step_index in 0..steps {
            let start = (sample_index * steps + step_index) * vocabulary;
            let row = &values[start..start + vocabulary];
            let (token_id, score) = finite_argmax(&descriptor.logits_output, start, row)?;
            match descriptor.decoder {
                RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id } => {
                    if token_id != blank_token_id && previous_token_id != Some(token_id) {
                        token_ids.push(token_id);
                        scores.push(score);
                    }
                    previous_token_id = Some(token_id);
                }
                RecipeTokenSequenceDecoder::Greedy {
                    begin_token_id,
                    end_token_id,
                } => {
                    if end_token_id == Some(token_id) {
                        if step_index > 0 {
                            break;
                        }
                        continue;
                    }
                    if begin_token_id != Some(token_id) {
                        token_ids.push(token_id);
                        scores.push(score);
                    }
                }
            }
        }
        outputs.push(TokenSequencePostprocessOutput::new(
            descriptor.task,
            token_ids,
            scores,
            descriptor.vocabulary_source,
        ));
    }
    Ok(outputs)
}

fn validate_decoder_token_ids(
    descriptor: &RecipeTokenSequencePostprocess,
    vocabulary: usize,
) -> Result<(), RecipePostprocessError> {
    match descriptor.decoder {
        RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id } => {
            validate_decoder_token_id(descriptor, "blank_token_id", blank_token_id, vocabulary)
        }
        RecipeTokenSequenceDecoder::Greedy {
            begin_token_id,
            end_token_id,
        } => {
            if let Some(token_id) = begin_token_id {
                validate_decoder_token_id(descriptor, "begin_token_id", token_id, vocabulary)?;
            }
            if let Some(token_id) = end_token_id {
                validate_decoder_token_id(descriptor, "end_token_id", token_id, vocabulary)?;
            }
            Ok(())
        }
    }
}

fn validate_decoder_token_id(
    descriptor: &RecipeTokenSequencePostprocess,
    field: &'static str,
    token_id: usize,
    vocabulary: usize,
) -> Result<(), RecipePostprocessError> {
    if token_id < vocabulary {
        return Ok(());
    }
    Err(RecipePostprocessError::TokenIdOutsideVocabulary {
        output: descriptor.logits_output.clone(),
        field,
        token_id,
        vocabulary,
    })
}

fn finite_argmax(
    output: &str,
    offset: usize,
    values: &[f32],
) -> Result<(usize, f32), RecipePostprocessError> {
    let Some((&first, remaining)) = values.split_first() else {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_owned(),
            expected: "a non-empty vocabulary dimension",
            actual: Vec::new(),
        });
    };
    if !first.is_finite() {
        return Err(RecipePostprocessError::NonFiniteTensorValue {
            output: output.to_owned(),
            index: offset,
            value: first,
        });
    }

    let mut best = (0, first);
    for (index, &value) in remaining.iter().enumerate() {
        if !value.is_finite() {
            return Err(RecipePostprocessError::NonFiniteTensorValue {
                output: output.to_owned(),
                index: offset + index + 1,
                value,
            });
        }
        // Torch and NumPy argmax keep the first index when values tie.
        if value > best.1 {
            best = (index + 1, value);
        }
    }
    Ok(best)
}

fn execute_output_hook_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeOutputHookPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let tensor = tensor_by_output_name(model_outputs, &descriptor.output)?;
    let target_sizes = descriptor
        .target_size_source
        .as_ref()
        .map(|source| {
            let sizes = resolve_image_sizes(source, preprocessing_output, context)?;
            validate_output_hook_size_batch(source, sizes, tensor)?;
            Ok::<Vec<ImageSize>, RecipePostprocessError>(sizes.to_vec())
        })
        .transpose()?;

    Ok(RecipePostprocessOutput::OutputHook {
        descriptor_index,
        output: OutputHookPostprocessOutput::new(
            descriptor.task,
            descriptor.output.clone(),
            tensor.clone(),
            target_sizes,
        ),
    })
}

fn execute_binary_masks_descriptor(
    descriptor_index: usize,
    descriptor: &RecipeBinaryMaskPostprocess,
    model_outputs: &ProcessorOutput,
    preprocessing_output: &ProcessorOutput,
    context: RecipePostprocessContext<'_>,
) -> Result<RecipePostprocessOutput, RecipePostprocessError> {
    let mask_tensor = tensor_by_output_name(model_outputs, &descriptor.mask_logits_output)?;
    let (batch, inferred_masks_per_image, inferred_mask_size) =
        binary_mask_logits_shape(&descriptor.mask_logits_output, mask_tensor)?;
    let mask_size = descriptor.mask_size.unwrap_or(inferred_mask_size);
    let masks_per_image = descriptor
        .masks_per_image
        .unwrap_or(inferred_masks_per_image);
    if mask_size != inferred_mask_size || masks_per_image != inferred_masks_per_image {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: descriptor.mask_logits_output.clone(),
            expected:
                "[batch, masks_per_image, mask_height, mask_width] matching descriptor counts",
            actual: mask_tensor.shape().to_vec(),
        });
    }

    let original_sizes = resolve_image_sizes(
        &descriptor.original_size_source,
        preprocessing_output,
        context,
    )?;
    let reshaped_input_sizes = resolve_image_sizes(
        &descriptor.reshaped_input_size_source,
        preprocessing_output,
        context,
    )?;
    validate_size_batch(&descriptor.original_size_source, original_sizes, batch)?;
    validate_size_batch(
        &descriptor.reshaped_input_size_source,
        reshaped_input_sizes,
        batch,
    )?;
    let padded_size = preprocessing_pixel_values_size(preprocessing_output)?;

    let mask_logits = mask_tensor.data().to_vec::<f32>();
    let mask_pixels = checked_pixels(mask_size)?;
    let masks_per_sample = checked_mul(masks_per_image, mask_pixels)?;
    let mut outputs = Vec::with_capacity(batch);
    for (sample_index, (&original_size, &reshaped_input_size)) in
        original_sizes.iter().zip(reshaped_input_sizes).enumerate()
    {
        let sample_start = checked_mul(sample_index, masks_per_sample)?;
        let mut masks = Vec::with_capacity(masks_per_image);
        for mask_index in 0..masks_per_image {
            let start = sample_start + checked_mul(mask_index, mask_pixels)?;
            masks.push(post_process_binary_mask(
                &mask_logits[start..start + mask_pixels],
                mask_size,
                padded_size,
                reshaped_input_size,
                original_size,
                descriptor.mask_threshold,
            )?);
        }
        outputs.push(BinaryMaskPostprocessOutput::new(original_size, masks));
    }

    Ok(RecipePostprocessOutput::BinaryMasks {
        descriptor_index,
        outputs,
    })
}

fn tensor_by_output_name<'a>(
    output: &'a ProcessorOutput,
    name: &str,
) -> Result<&'a Tensor, RecipePostprocessError> {
    output
        .tensor(&tensor_name(name))
        .ok_or_else(|| RecipePostprocessError::MissingTensor {
            name: name.to_string(),
        })
}

fn tensor_name(name: &str) -> ProcessorTensorName {
    match name {
        "pixel_values" => ProcessorTensorName::PixelValues,
        "pixel_mask" => ProcessorTensorName::PixelMask,
        other => ProcessorTensorName::Other(other.to_string()),
    }
}

fn tensor_postprocess_recipe_error(
    output: &str,
    error: TensorPostprocessError,
) -> RecipePostprocessError {
    match error {
        TensorPostprocessError::UnsupportedLayout(layout) => {
            RecipePostprocessError::UnsupportedTensorLayout {
                output: output.to_string(),
                expected: "BFCHW or BFHWC",
                actual: layout,
            }
        }
        TensorPostprocessError::Tensor(error) => RecipePostprocessError::Tensor(error),
        TensorPostprocessError::Transform(error) => RecipePostprocessError::Transform(error),
    }
}

fn metadata_name(name: &str) -> ProcessorMetadataName {
    match name {
        "original_sizes" => ProcessorMetadataName::OriginalSizes,
        "reshaped_input_sizes" => ProcessorMetadataName::ReshapedInputSizes,
        "image_grid_thw" => ProcessorMetadataName::ImageGridThw,
        "image_patch_counts" => ProcessorMetadataName::ImagePatchCounts,
        "labels" => ProcessorMetadataName::Labels,
        other => ProcessorMetadataName::Other(other.to_string()),
    }
}

fn resolve_image_sizes<'a>(
    source: &RecipeImageSizeSource,
    preprocessing_output: &'a ProcessorOutput,
    context: RecipePostprocessContext<'a>,
) -> Result<&'a [ImageSize], RecipePostprocessError> {
    match source {
        RecipeImageSizeSource::OriginalSizes => preprocessing_output
            .original_sizes()
            .ok_or_else(|| missing_size_source(source)),
        RecipeImageSizeSource::ReshapedInputSizes => preprocessing_output
            .reshaped_input_sizes()
            .ok_or_else(|| missing_size_source(source)),
        RecipeImageSizeSource::CallerProvided => context
            .target_sizes()
            .ok_or_else(|| missing_size_source(source)),
        RecipeImageSizeSource::Other(name) => match preprocessing_output
            .metadata_value(&metadata_name(name))
        {
            Some(ProcessorMetadataValue::ImageSizes(sizes)) => Ok(sizes),
            Some(_) => Err(RecipePostprocessError::InvalidImageSizeMetadata { name: name.clone() }),
            None => Err(RecipePostprocessError::MissingImageSizeMetadata { name: name.clone() }),
        },
    }
}

fn missing_size_source(source: &RecipeImageSizeSource) -> RecipePostprocessError {
    RecipePostprocessError::MissingImageSizeMetadata {
        name: image_size_source_name(source),
    }
}

fn validate_size_batch(
    source: &RecipeImageSizeSource,
    sizes: &[ImageSize],
    expected: usize,
) -> Result<(), RecipePostprocessError> {
    if sizes.len() == expected {
        Ok(())
    } else {
        Err(RecipePostprocessError::IncompatibleImageSizeBatch {
            size_source: image_size_source_name(source),
            expected,
            actual: sizes.len(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum DenseMapTargetSizeMode {
    PerMap,
    PerImage { target_count: usize },
}

impl DenseMapTargetSizeMode {
    fn index(self, sample_index: usize) -> usize {
        match self {
            Self::PerMap => sample_index,
            Self::PerImage { target_count } => sample_index / target_count,
        }
    }
}

fn dense_map_target_count(
    output: &str,
    actual_shape: &[usize],
    batch: usize,
    target_names: &[String],
) -> Result<usize, RecipePostprocessError> {
    if target_names.is_empty() {
        return Ok(1);
    }

    let target_count = target_names.len();
    if batch.is_multiple_of(target_count) {
        Ok(target_count)
    } else {
        Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "batch dimension divisible by target_names length",
            actual: actual_shape.to_vec(),
        })
    }
}

fn dense_map_target_name(target_names: &[String], sample_index: usize) -> Option<String> {
    if target_names.is_empty() {
        None
    } else {
        Some(target_names[sample_index % target_names.len()].clone())
    }
}

fn validate_dense_map_size_batch(
    source: &RecipeImageSizeSource,
    sizes: &[ImageSize],
    batch: usize,
    target_count: usize,
    has_target_names: bool,
) -> Result<DenseMapTargetSizeMode, RecipePostprocessError> {
    if !has_target_names {
        validate_size_batch(source, sizes, batch)?;
        return Ok(DenseMapTargetSizeMode::PerMap);
    }

    let image_count = batch / target_count;
    if sizes.len() == image_count {
        Ok(DenseMapTargetSizeMode::PerImage { target_count })
    } else if sizes.len() == batch {
        Ok(DenseMapTargetSizeMode::PerMap)
    } else {
        Err(RecipePostprocessError::IncompatibleImageSizeBatch {
            size_source: image_size_source_name(source),
            expected: image_count,
            actual: sizes.len(),
        })
    }
}

fn validate_output_hook_size_batch(
    source: &RecipeImageSizeSource,
    sizes: &[ImageSize],
    tensor: &Tensor,
) -> Result<(), RecipePostprocessError> {
    if let Some(batch) = tensor.batch() {
        return validate_size_batch(source, sizes, batch);
    }

    if sizes.len() == 1 || tensor.shape().first().copied() == Some(sizes.len()) {
        Ok(())
    } else {
        Err(RecipePostprocessError::IncompatibleImageSizeBatch {
            size_source: image_size_source_name(source),
            expected: tensor.shape().first().copied().unwrap_or(1),
            actual: sizes.len(),
        })
    }
}

fn object_detection_logits_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize, usize), RecipePostprocessError> {
    query_class_logits_shape(output, tensor)
}

fn query_class_logits_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize, usize), RecipePostprocessError> {
    match tensor.shape() {
        [num_queries, num_labels] => Ok((1, *num_queries, *num_labels)),
        [batch, num_queries, num_labels] => Ok((*batch, *num_queries, *num_labels)),
        actual => Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "[num_queries, num_labels] or [batch, num_queries, num_labels]",
            actual: actual.to_vec(),
        }),
    }
}

fn query_mask_logits_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize, ImageSize), RecipePostprocessError> {
    match tensor.shape() {
        [num_queries, height, width] => {
            Ok((1, *num_queries, ImageSize::new(*height, *width)?))
        }
        [batch, num_queries, height, width] => {
            Ok((*batch, *num_queries, ImageSize::new(*height, *width)?))
        }
        actual => Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "[num_queries, mask_height, mask_width] or [batch, num_queries, mask_height, mask_width]",
            actual: actual.to_vec(),
        }),
    }
}

fn object_detection_boxes_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize), RecipePostprocessError> {
    match tensor.shape() {
        [num_queries, 4] => Ok((1, *num_queries)),
        [batch, num_queries, 4] => Ok((*batch, *num_queries)),
        actual => Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "[num_queries, 4] or [batch, num_queries, 4]",
            actual: actual.to_vec(),
        }),
    }
}

fn coordinate_tensor_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize), RecipePostprocessError> {
    match tensor.shape() {
        [points, 2] => Ok((1, *points)),
        [batch, points, 2] => Ok((*batch, *points)),
        actual => Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "[points, 2] or [batch, points, 2]",
            actual: actual.to_vec(),
        }),
    }
}

fn coordinate_scores_from_tensor(
    descriptor: &RecipeCoordinatePostprocess,
    model_outputs: &ProcessorOutput,
    batch: usize,
    points_per_sample: usize,
) -> Result<Option<Vec<f32>>, RecipePostprocessError> {
    let Some(output_name) = &descriptor.scores_output else {
        return Ok(None);
    };
    let scores_tensor = tensor_by_output_name(model_outputs, output_name)?;
    let scores = scores_tensor.data().to_vec::<f32>();
    let expected = checked_mul(batch, points_per_sample)?;
    if scores.len() != expected {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: output_name.clone(),
            expected: "one score value per coordinate point",
            actual: scores_tensor.shape().to_vec(),
        });
    }

    Ok(Some(scores))
}

fn depth_tensor_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, ImageSize), RecipePostprocessError> {
    let shape = tensor.shape();
    match shape {
        [height, width] => Ok((1, ImageSize::new(*height, *width)?)),
        [batch, height, width] => Ok((*batch, ImageSize::new(*height, *width)?)),
        [batch, ..] if shape.len() == 4 => {
            let channel_axis = tensor.layout().channel_axis();
            if shape[channel_axis] != 1 {
                return Err(invalid_depth_tensor_shape(output, shape));
            }
            let height_axis = tensor.layout().height_axis();
            let width_axis = tensor.layout().width_axis();
            Ok((
                *batch,
                ImageSize::new(shape[height_axis], shape[width_axis])?,
            ))
        }
        actual => Err(invalid_depth_tensor_shape(output, actual)),
    }
}

fn invalid_depth_tensor_shape(output: &str, actual: &[usize]) -> RecipePostprocessError {
    RecipePostprocessError::InvalidTensorShape {
        output: output.to_string(),
        expected: "[height, width], [batch, height, width], or single-channel batched image layout",
        actual: actual.to_vec(),
    }
}

#[derive(Clone, Copy)]
struct DenseMapTensorShape {
    batch: usize,
    size: ImageSize,
    channels: usize,
    format: DenseMapTensorFormat,
}

#[derive(Clone, Copy)]
enum DenseMapTensorFormat {
    Hw,
    Chw,
    Hwc,
    Nchw,
    Nhwc,
}

fn dense_map_tensor_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<DenseMapTensorShape, RecipePostprocessError> {
    let shape = tensor.shape();
    match shape {
        [height, width] => Ok(DenseMapTensorShape {
            batch: 1,
            size: ImageSize::new(*height, *width)?,
            channels: 1,
            format: DenseMapTensorFormat::Hw,
        }),
        [..] if shape.len() == 3 => match tensor.layout() {
            Layout::CHW => Ok(DenseMapTensorShape {
                batch: 1,
                size: ImageSize::new(shape[1], shape[2])?,
                channels: shape[0],
                format: DenseMapTensorFormat::Chw,
            }),
            Layout::HWC => Ok(DenseMapTensorShape {
                batch: 1,
                size: ImageSize::new(shape[0], shape[1])?,
                channels: shape[2],
                format: DenseMapTensorFormat::Hwc,
            }),
            _ => Err(invalid_dense_map_tensor_shape(output, shape)),
        },
        [..] if shape.len() == 4 => match tensor.layout() {
            Layout::NCHW => Ok(DenseMapTensorShape {
                batch: shape[0],
                size: ImageSize::new(shape[2], shape[3])?,
                channels: shape[1],
                format: DenseMapTensorFormat::Nchw,
            }),
            Layout::NHWC => Ok(DenseMapTensorShape {
                batch: shape[0],
                size: ImageSize::new(shape[1], shape[2])?,
                channels: shape[3],
                format: DenseMapTensorFormat::Nhwc,
            }),
            _ => Err(invalid_dense_map_tensor_shape(output, shape)),
        },
        actual => Err(invalid_dense_map_tensor_shape(output, actual)),
    }
}

fn invalid_dense_map_tensor_shape(output: &str, actual: &[usize]) -> RecipePostprocessError {
    RecipePostprocessError::InvalidTensorShape {
        output: output.to_string(),
        expected:
            "[height, width], unbatched CHW/HWC image layout, or batched NCHW/NHWC image layout",
        actual: actual.to_vec(),
    }
}

fn dense_map_sample_values(
    values: &[f32],
    shape: DenseMapTensorShape,
    sample_index: usize,
) -> Result<Vec<f32>, RecipePostprocessError> {
    let pixels = checked_pixels(shape.size)?;
    let values_per_sample = checked_mul(pixels, shape.channels)?;
    let sample_start = checked_mul(sample_index, values_per_sample)?;

    match shape.format {
        DenseMapTensorFormat::Hw | DenseMapTensorFormat::Hwc | DenseMapTensorFormat::Nhwc => {
            Ok(values[sample_start..sample_start + values_per_sample].to_vec())
        }
        DenseMapTensorFormat::Chw | DenseMapTensorFormat::Nchw => {
            let mut sample_values = Vec::with_capacity(values_per_sample);
            for pixel_index in 0..pixels {
                for channel in 0..shape.channels {
                    sample_values.push(values[sample_start + channel * pixels + pixel_index]);
                }
            }
            Ok(sample_values)
        }
    }
}

fn post_process_nchw_images(
    values: &[f32],
    batch: usize,
    channels: usize,
    size: ImageSize,
) -> Result<Vec<ImageFrame>, TensorPostprocessError> {
    let item_len = checked_dense_value_count(size, channels)?;
    validate_tensor_value_count(values.len(), checked_transform_mul(batch, item_len)?)?;

    values
        .chunks_exact(item_len)
        .take(batch)
        .map(|sample| image_frame_from_chw_unit_values(sample, size, channels))
        .collect()
}

fn post_process_nhwc_images(
    values: &[f32],
    batch: usize,
    channels: usize,
    size: ImageSize,
) -> Result<Vec<ImageFrame>, TensorPostprocessError> {
    let item_len = checked_dense_value_count(size, channels)?;
    validate_tensor_value_count(values.len(), checked_transform_mul(batch, item_len)?)?;

    values
        .chunks_exact(item_len)
        .take(batch)
        .map(|sample| image_frame_from_hwc_unit_values(sample, size, channels))
        .collect()
}

fn post_process_bfchw_videos(
    values: &[f32],
    batch: usize,
    frames: usize,
    channels: usize,
    size: ImageSize,
) -> Result<Vec<Vec<ImageFrame>>, TensorPostprocessError> {
    post_process_batched_video_values(values, batch, frames, channels, size, |sample| {
        image_frame_from_chw_unit_values(sample, size, channels)
    })
}

fn post_process_bfhwc_videos(
    values: &[f32],
    batch: usize,
    frames: usize,
    channels: usize,
    size: ImageSize,
) -> Result<Vec<Vec<ImageFrame>>, TensorPostprocessError> {
    post_process_batched_video_values(values, batch, frames, channels, size, |sample| {
        image_frame_from_hwc_unit_values(sample, size, channels)
    })
}

fn post_process_batched_video_values(
    values: &[f32],
    batch: usize,
    frames: usize,
    channels: usize,
    size: ImageSize,
    mut frame_from_sample: impl FnMut(&[f32]) -> Result<ImageFrame, TensorPostprocessError>,
) -> Result<Vec<Vec<ImageFrame>>, TensorPostprocessError> {
    let item_len = checked_dense_value_count(size, channels)?;
    let frame_count = checked_transform_mul(batch, frames)?;
    validate_tensor_value_count(values.len(), checked_transform_mul(frame_count, item_len)?)?;

    let mut samples = values.chunks_exact(item_len);
    let mut clips = Vec::with_capacity(batch);
    for _ in 0..batch {
        let mut clip = Vec::with_capacity(frames);
        for _ in 0..frames {
            let sample = samples.next().ok_or({
                TensorPostprocessError::Transform(TransformError::InvalidBufferLength {
                    expected: values.len(),
                    actual: 0,
                })
            })?;
            clip.push(frame_from_sample(sample)?);
        }
        clips.push(clip);
    }
    Ok(clips)
}

fn image_frame_from_hwc_unit_values(
    values: &[f32],
    size: ImageSize,
    channels: usize,
) -> Result<ImageFrame, TensorPostprocessError> {
    let expected = checked_dense_value_count(size, channels)?;
    validate_tensor_value_count(values.len(), expected)?;
    let pixel_format = pixel_format_for_tensor_channels(channels)?;
    let data = values
        .iter()
        .copied()
        .map(unit_f32_to_u8)
        .collect::<Vec<_>>();
    Ok(ImageFrame::new(size.width, size.height, pixel_format, data)
        .map_err(transform_error_from_media)?)
}

fn image_frame_from_chw_unit_values(
    values: &[f32],
    size: ImageSize,
    channels: usize,
) -> Result<ImageFrame, TensorPostprocessError> {
    let pixels = checked_transform_pixels(size)?;
    let expected = checked_dense_value_count(size, channels)?;
    validate_tensor_value_count(values.len(), expected)?;
    let pixel_format = pixel_format_for_tensor_channels(channels)?;
    let mut data = Vec::with_capacity(expected);
    for pixel_index in 0..pixels {
        for channel in 0..channels {
            data.push(unit_f32_to_u8(values[channel * pixels + pixel_index]));
        }
    }
    Ok(ImageFrame::new(size.width, size.height, pixel_format, data)
        .map_err(transform_error_from_media)?)
}

fn pixel_format_for_tensor_channels(
    channels: usize,
) -> Result<PixelFormat, TensorPostprocessError> {
    match channels {
        1 => Ok(PixelFormat::Luma8),
        3 => Ok(PixelFormat::Rgb8),
        4 => Ok(PixelFormat::Rgba8),
        channels => Err(TransformError::UnsupportedChannels(channels).into()),
    }
}

fn unit_f32_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn validate_tensor_value_count(
    actual: usize,
    expected: usize,
) -> Result<(), TensorPostprocessError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidBufferLength { expected, actual }.into())
    }
}

fn checked_dense_value_count(size: ImageSize, channels: usize) -> Result<usize, TransformError> {
    let pixels = checked_transform_pixels(size)?;
    pixels
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn checked_transform_pixels(size: ImageSize) -> Result<usize, TransformError> {
    ImageSize::new(size.height, size.width)?;
    size.height
        .checked_mul(size.width)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn checked_transform_mul(left: usize, right: usize) -> Result<usize, TransformError> {
    left.checked_mul(right)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn transform_error_from_media(error: MediaError) -> TransformError {
    match error {
        MediaError::InvalidFrameDimensions { width, height } => {
            TransformError::InvalidSize { height, width }
        }
        MediaError::InvalidBufferLength { expected, actual } => {
            TransformError::InvalidBufferLength { expected, actual }
        }
        MediaError::FrameSizeOverflow => TransformError::ImageSizeOverflow,
        error => TransformError::ImageFrameInvariant(error.to_string()),
    }
}

fn interleave_channel_planes(
    planes: &[Vec<f32>],
    pixels: usize,
) -> Result<Vec<f32>, TransformError> {
    if planes.is_empty() {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 0,
            actual: 0,
        });
    }

    for plane in planes {
        if plane.len() != pixels {
            return Err(TransformError::InvalidBufferLength {
                expected: pixels,
                actual: plane.len(),
            });
        }
    }

    let channels = planes.len();
    let value_count = pixels
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut values = Vec::with_capacity(value_count);
    for pixel_index in 0..pixels {
        for plane in planes {
            values.push(plane[pixel_index]);
        }
    }
    Ok(values)
}

fn detection_center_boxes_from_tensor(
    output: &str,
    tensor: &Tensor,
) -> Result<Vec<DetectionCenterBox>, RecipePostprocessError> {
    let values = tensor.data().to_vec::<f32>();
    if !values.len().is_multiple_of(4) {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "a final box-coordinate dimension of 4",
            actual: tensor.shape().to_vec(),
        });
    }

    values
        .chunks_exact(4)
        .map(|bbox| DetectionCenterBox::new(bbox[0], bbox[1], bbox[2], bbox[3]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(RecipePostprocessError::Transform)
}

fn binary_mask_logits_shape(
    output: &str,
    tensor: &Tensor,
) -> Result<(usize, usize, ImageSize), RecipePostprocessError> {
    match tensor.shape() {
        [masks_per_image, height, width] => Ok((
            1,
            *masks_per_image,
            ImageSize::new(*height, *width)?,
        )),
        [batch, masks_per_image, height, width] => Ok((
            *batch,
            *masks_per_image,
            ImageSize::new(*height, *width)?,
        )),
        actual => Err(RecipePostprocessError::InvalidTensorShape {
            output: output.to_string(),
            expected: "[masks_per_image, mask_height, mask_width] or [batch, masks_per_image, mask_height, mask_width]",
            actual: actual.to_vec(),
        }),
    }
}

fn preprocessing_pixel_values_size(
    preprocessing_output: &ProcessorOutput,
) -> Result<ImageSize, RecipePostprocessError> {
    let tensor = preprocessing_output.pixel_values().ok_or_else(|| {
        RecipePostprocessError::MissingTensor {
            name: "pixel_values".to_string(),
        }
    })?;
    let shape = tensor.shape();
    let height_axis = tensor.layout().height_axis();
    let width_axis = tensor.layout().width_axis();
    if height_axis >= shape.len() || width_axis >= shape.len() {
        return Err(RecipePostprocessError::InvalidTensorShape {
            output: "pixel_values".to_string(),
            expected: "an image tensor layout with height and width axes",
            actual: shape.to_vec(),
        });
    }

    Ok(ImageSize::new(shape[height_axis], shape[width_axis])?)
}

fn checked_pixels(size: ImageSize) -> Result<usize, RecipePostprocessError> {
    checked_mul(size.height, size.width)
}

fn checked_mul(left: usize, right: usize) -> Result<usize, RecipePostprocessError> {
    left.checked_mul(right)
        .ok_or(RecipePostprocessError::Tensor(
            TensorError::ShapeElementCountOverflow,
        ))
}

fn coordinate_point_count(actual: usize) -> Result<usize, TransformError> {
    if actual.is_multiple_of(2) {
        Ok(actual / 2)
    } else {
        Err(TransformError::InvalidBufferLength {
            expected: actual.saturating_add(1),
            actual,
        })
    }
}

fn validate_coordinate_scores_len(actual: usize, expected: usize) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidBufferLength { expected, actual })
    }
}

fn validate_coordinate_point(point: [f32; 2]) -> Result<(), TransformError> {
    if point.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformError::InvalidPoint { point })
    }
}

fn validate_coordinate_score(score: f32) -> Result<(), TransformError> {
    if score.is_finite() {
        Ok(())
    } else {
        Err(TransformError::InvalidScoreValue(score))
    }
}

fn segmentation_options(
    descriptor: &RecipeSegmentationPostprocess,
    target_sizes: Option<&[ImageSize]>,
    sample_index: usize,
) -> SegmentationPostProcessOptions {
    SegmentationPostProcessOptions {
        score_threshold: descriptor.options.score_threshold,
        mask_threshold: descriptor.options.mask_threshold,
        overlap_mask_area_threshold: descriptor.options.overlap_mask_area_threshold,
        target_size: target_sizes.map(|sizes| sizes[sample_index]),
    }
}

fn image_size_source_name(source: &RecipeImageSizeSource) -> String {
    match source {
        RecipeImageSizeSource::OriginalSizes => "original_sizes".to_string(),
        RecipeImageSizeSource::ReshapedInputSizes => "reshaped_input_sizes".to_string(),
        RecipeImageSizeSource::CallerProvided => "caller_provided".to_string(),
        RecipeImageSizeSource::Other(name) => name.clone(),
    }
}
