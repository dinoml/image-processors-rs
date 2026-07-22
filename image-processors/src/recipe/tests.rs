use super::*;

fn rgb_recipe(stages: Vec<ProcessorRecipeStage>) -> Result<ProcessorRecipe, RecipeError> {
    ProcessorRecipe::new(
        "test.rgb",
        ProcessorRecipeInput::default(),
        stages,
        ProcessorRecipeOutput::batch(Layout::NCHW),
    )
}

#[test]
fn new_rejects_empty_recipe_id() {
    let err = ProcessorRecipe::new(
        " ",
        ProcessorRecipeInput::default(),
        vec![ProcessorRecipeStage::Binarize],
        ProcessorRecipeOutput::batch(Layout::NCHW),
    )
    .unwrap_err();

    assert_eq!(err, RecipeError::EmptyId);
}

#[test]
fn new_rejects_zero_resize_target() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::Resize {
        resize: RecipeResizeStage::fixed(
            ImageSize {
                height: 0,
                width: 224,
            },
            ResizeMode::Default,
            ResizeFilter::Bilinear,
            ResizeParity::Compatibility,
        ),
    }])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidResizeTarget {
            size: ImageSize {
                height: 0,
                width: 224,
            }
        }
    );
}

#[test]
fn new_rejects_stats_that_do_not_match_known_pixel_format() {
    let err = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::Normalize {
            mean: vec![0.5, 0.5],
            std: vec![0.5, 0.5],
        },
    ])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidNormalizationStats {
            stage_index: 1,
            channels: 3,
            actual: 2,
        }
    );
}

#[test]
fn deserialize_validates_recipe_data() {
    let err = serde_json::from_str::<ProcessorRecipe>(
        r#"{
            "id": "bad.resize",
            "input": {},
            "stages": [
                {
                    "stage": "resize",
                    "resize": {
                        "target": {
                            "kind": "fixed",
                            "size": {
                                "height": 0,
                                "width": 16
                            }
                        },
                        "mode": "Default",
                        "filter": "Bilinear",
                        "parity": "Compatibility"
                        }
                }
            ],
            "output": {
                "layout": "NCHW",
                "leading_axis": "Batch"
            }
        }"#,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("resize stage target"),
        "unexpected serde error: {err}"
    );
}

#[test]
fn lowers_fixed_recipe_to_generic_image_config() {
    let recipe = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::fixed(
                ImageSize {
                    height: 224,
                    width: 224,
                },
                ResizeMode::Crop,
                ResizeFilter::Bicubic,
                ResizeParity::Compatibility,
            ),
        },
        ProcessorRecipeStage::Rescale {
            factor: 1.0 / 255.0,
        },
        ProcessorRecipeStage::Normalize {
            mean: vec![0.48145466, 0.4578275, 0.40821073],
            std: vec![0.26862954, 0.261_302_6, 0.275_777_1],
        },
    ])
    .unwrap();

    let config = recipe.to_image_processor_config().unwrap();

    assert_eq!(config.height, Some(224));
    assert_eq!(config.width, Some(224));
    assert_eq!(config.resize_mode, ResizeMode::Crop);
    assert_eq!(config.resample, ResizeFilter::Bicubic);
    assert_eq!(config.resize_parity, ResizeParity::Compatibility);
    assert_eq!(config.pixel_format, Some(PixelFormat::Rgb8));
    assert!(config.do_rescale);
    assert!(config.do_normalize);
    assert_eq!(config.output_layout, Layout::NCHW);
    crate::image::ImageProcessor::new(config).unwrap();
}

#[test]
fn lowering_rejects_duplicate_generic_stages() {
    let recipe = rgb_recipe(vec![
        ProcessorRecipeStage::Rescale { factor: 1.0 },
        ProcessorRecipeStage::Rescale { factor: 0.5 },
    ])
    .unwrap();

    let err = recipe.to_image_processor_config().unwrap_err();

    assert_eq!(err, RecipeError::DuplicateStage { stage: "rescale" });
}

#[test]
fn serialize_roundtrips_valid_recipe_through_validation() {
    let recipe = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::dynamic(
                ResizeMode::Default,
                ResizeFilter::Bilinear,
                ResizeParity::Compatibility,
            ),
        },
        ProcessorRecipeStage::Rescale {
            factor: 1.0 / 255.0,
        },
    ])
    .unwrap();

    let encoded = serde_json::to_string(&recipe).unwrap();
    let decoded: ProcessorRecipe = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, recipe);
}

#[test]
fn postprocess_descriptors_roundtrip_and_preserve_generic_lowering() {
    let mut segmentation = RecipeSegmentationPostprocess::new(
        RecipeSegmentationTask::Panoptic,
        "pred_logits",
        "pred_masks",
        Some(4),
    );
    segmentation.num_queries = Some(2);
    segmentation.mask_size = Some(ImageSize::new(16, 16).unwrap());
    segmentation.options.target_size_source = Some(RecipeImageSizeSource::OriginalSizes);
    segmentation.label_ids_to_fuse = vec![0, 1];

    let mut binary_masks = RecipeBinaryMaskPostprocess::new(
        "pred_masks",
        RecipeImageSizeSource::OriginalSizes,
        RecipeImageSizeSource::ReshapedInputSizes,
        0.0,
    );
    binary_masks.mask_size = Some(ImageSize::new(256, 256).unwrap());
    binary_masks.masks_per_image = Some(3);

    let recipe = ProcessorRecipe::new_with_postprocess(
        "test.postprocess",
        ProcessorRecipeInput::default(),
        vec![
            ProcessorRecipeStage::ConvertPixelFormat {
                format: PixelFormat::Rgb8,
            },
            ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::fixed(
                    ImageSize::new(224, 224).unwrap(),
                    ResizeMode::Default,
                    ResizeFilter::Bilinear,
                    ResizeParity::Compatibility,
                ),
            },
        ],
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![
            ProcessorRecipePostprocess::ObjectDetection(RecipeObjectDetectionPostprocess::new(
                "logits",
                "pred_boxes",
                Some(92),
                0.5,
                RecipeImageSizeSource::CallerProvided,
            )),
            ProcessorRecipePostprocess::Segmentation(segmentation),
            ProcessorRecipePostprocess::BinaryMasks(binary_masks),
            ProcessorRecipePostprocess::Depth(RecipeDepthPostprocess::new(
                "predicted_depth",
                RecipeDepthUnit::Relative,
                Some(RecipeImageSizeSource::OriginalSizes),
                true,
            )),
            ProcessorRecipePostprocess::DenseMap(
                RecipeDenseMapPostprocess::new(
                    RecipeDenseMapTask::IntrinsicImage,
                    "predicted_intrinsics",
                    Some(RecipeImageSizeSource::OriginalSizes),
                    true,
                )
                .with_channels(3)
                .with_target_names(vec!["albedo".to_string(), "shading".to_string()]),
            ),
            ProcessorRecipePostprocess::VideoTensor(RecipeVideoTensorPostprocess::new(
                "decoded_video",
                RecipeVideoTransferFunction::LogC3,
            )),
            ProcessorRecipePostprocess::Coordinates(RecipeCoordinatePostprocess::new(
                RecipeCoordinateTask::Keypoints,
                "keypoints",
                Some("scores".to_string()),
                RecipeImageSizeSource::CallerProvided,
            )),
            ProcessorRecipePostprocess::TokenSequence(RecipeTokenSequencePostprocess::new(
                RecipeTokenSequenceTask::Ocr,
                "recognition_logits",
                RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id: 0 },
            )),
            ProcessorRecipePostprocess::OutputHook(RecipeOutputHookPostprocess::new(
                RecipeOutputHookTask::TableStructure,
                "structure_logits",
                Some(RecipeImageSizeSource::Other("document_sizes".to_string())),
            )),
        ],
    )
    .unwrap();

    assert_eq!(recipe.postprocess().len(), 9);
    assert_eq!(
        recipe.to_image_processor_config().unwrap().output_layout,
        Layout::NCHW
    );

    let encoded = serde_json::to_string(&recipe).unwrap();
    let decoded: ProcessorRecipe = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, recipe);
}

#[test]
fn postprocess_only_recipe_roundtrips_and_rejects_generic_lowering() {
    let recipe = ProcessorRecipe::logc3_hdr_video_postprocess(
        "diffusers.ltx2_video_hdr_postprocess",
        "decoded_video",
    )
    .unwrap();

    assert!(recipe.is_postprocess_only());
    assert!(recipe.stages().is_empty());
    assert_eq!(recipe.output(), ProcessorRecipeOutput::batch(Layout::BFHWC));
    assert_eq!(recipe.postprocess().len(), 1);
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::PostprocessOnlyCannotLower
    );

    let encoded = serde_json::to_string(&recipe).unwrap();
    let decoded: ProcessorRecipe = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, recipe);
}

#[test]
fn postprocess_only_recipe_rejects_empty_descriptors() {
    let err = ProcessorRecipe::new_postprocess_only(
        "test.empty_postprocess",
        ProcessorRecipeInput::default(),
        ProcessorRecipeOutput::batch(Layout::BFHWC),
        Vec::new(),
    )
    .unwrap_err();

    assert_eq!(err, RecipeError::EmptyPostprocessDescriptors);
}

#[test]
fn postprocess_descriptors_reject_invalid_class_counts() {
    let err = ProcessorRecipe::new_with_postprocess(
        "test.bad_detection",
        ProcessorRecipeInput::default(),
        vec![ProcessorRecipeStage::Binarize],
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![ProcessorRecipePostprocess::ObjectDetection(
            RecipeObjectDetectionPostprocess::new(
                "logits",
                "pred_boxes",
                Some(1),
                0.5,
                RecipeImageSizeSource::CallerProvided,
            ),
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidPostprocessClassCount {
            postprocess_index: 0,
            value: 1,
        }
    );
}

#[test]
fn postprocess_descriptors_reject_invalid_channel_counts() {
    let err = ProcessorRecipe::new_with_postprocess(
        "test.bad_dense_map",
        ProcessorRecipeInput::default(),
        vec![ProcessorRecipeStage::Binarize],
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![ProcessorRecipePostprocess::DenseMap(
            RecipeDenseMapPostprocess::new(
                RecipeDenseMapTask::Uncertainty,
                "uncertainty",
                None,
                false,
            )
            .with_channels(0),
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidPostprocessChannelCount {
            postprocess_index: 0,
            value: 0,
        }
    );
}

#[test]
fn postprocess_descriptors_reject_empty_dense_map_target_names() {
    let err = ProcessorRecipe::new_with_postprocess(
        "test.bad_dense_map_target",
        ProcessorRecipeInput::default(),
        vec![ProcessorRecipeStage::Binarize],
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![ProcessorRecipePostprocess::DenseMap(
            RecipeDenseMapPostprocess::new(
                RecipeDenseMapTask::IntrinsicImage,
                "intrinsics",
                None,
                false,
            )
            .with_target_names(vec!["albedo".to_string(), " ".to_string()]),
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::EmptyPostprocessTargetName {
            postprocess_index: 0,
            target_index: 1,
        }
    );
}

#[test]
fn postprocess_descriptors_reject_invalid_output_names() {
    let err = ProcessorRecipe::new_with_postprocess(
        "test.bad_outputs",
        ProcessorRecipeInput::default(),
        vec![ProcessorRecipeStage::Binarize],
        ProcessorRecipeOutput::batch(Layout::NCHW),
        vec![ProcessorRecipePostprocess::Coordinates(
            RecipeCoordinatePostprocess::new(
                RecipeCoordinateTask::Pose,
                " ",
                None,
                RecipeImageSizeSource::CallerProvided,
            ),
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::EmptyPostprocessOutputName {
            postprocess_index: 0,
            field: "coordinates_output",
        }
    );
}

#[test]
fn deserialize_validates_postprocess_descriptors() {
    let err = serde_json::from_str::<ProcessorRecipe>(
        r#"{
            "id": "bad.postprocess",
            "input": {},
            "stages": [
                { "stage": "binarize" }
            ],
            "output": {
                "layout": "NCHW",
                "leading_axis": "Batch"
            },
            "postprocess": [
                {
                    "task": "binary_masks",
                    "parameters": {
                        "mask_logits_output": "pred_masks",
                        "mask_size": { "height": 0, "width": 256 },
                        "masks_per_image": 1,
                        "original_size_source": { "source": "original_sizes" },
                        "reshaped_input_size_source": { "source": "reshaped_input_sizes" },
                        "mask_threshold": 0.0
                    }
                }
            ]
        }"#,
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("field mask_size has invalid image size"),
        "unexpected serde error: {err}"
    );
}

#[test]
fn common_transform_recipe_stages_roundtrip_through_validation() {
    let recipe = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::SampleFrames {
            sampling: FrameSampling::every(2).with_max_frames(4),
        },
        ProcessorRecipeStage::TemporalRepeatLast { multiple: 4 },
        ProcessorRecipeStage::PatchGrid {
            patch_grid: RecipePatchGridStage::new(
                vec![ImageSize::new(336, 672).unwrap()],
                ImageSize::new(336, 336).unwrap(),
                336,
            ),
        },
        ProcessorRecipeStage::ImageSplit {
            split: RecipeImageSplitStage::new(364, 182, true),
        },
        ProcessorRecipeStage::AspectRatioCrops {
            aspect_ratio_crops: RecipeAspectRatioCropStage::new(AspectRatioCropOptions {
                min_crop_size: 2,
                max_num_crops: 4,
                min_ratio_to_activate: 1.1,
            }),
        },
        ProcessorRecipeStage::TiledCanvas {
            tile: RecipeTiledCanvasStage::new(224, 4),
        },
        ProcessorRecipeStage::DocumentGeometry {
            document: RecipeDocumentGeometryStage::new(
                ImageSize::new(2560, 1920).unwrap(),
                true,
                true,
                true,
                true,
            ),
        },
        ProcessorRecipeStage::ValidateImageSize {
            constraints: ImageSizeConstraints {
                max_aspect_ratio: 8.0,
                min_side_length: 64,
            },
        },
        ProcessorRecipeStage::SelectAspectRatioBucket {
            candidates: vec![
                ImageSize::new(704, 1408).unwrap(),
                ImageSize::new(1024, 1024).unwrap(),
            ],
        },
        ProcessorRecipeStage::SelectVideoSizeBucket {
            candidates: vec![
                VideoSizeBucket::new(8, ImageSize::new(720, 1280).unwrap()).unwrap(),
                VideoSizeBucket::new(16, ImageSize::new(1024, 1024).unwrap()).unwrap(),
            ],
        },
        ProcessorRecipeStage::AreaResize {
            target_area: 1024 * 1024,
            filter: ResizeFilter::Bilinear,
        },
        ProcessorRecipeStage::Crop {
            crop: RecipeCropStage::center(ImageSize::new(512, 512).unwrap()),
        },
        ProcessorRecipeStage::Crop {
            crop: RecipeCropStage::absolute(ImageCropBox::new(0, 0, 512, 512)),
        },
        ProcessorRecipeStage::ResizeCenterCrop {
            size: ImageSize::new(512, 512).unwrap(),
            rounding: ResizeRounding::Ceil,
            filter: ResizeFilter::Bilinear,
        },
        ProcessorRecipeStage::Pad {
            pad: RecipePadStage::constant(Padding::symmetric(2, 4), vec![0, 0, 0]),
        },
        ProcessorRecipeStage::PadCanvas {
            padding: Padding::all(2),
            fill: CanvasFill::ReflectImage,
        },
        ProcessorRecipeStage::PadToMultiple {
            multiples: ImageSize::new(8, 8).unwrap(),
            fill: CanvasFill::ImageEdges,
        },
        ProcessorRecipeStage::Overlay {
            overlay: RecipeOverlayStage::new(OverlayPosition::new(2, 3)),
        },
        ProcessorRecipeStage::MaskComposite,
        ProcessorRecipeStage::RoundToMultiple {
            requested_size: Some(ImageSize::new(65, 97).unwrap()),
            multiples: ImageSize::new(8, 16).unwrap(),
        },
        ProcessorRecipeStage::ResizeFill {
            size: ImageSize::new(512, 512).unwrap(),
            fill: CanvasFill::constant(vec![0, 0, 0]),
            filter: ResizeFilter::Bilinear,
        },
        ProcessorRecipeStage::ResizeFillRgb {
            size: ImageSize::new(512, 512).unwrap(),
            fill: RgbCanvasFill::ConstantRgb([0, 0, 0]),
            filter: ResizeFilter::Bilinear,
        },
        ProcessorRecipeStage::ConcatenateHorizontallyRgb {
            fill: [255, 255, 255],
        },
    ])
    .unwrap();

    let encoded = serde_json::to_string(&recipe).unwrap();
    let decoded: ProcessorRecipe = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, recipe);
}

#[test]
fn vlm_geometry_recipe_stages_reject_invalid_parameters() {
    let patch_grid_err = rgb_recipe(vec![ProcessorRecipeStage::PatchGrid {
        patch_grid: RecipePatchGridStage::new(Vec::new(), ImageSize::new(336, 336).unwrap(), 336),
    }])
    .unwrap_err();

    assert!(matches!(
        patch_grid_err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "patch_grid",
            source: TransformError::EmptyResolutionCandidates,
        }
    ));

    let split_err = rgb_recipe(vec![ProcessorRecipeStage::ImageSplit {
        split: RecipeImageSplitStage::new(0, 182, true),
    }])
    .unwrap_err();

    assert!(matches!(
        split_err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "image_split",
            source: TransformError::InvalidScaleFactor(0),
        }
    ));

    let pan_err = rgb_recipe(vec![ProcessorRecipeStage::AspectRatioCrops {
        aspect_ratio_crops: RecipeAspectRatioCropStage::new(AspectRatioCropOptions {
            min_crop_size: 2,
            max_num_crops: 4,
            min_ratio_to_activate: 0.0,
        }),
    }])
    .unwrap_err();

    assert!(matches!(
        pan_err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "aspect_ratio_crops",
            source: TransformError::InvalidAspectRatioCropActivationRatio(0.0),
        }
    ));

    let tile_err = rgb_recipe(vec![ProcessorRecipeStage::TiledCanvas {
        tile: RecipeTiledCanvasStage::new(224, 0),
    }])
    .unwrap_err();

    assert!(matches!(
        tile_err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "tiled_canvas",
            source: TransformError::InvalidScaleFactor(0),
        }
    ));

    let document_err = rgb_recipe(vec![ProcessorRecipeStage::DocumentGeometry {
        document: RecipeDocumentGeometryStage::new(
            ImageSize {
                height: 0,
                width: 1920,
            },
            true,
            true,
            false,
            true,
        ),
    }])
    .unwrap_err();

    assert!(matches!(
        document_err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "document_geometry",
            source: TransformError::InvalidSize {
                height: 0,
                width: 1920,
            },
        }
    ));
}

#[test]
fn vlm_geometry_recipe_stages_do_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::PatchGrid {
        patch_grid: RecipePatchGridStage::new(
            vec![ImageSize::new(336, 672).unwrap()],
            ImageSize::new(336, 336).unwrap(),
            336,
        ),
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "patch_grid",
        }
    );
}

#[test]
fn document_geometry_recipe_stage_does_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::DocumentGeometry {
        document: RecipeDocumentGeometryStage::new(
            ImageSize::new(2560, 1920).unwrap(),
            true,
            true,
            false,
            true,
        ),
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "document_geometry",
        }
    );
}

#[test]
fn sample_frames_recipe_stage_rejects_invalid_sampling() {
    let stride_err = rgb_recipe(vec![ProcessorRecipeStage::SampleFrames {
        sampling: FrameSampling {
            stride: 0,
            ..FrameSampling::default()
        },
    }])
    .unwrap_err();

    assert_eq!(
        stride_err,
        RecipeError::InvalidFrameSamplingStride {
            stage_index: 0,
            stride: 0,
        }
    );

    let limit_err = rgb_recipe(vec![ProcessorRecipeStage::SampleFrames {
        sampling: FrameSampling {
            max_frames: Some(0),
            ..FrameSampling::default()
        },
    }])
    .unwrap_err();

    assert_eq!(
        limit_err,
        RecipeError::InvalidFrameSamplingLimit {
            stage_index: 0,
            max_frames: 0,
        }
    );
}

#[test]
fn sample_frames_recipe_stage_does_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::SampleFrames {
        sampling: FrameSampling::every(2).with_max_frames(4),
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "sample_frames",
        }
    );
}

#[test]
fn temporal_repeat_last_recipe_stage_rejects_zero_multiple() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::TemporalRepeatLast {
        multiple: 0,
    }])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "temporal_repeat_last",
            source: TransformError::InvalidTemporalMultiple(0),
        }
    );
}

#[test]
fn temporal_repeat_last_recipe_stage_does_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::TemporalRepeatLast {
        multiple: 4,
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "temporal_repeat_last",
        }
    );
}

#[test]
fn common_transform_recipe_stage_rejects_invalid_parameters() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::AreaResize {
        target_area: 0,
        filter: ResizeFilter::Bilinear,
    }])
    .unwrap_err();

    assert!(matches!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "area_resize",
            source: TransformError::InvalidTargetArea(0),
        }
    ));
}

#[test]
fn crop_recipe_stage_rejects_empty_absolute_box() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::Crop {
        crop: RecipeCropStage::absolute(ImageCropBox::new(1, 0, 1, 1)),
    }])
    .unwrap_err();

    assert!(matches!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "absolute_crop",
            source: TransformError::InvalidImageCropBox { .. },
        }
    ));
}

#[test]
fn pad_recipe_stage_rejects_fill_that_does_not_match_known_pixel_format() {
    let err = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::Pad {
            pad: RecipePadStage::constant(Padding::all(1), vec![1, 2]),
        },
    ])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 1,
            stage: "pad",
            source: TransformError::InvalidPaddingFill {
                channels: 3,
                actual: 2,
            },
        }
    );
}

#[test]
fn resize_fill_recipe_stage_rejects_fill_that_does_not_match_known_pixel_format() {
    let err = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::ResizeFill {
            size: ImageSize::new(2, 2).unwrap(),
            fill: CanvasFill::constant(vec![1, 2]),
            filter: ResizeFilter::Bilinear,
        },
    ])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 1,
            stage: "resize_fill",
            source: TransformError::InvalidPaddingFill {
                channels: 3,
                actual: 2,
            },
        }
    );
}

#[test]
fn pad_canvas_recipe_stage_rejects_fill_that_does_not_match_known_pixel_format() {
    let err = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::PadCanvas {
            padding: Padding::all(1),
            fill: CanvasFill::constant(vec![1, 2]),
        },
    ])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 1,
            stage: "pad_canvas",
            source: TransformError::InvalidPaddingFill {
                channels: 3,
                actual: 2,
            },
        }
    );
}

#[test]
fn pad_to_multiple_recipe_stage_rejects_fill_that_does_not_match_known_pixel_format() {
    let err = rgb_recipe(vec![
        ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        },
        ProcessorRecipeStage::PadToMultiple {
            multiples: ImageSize::new(8, 8).unwrap(),
            fill: CanvasFill::constant(vec![1, 2]),
        },
    ])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 1,
            stage: "pad_to_multiple",
            source: TransformError::InvalidPaddingFill {
                channels: 3,
                actual: 2,
            },
        }
    );
}

#[test]
fn pad_to_multiple_recipe_stage_rejects_zero_multiple() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::PadToMultiple {
        multiples: ImageSize {
            height: 0,
            width: 8,
        },
        fill: CanvasFill::ImageEdges,
    }])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "pad_to_multiple",
            source: TransformError::InvalidSize {
                height: 0,
                width: 8,
            },
        }
    );
}

#[test]
fn pad_to_multiple_recipe_stage_does_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::PadToMultiple {
        multiples: ImageSize::new(8, 8).unwrap(),
        fill: CanvasFill::ImageEdges,
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "pad_to_multiple",
        }
    );
}

#[test]
fn overlay_and_mask_composite_stages_do_not_lower_to_generic_image_config() {
    let overlay_recipe = rgb_recipe(vec![ProcessorRecipeStage::Overlay {
        overlay: RecipeOverlayStage::new(OverlayPosition::origin()),
    }])
    .unwrap();

    assert_eq!(
        overlay_recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage { stage: "overlay" }
    );

    let mask_recipe = rgb_recipe(vec![ProcessorRecipeStage::MaskComposite]).unwrap();

    assert_eq!(
        mask_recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "mask_composite",
        }
    );
}

#[test]
fn recipe_only_transform_stages_do_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::AreaResize {
        target_area: 1024 * 1024,
        filter: ResizeFilter::Bilinear,
    }])
    .unwrap();

    let err = recipe.to_image_processor_config().unwrap_err();

    assert_eq!(
        err,
        RecipeError::UnsupportedGenericStage {
            stage: "area_resize",
        }
    );
}

#[test]
fn select_video_size_bucket_recipe_stage_rejects_invalid_candidate() {
    let err = rgb_recipe(vec![ProcessorRecipeStage::SelectVideoSizeBucket {
        candidates: vec![VideoSizeBucket {
            frame_count: 0,
            size: ImageSize::new(720, 1280).unwrap(),
        }],
    }])
    .unwrap_err();

    assert_eq!(
        err,
        RecipeError::InvalidTransformStage {
            stage_index: 0,
            stage: "select_video_size_bucket",
            source: TransformError::InvalidFrameCount(0),
        }
    );
}

#[test]
fn select_video_size_bucket_recipe_stage_does_not_lower_to_generic_image_config() {
    let recipe = rgb_recipe(vec![ProcessorRecipeStage::SelectVideoSizeBucket {
        candidates: vec![VideoSizeBucket::new(8, ImageSize::new(720, 1280).unwrap()).unwrap()],
    }])
    .unwrap();

    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "select_video_size_bucket",
        }
    );
}

#[test]
fn shortest_and_longest_edge_resize_targets_roundtrip_but_do_not_lower() {
    let recipe = rgb_recipe(vec![
        ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::shortest_edge(
                800,
                Some(1333),
                ResizeFilter::Bilinear,
                ResizeParity::Compatibility,
            ),
        },
        ProcessorRecipeStage::ResizeCenterCrop {
            size: ImageSize::new(512, 512).unwrap(),
            rounding: ResizeRounding::Floor,
            filter: ResizeFilter::Bilinear,
        },
    ])
    .unwrap();

    let encoded = serde_json::to_string(&recipe).unwrap();
    let decoded: ProcessorRecipe = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, recipe);
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "resize_shortest_edge",
        }
    );

    let longest_edge_recipe = rgb_recipe(vec![ProcessorRecipeStage::Resize {
        resize: RecipeResizeStage::longest_edge(
            1024,
            ResizeFilter::Bilinear,
            ResizeParity::Compatibility,
        ),
    }])
    .unwrap();

    assert_eq!(
        longest_edge_recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "resize_longest_edge",
        }
    );
}

#[test]
fn smart_resize_patch_recipe_lowers_to_patch_source_layout() {
    let recipe = ProcessorRecipe::new(
        "test.qwen",
        ProcessorRecipeInput::default(),
        vec![
            ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::smart(
                    ResizeLimits {
                        factor: 28,
                        min_pixels: 28 * 28,
                        max_pixels: 28 * 28 * 16,
                    },
                    ResizeMode::Default,
                    ResizeFilter::Bicubic,
                    ResizeParity::Compatibility,
                ),
            },
            ProcessorRecipeStage::PatchFlatten {
                patch: RecipePatchStage::flatten(Layout::NCHW, 14, 2, 2),
            },
        ],
        ProcessorRecipeOutput {
            layout: Layout::NC,
            leading_axis: TensorLeadingAxis::Batch,
        },
    )
    .unwrap();

    let config = recipe.to_image_processor_config().unwrap();

    assert!(config.do_resize);
    assert_eq!(config.height, None);
    assert_eq!(config.width, None);
    assert_eq!(config.output_layout, Layout::NCHW);
}

#[test]
fn patch_stage_computes_image_and_temporal_grid_metadata() {
    let patch = RecipePatchStage::flatten(Layout::NCHW, 14, 2, 2);
    let target = ImageSize::new(28, 56).unwrap();

    assert_eq!(patch.image_grid_thw(target).unwrap(), [1, 2, 4]);
    assert_eq!(patch.temporal_grid_thw(3, target).unwrap(), [2, 2, 4]);
    assert_eq!(patch.temporal_grid_size(4).unwrap(), 2);
}

#[test]
fn patch_stage_flattens_channel_last_temporal_tensor_with_repeat_last() {
    let patch = RecipePatchStage::flatten(Layout::HWC, 2, 2, 1);
    let target = ImageSize::new(2, 2).unwrap();
    let frame = Tensor::new(
        TensorData::F32(vec![
            1.0, 11.0, 21.0, 2.0, 12.0, 22.0, 3.0, 13.0, 23.0, 4.0, 14.0, 24.0,
        ]),
        vec![2, 2, 3],
        Layout::HWC,
    )
    .unwrap();

    let flattened = patch
        .flatten_temporal_tensors(&[frame], target)
        .expect("channel-last frame should flatten");

    assert_eq!(flattened.layout(), Layout::NC);
    assert_eq!(flattened.shape(), [1, 24]);
    assert_eq!(
        flattened.data().to_vec::<f32>(),
        vec![
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0,
            21.0, 22.0, 23.0, 24.0, 21.0, 22.0, 23.0, 24.0,
        ]
    );
}

#[test]
fn patch_stage_repeats_last_frame_for_odd_temporal_groups_metamorphic() {
    // Metamorphic coverage for the shared executor, not upstream fixture evidence.
    let patch = RecipePatchStage::flatten(Layout::CHW, 2, 2, 1);
    let target = ImageSize::new(2, 2).unwrap();
    let frames = [
        Tensor::new(
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 2, 2],
            Layout::CHW,
        )
        .unwrap(),
        Tensor::new(
            TensorData::F32(vec![5.0, 6.0, 7.0, 8.0]),
            vec![1, 2, 2],
            Layout::CHW,
        )
        .unwrap(),
        Tensor::new(
            TensorData::F32(vec![9.0, 10.0, 11.0, 12.0]),
            vec![1, 2, 2],
            Layout::CHW,
        )
        .unwrap(),
    ];

    let flattened = patch
        .flatten_temporal_tensors(&frames, target)
        .expect("odd frame count should repeat the final frame");

    assert_eq!(flattened.shape(), [2, 8]);
    assert_eq!(
        flattened.data().to_vec::<f32>(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 12.0,]
    );
}

#[test]
fn patch_stage_reports_target_shape_when_source_size_mismatches() {
    let patch = RecipePatchStage::flatten(Layout::CHW, 2, 1, 1);
    let tensor = Tensor::new(TensorData::F32(vec![0.0; 24]), vec![3, 4, 2], Layout::CHW).unwrap();

    let err = patch
        .flatten_image_tensor(&tensor, ImageSize::new(2, 2).unwrap())
        .unwrap_err();

    assert_eq!(
        err,
        RecipePatchError::IncompatibleSourceShape {
            expected: vec![3, 2, 2],
            actual: vec![3, 4, 2],
        }
    );
}

#[test]
fn patch_stage_rejects_multi_sample_image_tensor() {
    let patch = RecipePatchStage::flatten(Layout::NCHW, 2, 1, 1);
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 24]),
        vec![2, 3, 2, 2],
        Layout::NCHW,
    )
    .unwrap();

    let err = patch
        .flatten_image_tensor(&tensor, ImageSize::new(2, 2).unwrap())
        .unwrap_err();

    assert_eq!(
        err,
        RecipePatchError::InvalidSourceShape {
            layout: Layout::NCHW,
            actual: vec![2, 3, 2, 2],
        }
    );
}

#[test]
fn patch_stage_rejects_invalid_target_for_merged_grid() {
    let patch = RecipePatchStage::flatten(Layout::NCHW, 14, 2, 2);
    let target = ImageSize::new(28, 42).unwrap();

    let err = patch.image_grid_thw(target).unwrap_err();

    assert_eq!(
        err,
        RecipePatchError::InvalidTarget {
            target_size: target,
            patch_size: 14,
            merge_size: 2,
        }
    );
}

#[test]
fn patch_stage_rejects_empty_temporal_grid() {
    let patch = RecipePatchStage::flatten(Layout::NCHW, 14, 2, 2);

    let err = patch.temporal_grid_size(0).unwrap_err();

    assert_eq!(err, RecipePatchError::EmptyFrameCount);
}

#[test]
fn smart_resize_patch_recipe_rejects_factor_mismatch() {
    let err = ProcessorRecipe::new(
        "test.bad_qwen",
        ProcessorRecipeInput::default(),
        vec![
            ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::smart(
                    ResizeLimits {
                        factor: 32,
                        min_pixels: 32 * 32,
                        max_pixels: 32 * 32 * 16,
                    },
                    ResizeMode::Default,
                    ResizeFilter::Bicubic,
                    ResizeParity::Compatibility,
                ),
            },
            ProcessorRecipeStage::PatchFlatten {
                patch: RecipePatchStage::flatten(Layout::NCHW, 14, 2, 2),
            },
        ],
        ProcessorRecipeOutput {
            layout: Layout::NC,
            leading_axis: TensorLeadingAxis::Batch,
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        RecipeError::InvalidPatchGeometry {
            resize_factor: 32,
            patch_size: 14,
            merge_size: 2,
            ..
        }
    ));
}
