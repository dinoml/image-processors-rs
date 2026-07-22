use super::*;

#[test]
fn clip_config_uses_explicit_device_agnostic_layout() {
    let config = ClipImageProcessorConfig {
        output_layout: ImageLayout::HeightWidthChannels,
        ..Default::default()
    };
    let generic = config.image_processor_config();

    assert_eq!(
        config.output_image_layout(),
        ImageLayout::HeightWidthChannels
    );
    assert_eq!(
        config.output_video_layout(),
        VideoLayout::FramesHeightWidthChannels
    );
    assert_eq!(generic.output_layout, Layout::NHWC);
}

#[test]
fn clip_config_uses_explicit_size_and_crop_controls() {
    let config = ClipImageProcessorConfig {
        size: ImageSize {
            height: 256,
            width: 256,
        },
        crop_size: ImageSize {
            height: 224,
            width: 224,
        },
        do_center_crop: true,
        ..Default::default()
    };
    let generic = config.image_processor_config();

    assert_eq!(generic.height, Some(224));
    assert_eq!(generic.width, Some(224));
    assert_eq!(generic.resize_mode, ResizeMode::Crop);
}

#[test]
fn vit_config_can_disable_center_crop() {
    let config = VitImageProcessorConfig {
        size: ImageSize {
            height: 384,
            width: 384,
        },
        crop_size: ImageSize {
            height: 224,
            width: 224,
        },
        do_center_crop: false,
        ..Default::default()
    };
    let generic = config.image_processor_config();

    assert_eq!(generic.height, Some(384));
    assert_eq!(generic.width, Some(384));
    assert_eq!(generic.resize_mode, ResizeMode::Default);
}

#[test]
fn fixed_family_recipes_lower_to_existing_generic_configs() {
    let clip = ClipImageProcessorConfig::default();
    let vit = VitImageProcessorConfig::default();

    assert_eq!(
        clip.processor_recipe()
            .unwrap()
            .to_image_processor_config()
            .unwrap(),
        clip.image_processor_config()
    );
    assert_eq!(
        vit.processor_recipe()
            .unwrap()
            .to_image_processor_config()
            .unwrap(),
        vit.image_processor_config()
    );
}

#[test]
fn detr_and_sam_recipes_describe_aspect_preserving_resize_targets() {
    let detr = DetrImageProcessorConfig::default();
    let sam = SamImageProcessorConfig::default();

    let detr_recipe = detr.processor_recipe().unwrap();
    let sam_recipe = sam.processor_recipe().unwrap();

    assert!(matches!(
        &detr_recipe.stages()[1],
        ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage {
                target: crate::recipe::RecipeResizeTarget::ShortestEdge {
                    shortest_edge: 800,
                    longest_edge: Some(1333),
                },
                ..
            }
        }
    ));
    assert_eq!(
        detr_recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "resize_shortest_edge",
        }
    );
    let [ProcessorRecipePostprocess::ObjectDetection(detr_postprocess)] = detr_recipe.postprocess()
    else {
        panic!(
            "expected one DETR object-detection postprocess descriptor, got {:?}",
            detr_recipe.postprocess()
        );
    };
    assert_eq!(detr_postprocess.logits_output, "logits");
    assert_eq!(detr_postprocess.boxes_output, "pred_boxes");
    assert_eq!(detr_postprocess.num_labels_with_background, None);
    assert_eq!(detr_postprocess.score_threshold, 0.5);
    assert_eq!(
        detr_postprocess.target_size_source,
        RecipeImageSizeSource::CallerProvided
    );

    assert!(matches!(
        &sam_recipe.stages()[1],
        ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage {
                target: crate::recipe::RecipeResizeTarget::LongestEdge { longest_edge: 1024 },
                ..
            }
        }
    ));
    assert_eq!(
        sam_recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "resize_longest_edge",
        }
    );
    let [ProcessorRecipePostprocess::BinaryMasks(mask_postprocess)] = sam_recipe.postprocess()
    else {
        panic!(
            "expected one binary mask postprocess descriptor, got {:?}",
            sam_recipe.postprocess()
        );
    };
    assert_eq!(mask_postprocess.mask_logits_output, "pred_masks");
    assert_eq!(mask_postprocess.mask_size, None);
    assert_eq!(mask_postprocess.masks_per_image, None);
    assert_eq!(
        mask_postprocess.original_size_source,
        RecipeImageSizeSource::OriginalSizes
    );
    assert_eq!(
        mask_postprocess.reshaped_input_size_source,
        RecipeImageSizeSource::ReshapedInputSizes
    );
    assert_eq!(mask_postprocess.mask_threshold, 0.0);
}

#[test]
fn vae_qwen_and_pixtral_recipes_lower_to_existing_generic_configs() {
    let vae = VaeImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        do_binarize: true,
        ..Default::default()
    };
    let qwen = QwenVlImageProcessorConfig::default();
    let pixtral = PixtralImageProcessorConfig::default();

    assert_eq!(
        vae.processor_recipe()
            .unwrap()
            .to_image_processor_config()
            .unwrap(),
        vae.image_processor_config()
    );
    assert_eq!(
        qwen.processor_recipe()
            .unwrap()
            .to_image_processor_config()
            .unwrap(),
        qwen.patch_image_processor_config()
    );
    assert_eq!(
        pixtral
            .processor_recipe()
            .unwrap()
            .to_image_processor_config()
            .unwrap(),
        pixtral.image_processor_config()
    );
}

#[test]
fn llava_next_recipe_records_patch_grid_geometry() {
    let llava_next = LlavaNextImageProcessorConfig::default();
    let recipe = llava_next.processor_recipe().unwrap();

    assert!(matches!(
        &recipe.stages()[0],
        ProcessorRecipeStage::PatchGrid { patch_grid }
            if patch_grid.candidate_sizes == llava_next.image_grid_pinpoints
                && patch_grid.base_size == llava_next.size
                && patch_grid.patch_size == llava_next.patch_size()
    ));
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "patch_grid",
        }
    );
}

#[test]
fn idefics3_recipe_records_split_geometry() {
    let idefics3 = Idefics3ImageProcessorConfig::default();
    let recipe = idefics3.processor_recipe().unwrap();

    assert!(matches!(
        &recipe.stages()[0],
        ProcessorRecipeStage::ImageSplit { split }
            if split.longest_edge == idefics3.longest_edge
                && split.max_image_size == idefics3.max_image_size
                && split.do_resize == idefics3.do_resize
    ));
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "image_split",
        }
    );
}

#[test]
fn mllama_recipe_records_tiled_canvas_geometry() {
    let mllama = MllamaImageProcessorConfig::default();
    let recipe = mllama.processor_recipe().unwrap();

    assert!(matches!(
        &recipe.stages()[0],
        ProcessorRecipeStage::TiledCanvas { tile }
            if tile.tile_size == mllama.tile_size
                && tile.max_image_tiles == mllama.max_image_tiles
    ));
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "tiled_canvas",
        }
    );
    assert_eq!(
        tiled_batch_layout_for_image_layout(mllama.output_layout),
        Layout::NIPCHW
    );
}

#[test]
fn document_ocr_recipe_records_document_geometry() {
    let document = DocumentOcrImageProcessorConfig::default();
    let recipe = document.processor_recipe().unwrap();

    assert!(matches!(
        &recipe.stages()[0],
        ProcessorRecipeStage::DocumentGeometry { document: geometry }
            if geometry.target_size == document.image_size
                && geometry.do_resize == document.do_resize
                && geometry.do_thumbnail == document.do_thumbnail
                && geometry.do_align_long_axis == document.do_align_long_axis
                && geometry.do_pad == document.do_pad
    ));
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "document_geometry",
        }
    );
}

#[test]
fn document_ocr_tensor_stage_recipe_lowers_when_geometry_is_disabled() {
    let document = DocumentOcrImageProcessorConfig {
        do_resize: false,
        do_thumbnail: false,
        do_align_long_axis: false,
        do_pad: false,
        ..Default::default()
    };
    let lowered = document
        .processor_recipe()
        .unwrap()
        .to_image_processor_config()
        .unwrap();

    assert!(!lowered.do_resize);
    assert!(lowered.do_rescale);
    assert!(lowered.do_normalize);
    assert_eq!(lowered, document.image_processor_config());
}

#[test]
fn videomae_recipe_lowers_to_tensor_stage_config() {
    let videomae = VideoMaeImageProcessorConfig::default();
    let lowered = videomae
        .processor_recipe()
        .unwrap()
        .to_image_processor_config()
        .unwrap();

    assert_eq!(lowered, videomae.image_processor_config());
    assert!(!lowered.do_resize);
    assert!(lowered.do_rescale);
    assert!(lowered.do_normalize);
    assert_eq!(
        videomae.output_video_layout(),
        VideoLayout::FramesChannelsHeightWidth
    );
}

#[test]
fn vivit_recipe_lowers_to_offset_tensor_stage_config() {
    let vivit = VivitImageProcessorConfig::default();
    let lowered = vivit
        .processor_recipe()
        .unwrap()
        .to_image_processor_config()
        .unwrap();

    assert_eq!(lowered, vivit.image_processor_config());
    assert!(!lowered.do_resize);
    assert!(lowered.do_rescale);
    assert!(lowered.do_normalize);
    assert_eq!(lowered.image_mean, vec![1.5, 1.5, 1.5]);
    assert_eq!(lowered.image_std, STANDARD_IMAGE_STD.to_vec());
}

#[test]
fn gemma3_recipe_lowers_to_generic_config() {
    let gemma3 = Gemma3ImageProcessorConfig::default();
    let lowered = gemma3
        .processor_recipe()
        .unwrap()
        .to_image_processor_config()
        .unwrap();

    assert_eq!(lowered, gemma3.image_processor_config());
}

#[test]
fn gemma3_recipe_records_enabled_aspect_ratio_crop_geometry() {
    let gemma3 = Gemma3ImageProcessorConfig {
        do_pan_and_scan: true,
        pan_and_scan_min_crop_size: 8,
        pan_and_scan_max_num_crops: 3,
        pan_and_scan_min_ratio_to_activate: 1.2,
        ..Default::default()
    };
    let recipe = gemma3.processor_recipe().unwrap();

    assert!(matches!(
        &recipe.stages()[1],
        ProcessorRecipeStage::AspectRatioCrops { aspect_ratio_crops }
            if aspect_ratio_crops.options == gemma3.pan_and_scan_options()
    ));
    assert_eq!(
        recipe.to_image_processor_config().unwrap_err(),
        RecipeError::UnsupportedGenericStage {
            stage: "aspect_ratio_crops",
        }
    );
}

#[test]
fn qwen_recipe_rejects_patch_geometry_mismatch() {
    let mut config = QwenVlImageProcessorConfig::default();
    config.resize_limits.factor += 1;

    let err = config.processor_recipe().unwrap_err();

    assert!(matches!(
        err,
        RecipeError::InvalidPatchGeometry {
            resize_factor,
            patch_size,
            merge_size,
            ..
        } if resize_factor == config.resize_limits.factor
            && patch_size == config.patch_size
            && merge_size == config.merge_size
    ));
}
