use super::*;

#[test]
fn hf_preprocessor_config_parses_clip_size_and_crop() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "CLIPImageProcessor",
            "size": {"shortest_edge": 336},
            "crop_size": {"height": 320, "width": 320},
            "do_center_crop": false,
            "resample": 3
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Clip(config) = parsed else {
        panic!("expected CLIP config");
    };
    assert_eq!(config.size, ImageSize::new(336, 336).unwrap());
    assert_eq!(config.crop_size, ImageSize::new(320, 320).unwrap());
    assert!(!config.do_center_crop);
    assert_eq!(config.resample, ResizeFilter::Bicubic);
}

#[test]
fn hf_preprocessor_config_parses_blip_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "BlipImageProcessor",
            "size": {"height": 64, "width": 96},
            "do_resize": false,
            "do_center_crop": false,
            "do_convert_rgb": false,
            "do_rescale": false,
            "rescale_factor": 0.5,
            "do_normalize": true,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.4, 0.5, 0.6],
            "resample": "nearest"
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Blip(config) = parsed else {
        panic!("expected BLIP config");
    };
    assert_eq!(config.size, ImageSize::new(64, 96).unwrap());
    assert!(!config.do_resize);
    assert!(!config.do_center_crop);
    assert_eq!(config.pixel_format, None);
    assert!(!config.do_rescale);
    assert_eq!(config.rescale_factor, 0.5);
    assert!(config.do_normalize);
    assert_eq!(config.image_mean, vec![0.1, 0.2, 0.3]);
    assert_eq!(config.image_std, vec![0.4, 0.5, 0.6]);
    assert_eq!(config.resample, ResizeFilter::Nearest);
}

#[test]
fn hf_preprocessor_config_parses_sam_longest_edge() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "SamImageProcessor",
            "size": {"longest_edge": 2048},
            "resample": "nearest"
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Sam(config) = parsed else {
        panic!("expected SAM config");
    };
    assert_eq!(config.image_size, ImageSize::new(2048, 2048).unwrap());
    assert_eq!(config.resample, ResizeFilter::Nearest);
}

#[test]
fn hf_preprocessor_config_parses_detr_shortest_and_longest_edge() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "DetrImageProcessor",
            "size": {"shortest_edge": 640, "longest_edge": 1024}
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Detr(config) = parsed else {
        panic!("expected DETR config");
    };
    assert_eq!(
        config.resize_size,
        ShortestEdgeResizeConfig {
            shortest_edge: 640,
            longest_edge: Some(1024),
        }
    );
    assert_eq!(config.image_size, ImageSize::new(640, 640).unwrap());
}

#[test]
fn hf_preprocessor_config_parses_videomae_video_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VideoMAEImageProcessor",
            "size": {"shortest_edge": 4},
            "crop_size": {"height": 4, "width": 4},
            "do_resize": true,
            "do_center_crop": false,
            "resample": "nearest",
            "do_rescale": false,
            "do_normalize": true,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.4, 0.5, 0.6]
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::VideoMae(config) = parsed else {
        panic!("expected VideoMAE config");
    };
    assert_eq!(config.size, 4);
    assert_eq!(config.crop_size, ImageSize::new(4, 4).unwrap());
    assert!(config.do_resize);
    assert!(!config.do_center_crop);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(!config.do_rescale);
    assert!(config.do_normalize);
    assert_eq!(config.image_mean, vec![0.1, 0.2, 0.3]);
    assert_eq!(config.image_std, vec![0.4, 0.5, 0.6]);
}

#[test]
fn hf_preprocessor_config_parses_vivit_video_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VivitImageProcessor",
            "size": {"shortest_edge": 4},
            "crop_size": {"height": 4, "width": 4},
            "do_resize": true,
            "do_center_crop": false,
            "resample": "nearest",
            "do_rescale": true,
            "rescale_factor": 0.25,
            "offset": false,
            "do_normalize": true,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.4, 0.5, 0.6]
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Vivit(config) = parsed else {
        panic!("expected ViViT config");
    };
    assert_eq!(config.size, 4);
    assert_eq!(config.crop_size, ImageSize::new(4, 4).unwrap());
    assert!(config.do_resize);
    assert!(!config.do_center_crop);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(config.do_rescale);
    assert_eq!(config.rescale_factor, 0.25);
    assert!(!config.offset);
    assert!(config.do_normalize);
    assert_eq!(config.image_mean, vec![0.1, 0.2, 0.3]);
    assert_eq!(config.image_std, vec![0.4, 0.5, 0.6]);
}

#[test]
fn hf_preprocessor_config_parses_qwen_patch_limits() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "Qwen2VLImageProcessor",
            "min_pixels": 3136,
            "max_pixels": 1003520,
            "patch_size": 14,
            "merge_size": 3,
            "temporal_patch_size": 4,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::QwenVl(config) = parsed else {
        panic!("expected Qwen config");
    };
    assert_eq!(config.resize_limits.min_pixels, 3136);
    assert_eq!(config.resize_limits.max_pixels, 1003520);
    assert_eq!(config.resize_limits.factor, 42);
    assert_eq!(config.temporal_patch_size, 4);
    assert!(!config.do_normalize);
}

#[test]
fn hf_preprocessor_config_parses_llava_next_anyres_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "LlavaNextImageProcessor",
            "size": {"shortest_edge": 2},
            "crop_size": {"height": 2, "width": 2},
            "image_grid_pinpoints": [[2, 2], [2, 4]],
            "do_pad": false,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::LlavaNext(config) = parsed else {
        panic!("expected LLaVA-NeXT config");
    };
    assert_eq!(config.size, ImageSize::new(2, 2).unwrap());
    assert_eq!(config.crop_size, ImageSize::new(2, 2).unwrap());
    assert_eq!(
        config.image_grid_pinpoints,
        vec![
            ImageSize {
                height: 2,
                width: 2
            },
            ImageSize {
                height: 2,
                width: 4
            }
        ]
    );
    assert!(!config.do_pad);
    assert!(!config.do_normalize);
}

#[test]
fn hf_preprocessor_config_parses_pixtral_patch_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "PixtralImageProcessor",
            "size": {"longest_edge": 512},
            "patch_size": {"height": 8, "width": 16},
            "resample": 3,
            "do_rescale": false,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Pixtral(config) = parsed else {
        panic!("expected Pixtral config");
    };
    assert_eq!(config.max_size, ImageSize::new(512, 512).unwrap());
    assert_eq!(config.patch_size, ImageSize::new(8, 16).unwrap());
    assert_eq!(config.resample, ResizeFilter::Bicubic);
    assert!(!config.do_rescale);
    assert!(!config.do_normalize);
}

#[test]
fn hf_preprocessor_config_parses_idefics3_split_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "Idefics3ImageProcessor",
            "size": {"longest_edge": 10},
            "max_image_size": {"longest_edge": 5},
            "resample": "nearest",
            "do_image_splitting": false,
            "do_pad": false,
            "do_rescale": false,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Idefics3(config) = parsed else {
        panic!("expected Idefics3 config");
    };
    assert_eq!(config.longest_edge, 10);
    assert_eq!(config.max_image_size, 5);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(!config.do_image_splitting);
    assert!(!config.do_pad);
    assert!(!config.do_rescale);
    assert!(!config.do_normalize);
}

#[test]
fn hf_preprocessor_config_parses_mllama_tiling_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "MllamaImageProcessor",
            "size": {"height": 5, "width": 5},
            "max_image_tiles": 6,
            "resample": "nearest",
            "do_rescale": false,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Mllama(config) = parsed else {
        panic!("expected Mllama config");
    };
    assert_eq!(config.tile_size, 5);
    assert_eq!(config.max_image_tiles, 6);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(!config.do_rescale);
    assert!(!config.do_normalize);
    assert!(config.do_resize);
    assert!(config.do_pad);
}

#[test]
fn hf_preprocessor_config_parses_gemma3_pan_and_scan_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "Gemma3ImageProcessor",
            "size": {"height": 6, "width": 8},
            "do_resize": false,
            "do_pan_and_scan": true,
            "pan_and_scan_min_crop_size": 3,
            "pan_and_scan_max_num_crops": 5,
            "pan_and_scan_min_ratio_to_activate": 1.5,
            "resample": "nearest",
            "do_rescale": false,
            "do_normalize": false
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Gemma3(config) = parsed else {
        panic!("expected Gemma3 config");
    };
    assert_eq!(config.size, ImageSize::new(6, 8).unwrap());
    assert!(!config.do_resize);
    assert!(config.do_pan_and_scan);
    assert_eq!(config.pan_and_scan_min_crop_size, 3);
    assert_eq!(config.pan_and_scan_max_num_crops, 5);
    assert_eq!(config.pan_and_scan_min_ratio_to_activate, 1.5);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(!config.do_rescale);
    assert!(!config.do_normalize);
}

#[test]
fn hf_preprocessor_config_parses_donut_document_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "DonutImageProcessor",
            "size": {"height": 12, "width": 8},
            "do_resize": true,
            "do_thumbnail": false,
            "do_align_long_axis": true,
            "do_pad": false,
            "resample": "nearest",
            "do_rescale": false,
            "rescale_factor": 0.5,
            "do_normalize": true,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.4, 0.5, 0.6]
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::DocumentOcr(config) = parsed else {
        panic!("expected document/OCR config");
    };
    assert_eq!(config.image_size, ImageSize::new(12, 8).unwrap());
    assert!(config.do_resize);
    assert!(!config.do_thumbnail);
    assert!(config.do_align_long_axis);
    assert!(!config.do_pad);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert!(!config.do_rescale);
    assert_eq!(config.rescale_factor, 0.5);
    assert!(config.do_normalize);
    assert_eq!(config.image_mean, vec![0.1, 0.2, 0.3]);
    assert_eq!(config.image_std, vec![0.4, 0.5, 0.6]);
}

#[test]
fn hf_preprocessor_config_parses_vae_conversion_flags() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VaeImageProcessor",
            "vae_scale_factor": 4,
            "vae_latent_channels": 8,
            "do_convert_grayscale": true,
            "do_normalize": false,
            "do_binarize": true
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Vae(config) = parsed else {
        panic!("expected VAE config");
    };
    assert_eq!(config.vae_scale_factor, 4);
    assert_eq!(config.vae_latent_channels, 8);
    assert_eq!(config.pixel_format, Some(PixelFormat::Luma8));
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
}

#[test]
fn hf_preprocessor_config_routes_ldm3d_before_generic_vae() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VaeImageProcessorLDM3D",
            "vae_scale_factor": 4
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::VaeLdm3d(config) = parsed else {
        panic!("expected LDM3D VAE config");
    };
    assert_eq!(config.vae_scale_factor, 4);
}

#[test]
fn hf_preprocessor_config_parses_flux2_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "Flux2ImageProcessor",
            "do_resize": true,
            "vae_scale_factor": 16,
            "vae_latent_channels": 32,
            "resample": "nearest",
            "do_convert_rgb": true,
            "do_normalize": false,
            "do_binarize": true
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Flux2(config) = parsed else {
        panic!("expected Flux2 config");
    };
    assert!(config.do_resize);
    assert_eq!(config.vae_scale_factor, FLUX2_VAE_SCALE_FACTOR);
    assert_eq!(config.vae_latent_channels, FLUX2_VAE_LATENT_CHANNELS);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert_eq!(config.pixel_format, Some(PixelFormat::Rgb8));
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
}

#[test]
fn hf_preprocessor_config_parses_visual_cloze_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VisualClozeProcessor",
            "resolution": 512,
            "do_resize": true,
            "vae_scale_factor": 16,
            "vae_latent_channels": 16,
            "resample": "nearest",
            "do_convert_rgb": true,
            "do_normalize": false,
            "do_binarize": true
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::VisualCloze(config) = parsed else {
        panic!("expected VisualCloze config");
    };
    assert_eq!(config.resolution, 512);
    assert!(config.do_resize);
    assert_eq!(config.vae_scale_factor, VISUAL_CLOZE_VAE_SCALE_FACTOR);
    assert_eq!(config.vae_latent_channels, VISUAL_CLOZE_VAE_LATENT_CHANNELS);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert_eq!(config.pixel_format, Some(PixelFormat::Rgb8));
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
}

#[test]
fn hf_preprocessor_config_parses_hunyuan_video_15_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "HunyuanVideo15ImageProcessor",
            "vae_scale_factor": 8,
            "vae_latent_channels": 16,
            "do_resize": false,
            "do_convert_rgb": false,
            "do_normalize": false,
            "do_binarize": true,
            "resample": "nearest"
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::HunyuanVideo15(config) = parsed else {
        panic!("expected HunyuanVideo 1.5 config");
    };
    assert_eq!(config.vae_scale_factor, 8);
    assert_eq!(config.vae_latent_channels, 16);
    assert!(!config.do_resize);
    assert_eq!(config.pixel_format, None);
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
    assert_eq!(config.resample, ResizeFilter::Nearest);
}

#[test]
fn hf_preprocessor_config_parses_marigold_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "MarigoldImageProcessor",
            "vae_scale_factor": 4,
            "do_normalize": false,
            "do_range_check": false,
            "resample": "nearest"
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Marigold(config) = parsed else {
        panic!("expected Marigold config");
    };
    assert_eq!(config.vae_scale_factor, 4);
    assert!(!config.do_normalize);
    assert!(!config.do_range_check);
    assert_eq!(config.resample, ResizeFilter::Nearest);
}

#[test]
fn hf_preprocessor_config_parses_joy_image_edit_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "JoyImageEditImageProcessor",
            "do_resize": true,
            "vae_scale_factor": 16,
            "basesize": 1024,
            "resample": "nearest",
            "do_convert_grayscale": true,
            "do_normalize": false,
            "do_binarize": true
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::JoyImageEdit(config) = parsed else {
        panic!("expected JoyImage edit config");
    };
    assert!(config.do_resize);
    assert_eq!(config.vae_scale_factor, 16);
    assert_eq!(config.basesize, JOY_IMAGE_BASE_SIZE);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert_eq!(config.pixel_format, Some(PixelFormat::Luma8));
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
}

#[test]
fn hf_preprocessor_config_parses_wan_animate_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "WanAnimateImageProcessor",
            "do_resize": true,
            "vae_scale_factor": 16,
            "vae_latent_channels": 32,
            "resample": "nearest",
            "do_convert_rgb": true,
            "do_normalize": false,
            "do_binarize": true
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::WanAnimate(config) = parsed else {
        panic!("expected Wan Animate config");
    };
    assert!(config.do_resize);
    assert_eq!(config.vae_scale_factor, 16);
    assert_eq!(config.vae_latent_channels, 32);
    assert_eq!(config.resample, ResizeFilter::Nearest);
    assert_eq!(config.pixel_format, Some(PixelFormat::Rgb8));
    assert!(!config.do_normalize);
    assert!(config.do_binarize);
}

#[test]
fn hf_preprocessor_config_parses_ltx2_video_hdr_fields() {
    let parsed = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "LTX2VideoHDRProcessor",
            "do_resize": false,
            "vae_scale_factor": 16,
            "resample": "nearest",
            "hdr_transform": "logc3"
        }"#,
    )
    .unwrap();

    let ProcessorFamilyConfig::Ltx2VideoHdr(config) = parsed else {
        panic!("expected LTX2 HDR config");
    };
    assert!(!config.do_resize);
    assert_eq!(config.vae_scale_factor, 16);
    assert_eq!(config.resample, ResizeFilter::Nearest);
}

#[test]
fn hf_preprocessor_config_rejects_conflicting_vae_conversions() {
    let err = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{
            "image_processor_type": "VaeImageProcessor",
            "do_convert_rgb": true,
            "do_convert_grayscale": true
        }"#,
    )
    .unwrap_err();

    assert_eq!(err, ProcessorConfigError::ConflictingPixelFormatConversion);
}

#[test]
fn hf_preprocessor_config_rejects_missing_processor_type() {
    let err = ProcessorFamilyConfig::from_hf_preprocessor_json(r#"{"size": 224}"#).unwrap_err();

    assert_eq!(err, ProcessorConfigError::MissingProcessorType);
}

#[test]
fn hf_preprocessor_config_rejects_unsupported_processor_type() {
    let err = ProcessorFamilyConfig::from_hf_preprocessor_json(
        r#"{"image_processor_type": "UnknownImageProcessor"}"#,
    )
    .unwrap_err();

    assert_eq!(
        err,
        ProcessorConfigError::UnsupportedProcessorType("UnknownImageProcessor".to_string())
    );
}
