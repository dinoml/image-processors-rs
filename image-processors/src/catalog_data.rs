use super::{
    CatalogEntry, CompatibilityStatus, FixtureEvidence, ProcessorFamilyKind, UpstreamLibrary,
};

const TRANSFORMERS_AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const TRANSFORMERS_FIXTURE_COMMIT: &str = TRANSFORMERS_AUDIT_COMMIT;
const DIFFUSERS_AUDIT_COMMIT: &str = "208704a27a6f362b67cd1a04fa1db0b98036d26f";

pub(super) fn fixture_evidence(class_name: &str) -> Option<FixtureEvidence> {
    let fixture_files: &'static [&'static str] = match class_name {
        "CLIPImageProcessor" | "ViTImageProcessor" => &["transformers_clip_vit.json"],
        "VideoMAEImageProcessor" => &[
            "transformers_videomae.json",
            "transformers_videomae_batch.json",
        ],
        "VivitImageProcessor" => &["transformers_vivit.json", "transformers_vivit_batch.json"],
        "Qwen2VLImageProcessor" => &[
            "transformers_qwen_vl.json",
            "transformers_qwen_vl_multi_image.json",
            "transformers_qwen_vl_video.json",
        ],
        "LlavaNextImageProcessor" => &[
            "transformers_llava_next.json",
            "transformers_llava_next_nhwc.json",
        ],
        "PixtralImageProcessor" => &["transformers_pixtral.json"],
        "Idefics3ImageProcessor" => &["transformers_idefics3.json"],
        "Gemma3ImageProcessor" => &["transformers_gemma3.json"],
        "MllamaImageProcessor" => &["transformers_mllama.json"],
        "DetrImageProcessor" => &["transformers_detr.json"],
        "SamImageProcessor" => &["transformers_sam.json"],
        "DonutImageProcessor" => &["transformers_donut.json"],
        _ => return None,
    };

    Some(FixtureEvidence {
        source_commit: TRANSFORMERS_FIXTURE_COMMIT,
        fixture_files,
    })
}

macro_rules! transformers_recipe_id {
    ("conditional_detr") => {
        "transformers.conditional_detr_image_processor"
    };
    ("deformable_detr") => {
        "transformers.conditional_detr_image_processor"
    };
    ("grounding_dino") => {
        "transformers.grounding_dino_image_processor"
    };
    ("rf_detr") => {
        "transformers.rf_detr_image_processor"
    };
    ("pp_doclayout_v2") => {
        "transformers.pp_doclayout_image_processor"
    };
    ("pp_doclayout_v3") => {
        "transformers.pp_doclayout_image_processor"
    };
    ("pp_ocrv5_server_det") => {
        "transformers.pp_ocr_detection_image_processor"
    };
    ("mask2former") => {
        "transformers.maskformer_image_processor"
    };
    ("maskformer") => {
        "transformers.maskformer_image_processor"
    };
    ("prompt_depth_anything") => {
        "transformers.dpt_image_processor"
    };
    ("tipsv2_dpt") => {
        "transformers.tipsv2_image_processor"
    };
    ("efficientloftr") => {
        "transformers.keypoint_matching_image_processor"
    };
    ("lightglue") => {
        "transformers.keypoint_matching_image_processor"
    };
    ("superglue") => {
        "transformers.keypoint_matching_image_processor"
    };
    ($model_type:literal) => {
        concat!("transformers.", $model_type, "_image_processor")
    };
}

macro_rules! transformers_fixture_entry {
    ($class_name:literal, [$($class_alias:literal),* $(,)?], $model_type:tt, $family:ident) => {
        CatalogEntry {
            library: UpstreamLibrary::Transformers,
            class_name: $class_name,
            class_aliases: &[$($class_alias),*],
            model_type: Some($model_type),
            processor_ids: &[],
            recipe_id: Some(transformers_recipe_id!($model_type)),
            family: ProcessorFamilyKind::$family,
            audit_commit: TRANSFORMERS_AUDIT_COMMIT,
            compatibility: CompatibilityStatus::FixtureParity,
        }
    };
}

pub(super) const CATALOG_ENTRIES: &[CatalogEntry] = &[
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "CLIPImageProcessor",
        class_aliases: &["CLIPImageProcessorPil"],
        model_type: Some("clip"),
        processor_ids: &["openai/clip-vit-base-patch32"],
        recipe_id: Some("transformers.clip_image_processor"),
        family: ProcessorFamilyKind::Clip,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "ViTImageProcessor",
        class_aliases: &["ViTImageProcessorPil"],
        model_type: Some("vit"),
        processor_ids: &["google/vit-base-patch16-224-in21k"],
        recipe_id: Some("transformers.vit_image_processor"),
        family: ProcessorFamilyKind::Vit,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "VideoMAEImageProcessor",
        class_aliases: &["VideoMAEImageProcessorPil"],
        model_type: Some("videomae"),
        processor_ids: &["MCG-NJU/videomae-base"],
        recipe_id: Some("transformers.videomae_image_processor"),
        family: ProcessorFamilyKind::VideoMae,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "VivitImageProcessor",
        class_aliases: &[],
        model_type: Some("vivit"),
        processor_ids: &["google/vivit-b-16x2-kinetics400"],
        recipe_id: Some("transformers.vivit_image_processor"),
        family: ProcessorFamilyKind::Vivit,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "Qwen2VLImageProcessor",
        class_aliases: &["Qwen2VLImageProcessorPil"],
        model_type: Some("qwen2_vl"),
        processor_ids: &["Qwen/Qwen2-VL-7B-Instruct"],
        recipe_id: Some("transformers.qwen_vl_image_processor"),
        family: ProcessorFamilyKind::QwenVl,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "LlavaNextImageProcessor",
        class_aliases: &["LlavaNextImageProcessorPil"],
        model_type: Some("llava_next"),
        processor_ids: &[],
        recipe_id: Some("transformers.llava_next_image_processor"),
        family: ProcessorFamilyKind::LlavaNext,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "PixtralImageProcessor",
        class_aliases: &["PixtralImageProcessorPil"],
        model_type: Some("pixtral"),
        processor_ids: &[],
        recipe_id: Some("transformers.pixtral_image_processor"),
        family: ProcessorFamilyKind::Pixtral,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "Idefics3ImageProcessor",
        class_aliases: &["Idefics3ImageProcessorPil"],
        model_type: Some("idefics3"),
        processor_ids: &["HuggingFaceM4/Idefics3-8B-Llama3"],
        recipe_id: Some("transformers.idefics3_image_processor"),
        family: ProcessorFamilyKind::Idefics3,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "Gemma3ImageProcessor",
        class_aliases: &["Gemma3ImageProcessorPil"],
        model_type: Some("gemma3"),
        processor_ids: &["google/gemma-3-4b-it"],
        recipe_id: Some("transformers.gemma3_image_processor"),
        family: ProcessorFamilyKind::Gemma3,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "MllamaImageProcessor",
        class_aliases: &["MllamaImageProcessorPil"],
        model_type: Some("mllama"),
        processor_ids: &["meta-llama/Llama-3.2-11B-Vision-Instruct"],
        recipe_id: Some("transformers.mllama_image_processor"),
        family: ProcessorFamilyKind::Mllama,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "DetrImageProcessor",
        class_aliases: &["DetrImageProcessorPil"],
        model_type: Some("detr"),
        processor_ids: &["facebook/detr-resnet-50"],
        recipe_id: Some("transformers.detr_image_processor"),
        family: ProcessorFamilyKind::Detr,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "SamImageProcessor",
        class_aliases: &["SamImageProcessorPil"],
        model_type: Some("sam"),
        processor_ids: &["facebook/sam-vit-base"],
        recipe_id: Some("transformers.sam_image_processor"),
        family: ProcessorFamilyKind::Sam,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Transformers,
        class_name: "DonutImageProcessor",
        class_aliases: &["DonutImageProcessorPil"],
        model_type: Some("donut"),
        processor_ids: &["naver-clova-ix/donut-base"],
        recipe_id: Some("transformers.document_ocr_image_processor"),
        family: ProcessorFamilyKind::DocumentOcr,
        audit_commit: TRANSFORMERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "VaeImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: Some("diffusers.vae_image_processor"),
        family: ProcessorFamilyKind::Vae,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "VaeImageProcessorLDM3D",
        class_aliases: &[],
        model_type: None,
        processor_ids: &["Intel/ldm3d"],
        recipe_id: Some("diffusers.vae_image_processor"),
        family: ProcessorFamilyKind::VaeLdm3d,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "IPAdapterMaskProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::AttentionMask,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "PixArtImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::AspectRatioBucket,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "BlipImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::ImageConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "Flux2ImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::ImageConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "JoyImageEditImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::ImageConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "VisualClozeProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::ImageConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "WanAnimateImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::ImageConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "HunyuanVideo15ImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::VideoConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "LTX2VideoHDRProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: Some("diffusers.ltx2_video_hdr_postprocess"),
        family: ProcessorFamilyKind::VideoConditioning,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    CatalogEntry {
        library: UpstreamLibrary::Diffusers,
        class_name: "MarigoldImageProcessor",
        class_aliases: &[],
        model_type: None,
        processor_ids: &[],
        recipe_id: None,
        family: ProcessorFamilyKind::DensePrediction,
        audit_commit: DIFFUSERS_AUDIT_COMMIT,
        compatibility: CompatibilityStatus::FixtureParity,
    },
    transformers_fixture_entry!(
        "AriaImageProcessor",
        ["AriaImageProcessorPil"],
        "aria",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "BeitImageProcessor",
        ["BeitImageProcessorPil"],
        "beit",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "BitImageProcessor",
        ["BitImageProcessorPil"],
        "bit",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "BlipImageProcessor",
        ["BlipImageProcessorPil"],
        "blip",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "BridgeTowerImageProcessor",
        ["BridgeTowerImageProcessorPil"],
        "bridgetower",
        VisionLanguage
    ),
    transformers_fixture_entry!("CHMv2ImageProcessor", [], "chmv2", DepthGeometry),
    transformers_fixture_entry!(
        "ChameleonImageProcessor",
        ["ChameleonImageProcessorPil"],
        "chameleon",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "ChineseCLIPImageProcessor",
        ["ChineseCLIPImageProcessorPil"],
        "chinese_clip",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "Cohere2VisionImageProcessor",
        [],
        "cohere2_vision",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "ConditionalDetrImageProcessor",
        ["ConditionalDetrImageProcessorPil"],
        "conditional_detr",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "ConvNextImageProcessor",
        ["ConvNextImageProcessorPil"],
        "convnext",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "DINOv3ViTImageProcessor",
        [],
        "dinov3_vit",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "DPTImageProcessor",
        ["DPTImageProcessorPil"],
        "dpt",
        DepthGeometry
    ),
    transformers_fixture_entry!(
        "DeepseekOcr2ImageProcessor",
        ["DeepseekOcr2ImageProcessorPil"],
        "deepseek_ocr2",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "DeepseekVLHybridImageProcessor",
        ["DeepseekVLHybridImageProcessorPil"],
        "deepseek_vl_hybrid",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "DeepseekVLImageProcessor",
        ["DeepseekVLImageProcessorPil"],
        "deepseek_vl",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "DeformableDetrImageProcessor",
        ["DeformableDetrImageProcessorPil"],
        "deformable_detr",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "DeiTImageProcessor",
        ["DeiTImageProcessorPil"],
        "deit",
        EncoderClassifier
    ),
    transformers_fixture_entry!("DepthProImageProcessor", [], "depth_pro", DepthGeometry),
    transformers_fixture_entry!(
        "EfficientLoFTRImageProcessor",
        ["EfficientLoFTRImageProcessorPil"],
        "efficientloftr",
        KeypointMatchingPose
    ),
    transformers_fixture_entry!(
        "EfficientNetImageProcessor",
        ["EfficientNetImageProcessorPil"],
        "efficientnet",
        EncoderClassifier
    ),
    transformers_fixture_entry!("Emu3ImageProcessor", [], "emu3", VisionLanguage),
    transformers_fixture_entry!(
        "EomtImageProcessor",
        ["EomtImageProcessorPil"],
        "eomt",
        Segmentation
    ),
    transformers_fixture_entry!(
        "Ernie4_5_VLMoeImageProcessor",
        [
            "Ernie4_5_VLMoeImageProcessorPil",
            "Ernie4_5_VL_MoeImageProcessor",
            "Ernie4_5_VL_MoeImageProcessorPil",
        ],
        "ernie4_5_vl_moe",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "FlavaImageProcessor",
        ["FlavaImageProcessorPil"],
        "flava",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "FuyuImageProcessor",
        ["FuyuImageProcessorPil"],
        "fuyu",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "GLPNImageProcessor",
        ["GLPNImageProcessorPil"],
        "glpn",
        DepthGeometry
    ),
    transformers_fixture_entry!(
        "Gemma4ImageProcessor",
        ["Gemma4ImageProcessorPil"],
        "gemma4",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Gemma4UnifiedImageProcessor",
        [],
        "gemma4_unified",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Glm46VImageProcessor",
        ["Glm46VImageProcessorPil"],
        "glm46v",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Glm4vImageProcessor",
        ["Glm4vImageProcessorPil"],
        "glm4v",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "GlmImageImageProcessor",
        ["GlmImageImageProcessorPil"],
        "glm_image",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "GlmgaImageProcessor",
        ["GlmgaImageProcessorPil"],
        "glmga",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "GotOcr2ImageProcessor",
        ["GotOcr2ImageProcessorPil"],
        "got_ocr2",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "GroundingDinoImageProcessor",
        ["GroundingDinoImageProcessorPil"],
        "grounding_dino",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "HunYuanVLImageProcessor",
        ["HunYuanVLImageProcessorPil"],
        "hunyuan_vl",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Idefics2ImageProcessor",
        ["Idefics2ImageProcessorPil"],
        "idefics2",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "IdeficsImageProcessor",
        ["IdeficsImageProcessorPil"],
        "idefics",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "ImageGPTImageProcessor",
        ["ImageGPTImageProcessorPil"],
        "imagegpt",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "JanusImageProcessor",
        ["JanusImageProcessorPil"],
        "janus",
        VisionLanguage
    ),
    transformers_fixture_entry!("Kimi_K25ImageProcessor", [], "kimi_k25", VisionLanguage),
    transformers_fixture_entry!(
        "Kosmos2_5ImageProcessor",
        ["Kosmos2_5ImageProcessorPil"],
        "kosmos2_5",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "LayoutLMv2ImageProcessor",
        ["LayoutLMv2ImageProcessorPil"],
        "layoutlmv2",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "LayoutLMv3ImageProcessor",
        ["LayoutLMv3ImageProcessorPil"],
        "layoutlmv3",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "LevitImageProcessor",
        ["LevitImageProcessorPil"],
        "levit",
        EncoderClassifier
    ),
    transformers_fixture_entry!("Lfm2VlImageProcessor", [], "lfm2_vl", VisionLanguage),
    transformers_fixture_entry!(
        "LightGlueImageProcessor",
        ["LightGlueImageProcessorPil"],
        "lightglue",
        KeypointMatchingPose
    ),
    transformers_fixture_entry!("Llama4ImageProcessor", [], "llama4", VisionLanguage),
    transformers_fixture_entry!(
        "LlavaImageProcessor",
        ["LlavaImageProcessorPil"],
        "llava",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "LlavaOnevisionImageProcessor",
        ["LlavaOnevisionImageProcessorPil"],
        "llava_onevision",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Mask2FormerImageProcessor",
        ["Mask2FormerImageProcessorPil"],
        "mask2former",
        Segmentation
    ),
    transformers_fixture_entry!(
        "MaskFormerImageProcessor",
        ["MaskFormerImageProcessorPil"],
        "maskformer",
        Segmentation
    ),
    transformers_fixture_entry!(
        "MiniCPMV4_6ImageProcessor",
        ["MiniCPMV4_6ImageProcessorPil"],
        "minicpmv4_6",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "MiniMaxM3VLImageProcessor",
        [],
        "minimax_m3_vl",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "MobileNetV1ImageProcessor",
        ["MobileNetV1ImageProcessorPil"],
        "mobilenet_v1",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "MobileNetV2ImageProcessor",
        ["MobileNetV2ImageProcessorPil"],
        "mobilenet_v2",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "MobileViTImageProcessor",
        ["MobileViTImageProcessorPil"],
        "mobilevit",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "NougatImageProcessor",
        ["NougatImageProcessorPil"],
        "nougat",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "OneFormerImageProcessor",
        ["OneFormerImageProcessorPil"],
        "oneformer",
        Segmentation
    ),
    transformers_fixture_entry!(
        "Ovis2ImageProcessor",
        ["Ovis2ImageProcessorPil"],
        "ovis2",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "OwlViTImageProcessor",
        ["OwlViTImageProcessorPil"],
        "owlvit",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "Owlv2ImageProcessor",
        ["Owlv2ImageProcessorPil"],
        "owlv2",
        DetectionGrounding
    ),
    transformers_fixture_entry!("PI0ImageProcessor", [], "pi0", VisionLanguage),
    transformers_fixture_entry!(
        "PPChart2TableImageProcessor",
        ["PPChart2TableImageProcessorPil"],
        "pp_chart2table",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "PPDocLayoutV2ImageProcessor",
        [],
        "pp_doclayout_v2",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "PPDocLayoutV3ImageProcessor",
        [],
        "pp_doclayout_v3",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "PPFormulaNetImageProcessor",
        [],
        "pp_formulanet",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!("PPLCNetImageProcessor", [], "pp_lcnet", EncoderClassifier),
    transformers_fixture_entry!(
        "PPOCRV5ServerDetImageProcessor",
        [],
        "pp_ocrv5_server_det",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "PPOCRV5ServerRecImageProcessor",
        [],
        "pp_ocrv5_server_rec",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "PPOCRV6SmallRecImageProcessor",
        [],
        "pp_ocrv6_small_rec",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "PaddleOCRVLImageProcessor",
        ["PaddleOCRVLImageProcessorPil"],
        "paddleocr_vl",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "PerceiverImageProcessor",
        ["PerceiverImageProcessorPil"],
        "perceiver",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "PerceptionLMImageProcessor",
        [],
        "perception_lm",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Phi4MultimodalImageProcessor",
        [],
        "phi4_multimodal",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "Pix2StructImageProcessor",
        ["Pix2StructImageProcessorPil"],
        "pix2struct",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "PoolFormerImageProcessor",
        ["PoolFormerImageProcessorPil"],
        "poolformer",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "PromptDepthAnythingImageProcessor",
        ["PromptDepthAnythingImageProcessorPil"],
        "prompt_depth_anything",
        DepthGeometry
    ),
    transformers_fixture_entry!(
        "PvtImageProcessor",
        ["PvtImageProcessorPil"],
        "pvt",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "RTDetrImageProcessor",
        ["RTDetrImageProcessorPil"],
        "rt_detr",
        DetectionGrounding
    ),
    transformers_fixture_entry!("RfDetrImageProcessor", [], "rf_detr", DetectionGrounding),
    transformers_fixture_entry!(
        "SLANeXtImageProcessor",
        [],
        "slanext",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!("Sam2ImageProcessor", [], "sam2", Segmentation),
    transformers_fixture_entry!("Sam3ImageProcessor", [], "sam3", Segmentation),
    transformers_fixture_entry!("Sapiens2ImageProcessor", [], "sapiens2", Segmentation),
    transformers_fixture_entry!(
        "SegGptImageProcessor",
        ["SegGptImageProcessorPil"],
        "seggpt",
        Segmentation
    ),
    transformers_fixture_entry!(
        "SegformerImageProcessor",
        ["SegformerImageProcessorPil"],
        "segformer",
        Segmentation
    ),
    transformers_fixture_entry!(
        "Siglip2ImageProcessor",
        ["Siglip2ImageProcessorPil"],
        "siglip2",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "SiglipImageProcessor",
        ["SiglipImageProcessorPil"],
        "siglip",
        EncoderClassifier
    ),
    transformers_fixture_entry!(
        "SmolVLMImageProcessor",
        ["SmolVLMImageProcessorPil"],
        "smolvlm",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "SuperGlueImageProcessor",
        ["SuperGlueImageProcessorPil"],
        "superglue",
        KeypointMatchingPose
    ),
    transformers_fixture_entry!(
        "SuperPointImageProcessor",
        ["SuperPointImageProcessorPil"],
        "superpoint",
        KeypointMatchingPose
    ),
    transformers_fixture_entry!(
        "Swin2SRImageProcessor",
        ["Swin2SRImageProcessorPil"],
        "swin2sr",
        ImageRestoration
    ),
    transformers_fixture_entry!(
        "TextNetImageProcessor",
        ["TextNetImageProcessorPil"],
        "textnet",
        DocumentUnderstanding
    ),
    transformers_fixture_entry!(
        "TimmWrapperImageProcessor",
        [],
        "timm_wrapper",
        EncoderClassifier
    ),
    transformers_fixture_entry!("Tipsv2DptImageProcessor", [], "tipsv2_dpt", DepthGeometry),
    transformers_fixture_entry!("Tipsv2ImageProcessor", [], "tipsv2", DepthGeometry),
    transformers_fixture_entry!("TvpImageProcessor", ["TvpImageProcessorPil"], "tvp", Video),
    transformers_fixture_entry!("UVDocImageProcessor", [], "uvdoc", DocumentUnderstanding),
    transformers_fixture_entry!(
        "VideoLlama3ImageProcessor",
        ["VideoLlama3ImageProcessorPil"],
        "video_llama_3",
        Video
    ),
    transformers_fixture_entry!("VideoLlavaImageProcessor", [], "video_llava", Video),
    transformers_fixture_entry!(
        "ViltImageProcessor",
        ["ViltImageProcessorPil"],
        "vilt",
        VisionLanguage
    ),
    transformers_fixture_entry!(
        "VitMatteImageProcessor",
        ["VitMatteImageProcessorPil"],
        "vitmatte",
        Segmentation
    ),
    transformers_fixture_entry!(
        "VitPoseImageProcessor",
        ["VitPoseImageProcessorPil"],
        "vitpose",
        KeypointMatchingPose
    ),
    transformers_fixture_entry!(
        "YolosImageProcessor",
        ["YolosImageProcessorPil"],
        "yolos",
        DetectionGrounding
    ),
    transformers_fixture_entry!(
        "ZoeDepthImageProcessor",
        ["ZoeDepthImageProcessorPil"],
        "zoedepth",
        DepthGeometry
    ),
];
