//! Compact audited presets for Transformers multimodal image processors.
//!
//! The upstream classes represented here share a small number of Rust-native
//! execution profiles. Presets retain the upstream class and source mapping
//! while exposing explicit fixed-square configurations. These compact presets
//! are distinct from upstream defaults: they accept square RGB images at or
//! above [`MultimodalPreset::minimum_input_edge`] and use the fixed output edge
//! reported by [`MultimodalPreset::output_edge`].

use std::collections::BTreeMap;

use half::bf16;
use thiserror::Error;

use crate::image::{BatchExecution, ImageProcessor, ImageProcessorConfig, ImageProcessorError};
use crate::media::{ImageDecodeBackend, ImageFrame, PixelFormat};
use crate::postprocess::{decode_token_sequences, RecipePostprocessError};
use crate::recipe::{
    ProcessorRecipe, ProcessorRecipeInput, ProcessorRecipeOutput, ProcessorRecipePostprocess,
    RecipeError, RecipePatchError, RecipePatchStage, RecipeTokenSequenceDecoder,
    RecipeTokenSequencePostprocess, RecipeTokenSequenceTask,
};
use crate::tensor::{Layout, Tensor, TensorData};
use crate::transforms::{ImageSize, ResizeFilter, ResizeMode, ResizeParity};

const AUDIT_COMMIT: &str = "6d960ca0a0eba0d2aebc920d8080a9353da468d3";
const COMPACT_EDGE: usize = 8;
const PATCH_SIZE: usize = 2;
const SLANEXT_BOS_TOKEN_ID: usize = 0;
const SLANEXT_EOS_TOKEN_ID: usize = 49;
const RESCALE_FACTOR: f32 = 1.0 / 255.0;
const HALF_MEAN: [f32; 3] = [0.5; 3];
const HALF_STD: [f32; 3] = [0.5; 3];
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PP_FORMULA_MEAN: [f32; 3] = [0.7931; 3];
const PP_FORMULA_STD: [f32; 3] = [0.1738; 3];
const ZERO_MEAN: [f32; 3] = [0.0; 3];
const ONE_MEAN: [f32; 3] = [1.0; 3];
const UNIT_STD: [f32; 3] = [1.0; 3];

/// Broad task category for an audited multimodal preset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalCategory {
    /// Vision-language image inputs.
    VisionLanguage,
    /// OCR, layout, table, or document inputs.
    DocumentUnderstanding,
    /// Video or temporal image inputs.
    Video,
}

/// Reusable execution profile shared by related upstream classes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalProfile {
    /// Fixed-size tensor preprocessing.
    Fixed,
    /// Fixed-size preprocessing with a validity mask or size metadata.
    FixedMask,
    /// One-or-more tiled image tensors.
    Tiled,
    /// Qwen-style spatial or temporal patch flattening.
    PatchGrid,
    /// Spatial patches kept as image-shaped frames.
    PatchFrames,
    /// Dynamically packed patch sequences.
    AdaptivePatches,
    /// Fuyu nested canvas output.
    Fuyu,
    /// Gemma 4 padded patch-token output.
    Gemma4,
    /// Gemma 4 unified padded patch-token output.
    Gemma4Unified,
    /// MiniCPM NaViT patch layout.
    MiniCpm,
    /// Phi-4 dynamic-HD image layout.
    Phi4,
    /// Batched frame tensor output.
    Video,
    /// UVDoc BGR input and retained original image.
    UvDoc,
}

/// Static mapping from an upstream class to its Rust execution profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimodalPreset {
    class_name: &'static str,
    model_type: &'static str,
    recipe_id: &'static str,
    source_path: &'static str,
    category: MultimodalCategory,
    profile: MultimodalProfile,
}

impl MultimodalPreset {
    /// Returns the canonical upstream processor class name.
    pub const fn class_name(self) -> &'static str {
        self.class_name
    }

    /// Returns the upstream model type.
    pub const fn model_type(self) -> &'static str {
        self.model_type
    }

    /// Returns the stable class-specific recipe identifier.
    pub const fn recipe_id(self) -> &'static str {
        self.recipe_id
    }

    /// Returns the source file audited at [`Self::audit_commit`].
    pub const fn source_path(self) -> &'static str {
        self.source_path
    }

    /// Returns the broad task category.
    pub const fn category(self) -> MultimodalCategory {
        self.category
    }

    /// Returns the reusable execution profile.
    pub const fn profile(self) -> MultimodalProfile {
        self.profile
    }

    /// Returns the exact upstream source revision used for the audit.
    pub const fn audit_commit(self) -> &'static str {
        AUDIT_COMMIT
    }

    /// Returns the fixed square output edge for this compact configuration.
    pub fn output_edge(self) -> usize {
        pixel_spec(self.class_name).edge
    }

    /// Returns the smallest square RGB input edge accepted by this preset.
    pub fn minimum_input_edge(self) -> usize {
        if self.class_name == "GlmImageImageProcessor" {
            4
        } else {
            COMPACT_EDGE
        }
    }

    /// Returns the input edge used by the audited full-value payload.
    pub fn audited_input_edge(self) -> usize {
        match self.class_name {
            "AriaImageProcessor" => 490,
            "Gemma4ImageProcessor" | "Gemma4UnifiedImageProcessor" => 16,
            "GlmImageImageProcessor" => 4,
            "MiniMaxM3VLImageProcessor" => 56,
            "Phi4MultimodalImageProcessor" => 12,
            _ => COMPACT_EDGE,
        }
    }

    /// Returns the spatial patch size when this profile emits patch tokens.
    pub const fn patch_size(self) -> Option<usize> {
        match self.profile {
            MultimodalProfile::PatchGrid
            | MultimodalProfile::PatchFrames
            | MultimodalProfile::AdaptivePatches
            | MultimodalProfile::Gemma4
            | MultimodalProfile::Gemma4Unified
            | MultimodalProfile::MiniCpm
            | MultimodalProfile::Phi4 => Some(PATCH_SIZE),
            MultimodalProfile::Fixed
            | MultimodalProfile::FixedMask
            | MultimodalProfile::Tiled
            | MultimodalProfile::Fuyu
            | MultimodalProfile::Video
            | MultimodalProfile::UvDoc => None,
        }
    }

    /// Builds and validates the typed execution recipe for this preset.
    ///
    /// # Errors
    ///
    /// Returns an error if the stable identifier or compact geometry encoded
    /// by the preset is internally inconsistent.
    pub fn processor_recipe(self) -> Result<MultimodalRecipe, MultimodalRecipeError> {
        MultimodalRecipe::try_from(self)
    }
}

/// Validated Rust-native execution recipe for a multimodal preset.
///
/// This descriptor records the complete compact geometry contract used by
/// [`MultimodalProcessor`] while retaining the stable class-specific recipe
/// identifier used by the compatibility catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimodalRecipe {
    preset: MultimodalPreset,
    output_edge: usize,
    minimum_input_edge: usize,
    audited_input_edge: usize,
    patch_size: Option<usize>,
}

impl MultimodalRecipe {
    /// Returns the stable class-specific recipe identifier.
    pub const fn id(self) -> &'static str {
        self.preset.recipe_id
    }

    /// Returns the preset represented by this recipe.
    pub const fn preset(self) -> MultimodalPreset {
        self.preset
    }

    /// Returns the canonical upstream processor class name.
    pub const fn class_name(self) -> &'static str {
        self.preset.class_name
    }

    /// Returns the upstream model type.
    pub const fn model_type(self) -> &'static str {
        self.preset.model_type
    }

    /// Returns the broad task category.
    pub const fn category(self) -> MultimodalCategory {
        self.preset.category
    }

    /// Returns the reusable execution profile.
    pub const fn profile(self) -> MultimodalProfile {
        self.preset.profile
    }

    /// Returns the fixed square output edge.
    pub const fn output_edge(self) -> usize {
        self.output_edge
    }

    /// Returns the smallest accepted square RGB input edge.
    pub const fn minimum_input_edge(self) -> usize {
        self.minimum_input_edge
    }

    /// Returns the input edge used by the audited parity payload.
    pub const fn audited_input_edge(self) -> usize {
        self.audited_input_edge
    }

    /// Returns the spatial patch size when the recipe emits patch tokens.
    pub const fn patch_size(self) -> Option<usize> {
        self.patch_size
    }

    /// Returns the shared patch-flatten stage used by patch-grid profiles.
    pub fn patch_stage(self) -> Option<RecipePatchStage> {
        (self.profile() == MultimodalProfile::PatchGrid).then(patch_grid_stage)
    }

    /// Builds the audited tokenizer-independent structured-output recipe, when
    /// this preset defines one.
    ///
    /// OCR presets expose CTC token-id decoding and SLANeXt exposes its
    /// BOS/EOS-aware table-token decoder. Token ids remain tied to the external
    /// model vocabulary; this recipe does not claim tokenizer or HTML-schema
    /// ownership.
    ///
    /// # Errors
    ///
    /// Returns an error if the internally defined postprocess-only recipe is
    /// invalid.
    pub fn structured_postprocess_recipe(self) -> Result<Option<ProcessorRecipe>, RecipeError> {
        let descriptor = match self.class_name() {
            "PPOCRV5ServerRecImageProcessor" | "PPOCRV6SmallRecImageProcessor" => {
                RecipeTokenSequencePostprocess::new(
                    RecipeTokenSequenceTask::Ocr,
                    "last_hidden_state",
                    RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id: 0 },
                )
            }
            "SLANeXtImageProcessor" => RecipeTokenSequencePostprocess::new(
                RecipeTokenSequenceTask::TableStructure,
                "last_hidden_state",
                RecipeTokenSequenceDecoder::Greedy {
                    begin_token_id: Some(SLANEXT_BOS_TOKEN_ID),
                    end_token_id: Some(SLANEXT_EOS_TOKEN_ID),
                },
            ),
            _ => return Ok(None),
        };
        ProcessorRecipe::new_postprocess_only(
            format!("{}.structured_output", self.id()),
            ProcessorRecipeInput::default(),
            ProcessorRecipeOutput::batch(Layout::CHW),
            vec![ProcessorRecipePostprocess::TokenSequence(descriptor)],
        )
        .map(Some)
    }

    /// Builds an executable processor from this validated recipe.
    ///
    /// # Errors
    ///
    /// Returns an error if the recipe's canonical class is no longer present
    /// in [`MULTIMODAL_PRESETS`].
    pub fn build(self) -> Result<MultimodalProcessor, MultimodalProcessorError> {
        MultimodalProcessor::for_class(self.class_name())
    }
}

impl TryFrom<MultimodalPreset> for MultimodalRecipe {
    type Error = MultimodalRecipeError;

    fn try_from(preset: MultimodalPreset) -> Result<Self, Self::Error> {
        let expected_id = format!("transformers.{}_image_processor", preset.model_type);
        if preset.recipe_id != expected_id {
            return Err(MultimodalRecipeError::InvalidIdentifier {
                class_name: preset.class_name,
                recipe_id: preset.recipe_id,
            });
        }

        let output_edge = preset.output_edge();
        let minimum_input_edge = preset.minimum_input_edge();
        let audited_input_edge = preset.audited_input_edge();
        if output_edge == 0 || minimum_input_edge == 0 || audited_input_edge < minimum_input_edge {
            return Err(MultimodalRecipeError::InvalidGeometry {
                class_name: preset.class_name,
                output_edge,
                minimum_input_edge,
                audited_input_edge,
            });
        }

        let patch_size = preset.patch_size();
        if let Some(patch_size) = patch_size {
            if patch_size == 0 || !output_edge.is_multiple_of(patch_size) {
                return Err(MultimodalRecipeError::InvalidPatchGeometry {
                    class_name: preset.class_name,
                    output_edge,
                    patch_size,
                });
            }
        }

        Ok(Self {
            preset,
            output_edge,
            minimum_input_edge,
            audited_input_edge,
            patch_size,
        })
    }
}

/// Validation errors for [`MultimodalRecipe`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MultimodalRecipeError {
    /// The stable recipe identifier does not match the preset model type.
    #[error("multimodal preset {class_name} has invalid recipe identifier {recipe_id}")]
    InvalidIdentifier {
        /// Canonical processor class.
        class_name: &'static str,
        /// Invalid recipe identifier.
        recipe_id: &'static str,
    },
    /// One or more fixed-square geometry values are invalid.
    #[error(
        "multimodal preset {class_name} has invalid geometry: output={output_edge}, minimum_input={minimum_input_edge}, audited_input={audited_input_edge}"
    )]
    InvalidGeometry {
        /// Canonical processor class.
        class_name: &'static str,
        /// Fixed output edge.
        output_edge: usize,
        /// Minimum accepted input edge.
        minimum_input_edge: usize,
        /// Audited parity input edge.
        audited_input_edge: usize,
    },
    /// Patch geometry does not evenly cover the fixed output.
    #[error(
        "multimodal preset {class_name} has invalid patch geometry: output={output_edge}, patch={patch_size}"
    )]
    InvalidPatchGeometry {
        /// Canonical processor class.
        class_name: &'static str,
        /// Fixed output edge.
        output_edge: usize,
        /// Spatial patch edge.
        patch_size: usize,
    },
}

macro_rules! preset {
    ($class:literal, $model:literal, $category:ident, $profile:ident, $source:literal) => {
        MultimodalPreset {
            class_name: $class,
            model_type: $model,
            recipe_id: concat!("transformers.", $model, "_image_processor"),
            source_path: concat!(
                "transformers/models/",
                $source,
                "/image_processing_",
                $source,
                ".py"
            ),
            category: MultimodalCategory::$category,
            profile: MultimodalProfile::$profile,
        }
    };
}

/// Every compact vision-language, document, and video Transformers preset.
pub const MULTIMODAL_PRESETS: &[MultimodalPreset] = &[
    preset!("AriaImageProcessor", "aria", VisionLanguage, Tiled, "aria"),
    preset!("BlipImageProcessor", "blip", VisionLanguage, Fixed, "blip"),
    preset!(
        "BridgeTowerImageProcessor",
        "bridgetower",
        VisionLanguage,
        FixedMask,
        "bridgetower"
    ),
    preset!(
        "ChameleonImageProcessor",
        "chameleon",
        VisionLanguage,
        Fixed,
        "chameleon"
    ),
    preset!(
        "Cohere2VisionImageProcessor",
        "cohere2_vision",
        VisionLanguage,
        Tiled,
        "cohere2_vision"
    ),
    preset!(
        "DeepseekOcr2ImageProcessor",
        "deepseek_ocr2",
        DocumentUnderstanding,
        Tiled,
        "deepseek_ocr2"
    ),
    preset!(
        "DeepseekVLHybridImageProcessor",
        "deepseek_vl_hybrid",
        VisionLanguage,
        Fixed,
        "deepseek_vl_hybrid"
    ),
    preset!(
        "DeepseekVLImageProcessor",
        "deepseek_vl",
        VisionLanguage,
        Fixed,
        "deepseek_vl"
    ),
    preset!(
        "Emu3ImageProcessor",
        "emu3",
        VisionLanguage,
        FixedMask,
        "emu3"
    ),
    preset!(
        "Ernie4_5_VLMoeImageProcessor",
        "ernie4_5_vl_moe",
        VisionLanguage,
        PatchGrid,
        "ernie4_5_vl_moe"
    ),
    preset!("FuyuImageProcessor", "fuyu", VisionLanguage, Fuyu, "fuyu"),
    preset!(
        "Gemma4ImageProcessor",
        "gemma4",
        VisionLanguage,
        Gemma4,
        "gemma4"
    ),
    preset!(
        "Gemma4UnifiedImageProcessor",
        "gemma4_unified",
        VisionLanguage,
        Gemma4Unified,
        "gemma4_unified"
    ),
    preset!(
        "Glm46VImageProcessor",
        "glm46v",
        VisionLanguage,
        PatchGrid,
        "glm46v"
    ),
    preset!(
        "Glm4vImageProcessor",
        "glm4v",
        VisionLanguage,
        PatchGrid,
        "glm4v"
    ),
    preset!(
        "GlmImageImageProcessor",
        "glm_image",
        VisionLanguage,
        PatchGrid,
        "glm_image"
    ),
    preset!(
        "GlmgaImageProcessor",
        "glmga",
        VisionLanguage,
        PatchGrid,
        "glmga"
    ),
    preset!(
        "GotOcr2ImageProcessor",
        "got_ocr2",
        DocumentUnderstanding,
        Tiled,
        "got_ocr2"
    ),
    preset!(
        "HunYuanVLImageProcessor",
        "hunyuan_vl",
        VisionLanguage,
        PatchGrid,
        "hunyuan_vl"
    ),
    preset!(
        "Idefics2ImageProcessor",
        "idefics2",
        VisionLanguage,
        FixedMask,
        "idefics2"
    ),
    preset!(
        "IdeficsImageProcessor",
        "idefics",
        VisionLanguage,
        Fixed,
        "idefics"
    ),
    preset!(
        "JanusImageProcessor",
        "janus",
        VisionLanguage,
        Fixed,
        "janus"
    ),
    preset!(
        "Kimi_K25ImageProcessor",
        "kimi_k25",
        VisionLanguage,
        PatchFrames,
        "kimi_k25"
    ),
    preset!(
        "Kosmos2_5ImageProcessor",
        "kosmos2_5",
        DocumentUnderstanding,
        AdaptivePatches,
        "kosmos2_5"
    ),
    preset!(
        "LayoutLMv2ImageProcessor",
        "layoutlmv2",
        DocumentUnderstanding,
        Fixed,
        "layoutlmv2"
    ),
    preset!(
        "LayoutLMv3ImageProcessor",
        "layoutlmv3",
        DocumentUnderstanding,
        Fixed,
        "layoutlmv3"
    ),
    preset!(
        "Lfm2VlImageProcessor",
        "lfm2_vl",
        VisionLanguage,
        AdaptivePatches,
        "lfm2_vl"
    ),
    preset!(
        "Llama4ImageProcessor",
        "llama4",
        VisionLanguage,
        Tiled,
        "llama4"
    ),
    preset!(
        "LlavaImageProcessor",
        "llava",
        VisionLanguage,
        Fixed,
        "llava"
    ),
    preset!(
        "LlavaOnevisionImageProcessor",
        "llava_onevision",
        VisionLanguage,
        Tiled,
        "llava_onevision"
    ),
    preset!(
        "MiniCPMV4_6ImageProcessor",
        "minicpmv4_6",
        VisionLanguage,
        MiniCpm,
        "minicpmv4_6"
    ),
    preset!(
        "MiniMaxM3VLImageProcessor",
        "minimax_m3_vl",
        VisionLanguage,
        PatchGrid,
        "minimax_m3_vl"
    ),
    preset!(
        "NougatImageProcessor",
        "nougat",
        DocumentUnderstanding,
        Fixed,
        "nougat"
    ),
    preset!(
        "Ovis2ImageProcessor",
        "ovis2",
        VisionLanguage,
        Tiled,
        "ovis2"
    ),
    preset!("PI0ImageProcessor", "pi0", VisionLanguage, FixedMask, "pi0"),
    preset!(
        "PPChart2TableImageProcessor",
        "pp_chart2table",
        DocumentUnderstanding,
        Fixed,
        "pp_chart2table"
    ),
    preset!(
        "PPFormulaNetImageProcessor",
        "pp_formulanet",
        DocumentUnderstanding,
        Fixed,
        "pp_formulanet"
    ),
    preset!(
        "PPOCRV5ServerRecImageProcessor",
        "pp_ocrv5_server_rec",
        DocumentUnderstanding,
        Fixed,
        "pp_ocrv5_server_rec"
    ),
    preset!(
        "PPOCRV6SmallRecImageProcessor",
        "pp_ocrv6_small_rec",
        DocumentUnderstanding,
        Fixed,
        "pp_ocrv6_small_rec"
    ),
    preset!(
        "PaddleOCRVLImageProcessor",
        "paddleocr_vl",
        DocumentUnderstanding,
        PatchFrames,
        "paddleocr_vl"
    ),
    preset!(
        "PerceptionLMImageProcessor",
        "perception_lm",
        VisionLanguage,
        Tiled,
        "perception_lm"
    ),
    preset!(
        "Phi4MultimodalImageProcessor",
        "phi4_multimodal",
        VisionLanguage,
        Phi4,
        "phi4_multimodal"
    ),
    preset!(
        "Pix2StructImageProcessor",
        "pix2struct",
        DocumentUnderstanding,
        AdaptivePatches,
        "pix2struct"
    ),
    preset!(
        "SLANeXtImageProcessor",
        "slanext",
        DocumentUnderstanding,
        FixedMask,
        "slanext"
    ),
    preset!(
        "Siglip2ImageProcessor",
        "siglip2",
        VisionLanguage,
        AdaptivePatches,
        "siglip2"
    ),
    preset!(
        "SmolVLMImageProcessor",
        "smolvlm",
        VisionLanguage,
        Tiled,
        "smolvlm"
    ),
    preset!(
        "TextNetImageProcessor",
        "textnet",
        DocumentUnderstanding,
        Fixed,
        "textnet"
    ),
    preset!("TvpImageProcessor", "tvp", Video, Video, "tvp"),
    preset!(
        "UVDocImageProcessor",
        "uvdoc",
        DocumentUnderstanding,
        UvDoc,
        "uvdoc"
    ),
    preset!(
        "VideoLlama3ImageProcessor",
        "video_llama_3",
        Video,
        PatchGrid,
        "video_llama_3"
    ),
    preset!(
        "VideoLlavaImageProcessor",
        "video_llava",
        Video,
        Video,
        "video_llava"
    ),
    preset!(
        "ViltImageProcessor",
        "vilt",
        VisionLanguage,
        FixedMask,
        "vilt"
    ),
];

/// Alternate upstream class names accepted by [`multimodal_preset`].
pub const MULTIMODAL_CLASS_ALIASES: &[(&str, &str)] = &[
    ("AriaImageProcessorPil", "AriaImageProcessor"),
    ("BlipImageProcessorPil", "BlipImageProcessor"),
    ("BridgeTowerImageProcessorPil", "BridgeTowerImageProcessor"),
    ("ChameleonImageProcessorPil", "ChameleonImageProcessor"),
    (
        "DeepseekOcr2ImageProcessorPil",
        "DeepseekOcr2ImageProcessor",
    ),
    (
        "DeepseekVLHybridImageProcessorPil",
        "DeepseekVLHybridImageProcessor",
    ),
    ("DeepseekVLImageProcessorPil", "DeepseekVLImageProcessor"),
    (
        "Ernie4_5_VLMoeImageProcessorPil",
        "Ernie4_5_VLMoeImageProcessor",
    ),
    (
        "Ernie4_5_VL_MoeImageProcessor",
        "Ernie4_5_VLMoeImageProcessor",
    ),
    (
        "Ernie4_5_VL_MoeImageProcessorPil",
        "Ernie4_5_VLMoeImageProcessor",
    ),
    ("FuyuImageProcessorPil", "FuyuImageProcessor"),
    ("Gemma4ImageProcessorPil", "Gemma4ImageProcessor"),
    ("Glm46VImageProcessorPil", "Glm46VImageProcessor"),
    ("Glm4vImageProcessorPil", "Glm4vImageProcessor"),
    ("GlmImageImageProcessorPil", "GlmImageImageProcessor"),
    ("GlmgaImageProcessorPil", "GlmgaImageProcessor"),
    ("GotOcr2ImageProcessorPil", "GotOcr2ImageProcessor"),
    ("HunYuanVLImageProcessorPil", "HunYuanVLImageProcessor"),
    ("Idefics2ImageProcessorPil", "Idefics2ImageProcessor"),
    ("IdeficsImageProcessorPil", "IdeficsImageProcessor"),
    ("JanusImageProcessorPil", "JanusImageProcessor"),
    ("Kosmos2_5ImageProcessorPil", "Kosmos2_5ImageProcessor"),
    ("LayoutLMv2ImageProcessorPil", "LayoutLMv2ImageProcessor"),
    ("LayoutLMv3ImageProcessorPil", "LayoutLMv3ImageProcessor"),
    ("LlavaImageProcessorPil", "LlavaImageProcessor"),
    (
        "LlavaOnevisionImageProcessorPil",
        "LlavaOnevisionImageProcessor",
    ),
    ("MiniCPMV4_6ImageProcessorPil", "MiniCPMV4_6ImageProcessor"),
    ("NougatImageProcessorPil", "NougatImageProcessor"),
    ("Ovis2ImageProcessorPil", "Ovis2ImageProcessor"),
    (
        "PPChart2TableImageProcessorPil",
        "PPChart2TableImageProcessor",
    ),
    ("PaddleOCRVLImageProcessorPil", "PaddleOCRVLImageProcessor"),
    ("Pix2StructImageProcessorPil", "Pix2StructImageProcessor"),
    ("Siglip2ImageProcessorPil", "Siglip2ImageProcessor"),
    ("SmolVLMImageProcessorPil", "SmolVLMImageProcessor"),
    ("TextNetImageProcessorPil", "TextNetImageProcessor"),
    ("TvpImageProcessorPil", "TvpImageProcessor"),
    ("VideoLlama3ImageProcessorPil", "VideoLlama3ImageProcessor"),
    ("ViltImageProcessorPil", "ViltImageProcessor"),
];

/// Returns the audited preset for `class_name`.
pub fn multimodal_preset(class_name: &str) -> Option<&'static MultimodalPreset> {
    let class_name = MULTIMODAL_CLASS_ALIASES
        .iter()
        .find_map(|&(alias, canonical)| (alias == class_name).then_some(canonical))
        .unwrap_or(class_name);
    MULTIMODAL_PRESETS
        .iter()
        .find(|preset| preset.class_name == class_name)
}

/// Owned values for arbitrary-rank multimodal arrays.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum MultimodalArrayData {
    /// 32-bit floating-point values.
    F32(Vec<f32>),
    /// Unsigned 8-bit values.
    U8(Vec<u8>),
    /// Signed 32-bit integer values.
    I32(Vec<i32>),
    /// Signed 64-bit integer values.
    I64(Vec<i64>),
    /// Boolean values stored as canonical zero-or-one bytes.
    Bool(Vec<u8>),
}

impl MultimodalArrayData {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U8(values) | Self::Bool(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
        }
    }

    fn into_f32(self) -> Result<Vec<f32>, MultimodalProcessorError> {
        match self {
            Self::F32(values) => Ok(values),
            _ => Err(MultimodalProcessorError::UnexpectedTensorDtype),
        }
    }
}

/// Validated arbitrary-rank array returned by a multimodal processor.
#[derive(Clone, Debug, PartialEq)]
pub struct MultimodalArray {
    shape: Vec<usize>,
    data: MultimodalArrayData,
}

impl MultimodalArray {
    /// Creates an array after validating its element count.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, the shape overflows, or the
    /// data length does not equal the shape element count.
    pub fn new(
        shape: impl Into<Vec<usize>>,
        data: MultimodalArrayData,
    ) -> Result<Self, MultimodalProcessorError> {
        let shape = shape.into();
        if shape.contains(&0) {
            return Err(MultimodalProcessorError::ZeroArrayDimension);
        }
        let expected = shape.iter().try_fold(1usize, |count, &dimension| {
            count
                .checked_mul(dimension)
                .ok_or(MultimodalProcessorError::ShapeOverflow)
        })?;
        if expected != data.len() {
            return Err(MultimodalProcessorError::InvalidArrayLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Returns the array shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the array data.
    pub fn data(&self) -> &MultimodalArrayData {
        &self.data
    }
}

/// A named output value, including arrays and structured document metadata.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum MultimodalValue {
    /// Numeric or boolean array.
    Array(MultimodalArray),
    /// Ordered nested values.
    Sequence(Vec<MultimodalValue>),
    /// Named nested values.
    Mapping(BTreeMap<String, MultimodalValue>),
    /// Signed integer scalar.
    Integer(i64),
    /// Floating-point scalar.
    Float(f64),
    /// Text scalar.
    Text(String),
}

/// Named outputs returned by [`MultimodalProcessor`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultimodalOutput {
    fields: BTreeMap<String, MultimodalValue>,
}

impl MultimodalOutput {
    /// Returns all output fields in stable lexical order.
    pub fn fields(&self) -> &BTreeMap<String, MultimodalValue> {
        &self.fields
    }

    /// Returns one named output field.
    pub fn get(&self, name: &str) -> Option<&MultimodalValue> {
        self.fields.get(name)
    }

    fn insert_array(&mut self, name: &str, array: MultimodalArray) {
        self.fields
            .insert(name.to_owned(), MultimodalValue::Array(array));
    }

    fn insert_value(&mut self, name: &str, value: MultimodalValue) {
        self.fields.insert(name.to_owned(), value);
    }
}

/// Errors returned by audited multimodal preprocessing.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MultimodalProcessorError {
    /// No audited preset exists for a class.
    #[error("unknown multimodal processor class {0}")]
    UnknownClass(String),
    /// The input is empty.
    #[error("multimodal processor input is empty")]
    EmptyInput,
    /// A profile that accepts one image received additional frames.
    #[error("multimodal processor {class_name} accepts exactly one image, got {actual} frames")]
    UnsupportedFrameCount {
        /// Canonical processor class.
        class_name: &'static str,
        /// Actual number of supplied frames.
        actual: usize,
    },
    /// Compact presets require RGB input.
    #[error("compact multimodal presets require RGB8 input, got {0:?}")]
    UnsupportedPixelFormat(PixelFormat),
    /// Compact presets require a square input.
    #[error("compact multimodal presets require a square input, got {width}x{height}")]
    NonSquareInput {
        /// Input width.
        width: usize,
        /// Input height.
        height: usize,
    },
    /// A compact preset received an input below its supported minimum edge.
    #[error(
        "compact multimodal preset requires an input edge of at least {minimum}, got {actual}"
    )]
    InputTooSmall {
        /// Minimum supported edge.
        minimum: usize,
        /// Actual input edge.
        actual: usize,
    },
    /// An array contained a zero dimension.
    #[error("multimodal array dimensions must be non-zero")]
    ZeroArrayDimension,
    /// An array shape overflowed `usize`.
    #[error("multimodal array element count overflowed usize")]
    ShapeOverflow,
    /// Array data and shape lengths differ.
    #[error("multimodal array expected {expected} values, got {actual}")]
    InvalidArrayLength {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// Document rectification received a zero image or output dimension.
    #[error("document rectification {field} dimensions must be non-zero, got {width}x{height}")]
    InvalidRectificationDimensions {
        /// Dimension group (`image` or `output`).
        field: &'static str,
        /// Invalid width.
        width: usize,
        /// Invalid height.
        height: usize,
    },
    /// Document rectification received a non-finite image value.
    #[error("document rectification image value at index {index} is not finite: {value}")]
    NonFiniteRectificationImageValue {
        /// Invalid flattened image index.
        index: usize,
        /// Non-finite value.
        value: f32,
    },
    /// Document rectification received a non-finite grid coordinate.
    #[error(
        "document rectification grid {axis} coordinate at index {index} is not finite: {value}"
    )]
    NonFiniteRectificationGridCoordinate {
        /// Invalid flattened grid index.
        index: usize,
        /// Coordinate axis (`x` or `y`).
        axis: &'static str,
        /// Non-finite value.
        value: f32,
    },
    /// Document rectification received a non-finite output scale.
    #[error("document rectification scale must be finite, got {0}")]
    NonFiniteRectificationScale(f32),
    /// An internal fixed processor returned an unexpected dtype.
    #[error("processor returned an unexpected tensor dtype")]
    UnexpectedTensorDtype,
    /// A logits shape was invalid.
    #[error("invalid logits shape: expected {expected} values, got {actual}")]
    InvalidLogitsLength {
        /// Expected value count.
        expected: usize,
        /// Actual value count.
        actual: usize,
    },
    /// A prediction referenced a vocabulary entry that does not exist.
    #[error("prediction index {index} is outside vocabulary length {vocabulary_len}")]
    VocabularyIndex {
        /// Invalid index.
        index: usize,
        /// Available vocabulary length.
        vocabulary_len: usize,
    },
    /// The generic image processor failed.
    #[error(transparent)]
    Image(#[from] ImageProcessorError),
    /// Shared patch flattening failed.
    #[error(transparent)]
    Patch(#[from] RecipePatchError),
    /// Shared structured token-sequence decoding failed.
    #[error(transparent)]
    TokenSequence(#[from] RecipePostprocessError),
}

/// Resize backend selected by a canonical or legacy processor class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalResizeBackend {
    /// Antialiased tensor resize used by canonical Transformers processors.
    Torchvision,
    /// Pillow-compatible resize used by legacy `*Pil` aliases.
    Pillow,
}

impl MultimodalResizeBackend {
    const fn parity(self) -> ResizeParity {
        match self {
            Self::Torchvision => ResizeParity::Torchvision,
            Self::Pillow => ResizeParity::Compatibility,
        }
    }
}

/// Executes one audited multimodal class through a shared Rust profile.
#[derive(Clone, Copy, Debug)]
pub struct MultimodalProcessor {
    preset: &'static MultimodalPreset,
    requested_class_name: &'static str,
    resize_backend: MultimodalResizeBackend,
}

impl MultimodalProcessor {
    /// Creates a processor for an audited upstream class.
    ///
    /// # Errors
    ///
    /// Returns an error when `class_name` is not in [`MULTIMODAL_PRESETS`].
    pub fn for_class(class_name: &str) -> Result<Self, MultimodalProcessorError> {
        let preset = multimodal_preset(class_name)
            .ok_or_else(|| MultimodalProcessorError::UnknownClass(class_name.to_owned()))?;
        let alias = MULTIMODAL_CLASS_ALIASES
            .iter()
            .find(|&&(alias, _)| alias == class_name)
            .map(|&(alias, _)| alias);
        let requested_class_name = alias.unwrap_or(preset.class_name);
        let resize_backend = if requested_class_name.ends_with("Pil") {
            MultimodalResizeBackend::Pillow
        } else {
            MultimodalResizeBackend::Torchvision
        };
        Ok(Self {
            preset,
            requested_class_name,
            resize_backend,
        })
    }

    /// Returns the selected static preset.
    pub const fn preset(self) -> &'static MultimodalPreset {
        self.preset
    }

    /// Returns the canonical class or exact alias used to construct this processor.
    pub const fn requested_class_name(self) -> &'static str {
        self.requested_class_name
    }

    /// Returns the resize backend selected by the requested class.
    pub const fn resize_backend(self) -> MultimodalResizeBackend {
        self.resize_backend
    }

    /// Preprocesses one decoded image.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported pixel storage or failed transforms.
    pub fn preprocess_image(
        self,
        image: &ImageFrame,
    ) -> Result<MultimodalOutput, MultimodalProcessorError> {
        self.preprocess_frames(std::slice::from_ref(image))
    }

    /// Preprocesses one image batch or temporal frame sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when `frames` is empty, frames are not RGB8, or a
    /// configured resize or tensor conversion fails.
    pub fn preprocess_frames(
        self,
        frames: &[ImageFrame],
    ) -> Result<MultimodalOutput, MultimodalProcessorError> {
        if frames.is_empty() {
            return Err(MultimodalProcessorError::EmptyInput);
        }
        if frames.len() > 1
            && !matches!(
                self.preset.profile,
                MultimodalProfile::PatchGrid | MultimodalProfile::Video
            )
        {
            return Err(MultimodalProcessorError::UnsupportedFrameCount {
                class_name: self.preset.class_name,
                actual: frames.len(),
            });
        }
        for frame in frames {
            ensure_input(self.preset, frame)?;
        }

        match self.preset.profile {
            MultimodalProfile::Fixed | MultimodalProfile::FixedMask => {
                preprocess_fixed(self, &frames[0])
            }
            MultimodalProfile::Tiled => preprocess_tiled(self, &frames[0]),
            MultimodalProfile::PatchGrid => preprocess_patch_grid(self, frames),
            MultimodalProfile::PatchFrames => preprocess_patch_frames(self, &frames[0]),
            MultimodalProfile::AdaptivePatches => preprocess_adaptive(self, &frames[0]),
            MultimodalProfile::Fuyu => preprocess_fuyu(self, &frames[0]),
            MultimodalProfile::Gemma4 | MultimodalProfile::Gemma4Unified => {
                preprocess_gemma4(self, &frames[0])
            }
            MultimodalProfile::MiniCpm => preprocess_minicpm(self, &frames[0]),
            MultimodalProfile::Phi4 => preprocess_phi4(self, &frames[0]),
            MultimodalProfile::Video => preprocess_video(self, frames),
            MultimodalProfile::UvDoc => preprocess_uvdoc(self, &frames[0]),
        }
    }
}

#[derive(Clone, Copy)]
struct PixelSpec {
    edge: usize,
    filter: ResizeFilter,
    rescale_factor: f32,
    do_rescale: bool,
    do_normalize: bool,
    mean: &'static [f32; 3],
    std: &'static [f32; 3],
    reverse_channels: bool,
}

fn pixel_spec(class_name: &str) -> PixelSpec {
    let edge = match class_name {
        "AriaImageProcessor" => 490,
        "Gemma4ImageProcessor" | "Gemma4UnifiedImageProcessor" => 16,
        "GlmImageImageProcessor" => 4,
        "MiniMaxM3VLImageProcessor" => 56,
        "Phi4MultimodalImageProcessor" => 12,
        _ => COMPACT_EDGE,
    };
    let filter = match class_name {
        "ChameleonImageProcessor" | "SmolVLMImageProcessor" => ResizeFilter::Lanczos,
        "FuyuImageProcessor"
        | "Idefics2ImageProcessor"
        | "LayoutLMv2ImageProcessor"
        | "LayoutLMv3ImageProcessor"
        | "Lfm2VlImageProcessor"
        | "NougatImageProcessor"
        | "PPFormulaNetImageProcessor"
        | "PPOCRV5ServerRecImageProcessor"
        | "PPOCRV6SmallRecImageProcessor"
        | "Phi4MultimodalImageProcessor"
        | "SLANeXtImageProcessor"
        | "Siglip2ImageProcessor"
        | "TextNetImageProcessor"
        | "TvpImageProcessor"
        | "UVDocImageProcessor" => ResizeFilter::Bilinear,
        _ => ResizeFilter::Bicubic,
    };
    let (mean, std) = match class_name {
        "BlipImageProcessor"
        | "BridgeTowerImageProcessor"
        | "Cohere2VisionImageProcessor"
        | "DeepseekVLHybridImageProcessor"
        | "DeepseekVLImageProcessor"
        | "Emu3ImageProcessor"
        | "Ernie4_5_VLMoeImageProcessor"
        | "Glm46VImageProcessor"
        | "Glm4vImageProcessor"
        | "GlmImageImageProcessor"
        | "GlmgaImageProcessor"
        | "GotOcr2ImageProcessor"
        | "HunYuanVLImageProcessor"
        | "IdeficsImageProcessor"
        | "JanusImageProcessor"
        | "LlavaImageProcessor"
        | "LlavaOnevisionImageProcessor"
        | "MiniMaxM3VLImageProcessor"
        | "Ovis2ImageProcessor"
        | "PPChart2TableImageProcessor"
        | "PaddleOCRVLImageProcessor"
        | "VideoLlavaImageProcessor" => (&CLIP_MEAN, &CLIP_STD),
        "NougatImageProcessor" | "SLANeXtImageProcessor" | "TextNetImageProcessor" => {
            (&IMAGENET_MEAN, &IMAGENET_STD)
        }
        "PPFormulaNetImageProcessor" => (&PP_FORMULA_MEAN, &PP_FORMULA_STD),
        "ChameleonImageProcessor" => (&ONE_MEAN, &UNIT_STD),
        "Gemma4ImageProcessor" | "Gemma4UnifiedImageProcessor" => (&ZERO_MEAN, &UNIT_STD),
        _ => (&HALF_MEAN, &HALF_STD),
    };
    PixelSpec {
        edge,
        filter,
        rescale_factor: if class_name == "ChameleonImageProcessor" {
            0.0078
        } else {
            RESCALE_FACTOR
        },
        do_rescale: class_name != "LayoutLMv2ImageProcessor",
        do_normalize: !matches!(
            class_name,
            "LayoutLMv2ImageProcessor"
                | "Gemma4ImageProcessor"
                | "Gemma4UnifiedImageProcessor"
                | "UVDocImageProcessor"
        ),
        mean,
        std,
        reverse_channels: matches!(
            class_name,
            "LayoutLMv2ImageProcessor"
                | "PPOCRV6SmallRecImageProcessor"
                | "TvpImageProcessor"
                | "UVDocImageProcessor"
        ),
    }
}

fn ensure_input(
    preset: &MultimodalPreset,
    frame: &ImageFrame,
) -> Result<(), MultimodalProcessorError> {
    if frame.pixel_format() != PixelFormat::Rgb8 {
        return Err(MultimodalProcessorError::UnsupportedPixelFormat(
            frame.pixel_format(),
        ));
    }
    if frame.width() != frame.height() {
        return Err(MultimodalProcessorError::NonSquareInput {
            width: frame.width(),
            height: frame.height(),
        });
    }
    if frame.width() < preset.minimum_input_edge() {
        return Err(MultimodalProcessorError::InputTooSmall {
            minimum: preset.minimum_input_edge(),
            actual: frame.width(),
        });
    }
    Ok(())
}

fn process_pixels(
    class_name: &str,
    frame: &ImageFrame,
    resize_backend: MultimodalResizeBackend,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    if class_name == "Llama4ImageProcessor" {
        return process_llama4_pixels(frame, resize_backend);
    }
    if matches!(
        class_name,
        "PPOCRV5ServerRecImageProcessor" | "PPOCRV6SmallRecImageProcessor"
    ) {
        return process_ppocr_pixels(class_name, frame);
    }
    if class_name == "SLANeXtImageProcessor" {
        return process_slanext_pixels(frame);
    }
    let spec = pixel_spec(class_name);
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(spec.edge),
        width: Some(spec.edge),
        resize_mode: ResizeMode::Default,
        resample: spec.filter,
        resize_parity: resize_backend.parity(),
        decode_backend: ImageDecodeBackend::ImageCrate,
        batch_execution: BatchExecution::Serial,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: spec.do_rescale,
        rescale_factor: spec.rescale_factor,
        do_normalize: spec.do_normalize,
        image_mean: spec.mean.to_vec(),
        image_std: spec.std.to_vec(),
        do_binarize: false,
        output_layout: Layout::NCHW,
    })?;
    let tensor = processor.preprocess_image(frame)?;
    let mut array = match tensor.data() {
        TensorData::F32(values) => MultimodalArray::new(
            tensor.shape().to_vec(),
            MultimodalArrayData::F32(values.clone()),
        )?,
        TensorData::U8(values) => MultimodalArray::new(
            tensor.shape().to_vec(),
            MultimodalArrayData::U8(values.clone()),
        )?,
        TensorData::F16(_)
        | TensorData::BF16(_)
        | TensorData::I8(_)
        | TensorData::QuantizedU8 { .. }
        | TensorData::QuantizedI8 { .. }
        | TensorData::PackedU4 { .. }
        | TensorData::PackedI4 { .. }
        | TensorData::I32(_)
        | TensorData::I64(_)
        | TensorData::Bool(_) => return Err(MultimodalProcessorError::UnexpectedTensorDtype),
    };
    if class_name == "LayoutLMv2ImageProcessor" {
        let MultimodalArrayData::F32(values) = array.data else {
            return Err(MultimodalProcessorError::UnexpectedTensorDtype);
        };
        array.data = MultimodalArrayData::U8(values.into_iter().map(|value| value as u8).collect());
    }
    if spec.reverse_channels {
        reverse_nchw_channels(&mut array)?;
    }
    Ok(array)
}

fn process_ppocr_pixels(
    class_name: &str,
    frame: &ImageFrame,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    let spec = pixel_spec(class_name);
    let values = frame
        .data()
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let resized = if class_name == "PPOCRV6SmallRecImageProcessor" {
        resize_f32_bilinear_no_antialias(
            &values,
            frame.height(),
            frame.width(),
            spec.edge,
            spec.edge,
        )
    } else {
        image_resize_kernels::resize_f32_torchvision(
            &values,
            image_resize_kernels::ImageSize {
                height: frame.height(),
                width: frame.width(),
            },
            3,
            image_resize_kernels::ImageSize {
                height: spec.edge,
                width: spec.edge,
            },
            image_resize_kernels::ResizeFilter::Bilinear,
        )
        .map_err(ImageProcessorError::ResizeKernel)?
    };
    let mut chw = vec![0.0; resized.len()];
    for y in 0..spec.edge {
        for x in 0..spec.edge {
            for output_channel in 0..3 {
                let input_channel = if spec.reverse_channels {
                    2 - output_channel
                } else {
                    output_channel
                };
                let mut value = resized[(y * spec.edge + x) * 3 + input_channel];
                if class_name == "PPOCRV6SmallRecImageProcessor" {
                    value = value.round().clamp(0.0, 255.0);
                }
                if spec.do_rescale {
                    value *= spec.rescale_factor;
                }
                if spec.do_normalize {
                    value = (value - spec.mean[output_channel]) / spec.std[output_channel];
                }
                chw[output_channel * spec.edge * spec.edge + y * spec.edge + x] = value;
            }
        }
    }
    MultimodalArray::new([1, 3, spec.edge, spec.edge], MultimodalArrayData::F32(chw))
}

fn process_slanext_pixels(frame: &ImageFrame) -> Result<MultimodalArray, MultimodalProcessorError> {
    let spec = pixel_spec("SLANeXtImageProcessor");
    let resized = resize_slanext_u8(frame, spec.edge);
    let mut chw = vec![0.0; resized.len()];
    for y in 0..spec.edge {
        for x in 0..spec.edge {
            for channel in 0..3 {
                let mut value = f32::from(resized[(y * spec.edge + x) * 3 + channel]);
                value *= spec.rescale_factor;
                value = (value - spec.mean[channel]) / spec.std[channel];
                chw[channel * spec.edge * spec.edge + y * spec.edge + x] = value;
            }
        }
    }
    MultimodalArray::new([1, 3, spec.edge, spec.edge], MultimodalArrayData::F32(chw))
}

fn resize_slanext_u8(frame: &ImageFrame, target_edge: usize) -> Vec<u8> {
    const WEIGHT_SCALE: i64 = 2048;
    let source_height = frame.height();
    let source_width = frame.width();
    let row_coordinates = fixed_bilinear_coordinates(source_height, target_edge);
    let column_coordinates = fixed_bilinear_coordinates(source_width, target_edge);
    let mut output = vec![0; target_edge * target_edge * 3];
    for (target_y, &(source_y, bottom_weight)) in row_coordinates.iter().enumerate() {
        let top_weight = WEIGHT_SCALE - bottom_weight;
        for (target_x, &(source_x, right_weight)) in column_coordinates.iter().enumerate() {
            let left_weight = WEIGHT_SCALE - right_weight;
            for channel in 0..3 {
                let top_left =
                    i64::from(frame.data()[(source_y * source_width + source_x) * 3 + channel]);
                let top_right =
                    i64::from(frame.data()[(source_y * source_width + source_x + 1) * 3 + channel]);
                let bottom_left = i64::from(
                    frame.data()[((source_y + 1) * source_width + source_x) * 3 + channel],
                );
                let bottom_right = i64::from(
                    frame.data()[((source_y + 1) * source_width + source_x + 1) * 3 + channel],
                );
                let interpolated = top_weight * (left_weight * top_left + right_weight * top_right)
                    + bottom_weight * (left_weight * bottom_left + right_weight * bottom_right);
                output[(target_y * target_edge + target_x) * 3 + channel] =
                    ((interpolated + (1 << 21)) >> 22).clamp(0, 255) as u8;
            }
        }
    }
    output
}

fn fixed_bilinear_coordinates(source: usize, target: usize) -> Vec<(usize, i64)> {
    let mut coordinates = Vec::with_capacity(target);
    for target_index in 0..target {
        let source_coordinate = (target_index as f32 + 0.5) * (source as f32 / target as f32) - 0.5;
        let mut source_floor = source_coordinate.floor() as isize;
        let mut fraction = source_coordinate - source_floor as f32;
        if source_floor < 0 {
            source_floor = 0;
            fraction = 0.0;
        } else if source_floor >= (source - 1) as isize {
            source_floor = (source - 2) as isize;
            fraction = 1.0;
        }
        let weight = (fraction * 2048.0 + 0.5).floor() as i64;
        coordinates.push((source_floor as usize, weight));
    }
    coordinates
}

fn resize_f32_bilinear_no_antialias(
    values: &[f32],
    source_height: usize,
    source_width: usize,
    target_height: usize,
    target_width: usize,
) -> Vec<f32> {
    let mut horizontal = vec![0.0; source_height * target_width * 3];
    let scale_x = source_width as f32 / target_width as f32;
    for y in 0..source_height {
        for x in 0..target_width {
            let source_x = ((x as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = (source_x.floor() as usize).min(source_width - 1);
            let x1 = (x0 + 1).min(source_width - 1);
            let x_weight = source_x - x0 as f32;
            for channel in 0..3 {
                let left = values[(y * source_width + x0) * 3 + channel];
                let right = values[(y * source_width + x1) * 3 + channel];
                horizontal[(y * target_width + x) * 3 + channel] =
                    (left + (right - left) * x_weight).round().clamp(0.0, 255.0);
            }
        }
    }

    let mut output = vec![0.0; target_height * target_width * 3];
    let scale_y = source_height as f32 / target_height as f32;
    for y in 0..target_height {
        let source_y = ((y as f32 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = (source_y.floor() as usize).min(source_height - 1);
        let y1 = (y0 + 1).min(source_height - 1);
        let y_weight = source_y - y0 as f32;
        for x in 0..target_width {
            for channel in 0..3 {
                let top = horizontal[(y0 * target_width + x) * 3 + channel];
                let bottom = horizontal[(y1 * target_width + x) * 3 + channel];
                output[(y * target_width + x) * 3 + channel] =
                    (top + (bottom - top) * y_weight).round().clamp(0.0, 255.0);
            }
        }
    }
    output
}

fn process_llama4_pixels(
    frame: &ImageFrame,
    resize_backend: MultimodalResizeBackend,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: true,
        height: Some(COMPACT_EDGE),
        width: Some(COMPACT_EDGE),
        resize_mode: ResizeMode::Default,
        resample: ResizeFilter::Bilinear,
        resize_parity: resize_backend.parity(),
        decode_backend: ImageDecodeBackend::ImageCrate,
        batch_execution: BatchExecution::Serial,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: false,
        rescale_factor: 1.0,
        do_normalize: false,
        image_mean: Vec::new(),
        image_std: Vec::new(),
        do_binarize: false,
        output_layout: Layout::NCHW,
    })?;
    let tensor = processor.preprocess_image(frame)?;
    let TensorData::F32(values) = tensor.data() else {
        return Err(MultimodalProcessorError::UnexpectedTensorDtype);
    };
    let values = values
        .iter()
        .map(|&value| {
            let value = bf16::from_f32(value).to_f32();
            let scaled = bf16::from_f32(value * RESCALE_FACTOR).to_f32();
            let centered = bf16::from_f32(scaled - 0.5).to_f32();
            bf16::from_f32(centered / 0.5).to_f32()
        })
        .collect();
    MultimodalArray::new(tensor.shape().to_vec(), MultimodalArrayData::F32(values))
}

fn reverse_nchw_channels(array: &mut MultimodalArray) -> Result<(), MultimodalProcessorError> {
    let [batch, channels, height, width] = array.shape.as_slice() else {
        return Err(MultimodalProcessorError::UnexpectedTensorDtype);
    };
    if *channels != 3 {
        return Err(MultimodalProcessorError::UnexpectedTensorDtype);
    }
    let plane = height
        .checked_mul(*width)
        .ok_or(MultimodalProcessorError::ShapeOverflow)?;
    match &mut array.data {
        MultimodalArrayData::F32(values) => {
            for sample in values.chunks_exact_mut(channels * plane) {
                for index in 0..plane {
                    sample.swap(index, 2 * plane + index);
                }
            }
        }
        MultimodalArrayData::U8(values) => {
            for sample in values.chunks_exact_mut(channels * plane) {
                for index in 0..plane {
                    sample.swap(index, 2 * plane + index);
                }
            }
        }
        MultimodalArrayData::I32(_)
        | MultimodalArrayData::I64(_)
        | MultimodalArrayData::Bool(_) => {
            return Err(MultimodalProcessorError::UnexpectedTensorDtype);
        }
    }
    let _ = batch;
    Ok(())
}

fn f32_pixels(
    class_name: &str,
    frame: &ImageFrame,
    resize_backend: MultimodalResizeBackend,
) -> Result<(Vec<f32>, usize), MultimodalProcessorError> {
    let array = process_pixels(class_name, frame, resize_backend)?;
    let edge = array.shape[2];
    match array.data {
        MultimodalArrayData::F32(values) => Ok((values, edge)),
        MultimodalArrayData::U8(_)
        | MultimodalArrayData::I32(_)
        | MultimodalArrayData::I64(_)
        | MultimodalArrayData::Bool(_) => Err(MultimodalProcessorError::UnexpectedTensorDtype),
    }
}

fn array_f32(
    shape: impl Into<Vec<usize>>,
    values: Vec<f32>,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    MultimodalArray::new(shape, MultimodalArrayData::F32(values))
}

fn array_i32(
    shape: impl Into<Vec<usize>>,
    values: Vec<i32>,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    MultimodalArray::new(shape, MultimodalArrayData::I32(values))
}

fn array_i64(
    shape: impl Into<Vec<usize>>,
    values: Vec<i64>,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    MultimodalArray::new(shape, MultimodalArrayData::I64(values))
}

fn array_bool(
    shape: impl Into<Vec<usize>>,
    values: Vec<u8>,
) -> Result<MultimodalArray, MultimodalProcessorError> {
    MultimodalArray::new(shape, MultimodalArrayData::Bool(values))
}

fn preprocess_fixed(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let mut output = MultimodalOutput::default();
    let pixels = process_pixels(preset.class_name, frame, processor.resize_backend)?;
    match preset.class_name {
        "DeepseekVLHybridImageProcessor" => {
            output.insert_array("high_res_pixel_values", pixels.clone());
            output.insert_array("pixel_values", pixels);
        }
        "Idefics2ImageProcessor" => {
            let values = pixels.data.into_f32()?;
            output.insert_array(
                "pixel_values",
                array_f32([1, 1, 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
            );
            let mask = match processor.resize_backend {
                MultimodalResizeBackend::Torchvision => array_f32(
                    [1, 1, COMPACT_EDGE, COMPACT_EDGE],
                    vec![1.0; COMPACT_EDGE * COMPACT_EDGE],
                )?,
                MultimodalResizeBackend::Pillow => array_i64(
                    [1, 1, COMPACT_EDGE, COMPACT_EDGE],
                    vec![1; COMPACT_EDGE * COMPACT_EDGE],
                )?,
            };
            output.insert_array("pixel_attention_mask", mask);
        }
        "BridgeTowerImageProcessor" | "ViltImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array(
                "pixel_mask",
                array_i64(
                    [1, COMPACT_EDGE, COMPACT_EDGE],
                    vec![1; COMPACT_EDGE * COMPACT_EDGE],
                )?,
            );
        }
        "Emu3ImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array(
                "image_sizes",
                array_i64([1, 2], vec![COMPACT_EDGE as i64, COMPACT_EDGE as i64])?,
            );
        }
        _ => output.insert_array("pixel_values", pixels),
    }
    Ok(output)
}

fn preprocess_tiled(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let mut output = MultimodalOutput::default();
    let pixels = process_pixels(preset.class_name, frame, processor.resize_backend)?;
    match preset.class_name {
        "AriaImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array("pixel_mask", array_bool([1, 490, 490], vec![1; 490 * 490])?);
            output.insert_array("num_crops", array_i64([1], vec![1])?);
        }
        "Cohere2VisionImageProcessor" | "GotOcr2ImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array("num_patches", array_i64([1], vec![1])?);
        }
        "DeepseekOcr2ImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array("num_local_patches", array_i64([1], vec![0])?);
        }
        "Ovis2ImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array("grids", array_i64([1, 2], vec![1, 1])?);
        }
        "Llama4ImageProcessor" => {
            output.insert_array("pixel_values", pixels);
            output.insert_array("aspect_ratios", array_i64([1, 2], vec![1, 1])?);
        }
        "LlavaOnevisionImageProcessor" => {
            let values = pixels.data.into_f32()?;
            let mut doubled = Vec::with_capacity(values.len() * 2);
            doubled.extend_from_slice(&values);
            doubled.extend_from_slice(&values);
            output.insert_array(
                "pixel_values",
                array_f32([1, 2, 3, COMPACT_EDGE, COMPACT_EDGE], doubled)?,
            );
            output.insert_array(
                "image_sizes",
                array_i64([1, 2], vec![frame.height() as i64, frame.width() as i64])?,
            );
            output.insert_array("batch_num_images", array_i64([1], vec![1])?);
        }
        "PerceptionLMImageProcessor" => {
            let values = pixels.data.into_f32()?;
            output.insert_array(
                "pixel_values",
                array_f32([1, 1, 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
            );
        }
        "SmolVLMImageProcessor" => {
            let values = pixels.data.into_f32()?;
            output.insert_array(
                "pixel_values",
                array_f32([1, 1, 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
            );
            let mask = match processor.resize_backend {
                MultimodalResizeBackend::Torchvision => array_f32(
                    [1, 1, COMPACT_EDGE, COMPACT_EDGE],
                    vec![1.0; COMPACT_EDGE * COMPACT_EDGE],
                )?,
                MultimodalResizeBackend::Pillow => array_i64(
                    [1, 1, COMPACT_EDGE, COMPACT_EDGE],
                    vec![1; COMPACT_EDGE * COMPACT_EDGE],
                )?,
            };
            output.insert_array("pixel_attention_mask", mask);
        }
        _ => output.insert_array("pixel_values", pixels),
    }
    Ok(output)
}

fn patchify_chw(values: &[f32], height: usize, width: usize, patch: usize) -> Vec<f32> {
    let grid_h = height / patch;
    let grid_w = width / patch;
    let mut patches = Vec::with_capacity(values.len());
    for grid_y in 0..grid_h {
        for grid_x in 0..grid_w {
            for channel in 0..3 {
                for patch_y in 0..patch {
                    for patch_x in 0..patch {
                        let y = grid_y * patch + patch_y;
                        let x = grid_x * patch + patch_x;
                        patches.push(values[channel * height * width + y * width + x]);
                    }
                }
            }
        }
    }
    patches
}

fn patchify_hwc(values: &[f32], height: usize, width: usize, patch: usize) -> Vec<f32> {
    let grid_h = height / patch;
    let grid_w = width / patch;
    let mut patches = Vec::with_capacity(values.len());
    for grid_y in 0..grid_h {
        for grid_x in 0..grid_w {
            for patch_y in 0..patch {
                for patch_x in 0..patch {
                    let y = grid_y * patch + patch_y;
                    let x = grid_x * patch + patch_x;
                    for channel in 0..3 {
                        patches.push(values[channel * height * width + y * width + x]);
                    }
                }
            }
        }
    }
    patches
}

fn preprocess_patch_grid(
    processor: MultimodalProcessor,
    frames: &[ImageFrame],
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let patch = patch_grid_stage();
    let mut tensors = Vec::with_capacity(frames.len());
    for frame in frames {
        let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
        tensors.push(
            Tensor::new(TensorData::F32(values), [3, edge, edge], Layout::CHW)
                .map_err(RecipePatchError::from)?,
        );
    }
    let edge = pixel_spec(preset.class_name).edge;
    let target = ImageSize {
        height: edge,
        width: edge,
    };
    let flattened = patch.flatten_temporal_tensors(&tensors, target)?;
    let image_grid = patch.image_grid_thw(target)?.map(|value| value as i64);
    let grids = std::iter::repeat_n(image_grid, frames.len())
        .flatten()
        .collect::<Vec<_>>();
    let (data, shape, _, _) = flattened.into_parts();
    let TensorData::F32(values) = data else {
        return Err(MultimodalProcessorError::UnexpectedTensorDtype);
    };
    let mut output = MultimodalOutput::default();
    output.insert_array("pixel_values", array_f32(shape, values)?);
    output.insert_array("image_grid_thw", array_i64([frames.len(), 3], grids)?);
    if preset.class_name == "VideoLlama3ImageProcessor" {
        output.insert_array(
            "image_merge_sizes",
            array_i64([frames.len()], vec![1; frames.len()])?,
        );
    }
    Ok(output)
}

fn patch_grid_stage() -> RecipePatchStage {
    RecipePatchStage::flatten(Layout::CHW, PATCH_SIZE, 1, 1)
}

fn preprocess_patch_frames(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
    let patches = patchify_chw(&values, edge, edge, PATCH_SIZE);
    let count = (edge / PATCH_SIZE) * (edge / PATCH_SIZE);
    let mut output = MultimodalOutput::default();
    output.insert_array(
        "pixel_values",
        array_f32([count, 3, PATCH_SIZE, PATCH_SIZE], patches)?,
    );
    output.insert_array(
        "image_grid_thw",
        array_i64(
            [1, 3],
            vec![1, (edge / PATCH_SIZE) as i64, (edge / PATCH_SIZE) as i64],
        )?,
    );
    Ok(output)
}

fn preprocess_fuyu(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let pixels = process_pixels(preset.class_name, frame, processor.resize_backend)?;
    let values = pixels.data.into_f32()?;
    let mut output = MultimodalOutput::default();
    output.insert_array(
        "images",
        array_f32([1, 1, 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
    );
    output.insert_array("image_unpadded_heights", array_i64([1, 1], vec![8])?);
    output.insert_array("image_unpadded_widths", array_i64([1, 1], vec![8])?);
    output.insert_array(
        "image_scale_factors",
        array_f32([1, 1], vec![COMPACT_EDGE as f32 / frame.height() as f32])?,
    );
    Ok(output)
}

fn preprocess_gemma4(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
    let real_patches = patchify_hwc(&values, edge, edge, PATCH_SIZE);
    let patch_depth = 3 * PATCH_SIZE * PATCH_SIZE;
    let real_count = (edge / PATCH_SIZE) * (edge / PATCH_SIZE);
    let max_patches = 70;
    let mut patches = vec![0.0; max_patches * patch_depth];
    patches[..real_patches.len()].copy_from_slice(&real_patches);
    let mut positions = vec![-1; max_patches * 2];
    for y in 0..edge / PATCH_SIZE {
        for x in 0..edge / PATCH_SIZE {
            let index = y * (edge / PATCH_SIZE) + x;
            positions[index * 2] = x as i64;
            positions[index * 2 + 1] = y as i64;
        }
    }
    let mut output = MultimodalOutput::default();
    output.insert_array(
        "pixel_values",
        array_f32([1, max_patches, patch_depth], patches)?,
    );
    output.insert_array(
        "image_position_ids",
        array_i64([1, max_patches, 2], positions)?,
    );
    output.insert_array(
        "num_soft_tokens_per_image",
        array_i64([1], vec![real_count as i64])?,
    );
    Ok(output)
}

fn standardized_pixels(
    frame: &ImageFrame,
    resize_backend: MultimodalResizeBackend,
) -> Result<Vec<f32>, MultimodalProcessorError> {
    let values = frame
        .data()
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let difference = *value - mean;
            difference * difference
        })
        .sum::<f32>()
        / match resize_backend {
            MultimodalResizeBackend::Torchvision => (values.len() - 1) as f32,
            MultimodalResizeBackend::Pillow => values.len() as f32,
        };
    let std = variance.sqrt().max(1.0 / (values.len() as f32).sqrt());
    let normalized = values
        .into_iter()
        .map(|value| (value - mean) / std)
        .collect::<Vec<_>>();
    let resized = image_resize_kernels::resize_f32_torchvision(
        &normalized,
        image_resize_kernels::ImageSize {
            height: frame.height(),
            width: frame.width(),
        },
        3,
        image_resize_kernels::ImageSize {
            height: COMPACT_EDGE,
            width: COMPACT_EDGE,
        },
        image_resize_kernels::ResizeFilter::Bilinear,
    )
    .map_err(ImageProcessorError::ResizeKernel)?;
    let mut chw = vec![0.0; resized.len()];
    for y in 0..COMPACT_EDGE {
        for x in 0..COMPACT_EDGE {
            for channel in 0..3 {
                chw[channel * COMPACT_EDGE * COMPACT_EDGE + y * COMPACT_EDGE + x] =
                    resized[(y * COMPACT_EDGE + x) * 3 + channel];
            }
        }
    }
    Ok(chw)
}

fn preprocess_adaptive(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let mut output = MultimodalOutput::default();
    let grid = COMPACT_EDGE / PATCH_SIZE;
    let patch_count = grid * grid;
    let patch_depth = 3 * PATCH_SIZE * PATCH_SIZE;
    match preset.class_name {
        "Pix2StructImageProcessor" | "Kosmos2_5ImageProcessor" => {
            let normalized = standardized_pixels(frame, processor.resize_backend)?;
            let patches = patchify_hwc(&normalized, COMPACT_EDGE, COMPACT_EDGE, PATCH_SIZE);
            let mut positioned = Vec::with_capacity(patch_count * (patch_depth + 2));
            for patch_index in 0..patch_count {
                positioned.push((patch_index / grid + 1) as f32);
                positioned.push((patch_index % grid + 1) as f32);
                let start = patch_index * patch_depth;
                positioned.extend_from_slice(&patches[start..start + patch_depth]);
            }
            output.insert_array(
                "flattened_patches",
                array_f32([1, patch_count, patch_depth + 2], positioned)?,
            );
            output.insert_array(
                "attention_mask",
                array_f32([1, patch_count], vec![1.0; patch_count])?,
            );
            if preset.class_name == "Kosmos2_5ImageProcessor" {
                output.insert_array("rows", array_i64([1], vec![grid as i64])?);
                output.insert_array("cols", array_i64([1], vec![grid as i64])?);
                output.insert_array("height", array_i64([1], vec![8])?);
                output.insert_array("width", array_i64([1], vec![8])?);
            }
        }
        "Siglip2ImageProcessor" | "Lfm2VlImageProcessor" => {
            let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
            let patches = patchify_hwc(&values, edge, edge, PATCH_SIZE);
            output.insert_array(
                "pixel_values",
                array_f32([1, patch_count, patch_depth], patches)?,
            );
            output.insert_array(
                "pixel_attention_mask",
                array_i32([1, patch_count], vec![1; patch_count])?,
            );
            output.insert_array(
                "spatial_shapes",
                array_i64([1, 2], vec![grid as i64, grid as i64])?,
            );
            if preset.class_name == "Lfm2VlImageProcessor" {
                output.insert_array("image_sizes", array_i64([1, 2], vec![8, 8])?);
                output.insert_array("image_rows", array_i64([1], vec![1])?);
                output.insert_array("image_cols", array_i64([1], vec![1])?);
            }
        }
        _ => {
            return Err(MultimodalProcessorError::UnknownClass(
                preset.class_name.to_owned(),
            ))
        }
    }
    Ok(output)
}

fn preprocess_minicpm(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
    let grid = edge / PATCH_SIZE;
    let mut reshaped = Vec::with_capacity(values.len());
    for channel in 0..3 {
        for patch_y in 0..PATCH_SIZE {
            for grid_y in 0..grid {
                for grid_x in 0..grid {
                    for patch_x in 0..PATCH_SIZE {
                        let y = grid_y * PATCH_SIZE + patch_y;
                        let x = grid_x * PATCH_SIZE + patch_x;
                        reshaped.push(values[channel * edge * edge + y * edge + x]);
                    }
                }
            }
        }
    }
    let mut output = MultimodalOutput::default();
    output.insert_array(
        "pixel_values",
        array_f32([1, 3, PATCH_SIZE, edge * edge / PATCH_SIZE], reshaped)?,
    );
    output.insert_array(
        "target_sizes",
        array_i32([1, 2], vec![grid as i32, grid as i32])?,
    );
    output.insert_value(
        "grids",
        MultimodalValue::Sequence(vec![MultimodalValue::Sequence(vec![
            MultimodalValue::Integer(0),
            MultimodalValue::Integer(0),
        ])]),
    );
    output.insert_value(
        "num_patches_per_image",
        MultimodalValue::Sequence(vec![MultimodalValue::Integer(1)]),
    );
    Ok(output)
}

fn preprocess_phi4(
    processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let (values, edge) = f32_pixels(preset.class_name, frame, processor.resize_backend)?;
    let mut doubled = Vec::with_capacity(values.len() * 2);
    doubled.extend_from_slice(&values);
    doubled.extend_from_slice(&values);
    let mask_edge = edge / PATCH_SIZE;
    let mut output = MultimodalOutput::default();
    output.insert_array(
        "image_pixel_values",
        array_f32([1, 2, 3, edge, edge], doubled)?,
    );
    output.insert_array(
        "image_attention_mask",
        array_f32(
            [1, 2, mask_edge, mask_edge],
            vec![1.0; 2 * mask_edge * mask_edge],
        )?,
    );
    output.insert_array(
        "image_sizes",
        array_i64([1, 2], vec![edge as i64, edge as i64])?,
    );
    let downsampled = mask_edge.div_ceil(2);
    let tokens = 256 + 1 + downsampled * downsampled + downsampled + 16;
    output.insert_array("num_img_tokens", array_i64([1], vec![tokens as i64])?);
    Ok(output)
}

fn preprocess_video(
    processor: MultimodalProcessor,
    frames: &[ImageFrame],
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let preset = processor.preset;
    let mut values = Vec::new();
    for frame in frames {
        let array = process_pixels(preset.class_name, frame, processor.resize_backend)?;
        match array.data {
            MultimodalArrayData::F32(frame_values) => values.extend(frame_values),
            _ => return Err(MultimodalProcessorError::UnexpectedTensorDtype),
        }
    }
    let mut output = MultimodalOutput::default();
    if preset.class_name == "TvpImageProcessor" {
        output.insert_array(
            "pixel_values",
            array_f32([1, frames.len(), 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
        );
    } else {
        output.insert_array(
            "pixel_values_images",
            array_f32([frames.len(), 3, COMPACT_EDGE, COMPACT_EDGE], values)?,
        );
    }
    Ok(output)
}

fn preprocess_uvdoc(
    _processor: MultimodalProcessor,
    frame: &ImageFrame,
) -> Result<MultimodalOutput, MultimodalProcessorError> {
    let mut original_values = vec![0.0; frame.height() * frame.width() * 3];
    for y in 0..frame.height() {
        for x in 0..frame.width() {
            for output_channel in 0..3 {
                let input_channel = 2 - output_channel;
                original_values
                    [output_channel * frame.height() * frame.width() + y * frame.width() + x] =
                    f32::from(frame.data()[(y * frame.width() + x) * 3 + input_channel])
                        * RESCALE_FACTOR;
            }
        }
    }
    let resized = resize_f32_chw_align_corners(
        &original_values,
        frame.height(),
        frame.width(),
        COMPACT_EDGE,
        COMPACT_EDGE,
    );
    let pixels = array_f32([1, 3, COMPACT_EDGE, COMPACT_EDGE], resized)?;
    let original = array_f32([3, frame.height(), frame.width()], original_values)?;
    let mut output = MultimodalOutput::default();
    output.insert_array("pixel_values", pixels);
    output.insert_value(
        "original_images",
        MultimodalValue::Sequence(vec![MultimodalValue::Array(original)]),
    );
    Ok(output)
}

fn resize_f32_chw_align_corners(
    values: &[f32],
    source_height: usize,
    source_width: usize,
    target_height: usize,
    target_width: usize,
) -> Vec<f32> {
    if source_height == target_height && source_width == target_width {
        return values.to_vec();
    }
    let scale_y = if target_height > 1 {
        (source_height - 1) as f32 / (target_height - 1) as f32
    } else {
        0.0
    };
    let scale_x = if target_width > 1 {
        (source_width - 1) as f32 / (target_width - 1) as f32
    } else {
        0.0
    };
    let mut output = vec![0.0; 3 * target_height * target_width];
    for channel in 0..3 {
        for y in 0..target_height {
            let source_y = y as f32 * scale_y;
            let y0 = (source_y.floor() as usize).min(source_height - 1);
            let y1 = (y0 + 1).min(source_height - 1);
            let y_weight = source_y - y0 as f32;
            for x in 0..target_width {
                let source_x = x as f32 * scale_x;
                let x0 = (source_x.floor() as usize).min(source_width - 1);
                let x1 = (x0 + 1).min(source_width - 1);
                let x_weight = source_x - x0 as f32;
                let plane = channel * source_height * source_width;
                let top_left = values[plane + y0 * source_width + x0];
                let top_right = values[plane + y0 * source_width + x1];
                let bottom_left = values[plane + y1 * source_width + x0];
                let bottom_right = values[plane + y1 * source_width + x1];
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                output[channel * target_height * target_width + y * target_width + x] =
                    top + (bottom - top) * y_weight;
            }
        }
    }
    output
}

/// One decoded OCR recognition result.
#[derive(Clone, Debug, PartialEq)]
pub struct TextRecognition {
    /// Greedy CTC-decoded text.
    pub text: String,
    /// Mean selected-token confidence.
    pub score: f32,
}

/// Greedily decodes PPOCR recognition logits with duplicate and blank removal.
///
/// `logits` uses `[batch, steps, vocabulary]` row-major storage.
///
/// # Errors
///
/// Returns an error when the shape does not match `logits` or an argmax token
/// is outside `characters`.
pub fn post_process_text_recognition(
    logits: &[f32],
    batch: usize,
    steps: usize,
    vocabulary: usize,
    characters: &[&str],
) -> Result<Vec<TextRecognition>, MultimodalProcessorError> {
    let expected = batch
        .checked_mul(steps)
        .and_then(|value| value.checked_mul(vocabulary))
        .ok_or(MultimodalProcessorError::ShapeOverflow)?;
    if logits.len() != expected {
        return Err(MultimodalProcessorError::InvalidLogitsLength {
            expected,
            actual: logits.len(),
        });
    }
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::Ocr,
        "last_hidden_state",
        RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id: 0 },
    );
    let sequences = decode_token_sequences(&descriptor, logits, batch, steps, vocabulary)?;
    let mut results = Vec::with_capacity(sequences.len());
    for sequence in sequences {
        let mut text = String::new();
        for &index in sequence.token_ids() {
            let character =
                characters
                    .get(index)
                    .ok_or(MultimodalProcessorError::VocabularyIndex {
                        index,
                        vocabulary_len: characters.len(),
                    })?;
            text.push_str(character);
        }
        results.push(TextRecognition {
            text,
            score: sequence.mean_score().unwrap_or(0.0),
        });
    }
    Ok(results)
}

/// Decoded SLANeXt table structure.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRecognition {
    /// HTML structure tokens including document wrapper tags.
    pub structure: Vec<String>,
    /// Mean selected-token confidence.
    pub score: f32,
}

/// Greedily decodes one SLANeXt table structure sequence.
///
/// # Errors
///
/// Returns an error when the logits shape is invalid or a predicted index is
/// outside `vocabulary`.
pub fn post_process_table_recognition(
    logits: &[f32],
    steps: usize,
    vocabulary: &[&str],
    bos_id: usize,
    eos_id: usize,
) -> Result<TableRecognition, MultimodalProcessorError> {
    let expected = steps
        .checked_mul(vocabulary.len())
        .ok_or(MultimodalProcessorError::ShapeOverflow)?;
    if logits.len() != expected {
        return Err(MultimodalProcessorError::InvalidLogitsLength {
            expected,
            actual: logits.len(),
        });
    }
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::TableStructure,
        "last_hidden_state",
        RecipeTokenSequenceDecoder::Greedy {
            begin_token_id: Some(bos_id),
            end_token_id: Some(eos_id),
        },
    );
    let sequence = decode_token_sequences(&descriptor, logits, 1, steps, vocabulary.len())?
        .into_iter()
        .next()
        .ok_or(MultimodalProcessorError::EmptyInput)?;
    let mut structure = vec![
        "<html>".to_owned(),
        "<body>".to_owned(),
        "<table>".to_owned(),
    ];
    for &index in sequence.token_ids() {
        let token = vocabulary
            .get(index)
            .ok_or(MultimodalProcessorError::VocabularyIndex {
                index,
                vocabulary_len: vocabulary.len(),
            })?;
        structure.push((*token).to_owned());
    }
    structure.extend([
        "</table>".to_owned(),
        "</body>".to_owned(),
        "</html>".to_owned(),
    ]);
    Ok(TableRecognition {
        structure,
        score: sequence.mean_score().unwrap_or(0.0),
    })
}

/// One OCR word with a LayoutLM-style normalized bounding box.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OcrWord {
    /// Recognized word text.
    pub text: String,
    /// Corner box normalized to the inclusive `0..=1000` page coordinate space.
    pub box_1000: [u16; 4],
}

/// Normalizes a pixel-space OCR word box into LayoutLM page coordinates.
///
/// # Errors
///
/// Returns an error when the page size is zero.
pub fn normalize_ocr_word(
    text: impl Into<String>,
    bounds: [usize; 4],
    page_width: usize,
    page_height: usize,
) -> Result<OcrWord, MultimodalProcessorError> {
    if page_width == 0 || page_height == 0 {
        return Err(MultimodalProcessorError::ZeroArrayDimension);
    }
    let x0 = (bounds[0].min(page_width) * 1000 / page_width) as u16;
    let y0 = (bounds[1].min(page_height) * 1000 / page_height) as u16;
    let x1 = (bounds[2].min(page_width) * 1000 / page_width) as u16;
    let y1 = (bounds[3].min(page_height) * 1000 / page_height) as u16;
    Ok(OcrWord {
        text: text.into(),
        box_1000: [x0, y0, x1, y1],
    })
}

/// Rectified BGR document image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RectifiedDocument {
    /// Output image width.
    pub width: usize,
    /// Output image height.
    pub height: usize,
    /// Interleaved BGR bytes.
    pub bgr: Vec<u8>,
}

/// Applies an align-corners bilinear UVDoc sampling grid to a CHW image.
///
/// Coordinates are normalized to `[-1, 1]`, matching `torch.grid_sample`.
///
/// # Errors
///
/// Returns an error when input or grid shapes do not match their data.
pub fn post_process_document_rectification(
    image_chw: &[f32],
    image_width: usize,
    image_height: usize,
    grid: &[[f32; 2]],
    output_width: usize,
    output_height: usize,
    scale: f32,
) -> Result<RectifiedDocument, MultimodalProcessorError> {
    if image_width == 0 || image_height == 0 {
        return Err(MultimodalProcessorError::InvalidRectificationDimensions {
            field: "image",
            width: image_width,
            height: image_height,
        });
    }
    if output_width == 0 || output_height == 0 {
        return Err(MultimodalProcessorError::InvalidRectificationDimensions {
            field: "output",
            width: output_width,
            height: output_height,
        });
    }
    if !scale.is_finite() {
        return Err(MultimodalProcessorError::NonFiniteRectificationScale(scale));
    }
    let expected_image = image_width
        .checked_mul(image_height)
        .and_then(|value| value.checked_mul(3))
        .ok_or(MultimodalProcessorError::ShapeOverflow)?;
    if image_chw.len() != expected_image {
        return Err(MultimodalProcessorError::InvalidArrayLength {
            expected: expected_image,
            actual: image_chw.len(),
        });
    }
    if let Some((index, &value)) = image_chw
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(MultimodalProcessorError::NonFiniteRectificationImageValue { index, value });
    }
    let expected_grid = output_width
        .checked_mul(output_height)
        .ok_or(MultimodalProcessorError::ShapeOverflow)?;
    if grid.len() != expected_grid {
        return Err(MultimodalProcessorError::InvalidArrayLength {
            expected: expected_grid,
            actual: grid.len(),
        });
    }
    for (index, &[x, y]) in grid.iter().enumerate() {
        if !x.is_finite() {
            return Err(
                MultimodalProcessorError::NonFiniteRectificationGridCoordinate {
                    index,
                    axis: "x",
                    value: x,
                },
            );
        }
        if !y.is_finite() {
            return Err(
                MultimodalProcessorError::NonFiniteRectificationGridCoordinate {
                    index,
                    axis: "y",
                    value: y,
                },
            );
        }
    }
    let mut bgr = Vec::with_capacity(expected_grid * 3);
    for &[grid_x, grid_y] in grid {
        let source_x = (grid_x + 1.0) * 0.5 * (image_width.saturating_sub(1) as f32);
        let source_y = (grid_y + 1.0) * 0.5 * (image_height.saturating_sub(1) as f32);
        for channel in (0..3).rev() {
            let value = bilinear_sample(
                image_chw,
                image_width,
                image_height,
                channel,
                source_x,
                source_y,
            );
            bgr.push((value * scale) as u8);
        }
    }
    Ok(RectifiedDocument {
        width: output_width,
        height: output_height,
        bgr,
    })
}

fn bilinear_sample(
    image: &[f32],
    width: usize,
    height: usize,
    channel: usize,
    x: f32,
    y: f32,
) -> f32 {
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let x_weight = x - x0 as f32;
    let y_weight = y - y0 as f32;
    let plane = width * height;
    let at =
        |sample_x: usize, sample_y: usize| image[channel * plane + sample_y * width + sample_x];
    let top = at(x0, y0) * (1.0 - x_weight) + at(x1, y0) * x_weight;
    let bottom = at(x0, y1) * (1.0 - x_weight) + at(x1, y1) * x_weight;
    top * (1.0 - y_weight) + bottom * y_weight
}
