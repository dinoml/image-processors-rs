//! Processor-family configurations built on shared Rust primitives.
//!
//! This module provides model-family-oriented Rust APIs without introducing
//! bindings or moving tensor ownership out of the crate.

use std::borrow::Cow;
use std::path::Path;

use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::image::{
    stack_frame_tensors_as_video_batch, BatchExecution, ImageProcessor, ImageProcessorConfig,
    ImageProcessorError, ImageProcessorOptions, ImageProcessorWorkspace,
};
use crate::media::{
    load_image_from_path_with_backend, DefaultMediaLoader, ImageDecodeBackend, ImageFrame,
    ImageSequence, LoadedMedia, MediaSource, PixelFormat, VideoClip,
};
use crate::output::{
    ProcessorMetadataName, ProcessorMetadataValue, ProcessorOutput, ProcessorTensorName,
};
pub use crate::postprocess::BinaryMaskPostprocessOutput;
use crate::postprocess::{
    post_process_logc3_hdr_video_tensor, post_process_video_tensor, TensorPostprocessError,
};
use crate::recipe::{
    ProcessorRecipe, ProcessorRecipeInput, ProcessorRecipeOutput, ProcessorRecipePostprocess,
    ProcessorRecipeStage, RecipeAspectRatioCropStage, RecipeBinaryMaskPostprocess,
    RecipeDocumentGeometryStage, RecipeError, RecipeImageSizeSource, RecipeImageSplitStage,
    RecipeObjectDetectionPostprocess, RecipePatchError, RecipePatchGridStage, RecipePatchStage,
    RecipeResizeStage, RecipeTiledCanvasStage,
};
use crate::tensor::{
    DType, ImageLayout, Layout, Tensor, TensorData, TensorError, TensorLeadingAxis, TensorView,
    VideoLayout,
};
mod encoder_presets;
/// Aspect-preserving SigLIP2 NaFlex patch preparation.
pub mod siglip2;
pub use encoder_presets::{
    EncoderGeometry, EncoderImageProcessor, EncoderImageProcessorConfig,
    EncoderImageProcessorPreset, PresetProcessorError, Swin2SrImageProcessor,
    Swin2SrImageProcessorConfig,
};
mod task_vision_presets;
pub use task_vision_presets::{
    PaddleDetectionLimit, TaskVisionColorMode, TaskVisionImageProcessor,
    TaskVisionImageProcessorConfig, TaskVisionKeypointMatchingOutput, TaskVisionPadding,
    TaskVisionPoseBox, TaskVisionProcessorFamily, TaskVisionProcessorPreset, TaskVisionResize,
    TaskVisionSegmentationRequest,
};
mod multimodal_presets;
pub use multimodal_presets::{
    multimodal_preset, normalize_ocr_word, post_process_document_rectification,
    post_process_table_recognition, post_process_text_recognition, MultimodalArray,
    MultimodalArrayData, MultimodalCategory, MultimodalOutput, MultimodalPreset,
    MultimodalProcessor, MultimodalProcessorError, MultimodalProfile, MultimodalRecipe,
    MultimodalRecipeError, MultimodalResizeBackend, MultimodalValue, OcrWord, RectifiedDocument,
    TableRecognition, TextRecognition, MULTIMODAL_CLASS_ALIASES, MULTIMODAL_PRESETS,
};
mod marigold;
pub use marigold::{
    MarigoldImageProcessor, MarigoldImageProcessorConfig, MarigoldPreprocessOutput,
};
mod vae;
use vae::validate_vae_config;
pub use vae::{
    DepthMapU16, InpaintOverlayContext, InpaintPreprocessOutput, Ldm3dPostprocessOutput,
    Ldm3dPreprocessOutput, VaeImageProcessor, VaeImageProcessorConfig, VaeImageProcessorLdm3d,
    VaeOutputType, VaePostprocessOutput,
};
mod visual_cloze;
pub use visual_cloze::{
    VisualClozeNestedTensors, VisualClozePreprocessOutput, VisualClozePreprocessParts,
    VisualClozeProcessor, VisualClozeProcessorConfig,
};
mod detection;
#[cfg(test)]
use detection::shortest_edge_resize_output_size;
use detection::{checked_spatial_pad_mul, pad_spatial_tensors, validate_padding_target};
pub use detection::{DetrImageProcessorConfig, SamImageProcessorConfig, ShortestEdgeResizeConfig};
mod hf_config;
use hf_config::{default_true, HfPreprocessorConfig};
mod video;
pub use video::{
    VideoMaeImageProcessor, VideoMaeImageProcessorConfig, VivitImageProcessor,
    VivitImageProcessorConfig,
};
mod diffusion;
pub use diffusion::{
    BlipImageProcessor, BlipImageProcessorConfig, Flux2ImageProcessor, Flux2ImageProcessorConfig,
    HunyuanVideo15ImageProcessor, HunyuanVideo15ImageProcessorConfig, JoyImageEditImageProcessor,
    JoyImageEditImageProcessorConfig, Ltx2VideoHdrProcessor, Ltx2VideoHdrProcessorConfig,
    WanAnimateImageProcessor, WanAnimateImageProcessorConfig,
};
mod vlm;
use vlm::validate_positive_dimension;
pub use vlm::{
    Gemma3ImageProcessor, Gemma3ImageProcessorConfig, Idefics3ImageProcessor,
    Idefics3ImageProcessorConfig, LlavaNextImageProcessor, LlavaNextImageProcessorConfig,
    MllamaImageProcessor, MllamaImageProcessorConfig, PixtralImageProcessor,
    PixtralImageProcessorConfig, QwenVlImageProcessor, QwenVlImageProcessorConfig,
};

use crate::transforms::{
    area_resize_plan, aspect_ratio_crop_plan, center_crop_frame, centered_padding,
    concatenate_frames_horizontally_rgb, convert_frame_pixel_format, crop_frame, crop_region,
    fit_inside_size, inpaint_overlay, nested_frame_batch_padding_plan, pad_frame,
    pad_frame_with_canvas_fill, patch_aligned_resize_plan, patch_grid_batch_plan,
    patch_grid_image_patches_with_decision, post_process_binary_mask,
    post_process_object_detection as post_process_detr_object_detection,
    resize_center_crop_frame_with_decision, resize_fill_plan, resize_frame,
    resize_frame_with_decision, scale_factor_resize_plan, select_aspect_ratio_bucket,
    shortest_edge_resize_size, should_rotate_to_match_orientation, spatial_batch_padding_plan,
    split_image_batch_metadata, split_image_encoder_size, split_image_resize_size,
    tiled_canvas_batch_metadata, tiled_canvas_plan, validate_image_size_constraints,
    AspectRatioCropOptions, AspectRatioCropPlan, CanvasFill, DetectionCenterBox, ImageCropBox,
    ImageSize, ImageSizeConstraints, ObjectDetectionPrediction, Padding, PatchAlignedResizePlan,
    PatchGridPlan, ResizeDecision, ResizeFilter, ResizeLimits, ResizeMode, ResizeParity,
    ResizeRounding, SplitImagePlan, TiledCanvasGrid, TiledCanvasPlan, TransformError,
};

const DEFAULT_RESCALE_FACTOR: f32 = 1.0 / 255.0;
const DIFFUSION_VAE_IMAGE_MEAN: [f32; 1] = [0.5];
const DIFFUSION_VAE_IMAGE_STD: [f32; 1] = [0.5];
const CLIP_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];
const STANDARD_IMAGE_MEAN: [f32; 3] = [0.5; 3];
const STANDARD_IMAGE_STD: [f32; 3] = [0.5; 3];
const VIVIT_RESCALE_FACTOR: f32 = 1.0 / 127.5;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const SAM_IMAGE_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const SAM_IMAGE_STD: [f32; 3] = [58.395, 57.12, 57.375];
const QWEN_VL_PATCH_SIZE: usize = 14;
const QWEN_VL_TEMPORAL_PATCH_SIZE: usize = 2;
const QWEN_VL_MERGE_SIZE: usize = 2;
const LLAVA_NEXT_PATCH_SIZE: usize = 224;
const PIXTRAL_MAX_EDGE: usize = 1024;
const PIXTRAL_PATCH_SIZE: usize = 16;
const BLIP_IMAGE_SIZE: usize = 224;
const FLUX2_VAE_SCALE_FACTOR: usize = 16;
const FLUX2_VAE_LATENT_CHANNELS: usize = 32;
const FLUX2_MIN_SIDE_LENGTH: usize = 64;
const FLUX2_MAX_ASPECT_RATIO: f64 = 8.0;
const FLUX2_DEFAULT_MAX_AREA: usize = 1024 * 1024;
const VISUAL_CLOZE_RESOLUTION: usize = 384;
const VISUAL_CLOZE_VAE_SCALE_FACTOR: usize = 16;
const VISUAL_CLOZE_VAE_LATENT_CHANNELS: usize = 16;
const HUNYUAN_VIDEO_15_VAE_SCALE_FACTOR: usize = 16;
const HUNYUAN_VIDEO_15_VAE_LATENT_CHANNELS: usize = 32;
const HUNYUAN_VIDEO_15_MAX_BUCKET_RATIO: f64 = 4.0;
const JOY_IMAGE_BASE_SIZE: usize = 1024;
const IDEFICS3_DEFAULT_MAX_IMAGE_EDGE: usize = 364;
const IDEFICS3_DEFAULT_LONGEST_EDGE: usize = 4 * IDEFICS3_DEFAULT_MAX_IMAGE_EDGE;
const GEMMA3_IMAGE_SIZE: usize = 224;
const GEMMA3_PAN_AND_SCAN_MIN_CROP_SIZE: usize = 256;
const GEMMA3_PAN_AND_SCAN_MAX_NUM_CROPS: usize = 4;
const GEMMA3_PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE: f64 = 1.2;
const MLLAMA_TILE_SIZE: usize = 224;
const MLLAMA_MAX_IMAGE_TILES: usize = 4;
const LLAVA_NEXT_DEFAULT_GRID_PINPOINTS: [ImageSize; 5] = [
    ImageSize {
        height: 336,
        width: 672,
    },
    ImageSize {
        height: 672,
        width: 336,
    },
    ImageSize {
        height: 672,
        width: 672,
    },
    ImageSize {
        height: 1008,
        width: 336,
    },
    ImageSize {
        height: 336,
        width: 1008,
    },
];
const JOY_IMAGE_BUCKETS_1024: [ImageSize; 49] = [
    ImageSize {
        height: 512,
        width: 1792,
    },
    ImageSize {
        height: 512,
        width: 1856,
    },
    ImageSize {
        height: 512,
        width: 1920,
    },
    ImageSize {
        height: 512,
        width: 1984,
    },
    ImageSize {
        height: 512,
        width: 2048,
    },
    ImageSize {
        height: 576,
        width: 1600,
    },
    ImageSize {
        height: 576,
        width: 1664,
    },
    ImageSize {
        height: 576,
        width: 1728,
    },
    ImageSize {
        height: 576,
        width: 1792,
    },
    ImageSize {
        height: 640,
        width: 1472,
    },
    ImageSize {
        height: 640,
        width: 1536,
    },
    ImageSize {
        height: 640,
        width: 1600,
    },
    ImageSize {
        height: 704,
        width: 1344,
    },
    ImageSize {
        height: 704,
        width: 1408,
    },
    ImageSize {
        height: 704,
        width: 1472,
    },
    ImageSize {
        height: 768,
        width: 1216,
    },
    ImageSize {
        height: 768,
        width: 1280,
    },
    ImageSize {
        height: 768,
        width: 1344,
    },
    ImageSize {
        height: 832,
        width: 1152,
    },
    ImageSize {
        height: 832,
        width: 1216,
    },
    ImageSize {
        height: 896,
        width: 1088,
    },
    ImageSize {
        height: 896,
        width: 1152,
    },
    ImageSize {
        height: 960,
        width: 1024,
    },
    ImageSize {
        height: 960,
        width: 1088,
    },
    ImageSize {
        height: 1024,
        width: 960,
    },
    ImageSize {
        height: 1024,
        width: 1024,
    },
    ImageSize {
        height: 1088,
        width: 896,
    },
    ImageSize {
        height: 1088,
        width: 960,
    },
    ImageSize {
        height: 1152,
        width: 832,
    },
    ImageSize {
        height: 1152,
        width: 896,
    },
    ImageSize {
        height: 1216,
        width: 768,
    },
    ImageSize {
        height: 1216,
        width: 832,
    },
    ImageSize {
        height: 1280,
        width: 768,
    },
    ImageSize {
        height: 1344,
        width: 704,
    },
    ImageSize {
        height: 1344,
        width: 768,
    },
    ImageSize {
        height: 1408,
        width: 704,
    },
    ImageSize {
        height: 1472,
        width: 640,
    },
    ImageSize {
        height: 1472,
        width: 704,
    },
    ImageSize {
        height: 1536,
        width: 640,
    },
    ImageSize {
        height: 1600,
        width: 576,
    },
    ImageSize {
        height: 1600,
        width: 640,
    },
    ImageSize {
        height: 1664,
        width: 576,
    },
    ImageSize {
        height: 1728,
        width: 576,
    },
    ImageSize {
        height: 1792,
        width: 512,
    },
    ImageSize {
        height: 1792,
        width: 576,
    },
    ImageSize {
        height: 1856,
        width: 512,
    },
    ImageSize {
        height: 1920,
        width: 512,
    },
    ImageSize {
        height: 1984,
        width: 512,
    },
    ImageSize {
        height: 2048,
        width: 512,
    },
];

/// Strong Rust configuration parsed from an HF `preprocessor_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessorFamilyConfig {
    /// Diffusers BLIP image processor configuration.
    Blip(BlipImageProcessorConfig),
    /// Diffusers VAE image processor configuration.
    Vae(VaeImageProcessorConfig),
    /// Diffusers LDM3D VAE image processor configuration.
    VaeLdm3d(VaeImageProcessorConfig),
    /// Diffusers Flux2 image processor configuration.
    Flux2(Flux2ImageProcessorConfig),
    /// Diffusers VisualCloze image processor configuration.
    VisualCloze(VisualClozeProcessorConfig),
    /// Diffusers HunyuanVideo 1.5 image/video processor configuration.
    HunyuanVideo15(HunyuanVideo15ImageProcessorConfig),
    /// Diffusers Marigold dense-prediction processor configuration.
    Marigold(MarigoldImageProcessorConfig),
    /// Diffusers JoyImage edit image processor configuration.
    JoyImageEdit(JoyImageEditImageProcessorConfig),
    /// Diffusers Wan Animate image processor configuration.
    WanAnimate(WanAnimateImageProcessorConfig),
    /// Diffusers LTX2 HDR video processor configuration.
    Ltx2VideoHdr(Ltx2VideoHdrProcessorConfig),
    /// CLIP image processor configuration.
    Clip(ClipImageProcessorConfig),
    /// ViT image processor configuration.
    Vit(VitImageProcessorConfig),
    /// VideoMAE image processor configuration.
    VideoMae(VideoMaeImageProcessorConfig),
    /// ViViT image processor configuration.
    Vivit(VivitImageProcessorConfig),
    /// Qwen/VLM image processor configuration.
    QwenVl(QwenVlImageProcessorConfig),
    /// LLaVA-NeXT AnyRes image processor configuration.
    LlavaNext(LlavaNextImageProcessorConfig),
    /// Pixtral patch-aligned image processor configuration.
    Pixtral(PixtralImageProcessorConfig),
    /// Idefics3 split-image processor configuration.
    Idefics3(Idefics3ImageProcessorConfig),
    /// Gemma3 pan-and-scan image processor configuration.
    Gemma3(Gemma3ImageProcessorConfig),
    /// Mllama tiled-image processor configuration.
    Mllama(MllamaImageProcessorConfig),
    /// DETR image processor configuration.
    Detr(DetrImageProcessorConfig),
    /// SAM image processor configuration.
    Sam(SamImageProcessorConfig),
    /// Document/OCR image processor configuration.
    DocumentOcr(DocumentOcrImageProcessorConfig),
}

impl ProcessorFamilyConfig {
    /// Parses an HF `preprocessor_config.json` string into a strong Rust config.
    ///
    /// # Errors
    ///
    /// Returns an error when JSON is invalid, the processor family is missing
    /// or unsupported, or size fields are invalid.
    pub fn from_hf_preprocessor_json(input: &str) -> Result<Self, ProcessorConfigError> {
        let raw: HfPreprocessorConfig = serde_json::from_str(input)?;
        raw.into_family_config()
    }
}

/// Errors returned while parsing processor-family configurations.
#[derive(Debug, Error, PartialEq)]
pub enum ProcessorConfigError {
    /// The JSON document could not be parsed.
    #[error("invalid preprocessor config JSON: {0}")]
    Json(String),
    /// No processor family discriminator was present.
    #[error("preprocessor config is missing image_processor_type or processor_class")]
    MissingProcessorType,
    /// The processor family is not supported by this crate.
    #[error("unsupported image processor type {0}")]
    UnsupportedProcessorType(String),
    /// A size field was invalid or incomplete.
    #[error("invalid size field {field}")]
    InvalidSize {
        /// Size field name.
        field: &'static str,
    },
    /// Diffusers VAE RGB and grayscale conversion flags were both enabled.
    #[error("do_convert_rgb and do_convert_grayscale cannot both be true")]
    ConflictingPixelFormatConversion,
    /// A requested HDR transfer function is not supported.
    #[error("unsupported HDR transform {0}")]
    UnsupportedHdrTransform(String),
}

impl From<serde_json::Error> for ProcessorConfigError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

/// Configuration for CLIP-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ClipImageProcessorConfig {
    /// Resize target used before optional center cropping.
    pub size: ImageSize,
    /// Center-crop target used when `do_center_crop` is enabled.
    pub crop_size: ImageSize,
    /// Whether resized images should be center-cropped.
    pub do_center_crop: bool,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl Default for ClipImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: square_size(224),
            crop_size: square_size(224),
            do_center_crop: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl ClipImageProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the video output axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        video_layout_for_image_layout(self.output_layout)
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the generic image processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        fixed_image_processor_config(FixedImageProcessorSpec {
            image_size: self.target_size(),
            output_layout: self.output_layout,
            resize_mode: self.resize_mode(),
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &CLIP_IMAGE_MEAN,
            image_std: &CLIP_IMAGE_STD,
        })
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        fixed_image_processor_recipe(FixedImageProcessorRecipeSpec {
            id: "transformers.clip_image_processor",
            image_size: self.target_size(),
            output_layout: self.output_layout,
            resize_mode: self.resize_mode(),
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &CLIP_IMAGE_MEAN,
            image_std: &CLIP_IMAGE_STD,
        })
    }

    fn target_size(&self) -> ImageSize {
        if self.do_center_crop {
            self.crop_size
        } else {
            self.size
        }
    }

    fn resize_mode(&self) -> ResizeMode {
        if self.do_center_crop {
            ResizeMode::Crop
        } else {
            ResizeMode::Default
        }
    }
}

/// Configuration for ViT-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VitImageProcessorConfig {
    /// Resize target used before optional center cropping.
    pub size: ImageSize,
    /// Center-crop target used when `do_center_crop` is enabled.
    pub crop_size: ImageSize,
    /// Whether resized images should be center-cropped.
    pub do_center_crop: bool,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl Default for VitImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: square_size(224),
            crop_size: square_size(224),
            do_center_crop: false,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl VitImageProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the video output axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        video_layout_for_image_layout(self.output_layout)
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the generic image processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        fixed_image_processor_config(FixedImageProcessorSpec {
            image_size: self.target_size(),
            output_layout: self.output_layout,
            resize_mode: self.resize_mode(),
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &STANDARD_IMAGE_MEAN,
            image_std: &STANDARD_IMAGE_STD,
        })
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        fixed_image_processor_recipe(FixedImageProcessorRecipeSpec {
            id: "transformers.vit_image_processor",
            image_size: self.target_size(),
            output_layout: self.output_layout,
            resize_mode: self.resize_mode(),
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &STANDARD_IMAGE_MEAN,
            image_std: &STANDARD_IMAGE_STD,
        })
    }

    fn target_size(&self) -> ImageSize {
        if self.do_center_crop {
            self.crop_size
        } else {
            self.size
        }
    }

    fn resize_mode(&self) -> ResizeMode {
        if self.do_center_crop {
            ResizeMode::Crop
        } else {
            ResizeMode::Default
        }
    }
}

/// Configuration for document/OCR image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DocumentOcrImageProcessorConfig {
    /// Target image size.
    pub image_size: ImageSize,
    /// Whether images should be resized before thumbnailing and padding.
    pub do_resize: bool,
    /// Whether resized images should be constrained to `image_size` while
    /// preserving aspect ratio.
    pub do_thumbnail: bool,
    /// Whether the image should be rotated to align its long axis with
    /// `image_size`.
    pub do_align_long_axis: bool,
    /// Whether images should be center-padded to `image_size`.
    pub do_pad: bool,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied to pixel values when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether per-channel normalization should be applied.
    pub do_normalize: bool,
    /// Per-channel or scalar normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel or scalar normalization standard deviations.
    pub image_std: Vec<f32>,
}

impl Default for DocumentOcrImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize {
                height: 2560,
                width: 1920,
            },
            do_resize: true,
            do_thumbnail: true,
            do_align_long_axis: false,
            do_pad: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl DocumentOcrImageProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the video output axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        video_layout_for_image_layout(self.output_layout)
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the generic tensor-stage processor
    /// config used after document geometry has been applied.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: ResizeParity::Resampling,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable document/OCR processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions or
    /// output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(4);
        if self.do_resize || self.do_thumbnail || self.do_align_long_axis || self.do_pad {
            stages.push(ProcessorRecipeStage::DocumentGeometry {
                document: RecipeDocumentGeometryStage::new(
                    self.image_size,
                    self.do_resize,
                    self.do_thumbnail,
                    self.do_align_long_axis,
                    self.do_pad,
                ),
            });
        }
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        });
        if self.do_rescale {
            stages.push(ProcessorRecipeStage::Rescale {
                factor: self.rescale_factor,
            });
        }
        if self.do_normalize {
            stages.push(ProcessorRecipeStage::Normalize {
                mean: self.image_mean.clone(),
                std: self.image_std.clone(),
            });
        }

        ProcessorRecipe::new(
            "transformers.document_ocr_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}

macro_rules! fixed_family_processor {
    ($processor:ident, $config:ident) => {
        #[doc = concat!("Image processor for [`", stringify!($config), "`].")]
        #[derive(Clone, Debug)]
        pub struct $processor {
            config: $config,
            processor: ImageProcessor,
        }

        impl $processor {
            /// Creates a processor from a validated family configuration.
            ///
            /// # Errors
            ///
            /// Returns an error when the derived generic image processor config is invalid.
            pub fn new(config: $config) -> Result<Self, ImageProcessorError> {
                let processor = ImageProcessor::new(config.image_processor_config())?;
                Ok(Self { config, processor })
            }

            /// Returns this processor's family configuration.
            pub fn config(&self) -> &$config {
                &self.config
            }

            /// Returns the underlying generic image processor.
            pub fn image_processor(&self) -> &ImageProcessor {
                &self.processor
            }

            /// Loads and preprocesses media from a filesystem path.
            ///
            /// # Errors
            ///
            /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
            pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
                self.processor.open(path)
            }

            /// Loads and preprocesses a media source.
            ///
            /// # Errors
            ///
            /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
            pub fn open_source(&self, source: MediaSource) -> Result<Tensor, ImageProcessorError> {
                self.processor.open_source(source)
            }

            /// Loads image paths and preprocesses them as a batch tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
            pub fn open_batch<P: AsRef<Path> + Sync>(
                &self,
                paths: &[P],
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.open_batch(paths)
            }

            /// Loads image paths using reusable buffers and returns a batch tensor.
            ///
            /// The returned tensor owns its output buffer. Call
            /// [`ImageProcessorWorkspace::recycle_tensor`] after consuming the tensor
            /// to make that buffer available to the next workspace call.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
            pub fn open_batch_into<P: AsRef<Path> + Sync>(
                &self,
                paths: &[P],
                workspace: &mut ImageProcessorWorkspace,
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.open_batch_into(paths, workspace)
            }

            /// Preprocesses one decoded image frame.
            ///
            /// # Errors
            ///
            /// Returns an error when resizing, normalization, or tensor conversion fails.
            pub fn preprocess_image(
                &self,
                image: &ImageFrame,
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_image(image)
            }

            /// Preprocesses one decoded image frame with per-call options.
            ///
            /// # Errors
            ///
            /// Returns an error when resizing, normalization, or tensor conversion fails.
            pub fn preprocess_image_with_options(
                &self,
                image: &ImageFrame,
                options: ImageProcessorOptions,
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_image_with_options(image, options)
            }

            /// Preprocesses decoded image frames as a batch tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty or shapes are incompatible.
            pub fn preprocess_images(
                &self,
                images: &[ImageFrame],
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_images(images)
            }

            /// Preprocesses an image sequence as a frame tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the sequence is empty or frame preprocessing fails.
            pub fn preprocess_image_sequence(
                &self,
                sequence: &ImageSequence,
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_image_sequence(sequence)
            }

            /// Preprocesses a video clip as a frame tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the clip is empty or frame preprocessing fails.
            pub fn preprocess_video(
                &self,
                video: &VideoClip,
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_video(video)
            }

            /// Preprocesses decoded video clips as a batched video tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty, any clip is empty, or
            /// processed clip shapes are incompatible.
            pub fn preprocess_videos(
                &self,
                videos: &[VideoClip],
            ) -> Result<Tensor, ImageProcessorError> {
                self.processor.preprocess_videos(videos)
            }
        }
    };
}

fixed_family_processor!(ClipImageProcessor, ClipImageProcessorConfig);
fixed_family_processor!(VitImageProcessor, VitImageProcessorConfig);
fixed_family_processor!(DetrImageProcessor, DetrImageProcessorConfig);
fixed_family_processor!(SamImageProcessor, SamImageProcessorConfig);

impl ClipImageProcessor {
    /// Loads and preprocesses image paths directly into caller-owned `f16` storage.
    ///
    /// The returned view borrows `output` and carries validated batch shape,
    /// layout, dtype, and leading-axis metadata. The slice length must equal
    /// `batch * 3 * target_height * target_width`.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, the output length is invalid,
    /// loading fails, or processed image shapes are incompatible.
    pub fn open_batch_f16_into_slice<'a, P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        output: &'a mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        self.processor
            .open_batch_f16_into_slice(paths, output, workspace)
    }

    /// Loads image paths into an owned `F16` tensor using reusable workspace storage.
    ///
    /// Call [`ImageProcessorWorkspace::recycle_tensor`] after consuming the
    /// tensor to make its allocation available to the next call.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, output sizing overflows,
    /// loading fails, or processed image shapes are incompatible.
    pub fn open_batch_f16_into<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError> {
        if paths.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let target = self.config.target_size();
        let output_len = paths
            .len()
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(target.height))
            .and_then(|elements| elements.checked_mul(target.width))
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let mut values = workspace.take_f16_values(output_len);
        let view = self
            .processor
            .open_batch_f16_into_slice(paths, &mut values, workspace)?;
        let shape = view.shape().to_vec();
        let layout = view.layout();
        drop(view);
        Tensor::new(TensorData::F16(values), shape, layout)
            .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
            .map_err(ImageProcessorError::Tensor)
    }
}

/// Document/OCR image processor compatible with Donut-style preprocessing.
#[derive(Clone, Debug)]
pub struct DocumentOcrImageProcessor {
    config: DocumentOcrImageProcessorConfig,
    processor: ImageProcessor,
}

impl DocumentOcrImageProcessor {
    /// Creates a processor from a validated document/OCR configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor-stage image processor config is invalid.
    pub fn new(config: DocumentOcrImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &DocumentOcrImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses media from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, document geometry, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        self.open_source(MediaSource::path(path.as_ref()))
    }

    /// Loads and preprocesses a media source.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, document geometry, or tensor
    /// conversion fails.
    pub fn open_source(&self, source: MediaSource) -> Result<Tensor, ImageProcessorError> {
        match DefaultMediaLoader.load_with_remote_options_and_image_decode_backend(
            source,
            &Default::default(),
            self.config.decode_backend,
        )? {
            LoadedMedia::Image(image) => self.preprocess_image(&image),
            LoadedMedia::ImageSequence(sequence) => self.preprocess_image_sequence(&sequence),
            LoadedMedia::Video(video) => self.preprocess_video(&video),
        }
    }

    /// Loads image paths and preprocesses them as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, document
    /// geometry fails, or shapes are incompatible.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images(&images)
    }

    /// Loads image paths and preprocesses them as a batch tensor.
    ///
    /// The workspace is accepted for API symmetry with fixed-family
    /// processors. Document geometry materializes intermediate frames before
    /// tensor conversion.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, document
    /// geometry fails, or shapes are incompatible.
    pub fn open_batch_into<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError> {
        let _ = workspace;
        self.open_batch(paths)
    }

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when document geometry or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame with per-call size options.
    ///
    /// # Errors
    ///
    /// Returns an error when document geometry or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            std::slice::from_ref(image),
            options,
            TensorLeadingAxis::Batch,
        )
    }

    /// Preprocesses decoded image frames as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, document geometry fails, or
    /// shapes are incompatible.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_images_with_options(images, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded image frames as a batch tensor with per-call size
    /// options.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, document geometry fails, or
    /// shapes are incompatible.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(images, options, TensorLeadingAxis::Batch)
    }

    /// Preprocesses an image sequence as a frame tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, document geometry fails, or
    /// frame tensor conversion fails.
    pub fn preprocess_image_sequence(
        &self,
        sequence: &ImageSequence,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            sequence.frames(),
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Frames,
        )
    }

    /// Preprocesses a video clip as a frame tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, document geometry fails, or
    /// frame tensor conversion fails.
    pub fn preprocess_video(&self, video: &VideoClip) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            video.frames().iter().map(|frame| frame.image()),
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Frames,
        )
    }

    fn preprocess_image_batch_with_options<'a>(
        &self,
        images: impl IntoIterator<Item = &'a ImageFrame>,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Tensor, ImageProcessorError> {
        let prepared = images
            .into_iter()
            .map(|image| document_ocr_prepare_frame(&self.config, image, options))
            .collect::<Result<Vec<_>, _>>()?;
        if prepared.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        self.processor
            .preprocess_images(&prepared)?
            .with_leading_axis(leading_axis)
            .map_err(ImageProcessorError::Tensor)
    }
}

macro_rules! pixel_values_output_methods {
    ($processor:ident) => {
        impl $processor {
            /// Loads and preprocesses media as typed processor output.
            ///
            /// # Errors
            ///
            /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
            pub fn open_output(
                &self,
                path: impl AsRef<Path>,
            ) -> Result<ProcessorOutput, ImageProcessorError> {
                self.open(path).map(ProcessorOutput::from_pixel_values)
            }

            /// Loads image paths and preprocesses them as typed processor output.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
            pub fn open_batch_output<P: AsRef<Path> + Sync>(
                &self,
                paths: &[P],
            ) -> Result<ProcessorOutput, ImageProcessorError> {
                self.open_batch(paths)
                    .map(ProcessorOutput::from_pixel_values)
            }

            /// Preprocesses one decoded image frame as typed processor output.
            ///
            /// # Errors
            ///
            /// Returns an error when resizing, normalization, or tensor conversion fails.
            pub fn preprocess_image_output(
                &self,
                image: &ImageFrame,
            ) -> Result<ProcessorOutput, ImageProcessorError> {
                self.preprocess_image(image)
                    .map(ProcessorOutput::from_pixel_values)
            }

            /// Preprocesses decoded image frames as typed processor output.
            ///
            /// # Errors
            ///
            /// Returns an error when the batch is empty or shapes are incompatible.
            pub fn preprocess_images_output(
                &self,
                images: &[ImageFrame],
            ) -> Result<ProcessorOutput, ImageProcessorError> {
                self.preprocess_images(images)
                    .map(ProcessorOutput::from_pixel_values)
            }
        }
    };
}

pixel_values_output_methods!(ClipImageProcessor);
pixel_values_output_methods!(VitImageProcessor);
pixel_values_output_methods!(DocumentOcrImageProcessor);

fn image_processor_error_from_tensor_postprocess(
    error: TensorPostprocessError,
) -> ImageProcessorError {
    match error {
        TensorPostprocessError::UnsupportedLayout(layout) => {
            ImageProcessorError::UnsupportedLayout(layout)
        }
        TensorPostprocessError::Tensor(error) => ImageProcessorError::Tensor(error),
        TensorPostprocessError::Transform(error) => ImageProcessorError::Transform(error),
    }
}

fn resize_to_fit_size(source: ImageSize, target: ImageSize) -> ImageSize {
    let height_scale = target.height as f64 / source.height as f64;
    let width_scale = target.width as f64 / source.width as f64;
    let scale = height_scale.min(width_scale);
    let height = ((source.height as f64 * scale).round() as usize)
        .max(1)
        .min(target.height);
    let width = ((source.width as f64 * scale).round() as usize)
        .max(1)
        .min(target.width);
    ImageSize { height, width }
}

fn tensor_only_processor(
    mut config: ImageProcessorConfig,
) -> Result<ImageProcessor, ImageProcessorError> {
    config.do_resize = false;
    ImageProcessor::new(config)
}

#[derive(Clone, Copy, Debug)]
struct FixedImageProcessorSpec<'a> {
    image_size: ImageSize,
    output_layout: ImageLayout,
    resize_mode: ResizeMode,
    resample: ResizeFilter,
    resize_parity: ResizeParity,
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: &'a [f32],
    image_std: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
struct FixedImageProcessorRecipeSpec<'a> {
    id: &'static str,
    image_size: ImageSize,
    output_layout: ImageLayout,
    resize_mode: ResizeMode,
    resample: ResizeFilter,
    resize_parity: ResizeParity,
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: &'a [f32],
    image_std: &'a [f32],
}

struct ImageNormalizationRecipeSpec<'a> {
    id: &'static str,
    input: ProcessorRecipeInput,
    stages: Vec<ProcessorRecipeStage>,
    output_layout: ImageLayout,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: &'a [f32],
    image_std: &'a [f32],
}

fn fixed_image_processor_config(spec: FixedImageProcessorSpec<'_>) -> ImageProcessorConfig {
    ImageProcessorConfig {
        do_resize: true,
        height: Some(spec.image_size.height),
        width: Some(spec.image_size.width),
        resize_mode: spec.resize_mode,
        resample: spec.resample,
        resize_parity: spec.resize_parity,
        decode_backend: spec.decode_backend,
        batch_execution: spec.batch_execution,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: spec.do_rescale,
        rescale_factor: spec.rescale_factor,
        do_normalize: spec.do_normalize,
        image_mean: spec.image_mean.to_vec(),
        image_std: spec.image_std.to_vec(),
        do_binarize: false,
        output_layout: batched_layout_for_image_layout(spec.output_layout),
    }
}

fn fixed_image_processor_recipe(
    spec: FixedImageProcessorRecipeSpec<'_>,
) -> Result<ProcessorRecipe, RecipeError> {
    let mut stages = Vec::with_capacity(4);
    stages.push(ProcessorRecipeStage::ConvertPixelFormat {
        format: PixelFormat::Rgb8,
    });
    stages.push(ProcessorRecipeStage::Resize {
        resize: RecipeResizeStage::fixed(
            spec.image_size,
            spec.resize_mode,
            spec.resample,
            spec.resize_parity,
        ),
    });
    if spec.do_rescale {
        stages.push(ProcessorRecipeStage::Rescale {
            factor: spec.rescale_factor,
        });
    }
    if spec.do_normalize {
        stages.push(ProcessorRecipeStage::Normalize {
            mean: spec.image_mean.to_vec(),
            std: spec.image_std.to_vec(),
        });
    }

    ProcessorRecipe::new(
        spec.id,
        ProcessorRecipeInput {
            decode_backend: spec.decode_backend,
            batch_execution: spec.batch_execution,
        },
        stages,
        ProcessorRecipeOutput::batch(batched_layout_for_image_layout(spec.output_layout)),
    )
}

fn image_normalization_recipe(
    spec: ImageNormalizationRecipeSpec<'_>,
) -> Result<ProcessorRecipe, RecipeError> {
    let mut stages = spec.stages;
    if spec.do_rescale {
        stages.push(ProcessorRecipeStage::Rescale {
            factor: spec.rescale_factor,
        });
    }
    if spec.do_normalize {
        stages.push(ProcessorRecipeStage::Normalize {
            mean: spec.image_mean.to_vec(),
            std: spec.image_std.to_vec(),
        });
    }

    ProcessorRecipe::new(
        spec.id,
        spec.input,
        stages,
        ProcessorRecipeOutput::batch(batched_layout_for_image_layout(spec.output_layout)),
    )
}

fn resize_decision_for_parts(filter: ResizeFilter, parity: ResizeParity) -> ResizeDecision {
    match parity {
        ResizeParity::Resampling => ResizeDecision::resampling(filter),
        ResizeParity::Compatibility => ResizeDecision::compatibility(filter),
        ResizeParity::Torchvision => ResizeDecision::torchvision(filter),
        ResizeParity::PixelExact => ResizeDecision::pixel_exact(),
    }
}

fn resize_options_for_target(
    target: ImageSize,
    options: ImageProcessorOptions,
) -> ImageProcessorOptions {
    ImageProcessorOptions {
        height: Some(target.height),
        width: Some(target.width),
        resize_mode: Some(options.resize_mode.unwrap_or(ResizeMode::Default)),
    }
}

fn prepare_video_shortest_edge_frame(
    image: &ImageFrame,
    shortest_edge: usize,
    crop_size: ImageSize,
    do_resize: bool,
    do_center_crop: bool,
    resize_decision: ResizeDecision,
) -> Result<ImageFrame, ImageProcessorError> {
    let mut frame = Cow::Borrowed(image);

    if frame.pixel_format() != PixelFormat::Rgb8 {
        frame = Cow::Owned(convert_frame_pixel_format(
            frame.as_ref(),
            PixelFormat::Rgb8,
        )?);
    }

    if do_resize {
        let resize_size = video_shortest_edge_resize_size(frame.as_ref(), shortest_edge)?;
        if resize_size.height != frame.height() || resize_size.width != frame.width() {
            frame = Cow::Owned(resize_frame_with_decision(
                frame.as_ref(),
                resize_size,
                resize_decision,
                ResizeMode::Default,
            )?);
        }
    }

    if do_center_crop {
        frame = Cow::Owned(center_crop_frame(frame.as_ref(), crop_size)?);
    }

    Ok(frame.into_owned())
}

fn video_shortest_edge_resize_size(
    frame: &ImageFrame,
    shortest_edge: usize,
) -> Result<ImageSize, TransformError> {
    let height = frame.height();
    let width = frame.width();
    if width <= height {
        ImageSize::new(
            video_scaled_dimension(height, shortest_edge, width)?,
            shortest_edge,
        )
    } else {
        ImageSize::new(
            shortest_edge,
            video_scaled_dimension(width, shortest_edge, height)?,
        )
    }
}

fn video_scaled_dimension(
    value: usize,
    numerator: usize,
    denominator: usize,
) -> Result<usize, TransformError> {
    value
        .checked_mul(numerator)
        .ok_or(TransformError::ImageSizeOverflow)
        .map(|product| product / denominator)
}

fn document_ocr_prepare_frame(
    config: &DocumentOcrImageProcessorConfig,
    image: &ImageFrame,
    options: ImageProcessorOptions,
) -> Result<ImageFrame, ImageProcessorError> {
    let target = ImageSize::new(
        options.height.unwrap_or(config.image_size.height),
        options.width.unwrap_or(config.image_size.width),
    )?;
    let mut frame = Cow::Borrowed(image);

    if frame.pixel_format() != PixelFormat::Rgb8 {
        frame = Cow::Owned(convert_frame_pixel_format(
            frame.as_ref(),
            PixelFormat::Rgb8,
        )?);
    }

    if config.do_align_long_axis
        && should_rotate_to_match_orientation(
            ImageSize::new(frame.height(), frame.width())?,
            target,
        )?
    {
        frame = Cow::Owned(rotate_frame_90_clockwise(frame.as_ref())?);
    }

    if config.do_resize {
        let resize_size = shortest_edge_resize_size(
            ImageSize::new(frame.height(), frame.width())?,
            target.height.min(target.width),
            None,
        )?;
        if resize_size.height != frame.height() || resize_size.width != frame.width() {
            frame = Cow::Owned(resize_frame_with_decision(
                frame.as_ref(),
                resize_size,
                config.resize_decision(),
                ResizeMode::Default,
            )?);
        }
    }

    if config.do_thumbnail {
        let thumbnail_size =
            fit_inside_size(ImageSize::new(frame.height(), frame.width())?, target)?;
        if thumbnail_size.height != frame.height() || thumbnail_size.width != frame.width() {
            frame = Cow::Owned(resize_frame_with_decision(
                frame.as_ref(),
                thumbnail_size,
                config.resize_decision(),
                ResizeMode::Default,
            )?);
        }
    }

    if config.do_pad {
        let padding = centered_padding(ImageSize::new(frame.height(), frame.width())?, target)?;
        frame = Cow::Owned(pad_frame(frame.as_ref(), padding, &[0])?);
    }

    Ok(frame.into_owned())
}

fn rotate_frame_90_clockwise(frame: &ImageFrame) -> Result<ImageFrame, ImageProcessorError> {
    let channels = frame.channels();
    let source_width = frame.width();
    let source_height = frame.height();
    let mut output = vec![0; frame.data().len()];

    for y in 0..source_height {
        for x in 0..source_width {
            let source = (y * source_width + x) * channels;
            let destination = (x * source_height + (source_height - 1 - y)) * channels;
            output[destination..destination + channels]
                .copy_from_slice(&frame.data()[source..source + channels]);
        }
    }

    Ok(
        ImageFrame::new(source_height, source_width, frame.pixel_format(), output)?
            .with_timing(frame.timing()),
    )
}

fn square_size(size: usize) -> ImageSize {
    ImageSize {
        height: size,
        width: size,
    }
}

fn batched_layout_for_image_layout(layout: ImageLayout) -> Layout {
    match layout {
        ImageLayout::ChannelsHeightWidth => Layout::NCHW,
        ImageLayout::HeightWidthChannels => Layout::NHWC,
    }
}

fn patch_batched_layout_for_image_layout(layout: ImageLayout) -> Layout {
    match layout {
        ImageLayout::ChannelsHeightWidth => Layout::NPCHW,
        ImageLayout::HeightWidthChannels => Layout::NPHWC,
    }
}

fn tiled_batch_layout_for_image_layout(layout: ImageLayout) -> Layout {
    match layout {
        ImageLayout::ChannelsHeightWidth => Layout::NIPCHW,
        ImageLayout::HeightWidthChannels => Layout::NIPHWC,
    }
}

fn video_layout_for_image_layout(layout: ImageLayout) -> VideoLayout {
    match layout {
        ImageLayout::ChannelsHeightWidth => VideoLayout::FramesChannelsHeightWidth,
        ImageLayout::HeightWidthChannels => VideoLayout::FramesHeightWidthChannels,
    }
}

#[cfg(test)]
mod tests;
