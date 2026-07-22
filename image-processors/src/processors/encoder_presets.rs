//! Encoder/classifier presets and Swin2SR restoration preprocessing.
//!
//! The encoder wrapper groups upstream image processors by observable geometry
//! and tensor behavior. Class names select stable presets; they do not create a
//! parallel Rust type hierarchy for equivalent upstream Python classes.

use std::path::Path;

use thiserror::Error;

use super::{
    square_size, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD, IMAGENET_MEAN, IMAGENET_STD, STANDARD_IMAGE_MEAN,
    STANDARD_IMAGE_STD,
};
use crate::image::{BatchExecution, ImageProcessor, ImageProcessorConfig, ImageProcessorError};
use crate::media::{
    load_image_from_path_with_backend, ImageDecodeBackend, ImageFrame, MediaError, PixelFormat,
};
use crate::output::{ProcessorOutput, ProcessorTensorName};
use crate::postprocess::{post_process_image_tensor, TensorPostprocessError};
use crate::recipe::{
    ProcessorRecipe, ProcessorRecipeInput, ProcessorRecipeOutput, ProcessorRecipeStage,
    RecipeCropStage, RecipeError, RecipeResizeStage,
};
use crate::tensor::{ImageLayout, Layout, Tensor, TensorData, TensorError, TensorLeadingAxis};
use crate::transforms::{
    center_crop_frame, convert_frame_pixel_format, pad_frame, pad_frame_symmetric_to_next_multiple,
    resize_frame_to_f32_torchvision, resize_frame_with_decision, shortest_edge_resize_size,
    ImageSize, Padding, ResizeDecision, ResizeFilter, ResizeMode, ResizeParity, TransformError,
};

const RESCALE_FACTOR: f32 = 1.0 / 255.0;
const PPLCNET_MEAN: [f32; 3] = [0.406, 0.456, 0.485];
const PPLCNET_STD: [f32; 3] = [0.225, 0.224, 0.229];

/// Transformers encoder/classifier preset supported by the shared wrapper.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EncoderImageProcessorPreset {
    /// BEiT fixed-square preprocessing.
    Beit,
    /// BiT shortest-edge CLIP preprocessing.
    Bit,
    /// Chinese-CLIP shortest-edge CLIP preprocessing.
    ChineseClip,
    /// ConvNeXT crop-percentage preprocessing.
    ConvNext,
    /// DINOv3 ViT rescale-before-resize preprocessing.
    Dinov3Vit,
    /// DeiT resize-then-center-crop preprocessing.
    DeiT,
    /// EfficientNet preprocessing with the classification-head normalization.
    EfficientNet,
    /// FLAVA CLIP-normalized image preprocessing.
    Flava,
    /// ImageGPT preprocessing with optional color-cluster quantization.
    ImageGpt,
    /// LeViT scaled-shortest-edge preprocessing.
    Levit,
    /// MobileNetV1 shortest-edge preprocessing.
    MobileNetV1,
    /// MobileNetV2 shortest-edge preprocessing.
    MobileNetV2,
    /// MobileViT shortest-edge preprocessing with BGR output.
    MobileVit,
    /// Paddle PPLCNet divisor-aligned preprocessing with BGR output.
    PpLcNet,
    /// Perceiver crop-before-resize preprocessing.
    Perceiver,
    /// PoolFormer crop-percentage preprocessing.
    PoolFormer,
    /// PVT fixed-square preprocessing.
    Pvt,
    /// SigLIP fixed-square preprocessing.
    Siglip,
    /// A timm evaluation-transform preset.
    TimmWrapper,
}

impl EncoderImageProcessorPreset {
    /// Every encoder/classifier preset covered by this wrapper.
    pub const ALL: [Self; 19] = [
        Self::Beit,
        Self::Bit,
        Self::ChineseClip,
        Self::ConvNext,
        Self::Dinov3Vit,
        Self::DeiT,
        Self::EfficientNet,
        Self::Flava,
        Self::ImageGpt,
        Self::Levit,
        Self::MobileNetV1,
        Self::MobileNetV2,
        Self::MobileVit,
        Self::PpLcNet,
        Self::Perceiver,
        Self::PoolFormer,
        Self::Pvt,
        Self::Siglip,
        Self::TimmWrapper,
    ];

    /// Returns the canonical Transformers class name for this preset.
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Beit => "BeitImageProcessor",
            Self::Bit => "BitImageProcessor",
            Self::ChineseClip => "ChineseCLIPImageProcessor",
            Self::ConvNext => "ConvNextImageProcessor",
            Self::Dinov3Vit => "DINOv3ViTImageProcessor",
            Self::DeiT => "DeiTImageProcessor",
            Self::EfficientNet => "EfficientNetImageProcessor",
            Self::Flava => "FlavaImageProcessor",
            Self::ImageGpt => "ImageGPTImageProcessor",
            Self::Levit => "LevitImageProcessor",
            Self::MobileNetV1 => "MobileNetV1ImageProcessor",
            Self::MobileNetV2 => "MobileNetV2ImageProcessor",
            Self::MobileVit => "MobileViTImageProcessor",
            Self::PpLcNet => "PPLCNetImageProcessor",
            Self::Perceiver => "PerceiverImageProcessor",
            Self::PoolFormer => "PoolFormerImageProcessor",
            Self::Pvt => "PvtImageProcessor",
            Self::Siglip => "SiglipImageProcessor",
            Self::TimmWrapper => "TimmWrapperImageProcessor",
        }
    }

    /// Returns the PIL-backend aliases that share this preset.
    pub const fn class_aliases(self) -> &'static [&'static str] {
        match self {
            Self::Beit => &["BeitImageProcessorPil"],
            Self::Bit => &["BitImageProcessorPil"],
            Self::ChineseClip => &["ChineseCLIPImageProcessorPil"],
            Self::ConvNext => &["ConvNextImageProcessorPil"],
            Self::Dinov3Vit | Self::PpLcNet | Self::TimmWrapper => &[],
            Self::DeiT => &["DeiTImageProcessorPil"],
            Self::EfficientNet => &["EfficientNetImageProcessorPil"],
            Self::Flava => &["FlavaImageProcessorPil"],
            Self::ImageGpt => &["ImageGPTImageProcessorPil"],
            Self::Levit => &["LevitImageProcessorPil"],
            Self::MobileNetV1 => &["MobileNetV1ImageProcessorPil"],
            Self::MobileNetV2 => &["MobileNetV2ImageProcessorPil"],
            Self::MobileVit => &["MobileViTImageProcessorPil"],
            Self::Perceiver => &["PerceiverImageProcessorPil"],
            Self::PoolFormer => &["PoolFormerImageProcessorPil"],
            Self::Pvt => &["PvtImageProcessorPil"],
            Self::Siglip => &["SiglipImageProcessorPil"],
        }
    }

    /// Resolves a canonical class name or PIL alias into a preset.
    pub fn from_class_name(class_name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| {
            preset.class_name() == class_name || preset.class_aliases().contains(&class_name)
        })
    }

    /// Returns the stable recipe identifier for this preset.
    pub const fn recipe_id(self) -> &'static str {
        match self {
            Self::Beit => "transformers.beit_image_processor",
            Self::Bit => "transformers.bit_image_processor",
            Self::ChineseClip => "transformers.chinese_clip_image_processor",
            Self::ConvNext => "transformers.convnext_image_processor",
            Self::Dinov3Vit => "transformers.dinov3_vit_image_processor",
            Self::DeiT => "transformers.deit_image_processor",
            Self::EfficientNet => "transformers.efficientnet_image_processor",
            Self::Flava => "transformers.flava_image_processor",
            Self::ImageGpt => "transformers.imagegpt_image_processor",
            Self::Levit => "transformers.levit_image_processor",
            Self::MobileNetV1 => "transformers.mobilenet_v1_image_processor",
            Self::MobileNetV2 => "transformers.mobilenet_v2_image_processor",
            Self::MobileVit => "transformers.mobilevit_image_processor",
            Self::PpLcNet => "transformers.pp_lcnet_image_processor",
            Self::Perceiver => "transformers.perceiver_image_processor",
            Self::PoolFormer => "transformers.poolformer_image_processor",
            Self::Pvt => "transformers.pvt_image_processor",
            Self::Siglip => "transformers.siglip_image_processor",
            Self::TimmWrapper => "transformers.timm_wrapper_image_processor",
        }
    }
}

/// Spatial preprocessing shared by one or more encoder presets.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EncoderGeometry {
    /// Warp directly to a fixed height and width.
    Fixed {
        /// Fixed output dimensions.
        size: ImageSize,
    },
    /// Warp to a fixed size and then center-crop, padding with zeros if needed.
    FixedThenCenterCrop {
        /// Intermediate fixed dimensions.
        size: ImageSize,
        /// Final center-crop dimensions.
        crop_size: ImageSize,
    },
    /// Preserve aspect ratio by resizing the shortest edge, then center-crop.
    ShortestEdgeThenCenterCrop {
        /// Intermediate shortest-edge target.
        shortest_edge: usize,
        /// Final center-crop dimensions.
        crop_size: ImageSize,
    },
    /// Scale a requested shortest edge before resizing, then center-crop.
    ScaledShortestEdgeThenCenterCrop {
        /// User-facing shortest-edge size.
        shortest_edge: usize,
        /// Integer numerator applied before shortest-edge resize.
        scale_numerator: usize,
        /// Integer denominator applied before shortest-edge resize.
        scale_denominator: usize,
        /// Final center-crop dimensions.
        crop_size: ImageSize,
    },
    /// Resize by a crop percentage and center-crop to a square output.
    CropPercentage {
        /// Final square edge.
        output_edge: usize,
        /// Fraction of the resized shortest edge retained by the crop.
        crop_percentage: f64,
        /// Edge at which preprocessing changes to a direct square warp.
        warp_at_or_above: Option<usize>,
    },
    /// Crop a fraction of the source minimum dimension before a fixed resize.
    PerceiverCropThenResize {
        /// Final fixed dimensions.
        size: ImageSize,
        /// Reference dimensions used to calculate the source-relative crop.
        crop_reference: ImageSize,
    },
    /// Paddle-style rounded shortest-edge resize with divisor alignment.
    PaddleShortestEdgeThenCenterCrop {
        /// Requested shortest edge.
        shortest_edge: usize,
        /// Dimension alignment divisor.
        size_divisor: usize,
        /// Final center-crop dimensions.
        crop_size: ImageSize,
    },
}

/// Rust-owned configuration for the shared encoder/classifier wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderImageProcessorConfig {
    /// Upstream behavior preset.
    pub preset: EncoderImageProcessorPreset,
    /// Spatial preprocessing behavior.
    pub geometry: EncoderGeometry,
    /// Device-agnostic output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter.
    pub resample: ResizeFilter,
    /// Resize parity implementation.
    pub resize_parity: ResizeParity,
    /// Decode backend used by path entrypoints.
    pub decode_backend: ImageDecodeBackend,
    /// Batch execution policy used by tensor conversion.
    pub batch_execution: BatchExecution,
    /// Optional decoded-frame pixel conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Whether to rescale byte values.
    pub do_rescale: bool,
    /// Pixel rescale factor.
    pub rescale_factor: f32,
    /// Whether to apply channel normalization.
    pub do_normalize: bool,
    /// Per-channel normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviations.
    pub image_std: Vec<f32>,
    /// Whether the upstream recipe rescales before resizing.
    pub rescale_before_resize: bool,
    /// Optional second per-channel standard-deviation division.
    pub additional_std: Option<Vec<f32>>,
    /// Whether the final tensor reverses its first three channels.
    pub do_flip_channel_order: bool,
    /// Whether ImageGPT returns color-cluster indices instead of pixels.
    pub do_color_quantize: bool,
    /// ImageGPT color clusters in normalized RGB space.
    pub color_clusters: Option<Vec<[f32; 3]>>,
}

impl Default for EncoderImageProcessorConfig {
    fn default() -> Self {
        Self::for_preset(EncoderImageProcessorPreset::Beit)
    }
}

impl EncoderImageProcessorConfig {
    /// Creates the audited Transformers defaults for `preset`.
    pub fn for_preset(preset: EncoderImageProcessorPreset) -> Self {
        let mut config = Self {
            preset,
            geometry: EncoderGeometry::Fixed {
                size: square_size(224),
            },
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Torchvision,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: None,
            do_rescale: true,
            rescale_factor: RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
            rescale_before_resize: false,
            additional_std: None,
            do_flip_channel_order: false,
            do_color_quantize: false,
            color_clusters: None,
        };

        match preset {
            EncoderImageProcessorPreset::Beit => {}
            EncoderImageProcessorPreset::Bit | EncoderImageProcessorPreset::ChineseClip => {
                config.geometry = EncoderGeometry::ShortestEdgeThenCenterCrop {
                    shortest_edge: 224,
                    crop_size: square_size(224),
                };
                config.pixel_format = Some(PixelFormat::Rgb8);
                config.set_normalization(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD);
            }
            EncoderImageProcessorPreset::ConvNext => {
                config.geometry = EncoderGeometry::CropPercentage {
                    output_edge: 384,
                    crop_percentage: 224.0 / 256.0,
                    warp_at_or_above: Some(384),
                };
            }
            EncoderImageProcessorPreset::Dinov3Vit => {
                config.resample = ResizeFilter::Bilinear;
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
                config.rescale_before_resize = true;
            }
            EncoderImageProcessorPreset::DeiT => {
                config.geometry = EncoderGeometry::FixedThenCenterCrop {
                    size: square_size(256),
                    crop_size: square_size(224),
                };
            }
            EncoderImageProcessorPreset::EfficientNet => {
                config.geometry = EncoderGeometry::Fixed {
                    size: square_size(346),
                };
                config.additional_std = Some(STANDARD_IMAGE_STD.to_vec());
            }
            EncoderImageProcessorPreset::Flava => {
                config.geometry = EncoderGeometry::FixedThenCenterCrop {
                    size: square_size(224),
                    crop_size: square_size(224),
                };
                config.set_normalization(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD);
            }
            EncoderImageProcessorPreset::ImageGpt => {
                config.geometry = EncoderGeometry::Fixed {
                    size: square_size(256),
                };
                config.resample = ResizeFilter::Bilinear;
                config.do_color_quantize = true;
            }
            EncoderImageProcessorPreset::Levit => {
                config.geometry = EncoderGeometry::ScaledShortestEdgeThenCenterCrop {
                    shortest_edge: 224,
                    scale_numerator: 256,
                    scale_denominator: 224,
                    crop_size: square_size(224),
                };
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
            }
            EncoderImageProcessorPreset::MobileNetV1 | EncoderImageProcessorPreset::MobileNetV2 => {
                config.geometry = EncoderGeometry::ShortestEdgeThenCenterCrop {
                    shortest_edge: 256,
                    crop_size: square_size(224),
                };
                config.resample = ResizeFilter::Bilinear;
            }
            EncoderImageProcessorPreset::MobileVit => {
                config.geometry = EncoderGeometry::ShortestEdgeThenCenterCrop {
                    shortest_edge: 224,
                    crop_size: square_size(256),
                };
                config.do_normalize = false;
                config.do_flip_channel_order = true;
            }
            EncoderImageProcessorPreset::PpLcNet => {
                config.geometry = EncoderGeometry::PaddleShortestEdgeThenCenterCrop {
                    shortest_edge: 256,
                    size_divisor: 1,
                    crop_size: square_size(224),
                };
                config.resample = ResizeFilter::Bilinear;
                config.set_normalization(PPLCNET_MEAN, PPLCNET_STD);
                config.do_flip_channel_order = true;
            }
            EncoderImageProcessorPreset::Perceiver => {
                config.geometry = EncoderGeometry::PerceiverCropThenResize {
                    size: square_size(224),
                    crop_reference: square_size(256),
                };
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
            }
            EncoderImageProcessorPreset::PoolFormer => {
                config.geometry = EncoderGeometry::CropPercentage {
                    output_edge: 224,
                    crop_percentage: 0.9,
                    warp_at_or_above: None,
                };
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
            }
            EncoderImageProcessorPreset::Pvt => {
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
            }
            EncoderImageProcessorPreset::Siglip => {
                config.pixel_format = Some(PixelFormat::Rgb8);
            }
            EncoderImageProcessorPreset::TimmWrapper => {
                config.geometry = EncoderGeometry::CropPercentage {
                    output_edge: 224,
                    crop_percentage: 224.0 / 256.0,
                    warp_at_or_above: None,
                };
                config.pixel_format = Some(PixelFormat::Rgb8);
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
            }
        }

        config
    }

    fn set_normalization(&mut self, mean: [f32; 3], std: [f32; 3]) {
        self.image_mean = mean.to_vec();
        self.image_std = std.to_vec();
    }

    /// Creates audited defaults for a canonical class name or PIL alias.
    ///
    /// Canonical classes use the Torchvision tensor backend at the pinned
    /// Transformers revision. `*ImageProcessorPil` aliases select Pillow's
    /// fixed-point resize behavior.
    pub fn for_class_name(class_name: &str) -> Option<Self> {
        let preset = EncoderImageProcessorPreset::from_class_name(class_name)?;
        let mut config = Self::for_preset(preset);
        if preset.class_aliases().contains(&class_name) {
            config.resize_parity = ResizeParity::Compatibility;
        }
        Some(config)
    }

    /// Returns the output image axis order.
    pub const fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the explicit resize implementation selected by this config.
    pub fn resize_decision(&self) -> Result<ResizeDecision, TransformError> {
        ResizeDecision::new(self.resample, self.resize_parity)
    }

    /// Returns the main output name emitted by the configured preset.
    pub fn main_tensor_name(&self) -> ProcessorTensorName {
        if self.preset == EncoderImageProcessorPreset::ImageGpt && self.do_color_quantize {
            ProcessorTensorName::other("input_ids")
        } else {
            ProcessorTensorName::PixelValues
        }
    }

    /// Converts tensor-stage settings into the generic processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: self.pixel_format,
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout(self.output_layout),
        }
    }

    /// Builds a reusable recipe for the preset's ordered common stages.
    ///
    /// Channel reversal and ImageGPT color quantization are terminal wrapper
    /// operations because the generic recipe vocabulary has no equivalent
    /// tensor stage.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, normalization, or output layout is invalid.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(7);
        if let Some(format) = self.pixel_format {
            stages.push(ProcessorRecipeStage::ConvertPixelFormat { format });
        }

        if self.rescale_before_resize && self.do_rescale {
            stages.push(ProcessorRecipeStage::Rescale {
                factor: self.rescale_factor,
            });
        }
        stages.extend(
            self.geometry
                .recipe_stages(self.resample, self.resize_parity),
        );
        if !self.rescale_before_resize && self.do_rescale {
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
        if let Some(std) = &self.additional_std {
            stages.push(ProcessorRecipeStage::Normalize {
                mean: vec![0.0; std.len()],
                std: std.clone(),
            });
        }

        ProcessorRecipe::new(
            self.preset.recipe_id(),
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout(self.output_layout)),
        )
    }
}

impl EncoderGeometry {
    fn recipe_stages(
        self,
        filter: ResizeFilter,
        parity: ResizeParity,
    ) -> Vec<ProcessorRecipeStage> {
        let resize = |resize| ProcessorRecipeStage::Resize { resize };
        let crop = |size| ProcessorRecipeStage::Crop {
            crop: RecipeCropStage::center(size),
        };
        match self {
            Self::Fixed { size } | Self::PerceiverCropThenResize { size, .. } => {
                vec![resize(RecipeResizeStage::fixed(
                    size,
                    ResizeMode::Default,
                    filter,
                    parity,
                ))]
            }
            Self::FixedThenCenterCrop { size, crop_size } => vec![
                resize(RecipeResizeStage::fixed(
                    size,
                    ResizeMode::Default,
                    filter,
                    parity,
                )),
                crop(crop_size),
            ],
            Self::ShortestEdgeThenCenterCrop {
                shortest_edge,
                crop_size,
            }
            | Self::PaddleShortestEdgeThenCenterCrop {
                shortest_edge,
                crop_size,
                ..
            } => vec![
                resize(RecipeResizeStage::shortest_edge(
                    shortest_edge,
                    None,
                    filter,
                    parity,
                )),
                crop(crop_size),
            ],
            Self::ScaledShortestEdgeThenCenterCrop {
                shortest_edge,
                scale_numerator,
                scale_denominator,
                crop_size,
            } => {
                let scaled = shortest_edge
                    .checked_mul(scale_numerator)
                    .and_then(|value| value.checked_div(scale_denominator))
                    .unwrap_or(0);
                vec![
                    resize(RecipeResizeStage::shortest_edge(
                        scaled, None, filter, parity,
                    )),
                    crop(crop_size),
                ]
            }
            Self::CropPercentage {
                output_edge,
                crop_percentage: _,
                warp_at_or_above,
            } if warp_at_or_above.is_some_and(|threshold| output_edge >= threshold) => {
                let size = square_size(output_edge);
                vec![resize(RecipeResizeStage::fixed(
                    size,
                    ResizeMode::Default,
                    filter,
                    parity,
                ))]
            }
            Self::CropPercentage {
                output_edge,
                crop_percentage,
                ..
            } => {
                let resized_edge = (output_edge as f64 / crop_percentage) as usize;
                vec![
                    resize(RecipeResizeStage::shortest_edge(
                        resized_edge,
                        None,
                        filter,
                        parity,
                    )),
                    crop(square_size(output_edge)),
                ]
            }
        }
    }
}

/// Errors returned by encoder-preset and restoration wrappers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PresetProcessorError {
    /// Generic image preprocessing failed.
    #[error(transparent)]
    Image(#[from] ImageProcessorError),
    /// Decoded media loading failed.
    #[error(transparent)]
    Media(#[from] MediaError),
    /// Spatial transformation failed.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Tensor-to-image restoration failed.
    #[error(transparent)]
    Postprocess(#[from] TensorPostprocessError),
    /// ImageGPT quantization was enabled without checkpoint color clusters.
    #[error("ImageGPT color quantization requires at least one RGB cluster")]
    MissingColorClusters,
    /// A color cluster contained a non-finite component.
    #[error("ImageGPT color cluster {cluster} component {component} must be finite")]
    NonFiniteColorCluster {
        /// Cluster index.
        cluster: usize,
        /// RGB component index.
        component: usize,
    },
    /// A crop percentage was zero, negative, or non-finite.
    #[error("crop_percentage must be finite and positive, got {0}")]
    InvalidCropPercentage(f64),
    /// A configured divisor was zero.
    #[error("size divisor must be positive")]
    InvalidSizeDivisor,
    /// A class name did not identify a supported encoder preset or alias.
    #[error("unsupported encoder image-processor class `{0}`")]
    UnsupportedClassName(String),
    /// Rescale-before-resize was requested for unsupported spatial geometry.
    #[error("rescale-before-resize requires fixed resize geometry")]
    UnsupportedRescaleBeforeResizeGeometry,
    /// Rescale-before-resize was requested without the Torchvision backend.
    #[error("rescale-before-resize requires Torchvision resize parity")]
    UnsupportedRescaleBeforeResizeBackend,
}

/// Shared concrete processor for Transformers encoder/classifier presets.
#[derive(Clone, Debug)]
pub struct EncoderImageProcessor {
    config: EncoderImageProcessorConfig,
    tensor_processor: ImageProcessor,
}

impl EncoderImageProcessor {
    /// Creates a processor from a Rust-owned preset configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when tensor-stage normalization or layout is invalid.
    pub fn new(config: EncoderImageProcessorConfig) -> Result<Self, PresetProcessorError> {
        validate_color_clusters(config.color_clusters.as_deref())?;
        let tensor_processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self {
            config,
            tensor_processor,
        })
    }

    /// Creates a processor from audited defaults for `preset`.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived tensor configuration is invalid.
    pub fn for_preset(preset: EncoderImageProcessorPreset) -> Result<Self, PresetProcessorError> {
        Self::new(EncoderImageProcessorConfig::for_preset(preset))
    }

    /// Creates a processor for a canonical Transformers class or PIL alias.
    ///
    /// # Errors
    ///
    /// Returns an error when `class_name` is unsupported or its derived
    /// configuration is invalid.
    pub fn for_class_name(class_name: &str) -> Result<Self, PresetProcessorError> {
        let config = EncoderImageProcessorConfig::for_class_name(class_name)
            .ok_or_else(|| PresetProcessorError::UnsupportedClassName(class_name.to_owned()))?;
        Self::new(config)
    }

    /// Returns this processor's Rust-owned configuration.
    pub fn config(&self) -> &EncoderImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.tensor_processor
    }

    /// Applies pixel conversion and spatial preprocessing without tensor normalization.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion, resizing, padding, or cropping fails.
    pub fn prepare_image(&self, image: &ImageFrame) -> Result<ImageFrame, PresetProcessorError> {
        let converted;
        let image = if let Some(format) = self.config.pixel_format {
            if image.pixel_format() != format {
                converted = convert_frame_pixel_format(image, format)?;
                &converted
            } else {
                image
            }
        } else {
            image
        };
        prepare_encoder_geometry(image, self.config.geometry, self.config.resize_decision()?)
    }

    /// Returns the full spatially processed tensor before normalization.
    ///
    /// The stage tensor is byte-valued for native Transformers processors except
    /// DINOv3, whose upstream pipeline rescales before interpolation and therefore
    /// emits the resized unit-range `f32` tensor here. Timm's evaluation pipeline
    /// also emits unit-range `f32` values at this point.
    /// Family channel reversal is included because MobileViT and PPLCNet apply
    /// it after geometry even when rescaling and normalization are disabled.
    ///
    /// # Errors
    ///
    /// Returns an error when image preparation or tensor construction fails.
    pub fn preprocess_image_stage(
        &self,
        image: &ImageFrame,
    ) -> Result<Tensor, PresetProcessorError> {
        if self.config.rescale_before_resize && self.config.do_rescale {
            return self.preprocess_rescale_before_resize(image, false);
        }
        let prepared = self.prepare_image(image)?;
        frame_stage_tensor(
            &prepared,
            self.config.output_layout,
            self.config.do_flip_channel_order,
            self.config.preset == EncoderImageProcessorPreset::TimmWrapper,
        )
    }

    /// Loads and preprocesses one filesystem image.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, geometry, tensor conversion, or quantization fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, PresetProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Preprocesses one decoded image into the preset's main tensor.
    ///
    /// ImageGPT returns `input_ids` with `NC` layout when color quantization is
    /// enabled. Other presets return `pixel_values` in `NCHW` or `NHWC` layout.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, tensor conversion, or quantization fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, PresetProcessorError> {
        if self.config.rescale_before_resize && self.config.do_rescale {
            return self.preprocess_rescale_before_resize(image, true);
        }
        let prepared = self.prepare_image(image)?;
        let tensor = self.tensor_processor.preprocess_image(&prepared)?;
        finish_encoder_tensor(&self.config, tensor)
    }

    /// Preprocesses decoded images as one batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or incompatible batches, failed transforms,
    /// tensor conversion, or ImageGPT quantization without clusters.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, PresetProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch.into());
        }
        if self.config.rescale_before_resize && self.config.do_rescale {
            let tensors = images
                .iter()
                .map(|image| self.preprocess_rescale_before_resize(image, true))
                .collect::<Result<Vec<_>, _>>()?;
            return stack_encoder_batch(tensors);
        }
        let prepared: Result<Vec<_>, _> = images
            .iter()
            .map(|image| self.prepare_image(image))
            .collect();
        let tensor = self.tensor_processor.preprocess_images(&prepared?)?;
        finish_encoder_tensor(&self.config, tensor)
    }

    /// Preprocesses one decoded image into a named processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing the main tensor fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, PresetProcessorError> {
        let tensor = self.preprocess_image(image)?;
        let mut output = ProcessorOutput::new();
        output.insert_tensor(self.config.main_tensor_name(), tensor);
        Ok(output)
    }

    fn preprocess_rescale_before_resize(
        &self,
        image: &ImageFrame,
        normalize: bool,
    ) -> Result<Tensor, PresetProcessorError> {
        if self.config.resize_parity != ResizeParity::Torchvision {
            return Err(PresetProcessorError::UnsupportedRescaleBeforeResizeBackend);
        }
        let EncoderGeometry::Fixed { size: target } = self.config.geometry else {
            return Err(PresetProcessorError::UnsupportedRescaleBeforeResizeGeometry);
        };

        let converted;
        let image = if let Some(format) = self.config.pixel_format {
            if image.pixel_format() != format {
                converted = convert_frame_pixel_format(image, format)?;
                &converted
            } else {
                image
            }
        } else {
            image
        };
        let mut values = resize_frame_to_f32_torchvision(
            image,
            target,
            self.config.resample,
            self.config.rescale_factor,
        )?;
        if normalize && self.config.do_normalize {
            normalize_interleaved_f32(
                &mut values,
                image.channels(),
                &self.config.image_mean,
                &self.config.image_std,
            )?;
        }
        let tensor =
            interleaved_f32_tensor(values, target, image.channels(), self.config.output_layout)?;
        finish_encoder_tensor(&self.config, tensor)
    }
}

/// Rust-owned Swin2SR preprocessing configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Swin2SrImageProcessorConfig {
    /// Divisor used by symmetric bottom/right padding.
    pub size_divisor: usize,
    /// Device-agnostic output axis order.
    pub output_layout: ImageLayout,
    /// Decode backend used by path entrypoints.
    pub decode_backend: ImageDecodeBackend,
    /// Batch execution policy used by tensor conversion.
    pub batch_execution: BatchExecution,
    /// Whether to rescale byte values into the unit interval.
    pub do_rescale: bool,
    /// Pixel rescale factor.
    pub rescale_factor: f32,
}

impl Default for Swin2SrImageProcessorConfig {
    fn default() -> Self {
        Self {
            size_divisor: 8,
            output_layout: ImageLayout::ChannelsHeightWidth,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            do_rescale: true,
            rescale_factor: RESCALE_FACTOR,
        }
    }
}

impl Swin2SrImageProcessorConfig {
    /// Stable recipe identifier used by the processor catalog.
    pub const RECIPE_ID: &'static str = "transformers.swin2sr_image_processor";

    /// Converts tensor-stage settings into the generic processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: None,
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: false,
            image_mean: Vec::new(),
            image_std: Vec::new(),
            do_binarize: false,
            output_layout: batched_layout(self.output_layout),
        }
    }

    /// Builds the exact Swin2SR preprocessing recipe descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when the divisor, rescale factor, or layout is invalid.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = vec![ProcessorRecipeStage::PadSymmetricToNextMultiple {
            multiples: square_size(self.size_divisor),
        }];
        if self.do_rescale {
            stages.push(ProcessorRecipeStage::Rescale {
                factor: self.rescale_factor,
            });
        }
        ProcessorRecipe::new(
            Self::RECIPE_ID,
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout(self.output_layout)),
        )
    }
}

/// Concrete image-restoration processor for Swin2SR.
#[derive(Clone, Debug)]
pub struct Swin2SrImageProcessor {
    config: Swin2SrImageProcessorConfig,
    tensor_processor: ImageProcessor,
}

impl Swin2SrImageProcessor {
    /// Creates a Swin2SR processor from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the divisor or tensor-stage config is invalid.
    pub fn new(config: Swin2SrImageProcessorConfig) -> Result<Self, PresetProcessorError> {
        if config.size_divisor == 0 {
            return Err(PresetProcessorError::InvalidSizeDivisor);
        }
        let tensor_processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self {
            config,
            tensor_processor,
        })
    }

    /// Returns this processor's restoration configuration.
    pub fn config(&self) -> &Swin2SrImageProcessorConfig {
        &self.config
    }

    /// Applies upstream-compatible symmetric bottom/right padding.
    ///
    /// Even dimensions already divisible by `size_divisor` advance by one full
    /// divisor, matching the audited Transformers implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when output dimensions overflow or frame validation fails.
    pub fn prepare_image(&self, image: &ImageFrame) -> Result<ImageFrame, PresetProcessorError> {
        Ok(pad_frame_symmetric_to_next_multiple(
            image,
            square_size(self.config.size_divisor),
        )?)
    }

    /// Returns the full symmetrically padded byte tensor before rescaling.
    ///
    /// # Errors
    ///
    /// Returns an error when padding or tensor construction fails.
    pub fn preprocess_image_stage(
        &self,
        image: &ImageFrame,
    ) -> Result<Tensor, PresetProcessorError> {
        let prepared = self.prepare_image(image)?;
        frame_stage_tensor(&prepared, self.config.output_layout, false, false)
    }

    /// Loads and preprocesses one filesystem image.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, symmetric padding, or tensor conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, PresetProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Preprocesses one decoded restoration input.
    ///
    /// # Errors
    ///
    /// Returns an error when symmetric padding or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, PresetProcessorError> {
        let prepared = self.prepare_image(image)?;
        Ok(self.tensor_processor.preprocess_image(&prepared)?)
    }

    /// Preprocesses decoded restoration inputs as a batch.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or incompatible batches, failed padding, or
    /// tensor conversion.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, PresetProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch.into());
        }
        let prepared: Result<Vec<_>, _> = images
            .iter()
            .map(|image| self.prepare_image(image))
            .collect();
        Ok(self.tensor_processor.preprocess_images(&prepared?)?)
    }

    /// Preprocesses one restoration input as named `pixel_values` output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, PresetProcessorError> {
        Ok(ProcessorOutput::from_pixel_values(
            self.preprocess_image(image)?,
        ))
    }

    /// Restores unit-range model output tensors to clamped owned image frames.
    ///
    /// Values below zero or above one are clamped before conversion to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported layouts, channel counts, or shapes.
    pub fn postprocess_image_tensor(
        &self,
        tensor: &Tensor,
    ) -> Result<Vec<ImageFrame>, PresetProcessorError> {
        Ok(post_process_image_tensor(tensor)?)
    }
}

fn prepare_encoder_geometry(
    image: &ImageFrame,
    geometry: EncoderGeometry,
    decision: ResizeDecision,
) -> Result<ImageFrame, PresetProcessorError> {
    match geometry {
        EncoderGeometry::Fixed { size } => Ok(resize_frame_with_decision(
            image,
            size,
            decision,
            ResizeMode::Default,
        )?),
        EncoderGeometry::FixedThenCenterCrop { size, crop_size } => {
            let resized = resize_frame_with_decision(image, size, decision, ResizeMode::Default)?;
            center_crop_with_padding(&resized, crop_size)
        }
        EncoderGeometry::ShortestEdgeThenCenterCrop {
            shortest_edge,
            crop_size,
        } => {
            let resized = resize_shortest_edge(image, shortest_edge, decision)?;
            center_crop_with_padding(&resized, crop_size)
        }
        EncoderGeometry::ScaledShortestEdgeThenCenterCrop {
            shortest_edge,
            scale_numerator,
            scale_denominator,
            crop_size,
        } => {
            if scale_denominator == 0 {
                return Err(TransformError::InvalidScaleFactor(scale_denominator).into());
            }
            let scaled = shortest_edge
                .checked_mul(scale_numerator)
                .ok_or(TransformError::ImageSizeOverflow)?
                / scale_denominator;
            let resized = resize_shortest_edge(image, scaled, decision)?;
            center_crop_with_padding(&resized, crop_size)
        }
        EncoderGeometry::CropPercentage {
            output_edge,
            crop_percentage,
            warp_at_or_above,
        } => {
            if !crop_percentage.is_finite() || crop_percentage <= 0.0 {
                return Err(PresetProcessorError::InvalidCropPercentage(crop_percentage));
            }
            let output = square_size(output_edge);
            if warp_at_or_above.is_some_and(|threshold| output_edge >= threshold) {
                Ok(resize_frame_with_decision(
                    image,
                    output,
                    decision,
                    ResizeMode::Default,
                )?)
            } else {
                let resized_edge = (output_edge as f64 / crop_percentage) as usize;
                let resized = resize_shortest_edge(image, resized_edge, decision)?;
                center_crop_with_padding(&resized, output)
            }
        }
        EncoderGeometry::PerceiverCropThenResize {
            size,
            crop_reference,
        } => {
            let min_dimension = image.height().min(image.width());
            let crop_size = ImageSize::new(
                checked_ratio(size.height, min_dimension, crop_reference.height)?,
                checked_ratio(size.width, min_dimension, crop_reference.width)?,
            )?;
            let cropped = center_crop_with_padding(image, crop_size)?;
            Ok(resize_frame_with_decision(
                &cropped,
                size,
                decision,
                ResizeMode::Default,
            )?)
        }
        EncoderGeometry::PaddleShortestEdgeThenCenterCrop {
            shortest_edge,
            size_divisor,
            crop_size,
        } => {
            if size_divisor == 0 {
                return Err(PresetProcessorError::InvalidSizeDivisor);
            }
            let scale = shortest_edge as f64 / image.height().min(image.width()) as f64;
            let height = align_up(
                (image.height() as f64 * scale).round() as usize,
                size_divisor,
            )?;
            let width = align_up(
                (image.width() as f64 * scale).round() as usize,
                size_divisor,
            )?;
            let resized = resize_frame_with_decision(
                image,
                ImageSize::new(height, width)?,
                decision,
                ResizeMode::Default,
            )?;
            center_crop_with_padding(&resized, crop_size)
        }
    }
}

fn resize_shortest_edge(
    image: &ImageFrame,
    shortest_edge: usize,
    decision: ResizeDecision,
) -> Result<ImageFrame, PresetProcessorError> {
    let size = shortest_edge_resize_size(
        ImageSize::new(image.height(), image.width())?,
        shortest_edge,
        None,
    )?;
    Ok(resize_frame_with_decision(
        image,
        size,
        decision,
        ResizeMode::Default,
    )?)
}

fn center_crop_with_padding(
    image: &ImageFrame,
    crop_size: ImageSize,
) -> Result<ImageFrame, PresetProcessorError> {
    let missing_height = crop_size.height.saturating_sub(image.height());
    let missing_width = crop_size.width.saturating_sub(image.width());
    let padded = if missing_height == 0 && missing_width == 0 {
        None
    } else {
        let top = missing_height / 2;
        let left = missing_width / 2;
        Some(pad_frame(
            image,
            Padding::new(top, missing_width - left, missing_height - top, left),
            &[0],
        )?)
    };
    Ok(center_crop_frame(
        padded.as_ref().unwrap_or(image),
        crop_size,
    )?)
}

fn normalize_interleaved_f32(
    values: &mut [f32],
    channels: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<(), PresetProcessorError> {
    for stats in [mean, std] {
        if stats.len() != 1 && stats.len() != channels {
            return Err(ImageProcessorError::InvalidNormalizationStats {
                channels,
                actual: stats.len(),
            }
            .into());
        }
    }
    for pixel in values.chunks_exact_mut(channels) {
        for (channel, value) in pixel.iter_mut().enumerate() {
            let mean = mean[if mean.len() == 1 { 0 } else { channel }];
            let std = std[if std.len() == 1 { 0 } else { channel }];
            *value = (*value - mean) / std;
        }
    }
    Ok(())
}

fn interleaved_f32_tensor(
    values: Vec<f32>,
    size: ImageSize,
    channels: usize,
    layout: ImageLayout,
) -> Result<Tensor, PresetProcessorError> {
    let (values, shape, tensor_layout) = match layout {
        ImageLayout::ChannelsHeightWidth => {
            let mut planar = Vec::with_capacity(values.len());
            for channel in 0..channels {
                planar.extend(values.chunks_exact(channels).map(|pixel| pixel[channel]));
            }
            (
                planar,
                vec![1, channels, size.height, size.width],
                Layout::NCHW,
            )
        }
        ImageLayout::HeightWidthChannels => (
            values,
            vec![1, size.height, size.width, channels],
            Layout::NHWC,
        ),
    };
    Ok(Tensor::new(TensorData::F32(values), shape, tensor_layout)?
        .with_leading_axis(TensorLeadingAxis::Batch)?)
}

fn stack_encoder_batch(tensors: Vec<Tensor>) -> Result<Tensor, PresetProcessorError> {
    let first = tensors.first().ok_or(ImageProcessorError::EmptyBatch)?;
    let mut shape = first.shape().to_vec();
    let layout = first.layout();
    let sample_shape = shape[1..].to_vec();
    let sample_values = sample_shape.iter().product::<usize>();
    let mut values = Vec::with_capacity(
        tensors
            .len()
            .checked_mul(sample_values)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    for tensor in tensors {
        if tensor.layout() != layout
            || tensor.leading_axis() != Some(TensorLeadingAxis::Batch)
            || tensor.shape().first() != Some(&1)
            || tensor.shape()[1..] != sample_shape
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes.into());
        }
        match tensor.data() {
            TensorData::F32(data) => values.extend_from_slice(data),
            _ => return Err(ImageProcessorError::ExpectedDataType("F32").into()),
        }
    }
    shape[0] = values.len() / sample_values;
    Ok(Tensor::new(TensorData::F32(values), shape, layout)?
        .with_leading_axis(TensorLeadingAxis::Batch)?)
}

fn finish_encoder_tensor(
    config: &EncoderImageProcessorConfig,
    tensor: Tensor,
) -> Result<Tensor, PresetProcessorError> {
    let (data, shape, layout, leading_axis) = tensor.into_parts();
    match data {
        TensorData::F32(mut values) => {
            if let Some(std) = &config.additional_std {
                divide_channels(&mut values, &shape, layout, std)?;
            }
            if config.do_flip_channel_order {
                flip_first_three_channels(&mut values, &shape, layout)?;
            }
            if config.preset == EncoderImageProcessorPreset::ImageGpt && config.do_color_quantize {
                return quantize_imagegpt(
                    &values,
                    &shape,
                    layout,
                    config.color_clusters.as_deref(),
                );
            }
            rebuild_tensor(TensorData::F32(values), shape, layout, leading_axis)
        }
        TensorData::U8(mut values)
            if config.additional_std.is_none() && !config.do_color_quantize =>
        {
            if config.do_flip_channel_order {
                flip_first_three_channels(&mut values, &shape, layout)?;
            }
            rebuild_tensor(TensorData::U8(values), shape, layout, leading_axis)
        }
        _ => Err(ImageProcessorError::ExpectedDataType("float32 or uint8").into()),
    }
}

fn divide_channels(
    values: &mut [f32],
    shape: &[usize],
    layout: Layout,
    std: &[f32],
) -> Result<(), PresetProcessorError> {
    let (batch, channels, height, width) = image_batch_shape(shape, layout)?;
    if std.len() != 1 && std.len() != channels {
        return Err(ImageProcessorError::InvalidNormalizationStats {
            channels,
            actual: std.len(),
        }
        .into());
    }
    for sample in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let index = image_value_index(sample, channel, y, x, shape, layout);
                    values[index] /= std[if std.len() == 1 { 0 } else { channel }];
                }
            }
        }
    }
    Ok(())
}

fn flip_first_three_channels<T>(
    values: &mut [T],
    shape: &[usize],
    layout: Layout,
) -> Result<(), PresetProcessorError> {
    let (batch, channels, height, width) = image_batch_shape(shape, layout)?;
    if channels < 3 {
        return Err(ImageProcessorError::InvalidNormalizationStats {
            channels: 3,
            actual: channels,
        }
        .into());
    }
    for sample in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let red = image_value_index(sample, 0, y, x, shape, layout);
                let blue = image_value_index(sample, 2, y, x, shape, layout);
                values.swap(red, blue);
            }
        }
    }
    Ok(())
}

fn quantize_imagegpt(
    values: &[f32],
    shape: &[usize],
    layout: Layout,
    clusters: Option<&[[f32; 3]]>,
) -> Result<Tensor, PresetProcessorError> {
    let clusters = clusters
        .filter(|clusters| !clusters.is_empty())
        .ok_or(PresetProcessorError::MissingColorClusters)?;
    let (batch, channels, height, width) = image_batch_shape(shape, layout)?;
    if channels != 3 {
        return Err(ImageProcessorError::InvalidNormalizationStats {
            channels: 3,
            actual: channels,
        }
        .into());
    }
    let pixels = height
        .checked_mul(width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut input_ids = Vec::with_capacity(
        batch
            .checked_mul(pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    for sample in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let rgb = [
                    values[image_value_index(sample, 0, y, x, shape, layout)],
                    values[image_value_index(sample, 1, y, x, shape, layout)],
                    values[image_value_index(sample, 2, y, x, shape, layout)],
                ];
                let (nearest, _) = clusters
                    .iter()
                    .enumerate()
                    .map(|(index, cluster)| {
                        let distance = rgb
                            .iter()
                            .zip(cluster)
                            .map(|(value, center)| (value - center).powi(2))
                            .sum::<f32>();
                        (index, distance)
                    })
                    .min_by(|left, right| left.1.total_cmp(&right.1))
                    .ok_or(PresetProcessorError::MissingColorClusters)?;
                input_ids.push(nearest as i64);
            }
        }
    }
    Ok(Tensor::new(
        TensorData::I64(input_ids),
        [batch, pixels],
        Layout::NC,
    )?)
}

fn image_batch_shape(
    shape: &[usize],
    layout: Layout,
) -> Result<(usize, usize, usize, usize), PresetProcessorError> {
    match (layout, shape) {
        (Layout::NCHW, [batch, channels, height, width])
        | (Layout::NHWC, [batch, height, width, channels]) => {
            Ok((*batch, *channels, *height, *width))
        }
        _ => Err(ImageProcessorError::UnsupportedLayout(layout).into()),
    }
}

fn image_value_index(
    sample: usize,
    channel: usize,
    y: usize,
    x: usize,
    shape: &[usize],
    layout: Layout,
) -> usize {
    match layout {
        Layout::NCHW => ((sample * shape[1] + channel) * shape[2] + y) * shape[3] + x,
        Layout::NHWC => ((sample * shape[1] + y) * shape[2] + x) * shape[3] + channel,
        _ => 0,
    }
}

fn rebuild_tensor(
    data: TensorData,
    shape: Vec<usize>,
    layout: Layout,
    leading_axis: Option<TensorLeadingAxis>,
) -> Result<Tensor, PresetProcessorError> {
    let tensor = Tensor::new(data, shape, layout)?;
    match leading_axis {
        Some(axis) => Ok(tensor.with_leading_axis(axis)?),
        None => Ok(tensor),
    }
}

fn validate_color_clusters(clusters: Option<&[[f32; 3]]>) -> Result<(), PresetProcessorError> {
    for (cluster_index, cluster) in clusters.unwrap_or_default().iter().enumerate() {
        for (component_index, component) in cluster.iter().enumerate() {
            if !component.is_finite() {
                return Err(PresetProcessorError::NonFiniteColorCluster {
                    cluster: cluster_index,
                    component: component_index,
                });
            }
        }
    }
    Ok(())
}

fn frame_stage_tensor(
    image: &ImageFrame,
    layout: ImageLayout,
    flip_channel_order: bool,
    unit_range: bool,
) -> Result<Tensor, PresetProcessorError> {
    let batch_layout = batched_layout(layout);
    let shape = match layout {
        ImageLayout::ChannelsHeightWidth => {
            vec![1, image.channels(), image.height(), image.width()]
        }
        ImageLayout::HeightWidthChannels => {
            vec![1, image.height(), image.width(), image.channels()]
        }
    };
    let bytes: Vec<u8> = match layout {
        ImageLayout::ChannelsHeightWidth => (0..image.channels())
            .flat_map(|channel| {
                image
                    .data()
                    .chunks_exact(image.channels())
                    .map(move |pixel| pixel[stage_source_channel(flip_channel_order, channel)])
            })
            .collect(),
        ImageLayout::HeightWidthChannels => image
            .data()
            .chunks_exact(image.channels())
            .flat_map(|pixel| {
                (0..image.channels())
                    .map(move |channel| pixel[stage_source_channel(flip_channel_order, channel)])
            })
            .collect(),
    };
    let data = if unit_range {
        TensorData::F32(
            bytes
                .into_iter()
                .map(|value| value as f32 * RESCALE_FACTOR)
                .collect(),
        )
    } else {
        TensorData::U8(bytes)
    };
    Ok(Tensor::new(data, shape, batch_layout)?.with_leading_axis(TensorLeadingAxis::Batch)?)
}

fn stage_source_channel(flip_channel_order: bool, channel: usize) -> usize {
    if flip_channel_order && channel < 3 {
        2 - channel
    } else {
        channel
    }
}

fn checked_ratio(value: usize, multiplier: usize, divisor: usize) -> Result<usize, TransformError> {
    if divisor == 0 {
        return Err(TransformError::InvalidScaleFactor(divisor));
    }
    let result = (value as u128)
        .checked_mul(multiplier as u128)
        .ok_or(TransformError::ImageSizeOverflow)?
        / divisor as u128;
    usize::try_from(result).map_err(|_| TransformError::ImageSizeOverflow)
}

fn align_up(value: usize, divisor: usize) -> Result<usize, TransformError> {
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor * divisor)
        .ok_or(TransformError::ImageSizeOverflow)
}

const fn batched_layout(layout: ImageLayout) -> Layout {
    match layout {
        ImageLayout::ChannelsHeightWidth => Layout::NCHW,
        ImageLayout::HeightWidthChannels => Layout::NHWC,
    }
}
