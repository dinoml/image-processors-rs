//! Shared task-vision presets audited against Transformers image processors.

use super::*;
use crate::postprocess::{CoordinatePoint, RecipePostprocessContext};
use crate::recipe::{
    RecipeCoordinatePostprocess, RecipeCoordinateTask, RecipeDenseMapPostprocess,
    RecipeDenseMapTask, RecipeDepthPostprocess, RecipeDepthUnit, RecipeDetectionScoreMode,
    RecipeSegmentationPostprocess, RecipeSegmentationStrategy, RecipeSegmentationTask,
};
use crate::tensor::TensorError;
use crate::transforms::CanvasFill;

/// Broad task family represented by a task-vision preset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskVisionProcessorFamily {
    /// Object detection, open-vocabulary detection, or document grounding.
    DetectionGrounding,
    /// Semantic, instance, panoptic, prompt-mask, or matting processing.
    Segmentation,
    /// Dense depth or image-geometry processing.
    DepthGeometry,
    /// Keypoint extraction, image matching, or pose processing.
    KeypointMatchingPose,
}

/// Transformers task-vision processor preset audited by this crate.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskVisionProcessorPreset {
    /// `ConditionalDetrImageProcessor`.
    ConditionalDetr,
    /// `DeformableDetrImageProcessor`.
    DeformableDetr,
    /// `GroundingDinoImageProcessor`.
    GroundingDino,
    /// `OwlViTImageProcessor`.
    OwlVit,
    /// `Owlv2ImageProcessor`.
    Owlv2,
    /// `PPDocLayoutV2ImageProcessor`.
    PpDocLayoutV2,
    /// `PPDocLayoutV3ImageProcessor`.
    PpDocLayoutV3,
    /// `PPOCRV5ServerDetImageProcessor`.
    PpOcrV5ServerDet,
    /// `RTDetrImageProcessor`.
    RtDetr,
    /// `RfDetrImageProcessor`.
    RfDetr,
    /// `YolosImageProcessor`.
    Yolos,
    /// `EomtImageProcessor`.
    Eomt,
    /// `Mask2FormerImageProcessor`.
    Mask2Former,
    /// `MaskFormerImageProcessor`.
    MaskFormer,
    /// `OneFormerImageProcessor`.
    OneFormer,
    /// `Sam2ImageProcessor`.
    Sam2,
    /// `Sam3ImageProcessor`.
    Sam3,
    /// `Sapiens2ImageProcessor`.
    Sapiens2,
    /// `SegGptImageProcessor`.
    SegGpt,
    /// `SegformerImageProcessor`.
    Segformer,
    /// `VitMatteImageProcessor`.
    VitMatte,
    /// `CHMv2ImageProcessor`.
    ChmV2,
    /// `DPTImageProcessor`.
    Dpt,
    /// `DepthProImageProcessor`.
    DepthPro,
    /// `GLPNImageProcessor`.
    Glpn,
    /// `PromptDepthAnythingImageProcessor`.
    PromptDepthAnything,
    /// `Tipsv2DptImageProcessor`.
    TipsV2Dpt,
    /// `Tipsv2ImageProcessor`.
    TipsV2,
    /// `ZoeDepthImageProcessor`.
    ZoeDepth,
    /// `EfficientLoFTRImageProcessor`.
    EfficientLoFtr,
    /// `LightGlueImageProcessor`.
    LightGlue,
    /// `SuperGlueImageProcessor`.
    SuperGlue,
    /// `SuperPointImageProcessor`.
    SuperPoint,
    /// `VitPoseImageProcessor`.
    VitPose,
}

impl TaskVisionProcessorPreset {
    /// Every task-vision preset in stable catalog order.
    pub const ALL: [Self; 34] = [
        Self::ConditionalDetr,
        Self::DeformableDetr,
        Self::GroundingDino,
        Self::OwlVit,
        Self::Owlv2,
        Self::PpDocLayoutV2,
        Self::PpDocLayoutV3,
        Self::PpOcrV5ServerDet,
        Self::RtDetr,
        Self::RfDetr,
        Self::Yolos,
        Self::Eomt,
        Self::Mask2Former,
        Self::MaskFormer,
        Self::OneFormer,
        Self::Sam2,
        Self::Sam3,
        Self::Sapiens2,
        Self::SegGpt,
        Self::Segformer,
        Self::VitMatte,
        Self::ChmV2,
        Self::Dpt,
        Self::DepthPro,
        Self::Glpn,
        Self::PromptDepthAnything,
        Self::TipsV2Dpt,
        Self::TipsV2,
        Self::ZoeDepth,
        Self::EfficientLoFtr,
        Self::LightGlue,
        Self::SuperGlue,
        Self::SuperPoint,
        Self::VitPose,
    ];

    /// Resolves a canonical Transformers class name or its `*Pil` alias.
    pub fn from_class_name(class_name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| {
            preset.class_name() == class_name || preset.class_aliases().contains(&class_name)
        })
    }

    /// Returns the canonical upstream class name.
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::ConditionalDetr => "ConditionalDetrImageProcessor",
            Self::DeformableDetr => "DeformableDetrImageProcessor",
            Self::GroundingDino => "GroundingDinoImageProcessor",
            Self::OwlVit => "OwlViTImageProcessor",
            Self::Owlv2 => "Owlv2ImageProcessor",
            Self::PpDocLayoutV2 => "PPDocLayoutV2ImageProcessor",
            Self::PpDocLayoutV3 => "PPDocLayoutV3ImageProcessor",
            Self::PpOcrV5ServerDet => "PPOCRV5ServerDetImageProcessor",
            Self::RtDetr => "RTDetrImageProcessor",
            Self::RfDetr => "RfDetrImageProcessor",
            Self::Yolos => "YolosImageProcessor",
            Self::Eomt => "EomtImageProcessor",
            Self::Mask2Former => "Mask2FormerImageProcessor",
            Self::MaskFormer => "MaskFormerImageProcessor",
            Self::OneFormer => "OneFormerImageProcessor",
            Self::Sam2 => "Sam2ImageProcessor",
            Self::Sam3 => "Sam3ImageProcessor",
            Self::Sapiens2 => "Sapiens2ImageProcessor",
            Self::SegGpt => "SegGptImageProcessor",
            Self::Segformer => "SegformerImageProcessor",
            Self::VitMatte => "VitMatteImageProcessor",
            Self::ChmV2 => "CHMv2ImageProcessor",
            Self::Dpt => "DPTImageProcessor",
            Self::DepthPro => "DepthProImageProcessor",
            Self::Glpn => "GLPNImageProcessor",
            Self::PromptDepthAnything => "PromptDepthAnythingImageProcessor",
            Self::TipsV2Dpt => "Tipsv2DptImageProcessor",
            Self::TipsV2 => "Tipsv2ImageProcessor",
            Self::ZoeDepth => "ZoeDepthImageProcessor",
            Self::EfficientLoFtr => "EfficientLoFTRImageProcessor",
            Self::LightGlue => "LightGlueImageProcessor",
            Self::SuperGlue => "SuperGlueImageProcessor",
            Self::SuperPoint => "SuperPointImageProcessor",
            Self::VitPose => "VitPoseImageProcessor",
        }
    }

    const fn class_aliases(self) -> &'static [&'static str] {
        match self {
            Self::ConditionalDetr => &["ConditionalDetrImageProcessorPil"],
            Self::DeformableDetr => &["DeformableDetrImageProcessorPil"],
            Self::GroundingDino => &["GroundingDinoImageProcessorPil"],
            Self::OwlVit => &["OwlViTImageProcessorPil"],
            Self::Owlv2 => &["Owlv2ImageProcessorPil"],
            Self::RtDetr => &["RTDetrImageProcessorPil"],
            Self::Yolos => &["YolosImageProcessorPil"],
            Self::Eomt => &["EomtImageProcessorPil"],
            Self::Mask2Former => &["Mask2FormerImageProcessorPil"],
            Self::MaskFormer => &["MaskFormerImageProcessorPil"],
            Self::OneFormer => &["OneFormerImageProcessorPil"],
            Self::SegGpt => &["SegGptImageProcessorPil"],
            Self::Segformer => &["SegformerImageProcessorPil"],
            Self::VitMatte => &["VitMatteImageProcessorPil"],
            Self::Dpt => &["DPTImageProcessorPil"],
            Self::Glpn => &["GLPNImageProcessorPil"],
            Self::PromptDepthAnything => &["PromptDepthAnythingImageProcessorPil"],
            Self::ZoeDepth => &["ZoeDepthImageProcessorPil"],
            Self::EfficientLoFtr => &["EfficientLoFTRImageProcessorPil"],
            Self::LightGlue => &["LightGlueImageProcessorPil"],
            Self::SuperGlue => &["SuperGlueImageProcessorPil"],
            Self::SuperPoint => &["SuperPointImageProcessorPil"],
            Self::VitPose => &["VitPoseImageProcessorPil"],
            Self::PpDocLayoutV2
            | Self::PpDocLayoutV3
            | Self::PpOcrV5ServerDet
            | Self::RfDetr
            | Self::Sam2
            | Self::Sam3
            | Self::Sapiens2
            | Self::ChmV2
            | Self::DepthPro
            | Self::TipsV2Dpt
            | Self::TipsV2 => &[],
        }
    }

    /// Returns the shared recipe id used by this preset.
    pub const fn recipe_id(self) -> &'static str {
        match self {
            Self::ConditionalDetr | Self::DeformableDetr => {
                "transformers.conditional_detr_image_processor"
            }
            Self::GroundingDino => "transformers.grounding_dino_image_processor",
            Self::OwlVit => "transformers.owlvit_image_processor",
            Self::Owlv2 => "transformers.owlv2_image_processor",
            Self::PpDocLayoutV2 | Self::PpDocLayoutV3 => {
                "transformers.pp_doclayout_image_processor"
            }
            Self::PpOcrV5ServerDet => "transformers.pp_ocr_detection_image_processor",
            Self::RtDetr => "transformers.rt_detr_image_processor",
            Self::RfDetr => "transformers.rf_detr_image_processor",
            Self::Yolos => "transformers.yolos_image_processor",
            Self::Eomt => "transformers.eomt_image_processor",
            Self::Mask2Former | Self::MaskFormer => "transformers.maskformer_image_processor",
            Self::OneFormer => "transformers.oneformer_image_processor",
            Self::Sam2 => "transformers.sam2_image_processor",
            Self::Sam3 => "transformers.sam3_image_processor",
            Self::Sapiens2 => "transformers.sapiens2_image_processor",
            Self::SegGpt => "transformers.seggpt_image_processor",
            Self::Segformer => "transformers.segformer_image_processor",
            Self::VitMatte => "transformers.vitmatte_image_processor",
            Self::ChmV2 => "transformers.chmv2_image_processor",
            Self::Dpt | Self::PromptDepthAnything => "transformers.dpt_image_processor",
            Self::DepthPro => "transformers.depth_pro_image_processor",
            Self::Glpn => "transformers.glpn_image_processor",
            Self::TipsV2Dpt | Self::TipsV2 => "transformers.tipsv2_image_processor",
            Self::ZoeDepth => "transformers.zoedepth_image_processor",
            Self::EfficientLoFtr | Self::LightGlue | Self::SuperGlue => {
                "transformers.keypoint_matching_image_processor"
            }
            Self::SuperPoint => "transformers.superpoint_image_processor",
            Self::VitPose => "transformers.vitpose_image_processor",
        }
    }

    /// Returns the broad task family represented by this preset.
    pub const fn family(self) -> TaskVisionProcessorFamily {
        match self {
            Self::ConditionalDetr
            | Self::DeformableDetr
            | Self::GroundingDino
            | Self::OwlVit
            | Self::Owlv2
            | Self::PpDocLayoutV2
            | Self::PpDocLayoutV3
            | Self::PpOcrV5ServerDet
            | Self::RtDetr
            | Self::RfDetr
            | Self::Yolos => TaskVisionProcessorFamily::DetectionGrounding,
            Self::Eomt
            | Self::Mask2Former
            | Self::MaskFormer
            | Self::OneFormer
            | Self::Sam2
            | Self::Sam3
            | Self::Sapiens2
            | Self::SegGpt
            | Self::Segformer
            | Self::VitMatte => TaskVisionProcessorFamily::Segmentation,
            Self::ChmV2
            | Self::Dpt
            | Self::DepthPro
            | Self::Glpn
            | Self::PromptDepthAnything
            | Self::TipsV2Dpt
            | Self::TipsV2
            | Self::ZoeDepth => TaskVisionProcessorFamily::DepthGeometry,
            Self::EfficientLoFtr
            | Self::LightGlue
            | Self::SuperGlue
            | Self::SuperPoint
            | Self::VitPose => TaskVisionProcessorFamily::KeypointMatchingPose,
        }
    }

    const fn expects_pair(self) -> bool {
        matches!(
            self,
            Self::EfficientLoFtr | Self::LightGlue | Self::SuperGlue
        )
    }
}

/// Color conversion applied before tensor conversion.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskVisionColorMode {
    /// Preserve RGB values.
    Rgb,
    /// Convert to luminance and replicate it across three RGB channels.
    GrayscaleRgb,
}

/// Resize geometry used by a task-vision preset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskVisionResize {
    /// Preserve decoded spatial dimensions.
    None,
    /// Resize directly to a fixed size.
    Fixed {
        /// Fixed output dimensions.
        size: ImageSize,
    },
    /// Preserve aspect ratio while targeting the shortest edge.
    ShortestEdge {
        /// Target shorter edge.
        shortest_edge: usize,
        /// Optional longer-edge cap.
        longest_edge: Option<usize>,
        /// Optional divisor applied to the resolved resize dimensions.
        #[serde(default)]
        multiple: Option<usize>,
    },
    /// Round both axes down to a divisor, as used by GLPN.
    RoundDownToMultiple {
        /// Required axis divisor.
        multiple: usize,
    },
    /// PaddleOCR detection resize with 32-pixel rounding.
    PaddleDetection {
        /// Limit used by the selected side policy.
        limit_side_len: usize,
        /// Side-limit policy.
        limit_type: PaddleDetectionLimit,
        /// Optional post-limit maximum edge.
        max_side_limit: Option<usize>,
    },
}

/// Side-limit policy used by PaddleOCR detection preprocessing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddleDetectionLimit {
    /// Downscale only when the longest edge exceeds the limit.
    Max,
    /// Upscale only when the shortest edge is below the limit.
    Min,
    /// Always resize the longest edge to the limit.
    ResizeLong,
}

/// Constant or reflected padding applied around task-vision images.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskVisionPadding {
    /// Do not pad.
    #[default]
    None,
    /// Pad zeros on the bottom and right to an axis divisor.
    BottomRightToMultiple {
        /// Required axis divisor.
        multiple: usize,
    },
    /// Center-pad zeros to an axis divisor.
    CenterToMultiple {
        /// Required axis divisor.
        multiple: usize,
    },
    /// Pad zeros on the bottom and right to a square.
    SquareBottomRight,
    /// Apply ZoeDepth's symmetric reflected context padding.
    ZoeReflect,
}

/// Shared preprocessing configuration for audited task-vision presets.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TaskVisionImageProcessorConfig {
    /// Upstream preset represented by this configuration.
    pub preset: TaskVisionProcessorPreset,
    /// Resize geometry.
    pub resize: TaskVisionResize,
    /// Padding applied before resize.
    #[serde(default)]
    pub pre_resize_padding: TaskVisionPadding,
    /// Padding applied after resize.
    #[serde(default)]
    pub post_resize_padding: TaskVisionPadding,
    /// Color conversion policy.
    pub color_mode: TaskVisionColorMode,
    /// Whether RGB channels are emitted in reverse order.
    #[serde(default)]
    pub reverse_channels: bool,
    /// Resize filter.
    pub resample: ResizeFilter,
    /// Resize parity implementation.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Whether tensor values are rescaled.
    pub do_rescale: bool,
    /// Tensor rescale factor.
    pub rescale_factor: f32,
    /// Whether per-channel normalization is applied.
    pub do_normalize: bool,
    /// Normalization means.
    pub image_mean: Vec<f32>,
    /// Normalization standard deviations.
    pub image_std: Vec<f32>,
    /// Device-independent output image layout.
    pub output_layout: ImageLayout,
    /// Whether preprocessing emits a valid-pixel mask.
    #[serde(default)]
    pub emit_pixel_mask: bool,
    /// Whether preprocessing emits upstream original-size metadata.
    #[serde(default)]
    pub emit_original_sizes: bool,
    /// Whether preprocessing emits resized-size metadata.
    #[serde(default)]
    pub emit_reshaped_input_sizes: bool,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch execution policy.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl TaskVisionImageProcessorConfig {
    /// Creates the audited default configuration for a preset.
    pub fn for_preset(preset: TaskVisionProcessorPreset) -> Self {
        let mut config = Self {
            preset,
            resize: TaskVisionResize::Fixed {
                size: ImageSize {
                    height: 224,
                    width: 224,
                },
            },
            pre_resize_padding: TaskVisionPadding::None,
            post_resize_padding: TaskVisionPadding::None,
            color_mode: TaskVisionColorMode::Rgb,
            reverse_channels: false,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Torchvision,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: IMAGENET_MEAN.to_vec(),
            image_std: IMAGENET_STD.to_vec(),
            output_layout: ImageLayout::ChannelsHeightWidth,
            emit_pixel_mask: false,
            emit_original_sizes: false,
            emit_reshaped_input_sizes: false,
        };

        match preset {
            TaskVisionProcessorPreset::ConditionalDetr
            | TaskVisionProcessorPreset::DeformableDetr
            | TaskVisionProcessorPreset::GroundingDino
            | TaskVisionProcessorPreset::RfDetr => {
                config.resize = shortest_edge(800, Some(1333));
                config.emit_pixel_mask = true;
                config.emit_original_sizes = true;
                config.emit_reshaped_input_sizes = true;
            }
            TaskVisionProcessorPreset::Yolos => {
                config.resize = TaskVisionResize::ShortestEdge {
                    shortest_edge: 800,
                    longest_edge: Some(1333),
                    multiple: Some(16),
                };
                config.emit_pixel_mask = true;
                config.emit_original_sizes = true;
                config.emit_reshaped_input_sizes = true;
            }
            TaskVisionProcessorPreset::OwlVit => {
                config.resize = fixed_size(768, 768);
                config.resample = ResizeFilter::Bicubic;
                config.set_normalization(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD);
            }
            TaskVisionProcessorPreset::Owlv2 => {
                config.resize = fixed_size(960, 960);
                config.pre_resize_padding = TaskVisionPadding::SquareBottomRight;
                config.set_normalization(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD);
            }
            TaskVisionProcessorPreset::PpDocLayoutV2 | TaskVisionProcessorPreset::PpDocLayoutV3 => {
                config.resize = fixed_size(800, 800);
                config.resample = ResizeFilter::Bicubic;
                config.set_normalization([0.0; 3], [1.0; 3]);
            }
            TaskVisionProcessorPreset::PpOcrV5ServerDet => {
                config.resize = TaskVisionResize::PaddleDetection {
                    limit_side_len: 960,
                    limit_type: PaddleDetectionLimit::Max,
                    max_side_limit: Some(4000),
                };
                config.reverse_channels = true;
                // Upstream normalizes RGB and then reverses the normalized channels.
                // This implementation reverses bytes first, so the statistics follow BGR.
                config.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
                config.emit_original_sizes = true;
            }
            TaskVisionProcessorPreset::RtDetr => {
                config.resize = fixed_size(640, 640);
                config.disable_normalization();
            }
            TaskVisionProcessorPreset::Eomt => {
                config.resize = shortest_edge(640, Some(640));
            }
            TaskVisionProcessorPreset::Mask2Former | TaskVisionProcessorPreset::MaskFormer => {
                config.resize = TaskVisionResize::ShortestEdge {
                    shortest_edge: 800,
                    longest_edge: Some(1333),
                    multiple: Some(32),
                };
                config.emit_pixel_mask = true;
            }
            TaskVisionProcessorPreset::OneFormer => {
                config.resize = shortest_edge(800, Some(1333));
                config.emit_pixel_mask = true;
            }
            TaskVisionProcessorPreset::Sam2 => {
                config.resize = fixed_size(1024, 1024);
                config.emit_original_sizes = true;
            }
            TaskVisionProcessorPreset::Sam3 => {
                config.resize = fixed_size(1008, 1008);
                config.set_normalization(STANDARD_IMAGE_MEAN, STANDARD_IMAGE_STD);
                config.emit_original_sizes = true;
            }
            TaskVisionProcessorPreset::Sapiens2 => {
                config.resize = fixed_size(1024, 768);
            }
            TaskVisionProcessorPreset::SegGpt => {
                config.resize = fixed_size(448, 448);
                config.resample = ResizeFilter::Bicubic;
            }
            TaskVisionProcessorPreset::Segformer => {
                config.resize = fixed_size(512, 512);
            }
            TaskVisionProcessorPreset::VitMatte => {
                config.resize = TaskVisionResize::None;
                config.post_resize_padding =
                    TaskVisionPadding::BottomRightToMultiple { multiple: 32 };
                config.set_normalization(STANDARD_IMAGE_MEAN, STANDARD_IMAGE_STD);
            }
            TaskVisionProcessorPreset::ChmV2 => {
                config.resize = TaskVisionResize::None;
                config.post_resize_padding = TaskVisionPadding::CenterToMultiple { multiple: 16 };
                config.resample = ResizeFilter::Bicubic;
                config.set_normalization([0.42, 0.411, 0.296], [0.213, 0.156, 0.143]);
            }
            TaskVisionProcessorPreset::Dpt | TaskVisionProcessorPreset::PromptDepthAnything => {
                config.resize = fixed_size(384, 384);
                config.resample = ResizeFilter::Bicubic;
                config.set_normalization(STANDARD_IMAGE_MEAN, STANDARD_IMAGE_STD);
            }
            TaskVisionProcessorPreset::DepthPro => {
                config.resize = fixed_size(1536, 1536);
                config.set_normalization(STANDARD_IMAGE_MEAN, STANDARD_IMAGE_STD);
            }
            TaskVisionProcessorPreset::Glpn => {
                config.resize = TaskVisionResize::RoundDownToMultiple { multiple: 32 };
                config.disable_normalization();
            }
            TaskVisionProcessorPreset::TipsV2Dpt | TaskVisionProcessorPreset::TipsV2 => {
                config.resize = fixed_size(448, 448);
                config.disable_normalization();
            }
            TaskVisionProcessorPreset::ZoeDepth => {
                config.resize = fixed_size(384, 512);
                config.pre_resize_padding = TaskVisionPadding::ZoeReflect;
                config.set_normalization(STANDARD_IMAGE_MEAN, STANDARD_IMAGE_STD);
            }
            TaskVisionProcessorPreset::EfficientLoFtr
            | TaskVisionProcessorPreset::LightGlue
            | TaskVisionProcessorPreset::SuperGlue
            | TaskVisionProcessorPreset::SuperPoint => {
                config.resize = fixed_size(480, 640);
                if preset != TaskVisionProcessorPreset::SuperPoint {
                    config.color_mode = TaskVisionColorMode::GrayscaleRgb;
                }
                config.disable_normalization();
            }
            TaskVisionProcessorPreset::VitPose => {
                config.resize = fixed_size(256, 192);
            }
        }
        config
    }

    fn disable_normalization(&mut self) {
        self.do_normalize = false;
        self.image_mean.clear();
        self.image_std.clear();
    }

    fn set_normalization(&mut self, mean: [f32; 3], std: [f32; 3]) {
        self.image_mean = mean.to_vec();
        self.image_std = std.to_vec();
    }

    /// Creates the audited configuration for a canonical or explicit PIL class name.
    ///
    /// Canonical task-vision classes use Transformers' Torchvision backend, while
    /// `*ImageProcessorPil` aliases select Pillow compatibility semantics.
    pub fn for_class_name(class_name: &str) -> Option<Self> {
        let preset = TaskVisionProcessorPreset::from_class_name(class_name)?;
        let mut config = Self::for_preset(preset);
        if class_name.ends_with("ImageProcessorPil") {
            config.resize_parity = ResizeParity::Compatibility;
        }
        Some(config)
    }

    /// Converts this configuration into a reusable validated recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, normalization, or postprocessing fields are invalid.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = vec![ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        }];
        if self.preset == TaskVisionProcessorPreset::Owlv2 {
            let TaskVisionResize::Fixed { size } = self.resize else {
                return Err(RecipeError::UnsupportedGenericStage {
                    stage: "owlv2_non_fixed_resize",
                });
            };
            let pad_to_square = self.pre_resize_padding == TaskVisionPadding::SquareBottomRight;
            if !matches!(
                self.pre_resize_padding,
                TaskVisionPadding::None | TaskVisionPadding::SquareBottomRight
            ) {
                append_padding_recipe_stage(&mut stages, self.pre_resize_padding);
            }
            stages.push(ProcessorRecipeStage::Owlv2AntialiasedResize {
                size,
                pad_to_square,
                do_rescale: self.do_rescale,
                rescale_factor: self.rescale_factor,
            });
            append_padding_recipe_stage(&mut stages, self.post_resize_padding);
            if self.do_normalize {
                stages.push(ProcessorRecipeStage::Normalize {
                    mean: self.image_mean.clone(),
                    std: self.image_std.clone(),
                });
            }
            return ProcessorRecipe::new_with_postprocess(
                self.preset.recipe_id(),
                ProcessorRecipeInput {
                    decode_backend: self.decode_backend,
                    batch_execution: self.batch_execution,
                },
                stages,
                ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
                task_postprocess_descriptors(self)?,
            );
        }
        if self.preset == TaskVisionProcessorPreset::VitPose {
            let TaskVisionResize::Fixed { size } = self.resize else {
                return Err(RecipeError::UnsupportedGenericStage {
                    stage: "vit_pose_non_fixed_affine",
                });
            };
            append_padding_recipe_stage(&mut stages, self.pre_resize_padding);
            stages.push(ProcessorRecipeStage::VitPoseAffine {
                size,
                normalize_factor: 200.0,
                padding_factor: 1.25,
            });
            append_padding_recipe_stage(&mut stages, self.post_resize_padding);
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
            return ProcessorRecipe::new_with_postprocess(
                self.preset.recipe_id(),
                ProcessorRecipeInput {
                    decode_backend: self.decode_backend,
                    batch_execution: self.batch_execution,
                },
                stages,
                ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
                task_postprocess_descriptors(self)?,
            );
        }
        append_padding_recipe_stage(&mut stages, self.pre_resize_padding);
        match self.resize {
            TaskVisionResize::None => {}
            TaskVisionResize::Fixed { size } => stages.push(ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::fixed(
                    size,
                    ResizeMode::Default,
                    self.resample,
                    self.resize_parity,
                ),
            }),
            TaskVisionResize::ShortestEdge {
                shortest_edge,
                longest_edge,
                multiple,
            } => {
                if self.preset == TaskVisionProcessorPreset::Yolos {
                    stages.push(ProcessorRecipeStage::Resize {
                        resize: RecipeResizeStage::shortest_edge_round_down(
                            shortest_edge,
                            longest_edge,
                            multiple.unwrap_or(16),
                            self.resample,
                            self.resize_parity,
                        ),
                    });
                } else {
                    stages.push(ProcessorRecipeStage::Resize {
                        resize: RecipeResizeStage::shortest_edge(
                            shortest_edge,
                            longest_edge,
                            self.resample,
                            self.resize_parity,
                        ),
                    });
                    if let Some(multiple) = multiple {
                        stages.push(ProcessorRecipeStage::RoundToMultiple {
                            requested_size: None,
                            multiples: ImageSize {
                                height: multiple,
                                width: multiple,
                            },
                        });
                    }
                }
            }
            TaskVisionResize::RoundDownToMultiple { multiple } => {
                let multiples = ImageSize {
                    height: multiple,
                    width: multiple,
                };
                stages.push(ProcessorRecipeStage::RoundToMultiple {
                    requested_size: None,
                    multiples,
                });
            }
            TaskVisionResize::PaddleDetection { .. } => {
                stages.push(ProcessorRecipeStage::Resize {
                    resize: RecipeResizeStage {
                        target: crate::recipe::RecipeResizeTarget::Dynamic,
                        mode: ResizeMode::Default,
                        filter: self.resample,
                        parity: self.resize_parity,
                    },
                });
                stages.push(ProcessorRecipeStage::RoundToMultiple {
                    requested_size: None,
                    multiples: ImageSize {
                        height: 32,
                        width: 32,
                    },
                });
            }
        }
        append_padding_recipe_stage(&mut stages, self.post_resize_padding);
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

        // Recipes describe the per-image tensor contract. Pair-based wrappers add
        // the image-slot axis after executing this shared image recipe.
        let layout = batched_layout_for_image_layout(self.output_layout);
        ProcessorRecipe::new_with_postprocess(
            self.preset.recipe_id(),
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(layout),
            task_postprocess_descriptors(self)?,
        )
    }

    fn tensor_processor_config(&self, pixel_format: PixelFormat) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }
}

/// Shared processor for task-vision preset preprocessing.
#[derive(Debug)]
pub struct TaskVisionImageProcessor {
    config: TaskVisionImageProcessorConfig,
    tensor_processor: ImageProcessor,
}

/// Runtime options for a query-mask segmentation interpretation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskVisionSegmentationRequest {
    task: RecipeSegmentationTask,
    label_ids_to_fuse: Vec<i64>,
}

impl TaskVisionSegmentationRequest {
    /// Creates a request for a semantic, instance, or panoptic map.
    pub fn new(task: RecipeSegmentationTask) -> Self {
        Self {
            task,
            label_ids_to_fuse: Vec::new(),
        }
    }

    /// Creates a panoptic request with dataset-provided stuff labels to fuse.
    pub fn panoptic(label_ids_to_fuse: impl IntoIterator<Item = i64>) -> Self {
        Self {
            task: RecipeSegmentationTask::Panoptic,
            label_ids_to_fuse: label_ids_to_fuse.into_iter().collect(),
        }
    }

    /// Returns the requested segmentation interpretation.
    pub fn task(&self) -> RecipeSegmentationTask {
        self.task
    }

    /// Returns dataset label ids whose panoptic instances share one segment id.
    pub fn label_ids_to_fuse(&self) -> &[i64] {
        &self.label_ids_to_fuse
    }
}

impl From<RecipeSegmentationTask> for TaskVisionSegmentationRequest {
    fn from(task: RecipeSegmentationTask) -> Self {
        Self::new(task)
    }
}

/// Matched keypoints restored to the two original image coordinate spaces.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskVisionKeypointMatchingOutput {
    keypoints0: Vec<CoordinatePoint>,
    keypoints1: Vec<CoordinatePoint>,
    matching_scores: Vec<f32>,
}

impl TaskVisionKeypointMatchingOutput {
    /// Returns matched keypoints in the first image.
    pub fn keypoints0(&self) -> &[CoordinatePoint] {
        &self.keypoints0
    }

    /// Returns the corresponding matched keypoints in the second image.
    pub fn keypoints1(&self) -> &[CoordinatePoint] {
        &self.keypoints1
    }

    /// Returns the confidence score for each matched keypoint pair.
    pub fn matching_scores(&self) -> &[f32] {
        &self.matching_scores
    }

    /// Consumes the output into first-image points, second-image points, and scores.
    pub fn into_parts(self) -> (Vec<CoordinatePoint>, Vec<CoordinatePoint>, Vec<f32>) {
        (self.keypoints0, self.keypoints1, self.matching_scores)
    }
}

/// One COCO-format `(top_left_x, top_left_y, width, height)` VitPose input box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaskVisionPoseBox {
    top_left_x: f32,
    top_left_y: f32,
    width: f32,
    height: f32,
}

impl TaskVisionPoseBox {
    /// Creates a finite, non-empty VitPose input box.
    ///
    /// # Errors
    ///
    /// Returns an error when a coordinate is non-finite or an extent is not positive.
    pub fn new(
        top_left_x: f32,
        top_left_y: f32,
        width: f32,
        height: f32,
    ) -> Result<Self, TransformError> {
        let bbox = [top_left_x, top_left_y, width, height];
        if bbox.iter().any(|value| !value.is_finite()) || width <= 0.0 || height <= 0.0 {
            return Err(TransformError::InvalidBoundingBox { bbox });
        }
        Ok(Self {
            top_left_x,
            top_left_y,
            width,
            height,
        })
    }

    /// Returns the top-left horizontal coordinate.
    pub fn top_left_x(self) -> f32 {
        self.top_left_x
    }

    /// Returns the top-left vertical coordinate.
    pub fn top_left_y(self) -> f32 {
        self.top_left_y
    }

    /// Returns the box width.
    pub fn width(self) -> f32 {
        self.width
    }

    /// Returns the box height.
    pub fn height(self) -> f32 {
        self.height
    }
}

impl TaskVisionImageProcessor {
    /// Creates a task-vision processor after validating its tensor configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, divisors, resize policy, or normalization values.
    pub fn new(config: TaskVisionImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_task_vision_config(&config)?;
        let tensor_processor =
            ImageProcessor::new(config.tensor_processor_config(PixelFormat::Rgb8))?;
        Ok(Self {
            config,
            tensor_processor,
        })
    }

    /// Returns this processor's configuration.
    pub fn config(&self) -> &TaskVisionImageProcessorConfig {
        &self.config
    }

    /// Preprocesses one decoded RGB image into typed task-vision outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry, padding, color conversion, or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        if self.config.preset == TaskVisionProcessorPreset::Owlv2 {
            return self.preprocess_owlv2_output(image);
        }
        if self.config.preset == TaskVisionProcessorPreset::ZoeDepth {
            return self.preprocess_zoe_output(image);
        }
        let (prepared, original_size, reshaped_size) = self.prepare_image(image)?;
        let tensor = self.tensor_processor.preprocess_image(&prepared)?;
        self.output_with_metadata(tensor, original_size, reshaped_size)
    }

    /// Preprocesses one image pair into a `[batch, pair, channel, height, width]` tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the preset is not pair-based or pair geometry is incompatible.
    pub fn preprocess_pair_output(
        &self,
        first: &ImageFrame,
        second: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        if !self.config.preset.expects_pair() {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        let (first, first_original, first_reshaped) = self.prepare_image(first)?;
        let (second, second_original, second_reshaped) = self.prepare_image(second)?;
        if first_reshaped != second_reshaped {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        let tensor = self.tensor_processor.preprocess_images(&[first, second])?;
        let (data, shape, _, _) = tensor.into_parts();
        let (channels, height, width, layout) = match self.config.output_layout {
            ImageLayout::ChannelsHeightWidth => (shape[1], shape[2], shape[3], Layout::NPCHW),
            ImageLayout::HeightWidthChannels => (shape[3], shape[1], shape[2], Layout::NPHWC),
        };
        let shape = match self.config.output_layout {
            ImageLayout::ChannelsHeightWidth => vec![1, 2, channels, height, width],
            ImageLayout::HeightWidthChannels => vec![1, 2, height, width, channels],
        };
        let tensor = Tensor::new(data, shape, layout)
            .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))?;
        let mut output = ProcessorOutput::from_pixel_values(tensor);
        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(vec![first_original, second_original]),
        );
        output.insert_metadata(
            ProcessorMetadataName::ReshapedInputSizes,
            ProcessorMetadataValue::ImageSizes(vec![first_reshaped, second_reshaped]),
        );
        Ok(output)
    }

    /// Preprocesses a VitMatte RGB image and trimap into a four-channel tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the preset is not VitMatte, image sizes differ, or tensor conversion fails.
    pub fn preprocess_matte_output(
        &self,
        image: &ImageFrame,
        trimap: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        if self.config.preset != TaskVisionProcessorPreset::VitMatte {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        let original_size = ImageSize::new(image.height(), image.width())?;
        if trimap.height() != image.height() || trimap.width() != image.width() {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        // Transformers concatenates normalized RGB with the rescaled trimap and
        // only then pads the four-channel float tensor with literal zeros.
        let mut unpadded_config = self.config.clone();
        unpadded_config.post_resize_padding = TaskVisionPadding::None;
        let unpadded_processor = Self::new(unpadded_config.clone())?;
        let (prepared, _, unpadded_size) = unpadded_processor.prepare_image(image)?;
        let prepared_trimap = prepare_auxiliary_frame(&unpadded_config, trimap)?;
        let image_tensor = unpadded_processor
            .tensor_processor
            .preprocess_image(&prepared)?;
        let mut trimap_config = unpadded_config.tensor_processor_config(PixelFormat::Luma8);
        trimap_config.do_normalize = false;
        trimap_config.image_mean.clear();
        trimap_config.image_std.clear();
        let trimap_tensor =
            ImageProcessor::new(trimap_config)?.preprocess_image(&prepared_trimap)?;
        let tensor =
            concatenate_matte_channels(image_tensor, trimap_tensor, self.config.output_layout)?;
        let reshaped_size = match self.config.post_resize_padding {
            TaskVisionPadding::BottomRightToMultiple { multiple } => ImageSize::new(
                round_up(unpadded_size.height, multiple)?,
                round_up(unpadded_size.width, multiple)?,
            )?,
            TaskVisionPadding::None => unpadded_size,
            _ => return Err(ImageProcessorError::IncompatibleBatchShapes),
        };
        let tensor = if reshaped_size == unpadded_size {
            tensor
        } else {
            pad_spatial_tensors(vec![tensor], reshaped_size, self.config.output_layout)?
        };
        let mut output = ProcessorOutput::from_pixel_values(tensor);
        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(vec![original_size]),
        );
        output.insert_metadata(
            ProcessorMetadataName::ReshapedInputSizes,
            ProcessorMetadataValue::ImageSizes(vec![reshaped_size]),
        );
        Ok(output)
    }

    /// Applies VitPose's default bounding-box affine transform and tensor conversion.
    ///
    /// Each input box produces one output batch sample. The transform follows the
    /// audited Transformers/SciPy bilinear warp, including zero fill outside the
    /// decoded image and integer output quantization before normalization.
    ///
    /// # Errors
    ///
    /// Returns an error for non-VitPose presets, empty box lists, unsupported resize
    /// geometry, buffer overflow, or tensor conversion failure.
    pub fn preprocess_pose_output(
        &self,
        image: &ImageFrame,
        boxes: &[TaskVisionPoseBox],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        if self.config.preset != TaskVisionProcessorPreset::VitPose {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "VitPose bounding-box affine preprocessing",
            });
        }
        if boxes.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let TaskVisionResize::Fixed { size: target_size } = self.config.resize else {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "VitPose affine resize geometry",
            });
        };
        let image = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        let transformed = boxes
            .iter()
            .copied()
            .map(|bbox| vitpose_affine_frame(&image, bbox, target_size))
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = self.tensor_processor.preprocess_images(&transformed)?;
        Ok(ProcessorOutput::from_pixel_values(tensor))
    }

    fn preprocess_owlv2_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let mut image = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        image = apply_task_padding(image, self.config.pre_resize_padding)?;
        let TaskVisionResize::Fixed { size: target_size } = self.config.resize else {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "Owlv2 float resize geometry",
            });
        };
        if self.config.post_resize_padding != TaskVisionPadding::None {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "Owlv2 post-resize padding",
            });
        }
        let source_size = ImageSize::new(image.height(), image.width())?;
        let mut values: Vec<f32> = image
            .data()
            .iter()
            .map(|&value| {
                let value = f32::from(value);
                if self.config.do_rescale {
                    value * self.config.rescale_factor
                } else {
                    value
                }
            })
            .collect();
        values = owlv2_antialiased_resize(&values, source_size, target_size)?;
        if self.config.do_normalize {
            for pixel in values.chunks_exact_mut(3) {
                for (channel, value) in pixel.iter_mut().enumerate() {
                    let mean = self.config.image_mean[channel % self.config.image_mean.len()];
                    let std = self.config.image_std[channel % self.config.image_std.len()];
                    *value = (*value - mean) / std;
                }
            }
        }
        let (data, shape, layout) = match self.config.output_layout {
            ImageLayout::HeightWidthChannels => (
                values,
                vec![1, target_size.height, target_size.width, 3],
                Layout::NHWC,
            ),
            ImageLayout::ChannelsHeightWidth => {
                let pixels = target_size
                    .height
                    .checked_mul(target_size.width)
                    .ok_or(TransformError::ImageSizeOverflow)?;
                let mut channels_first = vec![0.0_f32; pixels * 3];
                for (pixel_index, pixel) in values.chunks_exact(3).enumerate() {
                    for channel in 0..3 {
                        channels_first[channel * pixels + pixel_index] = pixel[channel];
                    }
                }
                (
                    channels_first,
                    vec![1, 3, target_size.height, target_size.width],
                    Layout::NCHW,
                )
            }
        };
        let tensor = Tensor::new(TensorData::F32(data), shape, layout)?
            .with_leading_axis(TensorLeadingAxis::Batch)?;
        self.output_with_metadata(tensor, original_size, target_size)
    }

    fn preprocess_zoe_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let mut image = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        image = apply_task_padding(image, self.config.pre_resize_padding)?;
        if self.config.post_resize_padding != TaskVisionPadding::None {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "ZoeDepth post-resize padding",
            });
        }
        let source_size = ImageSize::new(image.height(), image.width())?;
        let target_size = match self.config.resize {
            TaskVisionResize::None => source_size,
            TaskVisionResize::Fixed { size } => size,
            _ => {
                return Err(ImageProcessorError::UnsupportedProcessorOption {
                    field: "ZoeDepth float resize geometry",
                })
            }
        };
        let values = image
            .data()
            .iter()
            .map(|&value| {
                let value = f32::from(value);
                if self.config.do_rescale {
                    value * self.config.rescale_factor
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let mut values = resize_f32_align_corners(&values, source_size, target_size)?;
        if self.config.do_normalize {
            for pixel in values.chunks_exact_mut(3) {
                for (channel, value) in pixel.iter_mut().enumerate() {
                    let mean = self.config.image_mean[channel % self.config.image_mean.len()];
                    let std = self.config.image_std[channel % self.config.image_std.len()];
                    *value = (*value - mean) / std;
                }
            }
        }
        let tensor = float_hwc_to_tensor(values, target_size, self.config.output_layout)?;
        self.output_with_metadata(tensor, original_size, target_size)
    }

    fn prepare_image(
        &self,
        image: &ImageFrame,
    ) -> Result<(ImageFrame, ImageSize, ImageSize), ImageProcessorError> {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let mut frame = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        if self.config.color_mode == TaskVisionColorMode::GrayscaleRgb {
            frame = convert_frame_pixel_format(&frame, PixelFormat::Luma8)?;
            frame = convert_frame_pixel_format(&frame, PixelFormat::Rgb8)?;
        }
        frame = apply_task_padding(frame, self.config.pre_resize_padding)?;
        frame = apply_task_resize(frame, &self.config)?;
        frame = apply_task_padding(frame, self.config.post_resize_padding)?;
        if self.config.reverse_channels {
            frame = reverse_rgb_channels(frame)?;
        }
        let reshaped_size = ImageSize::new(frame.height(), frame.width())?;
        Ok((frame, original_size, reshaped_size))
    }

    fn output_with_metadata(
        &self,
        tensor: Tensor,
        original_size: ImageSize,
        reshaped_size: ImageSize,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let mut output = ProcessorOutput::from_pixel_values(tensor);
        if self.config.emit_pixel_mask {
            output.insert_tensor(
                ProcessorTensorName::PixelMask,
                full_task_pixel_mask(reshaped_size)?,
            );
        }
        if self.config.emit_original_sizes {
            let name = if self.config.preset == TaskVisionProcessorPreset::PpOcrV5ServerDet {
                ProcessorMetadataName::other("target_sizes")
            } else {
                ProcessorMetadataName::OriginalSizes
            };
            output.insert_metadata(
                name,
                ProcessorMetadataValue::ImageSizes(vec![original_size]),
            );
        }
        if self.config.emit_reshaped_input_sizes {
            output.insert_metadata(
                ProcessorMetadataName::ReshapedInputSizes,
                ProcessorMetadataValue::ImageSizes(vec![reshaped_size]),
            );
        }
        Ok(output)
    }

    /// Executes the postprocessing descriptors carried by this processor's recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when recipe construction or descriptor execution fails.
    pub fn post_process_outputs(
        &self,
        model_outputs: &ProcessorOutput,
        preprocessing_output: &ProcessorOutput,
        context: RecipePostprocessContext<'_>,
    ) -> Result<
        Vec<crate::postprocess::RecipePostprocessOutput>,
        crate::postprocess::RecipePostprocessError,
    > {
        let recipe = self.config.processor_recipe().map_err(|_| {
            crate::postprocess::RecipePostprocessError::UnsupportedDescriptor {
                task: "invalid_task_vision_recipe",
            }
        })?;
        crate::postprocess::post_process_recipe_outputs(
            &recipe,
            model_outputs,
            preprocessing_output,
            context,
        )
    }

    /// Executes a supported semantic, instance, or panoptic query-mask task.
    ///
    /// MaskFormer, Mask2Former, and EOMT support all three interpretations.
    /// OneFormer supports semantic and panoptic output; its instance path
    /// requires external `class_info_file` and thing-class metadata. Panoptic
    /// stuff-class fusion is supplied by
    /// [`TaskVisionSegmentationRequest::panoptic`], because it belongs to the
    /// model's dataset metadata rather than image preprocessing. This API
    /// returns indexed maps; COCO RLE and per-instance binary-map output modes
    /// are not exposed.
    ///
    /// # Errors
    ///
    /// Returns an error when this preset has no query-mask segmentation
    /// descriptor, recipe construction fails, descriptor execution rejects the
    /// supplied model outputs, or OneFormer instance output is requested; its
    /// external dataset metadata contract is not represented by this API.
    pub fn post_process_segmentation_outputs(
        &self,
        request: impl Into<TaskVisionSegmentationRequest>,
        model_outputs: &ProcessorOutput,
        preprocessing_output: &ProcessorOutput,
        context: RecipePostprocessContext<'_>,
    ) -> Result<
        Vec<crate::postprocess::RecipePostprocessOutput>,
        crate::postprocess::RecipePostprocessError,
    > {
        let request = request.into();
        let recipe = self.config.processor_recipe().map_err(|_| {
            crate::postprocess::RecipePostprocessError::UnsupportedDescriptor {
                task: "invalid_task_vision_recipe",
            }
        })?;
        let mut found = false;
        let descriptors = recipe
            .postprocess()
            .iter()
            .cloned()
            .map(|descriptor| match descriptor {
                ProcessorRecipePostprocess::Segmentation(mut segmentation) => {
                    segmentation.task = request.task;
                    segmentation.label_ids_to_fuse =
                        if request.task == RecipeSegmentationTask::Panoptic {
                            request.label_ids_to_fuse.clone()
                        } else {
                            Vec::new()
                        };
                    found = true;
                    ProcessorRecipePostprocess::Segmentation(segmentation)
                }
                descriptor => descriptor,
            })
            .collect();
        if !found {
            return Err(
                crate::postprocess::RecipePostprocessError::UnsupportedDescriptor {
                    task: "query_mask_segmentation",
                },
            );
        }
        let recipe = recipe.with_postprocess(descriptors).map_err(|_| {
            crate::postprocess::RecipePostprocessError::UnsupportedDescriptor {
                task: "invalid_segmentation_recipe",
            }
        })?;
        crate::postprocess::post_process_recipe_outputs(
            &recipe,
            model_outputs,
            preprocessing_output,
            context,
        )
    }

    /// Restores indexed LightGlue/SuperGlue matches to the two original image sizes.
    ///
    /// The input keypoint slices must contain only valid (unmasked) keypoints. Coordinates
    /// are normalized `(x, y)` values. Like Transformers, scaled coordinates are converted
    /// to signed 32-bit integers before filtering and returned as `f32` coordinate values.
    /// Matches with a score equal to the threshold, a negative index, or an out-of-range
    /// second-image index are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported presets or inconsistent first-image match arrays.
    pub fn post_process_indexed_keypoint_matching(
        &self,
        normalized_keypoints0: &[[f32; 2]],
        normalized_keypoints1: &[[f32; 2]],
        matches0: &[i64],
        scores0: &[f32],
        target_sizes: [ImageSize; 2],
        threshold: f32,
    ) -> Result<TaskVisionKeypointMatchingOutput, ImageProcessorError> {
        if !matches!(
            self.config.preset,
            TaskVisionProcessorPreset::LightGlue | TaskVisionProcessorPreset::SuperGlue
        ) {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "indexed keypoint matching postprocess",
            });
        }
        if normalized_keypoints0.len() != matches0.len()
            || normalized_keypoints0.len() != scores0.len()
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if !threshold.is_finite() {
            return Err(TransformError::InvalidScoreValue(threshold).into());
        }
        if let Some(&score) = scores0.iter().find(|score| !score.is_finite()) {
            return Err(TransformError::InvalidScoreValue(score).into());
        }
        let target_sizes = [
            ImageSize::new(target_sizes[0].height, target_sizes[0].width)?,
            ImageSize::new(target_sizes[1].height, target_sizes[1].width)?,
        ];

        let keypoints0 = restore_integer_keypoints(normalized_keypoints0, target_sizes[0])?;
        let keypoints1 = restore_integer_keypoints(normalized_keypoints1, target_sizes[1])?;
        let mut matched_keypoints0 = Vec::new();
        let mut matched_keypoints1 = Vec::new();
        let mut matching_scores = Vec::new();
        for ((keypoint0, &match_index), &score) in keypoints0.iter().zip(matches0).zip(scores0) {
            let Ok(match_index) = usize::try_from(match_index) else {
                continue;
            };
            let Some(&keypoint1) = keypoints1.get(match_index) else {
                continue;
            };
            if score > threshold {
                matched_keypoints0.push(*keypoint0);
                matched_keypoints1.push(keypoint1);
                matching_scores.push(score);
            }
        }

        Ok(TaskVisionKeypointMatchingOutput {
            keypoints0: matched_keypoints0,
            keypoints1: matched_keypoints1,
            matching_scores,
        })
    }

    /// Restores SAM2/SAM3 mask logits directly to caller-provided image sizes.
    ///
    /// `mask_logits` are flattened as `[batch, masks_per_image, height, width]`.
    /// Unlike SAM1, SAM2 and SAM3 resize mask planes directly and do not remove
    /// an encoder-padding region first.
    ///
    /// # Errors
    ///
    /// Returns an error for non-SAM2/SAM3 presets, inconsistent batch lengths,
    /// invalid mask dimensions, or non-finite mask values.
    pub fn post_process_direct_masks(
        &self,
        mask_logits: &[f32],
        mask_size: ImageSize,
        masks_per_image: usize,
        target_sizes: &[ImageSize],
        mask_threshold: f32,
    ) -> Result<Vec<Vec<Vec<bool>>>, ImageProcessorError> {
        if !matches!(
            self.config.preset,
            TaskVisionProcessorPreset::Sam2 | TaskVisionProcessorPreset::Sam3
        ) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if target_sizes.is_empty() || masks_per_image == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let mask_pixels =
            mask_size
                .height
                .checked_mul(mask_size.width)
                .ok_or(ImageProcessorError::Transform(
                    TransformError::ImageSizeOverflow,
                ))?;
        let masks_per_sample =
            masks_per_image
                .checked_mul(mask_pixels)
                .ok_or(ImageProcessorError::Transform(
                    TransformError::ImageSizeOverflow,
                ))?;
        let expected = target_sizes.len().checked_mul(masks_per_sample).ok_or(
            ImageProcessorError::Transform(TransformError::ImageSizeOverflow),
        )?;
        if mask_logits.len() != expected {
            return Err(ImageProcessorError::Transform(
                TransformError::InvalidBufferLength {
                    expected,
                    actual: mask_logits.len(),
                },
            ));
        }

        target_sizes
            .iter()
            .copied()
            .enumerate()
            .map(|(sample_index, target_size)| {
                (0..masks_per_image)
                    .map(|mask_index| {
                        let start = sample_index * masks_per_sample + mask_index * mask_pixels;
                        let depth = crate::postprocess::post_process_depth_map(
                            &mask_logits[start..start + mask_pixels],
                            mask_size,
                            RecipeDepthUnit::Unknown,
                            Some(target_size),
                        )?;
                        Ok(depth
                            .values()
                            .iter()
                            .map(|&value| value > mask_threshold)
                            .collect())
                    })
                    .collect()
            })
            .collect()
    }
}

fn restore_integer_keypoints(
    points: &[[f32; 2]],
    target_size: ImageSize,
) -> Result<Vec<CoordinatePoint>, TransformError> {
    let width = target_size.width as f32;
    let height = target_size.height as f32;
    points
        .iter()
        .map(|point| {
            CoordinatePoint::new(point[0], point[1])?;
            let scaled = CoordinatePoint::new(point[0] * width, point[1] * height)?;
            CoordinatePoint::new(scaled.x as i32 as f32, scaled.y as i32 as f32)
        })
        .collect()
}

fn vitpose_affine_frame(
    image: &ImageFrame,
    bbox: TaskVisionPoseBox,
    target_size: ImageSize,
) -> Result<ImageFrame, ImageProcessorError> {
    let aspect_ratio = target_size.width as f64 / target_size.height as f64;
    let mut box_width = f64::from(bbox.width);
    let mut box_height = f64::from(bbox.height);
    let center_x = (f64::from(bbox.top_left_x) + box_width * 0.5) as f32;
    let center_y = (f64::from(bbox.top_left_y) + box_height * 0.5) as f32;
    if box_width > aspect_ratio * box_height {
        box_height = box_width / aspect_ratio;
    } else if box_width < aspect_ratio * box_height {
        box_width = box_height * aspect_ratio;
    }

    // Transformers stores scale as float32, applies its 1.25 padding factor,
    // builds a float32 push matrix, and then inverts it in float64.
    let scale_width = ((box_width / 200.0) as f32 * 1.25) * 200.0;
    let scale_height = ((box_height / 200.0) as f32 * 1.25) * 200.0;
    let push_x = ((target_size.width - 1) as f64 / f64::from(scale_width)) as f32;
    let push_y = ((target_size.height - 1) as f64 / f64::from(scale_height)) as f32;
    let push_offset_x =
        (f64::from(push_x) * (-f64::from(center_x) + f64::from(scale_width) * 0.5)) as f32;
    let push_offset_y =
        (f64::from(push_y) * (-f64::from(center_y) + f64::from(scale_height) * 0.5)) as f32;
    let inverse_x = 1.0 / f64::from(push_x);
    let inverse_y = 1.0 / f64::from(push_y);
    let inverse_offset_x = -f64::from(push_offset_x) / f64::from(push_x);
    let inverse_offset_y = -f64::from(push_offset_y) / f64::from(push_y);

    let output_len = target_size
        .height
        .checked_mul(target_size.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(ImageProcessorError::Transform(
            TransformError::ImageSizeOverflow,
        ))?;
    let mut output = vec![0_u8; output_len];
    for target_y in 0..target_size.height {
        let source_y = inverse_y * target_y as f64 + inverse_offset_y;
        for target_x in 0..target_size.width {
            let source_x = inverse_x * target_x as f64 + inverse_offset_x;
            let output_start = (target_y * target_size.width + target_x) * 3;
            if source_x < 0.0
                || source_y < 0.0
                || source_x > (image.width() - 1) as f64
                || source_y > (image.height() - 1) as f64
            {
                continue;
            }
            let x0 = source_x.floor() as usize;
            let y0 = source_y.floor() as usize;
            let x1 = (x0 + 1).min(image.width() - 1);
            let y1 = (y0 + 1).min(image.height() - 1);
            let x_weight = source_x - x0 as f64;
            let y_weight = source_y - y0 as f64;
            for channel in 0..3 {
                let top_left = f64::from(image.data()[(y0 * image.width() + x0) * 3 + channel]);
                let top_right = f64::from(image.data()[(y0 * image.width() + x1) * 3 + channel]);
                let bottom_left = f64::from(image.data()[(y1 * image.width() + x0) * 3 + channel]);
                let bottom_right = f64::from(image.data()[(y1 * image.width() + x1) * 3 + channel]);
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                let value = top + (bottom - top) * y_weight;
                output[output_start + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(ImageFrame::new(
        target_size.width,
        target_size.height,
        PixelFormat::Rgb8,
        output,
    )?)
}

fn owlv2_antialiased_resize(
    values: &[f32],
    source: ImageSize,
    target: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    let expected = source
        .height
        .checked_mul(source.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    let sigma_y = ((source.height as f32 / target.height as f32) - 1.0).max(0.0) * 0.5;
    let sigma_x = ((source.width as f32 / target.width as f32) - 1.0).max(0.0) * 0.5;
    let filtered_y = gaussian_filter_owlv2(values, source, sigma_y, true)?;
    let filtered = gaussian_filter_owlv2(&filtered_y, source, sigma_x, false)?;
    let output_len = target
        .height
        .checked_mul(target.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut output = vec![0.0_f32; output_len];
    let scale_y = source.height as f64 / target.height as f64;
    let scale_x = source.width as f64 / target.width as f64;
    for target_y in 0..target.height {
        let source_y =
            mirror_zoom_coordinate((target_y as f64 + 0.5) * scale_y - 0.5, source.height);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(source.height - 1);
        let y_weight = (source_y - y0 as f64) as f32;
        for target_x in 0..target.width {
            let source_x =
                mirror_zoom_coordinate((target_x as f64 + 0.5) * scale_x - 0.5, source.width);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(source.width - 1);
            let x_weight = (source_x - x0 as f64) as f32;
            let output_start = (target_y * target.width + target_x) * 3;
            for channel in 0..3 {
                let top_left = filtered[(y0 * source.width + x0) * 3 + channel];
                let top_right = filtered[(y0 * source.width + x1) * 3 + channel];
                let bottom_left = filtered[(y1 * source.width + x0) * 3 + channel];
                let bottom_right = filtered[(y1 * source.width + x1) * 3 + channel];
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                output[output_start + channel] = top + (bottom - top) * y_weight;
            }
        }
    }
    Ok(output)
}

// Whole-sample symmetry for scipy.ndimage.zoom(mode="mirror", grid_mode=True).
fn mirror_zoom_coordinate(coordinate: f64, length: usize) -> f64 {
    if length <= 1 {
        return 0.0;
    }
    let last = (length - 1) as f64;
    if coordinate < 0.0 {
        -coordinate
    } else if coordinate > last {
        2.0 * last - coordinate
    } else {
        coordinate
    }
}

fn resize_f32_align_corners(
    values: &[f32],
    source: ImageSize,
    target: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    let expected = source
        .height
        .checked_mul(source.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    if source == target {
        return Ok(values.to_vec());
    }
    let output_len = target
        .height
        .checked_mul(target.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut output = vec![0.0_f32; output_len];
    let y_scale = if target.height > 1 {
        (source.height - 1) as f32 / (target.height - 1) as f32
    } else {
        0.0
    };
    let x_scale = if target.width > 1 {
        (source.width - 1) as f32 / (target.width - 1) as f32
    } else {
        0.0
    };
    for target_y in 0..target.height {
        let source_y = target_y as f32 * y_scale;
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(source.height - 1);
        let y_weight = source_y - y0 as f32;
        for target_x in 0..target.width {
            let source_x = target_x as f32 * x_scale;
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(source.width - 1);
            let x_weight = source_x - x0 as f32;
            let output_start = (target_y * target.width + target_x) * 3;
            for channel in 0..3 {
                let top_left = values[(y0 * source.width + x0) * 3 + channel];
                let top_right = values[(y0 * source.width + x1) * 3 + channel];
                let bottom_left = values[(y1 * source.width + x0) * 3 + channel];
                let bottom_right = values[(y1 * source.width + x1) * 3 + channel];
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                output[output_start + channel] = top + (bottom - top) * y_weight;
            }
        }
    }
    Ok(output)
}

fn float_hwc_to_tensor(
    values: Vec<f32>,
    size: ImageSize,
    output_layout: ImageLayout,
) -> Result<Tensor, ImageProcessorError> {
    let (data, shape, layout) = match output_layout {
        ImageLayout::HeightWidthChannels => {
            (values, vec![1, size.height, size.width, 3], Layout::NHWC)
        }
        ImageLayout::ChannelsHeightWidth => {
            let pixels = size
                .height
                .checked_mul(size.width)
                .ok_or(TransformError::ImageSizeOverflow)?;
            let mut channels_first = vec![0.0_f32; pixels * 3];
            for (pixel_index, pixel) in values.chunks_exact(3).enumerate() {
                for channel in 0..3 {
                    channels_first[channel * pixels + pixel_index] = pixel[channel];
                }
            }
            (
                channels_first,
                vec![1, 3, size.height, size.width],
                Layout::NCHW,
            )
        }
    };
    Ok(Tensor::new(TensorData::F32(data), shape, layout)?
        .with_leading_axis(TensorLeadingAxis::Batch)?)
}

fn gaussian_filter_owlv2(
    values: &[f32],
    size: ImageSize,
    sigma: f32,
    vertical: bool,
) -> Result<Vec<f32>, TransformError> {
    if sigma <= f32::EPSILON {
        return Ok(values.to_vec());
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let mut weights = (-radius..=radius)
        .map(|offset| {
            let offset = offset as f32;
            (-0.5 * offset * offset / (sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let weight_sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    let expected = size
        .height
        .checked_mul(size.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    let mut output = vec![0.0_f32; expected];
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..3 {
                let mut value = 0.0_f32;
                for (weight_index, &weight) in weights.iter().enumerate() {
                    let offset = weight_index as isize - radius;
                    let (source_y, source_x) = if vertical {
                        (mirror_index(y as isize + offset, size.height), x)
                    } else {
                        (y, mirror_index(x as isize + offset, size.width))
                    };
                    value += values[(source_y * size.width + source_x) * 3 + channel] * weight;
                }
                output[(y * size.width + x) * 3 + channel] = value;
            }
        }
    }
    Ok(output)
}

fn mirror_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let upper = length as isize - 1;
    while index < 0 || index > upper {
        index = if index < 0 { -index } else { 2 * upper - index };
    }
    index as usize
}

fn fixed_size(height: usize, width: usize) -> TaskVisionResize {
    TaskVisionResize::Fixed {
        size: ImageSize { height, width },
    }
}

fn shortest_edge(shortest_edge: usize, longest_edge: Option<usize>) -> TaskVisionResize {
    TaskVisionResize::ShortestEdge {
        shortest_edge,
        longest_edge,
        multiple: None,
    }
}

fn append_padding_recipe_stage(stages: &mut Vec<ProcessorRecipeStage>, padding: TaskVisionPadding) {
    match padding {
        TaskVisionPadding::None
        | TaskVisionPadding::SquareBottomRight
        | TaskVisionPadding::ZoeReflect => {}
        TaskVisionPadding::BottomRightToMultiple { multiple }
        | TaskVisionPadding::CenterToMultiple { multiple } => {
            stages.push(ProcessorRecipeStage::PadToMultiple {
                multiples: ImageSize {
                    height: multiple,
                    width: multiple,
                },
                fill: CanvasFill::Constant(vec![0]),
            });
        }
    }
}

fn task_postprocess_descriptors(
    config: &TaskVisionImageProcessorConfig,
) -> Result<Vec<ProcessorRecipePostprocess>, RecipeError> {
    let preset = config.preset;
    Ok(match preset.family() {
        TaskVisionProcessorFamily::DetectionGrounding => {
            if preset == TaskVisionProcessorPreset::PpOcrV5ServerDet {
                return Ok(vec![ProcessorRecipePostprocess::DenseMap(
                    RecipeDenseMapPostprocess::new(
                        RecipeDenseMapTask::Other,
                        "last_hidden_state",
                        Some(RecipeImageSizeSource::CallerProvided),
                        true,
                    )
                    .with_channels(1),
                )]);
            }
            let threshold = if matches!(
                preset,
                TaskVisionProcessorPreset::GroundingDino
                    | TaskVisionProcessorPreset::OwlVit
                    | TaskVisionProcessorPreset::Owlv2
            ) {
                0.1
            } else {
                0.5
            };
            let mut descriptor = RecipeObjectDetectionPostprocess::new(
                "logits",
                "pred_boxes",
                None,
                threshold,
                RecipeImageSizeSource::CallerProvided,
            );
            descriptor = match preset {
                TaskVisionProcessorPreset::ConditionalDetr
                | TaskVisionProcessorPreset::DeformableDetr => descriptor
                    .with_score_mode(RecipeDetectionScoreMode::SigmoidTopK)
                    .with_top_k(100),
                TaskVisionProcessorPreset::GroundingDino
                | TaskVisionProcessorPreset::OwlVit
                | TaskVisionProcessorPreset::Owlv2 => {
                    descriptor.with_score_mode(RecipeDetectionScoreMode::SigmoidBestPerQuery)
                }
                TaskVisionProcessorPreset::PpDocLayoutV2
                | TaskVisionProcessorPreset::PpDocLayoutV3
                | TaskVisionProcessorPreset::RtDetr
                | TaskVisionProcessorPreset::RfDetr => {
                    descriptor.with_score_mode(RecipeDetectionScoreMode::SigmoidTopK)
                }
                _ => descriptor,
            };
            vec![ProcessorRecipePostprocess::ObjectDetection(descriptor)]
        }
        TaskVisionProcessorFamily::Segmentation => match preset {
            TaskVisionProcessorPreset::Sam2 | TaskVisionProcessorPreset::Sam3 => {
                vec![ProcessorRecipePostprocess::DenseMap(
                    RecipeDenseMapPostprocess::new(
                        RecipeDenseMapTask::Other,
                        "pred_masks",
                        Some(RecipeImageSizeSource::CallerProvided),
                        true,
                    ),
                )]
            }
            TaskVisionProcessorPreset::VitMatte => vec![ProcessorRecipePostprocess::DenseMap(
                RecipeDenseMapPostprocess::new(
                    RecipeDenseMapTask::Other,
                    "alpha",
                    Some(RecipeImageSizeSource::CallerProvided),
                    true,
                )
                .with_channels(1),
            )],
            TaskVisionProcessorPreset::Segformer | TaskVisionProcessorPreset::Sapiens2 => {
                vec![ProcessorRecipePostprocess::DenseMap(
                    RecipeDenseMapPostprocess::new(
                        RecipeDenseMapTask::Other,
                        "logits",
                        Some(RecipeImageSizeSource::CallerProvided),
                        true,
                    ),
                )]
            }
            TaskVisionProcessorPreset::SegGpt => vec![ProcessorRecipePostprocess::DenseMap(
                RecipeDenseMapPostprocess::new(
                    RecipeDenseMapTask::Other,
                    "pred_masks",
                    Some(RecipeImageSizeSource::CallerProvided),
                    true,
                )
                .with_channels(3),
            )],
            _ => {
                let strategy = match preset {
                    TaskVisionProcessorPreset::MaskFormer => RecipeSegmentationStrategy::MaskFormer,
                    TaskVisionProcessorPreset::Mask2Former => {
                        RecipeSegmentationStrategy::Mask2Former
                    }
                    TaskVisionProcessorPreset::OneFormer => RecipeSegmentationStrategy::OneFormer,
                    TaskVisionProcessorPreset::Eomt => {
                        let TaskVisionResize::ShortestEdge {
                            shortest_edge,
                            longest_edge,
                            ..
                        } = config.resize
                        else {
                            return Err(RecipeError::UnsupportedGenericStage {
                                stage: "eomt_postprocess_geometry",
                            });
                        };
                        RecipeSegmentationStrategy::Eomt {
                            size: ImageSize {
                                height: shortest_edge,
                                width: longest_edge.unwrap_or(shortest_edge),
                            },
                        }
                    }
                    _ => {
                        return Err(RecipeError::UnsupportedGenericStage {
                            stage: "query_mask_postprocess_strategy",
                        });
                    }
                };
                let mut descriptor = RecipeSegmentationPostprocess::new(
                    RecipeSegmentationTask::Semantic,
                    "class_queries_logits",
                    "masks_queries_logits",
                    None,
                )
                .with_strategy(strategy);
                if preset == TaskVisionProcessorPreset::Eomt {
                    descriptor.options.score_threshold = 0.8;
                }
                descriptor.options.target_size_source = Some(RecipeImageSizeSource::CallerProvided);
                vec![ProcessorRecipePostprocess::Segmentation(descriptor)]
            }
        },
        TaskVisionProcessorFamily::DepthGeometry => {
            let unit = match preset {
                TaskVisionProcessorPreset::DepthPro
                | TaskVisionProcessorPreset::PromptDepthAnything
                | TaskVisionProcessorPreset::ZoeDepth => RecipeDepthUnit::Meters,
                _ => RecipeDepthUnit::Relative,
            };
            vec![ProcessorRecipePostprocess::Depth(
                RecipeDepthPostprocess::new(
                    "predicted_depth",
                    unit,
                    Some(RecipeImageSizeSource::CallerProvided),
                    true,
                ),
            )]
        }
        TaskVisionProcessorFamily::KeypointMatchingPose => {
            let task = match preset {
                TaskVisionProcessorPreset::EfficientLoFtr
                | TaskVisionProcessorPreset::LightGlue
                | TaskVisionProcessorPreset::SuperGlue => RecipeCoordinateTask::Matches,
                TaskVisionProcessorPreset::VitPose => RecipeCoordinateTask::Pose,
                _ => RecipeCoordinateTask::Keypoints,
            };
            vec![ProcessorRecipePostprocess::Coordinates(
                RecipeCoordinatePostprocess::new(
                    task,
                    "keypoints",
                    Some("scores".to_owned()),
                    RecipeImageSizeSource::CallerProvided,
                ),
            )]
        }
    })
}

fn validate_task_vision_config(
    config: &TaskVisionImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    match config.resize {
        TaskVisionResize::None => {}
        TaskVisionResize::Fixed { size } => {
            ImageSize::new(size.height, size.width)?;
        }
        TaskVisionResize::ShortestEdge {
            shortest_edge,
            longest_edge,
            multiple,
        } => {
            if shortest_edge == 0 || longest_edge == Some(0) {
                return Err(ImageProcessorError::Transform(
                    TransformError::InvalidScaleFactor(shortest_edge),
                ));
            }
            if let Some(multiple) = multiple {
                validate_multiple(multiple)?;
            }
        }
        TaskVisionResize::RoundDownToMultiple { multiple } => validate_multiple(multiple)?,
        TaskVisionResize::PaddleDetection { limit_side_len, .. } => {
            validate_multiple(limit_side_len)?;
        }
    }
    validate_padding(config.pre_resize_padding)?;
    validate_padding(config.post_resize_padding)?;
    Ok(())
}

fn validate_padding(padding: TaskVisionPadding) -> Result<(), ImageProcessorError> {
    match padding {
        TaskVisionPadding::BottomRightToMultiple { multiple }
        | TaskVisionPadding::CenterToMultiple { multiple } => validate_multiple(multiple),
        _ => Ok(()),
    }
}

fn validate_multiple(multiple: usize) -> Result<(), ImageProcessorError> {
    if multiple == 0 {
        return Err(ImageProcessorError::Transform(
            TransformError::InvalidScaleFactor(multiple),
        ));
    }
    Ok(())
}

fn apply_task_resize(
    frame: ImageFrame,
    config: &TaskVisionImageProcessorConfig,
) -> Result<ImageFrame, ImageProcessorError> {
    let source = ImageSize::new(frame.height(), frame.width())?;
    let target = match config.resize {
        TaskVisionResize::None => return Ok(frame),
        TaskVisionResize::Fixed { size } => size,
        TaskVisionResize::ShortestEdge {
            shortest_edge,
            longest_edge,
            multiple,
        } => {
            if config.preset == TaskVisionProcessorPreset::Yolos {
                let target = super::detection::shortest_edge_resize_output_size(
                    source,
                    ShortestEdgeResizeConfig {
                        shortest_edge,
                        longest_edge,
                    },
                )?;
                let multiple = multiple.unwrap_or(16);
                let target = ImageSize::new(
                    target.height / multiple * multiple,
                    target.width / multiple * multiple,
                )?;
                return resize_frame_with_decision(
                    &frame,
                    target,
                    ResizeDecision::new(config.resample, config.resize_parity)?,
                    ResizeMode::Default,
                )
                .map_err(ImageProcessorError::Transform);
            }
            let mut target = shortest_edge_resize_size(source, shortest_edge, longest_edge)?;
            if let Some(multiple) = multiple {
                target = ImageSize::new(
                    round_up(target.height, multiple)?,
                    round_up(target.width, multiple)?,
                )?;
            }
            target
        }
        TaskVisionResize::RoundDownToMultiple { multiple } => ImageSize::new(
            source.height / multiple * multiple,
            source.width / multiple * multiple,
        )?,
        TaskVisionResize::PaddleDetection {
            limit_side_len,
            limit_type,
            max_side_limit,
        } => paddle_detection_size(source, limit_side_len, limit_type, max_side_limit)?,
    };
    if target == source {
        return Ok(frame);
    }
    resize_frame_with_decision(
        &frame,
        target,
        ResizeDecision::new(config.resample, config.resize_parity)?,
        ResizeMode::Default,
    )
    .map_err(ImageProcessorError::Transform)
}

fn paddle_detection_size(
    source: ImageSize,
    limit_side_len: usize,
    limit_type: PaddleDetectionLimit,
    max_side_limit: Option<usize>,
) -> Result<ImageSize, ImageProcessorError> {
    let height = source.height as f64;
    let width = source.width as f64;
    let min_edge = source.height.min(source.width);
    let max_edge = source.height.max(source.width);
    let mut ratio = match limit_type {
        PaddleDetectionLimit::Max if max_edge > limit_side_len => {
            limit_side_len as f64 / max_edge as f64
        }
        PaddleDetectionLimit::Min if min_edge < limit_side_len => {
            limit_side_len as f64 / min_edge as f64
        }
        PaddleDetectionLimit::ResizeLong => limit_side_len as f64 / max_edge as f64,
        PaddleDetectionLimit::Max | PaddleDetectionLimit::Min => 1.0,
    };
    let mut resize_height = (height * ratio) as usize;
    let mut resize_width = (width * ratio) as usize;
    if let Some(max_side_limit) = max_side_limit {
        let current_max = resize_height.max(resize_width);
        if current_max > max_side_limit {
            ratio = max_side_limit as f64 / current_max as f64;
            resize_height = (resize_height as f64 * ratio) as usize;
            resize_width = (resize_width as f64 * ratio) as usize;
        }
    }
    let height = ((resize_height as f64 / 32.0).round() as usize * 32).max(32);
    let width = ((resize_width as f64 / 32.0).round() as usize * 32).max(32);
    Ok(ImageSize::new(height, width)?)
}

fn apply_task_padding(
    frame: ImageFrame,
    padding: TaskVisionPadding,
) -> Result<ImageFrame, ImageProcessorError> {
    let size = ImageSize::new(frame.height(), frame.width())?;
    let padding = match padding {
        TaskVisionPadding::None => return Ok(frame),
        TaskVisionPadding::BottomRightToMultiple { multiple } => {
            let target = ImageSize::new(
                round_up(size.height, multiple)?,
                round_up(size.width, multiple)?,
            )?;
            Padding::new(0, target.width - size.width, target.height - size.height, 0)
        }
        TaskVisionPadding::CenterToMultiple { multiple } => {
            let target = ImageSize::new(
                round_up(size.height, multiple)?,
                round_up(size.width, multiple)?,
            )?;
            let height = target.height - size.height;
            let width = target.width - size.width;
            Padding::new(
                height / 2,
                width - width / 2,
                height - height / 2,
                width / 2,
            )
        }
        TaskVisionPadding::SquareBottomRight => {
            let edge = size.height.max(size.width);
            Padding::new(0, edge - size.width, edge - size.height, 0)
        }
        TaskVisionPadding::ZoeReflect => {
            let vertical = ((size.height as f64 / 2.0).sqrt() * 3.0) as usize;
            let horizontal = ((size.width as f64 / 2.0).sqrt() * 3.0) as usize;
            return pad_frame_with_canvas_fill(
                &frame,
                Padding::new(vertical, horizontal, vertical, horizontal),
                CanvasFill::ReflectImage,
            )
            .map_err(ImageProcessorError::Transform);
        }
    };
    pad_frame(&frame, padding, &[0]).map_err(ImageProcessorError::Transform)
}

fn round_up(value: usize, multiple: usize) -> Result<usize, ImageProcessorError> {
    validate_multiple(multiple)?;
    let remainder = value % multiple;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(multiple - remainder)
        .ok_or(ImageProcessorError::Transform(
            TransformError::ImageSizeOverflow,
        ))
}

fn reverse_rgb_channels(frame: ImageFrame) -> Result<ImageFrame, ImageProcessorError> {
    let width = frame.width();
    let height = frame.height();
    let (_, _, _, mut data, timing) = frame.into_parts();
    for pixel in data.chunks_exact_mut(3) {
        pixel.swap(0, 2);
    }
    Ok(ImageFrame::new(width, height, PixelFormat::Rgb8, data)?.with_timing(timing))
}

fn full_task_pixel_mask(size: ImageSize) -> Result<Tensor, ImageProcessorError> {
    let elements = size
        .height
        .checked_mul(size.width)
        .ok_or(ImageProcessorError::Tensor(
            TensorError::ShapeElementCountOverflow,
        ))?;
    Tensor::new(
        TensorData::Bool(vec![1; elements]),
        vec![1, 1, size.height, size.width],
        Layout::NCHW,
    )
    .map_err(ImageProcessorError::Tensor)
}

fn prepare_auxiliary_frame(
    config: &TaskVisionImageProcessorConfig,
    frame: &ImageFrame,
) -> Result<ImageFrame, ImageProcessorError> {
    let mut frame = apply_task_padding(frame.clone(), config.pre_resize_padding)?;
    frame = apply_task_resize(frame, config)?;
    apply_task_padding(frame, config.post_resize_padding)
}

fn concatenate_matte_channels(
    image: Tensor,
    trimap: Tensor,
    layout: ImageLayout,
) -> Result<Tensor, ImageProcessorError> {
    let (image_data, image_shape, image_layout, _) = image.into_parts();
    let (trimap_data, trimap_shape, trimap_layout, _) = trimap.into_parts();
    if image_layout != trimap_layout
        || image_shape[image_layout.height_axis()] != trimap_shape[trimap_layout.height_axis()]
        || image_shape[image_layout.width_axis()] != trimap_shape[trimap_layout.width_axis()]
    {
        return Err(ImageProcessorError::IncompatibleBatchShapes);
    }
    let image_values = image_data.to_vec::<f32>();
    let trimap_values = trimap_data.to_vec::<f32>();
    let height = image_shape[image_layout.height_axis()];
    let width = image_shape[image_layout.width_axis()];
    let mut values = Vec::with_capacity(image_values.len() + trimap_values.len());
    let shape = match layout {
        ImageLayout::ChannelsHeightWidth => {
            values.extend(image_values);
            values.extend(trimap_values);
            vec![1, 4, height, width]
        }
        ImageLayout::HeightWidthChannels => {
            for (rgb, &trimap) in image_values.chunks_exact(3).zip(&trimap_values) {
                values.extend_from_slice(rgb);
                values.push(trimap);
            }
            vec![1, height, width, 4]
        }
    };
    Tensor::new(TensorData::F32(values), shape, image_layout)
        .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
        .map_err(ImageProcessorError::Tensor)
}

#[cfg(test)]
mod owlv2_resize_tests {
    use super::*;

    #[test]
    fn enlargement_mirrors_both_edges_instead_of_clamping() {
        let source = ImageSize::new(2, 2).expect("source");
        let target = ImageSize::new(4, 4).expect("target");
        let pixels: Vec<f32> = [0.0, 4.0, 8.0, 12.0]
            .into_iter()
            .flat_map(|v| [v; 3])
            .collect();
        let actual = owlv2_antialiased_resize(&pixels, source, target).expect("resize");
        // scipy.ndimage.zoom(pixels, (2, 2, 1), order=1, mode="mirror", grid_mode=True).
        let expected = [
            3., 3., 5., 5., 3., 3., 5., 5., 7., 7., 9., 9., 7., 7., 9., 9.,
        ];
        for (pixel, expected) in actual.chunks_exact(3).zip(expected) {
            assert_eq!(pixel, [expected; 3]);
        }
        let singleton =
            owlv2_antialiased_resize(&[7.; 3], ImageSize::new(1, 1).expect("source"), target)
                .expect("singleton resize");
        assert_eq!(singleton, vec![7.; 48]);
    }
}
