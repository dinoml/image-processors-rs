//! Reusable processor recipe descriptions.
//!
//! Recipes describe common preprocessing stages without mirroring upstream
//! Python processor classes. They are validated data that can be serialized for
//! fixtures or catalog records and lowered into [`ImageProcessorConfig`] when
//! the stage set is representable by the generic image processor.

use serde::{Deserialize, Deserializer, Serialize};

use crate::image::{BatchExecution, ImageProcessorConfig};
use crate::media::{FrameSampling, ImageDecodeBackend, PixelFormat};
use crate::tensor::{DType, Layout, Tensor, TensorData, TensorError, TensorLeadingAxis};
mod error;
mod patch;
mod postprocess_descriptors;
mod stages;
mod validation;

pub use error::RecipeError;
pub use patch::{RecipePatchError, RecipePatchStage};
pub use postprocess_descriptors::*;
pub use stages::*;

use validation::{
    generic_output_layout, validate_id, validate_output, validate_postprocess, validate_stages,
};

use crate::transforms::{
    area_resize_plan, aspect_ratio_crop_plan, centered_padding, fit_inside_size,
    longest_edge_resize_size, multiple_of_resize_plan, pad_to_multiple_plan, padded_image_size,
    patch_grid_plan, resize_center_crop_plan, resize_fill_plan, select_aspect_ratio_bucket,
    shortest_edge_resize_size, should_rotate_to_match_orientation, split_image_encoder_size,
    split_image_plan, temporal_repeat_last_plan, tiled_canvas_plan, validate_image_crop_box,
    validate_image_size_constraints, validate_padding_fill, video_size_bucket_plan,
    AspectRatioCropOptions, CanvasFill, ImageCropBox, ImageSize, ImageSizeConstraints,
    OverlayPosition, Padding, ResizeDecision, ResizeFilter, ResizeLimits, ResizeMode, ResizeParity,
    ResizeRounding, RgbCanvasFill, TemporalRepeatLastPlan, TransformError, VideoSizeBucket,
};

/// A validated reusable processor stage recipe.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProcessorRecipe {
    id: String,
    input: ProcessorRecipeInput,
    stages: Vec<ProcessorRecipeStage>,
    output: ProcessorRecipeOutput,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    postprocess: Vec<ProcessorRecipePostprocess>,
}

impl ProcessorRecipe {
    /// Creates a validated processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is empty, the stage list is empty, a stage
    /// has invalid dimensions or numeric parameters, or the output layout is
    /// unsupported by the generic image processor.
    pub fn new(
        id: impl Into<String>,
        input: ProcessorRecipeInput,
        stages: Vec<ProcessorRecipeStage>,
        output: ProcessorRecipeOutput,
    ) -> Result<Self, RecipeError> {
        Self::new_with_postprocess(id, input, stages, output, Vec::new())
    }

    /// Creates a validated processor recipe with task postprocessing descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is empty, a preprocessing stage is invalid,
    /// the output layout is unsupported, or a postprocess descriptor contains
    /// invalid output names, dimensions, counts, or thresholds.
    pub fn new_with_postprocess(
        id: impl Into<String>,
        input: ProcessorRecipeInput,
        stages: Vec<ProcessorRecipeStage>,
        output: ProcessorRecipeOutput,
        postprocess: Vec<ProcessorRecipePostprocess>,
    ) -> Result<Self, RecipeError> {
        let id = id.into();
        validate_id(&id)?;
        validate_stages(&stages)?;
        validate_output(output, &stages)?;
        validate_postprocess(&postprocess)?;

        Ok(Self {
            id,
            input,
            stages,
            output,
            postprocess,
        })
    }

    /// Creates a validated recipe that only describes postprocessing.
    ///
    /// Postprocess-only recipes do not describe source loading or tensor
    /// preprocessing stages, so they cannot be lowered into an
    /// [`ImageProcessorConfig`]. Their `output` contract describes the primary
    /// postprocessed tensor or task output shape expected from the recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is empty, no postprocess descriptors are
    /// supplied, or a postprocess descriptor contains invalid output names,
    /// dimensions, counts, or thresholds.
    pub fn new_postprocess_only(
        id: impl Into<String>,
        input: ProcessorRecipeInput,
        output: ProcessorRecipeOutput,
        postprocess: Vec<ProcessorRecipePostprocess>,
    ) -> Result<Self, RecipeError> {
        let id = id.into();
        validate_id(&id)?;
        if postprocess.is_empty() {
            return Err(RecipeError::EmptyPostprocessDescriptors);
        }
        validate_postprocess(&postprocess)?;

        Ok(Self {
            id,
            input,
            stages: Vec::new(),
            output,
            postprocess,
        })
    }

    /// Creates a postprocess-only LogC3 HDR video tensor recipe.
    ///
    /// The descriptor expects the model output named by `output` to contain a
    /// decoded `BFCHW` or `BFHWC` video tensor and restores it to linear HDR
    /// `BFHWC` values through [`RecipeVideoTransferFunction::LogC3`].
    ///
    /// # Errors
    ///
    /// Returns an error when the id or model output name is empty.
    pub fn logc3_hdr_video_postprocess(
        id: impl Into<String>,
        output: impl Into<String>,
    ) -> Result<Self, RecipeError> {
        Self::new_postprocess_only(
            id,
            ProcessorRecipeInput::default(),
            ProcessorRecipeOutput::batch(Layout::BFHWC),
            vec![ProcessorRecipePostprocess::VideoTensor(
                RecipeVideoTensorPostprocess::new(output, RecipeVideoTransferFunction::LogC3),
            )],
        )
    }

    /// Returns this recipe's stable identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the recipe input policy.
    pub fn input(&self) -> &ProcessorRecipeInput {
        &self.input
    }

    /// Returns the ordered preprocessing stages.
    pub fn stages(&self) -> &[ProcessorRecipeStage] {
        &self.stages
    }

    /// Returns the tensor output contract.
    pub fn output(&self) -> ProcessorRecipeOutput {
        self.output
    }

    /// Returns task-specific postprocessing descriptors.
    pub fn postprocess(&self) -> &[ProcessorRecipePostprocess] {
        &self.postprocess
    }

    /// Returns true when this recipe only describes postprocessing.
    pub fn is_postprocess_only(&self) -> bool {
        self.stages.is_empty()
    }

    /// Replaces task-specific postprocessing descriptors on this recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when any descriptor contains invalid output names,
    /// dimensions, counts, or thresholds.
    pub fn with_postprocess(
        mut self,
        postprocess: Vec<ProcessorRecipePostprocess>,
    ) -> Result<Self, RecipeError> {
        validate_postprocess(&postprocess)?;
        self.postprocess = postprocess;
        Ok(self)
    }

    /// Consumes this recipe into its parts.
    pub fn into_parts(
        self,
    ) -> (
        String,
        ProcessorRecipeInput,
        Vec<ProcessorRecipeStage>,
        ProcessorRecipeOutput,
        Vec<ProcessorRecipePostprocess>,
    ) {
        (
            self.id,
            self.input,
            self.stages,
            self.output,
            self.postprocess,
        )
    }

    /// Lowers this recipe into the generic image processor configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the recipe uses duplicate generic stages or a
    /// stage combination that cannot be represented by [`ImageProcessorConfig`].
    pub fn to_image_processor_config(&self) -> Result<ImageProcessorConfig, RecipeError> {
        if self.is_postprocess_only() {
            return Err(RecipeError::PostprocessOnlyCannotLower);
        }

        let mut config = ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Resampling,
            decode_backend: self.input.decode_backend,
            batch_execution: self.input.batch_execution,
            pixel_format: None,
            do_rescale: false,
            rescale_factor: 1.0,
            do_normalize: false,
            image_mean: Vec::new(),
            image_std: Vec::new(),
            do_binarize: false,
            output_layout: generic_output_layout(self.output, &self.stages),
        };

        let mut resize_seen = false;
        let mut pixel_format_seen = false;
        let mut rescale_seen = false;
        let mut normalize_seen = false;
        let mut binarize_seen = false;
        let mut patch_flatten_seen = false;

        for stage in &self.stages {
            match stage {
                ProcessorRecipeStage::ConvertPixelFormat { format } => {
                    if pixel_format_seen {
                        return Err(RecipeError::DuplicateStage {
                            stage: "convert_pixel_format",
                        });
                    }
                    pixel_format_seen = true;
                    config.pixel_format = Some(*format);
                }
                ProcessorRecipeStage::Resize { resize } => {
                    if resize_seen {
                        return Err(RecipeError::DuplicateStage { stage: "resize" });
                    }
                    resize_seen = true;
                    config.do_resize = true;
                    match resize.target {
                        RecipeResizeTarget::Fixed { size } => {
                            config.height = Some(size.height);
                            config.width = Some(size.width);
                        }
                        RecipeResizeTarget::Dynamic | RecipeResizeTarget::SmartResize { .. } => {
                            config.height = None;
                            config.width = None;
                        }
                        RecipeResizeTarget::ShortestEdge { .. }
                        | RecipeResizeTarget::LongestEdge { .. } => {
                            return Err(RecipeError::UnsupportedGenericStage {
                                stage: resize.target.kind(),
                            });
                        }
                    }
                    config.resize_mode = resize.mode;
                    config.resample = resize.filter;
                    config.resize_parity = resize.parity;
                }
                ProcessorRecipeStage::Rescale { factor } => {
                    if rescale_seen {
                        return Err(RecipeError::DuplicateStage { stage: "rescale" });
                    }
                    rescale_seen = true;
                    config.do_rescale = true;
                    config.rescale_factor = *factor;
                }
                ProcessorRecipeStage::Normalize { mean, std } => {
                    if normalize_seen {
                        return Err(RecipeError::DuplicateStage { stage: "normalize" });
                    }
                    normalize_seen = true;
                    config.do_normalize = true;
                    config.image_mean = mean.clone();
                    config.image_std = std.clone();
                }
                ProcessorRecipeStage::Binarize => {
                    if binarize_seen {
                        return Err(RecipeError::DuplicateStage { stage: "binarize" });
                    }
                    binarize_seen = true;
                    config.do_binarize = true;
                }
                ProcessorRecipeStage::PatchFlatten { patch } => {
                    if patch_flatten_seen {
                        return Err(RecipeError::DuplicateStage {
                            stage: "patch_flatten",
                        });
                    }
                    patch_flatten_seen = true;
                    config.output_layout = patch.source_layout;
                }
                ProcessorRecipeStage::SampleFrames { .. }
                | ProcessorRecipeStage::Owlv2AntialiasedResize { .. }
                | ProcessorRecipeStage::VitPoseAffine { .. }
                | ProcessorRecipeStage::TemporalRepeatLast { .. }
                | ProcessorRecipeStage::PatchGrid { .. }
                | ProcessorRecipeStage::ImageSplit { .. }
                | ProcessorRecipeStage::AspectRatioCrops { .. }
                | ProcessorRecipeStage::TiledCanvas { .. }
                | ProcessorRecipeStage::DocumentGeometry { .. }
                | ProcessorRecipeStage::ValidateImageSize { .. }
                | ProcessorRecipeStage::SelectAspectRatioBucket { .. }
                | ProcessorRecipeStage::SelectVideoSizeBucket { .. }
                | ProcessorRecipeStage::AreaResize { .. }
                | ProcessorRecipeStage::Crop { .. }
                | ProcessorRecipeStage::ResizeCenterCrop { .. }
                | ProcessorRecipeStage::Pad { .. }
                | ProcessorRecipeStage::PadCanvas { .. }
                | ProcessorRecipeStage::PadToMultiple { .. }
                | ProcessorRecipeStage::PadSymmetricToNextMultiple { .. }
                | ProcessorRecipeStage::Overlay { .. }
                | ProcessorRecipeStage::MaskComposite
                | ProcessorRecipeStage::RoundToMultiple { .. }
                | ProcessorRecipeStage::ResizeFill { .. }
                | ProcessorRecipeStage::ResizeFillRgb { .. }
                | ProcessorRecipeStage::ConcatenateHorizontallyRgb { .. } => {
                    return Err(RecipeError::UnsupportedGenericStage {
                        stage: stage.kind(),
                    });
                }
            }
        }

        Ok(config)
    }
}

impl<'de> Deserialize<'de> for ProcessorRecipe {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = ProcessorRecipeParts::deserialize(deserializer)?;
        if raw.stages.is_empty() && !raw.postprocess.is_empty() {
            Self::new_postprocess_only(raw.id, raw.input, raw.output, raw.postprocess)
                .map_err(serde::de::Error::custom)
        } else {
            Self::new_with_postprocess(raw.id, raw.input, raw.stages, raw.output, raw.postprocess)
                .map_err(serde::de::Error::custom)
        }
    }
}

#[derive(Deserialize)]
struct ProcessorRecipeParts {
    id: String,
    input: ProcessorRecipeInput,
    stages: Vec<ProcessorRecipeStage>,
    output: ProcessorRecipeOutput,
    #[serde(default)]
    postprocess: Vec<ProcessorRecipePostprocess>,
}

/// Input and execution policy shared by recipe stages.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessorRecipeInput {
    /// Image decode backend used by source-loading entrypoints.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch execution policy used by source and frame batch entrypoints.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

/// Tensor output contract for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessorRecipeOutput {
    /// Output tensor layout.
    pub layout: Layout,
    /// Semantic meaning of the leading axis for batched outputs.
    pub leading_axis: TensorLeadingAxis,
}

impl ProcessorRecipeOutput {
    /// Returns a batch output contract for a layout.
    pub fn batch(layout: Layout) -> Self {
        Self {
            layout,
            leading_axis: TensorLeadingAxis::Batch,
        }
    }
}

#[cfg(test)]
mod tests;
