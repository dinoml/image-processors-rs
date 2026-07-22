use super::*;
use thiserror::Error;

/// Errors returned while validating or lowering processor recipes.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum RecipeError {
    /// Recipe id was empty or only whitespace.
    #[error("processor recipe id cannot be empty")]
    EmptyId,
    /// Recipe did not contain any stages.
    #[error("processor recipe must contain at least one stage")]
    EmptyStages,
    /// A postprocess-only recipe was created without descriptors.
    #[error("postprocess-only recipe must contain at least one postprocess descriptor")]
    EmptyPostprocessDescriptors,
    /// A postprocess-only recipe cannot lower to a generic image processor.
    #[error("postprocess-only recipe cannot lower to the generic image processor config")]
    PostprocessOnlyCannotLower,
    /// A generic stage appeared more than once.
    #[error("processor recipe contains duplicate {stage} stage")]
    DuplicateStage {
        /// Duplicate stage name.
        stage: &'static str,
    },
    /// A recipe stage cannot lower to the generic image processor config.
    #[error("recipe stage {stage} cannot lower to the generic image processor config")]
    UnsupportedGenericStage {
        /// Unsupported stage name.
        stage: &'static str,
    },
    /// A resize stage used invalid dimensions.
    #[error("resize stage target must be positive, got {size:?}")]
    InvalidResizeTarget {
        /// Invalid target size.
        size: ImageSize,
    },
    /// A resize stage used an invalid filter and parity combination.
    #[error("resize stage {stage_index} has invalid resize parity: {source}")]
    InvalidResizeParity {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Underlying transform validation error.
        source: TransformError,
    },
    /// A smart-resize stage used invalid pixel limits.
    #[error("smart-resize stage {stage_index} has invalid limits: {source}")]
    InvalidSmartResizeLimits {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Underlying transform validation error.
        source: TransformError,
    },
    /// A transform-backed recipe stage used invalid parameters.
    #[error("transform recipe stage {stage_index} ({stage}) is invalid: {source}")]
    InvalidTransformStage {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Stage name.
        stage: &'static str,
        /// Underlying transform validation error.
        source: TransformError,
    },
    /// Rescale factor was not finite.
    #[error("rescale factor at stage {stage_index} must be finite, got {value}")]
    NonFiniteRescaleFactor {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Invalid factor value.
        value: f32,
    },
    /// Normalization statistics were empty.
    #[error("{field} normalization stats at stage {stage_index} cannot be empty")]
    EmptyNormalizationStats {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Field name.
        field: &'static str,
    },
    /// Normalization statistic was not finite.
    #[error("{field} normalization stat at stage {stage_index}, index {index} must be finite, got {value}")]
    NonFiniteNormalizationStat {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Field name.
        field: &'static str,
        /// Statistic index.
        index: usize,
        /// Invalid value.
        value: f32,
    },
    /// Normalization standard deviation was not positive.
    #[error(
        "normalization std at stage {stage_index}, index {index} must be positive, got {value}"
    )]
    NonPositiveNormalizationStd {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Statistic index.
        index: usize,
        /// Invalid value.
        value: f32,
    },
    /// Normalization statistics did not match a known pixel format.
    #[error("normalization stats at stage {stage_index} must have length 1 or channel count {channels}, got {actual}")]
    InvalidNormalizationStats {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Expected channel count.
        channels: usize,
        /// Actual statistics length.
        actual: usize,
    },
    /// Output layout is unsupported by generic preprocessing.
    #[error("unsupported recipe output layout {0:?}")]
    UnsupportedOutputLayout(Layout),
    /// Patch-flattening source layout is unsupported.
    #[error("unsupported patch-flatten source layout {layout:?} at stage {stage_index}")]
    UnsupportedPatchSourceLayout {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Unsupported source tensor layout.
        layout: Layout,
    },
    /// Patch dimension was zero.
    #[error("{field} at patch-flatten stage {stage_index} must be positive, got {value}")]
    InvalidPatchDimension {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Field name.
        field: &'static str,
        /// Invalid dimension value.
        value: usize,
    },
    /// Patch geometry arithmetic overflowed.
    #[error("patch geometry at stage {stage_index} overflowed")]
    PatchGeometryOverflow {
        /// Stage index in the recipe.
        stage_index: usize,
    },
    /// Frame sampling stride was zero.
    #[error("frame-sampling stage {stage_index} stride must be positive, got {stride}")]
    InvalidFrameSamplingStride {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Invalid stride.
        stride: usize,
    },
    /// Frame sampling limit was zero.
    #[error("frame-sampling stage {stage_index} max_frames must be positive when configured, got {max_frames}")]
    InvalidFrameSamplingLimit {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Invalid frame limit.
        max_frames: usize,
    },
    /// Smart-resize factor did not match patch geometry.
    #[error("patch-flatten stage {stage_index} expects resize factor {resize_factor} to equal patch_size * merge_size ({patch_size} * {merge_size})")]
    InvalidPatchGeometry {
        /// Stage index in the recipe.
        stage_index: usize,
        /// Smart-resize factor.
        resize_factor: usize,
        /// Spatial patch size.
        patch_size: usize,
        /// Spatial merge size.
        merge_size: usize,
    },
    /// A postprocess descriptor referenced an empty model output name.
    #[error("postprocess descriptor {postprocess_index} field {field} cannot be empty")]
    EmptyPostprocessOutputName {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Field name.
        field: &'static str,
    },
    /// A postprocess descriptor referenced an empty metadata source name.
    #[error(
        "postprocess descriptor {postprocess_index} field {field} source name cannot be empty"
    )]
    EmptyPostprocessSourceName {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Field name.
        field: &'static str,
    },
    /// A postprocess descriptor declared an empty target name.
    #[error(
        "postprocess descriptor {postprocess_index} target name {target_index} cannot be empty"
    )]
    EmptyPostprocessTargetName {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Target-name index in the descriptor.
        target_index: usize,
    },
    /// A postprocess descriptor used an invalid class count.
    #[error(
        "postprocess descriptor {postprocess_index} class count must be at least 2, got {value}"
    )]
    InvalidPostprocessClassCount {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Invalid class count.
        value: usize,
    },
    /// A postprocess descriptor used an invalid query count.
    #[error(
        "postprocess descriptor {postprocess_index} query count must be positive, got {value}"
    )]
    InvalidPostprocessQueryCount {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Invalid query count.
        value: usize,
    },
    /// A postprocess descriptor used an invalid mask count.
    #[error("postprocess descriptor {postprocess_index} mask count must be positive, got {value}")]
    InvalidPostprocessMaskCount {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Invalid mask count.
        value: usize,
    },
    /// A postprocess descriptor used an invalid channel count.
    #[error(
        "postprocess descriptor {postprocess_index} channel count must be positive, got {value}"
    )]
    InvalidPostprocessChannelCount {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Invalid channel count.
        value: usize,
    },
    /// A postprocess descriptor used an invalid threshold.
    #[error(
        "postprocess descriptor {postprocess_index} field {field} has invalid threshold {value}"
    )]
    InvalidPostprocessThreshold {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Threshold field name.
        field: &'static str,
        /// Invalid threshold value.
        value: f32,
    },
    /// A postprocess descriptor used an invalid image size.
    #[error(
        "postprocess descriptor {postprocess_index} field {field} has invalid image size: {source}"
    )]
    InvalidPostprocessImageSize {
        /// Postprocess descriptor index in the recipe.
        postprocess_index: usize,
        /// Image-size field name.
        field: &'static str,
        /// Underlying transform validation error.
        source: TransformError,
    },
}
