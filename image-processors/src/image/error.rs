use super::*;
use thiserror::Error;

/// Errors returned by image preprocessing.
#[derive(Debug, Error)]
pub enum ImageProcessorError {
    /// A batch or frame sequence had no images.
    #[error("empty image batch")]
    EmptyBatch,
    /// Processed batch item shapes or layouts differed.
    #[error("incompatible image batch shapes")]
    IncompatibleBatchShapes,
    /// A configured padding target could not contain a resized image.
    #[error("padding target {target_size:?} is smaller than resized image {image_size:?}")]
    InvalidPaddingTarget {
        /// Requested padded output dimensions.
        target_size: ImageSize,
        /// Resized image dimensions that must fit inside the padding target.
        image_size: ImageSize,
    },
    /// Qwen/VLM patch geometry was inconsistent.
    #[error(
        "resize factor {resize_factor} must equal patch_size * merge_size ({patch_size} * {merge_size})"
    )]
    InvalidPatchGeometry {
        /// Configured smart-resize factor.
        resize_factor: usize,
        /// Spatial patch size.
        patch_size: usize,
        /// Spatial merge size.
        merge_size: usize,
    },
    /// A Qwen/VLM resized target could not be represented as merged patches.
    #[error("target size {target_size:?} is not divisible by patch_size={patch_size} and merge_size={merge_size}")]
    InvalidPatchTarget {
        /// Resized image dimensions.
        target_size: ImageSize,
        /// Spatial patch size.
        patch_size: usize,
        /// Spatial merge size.
        merge_size: usize,
    },
    /// Per-sample Diffusers denormalization flags did not match the batch size.
    #[error("denormalize flag count must match sample count {expected}, got {actual}")]
    InvalidDenormalizeFlags {
        /// Expected flag count.
        expected: usize,
        /// Actual flag count.
        actual: usize,
    },
    /// Parallel batch processing was requested in a build without the `parallel` feature.
    #[error("parallel batch processing requires the parallel feature")]
    ParallelBatchUnavailable,
    /// The configured output layout is unsupported.
    #[error("unsupported output layout {0:?}")]
    UnsupportedLayout(Layout),
    /// A processor-family option is not supported by the Rust wrapper.
    #[error("{field} is not supported for this processor family")]
    UnsupportedProcessorOption {
        /// Unsupported option field name.
        field: &'static str,
    },
    /// The configured preprocessing cannot be represented by a borrowed view.
    #[error("zero-copy tensor view unavailable: {reason}")]
    ZeroCopyUnavailable {
        /// Reason owned preprocessing is required.
        reason: &'static str,
    },
    /// An internal tensor had an unexpected data type.
    #[error("expected tensor data type {0}")]
    ExpectedDataType(&'static str),
    /// Normalization statistics did not match the channel count.
    #[error("normalization stats must have length 1 or channel count {channels}, got {actual}")]
    InvalidNormalizationStats {
        /// Expected channel count.
        channels: usize,
        /// Actual statistics length.
        actual: usize,
    },
    /// Required normalization statistics were empty.
    #[error("{field} normalization stats cannot be empty")]
    EmptyNormalizationStats {
        /// Configuration field name.
        field: &'static str,
    },
    /// A normalization statistic was not finite.
    #[error("{field} normalization stat at index {index} must be finite, got {value}")]
    NonFiniteNormalizationStat {
        /// Configuration field name.
        field: &'static str,
        /// Statistic index.
        index: usize,
        /// Non-finite value.
        value: f32,
    },
    /// A normalization standard deviation was not positive.
    #[error("normalization std at index {index} must be positive, got {value}")]
    NonPositiveNormalizationStd {
        /// Statistic index.
        index: usize,
        /// Non-positive value.
        value: f32,
    },
    /// A normalization standard deviation was zero.
    #[error("normalization standard deviation cannot contain zero")]
    ZeroNormalizationStd,
    /// The configured rescale factor was not finite.
    #[error("rescale_factor must be finite, got {0}")]
    NonFiniteRescaleFactor(f32),
    /// A configured target dimension was zero.
    #[error("{field} must be positive when configured, got {value}")]
    InvalidConfiguredDimension {
        /// Configuration field name.
        field: &'static str,
        /// Invalid dimension value.
        value: usize,
    },
    /// Diffusers VAE latent channel count was zero.
    #[error("vae_latent_channels must be positive, got {0}")]
    InvalidVaeLatentChannels(usize),
    /// Media loading or decoding failed.
    #[error("media error: {0}")]
    Media(#[from] MediaError),
    /// Tensor construction or layout conversion failed.
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    /// Image transform failed.
    #[error("transform error: {0}")]
    Transform(#[from] TransformError),
    /// Resize kernel failed.
    #[error("resize kernel error: {0}")]
    ResizeKernel(#[from] KernelResizeError),
}
