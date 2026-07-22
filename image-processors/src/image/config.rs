use super::validation::resize_decision_for_parts;
use super::*;
use serde::{Deserialize, Serialize};

/// Configuration for image and video preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageProcessorConfig {
    /// Whether frames should be resized before tensor conversion.
    pub do_resize: bool,
    /// Target frame height used when resizing.
    pub height: Option<usize>,
    /// Target frame width used when resizing.
    pub width: Option<usize>,
    /// Resize mode used when resizing is enabled.
    pub resize_mode: ResizeMode,
    /// Resampling filter used for resize operations.
    pub resample: ResizeFilter,
    /// Resize parity policy used for resize operations.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend used by path, byte, and URL loading entrypoints.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
    /// Pixel format to convert frames into before tensor conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Whether pixel values should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied to pixel values when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether per-channel normalization should be applied.
    pub do_normalize: bool,
    /// Per-channel or scalar normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel or scalar normalization standard deviations.
    pub image_std: Vec<f32>,
    /// Whether values should be thresholded into 0 or 1.
    pub do_binarize: bool,
    /// Output tensor layout.
    pub output_layout: Layout,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Resampling,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            image_mean: Vec::new(),
            image_std: Vec::new(),
            do_binarize: false,
            output_layout: Layout::NCHW,
        }
    }
}

impl ImageProcessorConfig {
    /// Returns the image axis order implied by `output_layout`.
    pub fn output_image_layout(&self) -> ImageLayout {
        match self.output_layout {
            Layout::CHW | Layout::NCHW | Layout::NPCHW | Layout::BFCHW | Layout::NIPCHW => {
                ImageLayout::ChannelsHeightWidth
            }
            Layout::NC
            | Layout::HWC
            | Layout::NHWC
            | Layout::NPHWC
            | Layout::BFHWC
            | Layout::NIPHWC => ImageLayout::HeightWidthChannels,
        }
    }

    /// Returns the video axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        match self.output_image_layout() {
            ImageLayout::ChannelsHeightWidth => VideoLayout::FramesChannelsHeightWidth,
            ImageLayout::HeightWidthChannels => VideoLayout::FramesHeightWidthChannels,
        }
    }

    /// Returns the explicit resize decision implied by this config.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }
}

/// Default batch size where automatic execution switches to parallel work.
///
/// Smaller batches usually do not amortize Rayon scheduling overhead on local
/// CLIP-style image preprocessing workloads.
pub const DEFAULT_PARALLEL_BATCH_THRESHOLD: usize = 8;

/// Batch processing execution mode.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BatchExecution {
    /// Select serial or parallel execution from the batch size and build features.
    Auto,
    /// Process batch items on the calling thread.
    Serial,
    /// Process independent batch items in parallel when the `parallel` feature is enabled.
    Parallel,
}

impl Default for BatchExecution {
    fn default() -> Self {
        default_batch_execution()
    }
}

/// Returns the default batch execution mode for this build.
///
/// Defaults to [`BatchExecution::Auto`], which uses serial execution for small
/// batches and parallel execution for larger batches when the `parallel` feature
/// is enabled.
pub const fn default_batch_execution() -> BatchExecution {
    BatchExecution::Auto
}

/// Per-call preprocessing overrides.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageProcessorOptions {
    /// Per-call resize height override.
    pub height: Option<usize>,
    /// Per-call resize width override.
    pub width: Option<usize>,
    /// Per-call resize mode override.
    pub resize_mode: Option<ResizeMode>,
}
