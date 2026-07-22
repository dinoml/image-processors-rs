//! Vision-language processor families.

use super::*;

mod anyres;
mod common;
mod config;
mod nested;
mod qwen;

pub use anyres::{LlavaNextImageProcessor, PixtralImageProcessor};
use common::{checked_vlm_add, checked_vlm_mul, vlm_shape_overflow};
pub use config::{
    Gemma3ImageProcessorConfig, Idefics3ImageProcessorConfig, LlavaNextImageProcessorConfig,
    MllamaImageProcessorConfig, PixtralImageProcessorConfig, QwenVlImageProcessorConfig,
};
pub use nested::{Gemma3ImageProcessor, Idefics3ImageProcessor, MllamaImageProcessor};
pub(super) use qwen::validate_positive_dimension;
pub use qwen::QwenVlImageProcessor;
