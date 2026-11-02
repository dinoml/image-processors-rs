//! PP-OCR detection image preparation on shared resize and tensor primitives.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{ImageGptPreprocessingPolicy, PresetProcessorError};

use crate::tensor::{Layout, Tensor, TensorData, TensorError};
use crate::transforms::{
    convert_frame_pixel_format, resize_frame_to_f32_torchvision, ImageSize, ResizeFilter,
    TransformError,
};
use crate::{ImageFrame, PixelFormat};

/// Aspect-preserving detector resize strategy.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextDetectionLimitType {
    /// Downsample only when the longest side exceeds the limit.
    Max,
    /// Upsample only when the shortest side is below the limit.
    Min,
    /// Always resize the longest side to the limit.
    ResizeLong,
}

/// Settings read directly from a Transformers image processor JSON.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct TextDetectionImageConfig {
    /// Whether to apply the aspect-preserving resize policy.
    pub do_resize: bool,
    /// Whether to rescale byte values.
    pub do_rescale: bool,
    /// Whether to normalize channels before reversing them.
    pub do_normalize: bool,
    /// Multiplicative rescale factor.
    pub rescale_factor: f64,
    /// RGB means, before channel reversal.
    pub image_mean: Vec<f32>,
    /// RGB standard deviations, before channel reversal.
    pub image_std: Vec<f32>,
    /// PIL-compatible resampling identifier: 0 nearest, 2 bilinear, 3 bicubic.
    pub resample: u32,
    /// Side length used by the selected resize policy.
    pub limit_side_len: usize,
    /// Which side controls the initial resize ratio.
    pub limit_type: TextDetectionLimitType,
    /// Optional cap applied after the initial resize and before 32-pixel alignment.
    pub max_side_limit: Option<usize>,
}

impl Default for TextDetectionImageConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            rescale_factor: 1.0 / 255.0,
            image_mean: vec![0.406, 0.456, 0.485],
            image_std: vec![0.225, 0.224, 0.229],
            resample: 2,
            limit_side_len: 960,
            limit_type: TextDetectionLimitType::Max,
            max_side_limit: Some(4000),
        }
    }
}

/// Image preparation failure.
#[derive(Debug, Error)]
pub enum TextDetectionImageError {
    /// Invalid processor settings.
    #[error("invalid text detection image setting: {0}")]
    Invalid(&'static str),
    /// Invalid resize geometry.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Invalid output tensor.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Invalid shared preprocessing settings.
    #[error(transparent)]
    Preprocessing(#[from] PresetProcessorError),
}

impl TextDetectionImageConfig {
    /// Calculate detector dimensions with truncation, cap, and nearest-even alignment.
    ///
    /// # Errors
    /// Rejects zero limits and unrepresentable geometry.
    pub fn resized_size(&self, original: ImageSize) -> Result<ImageSize, TextDetectionImageError> {
        ImageSize::new(original.height, original.width)?;
        if !self.do_resize {
            return Ok(original);
        }
        if self.limit_side_len == 0 || self.max_side_limit == Some(0) {
            return Err(TextDetectionImageError::Invalid(
                "side limits must be positive",
            ));
        }
        let side = match self.limit_type {
            TextDetectionLimitType::Max | TextDetectionLimitType::ResizeLong => {
                original.height.max(original.width)
            }
            TextDetectionLimitType::Min => original.height.min(original.width),
        };
        let ratio = match self.limit_type {
            TextDetectionLimitType::Max if side <= self.limit_side_len => 1.0,
            TextDetectionLimitType::Min if side >= self.limit_side_len => 1.0,
            _ => self.limit_side_len as f64 / side as f64,
        };
        let mut height = (original.height as f64 * ratio).trunc();
        let mut width = (original.width as f64 * ratio).trunc();
        if let Some(limit) = self.max_side_limit {
            if height.max(width) > limit as f64 {
                let ratio = limit as f64 / height.max(width);
                height = (height * ratio).trunc();
                width = (width * ratio).trunc();
            }
        }
        let height = ((height / 32.0).round_ties_even() * 32.0).max(32.0);
        let width = ((width / 32.0).round_ties_even() * 32.0).max(32.0);
        if !height.is_finite()
            || !width.is_finite()
            || height >= usize::MAX as f64
            || width >= usize::MAX as f64
        {
            return Err(TextDetectionImageError::Invalid(
                "resize dimensions overflow",
            ));
        }
        Ok(ImageSize::new(height as usize, width as usize)?)
    }

    /// Prepare one decoded image as NCHW float32, with normalization before BGR reversal.
    ///
    /// Resize uses the shared floating-point Torchvision path, avoiding byte
    /// requantization between resize and normalization. The original image size
    /// remains with the caller for output restoration.
    ///
    /// # Errors
    /// Reports invalid settings, resize, normalization and tensor failures.
    pub fn preprocess(&self, image: &ImageFrame) -> Result<Tensor, TextDetectionImageError> {
        let target = self.resized_size(ImageSize::new(image.height(), image.width())?)?;
        let filter = match self.resample {
            0 => ResizeFilter::Nearest,
            2 => ResizeFilter::Bilinear,
            3 => ResizeFilter::Bicubic,
            _ => {
                return Err(TextDetectionImageError::Invalid(
                    "unsupported tensor resize filter",
                ));
            }
        };
        let rgb = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        let values = if self.do_resize {
            resize_frame_to_f32_torchvision(&rgb, target, filter, 1.0)?
        } else {
            rgb.data().iter().map(|value| f32::from(*value)).collect()
        };
        let tensor = Tensor::new(
            TensorData::F32(values),
            vec![1, target.height, target.width, 3],
            Layout::NHWC,
        )?;
        let tensor = ImageGptPreprocessingPolicy::Torchvision.finish_tensor(
            tensor,
            self.do_rescale.then_some(self.rescale_factor),
            self.do_normalize
                .then_some((self.image_mean.as_slice(), self.image_std.as_slice())),
            None,
        )?;
        let tensor = tensor.to_layout(Layout::NCHW)?;
        let (data, shape, layout, _) = tensor.into_parts();
        let TensorData::F32(mut values) = data else {
            return Err(TextDetectionImageError::Invalid(
                "processor did not produce float32",
            ));
        };
        let pixels = target
            .height
            .checked_mul(target.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        for pixel in 0..pixels {
            values.swap(pixel, 2 * pixels + pixel);
        }
        Ok(Tensor::new(TensorData::F32(values), shape, layout)?)
    }
}
