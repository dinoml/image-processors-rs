//! Dense multiclass logits with MediaPipe-compatible category/confidence maps.

use image_resize_kernels::{
    resize_f32, resize_f32_bilinear_interleaved, ImageSize as KernelImageSize, ResizeError,
    ResizeFilter, ResizeProfile,
};
use thiserror::Error;

use crate::transforms::ImageSize;

/// Independently requested category and per-class confidence masks.
#[derive(Clone, Copy, Debug)]
pub struct MulticlassSegmentationOptions {
    /// Resize raw logits with aligned corners before taking their argmax.
    pub category_mask: bool,
    /// Apply softmax at source resolution and resize probabilities with half-pixel coordinates.
    pub confidence_masks: bool,
}

impl Default for MulticlassSegmentationOptions {
    fn default() -> Self {
        Self {
            category_mask: true,
            confidence_masks: true,
        }
    }
}

/// Final masks in the caller's requested image coordinates.
#[derive(Clone, Debug)]
pub struct MulticlassSegmentationOutput {
    /// Spatial extent of each row-major mask.
    pub size: ImageSize,
    /// Category ids, with ties assigned to the first class; absent when unrequested.
    pub category_mask: Option<Vec<u8>>,
    /// Row-major FP32 planes in class order; empty when unrequested.
    pub confidence_masks: Vec<Vec<f32>>,
}

/// Invalid logits or a shared resize failure.
#[derive(Debug, Error)]
pub enum MulticlassSegmentationError {
    /// Shape, options or value contract violation.
    #[error("{0}")]
    Invalid(&'static str),
    /// Failure in the existing Rust resize owner.
    #[error(transparent)]
    Resize(#[from] ResizeError),
}

/// Restore dense CHW logits to category and confidence masks.
///
/// This preserves the two distinct MediaPipe CPU output paths: category ids
/// come from interpolated logits, whereas confidence masks interpolate softmax
/// probabilities. The existing Rust kernels own both resize conventions; this
/// function does not link OpenCV or implement a model-local resize.
///
/// # Errors
/// Rejects empty dimensions, nonfinite logits, inconsistent storage, more than
/// 256 classes, or a request with no outputs. Propagates shared resize errors.
pub fn post_process_multiclass_segmentation(
    logits: &[f32],
    channels: usize,
    source: ImageSize,
    target: ImageSize,
    options: MulticlassSegmentationOptions,
) -> Result<MulticlassSegmentationOutput, MulticlassSegmentationError> {
    use MulticlassSegmentationError::Invalid;
    if !(2..=256).contains(&channels) || (!options.category_mask && !options.confidence_masks) {
        return Err(Invalid("request masks and between two and 256 classes"));
    }
    let source_size = KernelImageSize::new(source.height, source.width)?;
    let target_size = KernelImageSize::new(target.height, target.width)?;
    let source_pixels = source
        .height
        .checked_mul(source.width)
        .ok_or(Invalid("source dimensions overflow"))?;
    let target_pixels = target
        .height
        .checked_mul(target.width)
        .ok_or(Invalid("target dimensions overflow"))?;
    let values = source_pixels
        .checked_mul(channels)
        .ok_or(Invalid("logit dimensions overflow"))?;
    if logits.len() != values || logits.iter().any(|value| !value.is_finite()) {
        return Err(Invalid(
            "expected finite CHW logits matching source dimensions",
        ));
    }
    let mut interleaved = vec![0.0; values];
    for (pixel, scores) in interleaved.chunks_exact_mut(channels).enumerate() {
        for (channel, value) in scores.iter_mut().enumerate() {
            *value = logits[channel * source_pixels + pixel];
        }
    }
    let category_mask = if options.category_mask {
        let resized = resize_f32_bilinear_interleaved(
            &interleaved,
            source_size,
            channels,
            target_size,
            true,
        )?;
        let mut category = Vec::with_capacity(target_pixels);
        for scores in resized.chunks_exact(channels) {
            let mut best = 0;
            for channel in 1..channels {
                if scores[channel] > scores[best] {
                    best = channel;
                }
            }
            category.push(u8::try_from(best).map_err(|_| Invalid("category id overflow"))?);
        }
        Some(category)
    } else {
        None
    };
    let confidence_masks = if options.confidence_masks {
        for scores in interleaved.chunks_exact_mut(channels) {
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for value in scores.iter_mut() {
                *value = (*value - maximum).exp();
                denominator += *value;
            }
            for value in scores {
                *value /= denominator;
            }
        }
        let resized = resize_f32(
            &interleaved,
            source_size,
            channels,
            target_size,
            ResizeFilter::Bilinear,
            ResizeProfile::OpenCvLinear,
        )?;
        let mut planes = vec![vec![0.0; target_pixels]; channels];
        for (pixel, scores) in resized.chunks_exact(channels).enumerate() {
            for (plane, value) in planes.iter_mut().zip(scores) {
                plane[pixel] = *value;
            }
        }
        planes
    } else {
        Vec::new()
    };
    Ok(MulticlassSegmentationOutput {
        size: target,
        category_mask,
        confidence_masks,
    })
}
