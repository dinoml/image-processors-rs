//! Dense multiclass logits with MediaPipe-compatible category/confidence maps.

use image_resize_kernels::{
    resize_f32, resize_f32_bilinear_interleaved, ImageSize as KernelImageSize, ResizeError,
    ResizeFilter, ResizeProfile,
};
use thiserror::Error;

use crate::tensor::DType;
use crate::transforms::{ImageSize, SemanticSegmentationPrediction};

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

/// Restore dense CHW semantic logits using half-pixel bilinear interpolation.
///
/// Inputs are decoded values of a tensor stored in `precision`. Resizing uses
/// the existing FP32-accumulating tensor resize kernel with `align_corners=false`.
/// Each interpolated value is rounded to the source tensor precision before
/// argmax, matching tensor interpolation followed by class selection. Ties keep
/// the first class. Returned scores are these raw rounded logits, without softmax.
/// Unlike category-byte outputs, class indices are not limited to 256 labels.
///
/// # Errors
/// Rejects empty dimensions, unsupported precision, nonfinite input logits,
/// inconsistent storage, or size overflow. Propagates shared resize failures.
pub fn post_process_dense_semantic_segmentation(
    logits: &[f32],
    channels: usize,
    source: ImageSize,
    target: ImageSize,
    precision: DType,
) -> Result<SemanticSegmentationPrediction, MulticlassSegmentationError> {
    use MulticlassSegmentationError::Invalid;
    if channels == 0 || i64::try_from(channels - 1).is_err() {
        return Err(Invalid(
            "semantic class count must fit positive i64 indices",
        ));
    }
    if !matches!(precision, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Invalid(
            "semantic logits require F32, F16 or BF16 precision",
        ));
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
    let input_len = channels
        .checked_mul(source_pixels)
        .ok_or(Invalid("source logit dimensions overflow"))?;
    let output_len = channels
        .checked_mul(target_pixels)
        .ok_or(Invalid("target logit dimensions overflow"))?;
    if logits.len() != input_len || logits.iter().any(|value| !value.is_finite()) {
        return Err(Invalid(
            "expected finite CHW semantic logits matching source dimensions",
        ));
    }
    let mut scores = Vec::with_capacity(output_len);
    let mut class_ids = vec![0; target_pixels];
    let mut best_scores = vec![f32::NEG_INFINITY; target_pixels];
    for (class, plane) in logits.chunks_exact(source_pixels).enumerate() {
        let resized = resize_f32_bilinear_interleaved(plane, source_size, 1, target_size, false)?;
        let class = i64::try_from(class).map_err(|_| Invalid("semantic class index overflow"))?;
        for (pixel, value) in resized.into_iter().enumerate() {
            let value = match precision {
                DType::F16 => half::f16::from_f32(value).to_f32(),
                DType::BF16 => half::bf16::from_f32(value).to_f32(),
                _ => value,
            };
            if value > best_scores[pixel] {
                best_scores[pixel] = value;
                class_ids[pixel] = class;
            }
            scores.push(value);
        }
    }
    Ok(SemanticSegmentationPrediction {
        size: target,
        num_labels: channels,
        class_ids,
        scores,
    })
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
