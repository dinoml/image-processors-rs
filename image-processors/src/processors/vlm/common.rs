//! Shared VLM shape helpers.

use super::*;

pub(super) fn checked_vlm_mul(left: usize, right: usize) -> Result<usize, ImageProcessorError> {
    left.checked_mul(right).ok_or_else(vlm_shape_overflow)
}

pub(super) fn checked_vlm_add(left: usize, right: usize) -> Result<usize, ImageProcessorError> {
    left.checked_add(right).ok_or_else(vlm_shape_overflow)
}

pub(super) fn vlm_shape_overflow() -> ImageProcessorError {
    ImageProcessorError::Tensor(crate::tensor::TensorError::ShapeElementCountOverflow)
}
