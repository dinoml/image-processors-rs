//! Document foreground bounds with upstream byte-grayscale and threshold semantics.

use super::{convert_frame_pixel_format, crop_frame, ImageCropBox, TransformError};
use crate::media::{ImageFrame, PixelFormat};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Byte-grayscale and normalization policy used to locate document margins.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MarginCropMode {
    /// Pillow RGB-to-L rounding followed by NumPy float64 normalization.
    #[default]
    Pillow,
    /// Torchvision float32 RGB-to-gray truncated to bytes, then float32 normalization.
    Torchvision,
}

/// Source policy for document thumbnail geometry and interpolation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentThumbnailMode {
    /// Pillow bicubic with integer reduction at reducing_gap=2.0.
    Pillow,
    /// Torchvision antialiased bicubic byte resize.
    Torchvision,
}

pub(crate) fn document_thumbnail(
    frame: &ImageFrame,
    target: super::ImageSize,
    mode: DocumentThumbnailMode,
) -> Result<ImageFrame, TransformError> {
    let (mut height, mut width) = (
        frame.height().min(target.height),
        frame.width().min(target.width),
    );
    if height == frame.height() && width == frame.width() {
        return Ok(frame.clone());
    }
    if frame.height() > frame.width() {
        width = frame
            .width()
            .checked_mul(height)
            .ok_or(TransformError::ImageSizeOverflow)?
            / frame.height();
    } else if frame.width() > frame.height() {
        height = frame
            .height()
            .checked_mul(width)
            .ok_or(TransformError::ImageSizeOverflow)?
            / frame.width();
    }
    let size = super::ImageSize::new(height, width)?;
    match mode {
        DocumentThumbnailMode::Torchvision => super::resize_frame_with_decision(
            frame,
            size,
            super::ResizeDecision::torchvision(super::ResizeFilter::Bicubic),
            super::ResizeMode::Default,
        ),
        DocumentThumbnailMode::Pillow => {
            let data = image_resize_kernels::resize_u8_pillow_reducing_gap(
                frame.data(),
                image_resize_kernels::ImageSize {
                    height: frame.height(),
                    width: frame.width(),
                },
                frame.channels(),
                image_resize_kernels::ImageSize {
                    height: size.height,
                    width: size.width,
                },
                image_resize_kernels::ResizeFilter::Bicubic,
                2.0,
            )
            .map_err(|error| TransformError::ResizeKernel {
                message: error.to_string(),
            })?;
            super::new_frame(width, height, frame.pixel_format(), data, frame.timing())
        }
    }
}

/// Crop to the inclusive bounds of pixels strictly below normalized gray 200.
/// Uniform images remain unchanged. The returned pixels retain the input format.
///
/// # Errors
/// Reports pixel conversion, invalid crop bounds or frame storage failures.
pub fn crop_document_margin(
    frame: &ImageFrame,
    mode: MarginCropMode,
) -> Result<ImageFrame, TransformError> {
    let rgb = if frame.pixel_format() == PixelFormat::Rgb8 {
        Cow::Borrowed(frame)
    } else {
        Cow::Owned(convert_frame_pixel_format(frame, PixelFormat::Rgb8)?)
    };
    let gray = rgb
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|pixel| match mode {
            MarginCropMode::Pillow => {
                ((19595 * u32::from(pixel[0])
                    + 38470 * u32::from(pixel[1])
                    + 7471 * u32::from(pixel[2])
                    + 32768)
                    >> 16) as u8
            }
            MarginCropMode::Torchvision => {
                (0.2989 * f32::from(pixel[0])
                    + 0.587 * f32::from(pixel[1])
                    + 0.114 * f32::from(pixel[2])) as u8
            }
        })
        .collect::<Vec<_>>();
    let minimum = gray.iter().copied().min().unwrap_or(0);
    let maximum = gray.iter().copied().max().unwrap_or(0);
    if minimum == maximum {
        return Ok(frame.clone());
    }
    let (mut left, mut top, mut right, mut bottom) = (frame.width(), frame.height(), 0, 0);
    for (index, value) in gray.into_iter().enumerate() {
        let foreground = match mode {
            MarginCropMode::Pillow => {
                f64::from(value - minimum) / f64::from(maximum - minimum) * 255.0 < 200.0
            }
            MarginCropMode::Torchvision => {
                f32::from(value - minimum) / f32::from(maximum - minimum) * 255.0 < 200.0
            }
        };
        if foreground {
            let column = index % frame.width();
            let row = index / frame.width();
            left = left.min(column);
            right = right.max(column + 1);
            top = top.min(row);
            bottom = bottom.max(row + 1);
        }
    }
    crop_frame(frame, ImageCropBox::new(left, top, right, bottom))
}
