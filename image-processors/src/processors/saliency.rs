//! U²-Net image preparation and source-sized foreground restoration.
//!
//! Follows xuebinqin/U-2-Net's `data_loader.py::RescaleT/ToTensorLab(flag=0)`
//! and `u2net_human_seg_test.py::normPRED/save_output` at revision
//! ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29.

use std::borrow::Cow;

use crate::image::ImageProcessorError;
use crate::media::{ImageFrame, PixelFormat};
use crate::tensor::{Layout, Tensor, TensorData};
use crate::transforms::{
    ImageSize, ResizeDecision, ResizeFilter, ResizeMode, TransformError, binarize_mask,
    convert_frame_pixel_format, resize_f32_image_bilinear, resize_f64_skimage_bilinear_constant,
    resize_frame_with_decision,
};

/// U²-Net image preprocessing at a caller-configured network resolution.
#[derive(Clone, Debug)]
pub struct U2NetImageProcessor {
    size: ImageSize,
}

/// Source-sized predictions and foreground visualization owned by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct U2NetPostprocessOutput {
    /// Bilinearly restored raw sigmoid probabilities, without visualization normalization.
    pub probabilities: Vec<f32>,
    /// Strict threshold of the restored raw probabilities, in row-major order.
    pub binary_mask: Vec<bool>,
    /// Upstream visualization mask: FP32 min-max normalization, byte conversion, then Pillow resize.
    pub mask: ImageFrame,
    /// Source RGB pixels with the visualization mask as alpha.
    pub rgba: ImageFrame,
}

impl U2NetImageProcessor {
    /// Create a processor with positive network dimensions.
    ///
    /// # Errors
    /// Reports invalid or overflowing image dimensions.
    pub fn new(size: ImageSize) -> Result<Self, ImageProcessorError> {
        ImageSize::new(size.height, size.width)?;
        size.height
            .checked_mul(size.width)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(TransformError::ImageSizeOverflow)?;
        Ok(Self { size })
    }

    /// Return the network input dimensions.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Prepare one decoded image as FP32 `[1, 3, height, width]`.
    ///
    /// Integer pixels are converted to FP64 before scikit-image constant-boundary
    /// resizing. Division by the resized image maximum and ImageNet normalization
    /// also use FP64; the result is cast once to FP32, as upstream does before
    /// inference. Grayscale repeats its first-channel statistics across RGB.
    /// RGBA participates in the source maximum before its alpha channel is dropped.
    /// The source's division by zero for an entirely black image is retained.
    ///
    /// # Errors
    /// Reports invalid geometry, storage, resize or tensor construction failures.
    pub fn preprocess(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        let source = ImageSize::new(image.height(), image.width())?;
        // img_as_float for uint8 multiplies by the FP64 reciprocal of 255.
        let values = image
            .data()
            .iter()
            .map(|&value| f64::from(value) * (1.0 / 255.0))
            .collect::<Vec<_>>();
        let resized =
            resize_f64_skimage_bilinear_constant(&values, source, image.channels(), self.size)?;
        let maximum = resized.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pixels = self.size.height * self.size.width;
        let mut normalized = vec![0.0_f32; pixels * 3];
        let mean = [0.485_f64, 0.456, 0.406];
        let std = [0.229_f64, 0.224, 0.225];
        let grayscale = image.pixel_format() == PixelFormat::Luma8;
        for channel in 0..3 {
            let source_channel = if grayscale { 0 } else { channel };
            let statistics_channel = if grayscale { 0 } else { channel };
            for pixel in 0..pixels {
                let value = resized[pixel * image.channels() + source_channel] / maximum;
                normalized[channel * pixels + pixel] =
                    ((value - mean[statistics_channel]) / std[statistics_channel]) as f32;
            }
        }
        Ok(Tensor::new(
            TensorData::F32(normalized),
            vec![1, 3, self.size.height, self.size.width],
            Layout::NCHW,
        )?)
    }

    /// Prepare independent images as one FP32 NCHW batch.
    ///
    /// # Errors
    /// Rejects an empty batch, overflowing storage or invalid source images.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let count = self.size.height.checked_mul(self.size.width)
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| value.checked_mul(images.len()))
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mut values = Vec::with_capacity(count);
        for image in images {
            let (TensorData::F32(pixels), _, _, _) = self.preprocess(image)?.into_parts() else {
                unreachable!("U2Net preprocessing always constructs FP32 data");
            };
            values.extend(pixels);
        }
        Ok(Tensor::new(TensorData::F32(values),
            vec![images.len(), 3, self.size.height, self.size.width], Layout::NCHW)?)
    }

    /// Restore one fused sigmoid head to the decoded source image.
    ///
    /// Raw probabilities stay floating point. The visualization follows upstream's
    /// FP32 min-max scaling and truncating mode-F-to-RGB byte conversion before
    /// Pillow bilinear resize. A constant prediction yields a black visualization,
    /// matching the byte conversion of upstream's undefined zero-span normalization.
    ///
    /// # Errors
    /// Reports incorrect prediction length, nonfinite probabilities or invalid threshold.
    pub fn postprocess(
        &self,
        probabilities: &[f32],
        image: &ImageFrame,
        threshold: f32,
    ) -> Result<U2NetPostprocessOutput, ImageProcessorError> {
        let expected = self.size.height * self.size.width;
        if probabilities.len() != expected {
            return Err(TransformError::InvalidBufferLength {
                expected,
                actual: probabilities.len(),
            }
            .into());
        }
        for &value in probabilities {
            if !value.is_finite() {
                return Err(TransformError::InvalidImageValue(value).into());
            }
        }
        let target = ImageSize::new(image.height(), image.width())?;
        let restored = resize_f32_image_bilinear(probabilities, self.size, target)?;
        let binary_mask = binarize_mask(&restored, threshold)?;
        let minimum = probabilities.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = probabilities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let span = maximum - minimum;
        let mask = probabilities
            .iter()
            .map(|&value| (((value - minimum) / span) * 255.0) as u8)
            .collect::<Vec<_>>();
        let mask = ImageFrame::new(self.size.width, self.size.height, PixelFormat::Luma8, mask)?;
        let mask = resize_frame_with_decision(
            &mask,
            target,
            ResizeDecision::compatibility(ResizeFilter::Bilinear),
            ResizeMode::Default,
        )?;
        let rgb = if image.pixel_format() == PixelFormat::Rgb8 {
            Cow::Borrowed(image)
        } else {
            Cow::Owned(convert_frame_pixel_format(image, PixelFormat::Rgb8)?)
        };
        let output_len = restored
            .len()
            .checked_mul(4)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mut rgba = Vec::with_capacity(output_len);
        for (pixel, &alpha) in rgb.data().chunks_exact(3).zip(mask.data()) {
            rgba.extend_from_slice(pixel);
            rgba.push(alpha);
        }
        let rgba = ImageFrame::new(image.width(), image.height(), PixelFormat::Rgba8, rgba)?;
        Ok(U2NetPostprocessOutput {
            probabilities: restored,
            binary_mask,
            mask,
            rgba,
        })
    }
}
