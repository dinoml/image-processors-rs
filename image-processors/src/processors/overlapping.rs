//! Overlapping image crops and averaged semantic score restoration.

use crate::transforms::{
    resize_f32_bilinear_interleaved, ImageCropBox, ImageSize, SemanticSegmentationPrediction,
    TransformError,
};

/// Geometry of overlapping crops along an image's longer axis.
#[derive(Clone, Debug)]
pub struct OverlappingImagePlan {
    /// Image size before preprocessing.
    original_size: ImageSize,
    /// Image size after aspect-preserving resize, before cropping.
    resized_size: ImageSize,
    /// Square crop side length.
    patch_size: usize,
    /// Ordered absolute crop boxes in the resized image.
    crops: Vec<ImageCropBox>,
}

impl OverlappingImagePlan {
    /// Return the image dimensions before preprocessing.
    pub fn original_size(&self) -> ImageSize {
        self.original_size
    }

    /// Return the image dimensions after aspect-preserving resize.
    pub fn resized_size(&self) -> ImageSize {
        self.resized_size
    }

    /// Return the square crop side length.
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Return ordered absolute crop boxes in the resized image.
    pub fn crops(&self) -> &[ImageCropBox] {
        &self.crops
    }
    /// Cover the longer axis with equally overlapping crops up to `patch_size`.
    ///
    /// Positions use Python's floating-point calculation followed by truncation,
    /// matching EoMT's processor, including its final-patch rounding behavior.
    ///
    /// # Errors
    /// Rejects zero dimensions, a zero patch size, or arithmetic overflow.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn new(
        original_size: ImageSize,
        resized_size: ImageSize,
        patch_size: usize,
    ) -> Result<Self, TransformError> {
        ImageSize::new(original_size.height, original_size.width)?;
        ImageSize::new(resized_size.height, resized_size.width)?;
        if patch_size == 0 {
            return Err(TransformError::InvalidSize {
                height: resized_size.height,
                width: resized_size.width,
            });
        }
        let longer = resized_size.height.max(resized_size.width);
        let count = longer.div_ceil(patch_size);
        let total = count
            .checked_mul(patch_size)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let overlap = if count > 1 {
            (total - longer) as f64 / (count - 1) as f64
        } else {
            0.0
        };
        let mut crops = Vec::with_capacity(count);
        for index in 0..count {
            let start = (index as f64 * (patch_size as f64 - overlap)) as usize;
            let end = start
                .checked_add(patch_size)
                .ok_or(TransformError::ImageSizeOverflow)?;
            let crop = if resized_size.height > resized_size.width {
                ImageCropBox::new(0, start, resized_size.width, end.min(resized_size.height))
            } else {
                ImageCropBox::new(start, 0, end.min(resized_size.width), resized_size.height)
            };
            crate::transforms::validate_image_crop_box(crop, resized_size)?;
            crops.push(crop);
        }
        Self::from_crops(original_size, resized_size, patch_size, crops)
    }

    pub(crate) fn from_crops(
        original_size: ImageSize,
        resized_size: ImageSize,
        patch_size: usize,
        crops: Vec<ImageCropBox>,
    ) -> Result<Self, TransformError> {
        ImageSize::new(original_size.height, original_size.width)?;
        ImageSize::new(resized_size.height, resized_size.width)?;
        for crop in &crops {
            crate::transforms::validate_image_crop_box(*crop, resized_size)?;
        }
        Ok(Self {
            original_size,
            resized_size,
            patch_size,
            crops,
        })
    }

    /// Average overlapping class-score planes, resize, and take the final argmax.
    ///
    /// Predictions must contain semantic score planes matching each crop's size.
    /// Averaging happens before restoration to the original image size.
    ///
    /// # Errors
    /// Reports mismatched prediction contracts, invalid geometry, or size overflow.
    #[allow(clippy::cast_precision_loss)]
    pub fn merge_semantic_predictions(
        &self,
        predictions: &[SemanticSegmentationPrediction],
    ) -> Result<SemanticSegmentationPrediction, TransformError> {
        if predictions.len() != self.crops.len() || predictions.is_empty() {
            return Err(TransformError::InvalidBufferLength {
                expected: self.crops.len(),
                actual: predictions.len(),
            });
        }
        let labels = predictions[0].num_labels;
        if labels == 0 {
            return Err(TransformError::InvalidClassCount(labels));
        }
        let pixels = self
            .resized_size
            .height
            .checked_mul(self.resized_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mut sums = vec![
            0.0_f32;
            pixels
                .checked_mul(labels)
                .ok_or(TransformError::ImageSizeOverflow)?
        ];
        let mut counts = vec![0.0_f32; pixels];
        for (prediction, crop) in predictions.iter().zip(&self.crops) {
            let height = crop.y_max - crop.y_min;
            let width = crop.x_max - crop.x_min;
            let patch_pixels = height
                .checked_mul(width)
                .ok_or(TransformError::ImageSizeOverflow)?;
            if prediction.num_labels != labels
                || prediction.size.height != height
                || prediction.size.width != width
            {
                return Err(TransformError::InvalidSize {
                    height: prediction.size.height,
                    width: prediction.size.width,
                });
            }
            let expected = labels
                .checked_mul(patch_pixels)
                .ok_or(TransformError::ImageSizeOverflow)?;
            if prediction.scores.len() != expected {
                return Err(TransformError::InvalidBufferLength {
                    expected,
                    actual: prediction.scores.len(),
                });
            }
            for y in 0..height {
                for x in 0..width {
                    let destination = (crop.y_min + y) * self.resized_size.width + crop.x_min + x;
                    counts[destination] += 1.0;
                    for label in 0..labels {
                        sums[label * pixels + destination] +=
                            prediction.scores[label * patch_pixels + y * width + x];
                    }
                }
            }
        }
        let target_pixels = self
            .original_size
            .height
            .checked_mul(self.original_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mut scores = Vec::with_capacity(
            labels
                .checked_mul(target_pixels)
                .ok_or(TransformError::ImageSizeOverflow)?,
        );
        for plane in sums.chunks_exact_mut(pixels) {
            for (value, count) in plane.iter_mut().zip(&counts) {
                *value /= count.max(1.0);
            }
            scores.extend(resize_f32_bilinear_interleaved(
                plane,
                self.resized_size,
                1,
                self.original_size,
                false,
            )?);
        }
        let mut class_ids = vec![0_i64; target_pixels];
        for pixel in 0..target_pixels {
            let mut best = 0;
            for label in 1..labels {
                if scores[label * target_pixels + pixel] > scores[best * target_pixels + pixel] {
                    best = label;
                }
            }
            class_ids[pixel] =
                i64::try_from(best).map_err(|_| TransformError::ImageSizeOverflow)?;
        }
        Ok(SemanticSegmentationPrediction {
            size: self.original_size,
            num_labels: labels,
            class_ids,
            scores,
        })
    }
}
