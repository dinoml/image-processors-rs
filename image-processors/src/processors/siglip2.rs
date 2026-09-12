//! Original SigLIP2 NaFlex resize, patch packing and padding.

use crate::{
    ImageFrame, ImageProcessor, ImageProcessorConfig, ImageProcessorError, ImageSize, Layout,
    ResizeParity, TensorData,
};
use thiserror::Error;

/// NaFlex geometry or image conversion failure.
#[derive(Debug, Error)]
pub enum Siglip2Error {
    /// Invalid or unrepresentable input/configuration dimensions.
    #[error("invalid SigLIP2 dimensions or patch budget")]
    InvalidGeometry,
    /// Image decoding, resize, normalization or tensor failure.
    #[error(transparent)]
    Image(#[from] ImageProcessorError),
}

/// Aspect-preserving resize and fixed-budget patch settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Siglip2ImageProcessor {
    patch_size: usize,
    max_num_patches: usize,
}

/// One image's row-major patches and original attention/spatial metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Siglip2ImageOutput {
    /// Shape `[1, max_num_patches, patch_size * patch_size * 3]`.
    pub shape: [usize; 3],
    /// Normalized patches; within each patch the order is height, width, RGB.
    pub pixel_values: Vec<f32>,
    /// One for a real patch and zero for padding.
    pub pixel_attention_mask: Vec<i64>,
    /// Real patch grid `[height, width]` before patch padding.
    pub spatial_shapes: [usize; 2],
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Resized pixel dimensions.
    pub resized_size: ImageSize,
}

impl Siglip2ImageProcessor {
    /// Creates an original RGB NaFlex processor with mean/std 0.5.
    /// # Errors
    /// Rejects zero dimensions and a patch buffer whose size cannot be represented.
    pub fn new(patch_size: usize, max_num_patches: usize) -> Result<Self, Siglip2Error> {
        if patch_size == 0
            || max_num_patches == 0
            || patch_size
                .checked_mul(patch_size)
                .and_then(|v| v.checked_mul(3))
                .and_then(|v| v.checked_mul(max_num_patches))
                .and_then(|v| v.checked_mul(4))
                .is_none()
            || patch_size > u32::MAX as usize
            || max_num_patches > u32::MAX as usize
        {
            return Err(Siglip2Error::InvalidGeometry);
        }
        Ok(Self {
            patch_size,
            max_num_patches,
        })
    }

    /// Returns the aspect-dependent pixel geometry using the original binary search.
    /// # Errors
    /// Rejects empty, oversized or unrepresentable dimensions.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Validated u32 dimensions use the original f64 binary-search geometry"
    )]
    pub fn resized_size(&self, original: ImageSize) -> Result<ImageSize, Siglip2Error> {
        if original.height == 0
            || original.width == 0
            || original.height > u32::MAX as usize
            || original.width > u32::MAX as usize
        {
            return Err(Siglip2Error::InvalidGeometry);
        }
        let patch = self.patch_size as f64;
        let scaled =
            |scale: f64, size: usize| ((size as f64 * scale / patch).ceil() * patch).max(patch);
        let mut low = 1e-6;
        let mut high = 100.0;
        while high - low >= 1e-5 {
            let scale = (low + high) / 2.0;
            let height = scaled(scale, original.height);
            let width = scaled(scale, original.width);
            if (height / patch) * (width / patch) <= self.max_num_patches as f64 {
                low = scale;
            } else {
                high = scale;
            }
        }
        let height = scaled(low, original.height);
        let width = scaled(low, original.width);
        if height > u32::MAX as f64
            || width > u32::MAX as f64
            || (height / patch) * (width / patch) > self.max_num_patches as f64
        {
            return Err(Siglip2Error::InvalidGeometry);
        }
        Ok(ImageSize {
            height: height as usize,
            width: width as usize,
        })
    }

    /// Resizes with Pillow-compatible bilinear filtering, normalizes and packs patches.
    /// # Errors
    /// Returns geometry, conversion, resize or tensor errors.
    pub fn preprocess(&self, frame: &ImageFrame) -> Result<Siglip2ImageOutput, Siglip2Error> {
        let original_size = ImageSize {
            height: frame.height(),
            width: frame.width(),
        };
        let resized_size = self.resized_size(original_size)?;
        let pixels = ImageProcessor::new(ImageProcessorConfig {
            do_resize: true,
            height: Some(resized_size.height),
            width: Some(resized_size.width),
            resize_parity: ResizeParity::Compatibility,
            do_normalize: true,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            output_layout: Layout::NHWC,
            ..ImageProcessorConfig::default()
        })?
        .preprocess_image(frame)?;
        let TensorData::F32(pixels) = pixels.data() else {
            return Err(ImageProcessorError::ExpectedDataType("float32").into());
        };
        let patch = self.patch_size;
        let depth = patch * patch * 3;
        let grid = [resized_size.height / patch, resized_size.width / patch];
        let count = grid[0] * grid[1];
        let mut pixel_values = vec![0.0; self.max_num_patches * depth];
        for row in 0..grid[0] {
            for column in 0..grid[1] {
                let offset = (row * grid[1] + column) * depth;
                for y in 0..patch {
                    let source = ((row * patch + y) * resized_size.width + column * patch) * 3;
                    let target = offset + y * patch * 3;
                    pixel_values[target..target + patch * 3]
                        .copy_from_slice(&pixels[source..source + patch * 3]);
                }
            }
        }
        let mut pixel_attention_mask = vec![0; self.max_num_patches];
        pixel_attention_mask[..count].fill(1);
        Ok(Siglip2ImageOutput {
            shape: [1, self.max_num_patches, depth],
            pixel_values,
            pixel_attention_mask,
            spatial_shapes: grid,
            original_size,
            resized_size,
        })
    }
}
