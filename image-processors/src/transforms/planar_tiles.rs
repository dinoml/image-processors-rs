//! Planar floating-point padding, halo tiles and non-blended valid-region assembly.

use super::{validate_image_crop_box, ImageCropBox, ImageSize, TransformError};

fn planar_len(size: ImageSize, channels: usize) -> Result<usize, TransformError> {
    ImageSize::new(size.height, size.width)?;
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }
    size.width
        .checked_mul(size.height)
        .and_then(|value| value.checked_mul(channels))
        .ok_or(TransformError::ImageSizeOverflow)
}

fn validate_data(values: &[f32], size: ImageSize, channels: usize) -> Result<(), TransformError> {
    let expected = planar_len(size, channels)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

/// Pad the bottom/right of planar FP32 data with edge-exclusive reflection.
///
/// # Errors
/// Rejects invalid storage, overflow, or padding not smaller than its source axis.
pub fn reflect_pad_planar_f32(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    right: usize,
    bottom: usize,
) -> Result<(Vec<f32>, ImageSize), TransformError> {
    validate_data(values, source, channels)?;
    if (right != 0 && right >= source.width) || (bottom != 0 && bottom >= source.height) {
        return Err(TransformError::InvalidReflectionPadding {
            source_size: source,
            right,
            bottom,
        });
    }
    let target = ImageSize::new(
        source
            .height
            .checked_add(bottom)
            .ok_or(TransformError::ImageSizeOverflow)?,
        source
            .width
            .checked_add(right)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;
    let mut output = vec![0.0; planar_len(target, channels)?];
    for channel in 0..channels {
        for row in 0..target.height {
            let source_row = if row < source.height {
                row
            } else {
                source.height - (row - source.height) - 2
            };
            for column in 0..target.width {
                let source_column = if column < source.width {
                    column
                } else {
                    source.width - (column - source.width) - 2
                };
                output[(channel * target.height + row) * target.width + column] =
                    values[(channel * source.height + source_row) * source.width + source_column];
            }
        }
    }
    Ok((output, target))
}

/// Copy a rectangular region from planar FP32 storage, retaining channel order.
///
/// # Errors
/// Rejects invalid source storage, crop bounds or arithmetic overflow.
pub fn crop_planar_f32(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    region: ImageCropBox,
) -> Result<Vec<f32>, TransformError> {
    validate_data(values, source, channels)?;
    validate_image_crop_box(region, source)?;
    let size = ImageSize::new(region.y_max - region.y_min, region.x_max - region.x_min)?;
    let mut output = Vec::with_capacity(planar_len(size, channels)?);
    for channel in 0..channels {
        for row in region.y_min..region.y_max {
            let start = (channel * source.height + row) * source.width + region.x_min;
            output.extend_from_slice(&values[start..start + size.width]);
        }
    }
    Ok(output)
}

/// A source halo rectangle and its scaled, non-overlapping output core.
#[derive(Clone, Debug)]
pub struct HaloTile {
    /// Rectangle supplied to independent tile inference.
    pub input: ImageCropBox,
    /// Required dimensions of that tile's network output.
    pub result_size: ImageSize,
    /// Valid rectangle within the tile result, excluding halo context.
    pub valid: ImageCropBox,
    /// Destination rectangle in the assembled output.
    pub destination: ImageCropBox,
}

/// Row-major halo tiles clipped to an unpadded valid output region.
#[derive(Clone, Debug)]
pub struct HaloTilePlan {
    /// Scaled size of the unpadded output.
    pub output_size: ImageSize,
    /// Non-overlapping cores with context clipped at the source boundary.
    pub tiles: Vec<HaloTile>,
}

impl HaloTilePlan {
    /// Plan halo inference without overlap blending or edge extension.
    /// A zero tile size selects a single complete source image.
    ///
    /// # Errors
    /// Rejects invalid geometry, a valid region exceeding the source, or overflow.
    pub fn new(
        source: ImageSize,
        valid: ImageSize,
        tile_size: usize,
        halo: usize,
        scale: usize,
    ) -> Result<Self, TransformError> {
        ImageSize::new(source.height, source.width)?;
        ImageSize::new(valid.height, valid.width)?;
        if valid.width > source.width || valid.height > source.height {
            return Err(TransformError::CropTooLarge {
                source_size: source,
                target_size: valid,
            });
        }
        if scale == 0 {
            return Err(TransformError::InvalidScaleFactor(scale));
        }
        let scaled = |value: usize| {
            value
                .checked_mul(scale)
                .ok_or(TransformError::ImageSizeOverflow)
        };
        let output_size = ImageSize::new(scaled(valid.height)?, scaled(valid.width)?)?;
        let tile = if tile_size == 0 {
            source.width.max(source.height)
        } else {
            tile_size
        };
        let halo = if tile_size == 0 { 0 } else { halo };
        let mut tiles = Vec::new();
        for top in (0..valid.height).step_by(tile) {
            for left in (0..valid.width).step_by(tile) {
                let right = left.saturating_add(tile).min(source.width);
                let bottom = top.saturating_add(tile).min(source.height);
                let input = ImageCropBox::new(
                    left.saturating_sub(halo),
                    top.saturating_sub(halo),
                    right.saturating_add(halo).min(source.width),
                    bottom.saturating_add(halo).min(source.height),
                );
                let destination = ImageCropBox::new(
                    scaled(left)?,
                    scaled(top)?,
                    scaled(right.min(valid.width))?,
                    scaled(bottom.min(valid.height))?,
                );
                let valid_left = scaled(left - input.x_min)?;
                let valid_top = scaled(top - input.y_min)?;
                tiles.push(HaloTile {
                    result_size: ImageSize::new(
                        scaled(input.y_max - input.y_min)?,
                        scaled(input.x_max - input.x_min)?,
                    )?,
                    valid: ImageCropBox::new(
                        valid_left,
                        valid_top,
                        valid_left
                            .checked_add(destination.x_max - destination.x_min)
                            .ok_or(TransformError::ImageSizeOverflow)?,
                        valid_top
                            .checked_add(destination.y_max - destination.y_min)
                            .ok_or(TransformError::ImageSizeOverflow)?,
                    ),
                    input,
                    destination,
                });
            }
        }
        Ok(Self { output_size, tiles })
    }

    /// Allocate the assembled planar output without imposing a pixel-value policy.
    ///
    /// # Errors
    /// Rejects zero channels or arithmetic overflow.
    pub fn allocate_output(&self, channels: usize) -> Result<Vec<f32>, TransformError> {
        Ok(vec![0.0; planar_len(self.output_size, channels)?])
    }

    /// Copy only this tile's valid core into its unique destination rectangle.
    ///
    /// # Errors
    /// Rejects incompatible tensor lengths, rectangles or core dimensions.
    pub fn assemble_tile(
        &self,
        tile: &HaloTile,
        values: &[f32],
        channels: usize,
        output: &mut [f32],
    ) -> Result<(), TransformError> {
        validate_data(values, tile.result_size, channels)?;
        validate_data(output, self.output_size, channels)?;
        validate_image_crop_box(tile.valid, tile.result_size)?;
        validate_image_crop_box(tile.destination, self.output_size)?;
        let width = tile.valid.x_max - tile.valid.x_min;
        let height = tile.valid.y_max - tile.valid.y_min;
        if width != tile.destination.x_max - tile.destination.x_min
            || height != tile.destination.y_max - tile.destination.y_min
        {
            return Err(TransformError::InvalidImageCropBox {
                crop_box: tile.valid,
                image_size: tile.result_size,
            });
        }
        for channel in 0..channels {
            for row in 0..height {
                let source = (channel * tile.result_size.height + tile.valid.y_min + row)
                    * tile.result_size.width
                    + tile.valid.x_min;
                let target = (channel * self.output_size.height + tile.destination.y_min + row)
                    * self.output_size.width
                    + tile.destination.x_min;
                output[target..target + width].copy_from_slice(&values[source..source + width]);
            }
        }
        Ok(())
    }
}
