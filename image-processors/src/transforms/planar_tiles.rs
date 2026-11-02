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
    /// Split once into overlapping quadrants with cores meeting at the image midpoint.
    ///
    /// Context extends to the next receptive-field multiple after each midpoint,
    /// as in cszn/SCUNet's `test_onesplit`. Short axes retain Python's negative
    /// suffix-slice offsets when their output core fits. Otherwise the whole
    /// axis is retained. Empty cores on one-pixel axes are skipped.
    ///
    /// # Errors
    /// Rejects invalid dimensions, zero receptive field/scale or arithmetic overflow.
    pub fn split_once(
        source: ImageSize,
        receptive_field: usize,
        scale: usize,
    ) -> Result<Self, TransformError> {
        ImageSize::new(source.height, source.width)?;
        if receptive_field == 0 || scale == 0 {
            return Err(TransformError::InvalidScaleFactor(0));
        }
        let scaled = |value: usize| {
            value
                .checked_mul(scale)
                .ok_or(TransformError::ImageSizeOverflow)
        };
        let context = |extent: usize| {
            (extent / 2 / receptive_field + 1)
                .checked_mul(receptive_field)
                .ok_or(TransformError::ImageSizeOverflow)
        };
        let height = context(source.height)?;
        let width = context(source.width)?;
        let middle_y = source.height / 2;
        let middle_x = source.width / 2;
        // Python translates a negative suffix start relative to the source end.
        // Retain a full short axis when that suffix would truncate its valid core.
        let bottom_start = if height <= source.height {
            source.height - height
        } else {
            source.height.saturating_sub(height - source.height)
        };
        let bottom_start = if bottom_start <= middle_y {
            bottom_start
        } else {
            0
        };
        let right_start = if width <= source.width {
            source.width - width
        } else {
            source.width.saturating_sub(width - source.width)
        };
        let right_start = if right_start <= middle_x {
            right_start
        } else {
            0
        };
        let mut tiles = Vec::with_capacity(4);
        for (top, bottom, input_top, input_bottom) in [
            (0, middle_y, 0, height.min(source.height)),
            (middle_y, source.height, bottom_start, source.height),
        ] {
            for (left, right, input_left, input_right) in [
                (0, middle_x, 0, width.min(source.width)),
                (middle_x, source.width, right_start, source.width),
            ] {
                if top == bottom || left == right {
                    continue;
                }
                tiles.push(HaloTile {
                    input: ImageCropBox::new(input_left, input_top, input_right, input_bottom),
                    result_size: ImageSize::new(
                        scaled(input_bottom - input_top)?,
                        scaled(input_right - input_left)?,
                    )?,
                    valid: ImageCropBox::new(
                        scaled(left - input_left)?,
                        scaled(top - input_top)?,
                        scaled(right - input_left)?,
                        scaled(bottom - input_top)?,
                    ),
                    destination: ImageCropBox::new(
                        scaled(left)?,
                        scaled(top)?,
                        scaled(right)?,
                        scaled(bottom)?,
                    ),
                });
            }
        }
        Ok(Self {
            output_size: ImageSize::new(scaled(source.height)?, scaled(source.width)?)?,
            tiles,
        })
    }

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

/// The eight image symmetries in cszn image-restoration augmentation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DihedralTransform {
    /// Preserve the image.
    Identity,
    /// Exchange rows and columns.
    Transpose,
    /// Reverse row order.
    FlipVertical,
    /// Rotate clockwise by a quarter turn.
    RotateClockwise,
    /// Reverse column order.
    FlipHorizontal,
    /// Rotate counterclockwise by a quarter turn.
    RotateCounterClockwise,
    /// Rotate by half a turn.
    RotateHalfTurn,
    /// Reflect across the opposite diagonal.
    Transverse,
}

impl DihedralTransform {
    /// All transformations in upstream `augment_img` mode order, zero through seven.
    pub const ALL: [Self; 8] = [
        Self::Identity,
        Self::Transpose,
        Self::FlipVertical,
        Self::RotateClockwise,
        Self::FlipHorizontal,
        Self::RotateCounterClockwise,
        Self::RotateHalfTurn,
        Self::Transverse,
    ];

    /// Transformation that restores the original orientation.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::RotateClockwise => Self::RotateCounterClockwise,
            Self::RotateCounterClockwise => Self::RotateClockwise,
            other => other,
        }
    }

    /// Resulting spatial dimensions, including the rectangular-image cases.
    #[must_use]
    pub const fn output_size(self, source: ImageSize) -> ImageSize {
        match self {
            Self::Transpose
            | Self::RotateClockwise
            | Self::RotateCounterClockwise
            | Self::Transverse => ImageSize {
                height: source.width,
                width: source.height,
            },
            _ => source,
        }
    }

    /// Transform channel-first image planes without changing their floating values.
    ///
    /// # Errors
    /// Rejects invalid dimensions, tensor storage or arithmetic overflow.
    pub fn apply_planar_f32(
        self,
        values: &[f32],
        source: ImageSize,
        channels: usize,
    ) -> Result<Vec<f32>, TransformError> {
        validate_data(values, source, channels)?;
        let target = self.output_size(source);
        let mut result = vec![0.0; values.len()];
        for channel in 0..channels {
            for row in 0..target.height {
                for column in 0..target.width {
                    let (source_row, source_column) = match self {
                        Self::Identity => (row, column),
                        Self::Transpose => (column, row),
                        Self::FlipVertical => (source.height - 1 - row, column),
                        Self::RotateClockwise => (source.height - 1 - column, row),
                        Self::FlipHorizontal => (row, source.width - 1 - column),
                        Self::RotateCounterClockwise => (column, source.width - 1 - row),
                        Self::RotateHalfTurn => {
                            (source.height - 1 - row, source.width - 1 - column)
                        }
                        Self::Transverse => (source.height - 1 - column, source.width - 1 - row),
                    };
                    result[(channel * target.height + row) * target.width + column] = values
                        [(channel * source.height + source_row) * source.width + source_column];
                }
            }
        }
        Ok(result)
    }
}

/// Streaming FP32 mean of identically shaped planar inference results.
pub struct PlanarMean {
    size: ImageSize,
    channels: usize,
    sum: Vec<f32>,
    count: u16,
}

impl PlanarMean {
    /// Allocate one accumulator without retaining individual ensemble results.
    ///
    /// # Errors
    /// Rejects invalid dimensions, zero channels or arithmetic overflow.
    pub fn new(size: ImageSize, channels: usize) -> Result<Self, TransformError> {
        Ok(Self {
            size,
            channels,
            sum: vec![0.0; planar_len(size, channels)?],
            count: 0,
        })
    }

    /// Add one result after restoring its orientation.
    ///
    /// # Errors
    /// Rejects incompatible storage or more than 65535 samples.
    pub fn add(&mut self, values: &[f32]) -> Result<(), TransformError> {
        validate_data(values, self.size, self.channels)?;
        let count = self
            .count
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)?;
        for (sum, value) in self.sum.iter_mut().zip(values) {
            *sum += value;
        }
        self.count = count;
        Ok(())
    }

    /// Finish the mean in FP32; callers retain their network output dtype boundary.
    ///
    /// # Errors
    /// Rejects an empty ensemble.
    pub fn finish(mut self) -> Result<Vec<f32>, TransformError> {
        if self.count == 0 {
            return Err(TransformError::InvalidFrameCount(0));
        }
        for value in &mut self.sum {
            *value /= f32::from(self.count);
        }
        Ok(self.sum)
    }
}
