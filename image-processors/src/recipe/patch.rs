//! Patch flattening recipe stages.

use super::*;
use thiserror::Error;

/// Patch-flattening parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipePatchStage {
    /// Generic image tensor layout consumed by the patch stage.
    pub source_layout: Layout,
    /// Spatial patch size.
    pub patch_size: usize,
    /// Temporal patch size.
    pub temporal_patch_size: usize,
    /// Spatial merge size used when ordering patches.
    pub merge_size: usize,
}

impl RecipePatchStage {
    /// Creates an image patch-flattening stage.
    pub fn flatten(
        source_layout: Layout,
        patch_size: usize,
        temporal_patch_size: usize,
        merge_size: usize,
    ) -> Self {
        Self {
            source_layout,
            patch_size,
            temporal_patch_size,
            merge_size,
        }
    }

    /// Returns image-style temporal-height-width grid metadata.
    ///
    /// The returned grid uses a temporal value of `1`, followed by the spatial
    /// patch grid implied by `target`.
    ///
    /// # Errors
    ///
    /// Returns an error when any patch dimension is zero, when `target` is not
    /// divisible by the patch and merge geometry, or when geometry arithmetic
    /// overflows.
    pub fn image_grid_thw(self, target: ImageSize) -> Result<[usize; 3], RecipePatchError> {
        self.validate_target(target)?;
        Ok([
            1,
            target.height / self.patch_size,
            target.width / self.patch_size,
        ])
    }

    /// Returns video-style temporal-height-width grid metadata.
    ///
    /// Frames are grouped by `temporal_patch_size`; an incomplete final group is
    /// counted as though the last frame were repeated.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame_count` is zero, any patch dimension is zero,
    /// `target` is not divisible by the patch and merge geometry, or geometry
    /// arithmetic overflows.
    pub fn temporal_grid_thw(
        self,
        frame_count: usize,
        target: ImageSize,
    ) -> Result<[usize; 3], RecipePatchError> {
        self.validate_target(target)?;
        Ok([
            self.temporal_grid_size(frame_count)?,
            target.height / self.patch_size,
            target.width / self.patch_size,
        ])
    }

    /// Returns the number of temporal patch groups for a frame count.
    ///
    /// An incomplete final group is counted as though the last frame were
    /// repeated to fill `temporal_patch_size`.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame_count` is zero, any patch dimension is zero,
    /// or padding the frame count overflows.
    pub fn temporal_grid_size(self, frame_count: usize) -> Result<usize, RecipePatchError> {
        self.validate_dimensions()?;
        temporal_repeat_last_plan(frame_count, self.temporal_patch_size)
            .map(|plan| plan.group_count())
            .map_err(recipe_patch_error_from_temporal_plan)
    }

    /// Flattens one preprocessed image tensor into patch rows.
    ///
    /// The tensor layout must match [`RecipePatchStage::source_layout`]. `CHW`
    /// and `HWC` tensors are accepted directly; `NCHW` and `NHWC` tensors must
    /// contain exactly one leading sample. Values are emitted as an `NC` tensor
    /// whose rows follow the merge-aware patch order described by this stage.
    ///
    /// # Errors
    ///
    /// Returns an error when patch geometry is invalid, the target size does
    /// not match the tensor, the tensor layout or data type is unsupported, or
    /// output tensor construction fails.
    pub fn flatten_image_tensor(
        self,
        tensor: &Tensor,
        target: ImageSize,
    ) -> Result<Tensor, RecipePatchError> {
        self.flatten_temporal_tensors(std::slice::from_ref(tensor), target)
    }

    /// Flattens preprocessed frame tensors into temporal patch rows.
    ///
    /// The tensor layouts must match [`RecipePatchStage::source_layout`].
    /// Frames are grouped by [`RecipePatchStage::temporal_patch_size`]; an
    /// incomplete final group repeats the final input tensor. The returned
    /// tensor uses `NC` layout, with rows ordered by temporal group and
    /// merge-aware spatial patch position.
    ///
    /// # Errors
    ///
    /// Returns an error when the input frame list is empty, patch geometry is
    /// invalid, tensor shapes are incompatible, the tensor layout or data type
    /// is unsupported, or output tensor construction fails.
    pub fn flatten_temporal_tensors(
        self,
        tensors: &[Tensor],
        target: ImageSize,
    ) -> Result<Tensor, RecipePatchError> {
        self.validate_target(target)?;
        let sources = tensors
            .iter()
            .map(|tensor| self.patch_source_for_tensor(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        flatten_patch_sources(self, &sources, target)
    }

    /// Validates that an image size can be represented by this patch geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when any patch dimension is zero or `target` is not
    /// divisible by both patch and merge geometry.
    pub fn validate_target(self, target: ImageSize) -> Result<(), RecipePatchError> {
        self.validate_dimensions()?;
        if !target.height.is_multiple_of(self.patch_size)
            || !target.width.is_multiple_of(self.patch_size)
            || !(target.height / self.patch_size).is_multiple_of(self.merge_size)
            || !(target.width / self.patch_size).is_multiple_of(self.merge_size)
        {
            return Err(RecipePatchError::InvalidTarget {
                target_size: target,
                patch_size: self.patch_size,
                merge_size: self.merge_size,
            });
        }
        Ok(())
    }

    fn validate_dimensions(self) -> Result<(), RecipePatchError> {
        validate_recipe_patch_dimension("patch_size", self.patch_size)?;
        validate_recipe_patch_dimension("temporal_patch_size", self.temporal_patch_size)?;
        validate_recipe_patch_dimension("merge_size", self.merge_size)
    }

    fn patch_source_for_tensor<'a>(
        self,
        tensor: &'a Tensor,
    ) -> Result<PatchFlattenSource<'a>, RecipePatchError> {
        if tensor.layout() != self.source_layout {
            return Err(RecipePatchError::UnsupportedSourceLayout {
                expected: self.source_layout,
                actual: tensor.layout(),
            });
        }

        let TensorData::F32(values) = tensor.data() else {
            return Err(RecipePatchError::ExpectedDataType {
                expected: DType::F32,
                actual: tensor.dtype(),
            });
        };

        let (channels, height, width, format) = match tensor.layout() {
            Layout::CHW => {
                let [channels, height, width] = tensor.shape() else {
                    return Err(invalid_patch_source_shape(tensor));
                };
                (
                    *channels,
                    *height,
                    *width,
                    PatchFlattenSourceFormat::ChannelsHeightWidth,
                )
            }
            Layout::HWC => {
                let [height, width, channels] = tensor.shape() else {
                    return Err(invalid_patch_source_shape(tensor));
                };
                (
                    *channels,
                    *height,
                    *width,
                    PatchFlattenSourceFormat::HeightWidthChannels,
                )
            }
            Layout::NCHW => {
                let [batch, channels, height, width] = tensor.shape() else {
                    return Err(invalid_patch_source_shape(tensor));
                };
                if *batch != 1 {
                    return Err(invalid_patch_source_shape(tensor));
                }
                (
                    *channels,
                    *height,
                    *width,
                    PatchFlattenSourceFormat::ChannelsHeightWidth,
                )
            }
            Layout::NHWC => {
                let [batch, height, width, channels] = tensor.shape() else {
                    return Err(invalid_patch_source_shape(tensor));
                };
                if *batch != 1 {
                    return Err(invalid_patch_source_shape(tensor));
                }
                (
                    *channels,
                    *height,
                    *width,
                    PatchFlattenSourceFormat::HeightWidthChannels,
                )
            }
            actual => {
                return Err(RecipePatchError::UnsupportedSourceLayout {
                    expected: self.source_layout,
                    actual,
                });
            }
        };

        Ok(PatchFlattenSource {
            values,
            shape: tensor.shape(),
            channels,
            height,
            width,
            format,
        })
    }
}

/// Errors returned by recipe patch-grid helper methods.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum RecipePatchError {
    /// A patch-stage dimension was zero.
    #[error("{field} must be positive, got {value}")]
    InvalidDimension {
        /// Invalid field name.
        field: &'static str,
        /// Invalid value.
        value: usize,
    },
    /// The target image size is not divisible by the patch geometry.
    #[error(
        "target size {target_size:?} is not divisible by patch_size={patch_size} and merge_size={merge_size}"
    )]
    InvalidTarget {
        /// Target dimensions after resizing.
        target_size: ImageSize,
        /// Spatial patch size.
        patch_size: usize,
        /// Spatial merge size.
        merge_size: usize,
    },
    /// Temporal grid calculation received no frames.
    #[error("temporal patch grid requires at least one frame")]
    EmptyFrameCount,
    /// A tensor used a layout that this patch stage cannot flatten.
    #[error(
        "patch source tensor layout {actual:?} is unsupported for expected layout {expected:?}"
    )]
    UnsupportedSourceLayout {
        /// Layout configured by the patch stage.
        expected: Layout,
        /// Actual source tensor layout.
        actual: Layout,
    },
    /// A tensor shape could not be interpreted as a single frame.
    #[error("patch source tensor with layout {layout:?} has invalid shape {actual:?}")]
    InvalidSourceShape {
        /// Source tensor layout.
        layout: Layout,
        /// Actual source tensor shape.
        actual: Vec<usize>,
    },
    /// Source frame tensors differed in shape or target dimensions.
    #[error("patch source shape {actual:?} is incompatible with expected shape {expected:?}")]
    IncompatibleSourceShape {
        /// Expected tensor-like shape.
        expected: Vec<usize>,
        /// Actual tensor-like shape.
        actual: Vec<usize>,
    },
    /// A tensor had the wrong data type for patch flattening.
    #[error("patch source tensor expected {expected:?} data, got {actual:?}")]
    ExpectedDataType {
        /// Expected tensor dtype.
        expected: DType,
        /// Actual tensor dtype.
        actual: DType,
    },
    /// Output tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Patch geometry arithmetic overflowed.
    #[error("patch geometry overflowed")]
    GeometryOverflow,
}

#[derive(Clone, Copy)]
struct PatchFlattenSource<'a> {
    values: &'a [f32],
    shape: &'a [usize],
    channels: usize,
    height: usize,
    width: usize,
    format: PatchFlattenSourceFormat,
}

#[derive(Clone, Copy)]
enum PatchFlattenSourceFormat {
    ChannelsHeightWidth,
    HeightWidthChannels,
}

#[derive(Clone, Copy)]
struct PatchFlattenPosition {
    temporal_group: usize,
    merged_y: usize,
    merged_x: usize,
    merge_y: usize,
    merge_x: usize,
}

fn flatten_patch_sources(
    patch: RecipePatchStage,
    sources: &[PatchFlattenSource<'_>],
    target: ImageSize,
) -> Result<Tensor, RecipePatchError> {
    patch.validate_target(target)?;
    let first = sources.first().ok_or(RecipePatchError::EmptyFrameCount)?;
    let expected_shape = patch_expected_source_shape(patch.source_layout, first, target);
    for source in sources {
        if source.shape != expected_shape.as_slice() {
            return Err(RecipePatchError::IncompatibleSourceShape {
                expected: expected_shape,
                actual: source.shape.to_vec(),
            });
        }
    }

    let temporal_plan = temporal_repeat_last_plan(sources.len(), patch.temporal_patch_size)
        .map_err(recipe_patch_error_from_temporal_plan)?;
    let grid_t = temporal_plan.group_count();
    let grid_h = target.height / patch.patch_size;
    let grid_w = target.width / patch.patch_size;
    let merged_grid_h = grid_h / patch.merge_size;
    let merged_grid_w = grid_w / patch.merge_size;
    let patches = checked_recipe_patch_mul(grid_t, checked_recipe_patch_mul(grid_h, grid_w)?)?;
    let feature_dim = checked_recipe_patch_mul(
        checked_recipe_patch_mul(first.channels, patch.temporal_patch_size)?,
        checked_recipe_patch_mul(patch.patch_size, patch.patch_size)?,
    )?;
    let output_len = checked_recipe_patch_mul(patches, feature_dim)?;
    let mut values = Vec::with_capacity(output_len);

    for temporal_group in 0..grid_t {
        for merged_y in 0..merged_grid_h {
            for merged_x in 0..merged_grid_w {
                for merge_y in 0..patch.merge_size {
                    for merge_x in 0..patch.merge_size {
                        append_patch_features(
                            sources,
                            &mut values,
                            patch,
                            temporal_plan,
                            PatchFlattenPosition {
                                temporal_group,
                                merged_y,
                                merged_x,
                                merge_y,
                                merge_x,
                            },
                        )?;
                    }
                }
            }
        }
    }

    Ok(Tensor::new(
        TensorData::F32(values),
        vec![patches, feature_dim],
        Layout::NC,
    )?)
}

fn patch_expected_source_shape(
    layout: Layout,
    source: &PatchFlattenSource<'_>,
    target: ImageSize,
) -> Vec<usize> {
    match layout {
        Layout::CHW => vec![source.channels, target.height, target.width],
        Layout::HWC => vec![target.height, target.width, source.channels],
        Layout::NCHW => vec![1, source.channels, target.height, target.width],
        Layout::NHWC => vec![1, target.height, target.width, source.channels],
        _ => source.shape.to_vec(),
    }
}

fn append_patch_features(
    sources: &[PatchFlattenSource<'_>],
    output: &mut Vec<f32>,
    patch: RecipePatchStage,
    temporal_plan: TemporalRepeatLastPlan,
    position: PatchFlattenPosition,
) -> Result<(), RecipePatchError> {
    let patch_origin_y = checked_recipe_patch_mul(
        checked_recipe_patch_add(
            checked_recipe_patch_mul(position.merged_y, patch.merge_size)?,
            position.merge_y,
        )?,
        patch.patch_size,
    )?;
    let patch_origin_x = checked_recipe_patch_mul(
        checked_recipe_patch_add(
            checked_recipe_patch_mul(position.merged_x, patch.merge_size)?,
            position.merge_x,
        )?,
        patch.patch_size,
    )?;

    let first = sources.first().ok_or(RecipePatchError::EmptyFrameCount)?;
    for channel in 0..first.channels {
        for temporal_offset in 0..patch.temporal_patch_size {
            let temporal_index = checked_recipe_patch_add(
                checked_recipe_patch_mul(position.temporal_group, patch.temporal_patch_size)?,
                temporal_offset,
            )?;
            let source_index = temporal_plan
                .source_index(temporal_index)
                .ok_or(RecipePatchError::GeometryOverflow)?;
            let source = sources[source_index];
            for patch_y in 0..patch.patch_size {
                for patch_x in 0..patch.patch_size {
                    let y = checked_recipe_patch_add(patch_origin_y, patch_y)?;
                    let x = checked_recipe_patch_add(patch_origin_x, patch_x)?;
                    let index = patch_source_index(source, channel, y, x)?;
                    let value = source
                        .values
                        .get(index)
                        .copied()
                        .ok_or(RecipePatchError::GeometryOverflow)?;
                    output.push(value);
                }
            }
        }
    }
    Ok(())
}

fn patch_source_index(
    source: PatchFlattenSource<'_>,
    channel: usize,
    y: usize,
    x: usize,
) -> Result<usize, RecipePatchError> {
    match source.format {
        PatchFlattenSourceFormat::ChannelsHeightWidth => {
            let channel_stride = checked_recipe_patch_mul(source.height, source.width)?;
            let channel_offset = checked_recipe_patch_mul(channel, channel_stride)?;
            let row_offset = checked_recipe_patch_mul(y, source.width)?;
            checked_recipe_patch_add(channel_offset, checked_recipe_patch_add(row_offset, x)?)
        }
        PatchFlattenSourceFormat::HeightWidthChannels => {
            let row_offset = checked_recipe_patch_mul(y, source.width)?;
            let pixel_index = checked_recipe_patch_add(row_offset, x)?;
            let pixel_offset = checked_recipe_patch_mul(pixel_index, source.channels)?;
            checked_recipe_patch_add(pixel_offset, channel)
        }
    }
}

fn invalid_patch_source_shape(tensor: &Tensor) -> RecipePatchError {
    RecipePatchError::InvalidSourceShape {
        layout: tensor.layout(),
        actual: tensor.shape().to_vec(),
    }
}

fn recipe_patch_error_from_temporal_plan(error: TransformError) -> RecipePatchError {
    match error {
        TransformError::EmptyFrameBatch | TransformError::EmptyImageBatch => {
            RecipePatchError::EmptyFrameCount
        }
        TransformError::InvalidTemporalMultiple(value)
        | TransformError::InvalidScaleFactor(value) => RecipePatchError::InvalidDimension {
            field: "temporal_patch_size",
            value,
        },
        _ => RecipePatchError::GeometryOverflow,
    }
}

fn checked_recipe_patch_mul(left: usize, right: usize) -> Result<usize, RecipePatchError> {
    left.checked_mul(right)
        .ok_or(RecipePatchError::GeometryOverflow)
}

fn checked_recipe_patch_add(left: usize, right: usize) -> Result<usize, RecipePatchError> {
    left.checked_add(right)
        .ok_or(RecipePatchError::GeometryOverflow)
}

fn validate_recipe_patch_dimension(
    field: &'static str,
    value: usize,
) -> Result<(), RecipePatchError> {
    if value == 0 {
        Err(RecipePatchError::InvalidDimension { field, value })
    } else {
        Ok(())
    }
}
