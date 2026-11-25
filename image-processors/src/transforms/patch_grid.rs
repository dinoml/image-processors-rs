//! Patch-grid image extraction helpers.

use super::*;

/// A row-major crop and its non-overlapping valid core, in crop pixel coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapMarginTile {
    /// Rectangle copied from the padded canvas.
    pub crop: ImageCropBox,
    /// Core retained after discarding interior overlap margins.
    pub valid: ImageCropBox,
}

/// Fixed-margin overlapping geometry for a caller-selected grid.
/// Selection policy, patch pooling and token order belong to the caller.
#[derive(Clone, Debug)]
pub struct OverlapMarginGridPlan {
    /// Padded canvas covered exactly by the crops.
    pub canvas_size: ImageSize,
    /// Row-major crop rectangles and valid cores.
    pub tiles: Vec<OverlapMarginTile>,
}

impl OverlapMarginGridPlan {
    /// Build a grid with leading/trailing pixel margins on interior edges.
    /// Outer edges retain their margins; the final crop ends at the canvas edge.
    ///
    /// # Errors
    /// Rejects zero dimensions, margins consuming a crop, or arithmetic overflow.
    pub fn new(
        rows: usize,
        columns: usize,
        crop_size: ImageSize,
        leading: usize,
        trailing: usize,
    ) -> Result<Self, TransformError> {
        ImageSize::new(rows, columns)?;
        ImageSize::new(crop_size.height, crop_size.width)?;
        let margins = leading
            .checked_add(trailing)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if margins >= crop_size.height || margins >= crop_size.width {
            return Err(TransformError::InvalidScaleFactor(margins));
        }
        let step_y = crop_size.height - margins;
        let step_x = crop_size.width - margins;
        let canvas_size = ImageSize::new(
            rows.checked_mul(step_y)
                .and_then(|v| v.checked_add(margins))
                .ok_or(TransformError::ImageSizeOverflow)?,
            columns
                .checked_mul(step_x)
                .and_then(|v| v.checked_add(margins))
                .ok_or(TransformError::ImageSizeOverflow)?,
        )?;
        let count = rows
            .checked_mul(columns)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mut tiles = Vec::with_capacity(count);
        for row in 0..rows {
            for column in 0..columns {
                let x = column * step_x;
                let y = row * step_y;
                tiles.push(OverlapMarginTile {
                    crop: ImageCropBox::new(x, y, x + crop_size.width, y + crop_size.height),
                    valid: ImageCropBox::new(
                        if column == 0 { 0 } else { leading },
                        if row == 0 { 0 } else { leading },
                        crop_size.width - if column + 1 == columns { 0 } else { trailing },
                        crop_size.height - if row + 1 == rows { 0 } else { trailing },
                    ),
                });
            }
        }
        Ok(Self { canvas_size, tiles })
    }
}

/// Interleaved floating pixels and a one-channel validity mask on a padded canvas.
#[derive(Clone, Debug)]
pub struct FloatResizePad {
    /// Canvas dimensions.
    pub size: ImageSize,
    /// Number of interleaved channels.
    pub channels: usize,
    /// Pixels, retaining FP32 resize precision.
    pub values: Vec<f32>,
    /// One for resized source pixels and zero for centered padding.
    pub valid: Vec<f32>,
}

impl FloatResizePad {
    /// Extract pixels and validity together, preserving row-major channel order.
    ///
    /// # Errors
    /// Rejects invalid crop bounds or inconsistent public storage.
    pub fn crop(&self, region: ImageCropBox) -> Result<(Vec<f32>, Vec<f32>), TransformError> {
        validate_image_crop_box(region, self.size)?;
        let pixels = self
            .size
            .height
            .checked_mul(self.size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let expected = pixels
            .checked_mul(self.channels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if self.channels == 0 || self.values.len() != expected || self.valid.len() != pixels {
            return Err(TransformError::InvalidBufferLength {
                expected,
                actual: self.values.len(),
            });
        }
        let mut values = Vec::new();
        let mut valid = Vec::new();
        for row in region.y_min..region.y_max {
            let start = row * self.size.width + region.x_min;
            let end = row * self.size.width + region.x_max;
            values.extend_from_slice(&self.values[start * self.channels..end * self.channels]);
            valid.extend_from_slice(&self.valid[start..end]);
        }
        Ok((values, valid))
    }
}

/// Antialiased bilinear resize with FP32 scale/truncation and centered padding.
/// Coefficients follow PyTorch's scalar-FP32 CPU interpolation boundaries.
/// Input pixels are unit-range interleaved floats. Resize results are clipped
/// to that range before padding; callers can normalize the complete canvas.
///
/// # Errors
/// Rejects invalid dimensions/storage, empty rounded dimensions or resize errors.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn resize_pad_f32_with_mask(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    pad_value: f32,
) -> Result<FloatResizePad, TransformError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    let scale = (target.height as f32 / source.height as f32)
        .min(target.width as f32 / source.width as f32);
    let resized = ImageSize::new(
        (source.height as f32 * scale) as usize,
        (source.width as f32 * scale) as usize,
    )?;
    if resized.height > target.height || resized.width > target.width || !pad_value.is_finite() {
        return Err(TransformError::InvalidSize {
            height: resized.height,
            width: resized.width,
        });
    }
    let pixels = target
        .height
        .checked_mul(target.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let len = pixels
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let resized_values =
        resize_f32_bilinear_antialiased_torch(values, source, channels, resized, false)?;
    let mut output = FloatResizePad {
        size: target,
        channels,
        values: vec![pad_value; len],
        valid: vec![0.0; pixels],
    };
    let top = (target.height - resized.height) / 2;
    let left = (target.width - resized.width) / 2;
    for row in 0..resized.height {
        for column in 0..resized.width {
            let destination = (row + top) * target.width + column + left;
            output.valid[destination] = 1.0;
            for channel in 0..channels {
                output.values[destination * channels + channel] = resized_values
                    [(row * resized.width + column) * channels + channel]
                    .clamp(0.0, 1.0);
            }
        }
    }
    Ok(output)
}

#[allow(missing_docs)]
pub fn flatten_hwc_patches(
    values: &[f32],
    size: ImageSize,
    channels: usize,
    patch: usize,
    source_layout: crate::tensor::Layout,
) -> Result<Vec<f32>, TransformError> {
    if patch == 0
        || size.height == 0
        || size.width == 0
        || !size.height.is_multiple_of(patch)
        || !size.width.is_multiple_of(patch)
    {
        return Err(TransformError::InvalidScaleFactor(patch));
    }
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }
    if !matches!(
        source_layout,
        crate::tensor::Layout::HWC | crate::tensor::Layout::CHW
    ) {
        return Err(TransformError::ImageFrameInvariant(
            "patch input must have HWC or CHW layout".into(),
        ));
    }
    let plane = size
        .height
        .checked_mul(size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let expected = plane
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    let mut patches = Vec::with_capacity(expected);
    for grid_y in 0..size.height / patch {
        for grid_x in 0..size.width / patch {
            for patch_y in 0..patch {
                for patch_x in 0..patch {
                    let y = grid_y * patch + patch_y;
                    let x = grid_x * patch + patch_x;
                    for channel in 0..channels {
                        let index = if source_layout == crate::tensor::Layout::CHW {
                            channel * plane + y * size.width + x
                        } else {
                            (y * size.width + x) * channels + channel
                        };
                        patches.push(values[index]);
                    }
                }
            }
        }
    }
    Ok(patches)
}

/// Patch-grid extraction plan for one image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchGridPlan {
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Candidate resolution selected for patch extraction.
    pub selected_size: ImageSize,
    /// Aspect-preserving size before centered padding to `selected_size`.
    pub resized_size: ImageSize,
    /// Centered padding that expands `resized_size` to `selected_size`.
    pub padding: Padding,
    /// Square patch edge length used when dividing the padded image.
    pub patch_size: usize,
    /// Number of patch rows in the selected resolution.
    pub patches_height: usize,
    /// Number of patch columns in the selected resolution.
    pub patches_width: usize,
}

impl PatchGridPlan {
    /// Returns the number of high-resolution tiled patches.
    ///
    /// # Errors
    ///
    /// Returns an error if patch-grid arithmetic overflows.
    pub fn tiled_patch_count(&self) -> Result<usize, TransformError> {
        self.patches_height
            .checked_mul(self.patches_width)
            .ok_or(TransformError::ImageSizeOverflow)
    }

    /// Returns the total patch-frame count including the resized base image.
    ///
    /// # Errors
    ///
    /// Returns an error if patch-grid arithmetic overflows.
    pub fn image_patch_count(&self) -> Result<usize, TransformError> {
        self.tiled_patch_count()?
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)
    }
}

/// Patch-grid frame output for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchGridFrames {
    /// Geometry plan used to create `frames`.
    pub plan: PatchGridPlan,
    /// Resized base image followed by high-resolution tiled patches.
    pub frames: Vec<ImageFrame>,
}

/// Patch-grid padding plan for a batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchGridBatchPlan {
    /// Maximum total patch-frame count in the batch.
    pub max_patches: usize,
    /// Per-image total patch-frame count before padding.
    pub patch_counts: Vec<usize>,
    /// Per-image number of zero patch frames needed to reach `max_patches`.
    pub padding_patches: Vec<usize>,
}

/// Selects the patch-grid target resolution for an image.
///
/// The selected resolution maximizes effective scaled image area and uses the
/// smallest wasted area as a tie-breaker, matching Transformers'
/// `select_best_resolution` helper.
///
/// # Errors
///
/// Returns an error when `possible_resolutions` is empty, any dimensions are
/// invalid, or area arithmetic overflows.
pub fn select_patch_grid_resolution(
    original_size: ImageSize,
    possible_resolutions: &[ImageSize],
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    if possible_resolutions.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates);
    }

    let original_pixels = checked_pixels(original_size)?;
    let mut best_fit = None;
    let mut max_effective_resolution = 0usize;
    let mut min_wasted_resolution = usize::MAX;

    for candidate in possible_resolutions.iter().copied() {
        validate_size(candidate)?;
        let scale_by_width = (candidate.width as f64) / (original_size.width as f64);
        let scale_by_height = (candidate.height as f64) / (original_size.height as f64);
        let scale = scale_by_width.min(scale_by_height);
        let downscaled_width = floor_scaled_dimension(original_size.width, scale)?;
        let downscaled_height = floor_scaled_dimension(original_size.height, scale)?;
        let scaled_pixels = checked_area(downscaled_height, downscaled_width)?;
        let effective_resolution = scaled_pixels.min(original_pixels);
        let candidate_pixels = checked_pixels(candidate)?;
        let wasted_resolution = candidate_pixels
            .checked_sub(effective_resolution)
            .ok_or(TransformError::ImageSizeOverflow)?;

        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = Some(candidate);
        }
    }

    best_fit.ok_or(TransformError::EmptyResolutionCandidates)
}

/// Computes the aspect-preserving image size before patch-grid padding.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn patch_grid_output_size(
    original_size: ImageSize,
    target_resolution: ImageSize,
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    validate_size(target_resolution)?;

    let width_limited = compare_ratios_less(
        target_resolution.width,
        original_size.width,
        target_resolution.height,
        original_size.height,
    )?;

    if width_limited {
        Ok(ImageSize {
            height: ceil_ratio_dimension(
                original_size.height,
                target_resolution.width,
                original_size.width,
            )?
            .min(target_resolution.height),
            width: target_resolution.width,
        })
    } else {
        Ok(ImageSize {
            height: target_resolution.height,
            width: ceil_ratio_dimension(
                original_size.width,
                target_resolution.height,
                original_size.height,
            )?
            .min(target_resolution.width),
        })
    }
}

#[allow(missing_docs)]
pub fn aspect_ratio_tile_grid(
    original_size: ImageSize,
    tile_size: ImageSize,
    minimum: usize,
    maximum: usize,
) -> Result<(usize, usize), TransformError> {
    validate_size(original_size)?;
    validate_size(tile_size)?;
    if minimum == 0 || maximum < minimum {
        return Err(TransformError::EmptyResolutionCandidates);
    }
    let source_area = checked_pixels(original_size)?;
    let tile_area = checked_pixels(tile_size)?;
    let mut grids = Vec::new();
    for columns in 1..=maximum {
        for rows in 1..=maximum {
            if columns
                .checked_mul(rows)
                .is_some_and(|count| count >= minimum && count <= maximum)
            {
                grids.push((columns, rows));
            }
        }
    }
    grids.sort_by_key(|&(columns, rows)| columns * rows);
    let ratio = original_size.width as f64 / original_size.height as f64;
    let mut best = (1, 1);
    let mut difference = f64::INFINITY;
    for (columns, rows) in grids {
        let candidate = (ratio - columns as f64 / rows as f64).abs();
        let area = tile_area
            .checked_mul(columns)
            .and_then(|area| area.checked_mul(rows))
            .ok_or(TransformError::ImageSizeOverflow)?;
        if candidate < difference
            || (candidate == difference && source_area as f64 > 0.5 * area as f64)
        {
            difference = candidate;
            best = (columns, rows);
        }
    }
    Ok(best)
}

#[allow(missing_docs)]
pub fn covering_area_tile_grid(
    original_size: ImageSize,
    tile_side: usize,
    maximum: usize,
    threshold: f64,
) -> Result<(usize, usize), TransformError> {
    validate_size(original_size)?;
    if tile_side == 0 || maximum == 0 || !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)
    {
        return Err(TransformError::EmptyResolutionCandidates);
    }
    let area = checked_pixels(original_size)? as f64;
    let mut candidates = Vec::new();
    for first in 1..=maximum {
        for second in 1..=maximum / first {
            candidates.push((first, second));
        }
    }
    candidates.sort_by_key(|&(first, second)| first * second);
    let mut evaluated = Vec::with_capacity(candidates.len());
    for (first, second) in candidates {
        let row_height = original_size.height / first;
        let column_width = original_size.width / second;
        let mut covered = 0.0;
        for row in 0..first {
            for column in 0..second {
                let height = if row + 1 == first {
                    original_size.height - row * row_height
                } else {
                    row_height
                };
                let width = if column + 1 == second {
                    original_size.width - column * column_width
                } else {
                    column_width
                };
                let longest = width.max(height) as f64;
                let shortest = width.min(height) as f64;
                covered += if longest > tile_side as f64 {
                    shortest / longest * (tile_side as f64).powi(2)
                } else {
                    longest * shortest
                };
            }
        }
        evaluated.push(((first, second), covered / area));
    }
    let sufficient = evaluated.iter().any(|(_, ratio)| *ratio > threshold);
    if sufficient {
        evaluated.retain(|(_, ratio)| *ratio > threshold);
    }
    evaluated.sort_by(|(left, left_ratio), (right, right_ratio)| {
        if sufficient {
            (left.0 * left.1)
                .cmp(&(right.0 * right.1))
                .then_with(|| right_ratio.total_cmp(left_ratio))
        } else {
            right_ratio
                .total_cmp(left_ratio)
                .then_with(|| (left.0 * left.1).cmp(&(right.0 * right.1)))
        }
    });
    evaluated
        .first()
        .map(|&(grid, _)| grid)
        .ok_or(TransformError::EmptyResolutionCandidates)
}

#[allow(missing_docs)]
pub fn aspect_ratio_tile_frames(
    image: &ImageFrame,
    tile_size: ImageSize,
    minimum: usize,
    maximum: usize,
    use_thumbnail: bool,
    decision: ResizeDecision,
) -> Result<Vec<ImageFrame>, TransformError> {
    let best = aspect_ratio_tile_grid(frame_size(image), tile_size, minimum, maximum)?;
    let canvas_size = ImageSize::new(
        tile_size
            .height
            .checked_mul(best.1)
            .ok_or(TransformError::ImageSizeOverflow)?,
        tile_size
            .width
            .checked_mul(best.0)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;
    let canvas = resize_frame_with_decision(image, canvas_size, decision, ResizeMode::Default)?;
    let mut tiles = Vec::new();
    for row in 0..best.1 {
        for column in 0..best.0 {
            tiles.push(crop_frame(
                &canvas,
                ImageCropBox::new(
                    column * tile_size.width,
                    row * tile_size.height,
                    (column + 1) * tile_size.width,
                    (row + 1) * tile_size.height,
                ),
            )?);
        }
    }
    if use_thumbnail && tiles.len() != 1 {
        tiles.push(resize_frame_with_decision(
            image,
            tile_size,
            decision,
            ResizeMode::Default,
        )?);
    }
    Ok(tiles)
}

/// Builds a Patch-grid extraction plan for one image.
///
/// # Errors
///
/// Returns an error when no resolution can be selected, `patch_size` is zero,
/// dimensions are invalid, or patch-grid arithmetic overflows.
pub fn patch_grid_plan(
    original_size: ImageSize,
    possible_resolutions: &[ImageSize],
    patch_size: usize,
) -> Result<PatchGridPlan, TransformError> {
    if patch_size == 0 {
        return Err(TransformError::InvalidScaleFactor(patch_size));
    }

    let selected_size = select_patch_grid_resolution(original_size, possible_resolutions)?;
    let resized_size = patch_grid_output_size(original_size, selected_size)?;
    let padding = centered_padding(resized_size, selected_size)?;
    let patches_height = selected_size.height.div_ceil(patch_size);
    let patches_width = selected_size.width.div_ceil(patch_size);
    if patches_height == 0 || patches_width == 0 {
        return Err(TransformError::InvalidScaleFactor(patch_size));
    }

    Ok(PatchGridPlan {
        original_size,
        selected_size,
        resized_size,
        padding,
        patch_size,
        patches_height,
        patches_width,
    })
}

/// Creates patch-grid frames for one decoded image.
///
/// The returned frames are ordered as Transformers orders them: the resized
/// base image first, followed by high-resolution tile patches in row-major
/// order.
///
/// # Errors
///
/// Returns an error when geometry selection, resizing, padding, or patch
/// extraction fails.
pub fn patch_grid_image_patches(
    frame: &ImageFrame,
    possible_resolutions: &[ImageSize],
    base_size: ImageSize,
    patch_size: usize,
    resample: ResizeFilter,
) -> Result<PatchGridFrames, TransformError> {
    patch_grid_image_patches_with_decision(
        frame,
        possible_resolutions,
        base_size,
        patch_size,
        resample.decision(),
    )
}

/// Creates patch-grid frames with an explicit resize implementation.
///
/// This is useful for processor-family wrappers that need to pin a
/// compatibility resize path while keeping the public resampling filter stable.
///
/// # Errors
///
/// Returns an error when geometry selection, resizing, padding, or patch
/// extraction fails.
pub fn patch_grid_image_patches_with_decision(
    frame: &ImageFrame,
    possible_resolutions: &[ImageSize],
    base_size: ImageSize,
    patch_size: usize,
    resize_decision: ResizeDecision,
) -> Result<PatchGridFrames, TransformError> {
    patch_grid_frames(
        frame,
        possible_resolutions,
        base_size,
        patch_size,
        resize_decision,
        None,
    )
}

/// Extract a padded global view and row-major local tiles with Pillow pad geometry.
/// Existing patch-grid extraction retains its original resize and zero-fill policy.
///
/// # Errors
/// Reports invalid geometry, fill, resampling, or crop failures.
pub fn pillow_patch_grid_image_patches(
    frame: &ImageFrame,
    possible_resolutions: &[ImageSize],
    base_size: ImageSize,
    patch_size: usize,
    resize_decision: ResizeDecision,
    fill: &[u8],
) -> Result<PatchGridFrames, TransformError> {
    patch_grid_frames(
        frame,
        possible_resolutions,
        base_size,
        patch_size,
        resize_decision,
        Some(fill),
    )
}

fn patch_grid_frames(
    frame: &ImageFrame,
    possible_resolutions: &[ImageSize],
    base_size: ImageSize,
    patch_size: usize,
    resize_decision: ResizeDecision,
    pillow_fill: Option<&[u8]>,
) -> Result<PatchGridFrames, TransformError> {
    let original_size = frame_size(frame);
    let mut plan = patch_grid_plan(original_size, possible_resolutions, patch_size)?;
    let base = if let Some(fill) = pillow_fill {
        let geometry = pillow_pad_plan(original_size, plan.selected_size)?;
        plan.resized_size = geometry.resized_size;
        plan.padding = geometry.padding;
        pillow_pad_frame(frame, base_size, fill, resize_decision)?
    } else {
        resize_frame_with_decision(frame, base_size, resize_decision, ResizeMode::Default)?
    };
    let resized = resize_frame_with_decision(
        frame,
        plan.resized_size,
        resize_decision,
        ResizeMode::Default,
    )?;
    let padded = pad_frame(&resized, plan.padding, pillow_fill.unwrap_or(&[0]))?;
    let mut frames = Vec::with_capacity(plan.image_patch_count()?);
    frames.push(base);

    for row in 0..plan.patches_height {
        let y = row
            .checked_mul(plan.patch_size)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let patch_height = plan.patch_size.min(plan.selected_size.height - y);
        for column in 0..plan.patches_width {
            let x = column
                .checked_mul(plan.patch_size)
                .ok_or(TransformError::ImageSizeOverflow)?;
            let patch_width = plan.patch_size.min(plan.selected_size.width - x);
            frames.push(crop_frame_region(
                &padded,
                x,
                y,
                ImageSize {
                    height: patch_height,
                    width: patch_width,
                },
            )?);
        }
    }

    Ok(PatchGridFrames { plan, frames })
}

/// Computes how many zero patch frames are needed to batch patch-grid outputs.
///
/// # Errors
///
/// Returns an error when `plans` is empty or patch-count arithmetic overflows.
pub fn patch_grid_batch_plan(
    plans: &[PatchGridPlan],
) -> Result<PatchGridBatchPlan, TransformError> {
    if plans.is_empty() {
        return Err(TransformError::EmptyPatchBatch);
    }

    let patch_counts = plans
        .iter()
        .map(PatchGridPlan::image_patch_count)
        .collect::<Result<Vec<_>, _>>()?;
    let max_patches = patch_counts
        .iter()
        .copied()
        .max()
        .ok_or(TransformError::EmptyPatchBatch)?;
    let padding_patches = patch_counts
        .iter()
        .map(|count| max_patches - count)
        .collect();

    Ok(PatchGridBatchPlan {
        max_patches,
        patch_counts,
        padding_patches,
    })
}
