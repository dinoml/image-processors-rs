//! Patch-grid image extraction helpers.

use super::*;

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
    let original_size = frame_size(frame);
    let plan = patch_grid_plan(original_size, possible_resolutions, patch_size)?;
    let base = resize_frame_with_decision(frame, base_size, resize_decision, ResizeMode::Default)?;
    let resized = resize_frame_with_decision(
        frame,
        plan.resized_size,
        resize_decision,
        ResizeMode::Default,
    )?;
    let padded = pad_frame(&resized, plan.padding, &[0])?;
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
