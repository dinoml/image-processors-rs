//! Layout, attention, split-image, and aspect-ratio planning transforms.

use super::*;

/// Attention mask values downsampled for scaled dot-product attention.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionMaskDownsample {
    /// Bicubic mask size selected from `num_queries` and the source aspect ratio.
    pub downsample_size: ImageSize,
    /// Number of leading batch entries in `values`.
    ///
    /// This is usually `batch_size`. If the source contains multiple masks but
    /// fewer than `batch_size`, Diffusers repeats the whole source mask batch
    /// `batch_size` times, so this may be larger than `batch_size`.
    pub batch_entries: usize,
    /// Number of attention queries in each batch entry.
    pub num_queries: usize,
    /// Repeated value-embedding dimension for each query.
    pub value_embed_dim: usize,
    /// Row-major values with shape `[batch_entries, num_queries, value_embed_dim]`.
    pub values: Vec<f32>,
}

impl AttentionMaskDownsample {
    /// Returns the logical tensor shape of `values`.
    pub fn shape(&self) -> [usize; 3] {
        [self.batch_entries, self.num_queries, self.value_embed_dim]
    }
}

/// Patch-aligned resize plan for one image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PatchAlignedResizePlan {
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Maximum height and width before patch alignment.
    pub max_size: ImageSize,
    /// Patch height and width used by the vision tower.
    pub patch_size: ImageSize,
    /// Resized dimensions aligned up to patch multiples.
    pub resized_size: ImageSize,
    /// Number of patch rows in the resized image.
    pub patch_rows: usize,
    /// Number of patch columns in the resized image.
    pub patch_columns: usize,
}

impl PatchAlignedResizePlan {
    /// Returns the number of image patch tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if patch-grid arithmetic overflows.
    pub fn patch_count(&self) -> Result<usize, TransformError> {
        self.patch_rows
            .checked_mul(self.patch_columns)
            .ok_or(TransformError::ImageSizeOverflow)
    }
}

/// Right/bottom spatial padding plan for a batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpatialBatchPaddingPlan {
    /// Padded batch dimensions.
    pub target_size: ImageSize,
    /// Per-image resized dimensions before batch padding.
    pub image_sizes: Vec<ImageSize>,
    /// Per-image padding that expands each image to `target_size`.
    pub padding: Vec<Padding>,
}

/// Position of a target cell inside a nested image grid.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NestedImageGridTarget {
    /// Zero-based grid row.
    pub row: usize,
    /// Zero-based grid column.
    pub column: usize,
}

impl NestedImageGridTarget {
    /// Creates a target-cell position.
    pub fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }
}

/// Metadata for processors that arrange images and per-cell masks in a grid.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NestedImageGridMetadata {
    /// Number of rows in the nested image grid.
    pub rows: usize,
    /// Number of columns in each row.
    pub columns: usize,
    /// Processed image sizes in row-column order.
    pub image_sizes: Vec<Vec<ImageSize>>,
    /// Per-cell target mask in row-column order.
    pub target_mask: Vec<Vec<bool>>,
    /// Flattened target cell positions in row-major order.
    pub target_positions: Vec<NestedImageGridTarget>,
}

impl NestedImageGridMetadata {
    /// Builds validated nested image-grid metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the grid is empty, rows have inconsistent column
    /// counts, mask dimensions do not match image dimensions, image sizes are
    /// invalid, or target-position arithmetic overflows.
    pub fn new(
        image_sizes: Vec<Vec<ImageSize>>,
        target_mask: Vec<Vec<bool>>,
    ) -> Result<Self, TransformError> {
        nested_image_grid_metadata(image_sizes, target_mask)
    }
}

/// Split-image geometry plan for one image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitImagePlan {
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Size after long-edge resize.
    pub resized_size: ImageSize,
    /// Size rounded up to the vision encoder tile multiple.
    pub vision_encoder_size: ImageSize,
    /// Square split frame edge length.
    pub max_image_size: usize,
    /// Number of split rows, or zero when no split crops are emitted.
    pub rows: usize,
    /// Number of split columns, or zero when no split crops are emitted.
    pub columns: usize,
    /// Number of output frames for this image.
    pub frame_count: usize,
}

/// Row/column split metadata for a nested image batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitImageBatchMetadata {
    /// Maximum flattened frame count in any batch sample.
    pub max_frames_per_sample: usize,
    /// Per-sample flattened frame counts.
    pub sample_frame_counts: Vec<usize>,
    /// Per-original-image frame counts, unpadded per sample.
    pub image_frame_counts: Vec<Vec<usize>>,
    /// Per-original-image split rows, unpadded per sample.
    pub rows: Vec<Vec<usize>>,
    /// Per-original-image split columns, unpadded per sample.
    pub columns: Vec<Vec<usize>>,
}

/// Nested-frame batch padding and attention-mask plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NestedFrameBatchPaddingPlan {
    /// Maximum flattened frame count in any batch sample.
    pub max_frames_per_sample: usize,
    /// Padded frame dimensions.
    pub target_size: ImageSize,
    /// Actual frame sizes, unpadded per sample.
    pub frame_sizes: Vec<Vec<ImageSize>>,
    /// Per-actual-frame right/bottom padding to `target_size`.
    pub padding: Vec<Vec<Padding>>,
    /// Padded pixel-attention mask in batch-sample-frame-pixel order.
    pub pixel_attention_mask: Vec<Vec<Vec<bool>>>,
}

/// Aspect-ratio crop selection options.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct AspectRatioCropOptions {
    /// Minimum crop edge length required for aspect-ratio crops.
    pub min_crop_size: usize,
    /// Maximum number of crops along the elongated image axis.
    pub max_num_crops: usize,
    /// Minimum elongated-to-short aspect ratio required to emit crops.
    pub min_ratio_to_activate: f64,
}

/// Aspect-ratio crop rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AspectRatioCrop {
    /// Horizontal crop origin in pixels.
    pub origin_x: usize,
    /// Vertical crop origin in pixels.
    pub origin_y: usize,
    /// Actual crop dimensions after clipping at the image edge.
    pub size: ImageSize,
}

/// Aspect-ratio crop geometry plan for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct AspectRatioCropPlan {
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Number of crop rows, or zero when aspect-ratio cropping is inactive.
    pub crop_rows: usize,
    /// Number of crop columns, or zero when aspect-ratio cropping is inactive.
    pub crop_columns: usize,
    /// Nominal crop size before edge clipping, if crops are emitted.
    pub crop_size: Option<ImageSize>,
    /// Crop rectangles in row-major order.
    pub crops: Vec<AspectRatioCrop>,
}

impl AspectRatioCropPlan {
    /// Returns the number of aspect-ratio crops for this image.
    pub fn crop_count(&self) -> usize {
        self.crops.len()
    }
}

/// Batch metadata containing one aspect-ratio crop count per original image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AspectRatioCropBatchMetadata {
    /// Number of aspect-ratio crops emitted for each original image.
    pub num_crops: Vec<usize>,
}

/// Pixel position of an overlay's top-left corner.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct OverlayPosition {
    /// Horizontal destination coordinate in pixels.
    pub x: isize,
    /// Vertical destination coordinate in pixels.
    pub y: isize,
}

impl OverlayPosition {
    /// Creates an overlay position from horizontal and vertical coordinates.
    pub fn new(x: isize, y: isize) -> Self {
        Self { x, y }
    }

    /// Returns the origin position.
    pub fn origin() -> Self {
        Self::new(0, 0)
    }
}

/// Pixel-count limits used by [`ImageSize::smart_resize`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResizeLimits {
    /// Dimension multiple required by the target processor.
    pub factor: usize,
    /// Minimum allowed output pixels.
    pub min_pixels: usize,
    /// Maximum allowed output pixels.
    pub max_pixels: usize,
}

/// Downsamples attention mask planes for scaled dot-product attention.
///
/// The output repeats each downsampled query value across `value_embed_dim`,
/// matching the packed attention-bias shape expected by diffusion processors.
///
/// # Errors
///
/// Returns an error when dimensions, batch shape, mask values, or output
/// buffer sizes are invalid.
pub fn downsample_attention_mask(
    mask: &[f32],
    mask_size: ImageSize,
    batch_size: usize,
    num_queries: usize,
    value_embed_dim: usize,
) -> Result<AttentionMaskDownsample, TransformError> {
    validate_size(mask_size)?;
    validate_attention_shape(batch_size, num_queries, value_embed_dim)?;

    let mask_pixels = checked_pixels(mask_size)?;
    if mask.is_empty() {
        return Err(TransformError::EmptyMaskBatch);
    }
    if !mask.len().is_multiple_of(mask_pixels) {
        return Err(TransformError::InvalidBufferLength {
            expected: mask.len().div_ceil(mask_pixels) * mask_pixels,
            actual: mask.len(),
        });
    }

    let mask_count = mask.len() / mask_pixels;
    let downsample_size = attention_mask_downsample_size(mask_size, num_queries)?;
    let downsample_pixels = checked_pixels(downsample_size)?;
    let repeat_factor = if mask_count < batch_size {
        batch_size
    } else {
        1
    };
    let batch_entries = mask_count
        .checked_mul(repeat_factor)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let output_len = batch_entries
        .checked_mul(num_queries)
        .and_then(|value| value.checked_mul(value_embed_dim))
        .ok_or(TransformError::ImageSizeOverflow)?;

    let mut resized_masks = Vec::with_capacity(
        mask_count
            .checked_mul(downsample_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    for plane in mask.chunks_exact(mask_pixels) {
        resized_masks.extend(resize_f32_bicubic(plane, mask_size, downsample_size)?);
    }

    let mut values = Vec::with_capacity(output_len);
    for _ in 0..repeat_factor {
        for plane in resized_masks.chunks_exact(downsample_pixels) {
            append_attention_query_values(&mut values, plane, num_queries, value_embed_dim)?;
        }
    }

    Ok(AttentionMaskDownsample {
        downsample_size,
        batch_entries,
        num_queries,
        value_embed_dim,
        values,
    })
}

/// Computes patch-aligned resize dimensions.
///
/// The image is first shrunk to fit inside `max_size` when needed, preserving
/// aspect ratio with floor rounding. The result is then rounded up to
/// `patch_size` multiples.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn patch_aligned_resize_size(
    original_size: ImageSize,
    max_size: ImageSize,
    patch_size: ImageSize,
) -> Result<ImageSize, TransformError> {
    Ok(patch_aligned_resize_plan(original_size, max_size, patch_size)?.resized_size)
}

/// Builds a patch-aligned resize plan for one image.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or patch-grid arithmetic
/// overflows.
pub fn patch_aligned_resize_plan(
    original_size: ImageSize,
    max_size: ImageSize,
    patch_size: ImageSize,
) -> Result<PatchAlignedResizePlan, TransformError> {
    validate_size(original_size)?;
    validate_size(max_size)?;
    validate_size(patch_size)?;

    let ratio = ((original_size.height as f64) / (max_size.height as f64))
        .max((original_size.width as f64) / (max_size.width as f64));
    let mut resized_height = original_size.height;
    let mut resized_width = original_size.width;
    if ratio > 1.0 {
        resized_height = floor_divided_dimension(original_size.height, ratio)?.max(1);
        resized_width = floor_divided_dimension(original_size.width, ratio)?.max(1);
    }

    let patch_rows = resized_height.div_ceil(patch_size.height);
    let patch_columns = resized_width.div_ceil(patch_size.width);
    let aligned_height = patch_rows
        .checked_mul(patch_size.height)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let aligned_width = patch_columns
        .checked_mul(patch_size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;

    Ok(PatchAlignedResizePlan {
        original_size,
        max_size,
        patch_size,
        resized_size: ImageSize {
            height: aligned_height,
            width: aligned_width,
        },
        patch_rows,
        patch_columns,
    })
}

/// Computes right/bottom padding for a batch of resized images.
///
/// # Errors
///
/// Returns an error when the image list is empty or dimensions are invalid.
pub fn spatial_batch_padding_plan(
    image_sizes: &[ImageSize],
) -> Result<SpatialBatchPaddingPlan, TransformError> {
    let first = image_sizes
        .first()
        .copied()
        .ok_or(TransformError::EmptyImageBatch)?;
    let target_size = image_sizes
        .iter()
        .copied()
        .try_fold(first, |target, size| {
            validate_size(size)?;
            Ok::<_, TransformError>(ImageSize {
                height: target.height.max(size.height),
                width: target.width.max(size.width),
            })
        })?;
    let image_sizes = image_sizes.to_vec();
    let padding = image_sizes
        .iter()
        .map(|size| {
            Padding::new(
                0,
                target_size.width - size.width,
                target_size.height - size.height,
                0,
            )
        })
        .collect();

    Ok(SpatialBatchPaddingPlan {
        target_size,
        image_sizes,
        padding,
    })
}

/// Builds nested image-grid metadata from image sizes and a per-cell target mask.
///
/// This helper is intentionally processor-neutral: rows may represent examples,
/// query panels, pages, or any other nested image grouping. Target positions are
/// derived from `target_mask` in row-major order.
///
/// # Errors
///
/// Returns an error when the grid is empty, rows have inconsistent column
/// counts, mask dimensions do not match image dimensions, image sizes are
/// invalid, or target-position arithmetic overflows.
pub fn nested_image_grid_metadata(
    image_sizes: Vec<Vec<ImageSize>>,
    target_mask: Vec<Vec<bool>>,
) -> Result<NestedImageGridMetadata, TransformError> {
    let rows = image_sizes.len();
    let columns = image_sizes
        .first()
        .map(Vec::len)
        .ok_or(TransformError::EmptyNestedImageGrid)?;
    if columns == 0 {
        return Err(TransformError::EmptyNestedImageGrid);
    }

    validate_nested_image_grid_rows("image_sizes", &image_sizes, rows, columns)?;
    validate_nested_image_grid_rows("target_mask", &target_mask, rows, columns)?;

    for row in &image_sizes {
        for &size in row {
            validate_size(size)?;
        }
    }

    rows.checked_mul(columns)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut target_positions = Vec::new();
    for (row, mask_row) in target_mask.iter().enumerate() {
        for (column, is_target) in mask_row.iter().copied().enumerate() {
            if is_target {
                target_positions.push(NestedImageGridTarget::new(row, column));
            }
        }
    }

    Ok(NestedImageGridMetadata {
        rows,
        columns,
        image_sizes,
        target_mask,
        target_positions,
    })
}

/// Computes a split-image long-edge resize output size.
///
/// The longest edge is set to `longest_edge`, the other dimension preserves
/// aspect ratio with truncation, odd short-side outputs are rounded up to an
/// even value, and the final result is capped at 4096 pixels on the longest
/// side.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `longest_edge` is zero, or
/// arithmetic overflows.
pub fn split_image_resize_size(
    original_size: ImageSize,
    longest_edge: usize,
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    if longest_edge == 0 {
        return Err(TransformError::InvalidScaleFactor(longest_edge));
    }

    let mut resized = if original_size.width >= original_size.height {
        ImageSize {
            height: round_odd_up(checked_ratio_dimension(
                original_size.height,
                longest_edge,
                original_size.width,
            )?)?,
            width: longest_edge,
        }
    } else {
        ImageSize {
            height: longest_edge,
            width: round_odd_up(checked_ratio_dimension(
                original_size.width,
                longest_edge,
                original_size.height,
            )?)?,
        }
    };

    resized = scale_size_below_upper_bound(resized, IDEFICS3_MAX_IMAGE_EDGE)?;
    ImageSize::new(resized.height, resized.width)
}

/// Computes a split-image vision-encoder multiple-aligned size.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `max_image_size` is zero, or
/// arithmetic overflows.
pub fn split_image_encoder_size(
    resized_size: ImageSize,
    max_image_size: usize,
) -> Result<ImageSize, TransformError> {
    validate_size(resized_size)?;
    if max_image_size == 0 {
        return Err(TransformError::InvalidScaleFactor(max_image_size));
    }

    if resized_size.width >= resized_size.height {
        let width = round_up_to_multiple_transform(resized_size.width, max_image_size)?;
        let height = round_up_to_multiple_transform(
            checked_ratio_dimension(resized_size.height, width, resized_size.width)?,
            max_image_size,
        )?;
        ImageSize::new(height, width)
    } else {
        let height = round_up_to_multiple_transform(resized_size.height, max_image_size)?;
        let width = round_up_to_multiple_transform(
            checked_ratio_dimension(resized_size.width, height, resized_size.height)?,
            max_image_size,
        )?;
        ImageSize::new(height, width)
    }
}

/// Builds a split-image plan for one image.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `longest_edge` or
/// `max_image_size` is zero, or arithmetic overflows.
pub fn split_image_plan(
    original_size: ImageSize,
    longest_edge: usize,
    max_image_size: usize,
) -> Result<SplitImagePlan, TransformError> {
    if max_image_size == 0 {
        return Err(TransformError::InvalidScaleFactor(max_image_size));
    }
    let resized_size = split_image_resize_size(original_size, longest_edge)?;
    let vision_encoder_size = split_image_encoder_size(resized_size, max_image_size)?;
    let (rows, columns, frame_count) = if vision_encoder_size.height > max_image_size
        || vision_encoder_size.width > max_image_size
    {
        let rows = vision_encoder_size.height.div_ceil(max_image_size);
        let columns = vision_encoder_size.width.div_ceil(max_image_size);
        let frame_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_add(1))
            .ok_or(TransformError::ImageSizeOverflow)?;
        (rows, columns, frame_count)
    } else {
        (0, 0, 1)
    };

    Ok(SplitImagePlan {
        original_size,
        resized_size,
        vision_encoder_size,
        max_image_size,
        rows,
        columns,
        frame_count,
    })
}

/// Builds row/column split metadata for a nested image batch.
///
/// Each outer entry is one batch sample and each inner entry is one original
/// image in that sample.
///
/// # Errors
///
/// Returns an error when no actual images are provided or frame-count
/// arithmetic overflows.
pub fn split_image_batch_metadata(
    sample_plans: &[Vec<SplitImagePlan>],
) -> Result<SplitImageBatchMetadata, TransformError> {
    if sample_plans.is_empty() || !sample_plans.iter().any(|sample| !sample.is_empty()) {
        return Err(TransformError::EmptyImageBatch);
    }

    let mut max_frames_per_sample = 0usize;
    let mut sample_frame_counts = Vec::with_capacity(sample_plans.len());
    let mut image_frame_counts = Vec::with_capacity(sample_plans.len());
    let mut rows = Vec::with_capacity(sample_plans.len());
    let mut columns = Vec::with_capacity(sample_plans.len());

    for sample in sample_plans {
        let mut sample_total = 0usize;
        let mut sample_image_counts = Vec::with_capacity(sample.len());
        let mut sample_rows = Vec::with_capacity(sample.len());
        let mut sample_columns = Vec::with_capacity(sample.len());

        for plan in sample {
            sample_total = sample_total
                .checked_add(plan.frame_count)
                .ok_or(TransformError::ImageSizeOverflow)?;
            sample_image_counts.push(plan.frame_count);
            sample_rows.push(plan.rows);
            sample_columns.push(plan.columns);
        }

        max_frames_per_sample = max_frames_per_sample.max(sample_total);
        sample_frame_counts.push(sample_total);
        image_frame_counts.push(sample_image_counts);
        rows.push(sample_rows);
        columns.push(sample_columns);
    }

    Ok(SplitImageBatchMetadata {
        max_frames_per_sample,
        sample_frame_counts,
        image_frame_counts,
        rows,
        columns,
    })
}

/// Builds nested-batch padding and `pixel_attention_mask` metadata.
///
/// Padded frame slots are all-false masks. Actual frame masks are true for the
/// top-left valid region and false for right/bottom padding.
///
/// # Errors
///
/// Returns an error when no actual frames are provided, dimensions are invalid,
/// or mask shape arithmetic overflows.
pub fn nested_frame_batch_padding_plan(
    frame_sizes: &[Vec<ImageSize>],
) -> Result<NestedFrameBatchPaddingPlan, TransformError> {
    if frame_sizes.is_empty() || !frame_sizes.iter().any(|sample| !sample.is_empty()) {
        return Err(TransformError::EmptyImageBatch);
    }

    let mut target_size = ImageSize {
        height: 1,
        width: 1,
    };
    for sample in frame_sizes {
        for size in sample {
            validate_size(*size)?;
            target_size.height = target_size.height.max(size.height);
            target_size.width = target_size.width.max(size.width);
        }
    }
    validate_size(target_size)?;

    let max_frames_per_sample = frame_sizes
        .iter()
        .map(Vec::len)
        .max()
        .ok_or(TransformError::EmptyImageBatch)?;
    let target_pixels = checked_pixels(target_size)?;
    frame_sizes
        .len()
        .checked_mul(max_frames_per_sample)
        .and_then(|slots| slots.checked_mul(target_pixels))
        .ok_or(TransformError::ImageSizeOverflow)?;

    let mut padding = Vec::with_capacity(frame_sizes.len());
    let mut pixel_attention_mask =
        vec![vec![vec![false; target_pixels]; max_frames_per_sample]; frame_sizes.len()];
    for (sample_index, sample) in frame_sizes.iter().enumerate() {
        let mut sample_padding = Vec::with_capacity(sample.len());
        for (frame_index, size) in sample.iter().copied().enumerate() {
            sample_padding.push(Padding::new(
                0,
                target_size.width - size.width,
                target_size.height - size.height,
                0,
            ));
            write_top_left_mask(
                &mut pixel_attention_mask[sample_index][frame_index],
                size,
                target_size,
            )?;
        }
        padding.push(sample_padding);
    }

    Ok(NestedFrameBatchPaddingPlan {
        max_frames_per_sample,
        target_size,
        frame_sizes: frame_sizes.to_vec(),
        padding,
        pixel_attention_mask,
    })
}

/// Builds an aspect-ratio crop plan for one image.
///
/// The image is split along its elongated axis only when the aspect ratio is
/// at least `options.min_ratio_to_activate`. Crop counts use half-up rounding
/// of the elongated-to-short ratio and the final crop on an edge may be
/// smaller than the nominal crop size after clipping.
///
/// # Errors
///
/// Returns an error when dimensions or options are invalid, or crop geometry
/// arithmetic overflows.
pub fn aspect_ratio_crop_plan(
    original_size: ImageSize,
    options: AspectRatioCropOptions,
) -> Result<AspectRatioCropPlan, TransformError> {
    validate_size(original_size)?;
    validate_aspect_ratio_crop_options(options)?;

    let inactive_plan = || AspectRatioCropPlan {
        original_size,
        crop_rows: 0,
        crop_columns: 0,
        crop_size: None,
        crops: Vec::new(),
    };

    let (crop_rows, crop_columns) = if original_size.width >= original_size.height {
        let aspect_ratio = (original_size.width as f64) / (original_size.height as f64);
        if aspect_ratio < options.min_ratio_to_activate {
            return Ok(inactive_plan());
        }

        let ideal_columns = half_up_ratio_to_usize(original_size.width, original_size.height)?;
        let minimum_size_columns = original_size.width / options.min_crop_size;
        let crop_columns = minimum_size_columns
            .min(ideal_columns)
            .max(2)
            .min(options.max_num_crops);
        (1, crop_columns)
    } else {
        let aspect_ratio = (original_size.height as f64) / (original_size.width as f64);
        if aspect_ratio < options.min_ratio_to_activate {
            return Ok(inactive_plan());
        }

        let ideal_rows = half_up_ratio_to_usize(original_size.height, original_size.width)?;
        let minimum_size_rows = original_size.height / options.min_crop_size;
        let crop_rows = minimum_size_rows
            .min(ideal_rows)
            .max(2)
            .min(options.max_num_crops);
        (crop_rows, 1)
    };

    let crop_width = original_size.width.div_ceil(crop_columns);
    let crop_height = original_size.height.div_ceil(crop_rows);
    if crop_columns > original_size.width
        || crop_rows > original_size.height
        || crop_width.min(crop_height) < options.min_crop_size
    {
        return Ok(inactive_plan());
    }

    let crop_count = crop_rows
        .checked_mul(crop_columns)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut crops = Vec::with_capacity(crop_count);
    for row in 0..crop_rows {
        let origin_y = row
            .checked_mul(crop_height)
            .ok_or(TransformError::ImageSizeOverflow)?;
        for column in 0..crop_columns {
            let origin_x = column
                .checked_mul(crop_width)
                .ok_or(TransformError::ImageSizeOverflow)?;
            let size = ImageSize::new(
                crop_height.min(original_size.height - origin_y),
                crop_width.min(original_size.width - origin_x),
            )?;
            crops.push(AspectRatioCrop {
                origin_x,
                origin_y,
                size,
            });
        }
    }

    Ok(AspectRatioCropPlan {
        original_size,
        crop_rows,
        crop_columns,
        crop_size: Some(ImageSize {
            height: crop_height,
            width: crop_width,
        }),
        crops,
    })
}

/// Builds crop-count metadata for a batch of original images.
///
/// # Errors
///
/// Returns an error when the batch is empty, dimensions or options are invalid,
/// or crop geometry arithmetic overflows.
pub fn aspect_ratio_crop_batch_metadata(
    image_sizes: &[ImageSize],
    options: AspectRatioCropOptions,
) -> Result<AspectRatioCropBatchMetadata, TransformError> {
    if image_sizes.is_empty() {
        return Err(TransformError::EmptyImageBatch);
    }

    let mut num_crops = Vec::with_capacity(image_sizes.len());
    for size in image_sizes.iter().copied() {
        num_crops.push(aspect_ratio_crop_plan(size, options)?.crop_count());
    }

    Ok(AspectRatioCropBatchMetadata { num_crops })
}

fn validate_attention_shape(
    batch_size: usize,
    num_queries: usize,
    value_embed_dim: usize,
) -> Result<(), TransformError> {
    if batch_size > 0 && num_queries > 0 && value_embed_dim > 0 {
        Ok(())
    } else {
        Err(TransformError::InvalidAttentionShape {
            batch_size,
            num_queries,
            value_embed_dim,
        })
    }
}

fn attention_mask_downsample_size(
    mask_size: ImageSize,
    num_queries: usize,
) -> Result<ImageSize, TransformError> {
    validate_size(mask_size)?;
    if num_queries == 0 {
        return Err(TransformError::InvalidAttentionShape {
            batch_size: 1,
            num_queries,
            value_embed_dim: 1,
        });
    }

    let ratio = mask_size.width as f64 / mask_size.height as f64;
    let height = (num_queries as f64 / ratio).sqrt().floor();
    if !height.is_finite() || height < 1.0 || height > usize::MAX as f64 {
        return Err(TransformError::InvalidSmartResizeParams);
    }
    let mut height = height as usize;
    if !num_queries.is_multiple_of(height) {
        height = height
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)?;
    }
    let width = num_queries / height;
    ImageSize::new(height, width).map_err(|_| TransformError::InvalidSmartResizeParams)
}

fn append_attention_query_values(
    output: &mut Vec<f32>,
    downsampled_mask: &[f32],
    num_queries: usize,
    value_embed_dim: usize,
) -> Result<(), TransformError> {
    for value in downsampled_mask.iter().copied().take(num_queries) {
        validate_mask_value(value)?;
        for _ in 0..value_embed_dim {
            output.push(value);
        }
    }
    let padding_queries = num_queries.saturating_sub(downsampled_mask.len());
    for _ in 0..padding_queries {
        for _ in 0..value_embed_dim {
            output.push(0.0);
        }
    }
    Ok(())
}

fn write_top_left_mask(
    mask: &mut [bool],
    valid_size: ImageSize,
    target_size: ImageSize,
) -> Result<(), TransformError> {
    if valid_size.height > target_size.height || valid_size.width > target_size.width {
        return Err(TransformError::CropTooLarge {
            source_size: target_size,
            target_size: valid_size,
        });
    }
    validate_len(mask.len(), checked_pixels(target_size)?)?;

    for row in 0..valid_size.height {
        let row_start = row
            .checked_mul(target_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        for column in 0..valid_size.width {
            mask[row_start + column] = true;
        }
    }
    Ok(())
}

fn validate_aspect_ratio_crop_options(
    options: AspectRatioCropOptions,
) -> Result<(), TransformError> {
    if options.min_crop_size == 0 {
        return Err(TransformError::InvalidScaleFactor(options.min_crop_size));
    }
    if options.max_num_crops == 0 {
        return Err(TransformError::InvalidScaleFactor(options.max_num_crops));
    }
    if !options.min_ratio_to_activate.is_finite() || options.min_ratio_to_activate <= 0.0 {
        return Err(TransformError::InvalidAspectRatioCropActivationRatio(
            options.min_ratio_to_activate,
        ));
    }
    Ok(())
}

fn half_up_ratio_to_usize(numerator: usize, denominator: usize) -> Result<usize, TransformError> {
    debug_assert!(denominator > 0);
    let double_denominator = (denominator as u128)
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let value = (numerator as u128)
        .checked_mul(2)
        .and_then(|value| value.checked_add(denominator as u128))
        .ok_or(TransformError::ImageSizeOverflow)?
        / double_denominator;
    if value > usize::MAX as u128 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(value as usize)
}

pub(super) fn compare_ratios_less(
    left_numerator: usize,
    left_denominator: usize,
    right_numerator: usize,
    right_denominator: usize,
) -> Result<bool, TransformError> {
    let left = (left_numerator as u128)
        .checked_mul(right_denominator as u128)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let right = (right_numerator as u128)
        .checked_mul(left_denominator as u128)
        .ok_or(TransformError::ImageSizeOverflow)?;
    Ok(left < right)
}

fn validate_nested_image_grid_rows<T>(
    field: &'static str,
    rows: &[Vec<T>],
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), TransformError> {
    if rows.len() != expected_rows {
        return Err(TransformError::InconsistentNestedImageGridRows {
            field,
            expected: expected_rows,
            actual: rows.len(),
        });
    }

    for (row, values) in rows.iter().enumerate() {
        if values.len() != expected_columns {
            return Err(TransformError::InconsistentNestedImageGridColumns {
                field,
                row,
                expected: expected_columns,
                actual: values.len(),
            });
        }
    }

    Ok(())
}
