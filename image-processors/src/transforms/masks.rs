//! SAM-style mask and crop transforms.

use super::*;

/// Image-space point in x-y order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePoint {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl ImagePoint {
    /// Creates an image-space point.
    ///
    /// # Errors
    ///
    /// Returns an error when either coordinate is non-finite.
    pub fn new(x: f32, y: f32) -> Result<Self, TransformError> {
        validate_point_values([x, y])?;
        Ok(Self { x, y })
    }

    /// Scales an image-space point.
    ///
    /// # Errors
    ///
    /// Returns an error when a scale factor or resulting coordinate is
    /// non-finite.
    pub fn scale(self, scale_x: f32, scale_y: f32) -> Result<Self, TransformError> {
        validate_scale_factor_f32(scale_x)?;
        validate_scale_factor_f32(scale_y)?;
        Self::new(self.x * scale_x, self.y * scale_y)
    }
}

/// Absolute crop box with an associated crop layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayeredCropBox {
    /// Left crop coordinate in pixels.
    pub x_min: usize,
    /// Top crop coordinate in pixels.
    pub y_min: usize,
    /// Right crop coordinate in pixels.
    pub x_max: usize,
    /// Bottom crop coordinate in pixels.
    pub y_max: usize,
    /// Crop layer index, with zero reserved for the full-image crop.
    pub layer: usize,
}

/// Options for generating overlapping layered crop boxes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropGenerationOptions {
    /// Number of additional crop layers beyond the full-image crop.
    pub crop_layers: usize,
    /// Fractional overlap between neighboring crops.
    pub overlap_ratio: f32,
}

/// Options for filtering generated mask logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskFilterOptions {
    /// Minimum predicted IoU score required when positive.
    pub pred_iou_threshold: f32,
    /// Minimum mask stability score required when positive.
    pub stability_score_threshold: f32,
    /// Threshold used to binarize mask logits.
    pub mask_threshold: f32,
    /// Offset used when computing mask stability.
    pub stability_score_offset: f32,
    /// Absolute tolerance for crop-edge filtering.
    pub crop_edge_tolerance: f32,
}

impl Default for MaskFilterOptions {
    fn default() -> Self {
        Self {
            pred_iou_threshold: 0.88,
            stability_score_threshold: 0.95,
            mask_threshold: 0.0,
            stability_score_offset: 1.0,
            crop_edge_tolerance: 20.0,
        }
    }
}

/// Uncompressed run-length encoded binary mask.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryRleMask {
    /// Mask dimensions in height-width order.
    pub size: ImageSize,
    /// Alternating false/true run lengths in COCO-compatible column-major order.
    pub counts: Vec<usize>,
}

/// One generated binary mask prediction after NMS.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedMaskPrediction {
    /// Predicted mask quality score.
    pub score: f32,
    /// Absolute corner-format mask box.
    pub bbox: DetectionBoundingBox,
    /// Binary mask in row-major order.
    pub mask: Vec<bool>,
    /// Uncompressed RLE for the same binary mask.
    pub rle: BinaryRleMask,
}

/// One generated mask retained after crop-level filtering.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredMask {
    /// Predicted mask quality score.
    pub score: f32,
    /// Mask stability score computed from logits.
    pub stability_score: f32,
    /// Absolute corner-format mask box in original-image coordinates.
    pub bbox: DetectionBoundingBox,
    /// Crop-local corner-format mask box before offsetting.
    pub crop_bbox: DetectionBoundingBox,
    /// Binary mask padded to the original image size.
    pub mask: Vec<bool>,
    /// Uncompressed RLE for the padded binary mask.
    pub rle: BinaryRleMask,
}

/// Scales an image-space point from one image size to another.
///
/// # Errors
///
/// Returns an error when either image size is invalid, the scale factors are
/// non-finite, or the scaled point is invalid.
pub fn scale_image_point(
    point: ImagePoint,
    original_size: ImageSize,
    resized_size: ImageSize,
) -> Result<ImagePoint, TransformError> {
    validate_size(original_size)?;
    validate_size(resized_size)?;
    let scale_x = resized_size.width as f32 / original_size.width as f32;
    let scale_y = resized_size.height as f32 / original_size.height as f32;
    point.scale(scale_x, scale_y)
}

/// Builds a normalized square point grid with samples at each cell center.
///
/// # Errors
///
/// Returns an error when `points_per_side` is zero or the grid size overflows.
pub fn normalized_point_grid(points_per_side: usize) -> Result<Vec<ImagePoint>, TransformError> {
    if points_per_side == 0 {
        return Err(TransformError::InvalidScaleFactor(points_per_side));
    }
    let denominator = points_per_side
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let offset = 1.0 / denominator as f32;
    let mut points = Vec::with_capacity(
        points_per_side
            .checked_mul(points_per_side)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    for y in 0..points_per_side {
        for x in 0..points_per_side {
            points.push(ImagePoint::new(
                offset + x as f32 / points_per_side as f32,
                offset + y as f32 / points_per_side as f32,
            )?);
        }
    }
    Ok(points)
}

/// Generates overlapping layered crop boxes for automatic mask generation.
///
/// The first crop is always the full image at layer `0`. Additional layers use
/// `2 ** layer` crops per side with the same overlap formula as Transformers.
///
/// # Errors
///
/// Returns an error when dimensions or options are invalid, or crop arithmetic
/// overflows.
pub fn generate_layered_crop_boxes(
    image_size: ImageSize,
    options: CropGenerationOptions,
) -> Result<Vec<LayeredCropBox>, TransformError> {
    validate_size(image_size)?;
    validate_overlap_ratio(options.overlap_ratio)?;

    let mut boxes = vec![LayeredCropBox {
        x_min: 0,
        y_min: 0,
        x_max: image_size.width,
        y_max: image_size.height,
        layer: 0,
    }];
    let short_side = image_size.height.min(image_size.width);

    for layer_index in 0..options.crop_layers {
        let layer = layer_index
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let shift = u32::try_from(layer).map_err(|_| TransformError::ImageSizeOverflow)?;
        let crops_per_side = 1usize
            .checked_shl(shift)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let overlap = layered_crop_overlap(options.overlap_ratio, short_side, crops_per_side)?;
        let crop_width = overlap
            .checked_mul(crops_per_side - 1)
            .and_then(|span| span.checked_add(image_size.width))
            .ok_or(TransformError::ImageSizeOverflow)?
            .div_ceil(crops_per_side);
        let crop_height = overlap
            .checked_mul(crops_per_side - 1)
            .and_then(|span| span.checked_add(image_size.height))
            .ok_or(TransformError::ImageSizeOverflow)?
            .div_ceil(crops_per_side);
        let step_x = crop_width
            .checked_sub(overlap)
            .ok_or(TransformError::InvalidOverlapRatio(options.overlap_ratio))?;
        let step_y = crop_height
            .checked_sub(overlap)
            .ok_or(TransformError::InvalidOverlapRatio(options.overlap_ratio))?;

        for x_index in 0..crops_per_side {
            let x_min = step_x
                .checked_mul(x_index)
                .ok_or(TransformError::ImageSizeOverflow)?;
            for y_index in 0..crops_per_side {
                let y_min = step_y
                    .checked_mul(y_index)
                    .ok_or(TransformError::ImageSizeOverflow)?;
                boxes.push(LayeredCropBox {
                    x_min,
                    y_min,
                    x_max: x_min
                        .checked_add(crop_width)
                        .ok_or(TransformError::ImageSizeOverflow)?
                        .min(image_size.width),
                    y_max: y_min
                        .checked_add(crop_height)
                        .ok_or(TransformError::ImageSizeOverflow)?
                        .min(image_size.height),
                    layer,
                });
            }
        }
    }

    Ok(boxes)
}

/// Computes a bounding box for a binary mask.
///
/// Returns `Ok(None)` for an empty mask. Non-empty boxes use the maximum true
/// pixel index for `x_max` and `y_max`.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, the mask length is wrong, or
/// arithmetic overflows.
pub fn binary_mask_to_box(
    mask: &[bool],
    size: ImageSize,
) -> Result<Option<DetectionBoundingBox>, TransformError> {
    validate_size(size)?;
    validate_len(mask.len(), checked_pixels(size)?)?;

    let mut x_min = size.width;
    let mut y_min = size.height;
    let mut x_max = 0usize;
    let mut y_max = 0usize;
    let mut found = false;
    for y in 0..size.height {
        let row_start = y
            .checked_mul(size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        for x in 0..size.width {
            if mask[row_start + x] {
                found = true;
                x_min = x_min.min(x);
                y_min = y_min.min(y);
                x_max = x_max.max(x);
                y_max = y_max.max(y);
            }
        }
    }

    if found {
        Ok(Some(DetectionBoundingBox::new(
            x_min as f32,
            y_min as f32,
            x_max as f32,
            y_max as f32,
        )?))
    } else {
        Ok(None)
    }
}

/// Binarizes mask logits with a strict greater-than threshold.
///
/// # Errors
///
/// Returns an error when the threshold or any mask value is non-finite.
pub fn binarize_mask(values: &[f32], threshold: f32) -> Result<Vec<bool>, TransformError> {
    validate_mask_value(threshold)?;
    values
        .iter()
        .copied()
        .map(|value| {
            validate_mask_value(value)?;
            Ok(value > threshold)
        })
        .collect()
}

/// Computes a mask stability score from two thresholded logit regions.
///
/// # Errors
///
/// Returns an error when `threshold`, `offset`, or any mask value is non-finite.
pub fn mask_stability_score(
    values: &[f32],
    threshold: f32,
    offset: f32,
) -> Result<f32, TransformError> {
    validate_mask_value(threshold)?;
    validate_scale_factor_f32(offset)?;

    let mut intersections = 0usize;
    let mut unions = 0usize;
    for value in values.iter().copied() {
        validate_mask_value(value)?;
        if value > threshold + offset {
            intersections += 1;
        }
        if value > threshold - offset {
            unions += 1;
        }
    }

    if unions == 0 {
        Ok(0.0)
    } else {
        Ok(intersections as f32 / unions as f32)
    }
}

/// Encodes a binary mask as uncompressed column-major RLE.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, the mask length is wrong, or
/// arithmetic overflows.
pub fn binary_mask_to_rle(mask: &[bool], size: ImageSize) -> Result<BinaryRleMask, TransformError> {
    validate_size(size)?;
    let pixels = checked_pixels(size)?;
    validate_len(mask.len(), pixels)?;

    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0usize;
    for x in 0..size.width {
        for y in 0..size.height {
            let index = y
                .checked_mul(size.width)
                .and_then(|row| row.checked_add(x))
                .ok_or(TransformError::ImageSizeOverflow)?;
            let value = mask[index];
            if value == current {
                run = run
                    .checked_add(1)
                    .ok_or(TransformError::ImageSizeOverflow)?;
            } else {
                counts.push(run);
                run = 1;
                current = value;
            }
        }
    }
    counts.push(run);

    Ok(BinaryRleMask { size, counts })
}

/// Decodes an uncompressed column-major RLE mask into row-major binary pixels.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, RLE counts overflow, or the
/// decoded length does not match `rle.size`.
pub fn binary_rle_to_mask(rle: &BinaryRleMask) -> Result<Vec<bool>, TransformError> {
    validate_size(rle.size)?;
    let pixels = checked_pixels(rle.size)?;
    let mut mask = vec![false; pixels];
    let mut flat_index = 0usize;
    let mut value = false;
    for count in rle.counts.iter().copied() {
        let next = flat_index
            .checked_add(count)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if next > pixels {
            return Err(TransformError::InvalidBufferLength {
                expected: pixels,
                actual: next,
            });
        }
        if value {
            for encoded_index in flat_index..next {
                let y = encoded_index % rle.size.height;
                let x = encoded_index / rle.size.height;
                let row_major = y
                    .checked_mul(rle.size.width)
                    .and_then(|row| row.checked_add(x))
                    .ok_or(TransformError::ImageSizeOverflow)?;
                mask[row_major] = true;
            }
        }
        flat_index = next;
        value = !value;
    }
    if flat_index != pixels {
        return Err(TransformError::InvalidBufferLength {
            expected: pixels,
            actual: flat_index,
        });
    }

    Ok(mask)
}

/// Resizes one padded mask-logit plane back to original image dimensions.
///
/// This resize sequence covers padded encoder canvases: resize low-resolution
/// mask logits to the padded canvas, remove right/bottom padding with
/// `reshaped_input_size`, then resize to `original_size`.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, input length is wrong, values
/// are non-finite, or `reshaped_input_size` exceeds `pad_size`.
pub fn resize_padded_mask_logits(
    mask_logits: &[f32],
    mask_size: ImageSize,
    pad_size: ImageSize,
    reshaped_input_size: ImageSize,
    original_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    validate_size(mask_size)?;
    validate_size(pad_size)?;
    validate_size(reshaped_input_size)?;
    validate_size(original_size)?;
    if reshaped_input_size.height > pad_size.height || reshaped_input_size.width > pad_size.width {
        return Err(TransformError::CropTooLarge {
            source_size: pad_size,
            target_size: reshaped_input_size,
        });
    }

    let padded_logits = resize_f32_bilinear(mask_logits, mask_size, pad_size)?;
    let mut unpadded_logits = Vec::with_capacity(checked_pixels(reshaped_input_size)?);
    for y in 0..reshaped_input_size.height {
        let row_start = y
            .checked_mul(pad_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        unpadded_logits
            .extend_from_slice(&padded_logits[row_start..row_start + reshaped_input_size.width]);
    }

    resize_f32_bilinear(&unpadded_logits, reshaped_input_size, original_size)
}

/// Resizes and binarizes one padded mask-logit plane.
///
/// # Errors
///
/// Returns an error when resizing inputs are invalid or `mask_threshold` is
/// non-finite.
pub fn post_process_binary_mask(
    mask_logits: &[f32],
    mask_size: ImageSize,
    pad_size: ImageSize,
    reshaped_input_size: ImageSize,
    original_size: ImageSize,
    mask_threshold: f32,
) -> Result<Vec<bool>, TransformError> {
    let resized = resize_padded_mask_logits(
        mask_logits,
        mask_size,
        pad_size,
        reshaped_input_size,
        original_size,
    )?;
    binarize_mask(&resized, mask_threshold)
}

/// Returns true when a crop-local box touches a crop edge.
///
/// Edges that coincide with the original image boundary are ignored, matching
/// crop-based automatic mask generation filters.
///
/// # Errors
///
/// Returns an error when the box, crop, original size, or tolerance is invalid.
pub fn box_near_crop_edge(
    bbox: DetectionBoundingBox,
    crop_box: LayeredCropBox,
    original_size: ImageSize,
    tolerance: f32,
) -> Result<bool, TransformError> {
    validate_detection_box_extents(bbox)?;
    validate_crop_box(crop_box, original_size)?;
    validate_scale_factor_f32(tolerance)?;

    let absolute = offset_crop_bbox(bbox, crop_box)?;
    let crop_edges = [
        crop_box.x_min as f32,
        crop_box.y_min as f32,
        crop_box.x_max as f32,
        crop_box.y_max as f32,
    ];
    let image_edges = [
        0.0,
        0.0,
        original_size.width as f32,
        original_size.height as f32,
    ];
    let box_edges = [
        absolute.x_min,
        absolute.y_min,
        absolute.x_max,
        absolute.y_max,
    ];

    Ok(box_edges.iter().zip(crop_edges).zip(image_edges).any(
        |((box_edge, crop_edge), image_edge)| {
            is_close(*box_edge, crop_edge, tolerance) && !is_close(*box_edge, image_edge, tolerance)
        },
    ))
}

/// Pads a crop-local binary mask back to the original image canvas.
///
/// The returned mask is row-major with dimensions `original_size`.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, the crop is outside the
/// original image, or `mask` length does not match `mask_size`.
pub fn pad_crop_mask(
    mask: &[bool],
    mask_size: ImageSize,
    crop_box: LayeredCropBox,
    original_size: ImageSize,
) -> Result<Vec<bool>, TransformError> {
    validate_size(mask_size)?;
    validate_crop_box(crop_box, original_size)?;
    validate_len(mask.len(), checked_pixels(mask_size)?)?;
    let crop_size = crop_box_size(crop_box)?;
    if crop_size != mask_size {
        return Err(TransformError::IncompatibleFrameSize {
            frame: "crop mask",
            expected: crop_size,
            actual: mask_size,
        });
    }

    if crop_box.x_min == 0
        && crop_box.y_min == 0
        && crop_box.x_max == original_size.width
        && crop_box.y_max == original_size.height
    {
        return Ok(mask.to_vec());
    }

    let mut padded = vec![false; checked_pixels(original_size)?];
    for y in 0..mask_size.height {
        let source_start = y
            .checked_mul(mask_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let target_start = (crop_box.y_min + y)
            .checked_mul(original_size.width)
            .and_then(|row| row.checked_add(crop_box.x_min))
            .ok_or(TransformError::ImageSizeOverflow)?;
        padded[target_start..target_start + mask_size.width]
            .copy_from_slice(&mask[source_start..source_start + mask_size.width]);
    }

    Ok(padded)
}

/// Filters generated crop masks and pads retained masks to the original canvas.
///
/// `mask_logits` are interpreted as `num_masks` row-major planes of
/// `mask_size`. The crop box describes where those planes sit in the original
/// image. Retained masks pass predicted-IoU filtering, stability filtering,
/// binarization, and crop-edge filtering before being padded and encoded as
/// RLE.
///
/// # Errors
///
/// Returns an error when dimensions, crop geometry, thresholds, scores, mask
/// values, or buffer lengths are invalid.
pub fn filter_generated_masks(
    mask_logits: &[f32],
    mask_size: ImageSize,
    iou_scores: &[f32],
    crop_box: LayeredCropBox,
    original_size: ImageSize,
    options: MaskFilterOptions,
) -> Result<Vec<FilteredMask>, TransformError> {
    validate_size(mask_size)?;
    validate_crop_box(crop_box, original_size)?;
    validate_mask_filter_options(options)?;
    let pixels = checked_pixels(mask_size)?;
    let expected_logits = iou_scores
        .len()
        .checked_mul(pixels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    validate_len(mask_logits.len(), expected_logits)?;

    let mut filtered = Vec::new();
    for (mask_index, score) in iou_scores.iter().copied().enumerate() {
        validate_score_value(score)?;
        if options.pred_iou_threshold > 0.0 && score <= options.pred_iou_threshold {
            continue;
        }
        let start = mask_index
            .checked_mul(pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mask_plane = &mask_logits[start..start + pixels];
        let stability_score = mask_stability_score(
            mask_plane,
            options.mask_threshold,
            options.stability_score_offset,
        )?;
        if options.stability_score_threshold > 0.0
            && stability_score <= options.stability_score_threshold
        {
            continue;
        }

        let mask = binarize_mask(mask_plane, options.mask_threshold)?;
        let crop_bbox = binary_mask_to_box(&mask, mask_size)?
            .unwrap_or(DetectionBoundingBox::new(0.0, 0.0, 0.0, 0.0)?);
        if box_near_crop_edge(
            crop_bbox,
            crop_box,
            original_size,
            options.crop_edge_tolerance,
        )? {
            continue;
        }

        let padded_mask = pad_crop_mask(&mask, mask_size, crop_box, original_size)?;
        let rle = binary_mask_to_rle(&padded_mask, original_size)?;
        filtered.push(FilteredMask {
            score,
            stability_score,
            bbox: offset_crop_bbox(crop_bbox, crop_box)?,
            crop_bbox,
            mask: padded_mask,
            rle,
        });
    }

    Ok(filtered)
}

/// Applies generated-mask NMS and decodes kept RLE masks.
///
/// This mirrors the final class-agnostic NMS step used after crop-based mask
/// generation. Returned predictions are sorted by descending score.
///
/// # Errors
///
/// Returns an error when input lengths differ, boxes or scores are invalid,
/// RLE masks fail to decode, or `nms_threshold` is outside `[0, 1]`.
pub fn post_process_generated_masks(
    rle_masks: &[BinaryRleMask],
    iou_scores: &[f32],
    mask_boxes: &[DetectionBoundingBox],
    nms_threshold: f32,
) -> Result<Vec<GeneratedMaskPrediction>, TransformError> {
    validate_len(rle_masks.len(), mask_boxes.len())?;
    validate_len(iou_scores.len(), mask_boxes.len())?;
    let keep = non_max_suppression(mask_boxes, iou_scores, nms_threshold)?;

    keep.into_iter()
        .map(|index| {
            Ok(GeneratedMaskPrediction {
                score: iou_scores[index],
                bbox: mask_boxes[index],
                mask: binary_rle_to_mask(&rle_masks[index])?,
                rle: rle_masks[index].clone(),
            })
        })
        .collect()
}

/// Converts finite mask values to binary floating-point values at a `0.5` threshold.
///
/// # Errors
///
/// Returns an error when any mask value is not finite.
pub fn binarize_mask_to_unit_f32(values: &[f32]) -> Result<Vec<f32>, TransformError> {
    values
        .iter()
        .copied()
        .map(|value| {
            validate_mask_value(value)?;
            Ok(if value < 0.5 { 0.0 } else { 1.0 })
        })
        .collect()
}

fn validate_mask_filter_options(options: MaskFilterOptions) -> Result<(), TransformError> {
    validate_score_value(options.pred_iou_threshold)?;
    validate_score_value(options.stability_score_threshold)?;
    validate_mask_value(options.mask_threshold)?;
    validate_scale_factor_f32(options.stability_score_offset)?;
    validate_scale_factor_f32(options.crop_edge_tolerance)?;
    Ok(())
}

fn validate_crop_box(
    crop_box: LayeredCropBox,
    original_size: ImageSize,
) -> Result<(), TransformError> {
    validate_size(original_size)?;
    if crop_box.x_min <= crop_box.x_max
        && crop_box.y_min <= crop_box.y_max
        && crop_box.x_max <= original_size.width
        && crop_box.y_max <= original_size.height
        && crop_box.x_min < crop_box.x_max
        && crop_box.y_min < crop_box.y_max
    {
        Ok(())
    } else {
        Err(TransformError::InvalidCropBox {
            crop_box,
            image_size: original_size,
        })
    }
}

fn crop_box_size(crop_box: LayeredCropBox) -> Result<ImageSize, TransformError> {
    ImageSize::new(
        crop_box.y_max - crop_box.y_min,
        crop_box.x_max - crop_box.x_min,
    )
}

fn offset_crop_bbox(
    bbox: DetectionBoundingBox,
    crop_box: LayeredCropBox,
) -> Result<DetectionBoundingBox, TransformError> {
    DetectionBoundingBox::new(
        bbox.x_min + crop_box.x_min as f32,
        bbox.y_min + crop_box.y_min as f32,
        bbox.x_max + crop_box.x_min as f32,
        bbox.y_max + crop_box.y_min as f32,
    )
}

fn is_close(left: f32, right: f32, tolerance: f32) -> bool {
    (left - right).abs() <= tolerance
}

fn layered_crop_overlap(
    overlap_ratio: f32,
    short_side: usize,
    crops_per_side: usize,
) -> Result<usize, TransformError> {
    let overlap = overlap_ratio * short_side as f32 * (2.0 / crops_per_side as f32);
    if !overlap.is_finite() || overlap < 0.0 || overlap > usize::MAX as f32 {
        return Err(TransformError::InvalidOverlapRatio(overlap_ratio));
    }
    Ok(overlap as usize)
}
