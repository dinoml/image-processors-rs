//! Detection annotation and segmentation transforms.

use super::*;

/// Absolute bounding box in corner format `(x_min, y_min, x_max, y_max)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionBoundingBox {
    /// Left box coordinate in pixels.
    pub x_min: f32,
    /// Top box coordinate in pixels.
    pub y_min: f32,
    /// Right box coordinate in pixels.
    pub x_max: f32,
    /// Bottom box coordinate in pixels.
    pub y_max: f32,
}

impl DetectionBoundingBox {
    /// Creates an absolute corner-format bounding box.
    ///
    /// # Errors
    ///
    /// Returns an error when any coordinate is non-finite.
    pub fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Result<Self, TransformError> {
        let bbox = [x_min, y_min, x_max, y_max];
        validate_bbox_values(bbox)?;
        Ok(Self {
            x_min,
            y_min,
            x_max,
            y_max,
        })
    }

    /// Converts a COCO `(x, y, width, height)` box into clipped corner format.
    ///
    /// Returns `Ok(None)` when clipping leaves an empty box, matching DETR's
    /// behavior of dropping invalid annotations.
    ///
    /// # Errors
    ///
    /// Returns an error when image dimensions are invalid or box arithmetic is
    /// non-finite.
    pub fn from_coco_xywh_clipped(
        bbox: [f32; 4],
        image_size: ImageSize,
    ) -> Result<Option<Self>, TransformError> {
        validate_size(image_size)?;
        validate_bbox_values(bbox)?;

        let [x_min, y_min, width, height] = bbox;
        let x_max = x_min + width;
        let y_max = y_min + height;
        validate_bbox_values([x_min, y_min, x_max, y_max])?;

        let clipped = Self::new(
            x_min.clamp(0.0, image_size.width as f32),
            y_min.clamp(0.0, image_size.height as f32),
            x_max.clamp(0.0, image_size.width as f32),
            y_max.clamp(0.0, image_size.height as f32),
        )?;
        if clipped.width() > 0.0 && clipped.height() > 0.0 {
            Ok(Some(clipped))
        } else {
            Ok(None)
        }
    }

    /// Returns the box width.
    pub fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    /// Returns the box height.
    pub fn height(self) -> f32 {
        self.y_max - self.y_min
    }

    /// Scales an absolute corner-format box.
    ///
    /// # Errors
    ///
    /// Returns an error when a scale factor or resulting coordinate is
    /// non-finite.
    pub fn scale(self, scale_x: f32, scale_y: f32) -> Result<Self, TransformError> {
        validate_scale_factor_f32(scale_x)?;
        validate_scale_factor_f32(scale_y)?;
        Self::new(
            self.x_min * scale_x,
            self.y_min * scale_y,
            self.x_max * scale_x,
            self.y_max * scale_y,
        )
    }

    /// Converts the box to center format without normalizing coordinates.
    pub fn to_center(self) -> DetectionCenterBox {
        DetectionCenterBox {
            center_x: (self.x_min + self.x_max) * 0.5,
            center_y: (self.y_min + self.y_max) * 0.5,
            width: self.width(),
            height: self.height(),
        }
    }

    /// Converts the box to normalized center format relative to `image_size`.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are invalid or normalization produces a
    /// non-finite value.
    pub fn to_normalized_center(
        self,
        image_size: ImageSize,
    ) -> Result<DetectionCenterBox, TransformError> {
        validate_size(image_size)?;
        let center = self.to_center();
        DetectionCenterBox::new(
            center.center_x / image_size.width as f32,
            center.center_y / image_size.height as f32,
            center.width / image_size.width as f32,
            center.height / image_size.height as f32,
        )
    }
}

/// Bounding box in center format `(center_x, center_y, width, height)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionCenterBox {
    /// Box center x coordinate.
    pub center_x: f32,
    /// Box center y coordinate.
    pub center_y: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
}

impl DetectionCenterBox {
    /// Creates a center-format bounding box.
    ///
    /// # Errors
    ///
    /// Returns an error when any value is non-finite.
    pub fn new(
        center_x: f32,
        center_y: f32,
        width: f32,
        height: f32,
    ) -> Result<Self, TransformError> {
        validate_bbox_values([center_x, center_y, width, height])?;
        Ok(Self {
            center_x,
            center_y,
            width,
            height,
        })
    }

    /// Scales a center-format box.
    ///
    /// # Errors
    ///
    /// Returns an error when a scale factor or resulting value is non-finite.
    pub fn scale(self, scale_x: f32, scale_y: f32) -> Result<Self, TransformError> {
        validate_scale_factor_f32(scale_x)?;
        validate_scale_factor_f32(scale_y)?;
        Self::new(
            self.center_x * scale_x,
            self.center_y * scale_y,
            self.width * scale_x,
            self.height * scale_y,
        )
    }

    /// Converts the box to absolute corner format.
    ///
    /// # Errors
    ///
    /// Returns an error when corner coordinates are non-finite.
    pub fn to_corners(self) -> Result<DetectionBoundingBox, TransformError> {
        let half_width = self.width * 0.5;
        let half_height = self.height * 0.5;
        DetectionBoundingBox::new(
            self.center_x - half_width,
            self.center_y - half_height,
            self.center_x + half_width,
            self.center_y + half_height,
        )
    }

    /// Converts a normalized center-format box to absolute corner format.
    ///
    /// # Errors
    ///
    /// Returns an error when `image_size` is invalid or scaled coordinates are
    /// non-finite.
    pub fn to_absolute_corners(
        self,
        image_size: ImageSize,
    ) -> Result<DetectionBoundingBox, TransformError> {
        validate_size(image_size)?;
        self.scale(image_size.width as f32, image_size.height as f32)?
            .to_corners()
    }
}

/// One COCO detection object annotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CocoObjectAnnotation {
    /// COCO category id.
    pub category_id: i64,
    /// COCO `(x, y, width, height)` bounding box in pixels.
    pub bbox: [f32; 4],
    /// COCO object area in pixels.
    pub area: f32,
    /// Whether the object is marked as crowd.
    pub iscrowd: bool,
}

/// DETR-style detection target with absolute corner-format boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionAnnotation {
    /// COCO image id.
    pub image_id: i64,
    /// Original image size before preprocessing.
    pub original_size: ImageSize,
    /// Current image size for `boxes`.
    pub size: ImageSize,
    /// Per-object class labels.
    pub class_labels: Vec<i64>,
    /// Per-object absolute corner-format boxes.
    pub boxes: Vec<DetectionBoundingBox>,
    /// Per-object areas in pixels for the current image size.
    pub area: Vec<f32>,
    /// Per-object crowd flags.
    pub iscrowd: Vec<bool>,
}

/// DETR-style detection target with normalized center-format boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedDetectionAnnotation {
    /// COCO image id.
    pub image_id: i64,
    /// Original image size before preprocessing.
    pub original_size: ImageSize,
    /// Current image size used to normalize `boxes`.
    pub size: ImageSize,
    /// Per-object class labels.
    pub class_labels: Vec<i64>,
    /// Per-object normalized center-format boxes.
    pub boxes: Vec<DetectionCenterBox>,
    /// Per-object areas in pixels for the current image size.
    pub area: Vec<f32>,
    /// Per-object crowd flags.
    pub iscrowd: Vec<bool>,
}

/// One post-processed object-detection prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDetectionPrediction {
    /// Foreground-class probability after softmax.
    pub score: f32,
    /// Predicted class label.
    pub class_label: i64,
    /// Absolute corner-format box in target-image coordinates.
    pub bbox: DetectionBoundingBox,
}

/// One post-processed semantic-segmentation prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticSegmentationPrediction {
    /// Spatial dimensions of `class_ids` and each score plane.
    pub size: ImageSize,
    /// Number of foreground class score planes.
    pub num_labels: usize,
    /// Row-major class id for each output pixel.
    pub class_ids: Vec<i64>,
    /// Row-major class score planes in class-height-width order.
    pub scores: Vec<f32>,
}

/// Options for DETR-style instance and panoptic segmentation post-processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentationPostProcessOptions {
    /// Minimum class probability required to keep a query.
    pub score_threshold: f32,
    /// Threshold used to measure whether a weighted query mask has support.
    pub mask_threshold: f32,
    /// Minimum assigned-area to original-area ratio required to keep a segment.
    pub overlap_mask_area_threshold: f32,
    /// Optional output size for the returned segment map.
    pub target_size: Option<ImageSize>,
}

impl Default for SegmentationPostProcessOptions {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            mask_threshold: 0.5,
            overlap_mask_area_threshold: 0.8,
            target_size: None,
        }
    }
}

/// Metadata for one post-processed instance or panoptic segment.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentationSegmentInfo {
    /// Segment id written into the segmentation map.
    ///
    /// MaskFormer-style panoptic output starts at `1`; MaskFormer-style
    /// instance output and EOMT output start at `0`.
    pub id: i64,
    /// Predicted semantic class id for this segment.
    pub label_id: i64,
    /// Whether this segment belongs to a class requested for panoptic fusion.
    pub was_fused: bool,
    /// Query class probability rounded to six decimal places.
    pub score: f32,
}

/// One post-processed instance or panoptic segmentation prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentationPostProcessPrediction {
    /// Spatial dimensions of `segmentation`.
    pub size: ImageSize,
    /// Row-major segment id per pixel.
    ///
    /// Non-negative values refer to entries in `segments`; exact background and
    /// starting-id conventions depend on the upstream processor family. When
    /// every query is removed before segment composition, the map is filled
    /// with `-1` to match Transformers' no-mask result.
    pub segmentation: Vec<i64>,
    /// Segment metadata in retained-query order.
    pub segments: Vec<SegmentationSegmentInfo>,
}

/// Converts COCO object annotations into a DETR-style detection annotation.
///
/// Crowd annotations are skipped, and boxes that become empty after clipping
/// to the image bounds are dropped.
///
/// # Errors
///
/// Returns an error when the image size, annotation area, or bounding-box
/// values are invalid.
pub fn prepare_coco_detection_annotation(
    image_id: i64,
    image_size: ImageSize,
    annotations: &[CocoObjectAnnotation],
) -> Result<DetectionAnnotation, TransformError> {
    validate_size(image_size)?;

    let mut class_labels = Vec::with_capacity(annotations.len());
    let mut boxes = Vec::with_capacity(annotations.len());
    let mut area = Vec::with_capacity(annotations.len());
    let mut iscrowd = Vec::with_capacity(annotations.len());

    for annotation in annotations {
        if annotation.iscrowd {
            continue;
        }
        validate_annotation_area(annotation.area)?;
        if let Some(bbox) =
            DetectionBoundingBox::from_coco_xywh_clipped(annotation.bbox, image_size)?
        {
            class_labels.push(annotation.category_id);
            boxes.push(bbox);
            area.push(annotation.area);
            iscrowd.push(false);
        }
    }

    Ok(DetectionAnnotation {
        image_id,
        original_size: image_size,
        size: image_size,
        class_labels,
        boxes,
        area,
        iscrowd,
    })
}

/// Resizes a DETR-style detection annotation to `target_size`.
///
/// Absolute corner-format boxes are scaled by width and height ratios. Areas
/// are scaled by the product of those ratios.
///
/// # Errors
///
/// Returns an error when sizes are invalid, annotation field lengths differ,
/// or scaled geometry is non-finite.
pub fn resize_detection_annotation(
    annotation: &DetectionAnnotation,
    target_size: ImageSize,
) -> Result<DetectionAnnotation, TransformError> {
    validate_detection_annotation(annotation)?;
    validate_size(target_size)?;

    let scale_y = target_size.height as f32 / annotation.size.height as f32;
    let scale_x = target_size.width as f32 / annotation.size.width as f32;
    validate_scale_factor_f32(scale_x)?;
    validate_scale_factor_f32(scale_y)?;
    let area_scale = scale_x * scale_y;
    validate_scale_factor_f32(area_scale)?;

    let boxes = annotation
        .boxes
        .iter()
        .copied()
        .map(|bbox| bbox.scale(scale_x, scale_y))
        .collect::<Result<Vec<_>, _>>()?;
    let area = annotation
        .area
        .iter()
        .copied()
        .map(|area| {
            validate_annotation_area(area)?;
            let scaled = area * area_scale;
            validate_annotation_area(scaled)?;
            Ok(scaled)
        })
        .collect::<Result<Vec<_>, TransformError>>()?;

    Ok(DetectionAnnotation {
        image_id: annotation.image_id,
        original_size: annotation.original_size,
        size: target_size,
        class_labels: annotation.class_labels.clone(),
        boxes,
        area,
        iscrowd: annotation.iscrowd.clone(),
    })
}

/// Converts absolute corner-format detection boxes to normalized center boxes.
///
/// # Errors
///
/// Returns an error when annotation field lengths differ or normalized box
/// values are non-finite.
pub fn normalize_detection_annotation(
    annotation: &DetectionAnnotation,
) -> Result<NormalizedDetectionAnnotation, TransformError> {
    validate_detection_annotation(annotation)?;
    let boxes = annotation
        .boxes
        .iter()
        .copied()
        .map(|bbox| bbox.to_normalized_center(annotation.size))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NormalizedDetectionAnnotation {
        image_id: annotation.image_id,
        original_size: annotation.original_size,
        size: annotation.size,
        class_labels: annotation.class_labels.clone(),
        boxes,
        area: annotation.area.clone(),
        iscrowd: annotation.iscrowd.clone(),
    })
}

/// Updates normalized detection boxes after padding to `padded_size`.
///
/// This mirrors DETR's `update_bboxes` path after annotation normalization:
/// center coordinates and box sizes are scaled by the resized image size divided
/// by the padded canvas size. Right/bottom padding therefore shrinks normalized
/// values without changing absolute object locations.
///
/// # Errors
///
/// Returns an error when sizes are invalid, padding would shrink the canvas, or
/// scaled boxes are non-finite.
pub fn pad_normalized_detection_annotation(
    annotation: &NormalizedDetectionAnnotation,
    padded_size: ImageSize,
) -> Result<NormalizedDetectionAnnotation, TransformError> {
    validate_normalized_detection_annotation(annotation)?;
    validate_size(padded_size)?;
    if padded_size.height < annotation.size.height || padded_size.width < annotation.size.width {
        return Err(TransformError::CropTooLarge {
            source_size: padded_size,
            target_size: annotation.size,
        });
    }

    let scale_y = annotation.size.height as f32 / padded_size.height as f32;
    let scale_x = annotation.size.width as f32 / padded_size.width as f32;
    validate_scale_factor_f32(scale_x)?;
    validate_scale_factor_f32(scale_y)?;
    let boxes = annotation
        .boxes
        .iter()
        .copied()
        .map(|bbox| bbox.scale(scale_x, scale_y))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NormalizedDetectionAnnotation {
        image_id: annotation.image_id,
        original_size: annotation.original_size,
        size: padded_size,
        class_labels: annotation.class_labels.clone(),
        boxes,
        area: annotation.area.clone(),
        iscrowd: annotation.iscrowd.clone(),
    })
}

/// Converts DETR object-detection logits and boxes into final predictions.
///
/// Logits are interpreted as `[num_queries, num_labels_with_background]`.
/// The last class is treated as DETR's no-object background class and is
/// excluded before selecting the best foreground label. Boxes are expected in
/// normalized center format and are scaled to `target_size`.
///
/// # Errors
///
/// Returns an error when dimensions, class counts, logits, boxes, or threshold
/// values are invalid, or when the logits length does not match the query and
/// class counts.
pub fn post_process_object_detection(
    logits: &[f32],
    pred_boxes: &[DetectionCenterBox],
    num_labels_with_background: usize,
    target_size: ImageSize,
    score_threshold: f32,
) -> Result<Vec<ObjectDetectionPrediction>, TransformError> {
    validate_size(target_size)?;
    validate_score_value(score_threshold)?;
    if num_labels_with_background < 2 {
        return Err(TransformError::InvalidClassCount(
            num_labels_with_background,
        ));
    }
    let expected_logits = pred_boxes
        .len()
        .checked_mul(num_labels_with_background)
        .ok_or(TransformError::ImageSizeOverflow)?;
    validate_len(logits.len(), expected_logits)?;

    let foreground_labels = num_labels_with_background - 1;
    let mut predictions = Vec::new();
    for (query_logits, bbox) in logits
        .chunks_exact(num_labels_with_background)
        .zip(pred_boxes.iter().copied())
    {
        validate_bbox_values([bbox.center_x, bbox.center_y, bbox.width, bbox.height])?;
        let (score, class_label) = best_foreground_softmax(query_logits, foreground_labels)?;
        if score > score_threshold {
            predictions.push(ObjectDetectionPrediction {
                score,
                class_label: i64::try_from(class_label)
                    .map_err(|_| TransformError::ImageSizeOverflow)?,
                bbox: bbox.to_absolute_corners(target_size)?,
            });
        }
    }

    Ok(predictions)
}

/// Converts sigmoid-scored detection logits by retaining the best class per query.
///
/// This matches open-vocabulary processors such as Grounding DINO, OWL-ViT,
/// and OWLv2. Logits are interpreted as `[num_queries, num_labels]`; boxes use
/// normalized center format and are restored to `target_size`.
///
/// # Errors
///
/// Returns an error for invalid dimensions, logits, boxes, or thresholds.
pub fn post_process_sigmoid_best_object_detection(
    logits: &[f32],
    pred_boxes: &[DetectionCenterBox],
    num_labels: usize,
    target_size: ImageSize,
    score_threshold: f32,
) -> Result<Vec<ObjectDetectionPrediction>, TransformError> {
    validate_detection_postprocess_inputs(
        logits,
        pred_boxes,
        num_labels,
        target_size,
        score_threshold,
    )?;

    let mut predictions = Vec::new();
    for (query_logits, bbox) in logits
        .chunks_exact(num_labels)
        .zip(pred_boxes.iter().copied())
    {
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (label, logit) in query_logits.iter().copied().enumerate() {
            let score = sigmoid_score(logit)?;
            if score > best_score {
                best_score = score;
                best_label = label;
            }
        }
        if best_score > score_threshold {
            predictions.push(ObjectDetectionPrediction {
                score: best_score,
                class_label: i64::try_from(best_label)
                    .map_err(|_| TransformError::ImageSizeOverflow)?,
                bbox: bbox.to_absolute_corners(target_size)?,
            });
        }
    }
    Ok(predictions)
}

/// Converts sigmoid-scored logits by ranking flattened query-class pairs.
///
/// This matches Conditional DETR, Deformable DETR, RT-DETR, RF-DETR, and
/// Paddle document-layout processors. Multiple classes can therefore select
/// the same query box. At most `top_k` entries are considered before score
/// thresholding.
///
/// # Errors
///
/// Returns an error for invalid dimensions, logits, boxes, or thresholds.
pub fn post_process_sigmoid_top_k_object_detection(
    logits: &[f32],
    pred_boxes: &[DetectionCenterBox],
    num_labels: usize,
    target_size: ImageSize,
    score_threshold: f32,
    top_k: usize,
) -> Result<Vec<ObjectDetectionPrediction>, TransformError> {
    validate_detection_postprocess_inputs(
        logits,
        pred_boxes,
        num_labels,
        target_size,
        score_threshold,
    )?;

    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(index, logit)| Ok((sigmoid_score(logit)?, index)))
        .collect::<Result<Vec<_>, TransformError>>()?;
    ranked.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked.truncate(top_k.min(ranked.len()));

    let mut predictions = Vec::with_capacity(ranked.len());
    for (score, flat_index) in ranked {
        if score <= score_threshold {
            continue;
        }
        let query_index = flat_index / num_labels;
        let class_label = flat_index % num_labels;
        predictions.push(ObjectDetectionPrediction {
            score,
            class_label: i64::try_from(class_label)
                .map_err(|_| TransformError::ImageSizeOverflow)?,
            bbox: pred_boxes[query_index].to_absolute_corners(target_size)?,
        });
    }
    Ok(predictions)
}

fn validate_detection_postprocess_inputs(
    logits: &[f32],
    pred_boxes: &[DetectionCenterBox],
    num_labels: usize,
    target_size: ImageSize,
    score_threshold: f32,
) -> Result<(), TransformError> {
    validate_size(target_size)?;
    validate_score_value(score_threshold)?;
    if num_labels == 0 {
        return Err(TransformError::InvalidClassCount(num_labels));
    }
    let expected_logits = pred_boxes
        .len()
        .checked_mul(num_labels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    validate_len(logits.len(), expected_logits)?;
    for bbox in pred_boxes.iter().copied() {
        validate_bbox_values([bbox.center_x, bbox.center_y, bbox.width, bbox.height])?;
    }
    Ok(())
}

/// Converts DETR segmentation logits into a semantic segmentation map.
///
/// `class_logits` are interpreted as
/// `[num_queries, num_labels_with_background]`; the last class is treated as
/// DETR's no-object background class. `mask_logits` are interpreted as
/// `[num_queries, mask_size.height, mask_size.width]`. The helper computes
/// class probabilities with softmax, mask probabilities with sigmoid, combines
/// them into foreground class score planes, optionally resizes those planes to
/// `target_size`, and returns per-pixel argmax class ids.
///
/// # Errors
///
/// Returns an error when dimensions, lengths, class counts, or logit values are
/// invalid.
pub fn post_process_semantic_segmentation(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    target_size: Option<ImageSize>,
) -> Result<SemanticSegmentationPrediction, TransformError> {
    if num_labels_with_background < 2 {
        return Err(TransformError::InvalidClassCount(
            num_labels_with_background,
        ));
    }
    validate_size(mask_size)?;
    let output_size = target_size.unwrap_or(mask_size);
    validate_size(output_size)?;

    let mask_pixels = checked_pixels(mask_size)?;
    validate_len(
        class_logits.len(),
        num_queries
            .checked_mul(num_labels_with_background)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;
    validate_len(
        mask_logits.len(),
        num_queries
            .checked_mul(mask_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;

    let foreground_labels = num_labels_with_background - 1;
    let score_len = foreground_labels
        .checked_mul(mask_pixels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut scores = vec![0.0; score_len];
    for query_index in 0..num_queries {
        let class_start = query_index
            .checked_mul(num_labels_with_background)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let class_probs = foreground_softmax(
            &class_logits[class_start..class_start + num_labels_with_background],
            foreground_labels,
        )?;
        let mask_start = query_index
            .checked_mul(mask_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mask_probs = mask_logits[mask_start..mask_start + mask_pixels]
            .iter()
            .copied()
            .map(sigmoid_score)
            .collect::<Result<Vec<_>, _>>()?;
        for (label_index, class_probability) in class_probs.iter().copied().enumerate() {
            let score_start = label_index
                .checked_mul(mask_pixels)
                .ok_or(TransformError::ImageSizeOverflow)?;
            for (pixel_index, mask_probability) in mask_probs.iter().copied().enumerate() {
                scores[score_start + pixel_index] += class_probability * mask_probability;
            }
        }
    }

    if output_size != mask_size {
        let target_pixels = checked_pixels(output_size)?;
        let mut resized_scores = Vec::with_capacity(
            foreground_labels
                .checked_mul(target_pixels)
                .ok_or(TransformError::ImageSizeOverflow)?,
        );
        for score_plane in scores.chunks_exact(mask_pixels) {
            resized_scores.extend(resize_f32_bilinear(score_plane, mask_size, output_size)?);
        }
        scores = resized_scores;
    }

    let class_ids = argmax_class_ids(&scores, foreground_labels, checked_pixels(output_size)?)?;

    Ok(SemanticSegmentationPrediction {
        size: output_size,
        num_labels: foreground_labels,
        class_ids,
        scores,
    })
}

/// Converts DETR mask logits into instance segmentation predictions.
///
/// `class_logits` are interpreted as
/// `[num_queries, num_labels_with_background]`; the last class is treated as
/// DETR's no-object background class. `mask_logits` are interpreted as
/// `[num_queries, mask_size.height, mask_size.width]`. Retained masks are
/// weighted by their query score, composed by per-pixel argmax, and filtered by
/// overlap area.
///
/// # Errors
///
/// Returns an error when dimensions, lengths, class counts, logit values, or
/// thresholds are invalid.
pub fn post_process_instance_segmentation(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    options: SegmentationPostProcessOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    post_process_segmentation(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        mask_size,
        &[],
        SegmentationExecutionOptions::new(options, SegmentationMaskResize::Bilinear),
    )
}

/// Converts DETR mask logits into panoptic segmentation predictions.
///
/// This follows [`post_process_instance_segmentation`] and additionally reuses
/// one segment id for all retained queries whose label id appears in
/// `label_ids_to_fuse`, matching DETR's "stuff" class fusion behavior.
///
/// # Errors
///
/// Returns an error when dimensions, lengths, class counts, logit values, or
/// thresholds are invalid.
pub fn post_process_panoptic_segmentation(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    label_ids_to_fuse: &[i64],
    options: SegmentationPostProcessOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    post_process_segmentation(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        mask_size,
        label_ids_to_fuse,
        SegmentationExecutionOptions::new(options, SegmentationMaskResize::Bilinear),
    )
}

pub(crate) fn post_process_oneformer_panoptic_segmentation(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    label_ids_to_fuse: &[i64],
    options: SegmentationPostProcessOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    post_process_segmentation(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        mask_size,
        label_ids_to_fuse,
        SegmentationExecutionOptions::new(options, SegmentationMaskResize::TorchvisionBilinear),
    )
}

/// Computes intersection-over-union for two corner-format boxes.
///
/// # Errors
///
/// Returns an error when either box has non-finite coordinates or a negative
/// extent.
pub fn detection_box_iou(
    first: DetectionBoundingBox,
    second: DetectionBoundingBox,
) -> Result<f32, TransformError> {
    validate_detection_box_extents(first)?;
    validate_detection_box_extents(second)?;

    let intersection_width =
        (first.x_max.min(second.x_max) - first.x_min.max(second.x_min)).max(0.0);
    let intersection_height =
        (first.y_max.min(second.y_max) - first.y_min.max(second.y_min)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let first_area = first.width() * first.height();
    let second_area = second.width() * second.height();
    let union = first_area + second_area - intersection;
    if union <= 0.0 {
        Ok(0.0)
    } else {
        let iou = intersection / union;
        validate_score_value(iou)?;
        Ok(iou)
    }
}

/// Applies class-agnostic non-maximum suppression to boxes and scores.
///
/// Returned values are the kept input indices in descending score order.
///
/// # Errors
///
/// Returns an error when lengths differ, values are non-finite, boxes have
/// negative extents, or `iou_threshold` is outside `[0, 1]`.
pub fn non_max_suppression(
    boxes: &[DetectionBoundingBox],
    scores: &[f32],
    iou_threshold: f32,
) -> Result<Vec<usize>, TransformError> {
    validate_len(scores.len(), boxes.len())?;
    validate_iou_threshold(iou_threshold)?;
    for bbox in boxes.iter().copied() {
        validate_detection_box_extents(bbox)?;
    }
    for score in scores.iter().copied() {
        validate_score_value(score)?;
    }

    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });

    let mut keep = Vec::new();
    'candidate: for candidate in order {
        for kept in keep.iter().copied() {
            if detection_box_iou(boxes[candidate], boxes[kept])? > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(candidate);
    }

    Ok(keep)
}

/// Scales a corner-format detection box from one image size to another.
///
/// # Errors
///
/// Returns an error when either image size is invalid or scaled coordinates are
/// non-finite.
pub fn scale_detection_box(
    bbox: DetectionBoundingBox,
    original_size: ImageSize,
    resized_size: ImageSize,
) -> Result<DetectionBoundingBox, TransformError> {
    validate_size(original_size)?;
    validate_size(resized_size)?;
    let scale_x = resized_size.width as f32 / original_size.width as f32;
    let scale_y = resized_size.height as f32 / original_size.height as f32;
    bbox.scale(scale_x, scale_y)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SegmentationQueryPrediction {
    query_index: usize,
    score: f32,
    label_id: i64,
}

#[derive(Clone, Copy)]
enum SegmentationMaskResize {
    Bilinear,
    TorchvisionBilinear,
}

#[derive(Clone, Copy)]
struct SegmentationExecutionOptions {
    postprocess: SegmentationPostProcessOptions,
    resize: SegmentationMaskResize,
}

impl SegmentationExecutionOptions {
    fn new(postprocess: SegmentationPostProcessOptions, resize: SegmentationMaskResize) -> Self {
        Self {
            postprocess,
            resize,
        }
    }
}

fn validate_segmentation_post_process_options(
    options: SegmentationPostProcessOptions,
) -> Result<(), TransformError> {
    validate_score_value(options.score_threshold)?;
    validate_mask_value(options.mask_threshold)?;
    validate_iou_threshold(options.overlap_mask_area_threshold)?;
    if let Some(target_size) = options.target_size {
        validate_size(target_size)?;
    }
    Ok(())
}

fn post_process_segmentation(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    label_ids_to_fuse: &[i64],
    execution: SegmentationExecutionOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    let options = execution.postprocess;
    if num_labels_with_background < 2 {
        return Err(TransformError::InvalidClassCount(
            num_labels_with_background,
        ));
    }
    validate_size(mask_size)?;
    validate_segmentation_post_process_options(options)?;

    let output_size = options.target_size.unwrap_or(mask_size);
    let mask_pixels = checked_pixels(mask_size)?;
    validate_len(
        class_logits.len(),
        num_queries
            .checked_mul(num_labels_with_background)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;
    validate_len(
        mask_logits.len(),
        num_queries
            .checked_mul(mask_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;
    for value in mask_logits.iter().copied() {
        validate_score_value(value)?;
    }

    let selected_queries = select_segmentation_queries(
        class_logits,
        num_queries,
        num_labels_with_background,
        options.score_threshold,
    )?;
    let output_pixels = checked_pixels(output_size)?;
    if selected_queries.is_empty() {
        return Ok(SegmentationPostProcessPrediction {
            size: output_size,
            segmentation: vec![-1; output_pixels],
            segments: Vec::new(),
        });
    }

    let mut weighted_masks = Vec::with_capacity(
        selected_queries
            .len()
            .checked_mul(output_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    for query in selected_queries.iter().copied() {
        let mask_start = query
            .query_index
            .checked_mul(mask_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mask_end = mask_start
            .checked_add(mask_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mask_probabilities = mask_logits[mask_start..mask_end]
            .iter()
            .copied()
            .map(sigmoid_score)
            .collect::<Result<Vec<_>, _>>()?;
        let output_mask = if output_size == mask_size {
            mask_probabilities
        } else {
            match execution.resize {
                SegmentationMaskResize::Bilinear => {
                    resize_f32_bilinear(&mask_probabilities, mask_size, output_size)?
                }
                SegmentationMaskResize::TorchvisionBilinear => {
                    resize_f32_image_torchvision_bilinear(
                        &mask_probabilities,
                        mask_size,
                        output_size,
                    )?
                }
            }
        };

        for probability in output_mask {
            let weighted = probability * query.score;
            validate_mask_value(weighted)?;
            weighted_masks.push(weighted);
        }
    }

    let fused_labels = label_ids_to_fuse.iter().copied().collect::<HashSet<_>>();
    let (segmentation, segments) = compute_segmentation_segments(
        &weighted_masks,
        &selected_queries,
        output_size,
        options.mask_threshold,
        options.overlap_mask_area_threshold,
        &fused_labels,
    )?;

    Ok(SegmentationPostProcessPrediction {
        size: output_size,
        segmentation,
        segments,
    })
}

fn select_segmentation_queries(
    class_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    score_threshold: f32,
) -> Result<Vec<SegmentationQueryPrediction>, TransformError> {
    let background_label = num_labels_with_background - 1;
    let mut selected = Vec::new();
    for query_index in 0..num_queries {
        let class_start = query_index
            .checked_mul(num_labels_with_background)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let class_end = class_start
            .checked_add(num_labels_with_background)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let (score, label_id) = best_softmax_label(&class_logits[class_start..class_end])?;
        if label_id != background_label && score > score_threshold {
            selected.push(SegmentationQueryPrediction {
                query_index,
                score,
                label_id: i64::try_from(label_id).map_err(|_| TransformError::ImageSizeOverflow)?,
            });
        }
    }
    Ok(selected)
}

fn compute_segmentation_segments(
    weighted_masks: &[f32],
    queries: &[SegmentationQueryPrediction],
    output_size: ImageSize,
    mask_threshold: f32,
    overlap_mask_area_threshold: f32,
    label_ids_to_fuse: &HashSet<i64>,
) -> Result<(Vec<i64>, Vec<SegmentationSegmentInfo>), TransformError> {
    let output_pixels = checked_pixels(output_size)?;
    validate_len(
        weighted_masks.len(),
        queries
            .len()
            .checked_mul(output_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )?;

    let mut mask_labels = vec![0usize; output_pixels];
    for pixel_index in 0..output_pixels {
        let mut best_query = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for query_index in 0..queries.len() {
            let score = weighted_masks[query_index * output_pixels + pixel_index];
            validate_mask_value(score)?;
            if score > best_score {
                best_score = score;
                best_query = query_index;
            }
        }
        mask_labels[pixel_index] = best_query;
    }

    let mut segmentation = vec![0i64; output_pixels];
    let mut segments = Vec::new();
    let mut current_segment_id = 0i64;
    let mut fused_segment_ids = HashMap::<i64, i64>::new();

    for (query_index, query) in queries.iter().copied().enumerate() {
        let mask_start = query_index
            .checked_mul(output_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let mask_end = mask_start
            .checked_add(output_pixels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let weighted_mask = &weighted_masks[mask_start..mask_end];
        let mask_area = mask_labels
            .iter()
            .filter(|&&label| label == query_index)
            .count();
        let original_area = weighted_mask
            .iter()
            .copied()
            .filter(|&score| score >= mask_threshold)
            .count();
        if mask_area == 0 || original_area == 0 {
            continue;
        }

        let area_ratio = mask_area as f32 / original_area as f32;
        validate_score_value(area_ratio)?;
        if area_ratio <= overlap_mask_area_threshold {
            continue;
        }

        let should_fuse = label_ids_to_fuse.contains(&query.label_id);
        let segment_id = if let Some(segment_id) = fused_segment_ids.get(&query.label_id) {
            *segment_id
        } else {
            current_segment_id = current_segment_id
                .checked_add(1)
                .ok_or(TransformError::ImageSizeOverflow)?;
            if should_fuse {
                fused_segment_ids.insert(query.label_id, current_segment_id);
            }
            current_segment_id
        };

        for (pixel_index, label) in mask_labels.iter().copied().enumerate() {
            if label == query_index {
                segmentation[pixel_index] = segment_id;
            }
        }
        segments.push(SegmentationSegmentInfo {
            id: segment_id,
            label_id: query.label_id,
            was_fused: should_fuse,
            score: round_score_six(query.score)?,
        });
    }

    Ok((segmentation, segments))
}

fn best_foreground_softmax(
    logits: &[f32],
    foreground_labels: usize,
) -> Result<(f32, usize), TransformError> {
    let probabilities = foreground_softmax(logits, foreground_labels)?;
    let mut best_label = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (label, score) in probabilities.iter().copied().enumerate() {
        if score > best_score {
            best_score = score;
            best_label = label;
        }
    }

    Ok((best_score, best_label))
}

fn best_softmax_label(logits: &[f32]) -> Result<(f32, usize), TransformError> {
    let mut max_logit = f32::NEG_INFINITY;
    for value in logits.iter().copied() {
        validate_score_value(value)?;
        max_logit = max_logit.max(value);
    }

    let mut denominator = 0.0f32;
    for value in logits.iter().copied() {
        denominator += (value - max_logit).exp();
    }
    validate_score_value(denominator)?;
    if denominator == 0.0 {
        return Err(TransformError::InvalidScoreValue(denominator));
    }

    let mut best_label = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (label, value) in logits.iter().copied().enumerate() {
        let score = (value - max_logit).exp() / denominator;
        validate_score_value(score)?;
        if score > best_score {
            best_score = score;
            best_label = label;
        }
    }

    Ok((best_score, best_label))
}

fn foreground_softmax(
    logits: &[f32],
    foreground_labels: usize,
) -> Result<Vec<f32>, TransformError> {
    let mut max_logit = f32::NEG_INFINITY;
    for value in logits.iter().copied() {
        validate_score_value(value)?;
        max_logit = max_logit.max(value);
    }

    let mut denominator = 0.0f32;
    for value in logits.iter().copied() {
        denominator += (value - max_logit).exp();
    }
    validate_score_value(denominator)?;
    if denominator == 0.0 {
        return Err(TransformError::InvalidScoreValue(denominator));
    }

    let mut probabilities = Vec::with_capacity(foreground_labels);
    for value in logits.iter().copied().take(foreground_labels) {
        let score = (value - max_logit).exp() / denominator;
        validate_score_value(score)?;
        probabilities.push(score);
    }

    Ok(probabilities)
}

fn sigmoid_score(value: f32) -> Result<f32, TransformError> {
    validate_score_value(value)?;
    let score = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    };
    validate_score_value(score)?;
    Ok(score)
}

fn argmax_class_ids(
    scores: &[f32],
    num_labels: usize,
    pixels: usize,
) -> Result<Vec<i64>, TransformError> {
    let expected = num_labels
        .checked_mul(pixels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    validate_len(scores.len(), expected)?;
    let mut class_ids = Vec::with_capacity(pixels);
    for pixel_index in 0..pixels {
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for label_index in 0..num_labels {
            let score = scores[label_index * pixels + pixel_index];
            validate_score_value(score)?;
            if score > best_score {
                best_score = score;
                best_label = label_index;
            }
        }
        class_ids.push(i64::try_from(best_label).map_err(|_| TransformError::ImageSizeOverflow)?);
    }
    Ok(class_ids)
}

fn round_score_six(value: f32) -> Result<f32, TransformError> {
    validate_score_value(value)?;
    let rounded = (value * 1_000_000.0).round() / 1_000_000.0;
    validate_score_value(rounded)?;
    Ok(rounded)
}

fn validate_annotation_area(area: f32) -> Result<(), TransformError> {
    if area.is_finite() && area >= 0.0 {
        Ok(())
    } else {
        Err(TransformError::InvalidAnnotationArea(area))
    }
}

fn validate_detection_annotation(annotation: &DetectionAnnotation) -> Result<(), TransformError> {
    validate_size(annotation.original_size)?;
    validate_size(annotation.size)?;
    let expected = annotation.class_labels.len();
    if annotation.boxes.len() != expected
        || annotation.area.len() != expected
        || annotation.iscrowd.len() != expected
    {
        return Err(TransformError::InconsistentDetectionAnnotationLengths {
            class_labels: expected,
            boxes: annotation.boxes.len(),
            area: annotation.area.len(),
            iscrowd: annotation.iscrowd.len(),
        });
    }
    for bbox in annotation.boxes.iter().copied() {
        validate_bbox_values([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])?;
    }
    for area in annotation.area.iter().copied() {
        validate_annotation_area(area)?;
    }
    Ok(())
}

fn validate_normalized_detection_annotation(
    annotation: &NormalizedDetectionAnnotation,
) -> Result<(), TransformError> {
    validate_size(annotation.original_size)?;
    validate_size(annotation.size)?;
    let expected = annotation.class_labels.len();
    if annotation.boxes.len() != expected
        || annotation.area.len() != expected
        || annotation.iscrowd.len() != expected
    {
        return Err(TransformError::InconsistentDetectionAnnotationLengths {
            class_labels: expected,
            boxes: annotation.boxes.len(),
            area: annotation.area.len(),
            iscrowd: annotation.iscrowd.len(),
        });
    }
    for bbox in annotation.boxes.iter().copied() {
        validate_bbox_values([bbox.center_x, bbox.center_y, bbox.width, bbox.height])?;
    }
    for area in annotation.area.iter().copied() {
        validate_annotation_area(area)?;
    }
    Ok(())
}
