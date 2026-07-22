use std::collections::HashMap;

use crate::transforms::{
    post_process_semantic_segmentation, resize_f32_image_bilinear,
    resize_f32_image_torchvision_bilinear, ImageSize, SegmentationPostProcessOptions,
    SegmentationPostProcessPrediction, SegmentationSegmentInfo, SemanticSegmentationPrediction,
    TransformError,
};

#[derive(Clone, Copy, Debug)]
struct QueryPrediction {
    query_index: usize,
    label_id: i64,
    score: f32,
}

pub(super) fn resize_mask_planes_bilinear(
    values: &[f32],
    plane_count: usize,
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    let source_pixels = pixels(source_size)?;
    validate_len(values.len(), checked_mul(plane_count, source_pixels)?)?;
    let mut resized = Vec::with_capacity(checked_mul(plane_count, pixels(target_size)?)?);
    for plane in values.chunks_exact(source_pixels) {
        resized.extend(resize_f32_image_bilinear(plane, source_size, target_size)?);
    }
    Ok(resized)
}

pub(super) fn post_process_oneformer_semantic(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    target_size: Option<ImageSize>,
) -> Result<SemanticSegmentationPrediction, TransformError> {
    let prediction = post_process_semantic_segmentation(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        mask_size,
        None,
    )?;
    let Some(output_size) = target_size else {
        return Ok(prediction);
    };
    if output_size == mask_size {
        return Ok(prediction);
    }

    let mask_pixels = pixels(mask_size)?;
    let output_pixels = pixels(output_size)?;
    let mut scores = Vec::with_capacity(checked_mul(prediction.num_labels, output_pixels)?);
    for plane in prediction.scores.chunks_exact(mask_pixels) {
        scores.extend(resize_f32_image_torchvision_bilinear(
            plane,
            mask_size,
            output_size,
        )?);
    }
    let mut class_ids = Vec::with_capacity(output_pixels);
    for pixel_index in 0..output_pixels {
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for label in 0..prediction.num_labels {
            let score = scores[label * output_pixels + pixel_index];
            validate_score(score)?;
            if score > best_score {
                best_label = label;
                best_score = score;
            }
        }
        class_ids.push(i64::try_from(best_label).map_err(|_| TransformError::ImageSizeOverflow)?);
    }
    Ok(SemanticSegmentationPrediction {
        size: output_size,
        num_labels: prediction.num_labels,
        class_ids,
        scores,
    })
}

pub(super) fn restore_eomt_mask_logits(
    values: &[f32],
    plane_count: usize,
    source_size: ImageSize,
    canvas_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    let canvas_logits = resize_mask_planes_bilinear(values, plane_count, source_size, canvas_size)?;
    let requested_crop_size = eomt_aspect_resize_size(target_size, canvas_size)?;
    // Python's top-left tensor slice truncates an overlong stop at the canvas
    // boundary. This matters when the rounded shortest edge already equals the
    // source edge while the unrounded longest-edge cap is slightly smaller.
    let crop_size = ImageSize::new(
        requested_crop_size.height.min(canvas_size.height),
        requested_crop_size.width.min(canvas_size.width),
    )?;

    let canvas_pixels = pixels(canvas_size)?;
    let crop_pixels = pixels(crop_size)?;
    let mut restored = Vec::with_capacity(checked_mul(plane_count, pixels(target_size)?)?);
    let mut cropped = Vec::with_capacity(crop_pixels);
    for plane in canvas_logits.chunks_exact(canvas_pixels) {
        cropped.clear();
        for row in plane.chunks_exact(canvas_size.width).take(crop_size.height) {
            cropped.extend_from_slice(&row[..crop_size.width]);
        }
        restored.extend(resize_f32_image_bilinear(&cropped, crop_size, target_size)?);
    }
    Ok(restored)
}

pub(super) fn post_process_maskformer_instance(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
    options: SegmentationPostProcessOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    validate_options(options)?;
    validate_query_mask_inputs(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        mask_size,
    )?;
    let foreground_labels = num_labels_with_background - 1;
    let mask_pixels = pixels(mask_size)?;
    let output_size = options.target_size.unwrap_or(mask_size);
    let output_pixels = pixels(output_size)?;

    let mut candidates = Vec::with_capacity(checked_mul(num_queries, foreground_labels)?);
    for query_index in 0..num_queries {
        let start = checked_mul(query_index, num_labels_with_background)?;
        let probabilities = softmax(&class_logits[start..start + num_labels_with_background])?;
        for (label_id, score) in probabilities[..foreground_labels]
            .iter()
            .copied()
            .enumerate()
        {
            candidates.push(QueryPrediction {
                query_index,
                label_id: i64::try_from(label_id).map_err(|_| TransformError::ImageSizeOverflow)?,
                score,
            });
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.query_index.cmp(&right.query_index))
            .then_with(|| left.label_id.cmp(&right.label_id))
    });
    candidates.truncate(num_queries);

    let mut segmentation = vec![-1; output_pixels];
    let mut segments = Vec::new();
    for candidate in candidates {
        let start = checked_mul(candidate.query_index, mask_pixels)?;
        let mask = &mask_logits[start..start + mask_pixels];
        let mask_score = binary_mask_score(mask)?;
        let score = candidate.score * mask_score;
        validate_score(score)?;
        if score < options.score_threshold {
            continue;
        }

        let binary = mask
            .iter()
            .map(|&value| u8::from(value > 0.0))
            .collect::<Vec<_>>();
        let output_mask = if output_size == mask_size {
            binary
        } else {
            resize_binary_mask_nearest(&binary, mask_size, output_size)?
        };
        if output_mask.iter().all(|&value| value == 0) {
            continue;
        }

        let id = i64::try_from(segments.len()).map_err(|_| TransformError::ImageSizeOverflow)?;
        for (pixel, &is_set) in segmentation.iter_mut().zip(&output_mask) {
            if is_set != 0 {
                *pixel = id;
            }
        }
        segments.push(SegmentationSegmentInfo {
            id,
            label_id: candidate.label_id,
            was_fused: false,
            score: round_six(score)?,
        });
    }

    Ok(SegmentationPostProcessPrediction {
        size: output_size,
        segmentation,
        segments,
    })
}

pub(super) fn post_process_eomt_instance(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    output_size: ImageSize,
    score_threshold: f32,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    validate_score(score_threshold)?;
    validate_query_mask_inputs(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        output_size,
    )?;
    let foreground_labels = num_labels_with_background - 1;
    let output_pixels = pixels(output_size)?;
    let mut segmentation = vec![-1; output_pixels];
    let mut segments = Vec::new();

    for query_index in 0..num_queries {
        let class_start = checked_mul(query_index, num_labels_with_background)?;
        let probabilities =
            softmax(&class_logits[class_start..class_start + num_labels_with_background])?;
        let (label_id, class_score) = best_label(&probabilities[..foreground_labels]);
        let mask_start = checked_mul(query_index, output_pixels)?;
        let mask = &mask_logits[mask_start..mask_start + output_pixels];
        let score = class_score * binary_mask_score(mask)?;
        validate_score(score)?;
        if score < score_threshold || mask.iter().all(|&value| value <= 0.0) {
            continue;
        }

        let id = i64::try_from(segments.len()).map_err(|_| TransformError::ImageSizeOverflow)?;
        for (pixel, &value) in segmentation.iter_mut().zip(mask) {
            if value > 0.0 {
                *pixel = id;
            }
        }
        segments.push(SegmentationSegmentInfo {
            id,
            label_id: i64::try_from(label_id).map_err(|_| TransformError::ImageSizeOverflow)?,
            was_fused: false,
            score: round_six(score)?,
        });
    }

    Ok(SegmentationPostProcessPrediction {
        size: output_size,
        segmentation,
        segments,
    })
}

pub(super) fn post_process_eomt_panoptic(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    output_size: ImageSize,
    label_ids_to_fuse: &[i64],
    options: SegmentationPostProcessOptions,
) -> Result<SegmentationPostProcessPrediction, TransformError> {
    validate_options(options)?;
    validate_query_mask_inputs(
        class_logits,
        mask_logits,
        num_queries,
        num_labels_with_background,
        output_size,
    )?;
    let background_label = num_labels_with_background - 1;
    let output_pixels = pixels(output_size)?;
    let mut selected = Vec::new();
    for query_index in 0..num_queries {
        let start = checked_mul(query_index, num_labels_with_background)?;
        let probabilities = softmax(&class_logits[start..start + num_labels_with_background])?;
        let (label_id, score) = best_label(&probabilities);
        if label_id != background_label && score > options.score_threshold {
            selected.push(QueryPrediction {
                query_index,
                label_id: i64::try_from(label_id).map_err(|_| TransformError::ImageSizeOverflow)?,
                score,
            });
        }
    }
    if selected.is_empty() {
        return Ok(SegmentationPostProcessPrediction {
            size: output_size,
            segmentation: vec![-1; output_pixels],
            segments: Vec::new(),
        });
    }

    let mut probabilities = Vec::with_capacity(checked_mul(selected.len(), output_pixels)?);
    for query in &selected {
        let start = checked_mul(query.query_index, output_pixels)?;
        for &value in &mask_logits[start..start + output_pixels] {
            probabilities.push(sigmoid(value)?);
        }
    }
    let mut assignments = vec![0usize; output_pixels];
    for pixel_index in 0..output_pixels {
        let mut best_query = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (selected_index, query) in selected.iter().enumerate() {
            let score = query.score * probabilities[selected_index * output_pixels + pixel_index];
            if score > best_score {
                best_query = selected_index;
                best_score = score;
            }
        }
        assignments[pixel_index] = best_query;
    }

    let mut segmentation = vec![-1; output_pixels];
    let mut segments = Vec::new();
    let mut next_id = 0i64;
    let mut fused_ids = HashMap::<i64, i64>::new();
    for (selected_index, query) in selected.iter().copied().enumerate() {
        let plane =
            &probabilities[selected_index * output_pixels..(selected_index + 1) * output_pixels];
        let assigned_area = assignments
            .iter()
            .filter(|&&assigned| assigned == selected_index)
            .count();
        let original_area = plane
            .iter()
            .filter(|&&value| value >= options.mask_threshold)
            .count();
        let final_area = assignments
            .iter()
            .zip(plane)
            .filter(|(assigned, value)| {
                **assigned == selected_index && **value >= options.mask_threshold
            })
            .count();
        if assigned_area == 0
            || original_area == 0
            || final_area == 0
            || assigned_area as f32 / original_area as f32 <= options.overlap_mask_area_threshold
        {
            continue;
        }

        let should_fuse = label_ids_to_fuse.contains(&query.label_id);
        if let Some(&id) = fused_ids.get(&query.label_id) {
            write_eomt_segment(
                &mut segmentation,
                &assignments,
                plane,
                selected_index,
                options.mask_threshold,
                id,
            );
            continue;
        }
        let id = next_id;
        next_id = next_id
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if should_fuse {
            fused_ids.insert(query.label_id, id);
        }
        write_eomt_segment(
            &mut segmentation,
            &assignments,
            plane,
            selected_index,
            options.mask_threshold,
            id,
        );
        segments.push(SegmentationSegmentInfo {
            id,
            label_id: query.label_id,
            was_fused: should_fuse,
            score: round_six(query.score)?,
        });
    }

    Ok(SegmentationPostProcessPrediction {
        size: output_size,
        segmentation,
        segments,
    })
}

fn write_eomt_segment(
    segmentation: &mut [i64],
    assignments: &[usize],
    probabilities: &[f32],
    query_index: usize,
    mask_threshold: f32,
    id: i64,
) {
    for ((pixel, &assigned), &probability) in
        segmentation.iter_mut().zip(assignments).zip(probabilities)
    {
        if assigned == query_index && probability >= mask_threshold {
            *pixel = id;
        }
    }
}

fn binary_mask_score(mask_logits: &[f32]) -> Result<f32, TransformError> {
    // Torch's tensor reduction does not accumulate a large mask serially in
    // `f32`; use a wider accumulator to avoid size-dependent score drift.
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &value in mask_logits {
        validate_score(value)?;
        if value > 0.0 {
            sum += f64::from(sigmoid(value)?);
            count += 1;
        }
    }
    let score = (sum / (count as f64 + 1.0e-6)) as f32;
    validate_score(score)?;
    Ok(score)
}

fn resize_binary_mask_nearest(
    values: &[u8],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<u8>, TransformError> {
    validate_len(values.len(), pixels(source_size)?)?;
    let mut resized = Vec::with_capacity(pixels(target_size)?);
    for target_y in 0..target_size.height {
        let source_y = target_y * source_size.height / target_size.height;
        for target_x in 0..target_size.width {
            let source_x = target_x * source_size.width / target_size.width;
            resized.push(values[source_y * source_size.width + source_x]);
        }
    }
    Ok(resized)
}

fn eomt_aspect_resize_size(
    original_size: ImageSize,
    canvas_size: ImageSize,
) -> Result<ImageSize, TransformError> {
    let source_shortest = original_size.height.min(original_size.width) as f64;
    let source_longest = original_size.height.max(original_size.width) as f64;
    let requested_shortest = canvas_size.height as f64;
    let raw_shortest = (source_longest / source_shortest * requested_shortest
        > canvas_size.width as f64)
        .then(|| canvas_size.width as f64 * source_shortest / source_longest);
    let shortest =
        raw_shortest.map_or(canvas_size.height, |value| value.round_ties_even() as usize);

    let already_at_shortest_edge = (original_size.height <= original_size.width
        && original_size.height == shortest)
        || (original_size.width <= original_size.height && original_size.width == shortest);
    let (height, width) = if already_at_shortest_edge {
        (original_size.height, original_size.width)
    } else if original_size.width < original_size.height {
        let height = raw_shortest.map_or_else(
            || shortest as f64 * original_size.height as f64 / original_size.width as f64,
            |raw| raw * original_size.height as f64 / original_size.width as f64,
        );
        (height as usize, shortest)
    } else {
        let width = raw_shortest.map_or_else(
            || shortest as f64 * original_size.width as f64 / original_size.height as f64,
            |raw| raw * original_size.width as f64 / original_size.height as f64,
        );
        (shortest, width as usize)
    };
    ImageSize::new(height, width)
}

fn validate_query_mask_inputs(
    class_logits: &[f32],
    mask_logits: &[f32],
    num_queries: usize,
    num_labels_with_background: usize,
    mask_size: ImageSize,
) -> Result<(), TransformError> {
    if num_labels_with_background < 2 {
        return Err(TransformError::InvalidClassCount(
            num_labels_with_background,
        ));
    }
    validate_len(
        class_logits.len(),
        checked_mul(num_queries, num_labels_with_background)?,
    )?;
    validate_len(
        mask_logits.len(),
        checked_mul(num_queries, pixels(mask_size)?)?,
    )?;
    for &value in class_logits.iter().chain(mask_logits) {
        validate_score(value)?;
    }
    Ok(())
}

fn validate_options(options: SegmentationPostProcessOptions) -> Result<(), TransformError> {
    validate_score(options.score_threshold)?;
    if !options.mask_threshold.is_finite() {
        return Err(TransformError::InvalidMaskValue(options.mask_threshold));
    }
    if !options.overlap_mask_area_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.overlap_mask_area_threshold)
    {
        return Err(TransformError::InvalidIouThreshold(
            options.overlap_mask_area_threshold,
        ));
    }
    if let Some(size) = options.target_size {
        ImageSize::new(size.height, size.width)?;
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, TransformError> {
    let mut maximum = f32::NEG_INFINITY;
    for &value in logits {
        validate_score(value)?;
        maximum = maximum.max(value);
    }
    let denominator = logits
        .iter()
        .map(|&value| (value - maximum).exp())
        .sum::<f32>();
    validate_score(denominator)?;
    if denominator == 0.0 {
        return Err(TransformError::InvalidScoreValue(denominator));
    }
    logits
        .iter()
        .map(|&value| {
            let probability = (value - maximum).exp() / denominator;
            validate_score(probability)?;
            Ok(probability)
        })
        .collect()
}

fn best_label(probabilities: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (label, &score) in probabilities.iter().enumerate() {
        if score > best.1 {
            best = (label, score);
        }
    }
    best
}

fn sigmoid(value: f32) -> Result<f32, TransformError> {
    validate_score(value)?;
    let score = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    };
    validate_score(score)?;
    Ok(score)
}

fn round_six(value: f32) -> Result<f32, TransformError> {
    validate_score(value)?;
    Ok((value * 1_000_000.0).round() / 1_000_000.0)
}

fn validate_score(value: f32) -> Result<(), TransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TransformError::InvalidScoreValue(value))
    }
}

fn pixels(size: ImageSize) -> Result<usize, TransformError> {
    ImageSize::new(size.height, size.width)?;
    checked_mul(size.height, size.width)
}

fn checked_mul(left: usize, right: usize) -> Result<usize, TransformError> {
    left.checked_mul(right)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn validate_len(actual: usize, expected: usize) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidBufferLength { expected, actual })
    }
}
