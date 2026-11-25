//! Differentiable-binarization text detection restoration.
//!
//! Follows Transformers PPOCRV5ServerDetImageProcessor: list retrieval of outer
//! and hole contours, minimum-area boxes, masked mean scores and miter offsets.

use crate::transforms::{ImageSize, TransformError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};
use thiserror::Error;

/// Detection settings for one probability map.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct TextDetectionOptions {
    /// Strict probability threshold used to build the bitmap.
    pub threshold: f32,
    /// Minimum mean probability inside the integer rasterized box.
    pub box_threshold: f64,
    /// Maximum number of contours considered, before score/size filtering.
    pub max_candidates: usize,
    /// Minimum short side before expansion; expansion additionally requires +2.
    pub min_size: f32,
    /// Offset distance multiplier: area * ratio / perimeter.
    pub unclip_ratio: f64,
}

impl Default for TextDetectionOptions {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            box_threshold: 0.6,
            max_candidates: 1000,
            min_size: 3.0,
            unclip_ratio: 1.5,
        }
    }
}

/// Text quadrilateral in original image coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDetection {
    /// Upstream corner order: upper-left, upper-right, lower-right, lower-left.
    pub points: [[i16; 2]; 4],
    /// Masked mean of the probability map, before polygon expansion.
    pub score: f32,
}

/// Invalid text detection geometry, settings or probability storage.
#[derive(Debug, Error)]
pub enum TextDetectionError {
    /// Invalid image dimensions, values or buffer length.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Invalid inference settings.
    #[error("invalid text detection settings: {0}")]
    Invalid(&'static str),
}

/// Restore one row-major probability map to original-image quadrilaterals.
///
/// Contours include holes and are considered in reverse raster-discovery order,
/// matching list retrieval. Geometry and miter arithmetic use floating point
/// vertices; output coordinates use nearest-even rounding and signed 16-bit storage.
///
/// # Errors
/// Rejects invalid dimensions, nonfinite probabilities and invalid options.
pub fn post_process_text_detection(
    probabilities: &[f32],
    map_size: ImageSize,
    original_size: ImageSize,
    options: TextDetectionOptions,
) -> Result<Vec<TextDetection>, TextDetectionError> {
    ImageSize::new(map_size.height, map_size.width)?;
    ImageSize::new(original_size.height, original_size.width)?;
    let count = map_size
        .height
        .checked_mul(map_size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    if probabilities.len() != count {
        return Err(TransformError::InvalidBufferLength {
            expected: count,
            actual: probabilities.len(),
        }
        .into());
    }
    if !options.threshold.is_finite()
        || !options.box_threshold.is_finite()
        || !options.min_size.is_finite()
        || options.min_size < 0.0
        || !options.unclip_ratio.is_finite()
        || options.unclip_ratio < 0.0
    {
        return Err(TextDetectionError::Invalid(
            "nonfinite threshold or negative size/expansion",
        ));
    }
    if let Some(value) = probabilities.iter().find(|value| !value.is_finite()) {
        return Err(TransformError::InvalidScoreValue(*value).into());
    }
    let contours = list_contours(probabilities, map_size, options.threshold, false)?;
    let mut detections = Vec::new();
    for contour in contours.into_iter().rev().take(options.max_candidates) {
        let (points, short_side) = minimum_box(&contour);
        if short_side < options.min_size {
            continue;
        }
        let score = box_score(probabilities, map_size, &points);
        if score < options.box_threshold {
            continue;
        }
        let expanded = unclip(&points, options.unclip_ratio);
        let (points, short_side) = minimum_box(&expanded);
        if short_side < options.min_size + 2.0 {
            continue;
        }
        let scale = [
            original_size.width as f32 / map_size.width as f32,
            original_size.height as f32 / map_size.height as f32,
        ];
        let bounds = [original_size.width as f32, original_size.height as f32];
        let points = points.map(|point| {
            std::array::from_fn(|axis| {
                // The final cast intentionally preserves numpy int16 wrapping.
                ((point[axis] * scale[axis])
                    .round_ties_even()
                    .clamp(0.0, bounds[axis]) as i64) as i16
            })
        });
        detections.push(TextDetection {
            points,
            score: score as f32,
        });
    }
    Ok(detections)
}

// Suzuki border following without hierarchy, with 8-connected foreground.
fn list_contours(
    values: &[f32],
    size: ImageSize,
    threshold: f32,
    external_only: bool,
) -> Result<Vec<Vec<[f32; 2]>>, TextDetectionError> {
    let stride = size
        .width
        .checked_add(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let rows = size
        .height
        .checked_add(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let length = stride
        .checked_mul(rows)
        .ok_or(TransformError::ImageSizeOverflow)?;
    // Border offsets below use signed pointer-sized arithmetic.
    if length > isize::MAX as usize || length > i32::MAX as usize {
        return Err(TransformError::ImageSizeOverflow.into());
    }
    let mut bitmap = vec![0_i32; length];
    for y in 0..size.height {
        for x in 0..size.width {
            bitmap[(y + 1) * stride + x + 1] = i32::from(values[y * size.width + x] > threshold);
        }
    }
    // RETR_EXTERNAL excludes holes and islands enclosed by holes. Flooding the
    // padded background with four-connectivity identifies the external side of
    // each eight-connected foreground border without changing border tracing.
    let mut exterior = Vec::new();
    if external_only {
        exterior.resize(length, false);
        exterior[0] = true;
        let mut pending = VecDeque::from([0_usize]);
        while let Some(index) = pending.pop_front() {
            let row = index / stride;
            let column = index % stride;
            let neighbors = [
                (row > 0).then(|| index - stride),
                (row + 1 < rows).then(|| index + stride),
                (column > 0).then(|| index - 1),
                (column + 1 < stride).then(|| index + 1),
            ];
            for neighbor in neighbors.into_iter().flatten() {
                if bitmap[neighbor] == 0 && !exterior[neighbor] {
                    exterior[neighbor] = true;
                    pending.push_back(neighbor);
                }
            }
        }
    }
    let offsets = [
        1,
        stride as isize + 1,
        stride as isize,
        stride as isize - 1,
        -1,
        -(stride as isize) - 1,
        -(stride as isize),
        -(stride as isize) + 1,
    ];
    let mut contours = Vec::new();
    let mut border = 1_i32;
    for y in 1..=size.height {
        for x in 1..=size.width {
            let start = y * stride + x;
            let initial = if bitmap[start] == 1 && bitmap[start - 1] == 0 {
                4
            } else if bitmap[start] >= 1 && bitmap[start + 1] == 0 {
                0
            } else {
                continue;
            };
            let retain = !external_only || (initial == 4 && exterior[start - 1]);
            border += 1;
            let first = (0..8)
                .map(|step| (initial + step) % 8)
                .map(|direction| start.wrapping_add_signed(offsets[direction]))
                .find(|&index| bitmap[index] != 0);
            let Some(first) = first else {
                bitmap[start] = -border;
                if retain {
                    contours.push(vec![[(x - 1) as f32, (y - 1) as f32]]);
                }
                continue;
            };
            let mut previous = first;
            let mut current = start;
            let mut contour = Vec::new();
            loop {
                contour.push([
                    ((current % stride) - 1) as f32,
                    ((current / stride) - 1) as f32,
                ]);
                let direction = offsets
                    .iter()
                    .position(|&offset| current.wrapping_add_signed(offset) == previous)
                    .ok_or(TextDetectionError::Invalid("invalid contour neighborhood"))?;
                let mut right_zero = false;
                let mut next = current;
                for step in 1..=8 {
                    let candidate_direction = (direction + 8 - step) % 8;
                    let candidate = current.wrapping_add_signed(offsets[candidate_direction]);
                    if bitmap[candidate] != 0 {
                        next = candidate;
                        break;
                    }
                    right_zero |= candidate_direction == 0;
                }
                if right_zero {
                    bitmap[current] = -border;
                } else if bitmap[current] == 1 {
                    bitmap[current] = border;
                }
                if next == start && current == first {
                    break;
                }
                previous = current;
                current = next;
            }
            // CHAIN_APPROX_SIMPLE retains endpoints of each straight digital run.
            let simplified = (0..contour.len())
                .filter_map(|index| {
                    let previous = contour[(index + contour.len() - 1) % contour.len()];
                    let current = contour[index];
                    let next = contour[(index + 1) % contour.len()];
                    let incoming = std::array::from_fn::<_, 2, _>(|axis| {
                        current[axis].total_cmp(&previous[axis])
                    });
                    let outgoing =
                        std::array::from_fn::<_, 2, _>(|axis| next[axis].total_cmp(&current[axis]));
                    (incoming != outgoing || contour.len() <= 2).then_some(current)
                })
                .collect();
            if retain {
                contours.push(simplified);
            }
        }
    }
    Ok(contours)
}

fn cross(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f64 {
    f64::from(a[0] - origin[0]) * f64::from(b[1] - origin[1])
        - f64::from(a[1] - origin[1]) * f64::from(b[0] - origin[0])
}

fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut points = points.to_vec();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() <= 2 {
        return points;
    }
    let mut hull = Vec::new();
    for &point in &points {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0 {
            hull.pop();
        }
        hull.push(point);
    }
    let lower = hull.len();
    for &point in points.iter().rev().skip(1) {
        while hull.len() > lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0
        {
            hull.pop();
        }
        hull.push(point);
    }
    hull.pop();
    hull
}

struct RotatedRectangle {
    center: [f32; 2],
    size: [f32; 2],
    angle: f32,
}

enum RectangleConvention {
    Existing,
    OpenCv,
}

// The caller supplies a nondegenerate convex hull. Both restoration policies
// share the same rotating calipers and FP32 center/size/angle boundary.
fn minimum_rectangle(hull: &[[f32; 2]], convention: RectangleConvention) -> RotatedRectangle {
    let edges = (0..hull.len())
        .map(|i| {
            [
                hull[(i + 1) % hull.len()][0] - hull[i][0],
                hull[(i + 1) % hull.len()][1] - hull[i][1],
            ]
        })
        .collect::<Vec<_>>();
    // Keep the reference's double square-root / float reciprocal boundary.
    let inverse_lengths = edges
        .iter()
        .map(|edge| {
            (1.0 / (f64::from(edge[0]) * f64::from(edge[0])
                + f64::from(edge[1]) * f64::from(edge[1]))
            .sqrt()) as f32
        })
        .collect::<Vec<_>>();
    let mut contacts = [0; 4]; // bottom, right, top, left in Cartesian coordinates
    for i in 1..hull.len() {
        if hull[i][1] < hull[contacts[0]][1] {
            contacts[0] = i;
        }
        if hull[i][0] > hull[contacts[1]][0] {
            contacts[1] = i;
        }
        if hull[i][1] > hull[contacts[2]][1] {
            contacts[2] = i;
        }
        if hull[i][0] < hull[contacts[3]][0] {
            contacts[3] = i;
        }
    }
    let mut best_area = f32::MAX;
    let mut best_state = ([1.0_f32, 0.0], 0.0, 0.0, 0, 0);
    for _ in 0..hull.len() {
        let candidates = [
            edges[contacts[0]],
            [edges[contacts[1]][1], -edges[contacts[1]][0]],
            [-edges[contacts[2]][0], -edges[contacts[2]][1]],
            [-edges[contacts[3]][1], edges[contacts[3]][0]],
        ];
        let mut leading = 0;
        for i in 1..4 {
            if candidates[i][1] * candidates[leading][0] - candidates[i][0] * candidates[leading][1]
                < 0.0
            {
                leading = i;
            }
        }
        let direction = candidates[leading].map(|value| value * inverse_lengths[contacts[leading]]);
        contacts[leading] = (contacts[leading] + 1) % hull.len();
        let horizontal = [
            hull[contacts[1]][0] - hull[contacts[3]][0],
            hull[contacts[1]][1] - hull[contacts[3]][1],
        ];
        let vertical = [
            hull[contacts[2]][0] - hull[contacts[0]][0],
            hull[contacts[2]][1] - hull[contacts[0]][1],
        ];
        let width = horizontal[0] * direction[0] + horizontal[1] * direction[1];
        let height = -vertical[0] * direction[1] + vertical[1] * direction[0];
        if width * height <= best_area {
            best_area = width * height;
            best_state = (direction, width, height, contacts[3], contacts[0]);
        }
    }
    let (u, width, height, left, bottom) = best_state;
    let v = [-u[1], u[0]];
    let left_projection = u[0] * hull[left][0] + hull[left][1] * u[1];
    let bottom_projection = v[0] * hull[bottom][0] + hull[bottom][1] * v[1];
    let inverse_determinant = 1.0 / (u[0] * v[1] - v[0] * u[1]);
    let corner = [
        (left_projection * v[1] - bottom_projection * u[1]) * inverse_determinant,
        (u[0] * bottom_projection - v[0] * left_projection) * inverse_determinant,
    ];
    let horizontal = u.map(|value| value * width);
    let vertical = v.map(|value| value * height);
    let center: [f32; 2] =
        std::array::from_fn(|axis| corner[axis] + (horizontal[axis] + vertical[axis]) * 0.5);
    let norm = |vector: [f32; 2]| {
        (f64::from(vector[0]) * f64::from(vector[0]) + f64::from(vector[1]) * f64::from(vector[1]))
            .sqrt() as f32
    };
    if matches!(convention, RectangleConvention::OpenCv) {
        // OpenCV minAreaRect rounds atan2 to FP32 radians before converting
        // to degrees. Its width follows out[1], with no vertex reordering.
        let radians = f64::from(f64::from(horizontal[1]).atan2(f64::from(horizontal[0])) as f32);
        return RotatedRectangle {
            center,
            size: [norm(horizontal), norm(vertical)],
            angle: (radians * 180.0 / std::f64::consts::PI) as f32,
        };
    }
    // minAreaRect's center/size/angle boundary is significant: truncating its
    // reconstructed corners later defines the score mask. Direct projection is
    // mathematically equivalent but loses those FP32 rounding boundaries.
    let mut box_width = norm(vertical);
    let mut box_height = norm(horizontal);
    let angle = if horizontal[0] == 0.0 && horizontal[1] > 0.0 {
        std::mem::swap(&mut box_width, &mut box_height);
        -90.0_f32
    } else {
        (-f64::from(horizontal[0]).atan2(f64::from(horizontal[1])) * 180.0 / std::f64::consts::PI)
            as f32
    };
    RotatedRectangle {
        center,
        size: [box_width, box_height],
        angle,
    }
}

fn minimum_box(points: &[[f32; 2]]) -> ([[f32; 2]; 4], f32) {
    let hull = convex_hull(points);
    if hull.len() <= 1 {
        return ([hull.first().copied().unwrap_or([0.0; 2]); 4], 0.0);
    }
    if hull.len() == 2 {
        return ([hull[0], hull[1], hull[1], hull[0]], 0.0);
    }
    let RotatedRectangle {
        center,
        size: [box_width, box_height],
        angle,
    } = minimum_rectangle(&hull, RectangleConvention::Existing);
    let radians = f64::from(angle) * std::f64::consts::PI / 180.0;
    let sine = radians.sin() as f32 * 0.5;
    let cosine = radians.cos() as f32 * 0.5;
    let ah = sine * box_height;
    let aw = sine * box_width;
    let bh = cosine * box_height;
    let bw = cosine * box_width;
    let mut best = [
        [center[0] - ah - bw, center[1] + bh - aw],
        [center[0] + ah - bw, center[1] - bh - aw],
        [center[0] + ah + bw, center[1] - bh + aw],
        [center[0] - ah + bw, center[1] + bh + aw],
    ];
    let short_side = box_width.min(box_height);
    best.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let (left_top, left_bottom) = if best[1][1] > best[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (right_top, right_bottom) = if best[3][1] > best[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };
    (
        [
            best[left_top],
            best[right_top],
            best[right_bottom],
            best[left_bottom],
        ],
        short_side,
    )
}

/// FAST restoration shape selected by the original `bbox_type` setting.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FastTextBoxType {
    /// Area-adjusted minimum-area rectangle, in original corner order.
    #[default]
    Rect,
    /// First external contour in reverse raster-discovery order.
    Poly,
}

/// Original FAST component filtering and output geometry settings.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct FastTextDetectionOptions {
    /// Minimum mean score over all pixels assigned to a component.
    pub min_score: f64,
    /// Minimum expanded component area, in resized-image pixels.
    pub min_area: usize,
    /// Rectangle or polygon output.
    pub bbox_type: FastTextBoxType,
}

impl Default for FastTextDetectionOptions {
    fn default() -> Self {
        Self {
            min_score: 0.88,
            min_area: 250,
            bbox_type: FastTextBoxType::Rect,
        }
    }
}

/// One FAST region, with coordinates truncated to original-image integer pixels.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FastTextDetection {
    /// Rectangle corners or external contour vertices in upstream order.
    pub points: Vec<[i32; 2]>,
    /// Mean score over the complete expanded component, before restoration.
    pub score: f64,
}

fn sum_fp32(values: &[f32]) -> f32 {
    match values {
        [] => 0.0,
        [value] => *value,
        _ => {
            let (left, right) = values.split_at(values.len() / 2);
            sum_fp32(left) + sum_fp32(right)
        }
    }
}

/// Restore one original FAST kernel-label plane and expanded label/score planes.
///
/// `kernel_labels` is the pre-expansion quarter-resolution output; `labels` and
/// `scores` have `image_size` pixels. Candidate keys come from the kernel plane,
/// including the original first-key skip. Polygon output selects the first
/// external contour rather than the largest contour. No DBNet unclip, clipping,
/// vertex reordering, or nearest-integer rounding is applied.
///
/// # Errors
/// Rejects invalid dimensions/storage, invalid filtering settings, nonfinite
/// scores and degenerate rectangle geometry that has zero denominator area.
pub fn post_process_fast_text_detection(
    kernel_labels: &[i64],
    labels: &[i64],
    scores: &[f32],
    image_size: ImageSize,
    original_size: ImageSize,
    options: FastTextDetectionOptions,
) -> Result<Vec<FastTextDetection>, TextDetectionError> {
    ImageSize::new(image_size.height, image_size.width)?;
    ImageSize::new(original_size.height, original_size.width)?;
    let pixels = image_size
        .height
        .checked_mul(image_size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    if !image_size.height.is_multiple_of(4)
        || !image_size.width.is_multiple_of(4)
        || labels.len() != pixels
        || scores.len() != pixels
        || kernel_labels.len() != pixels / 16
    {
        return Err(TextDetectionError::Invalid(
            "FAST label/score plane geometry differs",
        ));
    }
    if !options.min_score.is_finite() {
        return Err(TextDetectionError::Invalid(
            "FAST requires finite min_score",
        ));
    }
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(TextDetectionError::Invalid("nonfinite FAST score"));
    }
    let keys = kernel_labels.iter().copied().collect::<BTreeSet<_>>();
    let scales = [
        original_size.width as f64 / image_size.width as f64,
        original_size.height as f64 / image_size.height as f64,
    ];
    let mut detections = Vec::new();
    for key in keys.into_iter().skip(1) {
        let mut points = Vec::new();
        for (index, &label) in labels.iter().enumerate() {
            if label == key {
                points.push([
                    (index % image_size.width) as f32,
                    (index / image_size.width) as f32,
                ]);
            }
        }
        if points.len() < options.min_area {
            continue;
        }
        if points.is_empty() {
            return Err(TextDetectionError::Invalid("empty FAST region"));
        }
        let selected_scores = labels
            .iter()
            .zip(scores)
            .filter_map(|(&label, &score)| (label == key).then_some(score))
            .collect::<Vec<_>>();
        // The source mean has FP32 accumulation/output, followed by .item()
        // promotion for comparison with the Python binary64 threshold. A tree
        // reduction avoids linear FP32 accumulation drift over large regions.
        let score = f64::from(sum_fp32(&selected_scores) / points.len() as f32);
        if score < options.min_score {
            continue;
        }
        let vertices = match options.bbox_type {
            FastTextBoxType::Rect => {
                let mut hull = convex_hull(&points);
                if hull.len() < 3 {
                    return Err(TextDetectionError::Invalid("degenerate FAST rectangle"));
                }
                // The input points are in raster order. Match OpenCV's hull
                // starting at the last raster point so equal-area caliper
                // candidates preserve its angle and corner ordering.
                let start = hull
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0])))
                    .map_or(0, |(index, _)| index);
                hull.rotate_left(start);
                let RotatedRectangle {
                    center,
                    size,
                    angle,
                } = minimum_rectangle(&hull, RectangleConvention::OpenCv);
                let area = f64::from(size[0]) * f64::from(size[1]);
                if area <= 0.0 {
                    return Err(TextDetectionError::Invalid("degenerate FAST rectangle"));
                }
                let alpha = ((points.len() as f64 / area).sqrt()).sqrt();
                let width = (f64::from(size[0]) * alpha) as f32;
                let height = (f64::from(size[1]) * alpha) as f32;
                let radians = f64::from(angle) * std::f64::consts::PI / 180.0;
                let a = radians.sin() as f32 * 0.5;
                let b = radians.cos() as f32 * 0.5;
                let p0 = [
                    center[0] - a * height - b * width,
                    center[1] + b * height - a * width,
                ];
                let p1 = [
                    center[0] + a * height - b * width,
                    center[1] - b * height - a * width,
                ];
                vec![
                    p0,
                    p1,
                    [2.0 * center[0] - p0[0], 2.0 * center[1] - p0[1]],
                    [2.0 * center[0] - p1[0], 2.0 * center[1] - p1[1]],
                ]
            }
            FastTextBoxType::Poly => {
                let bitmap = labels
                    .iter()
                    .map(|&value| if value == key { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>();
                let mut contours = list_contours(&bitmap, image_size, 0.5, true)?;
                let Some(contour) = contours.pop() else {
                    continue;
                };
                contour
            }
        };
        let points = vertices
            .into_iter()
            .map(|point| std::array::from_fn(|axis| (f64::from(point[axis]) * scales[axis]) as i32))
            .collect();
        detections.push(FastTextDetection { points, score });
    }
    Ok(detections)
}

fn unclip(points: &[[f32; 2]; 4], ratio: f64) -> [[f32; 2]; 4] {
    let edges: [[f32; 2]; 4] = std::array::from_fn(|i| {
        [
            points[(i + 1) % 4][0] - points[i][0],
            points[(i + 1) % 4][1] - points[i][1],
        ]
    });
    let lengths = edges.map(|edge| (edge[0] * edge[0] + edge[1] * edge[1]).sqrt());
    let perimeter: f64 = lengths.iter().map(|value| f64::from(*value)).sum();
    let signed_area: f64 = (0..4)
        .map(|i| {
            f64::from(points[i][0]) * f64::from(points[(i + 1) % 4][1])
                - f64::from(points[i][1]) * f64::from(points[(i + 1) % 4][0])
        })
        .sum();
    let distance = (signed_area.abs() * 0.5 * ratio / perimeter.max(1e-6)) as f32;
    let directions: [[f32; 2]; 4] =
        std::array::from_fn(|i| edges[i].map(|value| value / lengths[i].max(1e-6)));
    let normals = directions.map(|edge| {
        if signed_area > 0.0 {
            [edge[1], -edge[0]]
        } else {
            [-edge[1], edge[0]]
        }
    });
    let shifted: [[f32; 2]; 4] = std::array::from_fn(|i| {
        [
            points[i][0] + distance * normals[i][0],
            points[i][1] + distance * normals[i][1],
        ]
    });
    std::array::from_fn(|i| {
        let previous = (i + 3) % 4;
        let determinant =
            directions[previous][0] * directions[i][1] - directions[previous][1] * directions[i][0];
        if determinant.abs() < 1e-6 {
            return std::array::from_fn(|axis| {
                points[i][axis] + 0.5 * distance * (normals[previous][axis] + normals[i][axis])
            });
        }
        let delta = [
            shifted[i][0] - shifted[previous][0],
            shifted[i][1] - shifted[previous][1],
        ];
        let parameter = (delta[0] * directions[i][1] - delta[1] * directions[i][0]) / determinant;
        std::array::from_fn(|axis| shifted[previous][axis] + directions[previous][axis] * parameter)
    })
}

fn clip_line(
    mut a: [i64; 2],
    mut b: [i64; 2],
    width: usize,
    height: usize,
) -> ([i64; 2], [i64; 2], bool) {
    let right = width as i64 - 1;
    let bottom = height as i64 - 1;
    let code = |p: [i64; 2]| {
        i32::from(p[0] < 0)
            | (i32::from(p[0] > right) << 1)
            | (i32::from(p[1] < 0) << 2)
            | (i32::from(p[1] > bottom) << 3)
    };
    let mut first = code(a);
    let mut second = code(b);
    if first & second != 0 {
        return (a, b, false);
    }
    if first | second != 0 {
        if first & 12 != 0 {
            let y = if first < 8 { 0 } else { bottom };
            a[0] += ((y - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = y;
            first = code(a);
        }
        if second & 12 != 0 {
            let y = if second < 8 { 0 } else { bottom };
            b[0] += ((y - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = y;
            second = code(b);
        }
        if first & second == 0 {
            if first != 0 {
                let x = if first == 1 { 0 } else { right };
                a[1] += ((x - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = x;
                first = 0;
            }
            if second != 0 {
                let x = if second == 1 { 0 } else { right };
                b[1] += ((x - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = x;
                second = 0;
            }
        }
    }
    (a, b, first | second == 0)
}

fn draw_line(mask: &mut [bool], width: usize, height: usize, mut a: [i64; 2], mut b: [i64; 2]) {
    if a[0] > b[0] {
        std::mem::swap(&mut a, &mut b);
    }
    let dx = b[0] - a[0];
    let dy = (b[1] - a[1]).abs();
    let vertical = dy > dx;
    let major = dx.max(dy);
    let minor = dx.min(dy);
    let y_direction = if b[1] >= a[1] { 1 } else { -1 };
    for step in 0..=major {
        // Nearest integer with midpoint ties toward the starting endpoint,
        // after orienting left-to-right, reproduces LINE_8's error<0 policy.
        let displacement = if major == 0 {
            0
        } else {
            (step * minor + (major - 1) / 2) / major
        };
        let x = a[0] + if vertical { displacement } else { step };
        let y = a[1] + y_direction * if vertical { step } else { displacement };
        if x >= 0 && x < width as i64 && y >= 0 && y < height as i64 {
            mask[y as usize * width + x as usize] = true;
        }
    }
}

fn box_score(values: &[f32], size: ImageSize, points: &[[f32; 2]; 4]) -> f64 {
    let xmin = points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (size.width - 1) as f32) as usize;
    let xmax = points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (size.width - 1) as f32) as usize;
    let ymin = points
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (size.height - 1) as f32) as usize;
    let ymax = points
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (size.height - 1) as f32) as usize;
    let width = xmax - xmin + 1;
    let height = ymax - ymin + 1;
    let polygon = points.map(|p| [(p[0] - xmin as f32) as i64, (p[1] - ymin as f32) as i64]);
    let mut mask = vec![false; width * height];
    let mut edges = Vec::new();
    for i in 0..4 {
        let a = polygon[i];
        let b = polygon[(i + 1) % 4];
        let (mut clipped_a, mut clipped_b, visible) = clip_line(a, b, width, height);
        if visible {
            draw_line(&mut mask, width, height, clipped_a, clipped_b);
        }
        if a[1] == b[1] {
            continue;
        }
        // The score mask is cropped, so clipping must also affect scanline
        // slopes. Original y extents are retained by integer polygon filling.
        if clipped_a[1] == clipped_b[1] {
            clipped_a[1] = a[1];
            clipped_b[1] = b[1];
        }
        let slope = ((clipped_b[0] - clipped_a[0]) << 16) / (clipped_b[1] - clipped_a[1]);
        let (start, end, clipped) = if a[1] < b[1] {
            (a, b, clipped_a)
        } else {
            (b, a, clipped_b)
        };
        let x = (clipped[0] << 16) + (start[1] - clipped[1]) * slope;
        edges.push((start[1], end[1], x, slope));
    }
    for y in 0..height {
        let mut intersections = Vec::new();
        for &(y0, y1, x, slope) in &edges {
            if y0 <= y as i64 && (y as i64) < y1 {
                intersections.push(x + (y as i64 - y0) * slope);
            }
        }
        intersections.sort_unstable();
        for pair in intersections.as_chunks::<2>().0 {
            let left = ((pair[0] + 65535) >> 16).max(0);
            let right = (pair[1] >> 16).min(width as i64 - 1);
            for x in left..=right {
                mask[y * width + x as usize] = true;
            }
        }
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] {
                sum += f64::from(values[(ymin + y) * size.width + xmin + x]);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
