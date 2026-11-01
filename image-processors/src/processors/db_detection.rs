type Vertex = [f32; 2];
type Quad = [Vertex; 4];
type Result<T> = std::result::Result<T, crate::transforms::TransformError>;

fn invalid(message: impl Into<String>) -> crate::transforms::TransformError {
    crate::transforms::TransformError::ImageFrameInvariant(message.into())
}

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub struct DetectionOptions {
    pub threshold: f32,
    pub box_threshold: f32,
    pub max_candidates: usize,
    pub min_size: f32,
    pub unclip_ratio: f32,
}

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct DetectedQuad {
    pub quadrilateral: [[i16; 2]; 4],
    pub score: f32,
}

fn cross(left: Vertex, right: Vertex) -> f32 {
    left[0] * right[1] - left[1] * right[0]
}

fn subtract(left: Vertex, right: Vertex) -> Vertex {
    [left[0] - right[0], left[1] - right[1]]
}

fn length(vector: Vertex) -> f32 {
    vector[0].hypot(vector[1])
}

// Build a floating-point convex hull so rectangle and expanded coordinates are
// not rounded before the upstream's final original-image coordinate conversion.
fn convex_hull(points: &[Vertex]) -> Vec<Vertex> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let mut lower: Vec<Vertex> = Vec::new();
    for &point in &sorted {
        while lower.len() >= 2
            && cross(
                subtract(lower[lower.len() - 1], lower[lower.len() - 2]),
                subtract(point, lower[lower.len() - 1]),
            ) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper: Vec<Vertex> = Vec::new();
    for &point in sorted.iter().rev() {
        while upper.len() >= 2
            && cross(
                subtract(upper[upper.len() - 1], upper[upper.len() - 2]),
                subtract(point, upper[upper.len() - 1]),
            ) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn mini_box(points: &[Vertex]) -> Option<(Quad, f32)> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let mut best = None;
    let mut best_area = f32::INFINITY;
    for index in 0..hull.len() {
        let edge = subtract(hull[(index + 1) % hull.len()], hull[index]);
        let norm = length(edge);
        if norm == 0.0 {
            continue;
        }
        let horizontal = [edge[0] / norm, edge[1] / norm];
        let vertical = [-horizontal[1], horizontal[0]];
        let mut bounds = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for point in &hull {
            let x = point[0] * horizontal[0] + point[1] * horizontal[1];
            let y = point[0] * vertical[0] + point[1] * vertical[1];
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].max(x);
            bounds[2] = bounds[2].min(y);
            bounds[3] = bounds[3].max(y);
        }
        let width = bounds[1] - bounds[0];
        let height = bounds[3] - bounds[2];
        if width * height >= best_area {
            continue;
        }
        best_area = width * height;
        let mut corners = [
            [bounds[0], bounds[2]],
            [bounds[1], bounds[2]],
            [bounds[1], bounds[3]],
            [bounds[0], bounds[3]],
        ]
        .map(|point| {
            [
                point[0] * horizontal[0] + point[1] * vertical[0],
                point[0] * horizontal[1] + point[1] * vertical[1],
            ]
        });
        // Match _get_mini_boxes' stable x-sort and within-pair y comparisons.
        corners.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let (first, fourth) = if corners[1][1] > corners[0][1] {
            (0, 1)
        } else {
            (1, 0)
        };
        let (second, third) = if corners[3][1] > corners[2][1] {
            (2, 3)
        } else {
            (3, 2)
        };
        best = Some((
            [
                corners[first],
                corners[second],
                corners[third],
                corners[fourth],
            ],
            width.min(height),
        ));
    }
    best
}

fn box_score(prediction: &[f32], width: usize, height: usize, corners: Quad) -> Result<f32> {
    let xmin = corners
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (width - 1) as f32) as usize;
    let xmax = corners
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (width - 1) as f32) as usize;
    let ymin = corners
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (height - 1) as f32) as usize;
    let ymax = corners
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (height - 1) as f32) as usize;
    let size = crate::ImageSize::new(ymax - ymin + 1, xmax - xmin + 1)?;
    let mask_length = size
        .height
        .checked_mul(size.width)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| invalid("rectangle mask size overflow"))?;
    let mut mask = vec![0; mask_length];
    let polygon = corners.map(|point| {
        [
            (point[0] - xmin as f32) as i32,
            (point[1] - ymin as f32) as i32,
        ]
    });
    crate::transforms::fill_convex_polygon_rgb(&mut mask, size, &polygon, [1; 3])
        .map_err(|error| invalid(error.to_string()))?;
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for (index, pixel) in mask.as_chunks::<3>().0.iter().enumerate() {
        if pixel[0] != 0 {
            sum += f64::from(
                prediction[(ymin + index / size.width) * width + xmin + index % size.width],
            );
            count += 1;
        }
    }
    Ok(if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    })
}

fn unclip(corners: Quad, ratio: f32) -> Quad {
    let area = (0..4)
        .map(|index| cross(corners[index], corners[(index + 1) % 4]))
        .sum::<f32>()
        * 0.5;
    let edges =
        std::array::from_fn::<_, 4, _>(|index| subtract(corners[(index + 1) % 4], corners[index]));
    let perimeter = edges.iter().map(|&edge| length(edge)).sum::<f32>();
    let offset = area.abs() * ratio / perimeter;
    let directions = edges.map(|edge| {
        let norm = length(edge).max(1e-6);
        [edge[0] / norm, edge[1] / norm]
    });
    let normals = directions.map(|edge| {
        if area > 0.0 {
            [edge[1], -edge[0]]
        } else {
            [-edge[1], edge[0]]
        }
    });
    let shifted = std::array::from_fn::<_, 4, _>(|index| {
        [
            corners[index][0] + offset * normals[index][0],
            corners[index][1] + offset * normals[index][1],
        ]
    });
    std::array::from_fn(|index| {
        let previous = (index + 3) % 4;
        let product = cross(directions[previous], directions[index]);
        if product.abs() < 1e-6 {
            return [
                corners[index][0] + 0.5 * offset * (normals[previous][0] + normals[index][0]),
                corners[index][1] + 0.5 * offset * (normals[previous][1] + normals[index][1]),
            ];
        }
        let parameter = cross(
            subtract(shifted[index], shifted[previous]),
            directions[index],
        ) / product;
        [
            shifted[previous][0] + directions[previous][0] * parameter,
            shifted[previous][1] + directions[previous][1] * parameter,
        ]
    })
}

fn bitmap_contours(
    prediction: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> Vec<Vec<Vertex>> {
    let stride = width + 2;
    let mut labels = vec![0_i32; stride * (height + 2)];
    for y in 0..height {
        for x in 0..width {
            labels[(y + 1) * stride + x + 1] = i32::from(prediction[y * width + x] > threshold);
        }
    }
    let directions = [
        (-1_isize, 0_isize),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];
    let neighbor = |point: (usize, usize), direction: usize| {
        let (dx, dy) = directions[direction];
        (
            point.0.wrapping_add_signed(dx),
            point.1.wrapping_add_signed(dy),
        )
    };
    let direction_to = |point: (usize, usize), adjacent: (usize, usize)| {
        let delta = (
            adjacent.0 as isize - point.0 as isize,
            adjacent.1 as isize - point.1 as isize,
        );
        directions
            .iter()
            .position(|&direction| direction == delta)
            .unwrap_or(0)
    };
    let mut border = 1_i32;
    let mut contours = Vec::new();
    for y in 1..=height {
        for x in 1..=width {
            let offset = y * stride + x;
            let previous = if labels[offset] == 1 && labels[offset - 1] == 0 {
                Some((x - 1, y))
            } else if labels[offset] > 0 && labels[offset + 1] == 0 {
                Some((x + 1, y))
            } else {
                None
            };
            let Some(previous) = previous else { continue };
            border += 1;
            let origin = (x, y);
            let initial = direction_to(origin, previous);
            let first = (0..8).find_map(|step| {
                let point = neighbor(origin, (initial + step) % 8);
                (labels[point.1 * stride + point.0] != 0).then_some(point)
            });
            let Some(first) = first else {
                labels[offset] = -border;
                contours.push(vec![[x as f32 - 1.0, y as f32 - 1.0]]);
                continue;
            };
            let mut previous = first;
            let mut current = origin;
            let mut points = Vec::new();
            loop {
                points.push([current.0 as f32 - 1.0, current.1 as f32 - 1.0]);
                let initial = direction_to(current, previous);
                let mut crossed_right = false;
                let mut next = previous;
                for step in 1..=8 {
                    let direction = (initial + 8 - step) % 8;
                    let candidate = neighbor(current, direction);
                    if labels[candidate.1 * stride + candidate.0] != 0 {
                        next = candidate;
                        break;
                    }
                    crossed_right |= direction == 4;
                }
                let index = current.1 * stride + current.0;
                if crossed_right {
                    labels[index] = -border;
                } else if labels[index] == 1 {
                    labels[index] = border;
                }
                if next == origin && current == first {
                    break;
                }
                previous = current;
                current = next;
            }
            contours.push(points);
        }
    }
    contours
}

#[allow(missing_docs)]
pub fn boxes_from_bitmap(
    prediction: &[f32],
    width: usize,
    height: usize,
    original_size: [usize; 2],
    options: DetectionOptions,
) -> Result<Vec<DetectedQuad>> {
    if width == 0
        || height == 0
        || width.checked_mul(height) != Some(prediction.len())
        || original_size.contains(&0)
        || width > i32::MAX as usize - 2
        || height > i32::MAX as usize - 2
        || (width + 2)
            .checked_mul(height + 2)
            .is_none_or(|size| size > i32::MAX as usize)
    {
        return Err(invalid(
            "invalid probability map or original image dimensions",
        ));
    }
    let contours = bitmap_contours(prediction, width, height, options.threshold);
    let mut detections = Vec::new();
    // RETR_LIST exposes contours in reverse discovery order. Include holes as
    // well as external boundaries, and cap candidates before score filtering.
    for contour in contours.iter().rev().take(options.max_candidates) {
        let Some((corners, short_side)) = mini_box(contour) else {
            continue;
        };
        if short_side < options.min_size {
            continue;
        }
        let score = box_score(prediction, width, height, corners)?;
        if score < options.box_threshold {
            continue;
        }
        let Some((corners, short_side)) = mini_box(&unclip(corners, options.unclip_ratio)) else {
            continue;
        };
        if short_side < options.min_size + 2.0 {
            continue;
        }
        let quadrilateral = corners.map(|point| {
            [
                (point[0] * original_size[1] as f32 / width as f32)
                    .round_ties_even()
                    .clamp(0.0, original_size[1] as f32) as i32 as i16,
                (point[1] * original_size[0] as f32 / height as f32)
                    .round_ties_even()
                    .clamp(0.0, original_size[0] as f32) as i32 as i16,
            ]
        });
        detections.push(DetectedQuad {
            quadrilateral,
            score,
        });
    }
    Ok(detections)
}
