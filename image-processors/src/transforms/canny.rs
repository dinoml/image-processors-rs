//! Byte-domain Canny with integer three-tap derivatives and replicated borders.

use super::{ImageSize, SpatialFilterError};

/// Detect edges in interleaved bytes, returning one byte per pixel (0 or 255).
///
/// This implements the aperture-three CPU semantics used by T2I-Adapter:
/// replicated Sobel borders, strongest-channel selection (first on ties),
/// integer directional suppression and eight-connected hysteresis. Thresholds
/// are reordered and floored; `l2_gradient` selects squared Euclidean magnitude
/// instead of the default Manhattan magnitude. No smoothing is added.
/// The numeric contract follows `modules/imgproc/src/canny.cpp` in OpenCV 4.10;
/// this implementation does not link or call that library.
///
/// # Errors
/// Returns [`SpatialFilterError::Geometry`] for inconsistent or overflowing
/// storage and [`SpatialFilterError::NonFinite`] for nonfinite thresholds.
pub fn canny_u8(
    values: &[u8],
    size: ImageSize,
    channels: usize,
    low_threshold: f64,
    high_threshold: f64,
    l2_gradient: bool,
) -> Result<Vec<u8>, SpatialFilterError> {
    let pixels = size.width.checked_mul(size.height);
    let stride = size.width.checked_add(2);
    let padded = stride.and_then(|n| size.height.checked_add(2)?.checked_mul(n));
    if size.width == 0
        || size.height == 0
        || channels == 0
        || pixels.and_then(|n| n.checked_mul(channels)) != Some(values.len())
        || padded.is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(SpatialFilterError::Geometry);
    }
    if !low_threshold.is_finite() || !high_threshold.is_finite() {
        return Err(SpatialFilterError::NonFinite);
    }
    let threshold = |value: f64| {
        let value = if l2_gradient {
            value.min(32767.0)
        } else {
            value
        };
        (if l2_gradient && value > 0.0 {
            value * value
        } else {
            value
        })
        .floor()
    };
    let low = threshold(low_threshold.min(high_threshold));
    let high = threshold(low_threshold.max(high_threshold));
    let stride = stride.ok_or(SpatialFilterError::Geometry)?;
    let padded = padded.ok_or(SpatialFilterError::Geometry)?;
    let pixels = pixels.ok_or(SpatialFilterError::Geometry)?;
    let mut magnitude = vec![0_i32; padded];
    let mut derivatives = vec![(0_i32, 0_i32); pixels];
    for y in 0..size.height {
        let rows = [y.saturating_sub(1), y, (y + 1).min(size.height - 1)];
        for x in 0..size.width {
            let columns = [x.saturating_sub(1), x, (x + 1).min(size.width - 1)];
            let mut best = -1;
            for channel in 0..channels {
                let pixel = |row: usize, column: usize| {
                    i32::from(
                        values[(rows[row] * size.width + columns[column]) * channels + channel],
                    )
                };
                let dx = pixel(0, 2) - pixel(0, 0) + 2 * (pixel(1, 2) - pixel(1, 0)) + pixel(2, 2)
                    - pixel(2, 0);
                let dy = pixel(2, 0) - pixel(0, 0) + 2 * (pixel(2, 1) - pixel(0, 1)) + pixel(2, 2)
                    - pixel(0, 2);
                let strength = if l2_gradient {
                    dx * dx + dy * dy
                } else {
                    dx.abs() + dy.abs()
                };
                if strength > best {
                    best = strength;
                    derivatives[y * size.width + x] = (dx, dy);
                }
            }
            magnitude[(y + 1) * stride + x + 1] = best;
        }
    }
    let mut map = vec![1_u8; padded];
    let mut strong = Vec::new();
    for y in 0..size.height {
        for x in 0..size.width {
            let index = (y + 1) * stride + x + 1;
            let strength = magnitude[index];
            if f64::from(strength) <= low {
                continue;
            }
            let (dx, dy) = derivatives[y * size.width + x];
            let horizontal = dx.abs() * 13573;
            let vertical = dy.abs() << 15;
            let peak = if vertical < horizontal {
                strength > magnitude[index - 1] && strength >= magnitude[index + 1]
            } else if vertical > horizontal + (dx.abs() << 16) {
                strength > magnitude[index - stride] && strength >= magnitude[index + stride]
            } else {
                let direction = if (dx ^ dy) < 0 { -1 } else { 1 };
                let previous = (index as isize - stride as isize - direction) as usize;
                let next = (index as isize + stride as isize + direction) as usize;
                strength > magnitude[previous] && strength > magnitude[next]
            };
            if peak {
                if f64::from(strength) > high {
                    map[index] = 2;
                    strong.push(index);
                } else {
                    map[index] = 0;
                }
            }
        }
    }
    let row = stride as isize;
    while let Some(index) = strong.pop() {
        for offset in [-row - 1, -row, -row + 1, -1, 1, row - 1, row, row + 1] {
            let neighbour = (index as isize + offset) as usize;
            if map[neighbour] == 0 {
                map[neighbour] = 2;
                strong.push(neighbour);
            }
        }
    }
    let mut output = Vec::with_capacity(pixels);
    for y in 0..size.height {
        output.extend((0..size.width).map(|x| {
            if map[(y + 1) * stride + x + 1] == 2 {
                255
            } else {
                0
            }
        }));
    }
    Ok(output)
}
