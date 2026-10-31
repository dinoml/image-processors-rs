//! Integer RGB raster drawing with scalar compatibility rounding.
use super::{FaceGeometryError, ImageSize};
mod degree_sine;
use degree_sine::DEGREE_SINE;

fn validate(values: &[u8], size: ImageSize) -> Result<(), FaceGeometryError> {
    if size.width == 0
        || size.height == 0
        || size.width > i32::MAX as usize
        || size.height > i32::MAX as usize
        || size
            .width
            .checked_mul(size.height)
            .and_then(|v| v.checked_mul(3))
            != Some(values.len())
    {
        return Err(FaceGeometryError::Geometry);
    }
    Ok(())
}
fn hline(values: &mut [u8], size: ImageSize, y: i64, left: i64, right: i64, color: [u8; 3]) {
    if y < 0 || y >= size.height as i64 {
        return;
    }
    let left = left.max(0);
    let right = right.min(size.width as i64 - 1);
    if left > right {
        return;
    }
    let row = y as usize * size.width;
    for x in left as usize..=right as usize {
        values[(row + x) * 3..(row + x) * 3 + 3].copy_from_slice(&color);
    }
}
fn clip(mut a: [i64; 2], mut b: [i64; 2], size: ImageSize) -> Option<([i64; 2], [i64; 2])> {
    let right = size.width as i64 - 1;
    let bottom = size.height as i64 - 1;
    let code = |p: [i64; 2]| {
        i32::from(p[0] < 0)
            + i32::from(p[0] > right) * 2
            + i32::from(p[1] < 0) * 4
            + i32::from(p[1] > bottom) * 8
    };
    let mut ca = code(a);
    let mut cb = code(b);
    if ca & cb == 0 && ca | cb != 0 {
        if ca & 12 != 0 {
            let y = if ca < 8 { 0 } else { bottom };
            a[0] += ((y - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = y;
            ca = code(a);
        }
        if cb & 12 != 0 {
            let y = if cb < 8 { 0 } else { bottom };
            b[0] += ((y - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = y;
            cb = code(b);
        }
        if ca & cb == 0 && ca | cb != 0 {
            if ca != 0 {
                let x = if ca == 1 { 0 } else { right };
                a[1] += ((x - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = x;
                ca = 0;
            }
            if cb != 0 {
                let x = if cb == 1 { 0 } else { right };
                b[1] += ((x - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = x;
                cb = 0;
            }
        }
    }
    if ca | cb == 0 {
        Some((a, b))
    } else {
        None
    }
}
fn line(values: &mut [u8], size: ImageSize, a: [i32; 2], b: [i32; 2], color: [u8; 3]) {
    let Some((mut a, mut b)) = clip(a.map(i64::from), b.map(i64::from), size) else {
        return;
    };
    if a[0] > b[0] {
        std::mem::swap(&mut a, &mut b);
    }
    let dx = b[0] - a[0];
    let dy = (b[1] - a[1]).abs();
    let sign = if b[1] >= a[1] { 1 } else { -1 };
    let steep = dy > dx;
    let (major, minor) = if steep { (dy, dx) } else { (dx, dy) };
    let mut error = major - 2 * minor;
    for _ in 0..=major {
        let offset = (a[1] as usize * size.width + a[0] as usize) * 3;
        values[offset..offset + 3].copy_from_slice(&color);
        let step = error < 0;
        error -= 2 * minor;
        if step {
            error += 2 * major;
        }
        if steep {
            a[1] += sign;
            if step {
                a[0] += 1;
            }
        } else {
            a[0] += 1;
            if step {
                a[1] += sign;
            }
        }
    }
}

/// Integer points of a rotated full ellipse, with one-degree sampling.
/// Repeated consecutive rounded points are removed.
/// # Errors
/// Rejects negative axes or coordinates exceeding signed integer bounds.
pub fn ellipse_polygon(
    center: [i32; 2],
    axes: [i32; 2],
    angle: i32,
) -> Result<Vec<[i32; 2]>, FaceGeometryError> {
    if axes.iter().any(|&v| v < 0) {
        return Err(FaceGeometryError::Geometry);
    }
    let angle = angle.rem_euclid(360) as usize;
    let alpha = f64::from(DEGREE_SINE[450 - angle]);
    let beta = f64::from(DEGREE_SINE[angle]);
    let mut points = Vec::with_capacity(361);
    for degree in 0..=360 {
        let x = f64::from(axes[0]) * f64::from(DEGREE_SINE[450 - degree]);
        let y = f64::from(axes[1]) * f64::from(DEGREE_SINE[degree]);
        let point = [
            (f64::from(center[0]) + x * alpha - y * beta).round_ties_even(),
            (f64::from(center[1]) + x * beta + y * alpha).round_ties_even(),
        ];
        if point
            .iter()
            .any(|&v| v < f64::from(i32::MIN) || v > f64::from(i32::MAX))
        {
            return Err(FaceGeometryError::Geometry);
        }
        let point = point.map(|v| v as i32);
        if points.last() != Some(&point) {
            points.push(point);
        }
    }
    if points.len() == 1 {
        points = vec![center; 2];
    }
    Ok(points)
}

/// Fill a convex integer polygon in an interleaved RGB byte image.
/// Rasterizes eight-connected borders and the two fixed-point scan edges.
/// # Errors
/// Rejects invalid image geometry or an empty polygon.
pub fn fill_convex_polygon_rgb(
    values: &mut [u8],
    size: ImageSize,
    points: &[[i32; 2]],
    color: [u8; 3],
) -> Result<(), FaceGeometryError> {
    validate(values, size)?;
    if points.is_empty() {
        return Err(FaceGeometryError::Geometry);
    }
    let n = points.len();
    let mut previous = points[n - 1];
    let mut minimum = 0;
    let mut ymin = points[0][1];
    let mut ymax = ymin;
    let mut xmin = points[0][0];
    let mut xmax = xmin;
    for (i, &point) in points.iter().enumerate() {
        if point[1] < ymin {
            ymin = point[1];
            minimum = i;
        }
        ymax = ymax.max(point[1]);
        xmin = xmin.min(point[0]);
        xmax = xmax.max(point[0]);
        line(values, size, previous, point, color);
        previous = point;
    }
    if n < 3 || xmax < 0 || ymax < 0 || xmin >= size.width as i32 || ymin >= size.height as i32 {
        return Ok(());
    }
    #[derive(Clone, Copy)]
    struct Edge {
        index: usize,
        direction: usize,
        x: i64,
        dx: i64,
        end: i64,
    }
    let mut edges = [
        Edge {
            index: minimum,
            direction: 1,
            x: -65536,
            dx: 0,
            end: i64::from(ymin),
        },
        Edge {
            index: minimum,
            direction: n - 1,
            x: -65536,
            dx: 0,
            end: i64::from(ymin),
        },
    ];
    let mut remaining = n as i64;
    let ymax = i64::from(ymax).min(size.height as i64 - 1);
    for y in i64::from(ymin)..=ymax {
        for edge in &mut edges {
            if y >= edge.end {
                let mut start = edge.index;
                let mut index = (start + edge.direction) % n;
                let mut found = false;
                while remaining > 0 {
                    remaining -= 1;
                    let end = i64::from(points[index][1]);
                    if end > y {
                        let first = i64::from(points[start][0]) << 16;
                        let last = i64::from(points[index][0]) << 16;
                        edge.end = end;
                        edge.dx = ((last - first) * 2 + (end - y)) / (2 * (end - y));
                        edge.x = first;
                        edge.index = index;
                        found = true;
                        break;
                    }
                    start = index;
                    index = (index + edge.direction) % n;
                }
                if !found {
                    remaining = -1;
                }
            }
        }
        if remaining < 0 {
            break;
        }
        let (left, right) = if edges[0].x <= edges[1].x {
            (edges[0].x, edges[1].x)
        } else {
            (edges[1].x, edges[0].x)
        };
        hline(
            values,
            size,
            y,
            (left + 32768) >> 16,
            (right + 32768) >> 16,
            color,
        );
        for edge in &mut edges {
            edge.x += edge.dx;
        }
    }
    Ok(())
}

/// Draw a filled integer circle in interleaved RGB storage using midpoint spans.
/// # Errors
/// Rejects invalid image geometry or a negative radius.
pub fn fill_circle_rgb(
    values: &mut [u8],
    size: ImageSize,
    center: [i32; 2],
    radius: i32,
    color: [u8; 3],
) -> Result<(), FaceGeometryError> {
    validate(values, size)?;
    if radius < 0 {
        return Err(FaceGeometryError::Geometry);
    }
    let (cx, cy) = (i64::from(center[0]), i64::from(center[1]));
    let (mut dx, mut dy, mut error, mut plus, mut minus) =
        (i64::from(radius), 0, 0, 1, i64::from(radius) * 2 - 1);
    while dx >= dy {
        for y in [cy - dy, cy + dy] {
            hline(values, size, y, cx - dx, cx + dx, color);
        }
        for y in [cy - dx, cy + dx] {
            hline(values, size, y, cx - dy, cx + dy, color);
        }
        dy += 1;
        error += plus;
        plus += 2;
        if error > 0 {
            error -= minus;
            dx -= 1;
            minus -= 2;
        }
    }
    Ok(())
}
