//! Fixed-point nonzero-winding grayscale coverage for hinted font outlines.

use std::collections::BTreeMap;

use super::{invalid, Result};

/// Outline coordinates are signed 26.6 pixels, before bitmap placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Position {
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Segment {
    Move(Position),
    Line(Position),
    Conic(Position, Position),
    Cubic(Position, Position, Position),
}

pub(crate) struct Bitmap {
    pub left: i32,
    pub top: i32,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

#[derive(Default)]
struct Cell {
    cover: i64,
    area: i64,
}

struct Raster {
    cells: BTreeMap<(i64, i64), Cell>,
    current: Position,
}

impl Raster {
    fn integrate(&mut self, x: i64, y: i64, dy: i64, sum_x: i64) {
        if dy != 0 {
            let cell = self.cells.entry((y, x)).or_default();
            cell.cover += dy;
            cell.area += dy * sum_x;
        }
    }

    // A line contributes signed height and twice its signed area to every
    // intersected pixel. Intersections are integer quotients in 24.8 space.
    fn line(&mut self, to: Position) {
        let from = self.current;
        self.current = to;
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        if dy == 0 {
            return;
        }
        let (end_x, end_y) = (to.x >> 8, to.y >> 8);
        let (mut x, mut y) = (from.x >> 8, from.y >> 8);
        let (mut fx, mut fy) = (from.x & 255, from.y & 255);
        if dx == 0 {
            while y != end_y {
                let boundary = if dy > 0 { 256 } else { 0 };
                self.integrate(x, y, boundary - fy, 2 * fx);
                y += dy.signum();
                fy = 256 - boundary;
            }
        } else {
            let mut determinant = dx * fy - dy * fx;
            // The reference raster uses a 32-bit reciprocal multiply rather
            // than exact division. Preserve its cell-boundary quantization.
            let reciprocal_x = if x != end_x { 0xffff_ffff_i64 / dx } else { 0 };
            let reciprocal_y = if y != end_y { 0xffff_ffff_i64 / dy } else { 0 };
            let intersection = |value: i64, reciprocal: i64| {
                ((value as u64).wrapping_mul(reciprocal as u64) >> 32) as i64
            };
            while x != end_x || y != end_y {
                if determinant - dx * 256 > 0 && determinant <= 0 {
                    let next_y = intersection(-determinant, -reciprocal_x);
                    determinant -= dy * 256;
                    self.integrate(x, y, next_y - fy, fx);
                    x -= 1;
                    fx = 256;
                    fy = next_y;
                } else if determinant - dx * 256 + dy * 256 > 0 && determinant - dx * 256 <= 0 {
                    determinant -= dx * 256;
                    let next_x = intersection(-determinant, reciprocal_y);
                    self.integrate(x, y, 256 - fy, fx + next_x);
                    y += 1;
                    fx = next_x;
                    fy = 0;
                } else if determinant + dy * 256 >= 0 && determinant - dx * 256 + dy * 256 <= 0 {
                    determinant += dy * 256;
                    let next_y = intersection(determinant, reciprocal_x);
                    self.integrate(x, y, next_y - fy, fx + 256);
                    x += 1;
                    fx = 0;
                    fy = next_y;
                } else {
                    let next_x = intersection(determinant, -reciprocal_y);
                    determinant += dx * 256;
                    self.integrate(x, y, -fy, fx + next_x);
                    y -= 1;
                    fx = next_x;
                    fy = 256;
                }
            }
        }
        self.integrate(x, y, (to.y & 255) - fy, fx + (to.x & 255));
    }

    fn conic(&mut self, control: Position, to: Position) {
        let from = self.current;
        let bx = control.x - from.x;
        let by = control.y - from.y;
        let ax = to.x - control.x - bx;
        let ay = to.y - control.y - by;
        let mut deviation = ax.abs().max(ay.abs());
        if deviation <= 64 {
            self.line(to);
            return;
        }
        let mut level = 0;
        while deviation > 64 {
            deviation >>= 2;
            level += 1;
        }
        // Forward differences of the exact quadratic, using a 32-bit fraction.
        let (rx, ry) = (ax << (33 - 2 * level), ay << (33 - 2 * level));
        let (mut qx, mut qy) = (
            (bx << (33 - level)) + (ax << (32 - 2 * level)),
            (by << (33 - level)) + (ay << (32 - 2 * level)),
        );
        let (mut px, mut py) = (from.x << 32, from.y << 32);
        for _ in 0..1_u32 << level {
            px += qx;
            py += qy;
            qx += rx;
            qy += ry;
            self.line(Position {
                x: px >> 32,
                y: py >> 32,
            });
        }
    }

    fn cubic(&mut self, first: Position, second: Position, to: Position) {
        let mut pending = vec![(self.current, first, second, to)];
        while let Some((a, b, c, d)) = pending.pop() {
            let flat = (2 * a.x - 3 * b.x + d.x)
                .abs()
                .max((2 * a.y - 3 * b.y + d.y).abs())
                .max((a.x - 3 * c.x + 2 * d.x).abs())
                .max((a.y - 3 * c.y + 2 * d.y).abs());
            if flat <= 128 {
                self.line(d);
                continue;
            }
            let ab = Position {
                x: (a.x + b.x) >> 1,
                y: (a.y + b.y) >> 1,
            };
            let cd = Position {
                x: (c.x + d.x) >> 1,
                y: (c.y + d.y) >> 1,
            };
            let abc = Position {
                x: (a.x + 2 * b.x + c.x) >> 2,
                y: (a.y + 2 * b.y + c.y) >> 2,
            };
            let bcd = Position {
                x: (b.x + 2 * c.x + d.x) >> 2,
                y: (b.y + 2 * c.y + d.y) >> 2,
            };
            let center = Position {
                x: (a.x + 3 * b.x + 3 * c.x + d.x) >> 3,
                y: (a.y + 3 * b.y + 3 * c.y + d.y) >> 3,
            };
            pending.push((center, bcd, cd, d));
            pending.push((a, ab, abc, center));
        }
    }
}

pub(crate) fn rasterize(segments: &[Segment]) -> Result<Bitmap> {
    let mut bounds = None;
    let mut include = |p: Position| {
        let limits = bounds.get_or_insert([p.x, p.y, p.x, p.y]);
        limits[0] = limits[0].min(p.x);
        limits[1] = limits[1].min(p.y);
        limits[2] = limits[2].max(p.x);
        limits[3] = limits[3].max(p.y);
    };
    for segment in segments {
        match *segment {
            Segment::Move(p) | Segment::Line(p) => include(p),
            Segment::Conic(c, p) => {
                include(c);
                include(p);
            }
            Segment::Cubic(a, b, p) => {
                include(a);
                include(b);
                include(p);
            }
        }
    }
    let [x0, y0, x1, y1] = bounds.unwrap_or([0; 4]);
    let left = x0 >> 6;
    let bottom = y0 >> 6;
    let right = (x1 + 63) >> 6;
    let top = (y1 + 63) >> 6;
    let width = usize::try_from(right - left).map_err(|_| invalid())?;
    let height = usize::try_from(top - bottom).map_err(|_| invalid())?;
    let mut pixels = vec![0; width.checked_mul(height).ok_or_else(invalid)?];
    let mut raster = Raster {
        cells: BTreeMap::new(),
        current: Position::default(),
    };
    let upscale = |p: Position| Position {
        x: p.x * 4,
        y: p.y * 4,
    };
    for segment in segments {
        match *segment {
            Segment::Move(p) => raster.current = upscale(p),
            Segment::Line(p) => raster.line(upscale(p)),
            Segment::Conic(c, p) => raster.conic(upscale(c), upscale(p)),
            Segment::Cubic(a, b, p) => raster.cubic(upscale(a), upscale(b), upscale(p)),
        }
    }
    for y in bottom..top {
        let mut winding = 0;
        let mut previous_x = left;
        for (&(_, x), cell) in raster.cells.range((y, i64::MIN)..=(y, i64::MAX)) {
            let coverage = winding >> 9_i32;
            let alpha = (if coverage < 0 { !coverage } else { coverage }).min(255) as u8;
            for column in previous_x.max(left)..x.min(right) {
                pixels[(top - 1 - y) as usize * width + (column - left) as usize] = alpha;
            }
            winding += cell.cover * 512;
            if (left..right).contains(&x) {
                let coverage = (winding - cell.area) >> 9_i32;
                pixels[(top - 1 - y) as usize * width + (x - left) as usize] =
                    (if coverage < 0 { !coverage } else { coverage }).min(255) as u8;
            }
            previous_x = x + 1;
        }
        let coverage = winding >> 9_i32;
        let alpha = (if coverage < 0 { !coverage } else { coverage }).min(255) as u8;
        for column in previous_x.max(left)..right {
            pixels[(top - 1 - y) as usize * width + (column - left) as usize] = alpha;
        }
    }
    Ok(Bitmap {
        left: i32::try_from(left).map_err(|_| invalid())?,
        top: i32::try_from(top).map_err(|_| invalid())?,
        width,
        height,
        pixels,
    })
}
