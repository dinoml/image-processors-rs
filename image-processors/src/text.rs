//! Dependency-free TrueType outline rasterization for image text overlays.
use std::{collections::BTreeMap, io};
type Result<T> = io::Result<T>;
fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid or unsupported TrueType font",
    )
}
fn u16_at(b: &[u8], p: usize) -> Result<u16> {
    Ok(u16::from_be_bytes(
        b.get(p..p + 2)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?,
    ))
}
fn i16_at(b: &[u8], p: usize) -> Result<i16> {
    Ok(u16_at(b, p)? as i16)
}
fn u32_at(b: &[u8], p: usize) -> Result<u32> {
    Ok(u32::from_be_bytes(
        b.get(p..p + 4)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?,
    ))
}
#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
    on: bool,
}
impl Point {
    fn midpoint(self, b: Self) -> Self {
        Self {
            x: (self.x + b.x) * 0.5,
            y: (self.y + b.y) * 0.5,
            on: true,
        }
    }
}
/// A parsed TrueType font. Rendering uses outlines, without native font libraries.
/// TrueType bytecode hinting and CFF outlines are not supported.
pub struct TrueTypeFont {
    data: Vec<u8>,
    tables: BTreeMap<[u8; 4], (usize, usize)>,
    units: f32,
    ascender: i16,
    metrics: usize,
    glyphs: usize,
    long_loca: bool,
}
impl TrueTypeFont {
    /// Parse an sfnt TrueType font containing Unicode mappings and quadratic outlines.
    /// # Errors
    /// Returns InvalidData for malformed tables or unsupported outline formats.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if u32_at(&data, 0)? != 0x00010000 && data.get(..4) != Some(b"true") {
            return Err(invalid());
        }
        let count = usize::from(u16_at(&data, 4)?);
        let mut tables = BTreeMap::new();
        for i in 0..count {
            let p = 12 + i * 16;
            let tag = data
                .get(p..p + 4)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?;
            let offset = u32_at(&data, p + 8)? as usize;
            let length = u32_at(&data, p + 12)? as usize;
            data.get(offset..offset.checked_add(length).ok_or_else(invalid)?)
                .ok_or_else(invalid)?;
            tables.insert(tag, (offset, length));
        }
        let table = |name: &[u8; 4]| -> Result<&[u8]> {
            let &(o, n) = tables.get(name).ok_or_else(invalid)?;
            data.get(o..o + n).ok_or_else(invalid)
        };
        let head = table(b"head")?;
        let units = f32::from(u16_at(head, 18)?);
        let loca = i16_at(head, 50)?;
        let hhea = table(b"hhea")?;
        let ascender = i16_at(hhea, 4)?;
        let metrics = usize::from(u16_at(hhea, 34)?);
        let glyphs = usize::from(u16_at(table(b"maxp")?, 4)?);
        if units == 0.0 || metrics == 0 || metrics > glyphs || !matches!(loca, 0 | 1) {
            return Err(invalid());
        }
        table(b"glyf")?;
        table(b"loca")?;
        table(b"hmtx")?;
        table(b"cmap")?;
        Ok(Self {
            data,
            tables,
            units,
            ascender,
            metrics,
            glyphs,
            long_loca: loca == 1,
        })
    }
    fn table(&self, name: &[u8; 4]) -> Result<&[u8]> {
        let &(o, n) = self.tables.get(name).ok_or_else(invalid)?;
        self.data.get(o..o + n).ok_or_else(invalid)
    }
    fn glyph(&self, c: char) -> Result<usize> {
        let cmap = self.table(b"cmap")?;
        let count = usize::from(u16_at(cmap, 2)?);
        let code = c as u32;
        for desired in [12, 4] {
            for i in 0..count {
                let record = 4 + i * 8;
                let platform = u16_at(cmap, record)?;
                let encoding = u16_at(cmap, record + 2)?;
                if platform != 0 && !(platform == 3 && matches!(encoding, 1 | 10)) {
                    continue;
                }
                let offset = u32_at(cmap, record + 4)? as usize;
                let sub = cmap.get(offset..).ok_or_else(invalid)?;
                if u16_at(sub, 0)? != desired {
                    continue;
                }
                if desired == 12 {
                    let length = u32_at(sub, 4)? as usize;
                    let sub = sub.get(..length).ok_or_else(invalid)?;
                    let groups = u32_at(sub, 12)? as usize;
                    if groups > sub.len().saturating_sub(16) / 12 {
                        return Err(invalid());
                    }
                    for g in 0..groups {
                        let p = 16 + g * 12;
                        let start = u32_at(sub, p)?;
                        let end = u32_at(sub, p + 4)?;
                        if code >= start && code <= end {
                            let id = u32_at(sub, p + 8)?
                                .checked_add(code - start)
                                .ok_or_else(invalid)? as usize;
                            if id == 0 || id >= self.glyphs {
                                return Err(invalid());
                            }
                            return Ok(id);
                        }
                    }
                } else if code <= 65535 {
                    let length = usize::from(u16_at(sub, 2)?);
                    let sub = sub.get(..length).ok_or_else(invalid)?;
                    let n = usize::from(u16_at(sub, 6)?) / 2;
                    let ends = 14;
                    let starts = ends + n * 2 + 2;
                    let deltas = starts + n * 2;
                    let ranges = deltas + n * 2;
                    for g in 0..n {
                        let end = u32::from(u16_at(sub, ends + g * 2)?);
                        let start = u32::from(u16_at(sub, starts + g * 2)?);
                        if code < start || code > end {
                            continue;
                        }
                        let delta = u16_at(sub, deltas + g * 2)?;
                        let range = usize::from(u16_at(sub, ranges + g * 2)?);
                        let id = if range == 0 {
                            (code as u16).wrapping_add(delta)
                        } else {
                            let p = ranges + g * 2 + range + (code - start) as usize * 2;
                            let id = u16_at(sub, p)?;
                            if id == 0 {
                                0
                            } else {
                                id.wrapping_add(delta)
                            }
                        };
                        if id == 0 || usize::from(id) >= self.glyphs {
                            return Err(invalid());
                        }
                        return Ok(usize::from(id));
                    }
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "font does not contain a requested character",
        ))
    }
    fn contours(&self, id: usize, depth: usize) -> Result<Vec<Vec<Point>>> {
        if id >= self.glyphs || depth > 16 {
            return Err(invalid());
        }
        let loca = self.table(b"loca")?;
        let offset = |i| -> Result<usize> {
            if self.long_loca {
                Ok(u32_at(loca, i * 4)? as usize)
            } else {
                Ok(usize::from(u16_at(loca, i * 2)?) * 2)
            }
        };
        let start = offset(id)?;
        let end = offset(id + 1)?;
        if start == end {
            return Ok(Vec::new());
        }
        let b = self.table(b"glyf")?.get(start..end).ok_or_else(invalid)?;
        let count = i16_at(b, 0)?;
        if count >= 0 {
            let n = count as usize;
            if n == 0 {
                return Ok(Vec::new());
            }
            let mut ends = Vec::with_capacity(n);
            for i in 0..n {
                let end = usize::from(u16_at(b, 10 + i * 2)?);
                if ends.last().is_some_and(|previous| end <= *previous) {
                    return Err(invalid());
                }
                ends.push(end);
            }
            let points = ends[n - 1] + 1;
            let instructions = 10 + n * 2;
            let mut pos = instructions + 2 + usize::from(u16_at(b, instructions)?);
            let mut flags = Vec::with_capacity(points);
            while flags.len() < points {
                let flag = *b.get(pos).ok_or_else(invalid)?;
                pos += 1;
                let repeat = if flag & 8 != 0 {
                    let v = usize::from(*b.get(pos).ok_or_else(invalid)?);
                    pos += 1;
                    v
                } else {
                    0
                };
                if flags.len() + repeat + 1 > points {
                    return Err(invalid());
                }
                flags.extend(std::iter::repeat_n(flag, repeat + 1));
            }
            let mut x = 0i32;
            let mut coords = Vec::with_capacity(points);
            for &flag in &flags {
                let delta = if flag & 2 != 0 {
                    let v = i32::from(*b.get(pos).ok_or_else(invalid)?);
                    pos += 1;
                    if flag & 16 != 0 {
                        v
                    } else {
                        -v
                    }
                } else if flag & 16 != 0 {
                    0
                } else {
                    let v = i32::from(i16_at(b, pos)?);
                    pos += 2;
                    v
                };
                x = x.checked_add(delta).ok_or_else(invalid)?;
                coords.push(Point {
                    x: x as f32,
                    y: 0.0,
                    on: flag & 1 != 0,
                });
            }
            let mut y = 0i32;
            for (point, &flag) in coords.iter_mut().zip(&flags) {
                let delta = if flag & 4 != 0 {
                    let v = i32::from(*b.get(pos).ok_or_else(invalid)?);
                    pos += 1;
                    if flag & 32 != 0 {
                        v
                    } else {
                        -v
                    }
                } else if flag & 32 != 0 {
                    0
                } else {
                    let v = i32::from(i16_at(b, pos)?);
                    pos += 2;
                    v
                };
                y = y.checked_add(delta).ok_or_else(invalid)?;
                point.y = y as f32;
            }
            let mut contours = Vec::with_capacity(n);
            let mut start = 0;
            for end in ends {
                contours.push(coords[start..=end].to_vec());
                start = end + 1;
            }
            Ok(contours)
        } else {
            let mut pos = 10;
            let mut contours = Vec::new();
            loop {
                let flags = u16_at(b, pos)?;
                let glyph = usize::from(u16_at(b, pos + 2)?);
                pos += 4;
                if flags & 2 == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point-attached composite glyphs are unsupported",
                    ));
                }
                let (dx, dy) = if flags & 1 != 0 {
                    let pair = (f32::from(i16_at(b, pos)?), f32::from(i16_at(b, pos + 2)?));
                    pos += 4;
                    pair
                } else {
                    let pair = (
                        *b.get(pos).ok_or_else(invalid)? as i8 as f32,
                        *b.get(pos + 1).ok_or_else(invalid)? as i8 as f32,
                    );
                    pos += 2;
                    pair
                };
                let mut matrix = [1.0, 0.0, 0.0, 1.0];
                if flags & 8 != 0 {
                    let scale = f32::from(i16_at(b, pos)?) / 16384.0;
                    pos += 2;
                    matrix = [scale, 0.0, 0.0, scale];
                } else if flags & 64 != 0 {
                    matrix[0] = f32::from(i16_at(b, pos)?) / 16384.0;
                    matrix[3] = f32::from(i16_at(b, pos + 2)?) / 16384.0;
                    pos += 4;
                } else if flags & 128 != 0 {
                    for v in &mut matrix {
                        *v = f32::from(i16_at(b, pos)?) / 16384.0;
                        pos += 2;
                    }
                }
                let (dx, dy) = if flags & 0x800 != 0 {
                    (
                        matrix[0] * dx + matrix[2] * dy,
                        matrix[1] * dx + matrix[3] * dy,
                    )
                } else {
                    (dx, dy)
                };
                for mut contour in self.contours(glyph, depth + 1)? {
                    for point in &mut contour {
                        let x = point.x;
                        let y = point.y;
                        point.x = matrix[0] * x + matrix[2] * y + dx;
                        point.y = matrix[1] * x + matrix[3] * y + dy;
                    }
                    contours.push(contour);
                }
                if flags & 32 == 0 {
                    break;
                }
            }
            Ok(contours)
        }
    }
    fn kerning(&self, left: usize, right: usize) -> Result<i16> {
        let Ok(kern) = self.table(b"kern") else {
            return Ok(0);
        };
        if u16_at(kern, 0)? != 0 {
            return Ok(0);
        }
        let mut pos = 4;
        let mut adjustment = 0i16;
        for _ in 0..u16_at(kern, 2)? {
            let length = usize::from(u16_at(kern, pos + 2)?);
            if length < 6 {
                return Err(invalid());
            }
            let sub = kern.get(pos..pos + length).ok_or_else(invalid)?;
            let coverage = u16_at(sub, 4)?;
            if coverage >> 8 == 0 && coverage & 7 == 1 {
                let pairs = usize::from(u16_at(sub, 6)?);
                if pairs > sub.len().saturating_sub(14) / 6 {
                    return Err(invalid());
                }
                for i in 0..pairs {
                    let p = 14 + i * 6;
                    if usize::from(u16_at(sub, p)?) == left
                        && usize::from(u16_at(sub, p + 2)?) == right
                    {
                        let v = i16_at(sub, p + 4)?;
                        adjustment = if coverage & 8 != 0 {
                            v
                        } else {
                            adjustment.saturating_add(v)
                        };
                        break;
                    }
                }
            }
            pos += length;
        }
        Ok(adjustment)
    }
    /// Draw black text with a rounded white outline at the top-left of an RGB image.
    /// Uses the font's ascender, advances, kerning, quadratic outlines and 4×4 coverage.
    /// # Errors
    /// Returns an error for invalid image geometry, font data or missing characters.
    pub fn draw_rgb(
        &self,
        pixels: &mut [u8],
        width: usize,
        height: usize,
        text: &str,
        pixel_size: u32,
        stroke_width: u32,
    ) -> Result<()> {
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(pixels.len())
            || pixel_size == 0
            || stroke_width > pixel_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid text overlay geometry",
            ));
        }
        let scale = pixel_size as f32 / self.units;
        let baseline = (f32::from(self.ascender) * scale).ceil();
        let mut pen = 0.0;
        let mut previous = None;
        let metrics = self.table(b"hmtx")?;
        let mut edges = Vec::new();
        for character in text.chars() {
            let id = self.glyph(character)?;
            if let Some(left) = previous {
                pen += f32::from(self.kerning(left, id)?) * scale;
            }
            for contour in self.contours(id, 0)? {
                let points: Vec<_> = contour
                    .into_iter()
                    .map(|p| Point {
                        x: pen + p.x * scale,
                        y: baseline - p.y * scale,
                        on: p.on,
                    })
                    .collect();
                outline(&points, &mut edges);
            }
            pen += f32::from(u16_at(metrics, id.min(self.metrics - 1) * 4)?) * scale;
            previous = Some(id);
        }
        if edges.is_empty() {
            return Ok(());
        }
        let radius = stroke_width as f32;
        let min_x = edges
            .iter()
            .map(|(a, b)| a.x.min(b.x))
            .fold(f32::INFINITY, f32::min);
        let max_x = edges
            .iter()
            .map(|(a, b)| a.x.max(b.x))
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = edges
            .iter()
            .map(|(a, b)| a.y.min(b.y))
            .fold(f32::INFINITY, f32::min);
        let max_y = edges
            .iter()
            .map(|(a, b)| a.y.max(b.y))
            .fold(f32::NEG_INFINITY, f32::max);
        let left = (min_x - radius).floor().max(0.0) as usize;
        let right = ((max_x + radius).ceil().max(0.0) as usize).min(width);
        let top = (min_y - radius).floor().max(0.0) as usize;
        let bottom = ((max_y + radius).ceil().max(0.0) as usize).min(height);
        let mut coverage = vec![[0u32; 2]; right.saturating_sub(left)];
        for y in top..bottom {
            coverage.fill([0, 0]);
            for sy in 0..4 {
                let py = y as f32 + (sy as f32 + 0.5) / 4.0;
                let mut crossings = Vec::new();
                let mut strokes = Vec::new();
                for &(a, b) in &edges {
                    if (a.y <= py && b.y > py) || (b.y <= py && a.y > py) {
                        crossings.push((
                            a.x + (py - a.y) * (b.x - a.x) / (b.y - a.y),
                            if b.y > a.y { 1 } else { -1 },
                        ));
                    }
                    if radius > 0.0 {
                        for point in [a, b] {
                            let dy = py - point.y;
                            if dy.abs() <= radius {
                                let dx = (radius * radius - dy * dy).max(0.0).sqrt();
                                strokes.push((point.x - dx, point.x + dx));
                            }
                        }
                        let dx = b.x - a.x;
                        let dy = b.y - a.y;
                        let length = dx.hypot(dy);
                        if length > 0.0 {
                            let nx = -dy * radius / length;
                            let ny = dx * radius / length;
                            let corners = [
                                (a.x + nx, a.y + ny),
                                (b.x + nx, b.y + ny),
                                (b.x - nx, b.y - ny),
                                (a.x - nx, a.y - ny),
                            ];
                            let mut lo = f32::INFINITY;
                            let mut hi = f32::NEG_INFINITY;
                            for i in 0..4 {
                                let (ax, ay) = corners[i];
                                let (bx, by) = corners[(i + 1) % 4];
                                if (ay <= py && by > py) || (by <= py && ay > py) {
                                    let x = ax + (py - ay) * (bx - ax) / (by - ay);
                                    lo = lo.min(x);
                                    hi = hi.max(x);
                                }
                            }
                            if lo <= hi {
                                strokes.push((lo, hi));
                            }
                        }
                    }
                }
                crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
                strokes.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut union: Vec<(f32, f32)> = Vec::new();
                for (lo, hi) in strokes {
                    if let Some(last) = union.last_mut().filter(|last| lo <= last.1) {
                        last.1 = last.1.max(hi);
                    } else {
                        union.push((lo, hi));
                    }
                }
                let (mut crossing, mut stroke, mut winding) = (0, 0, 0);
                for x in left..right {
                    for sx in 0..4 {
                        let px = x as f32 + (sx as f32 + 0.5) / 4.0;
                        while crossing < crossings.len() && crossings[crossing].0 <= px {
                            winding += crossings[crossing].1;
                            crossing += 1;
                        }
                        while stroke < union.len() && union[stroke].1 < px {
                            stroke += 1;
                        }
                        let inside = winding != 0;
                        coverage[x - left][0] += u32::from(inside);
                        coverage[x - left][1] += u32::from(
                            inside
                                || union
                                    .get(stroke)
                                    .is_some_and(|&(lo, hi)| px >= lo && px <= hi),
                        );
                    }
                }
            }
            for (x, [fill, stroke]) in (left..right).zip(coverage.iter().copied()) {
                let alpha = fill * 255 / 16;
                let outline_alpha = stroke * 255 / 16;
                for channel in &mut pixels[(y * width + x) * 3..(y * width + x) * 3 + 3] {
                    let background =
                        (255 * outline_alpha + u32::from(*channel) * (255 - outline_alpha) + 127)
                            / 255;
                    *channel = ((background * (255 - alpha) + 127) / 255) as u8;
                }
            }
        }
        Ok(())
    }
}
fn quadratic(a: Point, b: Point, c: Point, depth: u8, edges: &mut Vec<(Point, Point)>) {
    let dx = a.x - 2.0 * b.x + c.x;
    let dy = a.y - 2.0 * b.y + c.y;
    if depth >= 12 || dx * dx + dy * dy <= 0.16 {
        edges.push((a, c));
    } else {
        let ab = a.midpoint(b);
        let bc = b.midpoint(c);
        let mid = ab.midpoint(bc);
        quadratic(a, ab, mid, depth + 1, edges);
        quadratic(mid, bc, c, depth + 1, edges);
    }
}
fn outline(points: &[Point], edges: &mut Vec<(Point, Point)>) {
    if points.is_empty() {
        return;
    }
    let n = points.len();
    let last = points[n - 1];
    let (start, mut index) = if points[0].on {
        (points[0], 1)
    } else if last.on {
        (last, 0)
    } else {
        (last.midpoint(points[0]), 0)
    };
    let mut current = start;
    let end = index + n;
    while index < end {
        let p = points[index % n];
        if p.on {
            edges.push((current, p));
            current = p;
            index += 1;
        } else {
            let next = points[(index + 1) % n];
            let target = if next.on { next } else { p.midpoint(next) };
            quadratic(current, p, target, 0, edges);
            current = target;
            index += if next.on { 2 } else { 1 };
        }
    }
    if current.x != start.x || current.y != start.y {
        edges.push((current, start));
    }
}
