//! Integer glyph loading for the native TrueType instruction interpreter.

use super::{
    gray::{self, Bitmap, Position, Segment},
    hinting::{mul_div, Graphics, Interpreter, Zone},
    i16_at, invalid, u16_at, u32_at, Result, TrueTypeFont,
};

pub(super) struct Face {
    pub font: TrueTypeFont,
    pub interpreter: Interpreter,
    pub graphics: Graphics,
    pub ascender: i64,
    pub descender: i64,
    pub requires_autofit: bool,
    automatic: std::collections::BTreeMap<u16, super::autofit::Metrics>,
    styles: Vec<(u16, bool)>,
    cff: bool,
    cff_random: Vec<u32>,
}

pub(super) struct Glyph {
    pub outline: Zone,
    pub advance: i64,
    pub origin: i64,
    phantoms: [Position; 4],
}

pub(super) struct RenderedGlyph {
    pub bitmap: Bitmap,
    pub advance: i64,
}

impl Face {
    pub fn new(bytes: Vec<u8>, pixel_size: u32, seed: &mut u32) -> Result<Self> {
        let font = TrueTypeFont::from_sfnt(bytes, true)?;
        let cff = font.table(b"CFF ").is_ok();
        let head = font.table(b"head")?;
        let units = u16_at(head, 18)?;
        let maxp = font.table(b"maxp")?;
        let twilight = if cff {
            0
        } else {
            usize::from(u16_at(maxp, 16)?)
        };
        let storage = if cff {
            0
        } else {
            usize::from(u16_at(maxp, 18)?)
        };
        let cvt = font
            .table(b"cvt ")
            .unwrap_or(&[])
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_be_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        let mut interpreter = Interpreter::new(pixel_size, units, twilight, storage, &cvt)?;
        let fpgm = font.table(b"fpgm").unwrap_or(&[]);
        let prep = font.table(b"prep").unwrap_or(&[]);
        let requires_autofit = !cff && u16_at(maxp, 26)? == 0 && fpgm.is_empty() && prep.is_empty();
        interpreter.prepare(fpgm, prep)?;
        let graphics = interpreter.graphics.clone();
        let hhea = font.table(b"hhea")?;
        let (mut ascender, mut descender) =
            (i64::from(i16_at(hhea, 4)?), i64::from(i16_at(hhea, 6)?));
        if let Ok(os2) = font.table(b"OS/2") {
            if u16_at(os2, 0)? != 0xffff
                && (u16_at(os2, 62)? & 128 != 0 || ascender == 0 && descender == 0)
            {
                ascender = i64::from(i16_at(os2, 68)?);
                descender = i64::from(i16_at(os2, 70)?);
                if u16_at(os2, 62)? & 128 == 0 && ascender == 0 && descender == 0 {
                    ascender = i64::from(u16_at(os2, 74)? as i16);
                    descender = -i64::from(u16_at(os2, 76)? as i16);
                }
            }
        }
        let ascender = (mul_div(ascender, interpreter.scale, 65536)? + 63) & !63;
        let descender = mul_div(descender, interpreter.scale, 65536)? & !63;
        let cff_random = if cff {
            super::cff::Font::parse(font.table(b"CFF ")?)?.random_states(seed)
        } else {
            Vec::new()
        };
        let styles = if requires_autofit {
            super::autofit::styles(&font)?
        } else {
            Vec::new()
        };
        Ok(Self {
            font,
            interpreter,
            graphics,
            ascender,
            descender,
            requires_autofit,
            automatic: std::collections::BTreeMap::new(),
            styles,
            cff,
            cff_random,
        })
    }

    fn data(&self, id: usize) -> Result<Vec<u8>> {
        if id >= self.font.glyphs {
            return Err(invalid());
        }
        let loca = self.font.table(b"loca")?;
        let offset = |index: usize| -> Result<usize> {
            if self.font.long_loca {
                Ok(u32_at(loca, index * 4)? as usize)
            } else {
                Ok(usize::from(u16_at(loca, index * 2)?) * 2)
            }
        };
        self.font
            .table(b"glyf")?
            .get(offset(id)?..offset(id + 1)?)
            .map(<[u8]>::to_vec)
            .ok_or_else(invalid)
    }

    fn metrics(&self, id: usize, data: &[u8]) -> Result<(i64, [Position; 4])> {
        let hmtx = self.font.table(b"hmtx")?;
        let advance = i64::from(u16_at(hmtx, id.min(self.font.metrics - 1) * 4)?);
        let bearing = i64::from(i16_at(
            hmtx,
            if id < self.font.metrics {
                id * 4 + 2
            } else {
                self.font.metrics * 4 + (id - self.font.metrics) * 2
            },
        )?);
        let x_min = if data.is_empty() {
            0
        } else {
            i64::from(i16_at(data, 2)?)
        };
        let y_min = if data.is_empty() {
            0
        } else {
            i64::from(i16_at(data, 4)?)
        };
        let y_max = if data.is_empty() {
            0
        } else {
            i64::from(i16_at(data, 8)?)
        };
        let hhea = self.font.table(b"hhea")?;
        let vertical_advance = if let Ok(os2) = self.font.table(b"OS/2") {
            i64::from(i16_at(os2, 68)?) - i64::from(i16_at(os2, 70)?)
        } else {
            i64::from(i16_at(hhea, 4)?) - i64::from(i16_at(hhea, 6)?)
        };
        let mut top_bearing = (vertical_advance - (y_max - y_min)) / 2;
        let mut vertical_advance = vertical_advance;
        if let (Ok(vhea), Ok(vmtx)) = (self.font.table(b"vhea"), self.font.table(b"vmtx")) {
            let count = usize::from(u16_at(vhea, 34)?);
            if count > 0 {
                vertical_advance = i64::from(u16_at(vmtx, id.min(count - 1) * 4)?);
                top_bearing = i64::from(i16_at(
                    vmtx,
                    if id < count {
                        id * 4 + 2
                    } else {
                        count * 4 + (id - count) * 2
                    },
                )?);
            }
        }
        Ok((
            advance,
            [
                Position {
                    x: x_min - bearing,
                    y: 0,
                },
                Position {
                    x: x_min - bearing + advance,
                    y: 0,
                },
                Position {
                    x: advance / 2,
                    y: y_max + top_bearing,
                },
                Position {
                    x: advance / 2,
                    y: y_max + top_bearing - vertical_advance,
                },
            ],
        ))
    }

    pub fn load(&mut self, id: usize, depth: usize) -> Result<Glyph> {
        self.load_outline(id, depth, false)
    }

    pub fn unscaled(&mut self, id: usize) -> Result<Glyph> {
        let mut glyph = self.load_outline(id, 0, true)?;
        for point in &mut glyph.outline.current {
            point.x -= glyph.origin;
        }
        glyph.outline.original = glyph.outline.current.clone();
        glyph.outline.unscaled = glyph.outline.current.clone();
        glyph.origin = 0;
        Ok(glyph)
    }

    fn load_outline(&mut self, id: usize, depth: usize, unscaled: bool) -> Result<Glyph> {
        if depth > 32 {
            return Err(invalid());
        }
        let data = self.data(id)?;
        let (advance, phantoms) = self.metrics(id, &data)?;
        let count = if data.is_empty() {
            0
        } else {
            i16_at(&data, 0)?
        };
        let mut zone = Zone::default();
        let mut program = Vec::new();
        let scale = if unscaled {
            65536
        } else {
            self.interpreter.scale
        };
        let mut device_phantoms = [Position::default(); 4];
        for (target, source) in device_phantoms.iter_mut().zip(phantoms) {
            *target = Position {
                x: mul_div(source.x, scale, 65536)?,
                y: mul_div(source.y, scale, 65536)?,
            };
        }
        let mut origin = mul_div(phantoms[0].x, scale, 65536)?;
        let mut device_advance = mul_div(advance, scale, 65536)?;
        if count > 0 {
            for index in 0..count as usize {
                zone.ends.push(usize::from(u16_at(&data, 10 + index * 2)?));
            }
            let points = zone
                .ends
                .last()
                .copied()
                .ok_or_else(invalid)?
                .checked_add(1)
                .ok_or_else(invalid)?;
            if zone.ends.windows(2).any(|ends| ends[0] >= ends[1]) {
                return Err(invalid());
            }
            let mut cursor = 10 + count as usize * 2;
            let instructions = usize::from(u16_at(&data, cursor)?);
            cursor += 2;
            program.extend_from_slice(
                data.get(cursor..cursor + instructions)
                    .ok_or_else(invalid)?,
            );
            cursor += instructions;
            let mut flags = Vec::with_capacity(points);
            while flags.len() < points {
                let flag = *data.get(cursor).ok_or_else(invalid)?;
                cursor += 1;
                let repetitions = if flag & 8 != 0 {
                    let n = usize::from(*data.get(cursor).ok_or_else(invalid)?);
                    cursor += 1;
                    n + 1
                } else {
                    1
                };
                if repetitions > points - flags.len() {
                    return Err(invalid());
                }
                flags.extend(std::iter::repeat_n(flag, repetitions));
            }
            zone.unscaled = vec![Position::default(); points];
            for axis in 0..2 {
                let mut coordinate = 0;
                for (point, &flag) in zone.unscaled.iter_mut().zip(&flags) {
                    let short = if axis == 0 { 2 } else { 4 };
                    let same = if axis == 0 { 16 } else { 32 };
                    let delta = if flag & short != 0 {
                        let value = i64::from(*data.get(cursor).ok_or_else(invalid)?);
                        cursor += 1;
                        if flag & same != 0 {
                            value
                        } else {
                            -value
                        }
                    } else if flag & same != 0 {
                        0
                    } else {
                        let value = i64::from(i16_at(&data, cursor)?);
                        cursor += 2;
                        value
                    };
                    coordinate += delta;
                    if axis == 0 {
                        point.x = coordinate;
                    } else {
                        point.y = coordinate;
                    }
                }
            }
            zone.on_curve = flags.iter().map(|flag| flag & 1 != 0).collect();
            zone.current = zone
                .unscaled
                .iter()
                .map(|point| {
                    Ok(Position {
                        x: mul_div(point.x, scale, 65536)?,
                        y: mul_div(point.y, scale, 65536)?,
                    })
                })
                .collect::<Result<_>>()?;
        } else if count < 0 {
            let mut cursor = 10;
            let mut last_flags;
            loop {
                let flags = u16_at(&data, cursor)?;
                let child = usize::from(u16_at(&data, cursor + 2)?);
                cursor += 4;
                let signed = flags & 2 != 0;
                let (a, b) = if flags & 1 != 0 {
                    let values = if signed {
                        (
                            i64::from(i16_at(&data, cursor)?),
                            i64::from(i16_at(&data, cursor + 2)?),
                        )
                    } else {
                        (
                            i64::from(u16_at(&data, cursor)?),
                            i64::from(u16_at(&data, cursor + 2)?),
                        )
                    };
                    cursor += 4;
                    values
                } else {
                    let a = *data.get(cursor).ok_or_else(invalid)?;
                    let b = *data.get(cursor + 1).ok_or_else(invalid)?;
                    cursor += 2;
                    if signed {
                        (i64::from(a as i8), i64::from(b as i8))
                    } else {
                        (i64::from(a), i64::from(b))
                    }
                };
                let mut matrix = [16384, 0, 0, 16384];
                if flags & 8 != 0 {
                    let value = i64::from(i16_at(&data, cursor)?);
                    cursor += 2;
                    matrix = [value, 0, 0, value];
                } else if flags & 64 != 0 {
                    matrix[0] = i64::from(i16_at(&data, cursor)?);
                    matrix[3] = i64::from(i16_at(&data, cursor + 2)?);
                    cursor += 4;
                } else if flags & 128 != 0 {
                    for value in &mut matrix {
                        *value = i64::from(i16_at(&data, cursor)?);
                        cursor += 2;
                    }
                }
                let mut child = self.load_outline(child, depth + 1, unscaled)?;
                if flags & 0x200 != 0 {
                    device_phantoms = child.phantoms;
                    origin = device_phantoms[0].x;
                    device_advance = device_phantoms[1].x - origin;
                }
                for point in &mut child.outline.current {
                    let original = *point;
                    point.x = mul_div(original.x, matrix[0], 16384)?
                        + mul_div(original.y, matrix[2], 16384)?;
                    point.y = mul_div(original.x, matrix[1], 16384)?
                        + mul_div(original.y, matrix[3], 16384)?;
                }
                let offset = if signed {
                    let (a, b) = if flags & 0x800 != 0 && matrix != [16384, 0, 0, 16384] {
                        (
                            mul_div(a, vector_length(matrix[0] * 4, matrix[2] * 4)?, 65536)?,
                            mul_div(b, vector_length(matrix[3] * 4, matrix[1] * 4)?, 65536)?,
                        )
                    } else {
                        (a, b)
                    };
                    let x = mul_div(a, scale, 65536)?;
                    let y = mul_div(b, scale, 65536)?;
                    Position {
                        x,
                        y: if !unscaled
                            && self.graphics.instruction_control & 1 == 0
                            && flags & 4 != 0
                        {
                            (y + 32) & !63
                        } else {
                            y
                        },
                    }
                } else {
                    let parent = *zone
                        .current
                        .get(usize::try_from(a).map_err(|_| invalid())?)
                        .ok_or_else(invalid)?;
                    let child = *child
                        .outline
                        .current
                        .get(usize::try_from(b).map_err(|_| invalid())?)
                        .ok_or_else(invalid)?;
                    Position {
                        x: parent.x - child.x,
                        y: parent.y - child.y,
                    }
                };
                let start = zone.current.len();
                for point in &mut child.outline.current {
                    point.x += offset.x;
                    point.y += offset.y;
                }
                zone.ends
                    .extend(child.outline.ends.iter().map(|end| start + end));
                zone.current.extend(child.outline.current);
                zone.on_curve.extend(child.outline.on_curve);
                last_flags = flags;
                if flags & 32 == 0 {
                    break;
                }
            }
            if last_flags & 256 != 0 {
                let length = usize::from(u16_at(&data, cursor)?);
                cursor += 2;
                program.extend_from_slice(data.get(cursor..cursor + length).ok_or_else(invalid)?);
            }
            zone.unscaled = zone.current.clone();
        }
        if unscaled {
            zone.original = zone.current.clone();
            zone.unscaled = zone.current.clone();
            zone.touched = vec![[false; 2]; zone.current.len()];
            return Ok(Glyph {
                outline: zone,
                advance: device_advance,
                origin,
                phantoms: device_phantoms,
            });
        }
        let real_points = zone.current.len();
        for point in device_phantoms {
            zone.current.push(point);
            if count < 0 {
                zone.unscaled.push(point);
            }
        }
        if count >= 0 {
            zone.unscaled.extend(phantoms);
        }
        zone.original = zone.current.clone();
        zone.on_curve.resize(real_points + 4, true);
        zone.touched = vec![[false; 2]; real_points + 4];
        let hinted =
            self.graphics.instruction_control & 1 == 0 && (count >= 0 || !program.is_empty());
        if hinted {
            zone.current[real_points].x = (zone.current[real_points].x + 32) & !63;
            zone.current[real_points + 1].x = (zone.current[real_points + 1].x + 32) & !63;
            zone.current[real_points + 2].y = (zone.current[real_points + 2].y + 32) & !63;
            zone.current[real_points + 3].y = (zone.current[real_points + 3].y + 32) & !63;
        }
        self.interpreter.glyph(
            zone,
            if hinted { &program } else { &[] },
            &self.graphics,
            count < 0,
        )?;
        let mut zone = std::mem::take(&mut self.interpreter.zones[1]);
        if hinted && self.interpreter.graphics.instruction_control & 4 != 0 {
            origin = zone.current[real_points].x;
            device_advance = zone.current[real_points + 1].x - origin;
            device_phantoms.copy_from_slice(&zone.current[real_points..real_points + 4]);
        }
        zone.current.truncate(real_points);
        zone.original.truncate(real_points);
        zone.unscaled.truncate(real_points);
        zone.on_curve.truncate(real_points);
        zone.touched.truncate(real_points);
        Ok(Glyph {
            outline: zone,
            advance: (device_advance + 32) & !63,
            origin,
            phantoms: device_phantoms,
        })
    }

    pub fn render(&mut self, character: char) -> Result<RenderedGlyph> {
        let id = self.font.glyph_index(character)?;
        if self.cff {
            let font = super::cff::Font::parse(self.font.table(b"CFF ")?)?;
            let (matrix, top_units) =
                font.transform(id, i64::from(u16_at(self.font.table(b"head")?, 18)?))?;
            let scale = mul_div(self.interpreter.scale, top_units, matrix.units)?;
            let glyph = font.decode(id, &mut self.cff_random)?;
            let mut segments = if let Some((offset, base, accent)) = glyph.composite {
                let accent = font.seac(accent)?;
                let base = font.seac(base)?;
                let mut segments =
                    super::cff_hint::outline(&glyph, font.private(id)?, (scale + 32) / 64)?;
                let mut accent_glyph = font.decode(accent, &mut self.cff_random)?;
                let base_glyph = font.decode(base, &mut self.cff_random)?;
                if accent_glyph.composite.is_some() || base_glyph.composite.is_some() {
                    return Err(invalid());
                }
                let translate = |point: &mut super::cff::Point| {
                    point.x += offset.x;
                    point.y += offset.y;
                };
                for command in &mut accent_glyph.commands {
                    match command {
                        super::cff::Command::Move(point) | super::cff::Command::Line(point) => {
                            translate(point)
                        }
                        super::cff::Command::Curve(a, b, end) => {
                            translate(a);
                            translate(b);
                            translate(end);
                        }
                        super::cff::Command::Close => {}
                    }
                }
                for stem in &mut accent_glyph.stems {
                    stem.start += if stem.vertical { offset.x } else { offset.y };
                }
                segments.extend(super::cff_hint::outline(
                    &accent_glyph,
                    font.private(id)?,
                    (scale + 32) / 64,
                )?);
                segments.extend(super::cff_hint::outline(
                    &base_glyph,
                    font.private(id)?,
                    (scale + 32) / 64,
                )?);
                segments
            } else {
                super::cff_hint::outline(&glyph, font.private(id)?, (scale + 32) / 64)?
            };
            let transform = |point: &mut Position| -> Result<()> {
                let original = *point;
                point.x = mul_div(original.x, matrix.values[0], 65536)?
                    + mul_div(original.y, matrix.values[2], 65536)?
                    + matrix.offset[0];
                point.y = mul_div(original.x, matrix.values[1], 65536)?
                    + mul_div(original.y, matrix.values[3], 65536)?
                    + matrix.offset[1];
                Ok(())
            };
            for segment in &mut segments {
                match segment {
                    Segment::Move(point) | Segment::Line(point) => transform(point)?,
                    Segment::Conic(a, end) => {
                        transform(a)?;
                        transform(end)?;
                    }
                    Segment::Cubic(a, b, end) => {
                        transform(a)?;
                        transform(b)?;
                        transform(end)?;
                    }
                }
            }
            let hmtx = self.font.table(b"hmtx")?;
            let advance = mul_div(
                i64::from(u16_at(hmtx, id.min(self.font.metrics - 1) * 4)?),
                matrix.values[0],
                65536,
            )? + matrix.offset[0];
            let advance = mul_div(advance, scale, 65536)?;
            return Ok(RenderedGlyph {
                bitmap: gray::rasterize(&segments)?,
                advance: (advance + 32) & !63,
            });
        }
        let mut glyph = if self.requires_autofit {
            let (script, nonbase) = *self.styles.get(id).ok_or_else(invalid)?;
            let metrics = match self.automatic.remove(&script) {
                Some(metrics) => metrics,
                None => super::autofit::Metrics::new(self, script)?,
            };
            let glyph = metrics.apply(self, id, nonbase);
            self.automatic.insert(script, metrics);
            glyph?
        } else {
            self.interpreter.begin_glyph();
            self.load(id, 0)?
        };
        // FreeType translates the loaded outline before decomposition.  Do not
        // move this after implied conic midpoints: signed /2 rounds toward zero.
        for point in &mut glyph.outline.current {
            point.x -= glyph.origin;
        }
        let mut segments = Vec::new();
        let mut first = 0;
        for &end in &glyph.outline.ends {
            let points = glyph.outline.current.get(first..=end).ok_or_else(invalid)?;
            let flags = glyph
                .outline
                .on_curve
                .get(first..=end)
                .ok_or_else(invalid)?;
            let count = points.len();
            // The reference raster rebases the outline into its nonnegative
            // bitmap box before decomposition. Floor averaging here commutes
            // with that integer-pixel translation, including negative points.
            let midpoint = |a: Position, b: Position| Position {
                x: (a.x + b.x) >> 1,
                y: (a.y + b.y) >> 1,
            };
            let (start, mut index) = if flags[0] {
                (points[0], 1)
            } else if flags[count - 1] {
                (points[count - 1], 0)
            } else {
                (midpoint(points[0], points[count - 1]), 0)
            };
            segments.push(Segment::Move(start));
            let finish = index + count;
            while index < finish {
                let position = index % count;
                if flags[position] {
                    segments.push(Segment::Line(points[position]));
                    index += 1;
                } else {
                    let next = (position + 1) % count;
                    let target = if flags[next] {
                        points[next]
                    } else {
                        midpoint(points[position], points[next])
                    };
                    segments.push(Segment::Conic(points[position], target));
                    index += if flags[next] { 2 } else { 1 };
                }
            }
            segments.push(Segment::Line(start));
            first = end + 1;
        }
        Ok(RenderedGlyph {
            bitmap: gray::rasterize(&segments)?,
            advance: glyph.advance,
        })
    }
}

// The profile uses the integer CORDIC length, including its shrink-factor
// correction, to scale composite translations. An ideal square root differs
// at rounding boundaries and consequently changes the hinted outline.
pub(super) fn vector_length(mut x: i64, mut y: i64) -> Result<i64> {
    if x == 0 {
        return Ok(y.abs());
    }
    if y == 0 {
        return Ok(x.abs());
    }
    let bits = u32::try_from(x.abs() | y.abs()).map_err(|_| invalid())?;
    let shift = 29 - (31 - bits.leading_zeros()) as i32;
    if shift >= 0 {
        x <<= shift;
        y <<= shift;
    } else {
        x >>= -shift;
        y >>= -shift;
    }
    if y > x {
        if y > -x {
            (x, y) = (y, -x);
        } else {
            x = -x;
            y = -y;
        }
    } else if y < -x {
        (x, y) = (-y, x);
    }
    for index in 1..23 {
        let bias = 1_i64 << (index - 1);
        (x, y) = if y > 0 {
            (x + ((y + bias) >> index), y - ((x + bias) >> index))
        } else {
            (x - ((y + bias) >> index), y + ((x + bias) >> index))
        };
    }
    let length = i64::try_from((i128::from(x) * 0xdbd95b16_i128 + 0x40000000) >> 32)
        .map_err(|_| invalid())?;
    Ok(if shift > 0 {
        (length + (1_i64 << (shift - 1))) >> shift
    } else {
        length << -shift
    })
}
