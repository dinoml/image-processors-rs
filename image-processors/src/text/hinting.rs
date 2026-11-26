//! TrueType program execution using signed 26.6 and 2.14 coordinates.

use std::collections::BTreeMap;

use super::{gray::Position, invalid, Result};

#[cfg(test)]
#[path = "hinting_tests.rs"]
mod tests;

pub(super) fn mul_div(a: i64, b: i64, c: i64) -> Result<i64> {
    if c == 0 {
        return Err(invalid());
    }
    let sign = (a < 0) ^ (b < 0) ^ (c < 0);
    let product = i128::from(a).abs() * i128::from(b).abs();
    let divisor = i128::from(c).abs();
    let value = i64::try_from((product + divisor / 2) / divisor).map_err(|_| invalid())?;
    Ok(if sign { -value } else { value })
}

fn project(point: Position, vector: Position) -> i64 {
    let product = point.x * vector.x + point.y * vector.y;
    (product + 8192 - i64::from(product < 0)) >> 14
}

fn difference(a: Position, b: Position) -> Position {
    Position {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

#[derive(Clone, Default)]
pub(super) struct Zone {
    pub current: Vec<Position>,
    pub original: Vec<Position>,
    pub unscaled: Vec<Position>,
    pub touched: Vec<[bool; 2]>,
    pub on_curve: Vec<bool>,
    pub ends: Vec<usize>,
}

impl Zone {
    fn point(&self, index: usize, original: bool) -> Result<Position> {
        (if original {
            &self.original
        } else {
            &self.current
        })
        .get(index)
        .copied()
        .ok_or_else(invalid)
    }
}

#[derive(Clone, Copy)]
enum Rounding {
    Grid,
    HalfGrid,
    DoubleGrid,
    Down,
    Up,
    Off,
    Super {
        period: i64,
        phase: i64,
        threshold: i64,
    },
}

impl Rounding {
    fn apply(self, value: i64) -> i64 {
        let distance = value.abs();
        let rounded = match self {
            Self::Grid => (distance + 32) & !63,
            Self::HalfGrid => (distance & !63) + 32,
            Self::DoubleGrid => (distance + 16) & !31,
            Self::Down => distance & !63,
            Self::Up => (distance + 63) & !63,
            Self::Off => distance,
            Self::Super {
                period,
                phase,
                threshold,
            } => {
                let adjusted = distance - phase + threshold;
                let result = adjusted - adjusted.rem_euclid(period) + phase;
                result.max(phase)
            }
        };
        if value < 0 {
            -rounded
        } else {
            rounded
        }
    }
}

#[derive(Clone)]
pub(super) struct Graphics {
    reference: [usize; 3],
    zones: [usize; 3],
    freedom: Position,
    projection: Position,
    dual: Position,
    loop_count: usize,
    minimum: i64,
    cut_in: i64,
    single_width: i64,
    single_cut_in: i64,
    delta_base: i64,
    delta_shift: u32,
    auto_flip: bool,
    rounding: Rounding,
    pub instruction_control: i64,
}

impl Default for Graphics {
    fn default() -> Self {
        Self {
            reference: [0; 3],
            zones: [1; 3],
            freedom: Position { x: 16384, y: 0 },
            projection: Position { x: 16384, y: 0 },
            dual: Position { x: 16384, y: 0 },
            loop_count: 1,
            minimum: 64,
            cut_in: 68,
            single_width: 0,
            single_cut_in: 0,
            delta_base: 9,
            delta_shift: 3,
            auto_flip: true,
            rounding: Rounding::Grid,
            instruction_control: 0,
        }
    }
}

#[derive(Clone)]
pub(super) struct Interpreter {
    pub zones: [Zone; 2],
    pub graphics: Graphics,
    pub cvt: Vec<i64>,
    pub storage: Vec<i64>,
    pub scale: i64,
    pub ppem: i64,
    functions: BTreeMap<i64, Vec<u8>>,
    definitions: BTreeMap<u8, Vec<u8>>,
    stack: Vec<i64>,
    backward: bool,
    composite: bool,
    interpolated: [bool; 2],
    instructions_left: usize,
    size_cvt: Vec<i64>,
    size_storage: Vec<i64>,
}

impl Interpreter {
    pub fn new(
        ppem: u32,
        units: u16,
        twilight_points: usize,
        storage: usize,
        cvt: &[i16],
    ) -> Result<Self> {
        let scale = mul_div(i64::from(ppem) * 64, 65536, i64::from(units))?;
        let points = vec![Position::default(); twilight_points.checked_add(4).ok_or_else(invalid)?];
        let twilight = Zone {
            current: points.clone(),
            original: points.clone(),
            unscaled: points,
            touched: vec![[false; 2]; twilight_points + 4],
            on_curve: vec![true; twilight_points + 4],
            ends: Vec::new(),
        };
        Ok(Self {
            zones: [twilight, Zone::default()],
            graphics: Graphics::default(),
            cvt: cvt
                .iter()
                .map(|value| mul_div(i64::from(*value), scale, 65536))
                .collect::<Result<_>>()?,
            storage: vec![0; storage],
            scale,
            ppem: i64::from(ppem),
            functions: BTreeMap::new(),
            definitions: BTreeMap::new(),
            stack: Vec::new(),
            backward: true,
            composite: false,
            interpolated: [false; 2],
            instructions_left: 1_000_000,
            size_cvt: Vec::new(),
            size_storage: Vec::new(),
        })
    }

    pub fn prepare(&mut self, font_program: &[u8], size_program: &[u8]) -> Result<()> {
        let cvt = self.cvt.clone();
        let scale = self.scale;
        let ppem = self.ppem;
        self.cvt.fill(0);
        self.scale = 0;
        self.ppem = 0;
        self.execute(font_program, 0)?;
        self.cvt = cvt;
        self.scale = scale;
        self.ppem = ppem;
        self.storage.fill(0);
        self.stack.clear();
        self.graphics = Graphics::default();
        self.execute(size_program, 0)?;
        self.stack.clear();
        // The size program does not retain vectors, reference points or zone
        // selections for subsequent glyph programs.
        self.graphics.reference = [0; 3];
        self.graphics.zones = [1; 3];
        self.graphics.freedom = Position { x: 16384, y: 0 };
        self.graphics.projection = self.graphics.freedom;
        self.graphics.dual = self.graphics.freedom;
        self.graphics.loop_count = 1;
        self.size_cvt = self.cvt.clone();
        self.size_storage = self.storage.clone();
        Ok(())
    }

    pub fn begin_glyph(&mut self) {
        self.cvt.clone_from(&self.size_cvt);
        self.storage.clone_from(&self.size_storage);
    }

    pub fn glyph(
        &mut self,
        zone: Zone,
        program: &[u8],
        size_graphics: &Graphics,
        composite: bool,
    ) -> Result<()> {
        self.zones[1] = zone;
        self.graphics = size_graphics.clone();
        self.stack.clear();
        self.interpolated = [false; 2];
        self.instructions_left = 1_000_000;
        self.backward = self.graphics.instruction_control & 4 == 0;
        self.composite = composite;
        if self.graphics.instruction_control & 1 == 0 {
            self.execute(program, 0)?;
        }
        Ok(())
    }

    fn pop(&mut self) -> Result<i64> {
        self.stack.pop().ok_or_else(invalid)
    }
    fn index(&mut self) -> Result<usize> {
        usize::try_from(self.pop()?).map_err(|_| invalid())
    }
    fn point(&self, zone: usize, index: usize, original: bool) -> Result<Position> {
        self.zones
            .get(zone)
            .ok_or_else(invalid)?
            .point(index, original)
    }
    fn relative(&self, za: usize, a: usize, zb: usize, b: usize, original: bool) -> Result<i64> {
        if original && za == 1 && zb == 1 {
            let delta = difference(
                *self.zones[1].unscaled.get(a).ok_or_else(invalid)?,
                *self.zones[1].unscaled.get(b).ok_or_else(invalid)?,
            );
            return mul_div(
                project(delta, self.graphics.dual),
                if self.composite { 65536 } else { self.scale },
                65536,
            );
        }
        Ok(project(
            difference(self.point(za, a, original)?, self.point(zb, b, original)?),
            if original {
                self.graphics.dual
            } else {
                self.graphics.projection
            },
        ))
    }

    fn move_point(&mut self, zone: usize, index: usize, distance: i64) -> Result<()> {
        self.shift_point(zone, index, distance, true)
    }

    fn shift_point(&mut self, zone: usize, index: usize, distance: i64, touch: bool) -> Result<()> {
        let freedom = self.graphics.freedom;
        let dot =
            (freedom.x * self.graphics.projection.x + freedom.y * self.graphics.projection.y) >> 14;
        let dot = if dot.abs() < 1024 { 16384 } else { dot };
        let dx = mul_div(distance, freedom.x, dot)?;
        let dy = mul_div(distance, freedom.y, dot)?;
        let point = self.zones[zone]
            .current
            .get_mut(index)
            .ok_or_else(invalid)?;
        if freedom.x != 0 && !self.backward {
            point.x += dx;
        }
        if freedom.y != 0 && !(self.backward && self.interpolated[0] && self.interpolated[1]) {
            point.y += dy;
        }
        if touch {
            let touched = self.zones[zone]
                .touched
                .get_mut(index)
                .ok_or_else(invalid)?;
            touched[0] |= freedom.x != 0;
            touched[1] |= freedom.y != 0;
        }
        Ok(())
    }

    fn normalize(x: i64, y: i64) -> Result<Position> {
        if x == 0 && y == 0 {
            return Ok(Position { x: 16384, y: 0 });
        }
        if x == 0 {
            return Ok(Position {
                x: 0,
                y: 16384 * y.signum(),
            });
        }
        if y == 0 {
            return Ok(Position {
                x: 16384 * x.signum(),
                y: 0,
            });
        }
        // Match the profile's reciprocal-length iteration, including its 32-bit
        // wrap boundaries before reducing a 16.16 unit vector to 2.14.
        let mut a = u32::try_from(x.abs()).map_err(|_| invalid())?;
        let mut b = u32::try_from(y.abs()).map_err(|_| invalid())?;
        let estimate = a.max(b).wrapping_add(a.min(b) >> 1);
        let leading = estimate.leading_zeros();
        let shift = leading as i32 - 15 - i32::from(estimate >= (0xaaaa_aaaa_u32 >> leading));
        let length = if shift > 0 {
            a = a.wrapping_shl(shift as u32);
            b = b.wrapping_shl(shift as u32);
            a.max(b).wrapping_add(a.min(b) >> 1)
        } else {
            a >>= -shift;
            b >>= -shift;
            estimate >> -shift
        };
        let mut correction = 65536_i32.wrapping_sub(length as i32);
        let (normalized_x, normalized_y) = loop {
            let nx = (a as i32).wrapping_add((a as i32).wrapping_mul(correction) >> 16) as u32;
            let ny = (b as i32).wrapping_add((b as i32).wrapping_mul(correction) >> 16) as u32;
            let residual =
                (nx.wrapping_mul(nx).wrapping_add(ny.wrapping_mul(ny)) as i32).wrapping_neg() / 512;
            let update = residual.wrapping_mul(65536_i32.wrapping_add(correction) >> 8) / 65536;
            correction = correction.wrapping_add(update);
            if update <= 0 {
                break (nx, ny);
            }
        };
        Ok(Position {
            x: (i64::from(normalized_x) * x.signum()) / 4,
            y: (i64::from(normalized_y) * y.signum()) / 4,
        })
    }

    fn interpolate(&mut self, axis: usize) -> Result<()> {
        if self.backward && self.interpolated[0] && self.interpolated[1] {
            return Ok(());
        }
        self.interpolated[axis] = true;
        let coordinate = |p: Position| if axis == 0 { p.x } else { p.y };
        let mut first = 0;
        for end in self.zones[1].ends.clone() {
            let touched = (first..=end)
                .filter(|index| self.zones[1].touched[*index][axis])
                .collect::<Vec<_>>();
            if touched.len() == 1 {
                let reference = touched[0];
                let delta = coordinate(self.point(1, reference, false)?)
                    - coordinate(self.point(1, reference, true)?);
                for point in first..=end {
                    if point != reference {
                        let current = self.zones[1].current.get_mut(point).ok_or_else(invalid)?;
                        if axis == 0 {
                            current.x += delta;
                        } else {
                            current.y += delta;
                        }
                    }
                }
            } else if touched.len() > 1 {
                for index in 0..touched.len() {
                    let a = touched[index];
                    let b = touched[(index + 1) % touched.len()];
                    let mut point = if a == end { first } else { a + 1 };
                    while point != b {
                        let mut before = coordinate(self.point(1, a, true)?);
                        let mut after = coordinate(self.point(1, b, true)?);
                        let mut moved_before = coordinate(self.point(1, a, false)?);
                        let mut moved_after = coordinate(self.point(1, b, false)?);
                        let mut design_before =
                            coordinate(*self.zones[1].unscaled.get(a).ok_or_else(invalid)?);
                        let mut design_after =
                            coordinate(*self.zones[1].unscaled.get(b).ok_or_else(invalid)?);
                        if design_before > design_after {
                            std::mem::swap(&mut before, &mut after);
                            std::mem::swap(&mut moved_before, &mut moved_after);
                            std::mem::swap(&mut design_before, &mut design_after);
                        }
                        let original = coordinate(self.point(1, point, true)?);
                        let moved = if original <= before {
                            original + moved_before - before
                        } else if original >= after {
                            original + moved_after - after
                        } else if moved_before == moved_after || design_before == design_after {
                            moved_before
                        } else {
                            let scale = mul_div(
                                moved_after - moved_before,
                                65536,
                                design_after - design_before,
                            )?;
                            moved_before
                                + mul_div(
                                    coordinate(
                                        *self.zones[1].unscaled.get(point).ok_or_else(invalid)?,
                                    ) - design_before,
                                    scale,
                                    65536,
                                )?
                        };
                        let target = self.zones[1].current.get_mut(point).ok_or_else(invalid)?;
                        if axis == 0 {
                            target.x = moved;
                        } else {
                            target.y = moved;
                        }
                        point = if point == end { first } else { point + 1 };
                    }
                }
            }
            first = end + 1;
        }
        Ok(())
    }

    fn skip_instruction(program: &[u8], pc: &mut usize) -> Result<u8> {
        let opcode = *program.get(*pc).ok_or_else(invalid)?;
        *pc += 1;
        let bytes = match opcode {
            0x40 | 0x41 => {
                let count = usize::from(*program.get(*pc).ok_or_else(invalid)?);
                *pc += 1;
                count * if opcode == 0x41 { 2 } else { 1 }
            }
            0xb0..=0xb7 => usize::from(opcode - 0xaf),
            0xb8..=0xbf => usize::from(opcode - 0xb7) * 2,
            _ => 0,
        };
        *pc = pc
            .checked_add(bytes)
            .filter(|end| *end <= program.len())
            .ok_or_else(invalid)?;
        Ok(opcode)
    }

    fn execute(&mut self, program: &[u8], depth: usize) -> Result<()> {
        if depth > 64 {
            return Err(invalid());
        }
        let mut pc = 0;
        while pc < program.len() {
            self.instructions_left = self.instructions_left.checked_sub(1).ok_or_else(invalid)?;
            let opcode = program[pc];
            pc += 1;
            match opcode {
                0x00..=0x05 => {
                    let vector = if opcode & 1 == 0 {
                        Position { x: 0, y: 16384 }
                    } else {
                        Position { x: 16384, y: 0 }
                    };
                    if opcode <= 3 {
                        self.graphics.projection = vector;
                        self.graphics.dual = vector;
                    }
                    if opcode <= 1 || opcode >= 4 {
                        self.graphics.freedom = vector;
                    }
                }
                0x06..=0x09 | 0x86..=0x87 => {
                    let b = self.index()?;
                    let a = self.index()?;
                    let za = self.graphics.zones[1];
                    let zb = self.graphics.zones[2];
                    let mut perpendicular = opcode & 1 != 0;
                    // SDPVTL computes the dual vector first.  A coincident
                    // original pair disables rotation for the current pair too.
                    if opcode >= 0x86 {
                        let mut vector =
                            difference(self.point(za, a, true)?, self.point(zb, b, true)?);
                        if vector.x == 0 && vector.y == 0 {
                            vector.x = 16384;
                            perpendicular = false;
                        }
                        if perpendicular {
                            vector = Position {
                                x: -vector.y,
                                y: vector.x,
                            };
                        }
                        self.graphics.dual = Self::normalize(vector.x, vector.y)?;
                    }
                    let mut vector =
                        difference(self.point(za, a, false)?, self.point(zb, b, false)?);
                    if vector.x == 0 && vector.y == 0 {
                        vector.x = 16384;
                        perpendicular = false;
                    }
                    if perpendicular {
                        vector = Position {
                            x: -vector.y,
                            y: vector.x,
                        };
                    }
                    let vector = Self::normalize(vector.x, vector.y)?;
                    if opcode <= 7 || opcode >= 0x86 {
                        self.graphics.projection = vector;
                        if opcode < 0x86 {
                            self.graphics.dual = vector;
                        }
                    } else {
                        self.graphics.freedom = vector;
                    }
                }
                0x0a | 0x0b => {
                    let y = i64::from(self.pop()? as i16);
                    let x = i64::from(self.pop()? as i16);
                    let vector = if x == 0 && y == 0 {
                        if opcode == 0x0a {
                            self.graphics.projection
                        } else {
                            self.graphics.freedom
                        }
                    } else {
                        Self::normalize(x, y)?
                    };
                    if opcode == 0x0a {
                        self.graphics.projection = vector;
                        self.graphics.dual = vector;
                    } else {
                        self.graphics.freedom = vector;
                    }
                }
                0x0c | 0x0d => {
                    let vector = if opcode == 0x0c {
                        self.graphics.projection
                    } else {
                        self.graphics.freedom
                    };
                    self.stack.extend([vector.x, vector.y]);
                }
                0x0e => self.graphics.freedom = self.graphics.projection,
                0x0f => {
                    let b1 = self.index()?;
                    let b0 = self.index()?;
                    let a1 = self.index()?;
                    let a0 = self.index()?;
                    let point = self.index()?;
                    let a = self.point(self.graphics.zones[1], a0, false)?;
                    let aa = self.point(self.graphics.zones[1], a1, false)?;
                    let b = self.point(self.graphics.zones[0], b0, false)?;
                    let bb = self.point(self.graphics.zones[0], b1, false)?;
                    let da = difference(aa, a);
                    let db = difference(bb, b);
                    let offset = difference(b, a);
                    let cross = mul_div(da.x, -db.y, 64)? + mul_div(da.y, db.x, 64)?;
                    let dot = mul_div(da.x, db.x, 64)? + mul_div(da.y, db.y, 64)?;
                    let result = if 19 * cross.abs() > dot.abs() {
                        let distance = mul_div(offset.x, -db.y, 64)? + mul_div(offset.y, db.x, 64)?;
                        Position {
                            x: a.x + mul_div(distance, da.x, cross)?,
                            y: a.y + mul_div(distance, da.y, cross)?,
                        }
                    } else {
                        Position {
                            x: (a.x + aa.x + b.x + bb.x) / 4,
                            y: (a.y + aa.y + b.y + bb.y) / 4,
                        }
                    };
                    *self.zones[self.graphics.zones[2]]
                        .current
                        .get_mut(point)
                        .ok_or_else(invalid)? = result;
                    *self.zones[self.graphics.zones[2]]
                        .touched
                        .get_mut(point)
                        .ok_or_else(invalid)? = [true; 2];
                }
                0x10..=0x12 => {
                    self.graphics.reference[usize::from(opcode - 0x10)] = self.index()?
                }
                0x13..=0x16 => {
                    let zone = self.index()?;
                    if zone > 1 {
                        return Err(invalid());
                    }
                    if opcode == 0x16 {
                        self.graphics.zones = [zone; 3];
                    } else {
                        self.graphics.zones[usize::from(opcode - 0x13)] = zone;
                    }
                }
                0x17 => self.graphics.loop_count = self.index()?,
                0x18 => self.graphics.rounding = Rounding::Grid,
                0x19 => self.graphics.rounding = Rounding::HalfGrid,
                0x1a => self.graphics.minimum = self.pop()?,
                0x1b => {
                    let mut nesting = 0;
                    while pc < program.len() {
                        match Self::skip_instruction(program, &mut pc)? {
                            0x58 => nesting += 1,
                            0x59 if nesting == 0 => break,
                            0x59 => nesting -= 1,
                            _ => {}
                        }
                    }
                }
                0x1c => {
                    let offset = self.pop()?;
                    pc = usize::try_from(pc as i64 + offset - 1).map_err(|_| invalid())?;
                }
                0x1d => self.graphics.cut_in = self.pop()?,
                0x1e => self.graphics.single_cut_in = self.pop()?,
                0x1f => {
                    let value = self.pop()?;
                    self.graphics.single_width = mul_div(value, self.scale, 65536)?;
                }
                0x20 => self.stack.push(*self.stack.last().ok_or_else(invalid)?),
                0x21 => {
                    self.pop()?;
                }
                0x22 => self.stack.clear(),
                0x23 => {
                    let n = self.stack.len();
                    if n < 2 {
                        return Err(invalid());
                    }
                    self.stack.swap(n - 1, n - 2);
                }
                0x24 => self.stack.push(self.stack.len() as i64),
                0x25 | 0x26 => {
                    let index = self.index()?;
                    let index = self
                        .stack
                        .len()
                        .checked_sub(index)
                        .filter(|_| index != 0)
                        .ok_or_else(invalid)?;
                    let value = if opcode == 0x26 {
                        self.stack.remove(index)
                    } else {
                        self.stack[index]
                    };
                    self.stack.push(value);
                }
                0x27 => {
                    let b = self.index()?;
                    let a = self.index()?;
                    let distance =
                        self.relative(self.graphics.zones[1], b, self.graphics.zones[0], a, false)?
                            / 2;
                    self.move_point(self.graphics.zones[0], a, distance)?;
                    self.move_point(self.graphics.zones[1], b, -distance)?;
                }
                0x29 => {
                    let point = self.index()?;
                    let zone = self.graphics.zones[0];
                    let touched = self.zones[zone]
                        .touched
                        .get_mut(point)
                        .ok_or_else(invalid)?;
                    if self.graphics.freedom.x != 0 {
                        touched[0] = false;
                    }
                    if self.graphics.freedom.y != 0 {
                        touched[1] = false;
                    }
                }
                0x2a | 0x2b => {
                    let function = self.pop()?;
                    let count = if opcode == 0x2a { self.index()? } else { 1 };
                    let body = self.functions.get(&function).cloned().ok_or_else(invalid)?;
                    for _ in 0..count {
                        self.execute(&body, depth + 1)?;
                    }
                }
                0x2c | 0x89 => {
                    let name = self.pop()?;
                    let start = pc;
                    while pc < program.len() {
                        let before = pc;
                        if Self::skip_instruction(program, &mut pc)? == 0x2d {
                            let body = program[start..before].to_vec();
                            if opcode == 0x2c {
                                self.functions.insert(name, body);
                            } else {
                                self.definitions
                                    .insert(u8::try_from(name).map_err(|_| invalid())?, body);
                            }
                            break;
                        }
                    }
                }
                0x2d => return Ok(()),
                0x2e | 0x2f => {
                    let point = self.index()?;
                    let zone = self.graphics.zones[0];
                    let distance =
                        project(self.point(zone, point, false)?, self.graphics.projection);
                    let target = if opcode & 1 != 0 {
                        self.graphics.rounding.apply(distance)
                    } else {
                        distance
                    };
                    self.move_point(zone, point, target - distance)?;
                    self.graphics.reference[0] = point;
                    self.graphics.reference[1] = point;
                }
                0x30 | 0x31 => self.interpolate(if opcode & 1 != 0 { 0 } else { 1 })?,
                0x32..=0x37 => {
                    let reference_slot = if opcode & 1 != 0 { 1 } else { 2 };
                    let reference_zone =
                        self.graphics.zones[if reference_slot == 1 { 0 } else { 1 }];
                    let reference = self.graphics.reference[reference_slot];
                    let shift = project(
                        difference(
                            self.point(reference_zone, reference, false)?,
                            self.point(reference_zone, reference, true)?,
                        ),
                        self.graphics.projection,
                    );
                    if opcode <= 0x33 {
                        for _ in 0..self.graphics.loop_count {
                            let point = self.index()?;
                            self.move_point(self.graphics.zones[2], point, shift)?;
                        }
                        self.graphics.loop_count = 1;
                    } else {
                        let selection = self.index()?;
                        let zone = self.graphics.zones[2];
                        if opcode >= 0x36 && selection > 1 {
                            return Err(invalid());
                        }
                        let (first, end) = if opcode <= 0x35 {
                            if zone == 0 {
                                if selection != 0 {
                                    return Err(invalid());
                                }
                                (0, self.zones[0].current.len())
                            } else {
                                let end =
                                    *self.zones[zone].ends.get(selection).ok_or_else(invalid)?;
                                (
                                    if selection == 0 {
                                        0
                                    } else {
                                        self.zones[zone].ends[selection - 1] + 1
                                    },
                                    end + 1,
                                )
                            }
                        } else {
                            (
                                0,
                                if zone == 0 {
                                    self.zones[0].current.len()
                                } else {
                                    self.zones[1].ends.last().map_or(0, |end| end + 1)
                                },
                            )
                        };
                        for point in first..end {
                            if zone != reference_zone || point != reference {
                                self.shift_point(zone, point, shift, opcode <= 0x35)?;
                            }
                        }
                    }
                }
                0x38 => {
                    let distance = self.pop()?;
                    for _ in 0..self.graphics.loop_count {
                        let point = self.index()?;
                        let zone = self.graphics.zones[2];
                        if !self.backward
                            || zone == 0
                            || (!(self.interpolated[0] && self.interpolated[1])
                                && ((self.composite && self.graphics.freedom.y != 0)
                                    || self.zones[zone].touched.get(point).ok_or_else(invalid)?[1]))
                        {
                            self.move_point(zone, point, distance)?;
                        }
                    }
                    self.graphics.loop_count = 1;
                }
                0x39 => {
                    let a = self.graphics.reference[1];
                    let b = self.graphics.reference[2];
                    let za = self.graphics.zones[0];
                    let zb = self.graphics.zones[1];
                    let zone = self.graphics.zones[2];
                    let twilight = za == 0 || zb == 0 || zone == 0;
                    let original_point =
                        |this: &Self, zone: usize, point: usize| -> Result<Position> {
                            if twilight {
                                this.point(zone, point, true)
                            } else {
                                this.zones[zone]
                                    .unscaled
                                    .get(point)
                                    .copied()
                                    .ok_or_else(invalid)
                            }
                        };
                    let original = project(
                        difference(original_point(self, zb, b)?, original_point(self, za, a)?),
                        self.graphics.dual,
                    );
                    let moved = self.relative(zb, b, za, a, false)?;
                    for _ in 0..self.graphics.loop_count {
                        let point = self.index()?;
                        let offset = project(
                            difference(
                                original_point(self, zone, point)?,
                                original_point(self, za, a)?,
                            ),
                            self.graphics.dual,
                        );
                        let target = if original == 0 {
                            offset
                        } else {
                            mul_div(offset, moved, original)?
                        };
                        let current = self.relative(zone, point, za, a, false)?;
                        self.move_point(zone, point, target - current)?;
                    }
                    self.graphics.loop_count = 1;
                }
                0x3a | 0x3b => {
                    let distance = self.pop()?;
                    let point = self.index()?;
                    let reference = self.graphics.reference[0];
                    if self.graphics.zones[1] == 0 {
                        let origin = self.point(self.graphics.zones[0], reference, true)?;
                        let freedom = self.graphics.freedom;
                        let dot = (freedom.x * self.graphics.projection.x
                            + freedom.y * self.graphics.projection.y)
                            >> 14;
                        let dot = if dot.abs() < 1024 { 16384 } else { dot };
                        let initial = Position {
                            x: origin.x + mul_div(distance, freedom.x, dot)?,
                            y: origin.y + mul_div(distance, freedom.y, dot)?,
                        };
                        *self.zones[0].original.get_mut(point).ok_or_else(invalid)? = initial;
                        self.zones[0].current[point] = initial;
                    }
                    let current = self.relative(
                        self.graphics.zones[1],
                        point,
                        self.graphics.zones[0],
                        reference,
                        false,
                    )?;
                    self.move_point(self.graphics.zones[1], point, distance - current)?;
                    self.graphics.reference[1] = reference;
                    self.graphics.reference[2] = point;
                    if opcode & 1 != 0 {
                        self.graphics.reference[0] = point;
                    }
                }
                0x3c => {
                    for _ in 0..self.graphics.loop_count {
                        let point = self.index()?;
                        let distance = self.relative(
                            self.graphics.zones[1],
                            point,
                            self.graphics.zones[0],
                            self.graphics.reference[0],
                            false,
                        )?;
                        self.move_point(self.graphics.zones[1], point, -distance)?;
                    }
                    self.graphics.loop_count = 1;
                }
                0x3d => self.graphics.rounding = Rounding::DoubleGrid,
                0x3e | 0x3f => {
                    let cvt = self.index()?;
                    let point = self.index()?;
                    let zone = self.graphics.zones[0];
                    let mut target = *self.cvt.get(cvt).ok_or_else(invalid)?;
                    if zone == 0 {
                        let freedom = self.graphics.freedom;
                        let original = Position {
                            x: mul_div(target, freedom.x, 16384)?,
                            y: mul_div(target, freedom.y, 16384)?,
                        };
                        *self.zones[0].original.get_mut(point).ok_or_else(invalid)? = original;
                        self.zones[0].current[point] = original;
                    }
                    let current =
                        project(self.point(zone, point, false)?, self.graphics.projection);
                    if opcode & 1 != 0 {
                        if (target - current).abs() > self.graphics.cut_in {
                            target = current;
                        }
                        target = self.graphics.rounding.apply(target);
                    }
                    self.move_point(zone, point, target - current)?;
                    self.graphics.reference[0] = point;
                    self.graphics.reference[1] = point;
                }
                0x40 | 0x41 | 0xb0..=0xbf => {
                    let words = opcode == 0x41 || opcode >= 0xb8;
                    let count = if opcode <= 0x41 {
                        let n = usize::from(*program.get(pc).ok_or_else(invalid)?);
                        pc += 1;
                        n
                    } else if words {
                        usize::from(opcode - 0xb7)
                    } else {
                        usize::from(opcode - 0xaf)
                    };
                    for _ in 0..count {
                        let value = if words {
                            let pair: [u8; 2] = program
                                .get(pc..pc + 2)
                                .ok_or_else(invalid)?
                                .try_into()
                                .map_err(|_| invalid())?;
                            pc += 2;
                            i64::from(i16::from_be_bytes(pair))
                        } else {
                            let value = i64::from(*program.get(pc).ok_or_else(invalid)?);
                            pc += 1;
                            value
                        };
                        self.stack.push(value);
                    }
                }
                0x42 => {
                    let value = self.pop()?;
                    let index = self.index()?;
                    *self.storage.get_mut(index).ok_or_else(invalid)? = value;
                }
                0x43 => {
                    let index = self.index()?;
                    self.stack
                        .push(*self.storage.get(index).ok_or_else(invalid)?);
                }
                0x44 | 0x70 => {
                    let value = self.pop()?;
                    let index = self.index()?;
                    let value = if opcode == 0x70 {
                        mul_div(value, self.scale, 65536)?
                    } else {
                        value
                    };
                    *self.cvt.get_mut(index).ok_or_else(invalid)? = value;
                }
                0x45 => {
                    let index = self.index()?;
                    self.stack.push(*self.cvt.get(index).ok_or_else(invalid)?);
                }
                0x46 | 0x47 => {
                    let point = self.index()?;
                    self.stack.push(project(
                        self.point(self.graphics.zones[2], point, opcode & 1 != 0)?,
                        if opcode & 1 != 0 {
                            self.graphics.dual
                        } else {
                            self.graphics.projection
                        },
                    ));
                }
                0x48 => {
                    let target = self.pop()?;
                    let point = self.index()?;
                    let zone = self.graphics.zones[2];
                    let current =
                        project(self.point(zone, point, false)?, self.graphics.projection);
                    self.move_point(zone, point, target - current)?;
                    if zone == 0 {
                        self.zones[0].original[point] = self.zones[0].current[point];
                    }
                }
                0x49 | 0x4a => {
                    let b = self.index()?;
                    let a = self.index()?;
                    let value = self.relative(
                        self.graphics.zones[0],
                        a,
                        self.graphics.zones[1],
                        b,
                        opcode == 0x4a,
                    )?;
                    self.stack.push(value);
                }
                0x4b => self.stack.push(self.ppem),
                0x4c => self.stack.push(self.ppem * 64),
                0x4d => self.graphics.auto_flip = true,
                0x4e => self.graphics.auto_flip = false,
                0x4f => {
                    self.pop()?;
                }
                0x50..=0x55 | 0x5a | 0x5b => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(i64::from(match opcode {
                        0x50 => a < b,
                        0x51 => a <= b,
                        0x52 => a > b,
                        0x53 => a >= b,
                        0x54 => a == b,
                        0x55 => a != b,
                        0x5a => a != 0 && b != 0,
                        _ => a != 0 || b != 0,
                    }));
                }
                0x56 | 0x57 => {
                    let value = self.pop()?;
                    let odd = self.graphics.rounding.apply(value) & 127 == 64;
                    self.stack
                        .push(i64::from(if opcode == 0x56 { odd } else { !odd }));
                }
                0x58 => {
                    if self.pop()? == 0 {
                        let mut nesting = 0;
                        while pc < program.len() {
                            match Self::skip_instruction(program, &mut pc)? {
                                0x58 => nesting += 1,
                                0x59 if nesting == 0 => break,
                                0x59 => nesting -= 1,
                                0x1b if nesting == 0 => break,
                                _ => {}
                            }
                        }
                    }
                }
                0x59 => {}
                0x5c => {
                    let value = self.pop()?;
                    self.stack.push(i64::from(value == 0));
                }
                0x5d | 0x71..=0x75 => {
                    let count = self.index()?;
                    let band = match opcode {
                        0x71 | 0x74 => 16,
                        0x72 | 0x75 => 32,
                        _ => 0,
                    };
                    for _ in 0..count {
                        let index = self.index()?;
                        let argument = self.pop()?;
                        let target = self.graphics.delta_base + band + (argument >> 4);
                        if target == self.ppem {
                            let step = (argument & 15) - 8;
                            let step = if step >= 0 { step + 1 } else { step };
                            let delta = (step * 64) >> self.graphics.delta_shift;
                            if opcode >= 0x73 {
                                *self.cvt.get_mut(index).ok_or_else(invalid)? += delta;
                            } else {
                                let zone = self.graphics.zones[0];
                                if index < self.zones[zone].current.len()
                                    && (!self.backward
                                        || (!(self.interpolated[0] && self.interpolated[1])
                                            && ((self.composite && self.graphics.freedom.y != 0)
                                                || self.zones[zone].touched[index][1])))
                                {
                                    self.move_point(zone, index, delta)?;
                                }
                            }
                        }
                    }
                }
                0x5e => self.graphics.delta_base = self.pop()?,
                0x5f => {
                    let shift = self.index()?;
                    if shift > 6 {
                        return Err(invalid());
                    }
                    self.graphics.delta_shift = shift as u32;
                }
                0x60..=0x63 | 0x8b | 0x8c => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    // TrueType DIV truncates toward zero (FT_MulDiv_No_Round),
                    // whereas MUL and the coordinate helpers round to nearest.
                    // Bytecode results retain the reference's signed32-bit
                    // FT_Long width even though native coordinates use i64.
                    let value = match opcode {
                        0x60 => (a as i32).wrapping_add(b as i32) as i64,
                        0x61 => (a as i32).wrapping_sub(b as i32) as i64,
                        0x62 => {
                            if b == 0 {
                                return Err(invalid());
                            }
                            (i128::from(a) * 64 / i128::from(b)) as i32 as i64
                        }
                        0x63 => mul_div(a, b, 64)? as i32 as i64,
                        0x8b => a.max(b),
                        _ => a.min(b),
                    };
                    self.stack.push(value);
                }
                0x64..=0x67 => {
                    let value = self.pop()?;
                    self.stack.push(match opcode {
                        0x64 => value.abs(),
                        0x65 => -value,
                        0x66 => value & !63,
                        _ => (value + 63) & !63,
                    });
                }
                0x68..=0x6b => {
                    let value = self.pop()?;
                    self.stack.push(self.graphics.rounding.apply(value));
                }
                0x6c..=0x6f => {}
                0x76 | 0x77 => {
                    let value = self.pop()?;
                    let grid = if opcode == 0x77 { 0x2d41 } else { 0x4000 };
                    let period = match (value >> 6) & 3 {
                        0 => grid / 2,
                        2 => grid * 2,
                        _ => grid,
                    };
                    let phase = period * ((value >> 4) & 3) / 4;
                    let threshold = if value & 15 == 0 {
                        period - 1
                    } else {
                        ((value & 15) - 4) * period / 8
                    };
                    self.graphics.rounding = Rounding::Super {
                        period: period >> 8,
                        phase: phase >> 8,
                        threshold: threshold >> 8,
                    };
                }
                0x78 | 0x79 => {
                    let condition = self.pop()?;
                    let offset = self.pop()?;
                    if (condition != 0) == (opcode == 0x78) {
                        pc = usize::try_from(pc as i64 + offset - 1).map_err(|_| invalid())?;
                    }
                }
                0x7a => self.graphics.rounding = Rounding::Off,
                0x7c => self.graphics.rounding = Rounding::Up,
                0x7d => self.graphics.rounding = Rounding::Down,
                0x7e | 0x7f => {
                    self.pop()?;
                }
                0x80 => {
                    for _ in 0..self.graphics.loop_count {
                        let point = self.index()?;
                        if !(self.backward && self.interpolated[0] && self.interpolated[1]) {
                            let flag = self.zones[1].on_curve.get_mut(point).ok_or_else(invalid)?;
                            *flag = !*flag;
                        }
                    }
                    self.graphics.loop_count = 1;
                }
                0x81 | 0x82 => {
                    let end = self.index()?;
                    let start = self.index()?;
                    if !(self.backward && self.interpolated[0] && self.interpolated[1]) {
                        for flag in self.zones[1]
                            .on_curve
                            .get_mut(start..=end)
                            .ok_or_else(invalid)?
                        {
                            *flag = opcode == 0x81;
                        }
                    }
                }
                0x85 | 0x8d => {
                    self.pop()?;
                }
                0x88 => {
                    let selector = self.pop()?;
                    let mut answer = if selector & 1 != 0 { 40 } else { 0 };
                    for (selection, result) in [(64, 13), (1024, 17), (2048, 18), (4096, 19)] {
                        if selector & selection != 0 {
                            answer |= 1 << result;
                        }
                    }
                    self.stack.push(answer);
                }
                0x8a => {
                    let n = self.stack.len();
                    if n < 3 {
                        return Err(invalid());
                    }
                    self.stack[n - 3..].rotate_left(1);
                }
                0x8e => {
                    let selector = self.pop()?;
                    let value = self.pop()?;
                    if (1..=3).contains(&selector) {
                        let flag = 1 << (selector - 1);
                        if value != 0 && value != flag {
                            return Err(invalid());
                        }
                        self.graphics.instruction_control =
                            (self.graphics.instruction_control & !flag) | value;
                        self.backward = self.graphics.instruction_control & 4 == 0;
                    }
                }
                0xc0..=0xff => {
                    let cvt = if opcode >= 0xe0 {
                        Some(self.pop()?)
                    } else {
                        None
                    };
                    let point = self.index()?;
                    let reference = self.graphics.reference[0];
                    let zone = self.graphics.zones[1];
                    let reference_zone = self.graphics.zones[0];
                    let mut indirect = if let Some(cvt) = cvt {
                        if cvt == -1 {
                            0
                        } else {
                            *self
                                .cvt
                                .get(usize::try_from(cvt).map_err(|_| invalid())?)
                                .ok_or_else(invalid)?
                        }
                    } else {
                        0
                    };
                    if cvt.is_some()
                        && (indirect - self.graphics.single_width).abs()
                            < self.graphics.single_cut_in
                    {
                        indirect = if indirect >= 0 {
                            self.graphics.single_width
                        } else {
                            -self.graphics.single_width
                        };
                    }
                    if cvt.is_some() && zone == 0 {
                        let origin = self.point(reference_zone, reference, true)?;
                        let freedom = self.graphics.freedom;
                        let initial = Position {
                            x: origin.x + mul_div(indirect, freedom.x, 16384)?,
                            y: origin.y + mul_div(indirect, freedom.y, 16384)?,
                        };
                        *self.zones[0].original.get_mut(point).ok_or_else(invalid)? = initial;
                        self.zones[0].current[point] = initial;
                    }
                    let original = if cvt.is_some() {
                        project(
                            difference(
                                self.point(zone, point, true)?,
                                self.point(reference_zone, reference, true)?,
                            ),
                            self.graphics.dual,
                        )
                    } else {
                        self.relative(zone, point, reference_zone, reference, true)?
                    };
                    let mut target = if cvt.is_some() { indirect } else { original };
                    if cvt.is_some() && self.graphics.auto_flip && (original ^ target) < 0 {
                        target = -target;
                    }
                    if cvt.is_none()
                        && self.graphics.single_cut_in > 0
                        && target > self.graphics.single_width - self.graphics.single_cut_in
                        && target < self.graphics.single_width + self.graphics.single_cut_in
                    {
                        target = if target >= 0 {
                            self.graphics.single_width
                        } else {
                            -self.graphics.single_width
                        };
                    }
                    if opcode & 4 != 0 {
                        if cvt.is_some()
                            && zone == reference_zone
                            && (target - original).abs() > self.graphics.cut_in
                        {
                            target = original;
                        }
                        target = self.graphics.rounding.apply(target);
                    }
                    if opcode & 8 != 0 {
                        target = if original >= 0 {
                            target.max(self.graphics.minimum)
                        } else {
                            target.min(-self.graphics.minimum)
                        };
                    }
                    let current = self.relative(zone, point, reference_zone, reference, false)?;
                    self.move_point(zone, point, target - current)?;
                    self.graphics.reference[1] = reference;
                    self.graphics.reference[2] = point;
                    if opcode & 16 != 0 {
                        self.graphics.reference[0] = point;
                    }
                }
                _ => {
                    let body = self.definitions.get(&opcode).cloned().ok_or_else(invalid)?;
                    self.execute(&body, depth + 1)?;
                }
            }
            if self.stack.len() > 65536 || pc > program.len() {
                return Err(invalid());
            }
        }
        Ok(())
    }
}
