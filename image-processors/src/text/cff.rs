//! CFF Type 2 programs, subroutines and explicit stem-mask changes.

use std::collections::BTreeMap;

use super::{hinting::mul_div, invalid, Result};

const ONE: i64 = 65536;

#[derive(Clone, Copy, Default)]
pub(super) struct Point {
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy)]
pub(super) enum Command {
    Move(Point),
    Line(Point),
    Curve(Point, Point, Point),
    Close,
}

#[derive(Clone, Copy)]
pub(super) struct Stem {
    pub vertical: bool,
    pub start: i64,
    pub width: i64,
}

pub(super) struct HintMask {
    pub command: usize,
    pub counter: bool,
    pub bits: Vec<u8>,
}

pub(super) struct Glyph {
    pub commands: Vec<Command>,
    pub stems: Vec<Stem>,
    pub masks: Vec<HintMask>,
    pub advance: i64,
    pub composite: Option<(Point, u8, u8)>,
}

pub(super) struct Index<'a> {
    objects: Vec<&'a [u8]>,
}

impl<'a> Index<'a> {
    fn parse(data: &'a [u8], cursor: &mut usize) -> Result<Self> {
        let count = usize::from(u16::from_be_bytes(
            data.get(*cursor..*cursor + 2)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?,
        ));
        *cursor += 2;
        if count == 0 {
            return Ok(Self {
                objects: Vec::new(),
            });
        }
        let bytes = usize::from(*data.get(*cursor).ok_or_else(invalid)?);
        *cursor += 1;
        if !(1..=4).contains(&bytes) {
            return Err(invalid());
        }
        let mut offsets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            let mut offset = 0_usize;
            for byte in data.get(*cursor..*cursor + bytes).ok_or_else(invalid)? {
                offset = offset
                    .checked_mul(256)
                    .and_then(|v| v.checked_add(usize::from(*byte)))
                    .ok_or_else(invalid)?;
            }
            *cursor += bytes;
            offsets.push(offset.checked_sub(1).ok_or_else(invalid)?);
        }
        if offsets[0] != 0 || offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(invalid());
        }
        let mut objects = Vec::with_capacity(count);
        for pair in offsets.windows(2) {
            objects.push(
                data.get(*cursor + pair[0]..*cursor + pair[1])
                    .ok_or_else(invalid)?,
            );
        }
        *cursor = cursor
            .checked_add(offsets[count])
            .filter(|end| *end <= data.len())
            .ok_or_else(invalid)?;
        Ok(Self { objects })
    }

    fn subroutine(&self, number: i64) -> Result<&'a [u8]> {
        let bias = if self.objects.len() < 1240 {
            107
        } else if self.objects.len() < 33900 {
            1131
        } else {
            32768
        };
        let index = usize::try_from(number + bias).map_err(|_| invalid())?;
        self.objects.get(index).copied().ok_or_else(invalid)
    }
}

fn real(data: &[u8], cursor: &mut usize) -> Result<i64> {
    Ok(decimal(data, cursor, 0, false)?.0)
}

fn decimal(data: &[u8], cursor: &mut usize, power: i64, dynamic: bool) -> Result<(i64, i64)> {
    let mut nibbles = Vec::new();
    loop {
        let byte = *data.get(*cursor).ok_or_else(invalid)?;
        *cursor += 1;
        nibbles.push(byte >> 4);
        if byte >> 4 == 15 {
            break;
        }
        nibbles.push(byte & 15);
        if byte & 15 == 15 {
            break;
        }
    }
    let mut index = 0;
    let mut negative = false;
    let mut number = 0_i64;
    let mut integer = 0_i64;
    let mut fraction = 0_i64;
    let mut extra = 0_i64;
    while let Some(&digit) = nibbles.get(index) {
        if digit == 14 {
            negative = true;
        } else if digit > 9 {
            break;
        } else if number >= 0x0ccccccc {
            extra += 1;
        } else if digit != 0 || number != 0 {
            integer += 1;
            number = number * 10 + i64::from(digit);
        }
        index += 1;
    }
    if nibbles.get(index) == Some(&10) {
        index += 1;
        while let Some(&digit) = nibbles.get(index) {
            if digit > 9 {
                break;
            }
            if digit == 0 && number == 0 {
                extra -= 1;
            } else if number < 0x0ccccccc && fraction < 9 {
                fraction += 1;
                number = number * 10 + i64::from(digit);
            }
            index += 1;
        }
    }
    let mut exponent = 0_i64;
    let mut overflow = false;
    if matches!(nibbles.get(index), Some(11 | 12)) {
        let exponent_negative = nibbles[index] == 12;
        index += 1;
        while let Some(&digit) = nibbles.get(index) {
            if digit > 9 {
                break;
            }
            if exponent > 1000 {
                overflow = true;
            } else {
                exponent = exponent * 10 + i64::from(digit);
            }
            index += 1;
        }
        if exponent_negative {
            exponent = -exponent;
        }
    }
    let sign = if negative { -1 } else { 1 };
    if number == 0 {
        return Ok((0, 0));
    }
    if overflow {
        return Ok((
            if exponent < 0 {
                0
            } else {
                sign * i64::from(i32::MAX)
            },
            0,
        ));
    }
    exponent += power + extra;
    let tens = |power: i64| -> Result<i64> {
        10_i64
            .checked_pow(u32::try_from(power).map_err(|_| invalid())?)
            .ok_or_else(invalid)
    };
    let (result, scaling) = if dynamic {
        fraction += integer;
        exponent += integer;
        if fraction <= 5 {
            if number > 32767 {
                (mul_div(number, ONE, 10)?, exponent - fraction + 1)
            } else {
                if exponent > 0 {
                    let digits = exponent.min(5);
                    let shift = digits - fraction;
                    if shift > 0 {
                        exponent -= digits;
                        number *= tens(shift)?;
                        if number > 32767 {
                            number /= 10;
                            exponent += 1;
                        }
                    } else {
                        exponent -= fraction;
                    }
                } else {
                    exponent -= fraction;
                }
                (number * ONE, exponent)
            }
        } else if number / tens(fraction - 5)? > 32767 {
            (mul_div(number, ONE, tens(fraction - 4)?)?, exponent - 4)
        } else {
            (mul_div(number, ONE, tens(fraction - 5)?)?, exponent - 5)
        }
    } else {
        integer += exponent;
        fraction -= exponent;
        if integer > 5 {
            return Ok((sign * i64::from(i32::MAX), 0));
        }
        if integer < -5 {
            return Ok((0, 0));
        }
        if integer < 0 {
            number /= tens(-integer)?;
            fraction += integer;
        }
        if fraction == 10 {
            number /= 10;
            fraction -= 1;
        }
        if fraction > 0 {
            if number / tens(fraction)? > 32767 {
                return Ok((0, 0));
            }
            (mul_div(number, ONE, tens(fraction)?)?, 0)
        } else {
            number *= tens(-fraction)?;
            (
                if number > 32767 {
                    i64::from(i32::MAX)
                } else {
                    number * ONE
                },
                0,
            )
        }
    };
    Ok((sign * result, scaling))
}

#[derive(Clone, Copy)]
pub(super) struct Matrix {
    pub values: [i64; 4],
    pub offset: [i64; 2],
    pub units: i64,
}

impl Matrix {
    fn identity(units: i64) -> Self {
        Self {
            values: [ONE, 0, 0, ONE],
            offset: [0; 2],
            units,
        }
    }
    fn normalize(mut self) -> Result<Self> {
        let divisor = if self.values[3] != 0 {
            self.values[3].abs()
        } else {
            self.values[1].abs()
        };
        if divisor == 0 {
            return Ok(Self::identity(1));
        }
        if divisor != ONE {
            self.units = mul_div(self.units, ONE, divisor)?;
            for value in &mut self.values {
                *value = mul_div(*value, ONE, divisor)?;
            }
            for value in &mut self.offset {
                *value = mul_div(*value, ONE, divisor)?;
            }
        }
        for offset in &mut self.offset {
            *offset >>= 16;
        }
        Ok(self)
    }
}

fn font_matrix(data: &[u8]) -> Result<Option<Matrix>> {
    let mut cursor = 0;
    let mut starts = Vec::new();
    while cursor < data.len() {
        let start = cursor;
        if number(data, &mut cursor, true)?.is_some() {
            starts.push(start);
            continue;
        }
        let first = data[cursor];
        cursor += 1;
        let operator = if first == 12 {
            let second = *data.get(cursor).ok_or_else(invalid)?;
            cursor += 1;
            0x0c00 | u16::from(second)
        } else {
            u16::from(first)
        };
        if operator == 0x0c07 {
            if starts.len() != 6 {
                return Err(invalid());
            }
            let mut values = [0; 6];
            let mut scalings = [0; 6];
            for (index, &start) in starts.iter().enumerate() {
                let mut cursor = start;
                if data[start] == 30 {
                    cursor += 1;
                    (values[index], scalings[index]) = decimal(data, &mut cursor, 0, true)?;
                } else {
                    let integer = number(data, &mut cursor, true)?.ok_or_else(invalid)? / ONE;
                    if integer > 32767 {
                        let digits = integer.ilog10() as i64 + 1;
                        let power = if integer / 10_i64.pow((digits - 5) as u32) > 32767 {
                            digits - 4
                        } else {
                            digits - 5
                        };
                        values[index] = mul_div(integer, ONE, 10_i64.pow(power as u32))?;
                        scalings[index] = power;
                    } else {
                        values[index] = integer * ONE;
                    }
                }
            }
            let maximum = values
                .iter()
                .zip(scalings)
                .filter_map(|(&value, scale)| (value != 0).then_some(scale))
                .max()
                .unwrap_or(1);
            let minimum = values
                .iter()
                .zip(scalings)
                .filter_map(|(&value, scale)| (value != 0).then_some(scale))
                .min()
                .unwrap_or(1);
            if !(-9..=0).contains(&maximum) || !(0..=9).contains(&(maximum - minimum)) {
                return Ok(Some(Matrix::identity(1)));
            }
            for (value, scale) in values.iter_mut().zip(scalings) {
                let divisor = 10_i64.pow((maximum - scale).max(0) as u32);
                *value = if *value < 0 {
                    (*value - divisor / 2) / divisor
                } else {
                    (*value + divisor / 2) / divisor
                };
            }
            let magnitude = values[..4]
                .iter()
                .fold(0_i64, |magnitude, value| magnitude | value.abs());
            if magnitude == 0 || magnitude > i64::from(i32::MAX) {
                return Ok(Some(Matrix::identity(1)));
            }
            let shift = magnitude.ilog2().saturating_sub(12);
            let [xx, yx, xy, yy] = [
                values[0] >> shift,
                values[1] >> shift,
                values[2] >> shift,
                values[3] >> shift,
            ];
            if 32 * (xx * yy - xy * yx).abs() <= xx * xx + xy * xy + yx * yx + yy * yy {
                return Ok(Some(Matrix::identity(1)));
            }
            return Ok(Some(Matrix {
                values: values[..4].try_into().map_err(|_| invalid())?,
                offset: [values[4], values[5]],
                units: 10_i64.pow((-maximum) as u32),
            }));
        }
        starts.clear();
    }
    Ok(None)
}

fn number(data: &[u8], cursor: &mut usize, dictionary: bool) -> Result<Option<i64>> {
    let byte = *data.get(*cursor).ok_or_else(invalid)?;
    let value = match byte {
        28 => {
            *cursor += 1;
            let value = i16::from_be_bytes(
                data.get(*cursor..*cursor + 2)
                    .ok_or_else(invalid)?
                    .try_into()
                    .map_err(|_| invalid())?,
            );
            *cursor += 2;
            i64::from(value) * ONE
        }
        29 if dictionary => {
            *cursor += 1;
            let value = i32::from_be_bytes(
                data.get(*cursor..*cursor + 4)
                    .ok_or_else(invalid)?
                    .try_into()
                    .map_err(|_| invalid())?,
            );
            *cursor += 4;
            i64::from(value) * ONE
        }
        30 if dictionary => {
            *cursor += 1;
            real(data, cursor)?
        }
        32..=246 => {
            *cursor += 1;
            (i64::from(byte) - 139) * ONE
        }
        247..=250 => {
            *cursor += 1;
            let tail = i64::from(*data.get(*cursor).ok_or_else(invalid)?);
            *cursor += 1;
            ((i64::from(byte) - 247) * 256 + tail + 108) * ONE
        }
        251..=254 => {
            *cursor += 1;
            let tail = i64::from(*data.get(*cursor).ok_or_else(invalid)?);
            *cursor += 1;
            (-(i64::from(byte) - 251) * 256 - tail - 108) * ONE
        }
        255 if !dictionary => {
            *cursor += 1;
            let value = i32::from_be_bytes(
                data.get(*cursor..*cursor + 4)
                    .ok_or_else(invalid)?
                    .try_into()
                    .map_err(|_| invalid())?,
            );
            *cursor += 4;
            i64::from(value)
        }
        _ => return Ok(None),
    };
    Ok(Some(value))
}

fn dictionary(data: &[u8]) -> Result<BTreeMap<u16, Vec<i64>>> {
    let mut cursor = 0;
    let mut operands = Vec::new();
    let mut entries = BTreeMap::new();
    while cursor < data.len() {
        if let Some(value) = number(data, &mut cursor, true)? {
            operands.push(value);
            continue;
        }
        let operator = data[cursor];
        cursor += 1;
        let operator = if operator == 12 {
            let second = *data.get(cursor).ok_or_else(invalid)?;
            cursor += 1;
            0x0c00 | u16::from(second)
        } else {
            u16::from(operator)
        };
        entries.insert(operator, std::mem::take(&mut operands));
    }
    if !operands.is_empty() {
        return Err(invalid());
    }
    Ok(entries)
}

pub(super) struct Private<'a> {
    pub values: BTreeMap<u16, Vec<i64>>,
    local: Index<'a>,
    default_width: i64,
    nominal_width: i64,
}

impl<'a> Private<'a> {
    fn parse(data: &'a [u8], entry: Option<&Vec<i64>>) -> Result<Self> {
        let (length, offset) = if let Some(values) = entry {
            if values.len() != 2 {
                return Err(invalid());
            }
            (
                usize::try_from(values[0] / ONE).map_err(|_| invalid())?,
                usize::try_from(values[1] / ONE).map_err(|_| invalid())?,
            )
        } else {
            (0, 0)
        };
        let values = dictionary(
            data.get(offset..offset.checked_add(length).ok_or_else(invalid)?)
                .ok_or_else(invalid)?,
        )?;
        let local = if let Some(relative) = values.get(&19).and_then(|v| v.first()) {
            let mut cursor = offset
                .checked_add(usize::try_from(relative / ONE).map_err(|_| invalid())?)
                .ok_or_else(invalid)?;
            Index::parse(data, &mut cursor)?
        } else {
            Index {
                objects: Vec::new(),
            }
        };
        let default_width = values
            .get(&20)
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0);
        let nominal_width = values
            .get(&21)
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0);
        Ok(Self {
            values,
            local,
            default_width,
            nominal_width,
        })
    }
}

pub(super) struct Font<'a> {
    pub top: BTreeMap<u16, Vec<i64>>,
    global: Index<'a>,
    charstrings: Index<'a>,
    private: Vec<Private<'a>>,
    selection: Vec<usize>,
    matrix: Option<Matrix>,
    matrices: Vec<Option<Matrix>>,
    charset: Vec<u16>,
}

impl<'a> Font<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.first() != Some(&1) {
            return Err(invalid());
        }
        let mut cursor = usize::from(*data.get(2).ok_or_else(invalid)?);
        let names = Index::parse(data, &mut cursor)?;
        let dictionaries = Index::parse(data, &mut cursor)?;
        if names.objects.len() != 1 || dictionaries.objects.len() != 1 {
            return Err(invalid());
        }
        let top = dictionary(dictionaries.objects[0])?;
        let matrix = font_matrix(dictionaries.objects[0])?;
        Index::parse(data, &mut cursor)?;
        let global = Index::parse(data, &mut cursor)?;
        let mut cursor =
            usize::try_from(*top.get(&17).and_then(|v| v.first()).ok_or_else(invalid)? / ONE)
                .map_err(|_| invalid())?;
        let charstrings = Index::parse(data, &mut cursor)?;
        let mut private = Vec::new();
        let mut matrices = Vec::new();
        let mut selection = vec![0; charstrings.objects.len()];
        if let Some(offset) = top.get(&0x0c24).and_then(|v| v.first()) {
            let mut cursor = usize::try_from(offset / ONE).map_err(|_| invalid())?;
            let dictionaries = Index::parse(data, &mut cursor)?;
            for entry in dictionaries.objects {
                let values = dictionary(entry)?;
                private.push(Private::parse(data, values.get(&18))?);
                matrices.push(font_matrix(entry)?);
            }
            let mut cursor = usize::try_from(
                *top.get(&0x0c25)
                    .and_then(|v| v.first())
                    .ok_or_else(invalid)?
                    / ONE,
            )
            .map_err(|_| invalid())?;
            let format = *data.get(cursor).ok_or_else(invalid)?;
            cursor += 1;
            match format {
                0 => {
                    for slot in &mut selection {
                        *slot = usize::from(*data.get(cursor).ok_or_else(invalid)?);
                        cursor += 1;
                    }
                }
                3 => {
                    let ranges = usize::from(u16::from_be_bytes(
                        data.get(cursor..cursor + 2)
                            .ok_or_else(invalid)?
                            .try_into()
                            .map_err(|_| invalid())?,
                    ));
                    cursor += 2;
                    let mut previous = 0;
                    for index in 0..ranges {
                        let start = usize::from(u16::from_be_bytes(
                            data.get(cursor..cursor + 2)
                                .ok_or_else(invalid)?
                                .try_into()
                                .map_err(|_| invalid())?,
                        ));
                        let fd = usize::from(*data.get(cursor + 2).ok_or_else(invalid)?);
                        cursor += 3;
                        let end = usize::from(u16::from_be_bytes(
                            data.get(cursor..cursor + 2)
                                .ok_or_else(invalid)?
                                .try_into()
                                .map_err(|_| invalid())?,
                        ));
                        if start != previous
                            || end <= start
                            || end > selection.len()
                            || (index + 1 == ranges && end != selection.len())
                        {
                            return Err(invalid());
                        }
                        selection[start..end].fill(fd);
                        previous = end;
                    }
                }
                _ => return Err(invalid()),
            }
            if selection.iter().any(|&index| index >= private.len()) {
                return Err(invalid());
            }
        } else {
            private.push(Private::parse(data, top.get(&18))?);
            matrices.push(None);
        }
        let charset = charset(
            data,
            top.get(&15)
                .and_then(|value| value.first())
                .copied()
                .unwrap_or(0)
                / ONE,
            charstrings.objects.len(),
        )?;
        Ok(Self {
            top,
            global,
            charstrings,
            private,
            selection,
            matrix,
            matrices,
            charset,
        })
    }

    pub fn random_states(&self, seed: &mut u32) -> Vec<u32> {
        self.private
            .iter()
            .map(|private| {
                let current = *seed;
                if *seed != 0 {
                    loop {
                        *seed = random(*seed);
                        if (*seed as i32) >= 0 {
                            break;
                        }
                    }
                }
                if current != 0 {
                    current
                } else {
                    (private
                        .values
                        .get(&0x0c13)
                        .and_then(|values| values.first())
                        .copied()
                        .unwrap_or(987654321 * ONE)
                        >> 16) as u32
                }
            })
            .collect()
    }

    pub fn decode(&self, id: usize, random_states: &mut [u32]) -> Result<Glyph> {
        let selected = *self.selection.get(id).ok_or_else(invalid)?;
        let private = self.private.get(selected).ok_or_else(invalid)?;
        let program = *self.charstrings.objects.get(id).ok_or_else(invalid)?;
        let mut decoder = Decoder {
            private,
            global: &self.global,
            position: Point::default(),
            stack: Vec::new(),
            transient: [0; 32],
            random: *random_states.get(selected).ok_or_else(invalid)?,
            width_seen: false,
            contour: false,
            instructions_left: 1_000_000,
            glyph: Glyph {
                commands: Vec::new(),
                stems: Vec::new(),
                masks: Vec::new(),
                advance: private.default_width,
                composite: None,
            },
        };
        decoder.execute(program, 0)?;
        decoder.close();
        random_states[selected] = decoder.random;
        Ok(decoder.glyph)
    }

    pub fn private(&self, id: usize) -> Result<&Private<'a>> {
        self.private
            .get(*self.selection.get(id).ok_or_else(invalid)?)
            .ok_or_else(invalid)
    }

    pub fn transform(&self, id: usize, units: i64) -> Result<(Matrix, i64)> {
        let root = self
            .matrix
            .unwrap_or_else(|| Matrix::identity(units))
            .normalize()?;
        let mut matrix = if let Some(mut sub) = self
            .matrices
            .get(*self.selection.get(id).ok_or_else(invalid)?)
            .copied()
            .flatten()
        {
            if self.matrix.is_some() {
                let scaling = if root.units > 1 && sub.units > 1 {
                    root.units.min(sub.units)
                } else {
                    1
                };
                let a = root.values;
                let b = sub.values;
                sub.values = [
                    mul_div(a[0], b[0], scaling * ONE)? + mul_div(a[2], b[1], scaling * ONE)?,
                    mul_div(a[1], b[0], scaling * ONE)? + mul_div(a[3], b[1], scaling * ONE)?,
                    mul_div(a[0], b[2], scaling * ONE)? + mul_div(a[2], b[3], scaling * ONE)?,
                    mul_div(a[1], b[2], scaling * ONE)? + mul_div(a[3], b[3], scaling * ONE)?,
                ];
                let [x, y] = sub.offset;
                sub.offset = [
                    mul_div(a[0], x, scaling * ONE)? + mul_div(a[2], y, scaling * ONE)?,
                    mul_div(a[1], x, scaling * ONE)? + mul_div(a[3], y, scaling * ONE)?,
                ];
                sub.units = mul_div(sub.units, root.units, scaling)?;
            }
            sub.normalize()?
        } else {
            root
        };
        if matrix.units <= 0 {
            matrix = Matrix::identity(1);
        }
        Ok((matrix, root.units))
    }

    pub fn seac(&self, character: u8) -> Result<usize> {
        if self.top.contains_key(&0x0c1e) {
            return Err(invalid());
        }
        let sid = match character {
            32..=126 => u16::from(character) - 31,
            161..=175 => u16::from(character) - 65,
            177..=180 => u16::from(character) - 66,
            182..=189 => u16::from(character) - 67,
            191 => 123,
            193..=200 => u16::from(character) - 69,
            202 => 132,
            203 => 133,
            205..=208 => u16::from(character) - 71,
            225 => 138,
            227 => 139,
            232..=235 => u16::from(character) - 92,
            241 => 144,
            245 => 145,
            248..=251 => u16::from(character) - 102,
            _ => 0,
        };
        self.charset
            .iter()
            .position(|&value| value == sid)
            .ok_or_else(invalid)
    }
}

fn random(mut value: u32) -> u32 {
    value ^= value << 13;
    value ^= value >> 17;
    value ^ (value << 5)
}

fn add(a: i64, b: i64) -> i64 {
    i64::from((a as i32).wrapping_add(b as i32))
}

fn charset(data: &[u8], offset: i64, count: usize) -> Result<Vec<u16>> {
    const EXPERT: &[u16] = &[
        0, 1, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 13, 14, 15, 99, 239, 240, 241, 242,
        243, 244, 245, 246, 247, 248, 27, 28, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
        259, 260, 261, 262, 263, 264, 265, 266, 109, 110, 267, 268, 269, 270, 271, 272, 273, 274,
        275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
        293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
        311, 312, 313, 314, 315, 316, 317, 318, 158, 155, 163, 319, 320, 321, 322, 323, 324, 325,
        326, 150, 164, 169, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358,
        359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
        377, 378,
    ];
    const SUBSET: &[u16] = &[
        0, 1, 231, 232, 235, 236, 237, 238, 13, 14, 15, 99, 239, 240, 241, 242, 243, 244, 245, 246,
        247, 248, 27, 28, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
        264, 265, 266, 109, 110, 267, 268, 269, 270, 272, 300, 301, 302, 305, 314, 315, 158, 155,
        163, 320, 321, 322, 323, 324, 325, 326, 150, 164, 169, 327, 328, 329, 330, 331, 332, 333,
        334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346,
    ];
    if offset == 0 {
        if count > 229 {
            return Err(invalid());
        }
        return Ok((0..count as u16).collect());
    }
    if offset == 1 || offset == 2 {
        return (if offset == 1 { EXPERT } else { SUBSET })
            .get(..count)
            .map(<[u16]>::to_vec)
            .ok_or_else(invalid);
    }
    let mut cursor = usize::try_from(offset).map_err(|_| invalid())?;
    let format = *data.get(cursor).ok_or_else(invalid)?;
    cursor += 1;
    let mut charset = vec![0];
    while charset.len() < count {
        let first = super::u16_at(data, cursor)?;
        cursor += 2;
        let left = match format {
            0 => 0,
            1 => {
                let value = usize::from(*data.get(cursor).ok_or_else(invalid)?);
                cursor += 1;
                value
            }
            2 => {
                let value = usize::from(super::u16_at(data, cursor)?);
                cursor += 2;
                value
            }
            _ => return Err(invalid()),
        };
        if left >= count - charset.len() {
            return Err(invalid());
        }
        for index in 0..=left {
            charset.push(first.checked_add(index as u16).ok_or_else(invalid)?);
        }
    }
    Ok(charset)
}

struct Decoder<'a, 'font> {
    private: &'font Private<'a>,
    global: &'font Index<'a>,
    position: Point,
    stack: Vec<i64>,
    transient: [i64; 32],
    random: u32,
    width_seen: bool,
    contour: bool,
    instructions_left: usize,
    glyph: Glyph,
}

impl Decoder<'_, '_> {
    fn pop(&mut self) -> Result<i64> {
        self.stack.pop().ok_or_else(invalid)
    }
    fn width(&mut self, expected: usize, parity: bool) {
        if self.width_seen {
            return;
        }
        let present = if parity {
            self.stack.len() & 1 != 0
        } else {
            self.stack.len() > expected
        };
        if present {
            self.glyph.advance = add(self.private.nominal_width, self.stack.remove(0));
        }
        self.width_seen = true;
    }
    fn close(&mut self) {
        if self.contour {
            self.glyph.commands.push(Command::Close);
            self.contour = false;
        }
    }
    fn move_to(&mut self, dx: i64, dy: i64) {
        self.close();
        self.position.x = add(self.position.x, dx);
        self.position.y = add(self.position.y, dy);
        self.glyph.commands.push(Command::Move(self.position));
        self.contour = true;
    }
    fn line(&mut self, dx: i64, dy: i64) {
        self.position.x = add(self.position.x, dx);
        self.position.y = add(self.position.y, dy);
        self.glyph.commands.push(Command::Line(self.position));
    }
    fn curve(&mut self, delta: &[i64]) -> Result<()> {
        if delta.len() != 6 {
            return Err(invalid());
        }
        self.position.x = add(self.position.x, delta[0]);
        self.position.y = add(self.position.y, delta[1]);
        let first = self.position;
        self.position.x = add(self.position.x, delta[2]);
        self.position.y = add(self.position.y, delta[3]);
        let second = self.position;
        self.position.x = add(self.position.x, delta[4]);
        self.position.y = add(self.position.y, delta[5]);
        self.glyph
            .commands
            .push(Command::Curve(first, second, self.position));
        Ok(())
    }
    fn stems(&mut self, vertical: bool) -> Result<()> {
        self.width(0, true);
        if self.stack.len() & 1 != 0 {
            return Err(invalid());
        }
        let mut position = 0;
        for pair in self.stack.chunks_exact(2) {
            position = add(position, pair[0]);
            self.glyph.stems.push(Stem {
                vertical,
                start: position,
                width: pair[1],
            });
            position = add(position, pair[1]);
        }
        self.stack.clear();
        Ok(())
    }

    fn execute(&mut self, program: &[u8], depth: usize) -> Result<()> {
        if depth > 16 {
            return Err(invalid());
        }
        let mut cursor = 0;
        while cursor < program.len() {
            self.instructions_left = self.instructions_left.checked_sub(1).ok_or_else(invalid)?;
            if let Some(value) = number(program, &mut cursor, false)? {
                self.stack.push(value);
                if self.stack.len() > 513 {
                    return Err(invalid());
                }
                continue;
            }
            let operator = program[cursor];
            cursor += 1;
            match operator {
                1 | 18 => self.stems(false)?,
                3 | 23 => self.stems(true)?,
                4 | 21 | 22 => {
                    self.width(if operator == 21 { 2 } else { 1 }, false);
                    let (x, y) = match (operator, self.stack.as_slice()) {
                        (4, [y]) => (0, *y),
                        (21, [x, y]) => (*x, *y),
                        (22, [x]) => (*x, 0),
                        _ => return Err(invalid()),
                    };
                    self.move_to(x, y);
                    self.stack.clear();
                }
                5 => {
                    let values = std::mem::take(&mut self.stack);
                    if !values.len().is_multiple_of(2) {
                        return Err(invalid());
                    }
                    for pair in values.chunks_exact(2) {
                        self.line(pair[0], pair[1]);
                    }
                }
                6 | 7 => {
                    let values = std::mem::take(&mut self.stack);
                    for (index, value) in values.into_iter().enumerate() {
                        if (index & 1 == 0) == (operator == 6) {
                            self.line(value, 0);
                        } else {
                            self.line(0, value);
                        }
                    }
                }
                8 => {
                    let values = std::mem::take(&mut self.stack);
                    if !values.len().is_multiple_of(6) {
                        return Err(invalid());
                    }
                    for delta in values.chunks_exact(6) {
                        self.curve(delta)?;
                    }
                }
                10 | 29 => {
                    let number = self.pop()? / ONE;
                    let body = if operator == 10 {
                        self.private.local.subroutine(number)?
                    } else {
                        self.global.subroutine(number)?
                    };
                    self.execute(body, depth + 1)?;
                }
                11 => return Ok(()),
                12 => {
                    let escape = *program.get(cursor).ok_or_else(invalid)?;
                    cursor += 1;
                    self.escape(escape)?;
                }
                14 => {
                    self.width(if self.stack.len() >= 4 { 4 } else { 0 }, false);
                    if let [x, y, base, accent] = self.stack.as_slice() {
                        self.glyph.composite = Some((
                            Point { x: *x, y: *y },
                            u8::try_from(base / ONE).map_err(|_| invalid())?,
                            u8::try_from(accent / ONE).map_err(|_| invalid())?,
                        ));
                    } else if !self.stack.is_empty() {
                        return Err(invalid());
                    }
                    self.stack.clear();
                    self.close();
                    return Ok(());
                }
                19 | 20 => {
                    self.stems(true)?;
                    let bytes = self.glyph.stems.len().div_ceil(8);
                    let bits = program
                        .get(cursor..cursor + bytes)
                        .ok_or_else(invalid)?
                        .to_vec();
                    cursor += bytes;
                    self.glyph.masks.push(HintMask {
                        command: self.glyph.commands.len(),
                        counter: operator == 20,
                        bits,
                    });
                }
                24 => {
                    let values = std::mem::take(&mut self.stack);
                    if values.len() < 8 || !(values.len() - 2).is_multiple_of(6) {
                        return Err(invalid());
                    }
                    let n = values.len() - 2;
                    for delta in values[..n].chunks_exact(6) {
                        self.curve(delta)?;
                    }
                    self.line(values[n], values[n + 1]);
                }
                25 => {
                    let values = std::mem::take(&mut self.stack);
                    if values.len() < 8 || !(values.len() - 6).is_multiple_of(2) {
                        return Err(invalid());
                    }
                    let n = values.len() - 6;
                    for pair in values[..n].chunks_exact(2) {
                        self.line(pair[0], pair[1]);
                    }
                    self.curve(&values[n..])?;
                }
                26 | 27 => {
                    let mut values = std::mem::take(&mut self.stack);
                    let mut optional = if values.len() & 1 != 0 {
                        values.remove(0)
                    } else {
                        0
                    };
                    if !values.len().is_multiple_of(4) {
                        return Err(invalid());
                    }
                    for delta in values.chunks_exact(4) {
                        if operator == 26 {
                            self.curve(&[optional, delta[0], delta[1], delta[2], 0, delta[3]])?;
                        } else {
                            self.curve(&[delta[0], optional, delta[1], delta[2], delta[3], 0])?;
                        }
                        optional = 0;
                    }
                }
                30 | 31 => {
                    let values = std::mem::take(&mut self.stack);
                    let mut cursor = 0;
                    let mut horizontal = operator == 31;
                    if values.len() % 4 > 1 {
                        return Err(invalid());
                    }
                    while cursor + 4 <= values.len() {
                        let delta = &values[cursor..cursor + 4];
                        cursor += 4;
                        let extra = if values.len() - cursor == 1 {
                            let value = values[cursor];
                            cursor += 1;
                            value
                        } else {
                            0
                        };
                        if horizontal {
                            self.curve(&[delta[0], 0, delta[1], delta[2], extra, delta[3]])?;
                        } else {
                            self.curve(&[0, delta[0], delta[1], delta[2], delta[3], extra])?;
                        }
                        horizontal = !horizontal;
                    }
                    if cursor != values.len() {
                        return Err(invalid());
                    }
                }
                _ => return Err(invalid()),
            }
        }
        Ok(())
    }

    fn escape(&mut self, operator: u8) -> Result<()> {
        match operator {
            3 | 4 | 10 | 11 | 12 | 15 | 24 => {
                let b = self.pop()?;
                let a = self.pop()?;
                let value = match operator {
                    3 => i64::from(a != 0 && b != 0) * ONE,
                    4 => i64::from(a != 0 || b != 0) * ONE,
                    10 => a + b,
                    11 => a - b,
                    12 => mul_div(a, ONE, b)?,
                    15 => i64::from(a == b) * ONE,
                    _ => mul_div(a, b, ONE)?,
                };
                self.stack.push(i64::from(value as i32));
            }
            5 | 9 | 14 | 26 => {
                let value = self.pop()?;
                let value = match operator {
                    5 => i64::from(value == 0) * ONE,
                    9 => value.abs(),
                    14 => -value,
                    _ => {
                        if value <= 0 {
                            0
                        } else {
                            let mut root = if value < 10 { value } else { value >> 1 };
                            loop {
                                let next = (root + mul_div(value, ONE, root)? + 1) >> 1;
                                if next == root {
                                    break next;
                                }
                                root = next;
                            }
                        }
                    }
                };
                self.stack.push(i64::from(value as i32));
            }
            18 => {
                self.pop()?;
            }
            20 => {
                let index = usize::try_from(self.pop()? / ONE).map_err(|_| invalid())?;
                let value = self.pop()?;
                *self.transient.get_mut(index).ok_or_else(invalid)? = value;
            }
            21 => {
                let index = usize::try_from(self.pop()? / ONE).map_err(|_| invalid())?;
                self.stack
                    .push(*self.transient.get(index).ok_or_else(invalid)?);
            }
            22 => {
                let v2 = self.pop()?;
                let v1 = self.pop()?;
                let s2 = self.pop()?;
                let s1 = self.pop()?;
                self.stack.push(if v1 <= v2 { s1 } else { s2 });
            }
            23 => {
                self.stack.push(i64::from((self.random & 65535) + 1));
                self.random = random(self.random);
            }
            27 => self.stack.push(*self.stack.last().ok_or_else(invalid)?),
            28 => {
                let n = self.stack.len();
                if n < 2 {
                    return Err(invalid());
                }
                self.stack.swap(n - 1, n - 2);
            }
            29 => {
                let index = (self.pop()? / ONE).max(0) as usize;
                let index = self
                    .stack
                    .len()
                    .checked_sub(1 + index.min(self.stack.len().saturating_sub(1)))
                    .ok_or_else(invalid)?;
                self.stack.push(self.stack[index]);
            }
            30 => {
                let shift = self.pop()? / ONE;
                let count = usize::try_from(self.pop()? / ONE).map_err(|_| invalid())?;
                let first = self.stack.len().checked_sub(count).ok_or_else(invalid)?;
                if count != 0 {
                    self.stack[first..].rotate_right(shift.rem_euclid(count as i64) as usize);
                }
            }
            34..=37 => {
                let values = std::mem::take(&mut self.stack);
                match (operator, values.as_slice()) {
                    (34, [a, b, c, d, e, f, g]) => {
                        self.curve(&[*a, 0, *b, *c, *d, 0])?;
                        self.curve(&[*e, 0, *f, -*c, *g, 0])?;
                    }
                    (35, values) if values.len() == 13 => {
                        self.curve(&values[..6])?;
                        self.curve(&values[6..12])?;
                    }
                    (36, [a, b, c, d, e, f, g, h, i]) => {
                        self.curve(&[*a, *b, *c, *d, *e, 0])?;
                        self.curve(&[*f, 0, *g, *h, *i, -b - d - h])?;
                    }
                    (37, values) if values.len() == 11 => {
                        let x = values[..10].iter().step_by(2).sum::<i64>();
                        let y = values[1..10].iter().step_by(2).sum::<i64>();
                        self.curve(&values[..6])?;
                        self.curve(&[
                            values[6],
                            values[7],
                            values[8],
                            values[9],
                            if x.abs() > y.abs() { values[10] } else { -x },
                            if x.abs() > y.abs() { -y } else { values[10] },
                        ])?;
                    }
                    _ => return Err(invalid()),
                }
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }
}
