//! BASIC caption layout and grayscale-mask composition for the FIBO profile.

use super::{hinting::mul_div, invalid, truetype::Face, Result};

pub(super) struct TextMask {
    pub width: usize,
    pub height: usize,
    pub offset: [i64; 2],
    pub advance: i64,
    pub pixels: Vec<u8>,
}

fn pixel(value: i64) -> i64 {
    (value + 32) >> 6
}

fn mask(font: &mut Face, text: &str, fractional_x: i64) -> Result<TextMask> {
    let mut glyphs = Vec::new();
    let mut previous = 0;
    for character in text.chars() {
        let index = font.font.glyph_index(character)?;
        if previous != 0 && index != 0 {
            let raw = i64::from(font.font.kerning(previous, index)?);
            let mut kerning = mul_div(raw, font.interpreter.scale, 65536)?;
            if font.interpreter.ppem < 25 {
                kerning = mul_div(kerning, font.interpreter.ppem, 25)?;
            }
            kerning = (kerning + 32) & !63;
            if let Some((_, advance)) = glyphs.last_mut() {
                *advance += pixel(kerning);
            }
        }
        let glyph = font.render(character)?;
        let advance = glyph.advance;
        glyphs.push((glyph.bitmap, advance));
        previous = index;
    }
    let (mut pen, mut left, mut right, mut bottom, mut top) = (0, 0, 0, 0, 0);
    for (bitmap, advance) in &glyphs {
        let x = pixel(pen);
        left = left.min(x + i64::from(bitmap.left));
        right = right.max(x + i64::from(bitmap.left) + bitmap.width as i64);
        top = top.max(i64::from(bitmap.top));
        bottom = bottom.min(i64::from(bitmap.top) - bitmap.height as i64);
        pen += advance;
        right = right.max(pixel(pen));
    }
    let width =
        usize::try_from(right - left + i64::from(fractional_x != 0)).map_err(|_| invalid())?;
    let height = usize::try_from(top - bottom).map_err(|_| invalid())?;
    let mut pixels = vec![0_u8; width.checked_mul(height).ok_or_else(invalid)?];
    let advance = pen;
    pen = -left * 64 + fractional_x;
    for (bitmap, advance) in glyphs {
        let x = pixel(pen) + i64::from(bitmap.left);
        let y = top - i64::from(bitmap.top);
        for row in 0..bitmap.height {
            for column in 0..bitmap.width {
                let destination_x = x + column as i64;
                let destination_y = y + row as i64;
                if (0..width as i64).contains(&destination_x)
                    && (0..height as i64).contains(&destination_y)
                {
                    let destination =
                        &mut pixels[destination_y as usize * width + destination_x as usize];
                    // Pillow accumulates overlapping glyph coverage with
                    // rounded alpha-over, including faint adjacent edges.
                    let alpha = u32::from(bitmap.pixels[row * bitmap.width + column]);
                    let product = u32::from(*destination) * (255 - alpha) + 128;
                    *destination = (alpha + ((product + (product >> 8)) >> 8)).min(255) as u8;
                }
            }
        }
        pen += advance;
    }
    Ok(TextMask {
        width,
        height,
        offset: [left, font.ascender / 64 - top],
        advance,
        pixels,
    })
}

fn has_glyph(font: &mut Face, character: char) -> Result<bool> {
    let missing = mask(font, "\u{fffe}", 0)?;
    let glyph = mask(font, &character.to_string(), 0)?;
    Ok(missing.width != glyph.width
        || missing.height != glyph.height
        || missing.pixels != glyph.pixels)
}

fn paint(destination: &mut [u8], width: usize, height: usize, mask: &TextMask, origin: [i64; 2]) {
    for row in 0..mask.height {
        let y = origin[1] + row as i64;
        if !(0..height as i64).contains(&y) {
            continue;
        }
        for column in 0..mask.width {
            let x = origin[0] + column as i64;
            if !(0..width as i64).contains(&x) {
                continue;
            }
            let alpha = u32::from(mask.pixels[row * mask.width + column]);
            let start = (y as usize * width + x as usize) * 3;
            for channel in &mut destination[start..start + 3] {
                let product = u32::from(*channel) * (255 - alpha) + 128;
                *channel = ((product + (product >> 8)) >> 8) as u8;
            }
        }
    }
}

fn lines(text: &str) -> Vec<String> {
    let whitespace = |c| matches!(c, '\u{0009}'..='\u{000d}' | '\u{001c}'..='\u{0020}' | '\u{0085}' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}');
    let compact = text
        .split(whitespace)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let characters = compact.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        vec![String::new()]
    } else {
        characters
            .chunks(256)
            .map(|line| line.iter().collect())
            .collect()
    }
}

pub(super) fn render(fonts: &mut [Face], text: &str) -> Result<crate::media::ImageFrame> {
    if fonts.is_empty() {
        return Err(invalid());
    }
    let lines = lines(text);
    let mut mixed = false;
    for character in lines.iter().flat_map(|line| line.chars()) {
        if !character.is_ascii() && !has_glyph(&mut fonts[0], character)? {
            mixed = true;
            break;
        }
    }
    let (width, height, rows) = if !mixed {
        let minimum = mask(&mut fonts[0], "Ag", 0)?.height.max(1);
        let mut width = 1;
        let mut height = 0;
        let mut rows = Vec::new();
        for line in &lines {
            let shape = mask(&mut fonts[0], if line.is_empty() { " " } else { line }, 0)?;
            let row_height = minimum.max(shape.height);
            width = width.max(shape.width);
            rows.push((height + 2, vec![(2_i64, shape)]));
            height += row_height + 4;
        }
        (width + 4, height, rows)
    } else {
        let ascender = fonts[0].ascender / 64;
        let descent = -fonts[0].descender / 64;
        let line_height = usize::try_from((ascender + descent).max(1)).map_err(|_| invalid())?;
        let mut maximum = 1;
        let mut rows = Vec::new();
        for (line_index, line) in lines.iter().enumerate() {
            let mut runs: Vec<(usize, String)> = Vec::new();
            for character in (if line.is_empty() { " " } else { line }).chars() {
                let mut selected = 0;
                if !character.is_ascii() && !has_glyph(&mut fonts[0], character)? {
                    for (index, font) in fonts.iter_mut().enumerate().skip(1) {
                        if has_glyph(font, character)? {
                            selected = index;
                            break;
                        }
                    }
                }
                if let Some((_, run)) = runs.last_mut().filter(|(index, _)| *index == selected) {
                    run.push(character);
                } else {
                    runs.push((selected, character.to_string()));
                }
            }
            let mut row = Vec::new();
            let mut pen: i64 = 0;
            for (index, text) in runs {
                let shape = mask(&mut fonts[index], &text, pen.rem_euclid(64))?;
                let advance = shape.advance;
                let x = 2 + pen.div_euclid(64) + shape.offset[0];
                // Mixed runs use anchor `ls`, rather than the default `la`.
                let baseline_offset = shape.offset[1] - fonts[index].ascender / 64;
                row.push((x, baseline_offset, shape));
                pen += advance;
            }
            let whole = pen.div_euclid(64);
            let remainder = pen.rem_euclid(64);
            let rounded = whole + i64::from(remainder > 32 || (remainder == 32 && whole & 1 != 0));
            maximum = maximum.max(usize::try_from(rounded.max(1)).map_err(|_| invalid())?);
            let y = 2 + line_index * (line_height + 4);
            rows.push((y, row));
        }
        let height = line_height
            .checked_mul(lines.len())
            .and_then(|v| v.checked_add(4 * (lines.len() - 1)))
            .and_then(|v| v.checked_add(4))
            .ok_or_else(invalid)?;
        let width = maximum + 4;
        let mut pixels = vec![
            255;
            width
                .checked_mul(height)
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(invalid)?
        ];
        for (y, row) in rows {
            for (x, offset, shape) in row {
                paint(
                    &mut pixels,
                    width,
                    height,
                    &shape,
                    [x, y as i64 + ascender + offset],
                );
            }
        }
        return crate::media::ImageFrame::new(
            width,
            height,
            crate::media::PixelFormat::Rgb8,
            pixels,
        )
        .map_err(std::io::Error::other);
    };
    let mut pixels = vec![
        255;
        width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(invalid)?
    ];
    for (y, row) in rows {
        for (x, shape) in row {
            paint(&mut pixels, width, height, &shape, [x, y as i64]);
        }
    }
    crate::media::ImageFrame::new(width, height, crate::media::PixelFormat::Rgb8, pixels)
        .map_err(std::io::Error::other)
}
