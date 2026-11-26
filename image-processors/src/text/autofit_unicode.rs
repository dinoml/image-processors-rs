//! Unicode metadata for the original automatic-hinter sample repertoire.
//!
//! Caption text itself uses BASIC layout.  These operations consume the fixed
//! standard/blue strings in `afscript.h` and `afblue.c`, not arbitrary caption
//! text.  The canonical closure and properties below follow Unicode 15.1.0,
//! the database used by the reference HarfBuzz 8.5.0 build.

use super::{Result, TrueTypeFont};

pub(super) fn samples(text: &str) -> Vec<&str> {
    if text.contains(' ') {
        return text
            .split(' ')
            .filter(|sample| !sample.is_empty())
            .collect();
    }
    // Original single-character samples are stored without their separators.
    text.char_indices()
        .map(|(start, character)| &text[start..start + character.len_utf8()])
        .collect()
}

pub(super) fn mark(character: char) -> bool {
    matches!(character as u32,
        0x655 | 0x93F..=0x941 | 0x943 | 0x94B..=0x94C |
        0x9BF..=0x9C0 | 0x9C8 | 0x9D7 | 0xA3F..=0xA40 |
        0xABF..=0xAC1 | 0xAC3..=0xAC4 | 0xACD | 0xCBE | 0xD4D |
        0xDD4 | 0xDD6 | 0x102B | 0x102D | 0x103C |
        0x17B6..=0x17D3 |
        0xA8BA | 0xA92C..=0xA92D | 0x11133)
}

pub(super) fn combining(character: char) -> u8 {
    match character as u32 {
        0x655 | 0xA92C..=0xA92D => 220,
        0xACD | 0xD4D | 0x17D2 | 0x11133 => 9,
        _ => 0,
    }
}

pub(super) fn nonspacing(character: char) -> bool {
    matches!(character as u32, 0x655 | 0x941 | 0x943 | 0xAC1 | 0xAC3..=0xAC4 | 0xACD |
        0xD4D | 0xDD4 | 0xDD6 | 0x102D | 0x17B7..=0x17BD | 0x17C6 | 0x17C9..=0x17D3 | 0xA92C..=0xA92D | 0x11133)
}

pub(super) fn script(character: char) -> Option<&'static str> {
    let code = character as u32;
    if matches!(code, 0x20 | 0x30..=0x39 | 0x5F | 0x7C | 0xB2..=0xB3 | 0xB9 |
        0x640 | 0x655 | 0x200C..=0x200D | 0x2070 | 0x2075 | 0x2077 |
        0x2080..=0x2083 | 0x2085 | 0x2087..=0x2088 | 0x25CC)
    {
        return None;
    }
    if (0x3040..=0x30FF).contains(&code) {
        return Some("kana");
    }
    if (0xAC00..=0xD7A3).contains(&code) || (0x1100..=0x11FF).contains(&code) {
        return Some("hang");
    }
    if (0x3100..=0x312F).contains(&code) {
        return Some("bopo");
    }
    super::autofit_scripts::SCRIPTS
        .iter()
        .find(|script| {
            script
                .ranges
                .iter()
                .any(|&(first, last)| code >= first && code <= last)
        })
        .and_then(|script| match script.name {
            "latb" | "latp" => Some("latn"),
            "geok" => Some("geor"),
            "khms" | "none" => None,
            name => Some(name),
        })
}

pub(super) fn normalize(font: &TrueTypeFont, text: &str) -> Result<Vec<char>> {
    let mut result = Vec::new();
    for character in text.chars() {
        // This is the sole canonical decomposition in the original standard /
        // blue repertoire and its recursive closure.  A supported precomposed
        // simple cluster is retained by the reference's normalization shortcut.
        if character == '\u{625}'
            && font.glyph_index(character)? == 0
            && font.glyph_index('\u{627}')? != 0
            && font.glyph_index('\u{655}')? != 0
        {
            result.extend(['\u{627}', '\u{655}']);
        } else if matches!(character as u32, 0x17BE..=0x17C0 | 0x17C4..=0x17C5)
            && font.glyph_index(character)? != 0
            && font.glyph_index('\u{17C1}')? != 0
        {
            // Khmer's shaper decomposes split matras even when the original
            // scalar has a glyph; its normalization mode forbids shortcuts.
            result.extend(['\u{17C1}', character]);
        } else {
            result.push(character);
        }
    }
    // Stable canonical ordering within each starter sequence.  The repertoire
    // contains viramas and below marks; equal classes retain their source order.
    for index in 1..result.len() {
        let order = combining(result[index]);
        if order == 0 {
            continue;
        }
        let mut position = index;
        while position > 0 {
            let previous = combining(result[position - 1]);
            if previous == 0 || previous <= order {
                break;
            }
            result.swap(position - 1, position);
            position -= 1;
        }
    }
    let mut composed = Vec::with_capacity(result.len());
    let mut starter = None;
    let mut previous = 0;
    for character in result {
        let order = combining(character);
        if character == '\u{655}' && (previous == 0 || previous < order) {
            if let Some(index) = starter {
                if composed[index] == '\u{627}' && font.glyph_index('\u{625}')? != 0 {
                    composed[index] = '\u{625}';
                    continue;
                }
            }
        }
        if order == 0 {
            starter = Some(composed.len());
        }
        composed.push(character);
        previous = order;
    }
    Ok(composed)
}
