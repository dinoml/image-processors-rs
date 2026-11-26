//! Arabic presentation-form metadata for the native automatic hinter.

use super::{invalid, u16_at, Element, Result, TrueTypeFont};

type LigatureRule = (Vec<usize>, usize);
type LigatureSet = (usize, Vec<LigatureRule>);
type ScalarLigature<const N: usize> = ([u16; N], u16);
type ScalarLigatureSet<'a, const N: usize> = (u16, &'a [ScalarLigature<N>]);

pub(super) enum Substitution {
    Single(Vec<(usize, usize)>),
    Ligature(Vec<LigatureSet>),
}

pub(super) struct Lookup {
    pub mask: u64,
    pub flags: u16,
    pub substitution: Substitution,
}

// Unicode Arabic presentation forms: nominal scalar, isolated form, number
// of contiguous forms. Four-form letters use isolated/final/initial/medial.
// ALEF MAKSURA has its initial/medial forms in Presentation Forms-A.
const FORMS: &[(u16, u16, u8)] = &[
    (0x621, 0xFE80, 1),
    (0x622, 0xFE81, 2),
    (0x623, 0xFE83, 2),
    (0x624, 0xFE85, 2),
    (0x625, 0xFE87, 2),
    (0x626, 0xFE89, 4),
    (0x627, 0xFE8D, 2),
    (0x628, 0xFE8F, 4),
    (0x629, 0xFE93, 2),
    (0x62A, 0xFE95, 4),
    (0x62B, 0xFE99, 4),
    (0x62C, 0xFE9D, 4),
    (0x62D, 0xFEA1, 4),
    (0x62E, 0xFEA5, 4),
    (0x62F, 0xFEA9, 2),
    (0x630, 0xFEAB, 2),
    (0x631, 0xFEAD, 2),
    (0x632, 0xFEAF, 2),
    (0x633, 0xFEB1, 4),
    (0x634, 0xFEB5, 4),
    (0x635, 0xFEB9, 4),
    (0x636, 0xFEBD, 4),
    (0x637, 0xFEC1, 4),
    (0x638, 0xFEC5, 4),
    (0x639, 0xFEC9, 4),
    (0x63A, 0xFECD, 4),
    (0x641, 0xFED1, 4),
    (0x642, 0xFED5, 4),
    (0x643, 0xFED9, 4),
    (0x644, 0xFEDD, 4),
    (0x645, 0xFEE1, 4),
    (0x646, 0xFEE5, 4),
    (0x647, 0xFEE9, 4),
    (0x648, 0xFEED, 2),
    (0x649, 0xFEEF, 2),
    (0x64A, 0xFEF1, 4),
    (0x671, 0xFB50, 2),
    (0x677, 0xFBDD, 1),
    (0x679, 0xFB66, 4),
    (0x67A, 0xFB5E, 4),
    (0x67B, 0xFB52, 4),
    (0x67E, 0xFB56, 4),
    (0x67F, 0xFB62, 4),
    (0x680, 0xFB5A, 4),
    (0x683, 0xFB76, 4),
    (0x684, 0xFB72, 4),
    (0x686, 0xFB7A, 4),
    (0x687, 0xFB7E, 4),
    (0x688, 0xFB88, 2),
    (0x68C, 0xFB84, 2),
    (0x68D, 0xFB82, 2),
    (0x68E, 0xFB86, 2),
    (0x691, 0xFB8C, 2),
    (0x698, 0xFB8A, 2),
    (0x6A4, 0xFB6A, 4),
    (0x6A6, 0xFB6E, 4),
    (0x6A9, 0xFB8E, 4),
    (0x6AD, 0xFBD3, 4),
    (0x6AF, 0xFB92, 4),
    (0x6B1, 0xFB9A, 4),
    (0x6B3, 0xFB96, 4),
    (0x6BA, 0xFB9E, 2),
    (0x6BB, 0xFBA0, 4),
    (0x6BE, 0xFBAA, 4),
    (0x6C0, 0xFBA4, 2),
    (0x6C1, 0xFBA6, 4),
    (0x6C5, 0xFBE0, 2),
    (0x6C6, 0xFBD9, 2),
    (0x6C7, 0xFBD7, 2),
    (0x6C8, 0xFBDB, 2),
    (0x6C9, 0xFBE2, 2),
    (0x6CB, 0xFBDE, 2),
    (0x6CC, 0xFBFC, 4),
    (0x6D0, 0xFBE4, 4),
    (0x6D2, 0xFBAE, 2),
    (0x6D3, 0xFBB0, 2),
];

const PAIRS: &[ScalarLigatureSet<'_, 1>] = &[
    (
        0xFE91,
        &[
            ([0xFEE2], 0xFC08),
            ([0xFEE4], 0xFC9F),
            ([0xFEA0], 0xFC9C),
            ([0xFEA4], 0xFC9D),
            ([0xFEA8], 0xFC9E),
        ],
    ),
    (
        0xFE92,
        &[([0xFEAE], 0xFC6A), ([0xFEE6], 0xFC6D), ([0xFEF2], 0xFC6F)],
    ),
    (
        0xFE97,
        &[
            ([0xFEE2], 0xFC0E),
            ([0xFEE4], 0xFCA4),
            ([0xFEA0], 0xFCA1),
            ([0xFEA4], 0xFCA2),
            ([0xFEA8], 0xFCA3),
        ],
    ),
    (
        0xFE98,
        &[([0xFEAE], 0xFC70), ([0xFEE6], 0xFC73), ([0xFEF2], 0xFC75)],
    ),
    (0xFE9B, &[([0xFEE2], 0xFC12)]),
    (0xFE9F, &[([0xFEE4], 0xFCA8)]),
    (0xFEA3, &[([0xFEE4], 0xFCAA)]),
    (0xFEA7, &[([0xFEE4], 0xFCAC)]),
    (0xFEB3, &[([0xFEE4], 0xFCB0)]),
    (0xFEB7, &[([0xFEE4], 0xFD30)]),
    (0xFED3, &[([0xFEF2], 0xFC32)]),
    (
        0xFEDF,
        &[
            ([0xFE9E], 0xFC3F),
            ([0xFEA0], 0xFCC9),
            ([0xFEA2], 0xFC40),
            ([0xFEA4], 0xFCCA),
            ([0xFEA6], 0xFC41),
            ([0xFEA8], 0xFCCB),
            ([0xFEE2], 0xFC42),
            ([0xFEE4], 0xFCCC),
            ([0xFEF2], 0xFC44),
            ([0xFEEC], 0xFCCD),
            ([0xFE82], 0xFEF5),
            ([0xFE84], 0xFEF7),
            ([0xFE88], 0xFEF9),
            ([0xFE8E], 0xFEFB),
        ],
    ),
    (
        0xFEE0,
        &[
            ([0xFEF0], 0xFC86),
            ([0xFE82], 0xFEF6),
            ([0xFE84], 0xFEF8),
            ([0xFE88], 0xFEFA),
            ([0xFE8E], 0xFEFC),
        ],
    ),
    (
        0xFEE3,
        &[
            ([0xFEA0], 0xFCCE),
            ([0xFEA4], 0xFCCF),
            ([0xFEA8], 0xFCD0),
            ([0xFEE4], 0xFCD1),
        ],
    ),
    (
        0xFEE7,
        &[
            ([0xFEE2], 0xFC4E),
            ([0xFEE4], 0xFCD5),
            ([0xFEA0], 0xFCD2),
            ([0xFEA4], 0xFCD3),
        ],
    ),
    (0xFEE8, &[([0xFEF2], 0xFC8F)]),
    (
        0xFEF3,
        &[
            ([0xFEA0], 0xFCDA),
            ([0xFEA4], 0xFCDB),
            ([0xFEA8], 0xFCDC),
            ([0xFEE4], 0xFCDD),
        ],
    ),
    (0xFEF4, &[([0xFEAE], 0xFC91), ([0xFEE6], 0xFC94)]),
];
const MARKS: &[ScalarLigatureSet<'_, 1>] = &[(
    0x651,
    &[
        ([0x64C], 0xFC5E),
        ([0x64E], 0xFC60),
        ([0x64F], 0xFC61),
        ([0x650], 0xFC62),
        ([0x64B], 0xF2EE),
    ],
)];
const TRIPLES: &[ScalarLigatureSet<'_, 2>] = &[(
    0xFEDF,
    &[
        ([0xFEE4, 0xFEA4], 0xFD88),
        ([0xFEE0, 0xFEEA], 0xF201),
        ([0xFEE4, 0xFEA0], 0xF211),
    ],
)];

fn glyph(font: &TrueTypeFont, scalar: u16) -> Result<usize> {
    font.glyph_index(char::from_u32(u32::from(scalar)).ok_or_else(super::invalid)?)
}

fn ligatures<const N: usize>(
    font: &TrueTypeFont,
    metadata: &[ScalarLigatureSet<'_, N>],
    flags: u16,
) -> Result<Option<Lookup>> {
    let mut groups = Vec::new();
    for &(first, variants) in metadata {
        let id = glyph(font, first)?;
        if id == 0 {
            continue;
        }
        groups.push((id, variants));
    }
    groups.sort_by_key(|&(id, _)| id);
    let mut rules = Vec::new();
    let mut components = Vec::new();
    let mut cursor = 0;
    let mut any = false;
    for (id, variants) in groups {
        let mut replacements = Vec::new();
        for &(scalars, target) in variants {
            let target = glyph(font, target)?;
            if target == 0 {
                continue;
            }
            let mut matched = true;
            for scalar in scalars {
                let component = glyph(font, scalar)?;
                if scalar == 0 || component == 0 {
                    matched = false;
                    break;
                }
                components.push(component);
            }
            if !matched {
                continue;
            }
            // The reference synthesizer appends found components before a
            // failed candidate is discarded. Preserve its serialized cursor.
            replacements.push((components[cursor..cursor + N].to_vec(), target));
            cursor += N;
            any = true;
        }
        rules.push((id, replacements));
    }
    Ok(any.then_some(Lookup {
        mask: 1,
        flags,
        substitution: Substitution::Ligature(rules),
    }))
}

pub(super) fn fallback(font: &TrueTypeFont) -> Result<Vec<Lookup>> {
    let mut lookups = Vec::new();
    for (form, mask) in [(2, 8), (3, 16), (1, 32), (0, 4)] {
        let mut replacements = Vec::new();
        for &(scalar, isolated, count) in FORMS {
            let presentation = if scalar == 0x649 && form >= 2 {
                0xFBE8 + form - 2
            } else if form < u16::from(count) {
                isolated + form
            } else {
                continue;
            };
            let from = glyph(font, scalar)?;
            let to = glyph(font, presentation)?;
            if from != 0 && to != 0 && from != to {
                replacements.push((from, to));
            }
        }
        replacements.sort_by_key(|&(from, _)| from);
        if !replacements.is_empty() {
            lookups.push(Lookup {
                mask,
                flags: 8,
                substitution: Substitution::Single(replacements),
            });
        }
    }
    if let Some(lookup) = ligatures(font, TRIPLES, 8)? {
        lookups.push(lookup);
    }
    if let Some(lookup) = ligatures(font, PAIRS, 8)? {
        lookups.push(lookup);
    }
    if let Some(lookup) = ligatures(font, MARKS, 0)? {
        lookups.push(lookup);
    }
    if !lookups.is_empty() {
        return Ok(lookups);
    }
    if glyph(font, 0x627)? == 199
        && glyph(font, 0x644)? == 225
        && glyph(font, 0x649)? == 236
        && glyph(font, 0x64A)? == 237
        && glyph(font, 0x652)? == 250
    {
        // The Windows reference enables the legacy Windows-1256 fallback.
        lookups.push(Lookup {
            mask: 1,
            flags: 8,
            substitution: Substitution::Ligature(vec![(
                225,
                vec![
                    (vec![199], 165),
                    (vec![195], 178),
                    (vec![194], 180),
                    (vec![197], 252),
                ],
            )]),
        });
        let common = [
            (198, 162),
            (200, 4),
            (201, 5),
            (202, 5),
            (203, 6),
            (204, 7),
            (205, 9),
            (206, 11),
            (211, 13),
            (212, 14),
            (213, 15),
            (214, 26),
            (223, 140),
            (225, 141),
            (227, 142),
            (228, 143),
            (236, 154),
            (237, 154),
        ];
        let lam_alef = [(165, 170), (178, 179), (180, 185), (252, 255)];
        for (mask, specific) in [
            (
                8,
                vec![(218, 27), (219, 30), (221, 128), (222, 131), (229, 144)],
            ),
            (
                16,
                vec![(218, 28), (219, 31), (221, 129), (222, 138), (229, 149)],
            ),
        ] {
            let mut replacements = common.to_vec();
            replacements.extend(specific);
            if mask == 16 {
                replacements.extend(lam_alef);
            }
            replacements.sort_by_key(|&(from, _)| from);
            lookups.push(Lookup {
                mask,
                flags: 8,
                substitution: Substitution::Single(replacements),
            });
        }
        let mut replacements = vec![
            (194, 2),
            (195, 1),
            (197, 3),
            (198, 181),
            (199, 0),
            (201, 159),
            (204, 8),
            (205, 10),
            (206, 12),
            (218, 29),
            (219, 127),
            (229, 152),
            (236, 160),
            (237, 156),
        ];
        replacements.extend(lam_alef);
        replacements.sort_by_key(|&(from, _)| from);
        lookups.push(Lookup {
            mask: 32,
            flags: 8,
            substitution: Substitution::Single(replacements),
        });
        lookups.push(Lookup {
            mask: 1,
            flags: 0,
            substitution: Substitution::Ligature(vec![(
                248,
                vec![(vec![243], 172), (vec![245], 173), (vec![246], 175)],
            )]),
        });
    }
    Ok(lookups)
}

pub(super) fn stretch(font: &TrueTypeFont, glyphs: &mut Vec<Element>, rtl: bool) -> Result<()> {
    if !glyphs.iter().any(|glyph| glyph.stretch != 0) {
        return Ok(());
    }
    let hmtx = font.table(b"hmtx")?;
    let advance = |id: usize| -> Result<i64> {
        if font.metrics == 0 || id >= font.glyphs {
            return Err(invalid());
        }
        Ok(i64::from(u16_at(hmtx, 4 * id.min(font.metrics - 1))?))
    };
    if !rtl {
        glyphs.reverse();
    }
    let mut reverse = Vec::new();
    let mut end = glyphs.len();
    while end != 0 {
        if glyphs[end - 1].stretch == 0 {
            reverse.push(glyphs[end - 1].clone());
            end -= 1;
            continue;
        }
        let mut start = end;
        let mut fixed = 0;
        let mut repeating = 0;
        let mut repeating_count = 0;
        while start != 0 && glyphs[start - 1].stretch != 0 {
            start -= 1;
            let width = advance(glyphs[start].id)?;
            if glyphs[start].stretch == 1 {
                fixed += width;
            } else {
                repeating += width;
                repeating_count += 1;
            }
        }
        let mut context = start;
        let mut total = 0;
        while context != 0 && glyphs[context - 1].stretch == 0 {
            let glyph = &glyphs[context - 1];
            // Every Arabic scalar in afscript/afblue is a letter or a mark;
            // substitution retains those Unicode properties. No caption text
            // is passed through this font-unit metric shaper.
            let word = matches!(glyph.character, 0x625 | 0x627 | 0x62A..=0x62B | 0x62D | 0x637..=0x638 | 0x640 | 0x643..=0x644 | 0x655);
            let ignorable = !glyph.substituted && matches!(glyph.character, 0x200C..=0x200D);
            if !word && !ignorable {
                break;
            }
            context -= 1;
            total += glyph.advance;
        }
        let mut remaining = total - fixed;
        let mut copies = if remaining > repeating && repeating > 0 {
            remaining / repeating - 1
        } else {
            0
        };
        let mut overlap = 0;
        let shortfall = remaining - repeating * (copies + 1);
        if shortfall > 0 && repeating_count > 0 {
            copies += 1;
            let excess = (copies + 1) * repeating - remaining;
            if excess > 0 {
                overlap = excess / (copies * repeating_count);
                remaining = 0;
            }
        }
        let mut offset = remaining / 2;
        for index in (start..end).rev() {
            let width = advance(glyphs[index].id)?;
            let repeat = if glyphs[index].stretch == 2 {
                usize::try_from(copies + 1).map_err(|_| invalid())?
            } else {
                1
            };
            reverse.try_reserve(repeat).map_err(|_| invalid())?;
            for copy in 0..repeat {
                if rtl {
                    offset -= width;
                    if copy != 0 {
                        offset += overlap;
                    }
                }
                let mut glyph = glyphs[index].clone();
                glyph.advance = 0;
                glyph.x_offset = offset;
                reverse.push(glyph);
                if !rtl {
                    offset += width;
                    if copy != 0 {
                        offset -= overlap;
                    }
                }
            }
        }
        end = start;
    }
    reverse.reverse();
    if !rtl {
        reverse.reverse();
    }
    *glyphs = reverse;
    Ok(())
}
