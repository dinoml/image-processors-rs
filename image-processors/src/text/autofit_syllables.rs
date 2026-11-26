//! Native maximal-munch recognition of automatic-hinter sample syllables.

use std::collections::BTreeSet;

#[derive(Clone)]
enum Pattern {
    Category(u64),
    Sequence(Vec<Self>),
    Alternative(Vec<Self>),
    Repeat(Box<Self>),
}

impl Pattern {
    fn ends(&self, input: &[u8], start: usize) -> BTreeSet<usize> {
        match self {
            Self::Category(set) => input
                .get(start)
                .filter(|&&category| category < 64 && set & (1u64 << category) != 0)
                .map(|_| BTreeSet::from([start + 1]))
                .unwrap_or_default(),
            Self::Sequence(parts) => {
                let mut positions = BTreeSet::from([start]);
                for part in parts {
                    positions = positions
                        .into_iter()
                        .flat_map(|position| part.ends(input, position))
                        .collect();
                    if positions.is_empty() {
                        break;
                    }
                }
                positions
            }
            Self::Alternative(parts) => parts
                .iter()
                .flat_map(|part| part.ends(input, start))
                .collect(),
            Self::Repeat(part) => {
                let mut reached = BTreeSet::from([start]);
                let mut pending = vec![start];
                while let Some(position) = pending.pop() {
                    for end in part.ends(input, position) {
                        if reached.insert(end) {
                            pending.push(end);
                        }
                    }
                }
                reached
            }
        }
    }
}

fn category(values: &[u8]) -> Pattern {
    Pattern::Category(values.iter().fold(0, |set, &value| set | (1u64 << value)))
}
fn sequence(parts: Vec<Pattern>) -> Pattern {
    Pattern::Sequence(parts)
}
fn alternative(parts: Vec<Pattern>) -> Pattern {
    Pattern::Alternative(parts)
}
fn repeat(part: Pattern) -> Pattern {
    Pattern::Repeat(Box::new(part))
}
fn optional(part: Pattern) -> Pattern {
    alternative(vec![sequence(Vec::new()), part])
}

fn recognize(input: &[u8], patterns: &[(u8, Pattern)], other: u8) -> Vec<u8> {
    let mut syllables = vec![0; input.len()];
    let mut start = 0;
    let mut serial = 1u8;
    while start < input.len() {
        let mut end = start;
        let mut kind = other;
        for (pattern_kind, pattern) in patterns {
            if let Some(&candidate) = pattern.ends(input, start).last() {
                // A longer match wins; grammar order resolves equal lengths.
                if candidate > end {
                    end = candidate;
                    kind = *pattern_kind;
                }
            }
        }
        if end == start {
            end += 1;
        }
        syllables[start..end].fill((serial << 4) | kind);
        serial = if serial == 15 { 1 } else { serial + 1 };
        start = end;
    }
    syllables
}

pub(super) fn universal(input: &[u8], marks: &[bool]) -> Vec<u8> {
    let halant = category(&[12, 53, 44, 48]);
    let modifiers = sequence(vec![
        repeat(category(&[31])),
        repeat(category(&[32])),
        repeat(sequence(vec![
            alternative(vec![
                sequence(vec![halant, category(&[1])]),
                category(&[11]),
            ]),
            repeat(category(&[31])),
            repeat(category(&[32])),
        ])),
    ]);
    let medial = sequence(vec![
        optional(category(&[30])),
        optional(category(&[27])),
        optional(category(&[28])),
        optional(category(&[29])),
    ]);
    let dependent = alternative(vec![
        sequence(vec![
            repeat(category(&[22])),
            repeat(category(&[33])),
            repeat(category(&[34])),
            repeat(category(&[35])),
        ]),
        category(&[12]),
    ]);
    let vowel_modifiers = sequence(vec![
        optional(category(&[53])),
        repeat(category(&[23])),
        repeat(category(&[37])),
        repeat(category(&[38])),
        repeat(category(&[39])),
    ]);
    let middle = sequence(vec![
        modifiers.clone(),
        medial,
        dependent,
        vowel_modifiers,
        repeat(sequence(vec![category(&[48]), category(&[1])])),
    ]);
    let tail = sequence(vec![
        middle.clone(),
        repeat(category(&[24])),
        repeat(category(&[25])),
        repeat(category(&[26])),
        alternative(vec![
            sequence(vec![repeat(category(&[45])), repeat(category(&[46]))]),
            optional(category(&[47])),
        ]),
    ]);
    let symbol_tail = alternative(vec![
        sequence(vec![
            category(&[41]),
            repeat(category(&[41])),
            repeat(category(&[42])),
        ]),
        sequence(vec![category(&[42]), repeat(category(&[42]))]),
    ]);
    let virama_tail = sequence(vec![modifiers, category(&[44])]);
    let sakot_tail = sequence(vec![middle, category(&[48])]);
    let any_tail = alternative(vec![
        tail.clone(),
        sakot_tail.clone(),
        symbol_tail,
        virama_tail.clone(),
    ]);
    let number_pair = sequence(vec![category(&[13]), category(&[4])]);
    let number_joiner = sequence(vec![repeat(number_pair.clone()), category(&[13])]);
    let numeral = sequence(vec![number_pair.clone(), repeat(number_pair)]);
    let start = sequence(vec![optional(category(&[18, 43])), category(&[1, 5])]);
    let hieroglyph = sequence(vec![
        repeat(category(&[51])),
        category(&[49]),
        optional(category(&[55])),
        optional(category(&[54])),
        repeat(category(&[52])),
        repeat(sequence(vec![
            category(&[50]),
            repeat(category(&[51])),
            optional(sequence(vec![
                category(&[49]),
                optional(category(&[55])),
                optional(category(&[54])),
                repeat(category(&[52])),
            ])),
        ])),
    ]);
    let patterns = [
        sequence(vec![start.clone(), virama_tail]),
        sequence(vec![start.clone(), sakot_tail]),
        sequence(vec![start, tail]),
        sequence(vec![category(&[4]), number_joiner.clone()]),
        sequence(vec![category(&[4]), optional(numeral.clone())]),
        sequence(vec![category(&[0, 5, 51]), optional(any_tail.clone())]),
        hieroglyph,
        sequence(vec![
            optional(category(&[18])),
            alternative(vec![any_tail, number_joiner, numeral]),
        ]),
    ]
    .into_iter()
    .enumerate()
    .map(|(kind, pattern)| {
        (
            kind as u8,
            sequence(vec![pattern, optional(category(&[14]))]),
        )
    })
    .collect::<Vec<_>>();
    // The machine excludes CGJ and a ZWNJ immediately preceding a mark,
    // while retaining their original positions for syllable range labels.
    let filtered = input
        .iter()
        .enumerate()
        .filter(|&(index, &category)| {
            category != 6
                && !(category == 14
                    && (index + 1..input.len())
                        .find(|&next| input[next] != 6)
                        .is_some_and(|next| marks[next]))
        })
        .map(|(index, &category)| (index, category))
        .collect::<Vec<_>>();
    let labels = recognize(
        &filtered
            .iter()
            .map(|&(_, category)| category)
            .collect::<Vec<_>>(),
        &patterns,
        8,
    );
    let mut result = vec![0; input.len()];
    for (index, &(original, _)) in filtered.iter().enumerate() {
        let end = filtered
            .get(index + 1)
            .map_or(input.len(), |&(next, _)| next);
        result[original..end].fill(labels[index]);
    }
    result
}

pub(super) fn universal_category(character: u32, script: &str) -> u8 {
    // These properties cover afscript/afblue's fixed sample repertoire and
    // its normalization closure. They are Unicode categories, not glyph IDs.
    match character {
        0xA73 | 0x104B => 5,
        0x200D => 6,
        0x200C => 14,
        0xACD | 0xD4D => 12,
        0x93F | 0x9BF | 0x9C8 | 0xA3F | 0xABF | 0x17BF..=0x17C0 | 0x17C3 => 22,
        0x103C => 30,
        0x102D => 33,
        0x941 | 0x943 | 0xAC1 | 0xAC3..=0xAC4 | 0xDD4 | 0xDD6 => 34,
        0x940
        | 0x94B..=0x94C
        | 0x9C0
        | 0x9D7
        | 0xA40
        | 0xAC0
        | 0xCBE
        | 0x102B
        | 0x17B6
        | 0xA8BA => 35,
        0xA92C..=0xA92D => 38,
        0x17D2 | 0x11133 => 44,
        0xB2..=0xB3 | 0x2082..=0x2083 => 47,
        _ if script == "medf" || (script == "nkoo" && (0x7C0..=0x7C9).contains(&character)) => 0,
        _ => 1,
    }
}

pub(super) fn khmer(input: &[u8]) -> Vec<u8> {
    // HarfBuzz 8.5 Khmer grammar, expressed as independent category-set
    // matching rather than the generated Ragel transition tables.
    let consonant = category(&[1, 15, 2]);
    let joiner = category(&[5, 6]);
    let cn = sequence(vec![
        consonant.clone(),
        optional(sequence(vec![optional(joiner.clone()), category(&[25])])),
    ]);
    let xgroup = repeat(sequence(vec![repeat(joiner.clone()), category(&[26])]));
    let matra = sequence(vec![
        optional(category(&[22])),
        xgroup.clone(),
        optional(category(&[21])),
        xgroup.clone(),
        optional(sequence(vec![optional(joiner), category(&[20])])),
        xgroup.clone(),
        optional(category(&[23])),
    ]);
    let tail = sequence(vec![
        xgroup.clone(),
        matra,
        xgroup,
        optional(sequence(vec![category(&[4]), consonant])),
        repeat(category(&[27])),
    ]);
    let broken = sequence(vec![
        optional(category(&[25])),
        repeat(sequence(vec![category(&[4]), cn.clone()])),
        alternative(vec![category(&[4]), tail]),
    ]);
    let valid = sequence(vec![
        alternative(vec![cn, category(&[10, 11])]),
        broken.clone(),
    ]);
    recognize(input, &[(0, valid), (1, broken)], 2)
}

pub(super) fn indic(input: &[u8]) -> Vec<u8> {
    let consonant = category(&[1, 15]);
    let joiner = category(&[5, 6]);
    let nukta = sequence(vec![
        optional(sequence(vec![optional(category(&[5])), category(&[12])])),
        optional(sequence(vec![category(&[3]), optional(category(&[3]))])),
    ]);
    let reph = alternative(vec![
        sequence(vec![category(&[15]), category(&[4])]),
        category(&[14]),
    ]);
    let cn = sequence(vec![
        consonant,
        optional(category(&[6])),
        optional(nukta.clone()),
    ]);
    let matra = sequence(vec![
        repeat(joiner.clone()),
        alternative(vec![
            category(&[7]),
            sequence(vec![optional(category(&[8])), category(&[13])]),
        ]),
        optional(category(&[3])),
        optional(category(&[4])),
    ]);
    let tail = sequence(vec![
        optional(sequence(vec![
            optional(joiner.clone()),
            category(&[8]),
            optional(category(&[8])),
            optional(category(&[5])),
        ])),
        repeat(category(&[9])),
    ]);
    let halant = sequence(vec![
        optional(joiner),
        category(&[4]),
        optional(sequence(vec![category(&[6]), optional(category(&[3]))])),
    ]);
    let complex = sequence(vec![
        repeat(sequence(vec![halant.clone(), cn.clone()])),
        optional(category(&[16])),
        alternative(vec![
            halant,
            sequence(vec![category(&[4]), category(&[5])]),
            repeat(matra),
        ]),
        tail.clone(),
    ]);
    let valid = sequence(vec![optional(category(&[14, 18])), cn, complex.clone()]);
    let vowel = sequence(vec![
        optional(reph.clone()),
        category(&[2]),
        optional(nukta.clone()),
        alternative(vec![category(&[6]), complex.clone()]),
    ]);
    let standalone = sequence(vec![
        alternative(vec![
            sequence(vec![optional(category(&[14, 18])), category(&[10])]),
            sequence(vec![optional(reph.clone()), category(&[11])]),
        ]),
        optional(nukta.clone()),
        complex.clone(),
    ]);
    let symbol = sequence(vec![category(&[17]), optional(category(&[3])), tail]);
    let broken = sequence(vec![optional(reph), optional(nukta), complex]);
    recognize(
        input,
        &[
            (0, valid),
            (1, vowel),
            (2, standalone),
            (3, symbol),
            (4, broken),
        ],
        5,
    )
}

pub(super) fn indic_properties(character: u32) -> (u8, u8) {
    // Original afscript/afblue repertoire, using the reference Indic Unicode
    // categories and visual positions. All remaining sample scalars are C/C.
    match character {
        0x905..=0x906
        | 0x908..=0x909
        | 0x910
        | 0x913..=0x914
        | 0x985
        | 0x987
        | 0x98F
        | 0x993
        | 0xA05
        | 0xA07..=0xA09
        | 0xA0F
        | 0xA13
        | 0xA87..=0xA88
        | 0xA8A..=0xA8C
        | 0xB88..=0xB89
        | 0xB92..=0xB93
        | 0xC05
        | 0xC07
        | 0xC0C
        | 0xC85
        | 0xC87
        | 0xC89..=0xC8A
        | 0xC8E
        | 0xC90
        | 0xD12 => (2, 4),
        0xACD => (4, 8),
        0xD4D => (4, 6),
        0x200C => (5, 14),
        0x200D => (6, 14),
        0x93F | 0x9BF | 0x9C8 | 0xA3F | 0xABF => (7, 2),
        0x940..=0x941 | 0x943 | 0x94B..=0x94C => (7, 9),
        0x9C0 | 0x9D7 | 0xAC0..=0xAC1 | 0xAC3..=0xAC4 => (7, 12),
        0xCBE => (7, 7),
        0xA40 => (13, 12),
        0x9E6
        | 0x9EA
        | 0xA66..=0xA69
        | 0xA6D
        | 0xAE6..=0xAE9
        | 0xAED
        | 0xBE6
        | 0xC66..=0xC68
        | 0xC6C
        | 0xC6F
        | 0xCE6
        | 0xCE8
        | 0xCEC..=0xCED => (10, 4),
        0x25CC => (11, 4),
        0xA30 | 0xAB0 | 0xC30 | 0xCB0 => (15, 4),
        0xC3D => (17, 13),
        _ => (1, 4),
    }
}

pub(super) fn myanmar(input: &[u8]) -> Vec<u8> {
    let joiner = category(&[5, 6]);
    let kinzi = sequence(vec![category(&[15]), category(&[32]), category(&[4])]);
    let consonant = category(&[1, 15]);
    let medial = sequence(vec![
        optional(category(&[38])),
        optional(category(&[32])),
        optional(category(&[36])),
        optional(sequence(vec![
            alternative(vec![
                sequence(vec![
                    category(&[37]),
                    optional(category(&[35])),
                    optional(category(&[41])),
                ]),
                sequence(vec![category(&[35]), optional(category(&[41]))]),
                category(&[41]),
            ]),
            optional(category(&[32])),
        ])),
    ]);
    let dot = optional(sequence(vec![category(&[3]), optional(category(&[32]))]));
    let main_vowel = sequence(vec![
        repeat(sequence(vec![category(&[22]), optional(category(&[40]))])),
        repeat(category(&[20])),
        repeat(category(&[21])),
        repeat(category(&[9])),
        dot.clone(),
    ]);
    let post_vowel = sequence(vec![
        category(&[23]),
        optional(category(&[35])),
        optional(category(&[41])),
        repeat(category(&[32])),
        repeat(category(&[20])),
        repeat(category(&[9])),
        dot,
    ]);
    let pwo = sequence(vec![
        category(&[39]),
        repeat(category(&[9])),
        optional(category(&[3])),
        optional(category(&[32])),
    ]);
    let complex_tail = sequence(vec![
        repeat(category(&[32])),
        medial,
        main_vowel,
        repeat(post_vowel),
        repeat(pwo),
        repeat(category(&[8])),
        optional(joiner.clone()),
    ]);
    let tail = sequence(vec![
        repeat(sequence(vec![
            category(&[4]),
            alternative(vec![consonant.clone(), category(&[2])]),
            optional(category(&[40])),
        ])),
        alternative(vec![category(&[4]), complex_tail]),
    ]);
    let valid = sequence(vec![
        optional(alternative(vec![kinzi.clone(), category(&[18])])),
        alternative(vec![consonant, category(&[2, 10, 11])]),
        optional(category(&[40])),
        tail.clone(),
    ]);
    let broken = sequence(vec![optional(kinzi), optional(category(&[40])), tail]);
    recognize(input, &[(0, valid), (2, joiner), (1, broken)], 2)
}

pub(super) fn myanmar_category(character: u32) -> u8 {
    match character {
        0x1004 | 0x101B | 0x105A => 15,
        0x1000..=0x1020
        | 0x103F
        | 0x104E
        | 0x1050..=0x1051
        | 0x105B..=0x105D
        | 0x1061
        | 0x1065..=0x1066
        | 0x106E..=0x1070
        | 0x1075..=0x1081
        | 0x108E
        | 0xA9E0..=0xA9E4
        | 0xA9E7..=0xA9EF
        | 0xA9FA..=0xA9FE
        | 0xAA60..=0xAA6F
        | 0xAA71..=0xAA73
        | 0xAA7A
        | 0xAA7E..=0xAA7F => 1,
        0x1021..=0x102A | 0x1052..=0x1055 => 2,
        0x1037 | 0xAA7C..=0xAA7D => 3,
        0x1039 => 4,
        0x200C => 5,
        0x200D => 6,
        0x1038 | 0x1087..=0x108D | 0x108F | 0x109A..=0x109C => 8,
        0x1032 | 0x1036 => 9,
        0x1040..=0x104B | 0x1090..=0x1099 | 0xA9F0..=0xA9F9 | 0xAA74..=0xAA76 => 10,
        0x25CC => 11,
        0x102D..=0x102E | 0x1033..=0x1035 | 0x1071..=0x1074 | 0x1085..=0x1086 | 0x109D | 0xA9E5 => {
            20
        }
        0x102F..=0x1030 | 0x1058..=0x1059 => 21,
        0x1031 | 0x1084 => 22,
        0x102B..=0x102C | 0x1056..=0x1057 | 0x1062 | 0x1067..=0x1068 | 0x1083 => 23,
        0x103A => 32,
        0x103E => 35,
        0x103C => 36,
        0x103D | 0x1082 => 37,
        0x103B | 0x105E..=0x105F => 38,
        0x1063..=0x1064 | 0x1069..=0x106D | 0xAA7B => 39,
        0xFE00..=0xFE0F => 40,
        0x1060 => 41,
        _ => 0,
    }
}

pub(super) fn khmer_category(character: u32) -> u8 {
    match character {
        0x179A => 15,
        0x1780..=0x17A2 => 1,
        0x17A3..=0x17B3 => 2,
        0x17D2 => 4,
        0x200C => 5,
        0x200D => 6,
        0x17D9 => 10,
        0x25CC => 11,
        0x17B7..=0x17BA | 0x17BE => 20,
        0x17BB..=0x17BD => 21,
        0x17C1..=0x17C3 => 22,
        0x17B6 | 0x17BF..=0x17C0 | 0x17C4..=0x17C5 => 23,
        0x17C9..=0x17CA | 0x17CC => 25,
        0x17C6 | 0x17CB | 0x17CD..=0x17D1 => 26,
        0x17C7..=0x17C8 | 0x17D3 | 0x17DD => 27,
        _ => 0,
    }
}
