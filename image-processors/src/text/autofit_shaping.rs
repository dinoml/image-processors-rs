//! Font-unit OpenType operations for automatic-hinter metric samples.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    autofit_layout::{coverage, offset, tags},
    i16_at, invalid, u16_at, u32_at, Result, TrueTypeFont,
};

#[path = "autofit_syllables.rs"]
mod syllables;

#[path = "autofit_arabic.rs"]
mod arabic;

#[path = "autofit_indic.rs"]
mod indic;

const KHMER_BASIC: &[[u8; 4]] = &[
    *b"locl", *b"ccmp", *b"pref", *b"blwf", *b"abvf", *b"pstf", *b"cfar",
];
const PRESENTATION_FEATURES: &[[u8; 4]] = &[*b"pres", *b"abvs", *b"blws", *b"psts"];
const MYANMAR_BASIC: &[[u8; 4]] = &[*b"rphf", *b"pref", *b"blwf", *b"pstf"];
const USE_PRE: &[[u8; 4]] = &[*b"locl", *b"ccmp", *b"nukt", *b"akhn"];
const USE_BASIC: &[[u8; 4]] = &[
    *b"rkrf", *b"abvf", *b"blwf", *b"half", *b"pstf", *b"vatu", *b"cjct",
];
const USE_FORMS: &[[u8; 4]] = &[*b"isol", *b"init", *b"medi", *b"fina"];
const USE_FINAL: &[[u8; 4]] = &[*b"abvs", *b"blws", *b"haln", *b"pres", *b"psts"];
const USE_SCRIPTS: &[&str] = &[
    "tibt", "mong", "sinh", "buhd", "limb", "sylo", "tfng", "nkoo", "kali", "saur", "sund", "mtei",
    "medf", "tavt", "cakm", "adlm", "rohg",
];
const INDIC_SCRIPTS: &[&str] = &[
    "beng", "deva", "gujr", "guru", "knda", "mlym", "orya", "taml", "telu",
];
const ARABIC_FORMS: &[([u8; 4], u64)] = &[
    (*b"isol", 4),
    (*b"fina", 32),
    (*b"fin2", 64),
    (*b"fin3", 128),
    (*b"medi", 16),
    (*b"med2", 256),
    (*b"init", 8),
];
const RANDOM_MASK: u64 = 255 << 16;
const PRELIMINARY: &[[u8; 4]] = &[
    *b"ltra", *b"ltrm", *b"rtla", *b"rtlm", *b"frac", *b"numr", *b"dnom", *b"rand", *b"trak",
    *b"Harf", *b"HARF",
];

#[derive(Clone, Default)]
pub(super) struct Element {
    pub id: usize,
    pub advance: i64,
    pub y_offset: i64,
    x_offset: i64,
    slot: usize,
    y_advance: i64,
    attachment: Option<(usize, bool)>,
    ligature: usize,
    component: usize,
    components: usize,
    multiple: usize,
    multiplied: bool,
    stretch: u8,
    class: u16,
    ligated: bool,
    unicode_mark: bool,
    nonspacing: bool,
    combining: u8,
    character: u32,
    mask: u64,
    syllable: u8,
    category: u8,
    position: u8,
    substituted: bool,
}

#[derive(Clone, Copy)]
struct LookupRule {
    mask: u64,
    auto_zwj: bool,
    auto_zwnj: bool,
    per_syllable: bool,
    random: bool,
}

impl Default for LookupRule {
    fn default() -> Self {
        Self {
            mask: 1,
            auto_zwj: true,
            auto_zwnj: true,
            per_syllable: false,
            random: false,
        }
    }
}

#[derive(Clone, Copy)]
enum Match {
    Any,
    Glyph(usize),
    Class(usize, u16),
    Coverage(usize),
}

pub(super) const FEATURES: &[[u8; 4]] = &[
    *b"c2cp", *b"c2sc", *b"ordn", *b"pcap", *b"sinf", *b"smcp", *b"subs", *b"sups", *b"titl",
];

fn unsupported() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "native automatic-hinter shaping operation is not implemented",
    )
}

fn class(data: &[u8], table: usize, glyph: usize) -> Result<u16> {
    match u16_at(data, table)? {
        1 => {
            let start = usize::from(u16_at(data, table + 2)?);
            let count = usize::from(u16_at(data, table + 4)?);
            if glyph < start || glyph - start >= count {
                Ok(0)
            } else {
                u16_at(data, table + 6 + 2 * (glyph - start))
            }
        }
        2 => {
            for index in 0..usize::from(u16_at(data, table + 2)?) {
                let record = table + 4 + 6 * index;
                if glyph >= usize::from(u16_at(data, record)?)
                    && glyph <= usize::from(u16_at(data, record + 2)?)
                {
                    return u16_at(data, record + 4);
                }
            }
            Ok(0)
        }
        _ => Err(invalid()),
    }
}

fn covered(data: &[u8], table: usize, glyph: usize) -> Result<Option<usize>> {
    Ok(coverage(data, table)?
        .iter()
        .position(|&value| usize::from(value) == glyph))
}

fn language(data: &[u8], script: &str) -> Result<Option<usize>> {
    let list = usize::from(u16_at(data, 4)?);
    let mut language = None;
    // Shape with the selected script's default language, unlike coverage,
    // which collects every language.  DFLT and latn are shaping fallbacks.
    let mut candidates = tags(script, false);
    candidates.extend([*b"DFLT", *b"latn"]);
    for wanted in candidates {
        for index in 0..usize::from(u16_at(data, list)?) {
            let record = list + 2 + 6 * index;
            if data.get(record..record + 4) != Some(wanted.as_slice()) {
                continue;
            }
            let table = offset(data, list, record + 4)?;
            let default = u16_at(data, table)?;
            language = (default != 0).then_some(table + usize::from(default));
            // A selected script without a default LangSys has no features.
            if language.is_none() {
                return Ok(None);
            }
            break;
        }
        if language.is_some() {
            break;
        }
    }
    Ok(language)
}

fn has_feature(font: &TrueTypeFont, script: &str, wanted: [u8; 4]) -> Result<bool> {
    for tag in [b"GSUB", b"GPOS"] {
        if !font.tables.contains_key(tag) {
            continue;
        }
        let data = font.table(tag)?;
        let Some(language) = language(data, script)? else {
            continue;
        };
        let features = usize::from(u16_at(data, 6)?);
        for index in 0..usize::from(u16_at(data, language + 4)?) {
            let feature = usize::from(u16_at(data, language + 6 + 2 * index)?);
            if feature >= usize::from(u16_at(data, features)?) {
                return Err(invalid());
            }
            let record = features + 2 + 6 * feature;
            if data.get(record..record + 4) == Some(wanted.as_slice()) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn feature_masks(
    font: &TrueTypeFont,
    script: &str,
    features: &[[u8; 4]],
    local: &[([u8; 4], u64)],
) -> Result<[u32; 24]> {
    // Keep semantic masks on glyphs while reproducing the map builder's
    // lexicographic allocation for AlternateSubst's numeric feature values.
    // The reference reserves bits 0..=2 for glyph flags, starts local feature
    // values at bit 4, and uses the high bit for single-valued global features.
    let mut mapping = [0; 24];
    mapping[0] = 1 << 31;
    let mut next = 4;
    for tag in features.iter().copied().collect::<BTreeSet<_>>() {
        let (mask, bits) = if tag == *b"rand" {
            (RANDOM_MASK, 8)
        } else if let Some(&(_, mask)) = local.iter().find(|&&(feature, _)| feature == tag) {
            (mask, 1)
        } else {
            continue;
        };
        if !has_feature(font, script, tag)? {
            continue;
        }
        if next + bits >= 31 {
            continue;
        }
        if mask != 0 {
            let shift = mask.trailing_zeros() as usize;
            for bit in 0..bits {
                mapping[shift + bit] = 1u32 << (next + bit);
            }
        }
        next += bits;
    }
    Ok(mapping)
}

fn plan(
    data: &[u8],
    script: &str,
    enabled: &[[u8; 4]],
    stage_features: &[[u8; 4]],
    all_features: &[[u8; 4]],
    first_stage: bool,
    rules: &[([u8; 4], LookupRule)],
) -> Result<BTreeMap<usize, LookupRule>> {
    let features = usize::from(u16_at(data, 6)?);
    let Some(language) = language(data, script)? else {
        return Ok(BTreeMap::new());
    };
    let required = usize::from(u16_at(data, language + 2)?);
    let mut selected = vec![required];
    for index in 0..usize::from(u16_at(data, language + 4)?) {
        selected.push(usize::from(u16_at(data, language + 6 + 2 * index)?));
    }
    selected.sort_by_key(|&feature| {
        let record = features + 2 + 6 * feature;
        (
            feature != required,
            data.get(record..record + 4).unwrap_or(&[]),
        )
    });
    let mut lookups: BTreeMap<usize, LookupRule> = BTreeMap::new();
    for feature in selected {
        if feature >= usize::from(u16_at(data, features)?) {
            continue;
        }
        let record = features + 2 + 6 * feature;
        let tag: [u8; 4] = data
            .get(record..record + 4)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?;
        if feature == required {
            if !(stage_features.contains(&tag) || first_stage && !all_features.contains(&tag)) {
                continue;
            }
        } else if !enabled.contains(&tag) {
            continue;
        }
        let mut rule = rules
            .iter()
            .find(|(feature, _)| *feature == tag)
            .map(|(_, rule)| *rule)
            .unwrap_or_default();
        if tag == *b"rand" {
            rule.mask = RANDOM_MASK;
            rule.random = true;
        }
        if matches!(&tag, b"mark" | b"mkmk") {
            rule.auto_zwj = false;
            rule.auto_zwnj = false;
        }
        if feature == required {
            rule = LookupRule::default();
        }
        let table = offset(data, features, record + 4)?;
        for index in 0..usize::from(u16_at(data, table + 2)?) {
            let lookup = usize::from(u16_at(data, table + 4 + 2 * index)?);
            lookups
                .entry(lookup)
                .and_modify(|existing| {
                    existing.mask |= rule.mask;
                    existing.auto_zwj &= rule.auto_zwj;
                    existing.auto_zwnj &= rule.auto_zwnj;
                })
                .or_insert(rule);
        }
    }
    Ok(lookups)
}

fn chosen_script(font: &TrueTypeFont, script: &str) -> Result<Option<[u8; 4]>> {
    if !font.tables.contains_key(b"GSUB") {
        return Ok(None);
    }
    let data = font.table(b"GSUB")?;
    let list = usize::from(u16_at(data, 4)?);
    let mut candidates = tags(script, false);
    candidates.extend([*b"DFLT", *b"latn"]);
    for wanted in candidates {
        for index in 0..usize::from(u16_at(data, list)?) {
            let record = list + 2 + 6 * index;
            if data.get(record..record + 4) == Some(wanted.as_slice()) {
                return Ok(Some(wanted));
            }
        }
    }
    Ok(None)
}

fn joining_type(character: u32) -> u8 {
    // U/L/R/D/T properties needed by the fixed automatic-hinter samples,
    // including their Mongolian ZWJ endpoints and Arabic normalization.
    match character {
        0x10D00 => 1,
        0x625 | 0x627 | 0x10D22 => 2,
        0x62A..=0x62B
        | 0x62D
        | 0x637..=0x638
        | 0x640
        | 0x643..=0x644
        | 0x7CA..=0x7EA
        | 0x1807
        | 0x180A
        | 0x1820..=0x1878
        | 0x1887..=0x18A8
        | 0x18AA
        | 0x200D
        | 0x10D01..=0x10D21
        | 0x10D23
        | 0x1E900..=0x1E943 => 3,
        0x655
        | 0x7EB..=0x7F3
        | 0x180B..=0x180D
        | 0x180F
        | 0x1885..=0x1886
        | 0x18A9
        | 0x1E944..=0x1E94B => 4,
        _ => 0,
    }
}

fn joining_masks(glyphs: &mut [Element]) {
    let mut previous: Option<(usize, u8)> = None;
    for index in 0..glyphs.len() {
        let kind = joining_type(glyphs[index].character);
        if kind == 4 {
            continue;
        }
        if kind == 0 {
            previous = None;
            continue;
        }
        glyphs[index].mask |= 4;
        if let Some((position, prior)) = previous {
            if matches!(prior, 1 | 3) && matches!(kind, 2 | 3) {
                let form = if glyphs[position].mask & 32 != 0 {
                    16
                } else {
                    8
                };
                glyphs[position].mask = (glyphs[position].mask & !60) | form;
                glyphs[index].mask = (glyphs[index].mask & !60) | 32;
            }
        }
        previous = Some((index, kind));
    }
}

struct Engine<'a> {
    font: &'a TrueTypeFont,
    data: &'a [u8],
    positioning: bool,
    coverage_query: bool,
    zero_context: bool,
    mask_map: [u32; 24],
    random_state: u32,
    rtl: bool,
    next_ligature: usize,
    next_slot: usize,
    rule: LookupRule,
    glyphs: Vec<Element>,
}

impl Engine<'_> {
    fn probe(
        &self,
        script: &str,
        tag: [u8; 4],
        ids: &[usize],
        zero_context: bool,
        all: &[[u8; 4]],
    ) -> Result<bool> {
        if self.data.is_empty() || ids.is_empty() {
            return Ok(false);
        }
        for lookup in plan(self.data, script, &[tag], &[tag], all, false, &[])?.into_keys() {
            let glyphs = ids
                .iter()
                .enumerate()
                .map(|(slot, &id)| Element {
                    id,
                    slot,
                    mask: 1,
                    class: 1,
                    ..Element::default()
                })
                .collect::<Vec<_>>();
            let mut probe = Engine {
                font: self.font,
                data: self.data,
                positioning: false,
                coverage_query: true,
                zero_context,
                mask_map: self.mask_map,
                random_state: 1,
                rtl: false,
                next_ligature: 1,
                next_slot: glyphs.len(),
                rule: LookupRule::default(),
                glyphs,
            };
            if probe.lookup(lookup, 0, 64)?.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn arabic(&mut self, enabled: &[[u8; 4]], common: &[[u8; 4]], all: &[[u8; 4]]) -> Result<()> {
        let manual = LookupRule {
            auto_zwj: false,
            ..LookupRule::default()
        };
        let first_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| PRELIMINARY.contains(tag))
            .chain([*b"stch"])
            .collect::<Vec<_>>();
        let first_features = PRELIMINARY
            .iter()
            .copied()
            .chain([*b"stch"])
            .collect::<Vec<_>>();
        if !self.data.is_empty() {
            self.apply(plan(
                self.data,
                "arab",
                &first_enabled,
                &first_features,
                all,
                false,
                &[],
            )?)?;
        }
        if has_feature(self.font, "arab", *b"stch")? {
            for glyph in &mut self.glyphs {
                if glyph.multiplied {
                    let component = if glyph.ligature == 0 {
                        glyph.multiple
                    } else {
                        glyph.component
                    };
                    glyph.stretch = if component % 2 == 0 { 1 } else { 2 };
                }
            }
        }
        let initial = [*b"ccmp", *b"locl"];
        let initial_rules = initial.map(|tag| (tag, manual));
        if !self.data.is_empty() {
            self.apply(plan(
                self.data,
                "arab",
                &initial,
                &initial,
                all,
                false,
                &initial_rules,
            )?)?;
            for &(tag, mask) in ARABIC_FORMS {
                self.apply(plan(
                    self.data,
                    "arab",
                    &[tag],
                    &[tag],
                    all,
                    false,
                    &[(tag, LookupRule { mask, ..manual })],
                )?)?;
            }
            self.apply(plan(
                self.data,
                "arab",
                &[*b"rlig"],
                &[*b"rlig"],
                all,
                false,
                &[(*b"rlig", manual)],
            )?)?;
        }
        let mut fallback = true;
        for &tag in USE_FORMS {
            fallback &= !has_feature(self.font, "arab", tag)?;
        }
        if fallback {
            self.fallback()?;
        }
        if self.data.is_empty() {
            return Ok(());
        }
        let separate_calt = !has_feature(self.font, "arab", *b"rclt")?;
        if separate_calt {
            self.apply(plan(
                self.data,
                "arab",
                &[*b"calt"],
                &[*b"calt"],
                all,
                false,
                &[(*b"calt", manual)],
            )?)?;
        }
        let mut earlier = initial.to_vec();
        earlier.extend(ARABIC_FORMS.iter().map(|&(tag, _)| tag));
        earlier.extend([*b"stch", *b"rlig"]);
        earlier.extend_from_slice(PRELIMINARY);
        if separate_calt {
            earlier.push(*b"calt");
        }
        let final_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain([*b"mset"])
            .collect::<Vec<_>>();
        let final_features = common
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain([*b"mset"])
            .collect::<Vec<_>>();
        let rules = [*b"calt", *b"rclt", *b"liga", *b"clig", *b"mset"].map(|tag| (tag, manual));
        self.apply(plan(
            self.data,
            "arab",
            &final_enabled,
            &final_features,
            all,
            false,
            &rules,
        )?)
    }

    fn insert_dotted_circles(&mut self, broken: u8, category: u8, repha: Option<u8>) -> Result<()> {
        let dotted = self.font.glyph_index('\u{25CC}')?;
        if dotted == 0 {
            return Ok(());
        }
        let mut start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let mut end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            if label & 15 == broken {
                let circle = Element {
                    id: dotted,
                    character: 0x25CC,
                    category,
                    position: 14,
                    class: 1,
                    mask: self.glyphs[start].mask,
                    syllable: label,
                    slot: self.next_slot,
                    ..Element::default()
                };
                self.next_slot += 1;
                let position = (start..end)
                    .find(|&index| Some(self.glyphs[index].category) != repha)
                    .unwrap_or(end);
                self.glyphs.insert(position, circle);
                end += 1;
            }
            start = end;
        }
        Ok(())
    }

    fn universal(
        &mut self,
        script: &str,
        enabled: &[[u8; 4]],
        common: &[[u8; 4]],
        all: &[[u8; 4]],
    ) -> Result<()> {
        let labels = syllables::universal(
            &self
                .glyphs
                .iter()
                .map(|glyph| glyph.category)
                .collect::<Vec<_>>(),
            &self
                .glyphs
                .iter()
                .map(|glyph| glyph.unicode_mark)
                .collect::<Vec<_>>(),
        );
        for (glyph, label) in self.glyphs.iter_mut().zip(labels) {
            glyph.syllable = label;
        }
        let arabic_joining = matches!(script, "mong" | "adlm" | "nkoo" | "rohg");
        let mut start = 0;
        let mut prior: Option<(usize, u64)> = None;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            let prefix = if self.glyphs[start].category == 18 {
                1
            } else {
                3.min(end - start)
            };
            for glyph in &mut self.glyphs[start..start + prefix] {
                glyph.mask |= 2;
            }
            if !arabic_joining {
                if matches!(label & 15, 6 | 8) {
                    prior = None;
                } else {
                    let form = if let Some((previous, form)) = prior {
                        let replacement = if form == 32 { 16 } else { 8 };
                        for glyph in &mut self.glyphs[previous..start] {
                            glyph.mask = (glyph.mask & !60) | replacement;
                        }
                        32
                    } else {
                        4
                    };
                    for glyph in &mut self.glyphs[start..end] {
                        glyph.mask = (glyph.mask & !60) | form;
                    }
                    prior = Some((start, form));
                }
            }
            start = end;
        }
        let pre_rules = USE_PRE
            .iter()
            .map(|&tag| {
                (
                    tag,
                    LookupRule {
                        auto_zwj: tag != *b"akhn",
                        per_syllable: true,
                        ..LookupRule::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        self.apply(plan(
            self.data, script, USE_PRE, USE_PRE, all, false, &pre_rules,
        )?)?;
        for glyph in &mut self.glyphs {
            glyph.substituted = false;
        }
        let per_syllable = LookupRule {
            auto_zwj: false,
            per_syllable: true,
            ..LookupRule::default()
        };
        self.apply(plan(
            self.data,
            script,
            &[*b"rphf"],
            &[*b"rphf"],
            all,
            false,
            &[(
                *b"rphf",
                LookupRule {
                    mask: 2,
                    ..per_syllable
                },
            )],
        )?)?;
        let mut previous = None;
        let mut found = false;
        let mut prefix = true;
        for glyph in &mut self.glyphs {
            if previous != Some(glyph.syllable) {
                previous = Some(glyph.syllable);
                found = false;
                prefix = true;
            }
            prefix &= glyph.mask & 2 != 0;
            if prefix && !found && glyph.substituted {
                glyph.category = 18;
                found = true;
            }
            glyph.substituted = false;
        }
        self.apply(plan(
            self.data,
            script,
            &[*b"pref"],
            &[*b"pref"],
            all,
            false,
            &[(*b"pref", per_syllable)],
        )?)?;
        previous = None;
        found = false;
        for glyph in &mut self.glyphs {
            if previous != Some(glyph.syllable) {
                previous = Some(glyph.syllable);
                found = false;
            }
            if !found && glyph.substituted {
                glyph.category = 22;
                found = true;
            }
        }
        let basic_rules = USE_BASIC
            .iter()
            .map(|&tag| (tag, per_syllable))
            .collect::<Vec<_>>();
        self.apply(plan(
            self.data,
            script,
            USE_BASIC,
            USE_BASIC,
            all,
            false,
            &basic_rules,
        )?)?;
        self.insert_dotted_circles(7, 1, Some(18))?;
        start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            if matches!(label & 15, 0 | 1 | 2 | 5 | 7) {
                if self.glyphs[start].category == 18 && end - start > 1 {
                    let target = (start + 1..end)
                        .find(|&index| {
                            matches!(self.glyphs[index].category,
                        22..=30 | 33..=35 | 37..=39 | 45..=47)
                                || (!self.glyphs[index].ligated
                                    && matches!(self.glyphs[index].category, 12 | 44 | 53))
                        })
                        .unwrap_or(end);
                    self.glyphs[start..target].rotate_left(1);
                }
                let mut target = start;
                for index in start..end {
                    let glyph = &self.glyphs[index];
                    if !glyph.ligated && matches!(glyph.category, 12 | 44 | 53) {
                        target = index + 1;
                    } else if matches!(glyph.category, 22 | 23)
                        && (if glyph.ligature == 0 {
                            glyph.multiple
                        } else {
                            glyph.component
                        }) == 0
                        && target < index
                    {
                        self.glyphs[target..index + 1].rotate_right(1);
                    }
                }
            }
            start = end;
        }
        for glyph in &mut self.glyphs {
            glyph.syllable = 0;
        }
        let form_rules = USE_FORMS
            .iter()
            .enumerate()
            .map(|(index, &tag)| {
                (
                    tag,
                    LookupRule {
                        mask: 4 << index,
                        ..LookupRule::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        self.apply(plan(
            self.data,
            script,
            USE_FORMS,
            USE_FORMS,
            all,
            false,
            &form_rules,
        )?)?;
        let earlier = USE_PRE
            .iter()
            .chain(USE_BASIC)
            .chain(USE_FORMS)
            .copied()
            .chain([*b"rphf", *b"pref"])
            .collect::<Vec<_>>();
        let final_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(USE_FINAL.iter().copied())
            .collect::<Vec<_>>();
        let final_features = common
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(USE_FINAL.iter().copied())
            .collect::<Vec<_>>();
        let final_rules = USE_FINAL
            .iter()
            .map(|&tag| {
                (
                    tag,
                    LookupRule {
                        auto_zwj: false,
                        ..LookupRule::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        self.apply(plan(
            self.data,
            script,
            &final_enabled,
            &final_features,
            all,
            false,
            &final_rules,
        )?)
    }

    fn myanmar(
        &mut self,
        script: &str,
        enabled: &[[u8; 4]],
        common: &[[u8; 4]],
        all: &[[u8; 4]],
    ) -> Result<()> {
        let labels = syllables::myanmar(
            &self
                .glyphs
                .iter()
                .map(|glyph| glyph.category)
                .collect::<Vec<_>>(),
        );
        for (glyph, label) in self.glyphs.iter_mut().zip(labels) {
            glyph.syllable = label;
        }
        let initial = [*b"locl", *b"ccmp"];
        let rules = initial.map(|tag| {
            (
                tag,
                LookupRule {
                    per_syllable: true,
                    ..LookupRule::default()
                },
            )
        });
        self.apply(plan(
            self.data, script, &initial, &initial, all, false, &rules,
        )?)?;
        self.insert_dotted_circles(1, 11, None)?;
        let mut start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            if label & 15 <= 1 {
                let reph = start + 3 <= end
                    && self.glyphs[start].category == 15
                    && self.glyphs[start + 1].category == 32
                    && self.glyphs[start + 2].category == 4;
                let limit = start + if reph { 3 } else { 0 };
                let base = (limit..end)
                    .find(|&index| {
                        !self.glyphs[index].ligated
                            && matches!(self.glyphs[index].category, 1 | 2 | 10 | 11 | 15 | 18)
                    })
                    .unwrap_or(if reph { start } else { limit });
                let mut index = start;
                while index < start + (if reph { 3 } else { 0 }) {
                    self.glyphs[index].position = 5;
                    index += 1;
                }
                while index < base {
                    self.glyphs[index].position = 3;
                    index += 1;
                }
                if index < end {
                    self.glyphs[index].position = 4;
                    index += 1;
                }
                let mut position = 5;
                while index < end {
                    let category = self.glyphs[index].category;
                    self.glyphs[index].position = match category {
                        36 => 3,
                        22 => 2,
                        40 => self.glyphs[index - 1].position,
                        21 if position == 5 || position == 8 => {
                            position = 8;
                            position
                        }
                        9 if position == 8 => 7,
                        _ => {
                            if position == 8 {
                                position = 9;
                            }
                            position
                        }
                    };
                    index += 1;
                }
                self.glyphs[start..end].sort_by_key(|glyph| glyph.position);
                let first = (start..end).find(|&index| self.glyphs[index].position == 2);
                let last = (start..end).rfind(|&index| self.glyphs[index].position == 2);
                if let (Some(first), Some(last)) = (first, last) {
                    if first < last {
                        self.glyphs[first..last + 1].reverse();
                        let mut group = first;
                        for index in first..=last {
                            if self.glyphs[index].category == 22 {
                                self.glyphs[group..index + 1].reverse();
                                group = index + 1;
                            }
                        }
                    }
                }
            }
            start = end;
        }
        let rule = LookupRule {
            auto_zwj: false,
            per_syllable: true,
            ..LookupRule::default()
        };
        for &tag in MYANMAR_BASIC {
            self.apply(plan(
                self.data,
                script,
                &[tag],
                &[tag],
                all,
                false,
                &[(tag, rule)],
            )?)?;
        }
        for glyph in &mut self.glyphs {
            glyph.syllable = 0;
        }
        let earlier = initial
            .into_iter()
            .chain(MYANMAR_BASIC.iter().copied())
            .collect::<Vec<_>>();
        let final_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(PRESENTATION_FEATURES.iter().copied())
            .collect::<Vec<_>>();
        let final_features = common
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(PRESENTATION_FEATURES.iter().copied())
            .collect::<Vec<_>>();
        let rules = PRESENTATION_FEATURES
            .iter()
            .map(|&tag| {
                (
                    tag,
                    LookupRule {
                        auto_zwj: false,
                        ..LookupRule::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        self.apply(plan(
            self.data,
            script,
            &final_enabled,
            &final_features,
            all,
            false,
            &rules,
        )?)
    }

    fn khmer(
        &mut self,
        script: &str,
        enabled: &[[u8; 4]],
        common: &[[u8; 4]],
        all: &[[u8; 4]],
    ) -> Result<()> {
        let labels = syllables::khmer(
            &self
                .glyphs
                .iter()
                .map(|glyph| glyph.category)
                .collect::<Vec<_>>(),
        );
        for (glyph, label) in self.glyphs.iter_mut().zip(labels) {
            glyph.syllable = label;
        }
        self.insert_dotted_circles(1, 11, None)?;
        let mut start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            if label & 15 <= 1 {
                for glyph in &mut self.glyphs[start + 1..end] {
                    glyph.mask |= 4 | 8 | 16;
                }
                let mut coengs = 0;
                for index in start + 1..end {
                    if self.glyphs[index].category == 4 && coengs <= 2 && index + 1 < end {
                        coengs += 1;
                        if self.glyphs[index + 1].category == 15 {
                            self.glyphs[index].mask |= 2;
                            self.glyphs[index + 1].mask |= 2;
                            self.glyphs[start..index + 2].rotate_right(2);
                            for glyph in &mut self.glyphs[index + 2..end] {
                                glyph.mask |= 32;
                            }
                            coengs = 2;
                        }
                    } else if self.glyphs[index].category == 22 {
                        self.glyphs[start..index + 1].rotate_right(1);
                    }
                }
            }
            start = end;
        }
        let rules = KHMER_BASIC
            .iter()
            .enumerate()
            .map(|(index, &tag)| {
                (
                    tag,
                    LookupRule {
                        mask: if index < 2 { 1 } else { 1 << (index - 1) },
                        auto_zwj: index < 2,
                        auto_zwnj: index < 2,
                        per_syllable: true,
                        random: false,
                    },
                )
            })
            .collect::<Vec<_>>();
        if !self.data.is_empty() {
            self.apply(plan(
                self.data,
                script,
                KHMER_BASIC,
                KHMER_BASIC,
                all,
                false,
                &rules,
            )?)?;
        }
        for glyph in &mut self.glyphs {
            glyph.syllable = 0;
        }
        let final_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| !KHMER_BASIC.contains(tag))
            .chain(PRESENTATION_FEATURES.iter().copied())
            .collect::<Vec<_>>();
        let final_features = common
            .iter()
            .copied()
            .filter(|tag| !KHMER_BASIC.contains(tag))
            .chain(PRESENTATION_FEATURES.iter().copied())
            .collect::<Vec<_>>();
        let final_rules = PRESENTATION_FEATURES
            .iter()
            .map(|&tag| {
                (
                    tag,
                    LookupRule {
                        auto_zwj: false,
                        auto_zwnj: false,
                        ..LookupRule::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        if !self.data.is_empty() {
            self.apply(plan(
                self.data,
                script,
                &final_enabled,
                &final_features,
                all,
                false,
                &final_rules,
            )?)?;
        }
        Ok(())
    }

    fn glyph_class(&self, position: usize) -> Result<u16> {
        let glyph = &self.glyphs[position];
        if !self.font.tables.contains_key(b"GDEF") {
            return Ok(glyph.class);
        }
        let data = self.font.table(b"GDEF")?;
        let definition = u16_at(data, 4)?;
        if definition == 0 {
            Ok(glyph.class)
        } else {
            class(data, usize::from(definition), glyph.id)
        }
    }

    fn component_count(&self, position: usize) -> Result<usize> {
        if self.glyphs[position].ligated && self.glyph_class(position)? == 2 {
            Ok(self.glyphs[position].components & 15)
        } else {
            Ok(1)
        }
    }

    fn extents(&self, glyph: usize) -> Result<[i64; 4]> {
        if glyph >= self.font.glyphs {
            return Err(invalid());
        }
        let loca = self.font.table(b"loca")?;
        let location = |index| -> Result<usize> {
            if self.font.long_loca {
                usize::try_from(u32_at(loca, 4 * index)?).map_err(|_| invalid())
            } else {
                Ok(2 * usize::from(u16_at(loca, 2 * index)?))
            }
        };
        let start = location(glyph)?;
        let end = location(glyph + 1)?;
        if start == end {
            return Ok([0; 4]);
        }
        let data = self
            .font
            .table(b"glyf")?
            .get(start..end)
            .ok_or_else(invalid)?;
        let left = i64::from(i16_at(data, 2)?);
        let bottom = i64::from(i16_at(data, 4)?);
        let right = i64::from(i16_at(data, 6)?);
        let top = i64::from(i16_at(data, 8)?);
        Ok([left, top, right - left, bottom - top])
    }

    fn fallback_metric_positions(&mut self) -> Result<()> {
        // The hinter consumes advances and vertical offsets, never horizontal
        // pen positions.  Preserve those parts of fallback mark placement,
        // including class/component resets and the reference below-mark gap.
        let gap = i64::from(u16_at(self.font.table(b"head")?, 18)?) / 16;
        let mut base = 0;
        while base < self.glyphs.len() {
            if self.glyphs[base].unicode_mark {
                base += 1;
                continue;
            }
            let end = (base + 1..self.glyphs.len())
                .find(|&index| !self.glyphs[index].unicode_mark)
                .unwrap_or(self.glyphs.len());
            let mut original = self.extents(self.glyphs[base].id)?;
            original[1] += self.glyphs[base].y_offset;
            let mut extents = original;
            let count = self.component_count(base)?;
            let mut component = -1;
            let mut combining = 255;
            let mut offset = if self.rtl {
                0
            } else {
                -self.glyphs[base].y_advance
            };
            for index in base + 1..end {
                let order = match self.glyphs[index].combining {
                    22 => 220,
                    26 => 230,
                    order => order,
                };
                if order == 0 {
                    offset += if self.rtl {
                        self.glyphs[index].y_advance
                    } else {
                        -self.glyphs[index].y_advance
                    };
                    continue;
                }
                if count > 1 {
                    let parent = &self.glyphs[base];
                    let mark = &self.glyphs[index];
                    let next = if parent.ligature == 0
                        || parent.ligature != mark.ligature
                        || mark.component > count
                    {
                        count as isize - 1
                    } else {
                        mark.component as isize - 1
                    };
                    if component != next {
                        component = next;
                        combining = 255;
                    }
                }
                if combining != order {
                    combining = order;
                    extents = original;
                }
                let mark = self.extents(self.glyphs[index].id)?;
                self.glyphs[index].y_offset = 0;
                if order == 220 {
                    extents[3] -= gap;
                    let mut value = extents[1] + extents[3] - mark[1];
                    if (gap > 0) == (value > 0) {
                        extents[3] -= value;
                        value = 0;
                    }
                    extents[3] += mark[3];
                    self.glyphs[index].y_offset = value;
                }
                self.glyphs[index].y_offset += offset;
                self.glyphs[index].advance = 0;
                self.glyphs[index].y_advance = 0;
            }
            base = end;
        }
        Ok(())
    }

    fn ignored(&self, position: usize, flags: u16, mark_set: Option<usize>) -> Result<bool> {
        let kind = self.glyph_class(position)?;
        if (kind == 1 && flags & 2 != 0)
            || (kind == 2 && flags & 4 != 0)
            || (kind == 3 && flags & 8 != 0)
        {
            return Ok(true);
        }
        if kind != 3 {
            return Ok(false);
        }
        if !self.font.tables.contains_key(b"GDEF") {
            return Ok(mark_set.is_some() || flags & 0xff00 != 0);
        }
        let data = self.font.table(b"GDEF")?;
        let glyph = self.glyphs[position].id;
        if let Some(mark_set) = mark_set {
            if u16_at(data, 2)? < 2 {
                return Err(invalid());
            }
            let sets = usize::from(u16_at(data, 12)?);
            if sets == 0
                || u16_at(data, sets)? != 1
                || mark_set >= usize::from(u16_at(data, sets + 2)?)
            {
                return Err(invalid());
            }
            let table = sets
                .checked_add(
                    usize::try_from(u32_at(data, sets + 4 + 4 * mark_set)?)
                        .map_err(|_| invalid())?,
                )
                .ok_or_else(invalid)?;
            return Ok(covered(data, table, glyph)?.is_none());
        }
        let attachment = flags >> 8;
        if attachment != 0 {
            let definition = u16_at(data, 10)?;
            return Ok(
                definition == 0 || class(data, usize::from(definition), glyph)? != attachment
            );
        }
        Ok(false)
    }

    fn positions(
        &self,
        start: usize,
        flags: u16,
        mark_set: Option<usize>,
        backward: bool,
    ) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        if backward {
            for index in (0..start).rev() {
                if !self.ignored(index, flags, mark_set)? {
                    result.push(index);
                }
            }
        } else {
            for index in start..self.glyphs.len() {
                if !self.ignored(index, flags, mark_set)? {
                    result.push(index);
                }
            }
        }
        Ok(result)
    }

    fn match_sequence(
        &self,
        start: usize,
        backward: bool,
        (flags, mark_set): (u16, Option<usize>),
        expected: &[Match],
        context: bool,
        syllable: u8,
    ) -> Result<Option<Vec<usize>>> {
        let mut cursor = start;
        let mut result = Vec::with_capacity(expected.len());
        for wanted in expected {
            loop {
                let position = if backward {
                    let Some(previous) = cursor.checked_sub(1) else {
                        return Ok(None);
                    };
                    cursor = previous;
                    previous
                } else {
                    if cursor >= self.glyphs.len() {
                        return Ok(None);
                    }
                    let position = cursor;
                    cursor += 1;
                    position
                };
                if self.ignored(position, flags, mark_set)? {
                    continue;
                }
                let glyph = &self.glyphs[position];
                let automatic = !glyph.substituted
                    && match glyph.character {
                        0x200C => self.positioning || (context && self.rule.auto_zwnj),
                        0x200D => context || self.rule.auto_zwj,
                        _ => false,
                    };
                let eligible = context
                    || ((glyph.mask & self.rule.mask != 0)
                        && (self.positioning
                            || !self.rule.per_syllable
                            || glyph.syllable == syllable));
                let matches = eligible
                    && match *wanted {
                        Match::Any => !automatic,
                        Match::Glyph(id) => glyph.id == id,
                        Match::Class(table, value) => class(self.data, table, glyph.id)? == value,
                        Match::Coverage(table) => covered(self.data, table, glyph.id)?.is_some(),
                    };
                if matches {
                    result.push(position);
                    break;
                }
                if !automatic {
                    return Ok(None);
                }
            }
        }
        Ok(Some(result))
    }

    fn lookup(&mut self, index: usize, at: usize, depth: usize) -> Result<Option<usize>> {
        if depth == 0 || at >= self.glyphs.len() {
            return Ok(None);
        }
        if !self.coverage_query && self.glyphs[at].mask & self.rule.mask == 0 {
            return Ok(None);
        }
        let list = usize::from(u16_at(self.data, 8)?);
        if index >= usize::from(u16_at(self.data, list)?) {
            return Ok(None);
        }
        let table = offset(self.data, list, list + 2 + 2 * index)?;
        let kind = u16_at(self.data, table)?;
        let flags = if self.coverage_query {
            0
        } else {
            u16_at(self.data, table + 2)?
        };
        let count = usize::from(u16_at(self.data, table + 4)?);
        let mark_set = if flags & 16 != 0 {
            Some(usize::from(u16_at(self.data, table + 6 + 2 * count)?))
        } else {
            None
        };
        if self.ignored(at, flags, mark_set)? {
            return Ok(None);
        }
        for index in 0..count {
            if let Some(next) = self.subtable(
                kind,
                offset(self.data, table, table + 6 + 2 * index)?,
                at,
                flags,
                mark_set,
                depth,
            )? {
                return Ok(Some(next));
            }
        }
        Ok(None)
    }

    fn context(
        &mut self,
        table: usize,
        at: usize,
        flags: u16,
        mark_set: Option<usize>,
        chained: bool,
        depth: usize,
    ) -> Result<Option<usize>> {
        let data = self.data;
        let format = u16_at(data, table)?;
        let mut rules = Vec::new();
        if format == 3 {
            rules.push(table + 2);
        } else {
            let Some(coverage_index) =
                covered(data, offset(data, table, table + 2)?, self.glyphs[at].id)?
            else {
                return Ok(None);
            };
            let (index, sets) = match (chained, format) {
                (_, 1) => (coverage_index, table + 4),
                (false, 2) => (
                    usize::from(class(
                        data,
                        offset(data, table, table + 4)?,
                        self.glyphs[at].id,
                    )?),
                    table + 6,
                ),
                (true, 2) => (
                    usize::from(class(
                        data,
                        offset(data, table, table + 6)?,
                        self.glyphs[at].id,
                    )?),
                    table + 10,
                ),
                _ => return Err(invalid()),
            };
            if index >= usize::from(u16_at(data, sets)?) {
                return Ok(None);
            }
            let relative = u16_at(data, sets + 2 + 2 * index)?;
            if relative == 0 {
                return Ok(None);
            }
            let set = table + usize::from(relative);
            for index in 0..usize::from(u16_at(data, set)?) {
                rules.push(offset(data, set, set + 2 + 2 * index)?);
            }
        }
        for rule in rules {
            let mut field = rule;
            let syllable = self.glyphs[at].syllable;
            let expected = |value: u16, definition: usize| -> Result<Match> {
                Ok(match format {
                    1 => Match::Glyph(usize::from(value)),
                    2 => Match::Class(offset(data, table, table + definition)?, value),
                    3 => Match::Coverage(table + usize::from(value)),
                    _ => return Err(invalid()),
                })
            };
            if chained {
                let count = usize::from(u16_at(data, field)?);
                field += 2;
                let wanted = (0..count)
                    .map(|index| expected(u16_at(data, field + 2 * index)?, 4))
                    .collect::<Result<Vec<_>>>()?;
                field += 2 * count;
                if self.coverage_query {
                    if self.zero_context && count != 0 {
                        continue;
                    }
                } else if self
                    .match_sequence(at, true, (flags, mark_set), &wanted, true, syllable)?
                    .is_none()
                {
                    continue;
                }
            }
            let count = usize::from(u16_at(data, field)?);
            field += 2;
            if count == 0 {
                continue;
            }
            if self.coverage_query && count != self.glyphs.len() {
                continue;
            }
            let early_count = if !chained {
                let count = usize::from(u16_at(data, field)?);
                field += 2;
                Some(count)
            } else {
                None
            };
            let first = usize::from(format != 3);
            if first == 0
                && covered(data, offset(data, table, field)?, self.glyphs[at].id)?.is_none()
            {
                continue;
            }
            let wanted = (1..count)
                .map(|index| {
                    expected(
                        u16_at(data, field + 2 * (index - first))?,
                        if chained { 6 } else { 4 },
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let Some(mut inputs) =
                self.match_sequence(at + 1, false, (flags, mark_set), &wanted, false, syllable)?
            else {
                continue;
            };
            inputs.insert(0, at);
            field += 2 * (count - first);
            if chained {
                let after_count = usize::from(u16_at(data, field)?);
                field += 2;
                let wanted = (0..after_count)
                    .map(|index| expected(u16_at(data, field + 2 * index)?, 8))
                    .collect::<Result<Vec<_>>>()?;
                field += 2 * after_count;
                if self.coverage_query {
                    if self.zero_context && after_count != 0 {
                        continue;
                    }
                } else if self
                    .match_sequence(
                        inputs[count - 1] + 1,
                        false,
                        (flags, mark_set),
                        &wanted,
                        true,
                        syllable,
                    )?
                    .is_none()
                {
                    continue;
                }
            }
            if self.coverage_query {
                return Ok(Some(at + count));
            }
            let record_count = if let Some(count) = early_count {
                count
            } else {
                let count = usize::from(u16_at(data, field)?);
                field += 2;
                count
            };
            if !self.compatible(&inputs, flags, mark_set)? {
                continue;
            }
            let slots = inputs[..count]
                .iter()
                .map(|&position| self.glyphs[position].slot)
                .collect::<Vec<_>>();
            let boundary = self
                .glyphs
                .get(inputs[count - 1] + 1)
                .map(|glyph| glyph.slot);
            for record in 0..record_count {
                let sequence = usize::from(u16_at(data, field + 4 * record)?);
                let Some(&slot) = slots.get(sequence) else {
                    continue;
                };
                let Some(position) = self.glyphs.iter().position(|glyph| glyph.slot == slot) else {
                    continue;
                };
                self.lookup(
                    usize::from(u16_at(data, field + 4 * record + 2)?),
                    position,
                    depth - 1,
                )?;
            }
            return Ok(Some(
                boundary
                    .and_then(|slot| self.glyphs.iter().position(|glyph| glyph.slot == slot))
                    .unwrap_or(self.glyphs.len()),
            ));
        }
        Ok(None)
    }

    fn value(&mut self, mut field: usize, format: u16, at: usize) -> Result<usize> {
        for bit in 0..16 {
            if format & (1 << bit) == 0 {
                continue;
            }
            let value = i64::from(i16_at(self.data, field)?);
            match bit {
                0 => self.glyphs[at].x_offset += value,
                1 => self.glyphs[at].y_offset += value,
                2 => self.glyphs[at].advance += value,
                3 => self.glyphs[at].y_advance += value,
                // Device tables are inactive for a font-unit shaping font.
                4..=7 => {}
                _ => return Err(invalid()),
            }
            field += 2;
        }
        Ok(field)
    }

    fn compatible(&self, positions: &[usize], flags: u16, mark_set: Option<usize>) -> Result<bool> {
        let Some(&first_position) = positions.first() else {
            return Ok(false);
        };
        let first = &self.glyphs[first_position];
        for &position in &positions[1..] {
            let glyph = &self.glyphs[position];
            if first.ligature != 0 && first.component != 0 {
                if glyph.ligature == first.ligature && glyph.component == first.component {
                    continue;
                }
                let mut base = None;
                for position in (0..first_position).rev() {
                    let glyph = &self.glyphs[position];
                    if glyph.ligature != first.ligature {
                        break;
                    }
                    if glyph.component == 0 {
                        base = Some(position);
                        break;
                    }
                }
                let Some(base) = base else {
                    return Ok(false);
                };
                if !self.ignored(base, flags, mark_set)? {
                    return Ok(false);
                }
            } else if glyph.ligature != 0
                && glyph.component != 0
                && glyph.ligature != first.ligature
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn ligate(
        &mut self,
        at: usize,
        positions: &[usize],
        id: usize,
        flags: u16,
        mark_set: Option<usize>,
    ) -> Result<Option<usize>> {
        if positions.is_empty() || !self.compatible(positions, flags, mark_set)? {
            return Ok(None);
        }
        let count = positions.len();
        if count == 1 {
            self.glyphs[at].id = id;
            self.glyphs[at].substituted = true;
            return Ok(Some(at + 1));
        }
        let original = positions
            .iter()
            .map(|&position| self.glyphs[position].clone())
            .collect::<Vec<_>>();
        let counts = positions
            .iter()
            .map(|&position| self.component_count(position))
            .collect::<Result<Vec<_>>>()?;
        let all_marks = positions[1..]
            .iter()
            .map(|&position| self.glyph_class(position))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .all(|&class| class == 3);
        let first_class = self.glyph_class(at)?;
        let base_ligature = first_class == 1 && all_marks;
        let mark_ligature = first_class == 3 && all_marks;
        let real_ligature = !base_ligature && !mark_ligature;
        let ligature_id = if real_ligature {
            let id = self.next_ligature;
            self.next_ligature = if id == 7 { 1 } else { id + 1 };
            id
        } else {
            0
        };
        let components = counts.iter().sum::<usize>();
        let mut accumulated = counts[0];
        for index in 1..count {
            let previous = counts[index - 1];
            if real_ligature {
                for position in positions[index - 1] + 1..positions[index] {
                    let old = self.glyphs[position].component;
                    self.glyphs[position].ligature = ligature_id;
                    self.glyphs[position].component = (accumulated - previous
                        + if old == 0 {
                            previous
                        } else {
                            old.min(previous)
                        })
                        & 15;
                }
            }
            accumulated += counts[index];
        }
        let last = &original[count - 1];
        if !mark_ligature && last.ligature != 0 {
            for position in positions[count - 1] + 1..self.glyphs.len() {
                let glyph = &mut self.glyphs[position];
                if glyph.ligature != last.ligature || glyph.component == 0 {
                    break;
                }
                glyph.ligature = ligature_id;
                glyph.component =
                    (accumulated - counts[count - 1] + glyph.component.min(counts[count - 1])) & 15;
            }
        }
        self.glyphs[at].id = id;
        self.glyphs[at].substituted = true;
        self.glyphs[at].multiple = 0;
        self.glyphs[at].multiplied = false;
        self.glyphs[at].ligated = true;
        if real_ligature {
            self.glyphs[at].class = 2;
            if self.glyphs[at].nonspacing {
                self.glyphs[at].unicode_mark = false;
                self.glyphs[at].nonspacing = false;
            }
            self.glyphs[at].ligature = ligature_id;
            self.glyphs[at].component = 0;
            self.glyphs[at].components = components & 15;
        }
        for &position in positions[1..].iter().rev() {
            self.glyphs.remove(position);
        }
        Ok(Some(at + 1))
    }

    fn fallback(&mut self) -> Result<()> {
        for lookup in arabic::fallback(self.font)? {
            self.rule = LookupRule {
                mask: lookup.mask,
                ..LookupRule::default()
            };
            let mut at = 0;
            while at < self.glyphs.len() {
                if self.glyphs[at].mask & lookup.mask == 0
                    || self.ignored(at, lookup.flags, None)?
                {
                    at += 1;
                    continue;
                }
                match &lookup.substitution {
                    arabic::Substitution::Single(replacements) => {
                        if let Some(&(_, id)) = replacements
                            .iter()
                            .find(|&&(from, _)| from == self.glyphs[at].id)
                        {
                            self.glyphs[at].id = id;
                            self.glyphs[at].substituted = true;
                        }
                    }
                    arabic::Substitution::Ligature(groups) => {
                        if let Some((_, replacements)) =
                            groups.iter().find(|(id, _)| *id == self.glyphs[at].id)
                        {
                            for (components, id) in replacements {
                                let wanted = components
                                    .iter()
                                    .copied()
                                    .map(Match::Glyph)
                                    .collect::<Vec<_>>();
                                let Some(mut positions) = self.match_sequence(
                                    at + 1,
                                    false,
                                    (lookup.flags, None),
                                    &wanted,
                                    false,
                                    0,
                                )?
                                else {
                                    continue;
                                };
                                positions.insert(0, at);
                                if self
                                    .ligate(at, &positions, *id, lookup.flags, None)?
                                    .is_some()
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                at += 1;
            }
        }
        Ok(())
    }

    fn anchor(&self, base: usize, field: usize) -> Result<Option<[i64; 2]>> {
        let relative = u16_at(self.data, field)?;
        if relative == 0 {
            return Ok(None);
        }
        let table = base + usize::from(relative);
        if !matches!(u16_at(self.data, table)?, 1..=3) {
            return Err(invalid());
        }
        // The automatic hinter creates an OT font with zero ppem and shapes
        // at UPEM.  Contour-point and device adjustments are therefore inactive.
        Ok(Some([
            i64::from(i16_at(self.data, table + 2)?),
            i64::from(i16_at(self.data, table + 4)?),
        ]))
    }

    fn reverse_cursive(&mut self, child: usize, parent: usize, depth: usize) {
        if depth == 0 {
            return;
        }
        let Some((previous, false)) = self.glyphs[child].attachment else {
            return;
        };
        self.glyphs[child].attachment = None;
        if previous == parent {
            return;
        }
        self.reverse_cursive(previous, parent, depth - 1);
        self.glyphs[previous].y_offset = -self.glyphs[child].y_offset;
        self.glyphs[previous].attachment = Some((child, false));
    }

    fn finish_attachment(&mut self, at: usize, depth: usize) {
        let Some((parent, mark)) = self.glyphs[at].attachment.take() else {
            return;
        };
        if depth == 0 || parent >= self.glyphs.len() {
            return;
        }
        self.finish_attachment(parent, depth - 1);
        let parent_y = self.glyphs[parent].y_offset;
        self.glyphs[at].y_offset += parent_y;
        if mark {
            let parent_x = self.glyphs[parent].x_offset;
            self.glyphs[at].x_offset += parent_x;
            if parent < at {
                let range = if self.rtl {
                    parent + 1..at + 1
                } else {
                    parent..at
                };
                let sign = if self.rtl { 1 } else { -1 };
                for position in range {
                    let advance = self.glyphs[position].advance;
                    let y_advance = self.glyphs[position].y_advance;
                    self.glyphs[at].x_offset += sign * advance;
                    self.glyphs[at].y_offset += sign * y_advance;
                }
            }
        }
    }

    fn mark(
        &mut self,
        kind: u16,
        table: usize,
        index: usize,
        at: usize,
        flags: u16,
        mark_set: Option<usize>,
    ) -> Result<Option<usize>> {
        let data = self.data;
        let positions = self.positions(
            at,
            if kind == 6 { flags & !14 } else { 8 },
            if kind == 6 { mark_set } else { None },
            true,
        )?;
        let parent = if kind == 4 {
            positions
                .into_iter()
                .find(|&position| self.glyphs[position].multiple == 0)
        } else {
            positions.first().copied()
        };
        let Some(parent) = parent else {
            return Ok(None);
        };
        if kind == 6 {
            if self.glyph_class(parent)? != 3 {
                return Ok(None);
            }
            let first = &self.glyphs[at];
            let second = &self.glyphs[parent];
            let compatible = if first.ligature == second.ligature {
                first.ligature == 0 || first.component == second.component
            } else {
                (first.ligature != 0 && first.component == 0)
                    || (second.ligature != 0 && second.component == 0)
            };
            if !compatible {
                return Ok(None);
            }
        }
        let Some(parent_index) = covered(
            data,
            offset(data, table, table + 4)?,
            self.glyphs[parent].id,
        )?
        else {
            return Ok(None);
        };
        let classes = usize::from(u16_at(data, table + 6)?);
        let marks = offset(data, table, table + 8)?;
        if index >= usize::from(u16_at(data, marks)?) {
            return Err(invalid());
        }
        let mark_class = usize::from(u16_at(data, marks + 2 + 4 * index)?);
        if mark_class >= classes {
            return Err(invalid());
        }
        let Some(mark_anchor) = self.anchor(marks, marks + 4 + 4 * index)? else {
            return Ok(None);
        };
        let array = offset(data, table, table + 10)?;
        if parent_index >= usize::from(u16_at(data, array)?) {
            return Err(invalid());
        }
        let (base, row) = if kind == 5 {
            let attach = offset(data, array, array + 2 + 2 * parent_index)?;
            let count = usize::from(u16_at(data, attach)?);
            if count == 0 {
                return Ok(None);
            }
            let mark = &self.glyphs[at];
            let ligature = &self.glyphs[parent];
            let component = if ligature.ligature != 0
                && ligature.ligature == mark.ligature
                && mark.component != 0
            {
                mark.component.min(count) - 1
            } else {
                count - 1
            };
            (attach, component)
        } else {
            (array, parent_index)
        };
        let Some(base_anchor) = self.anchor(base, base + 2 + 2 * (row * classes + mark_class))?
        else {
            return Ok(None);
        };
        self.glyphs[at].x_offset = base_anchor[0] - mark_anchor[0];
        self.glyphs[at].y_offset = base_anchor[1] - mark_anchor[1];
        self.glyphs[at].attachment = Some((parent, true));
        Ok(Some(at + 1))
    }

    fn subtable(
        &mut self,
        kind: u16,
        table: usize,
        at: usize,
        flags: u16,
        mark_set: Option<usize>,
        depth: usize,
    ) -> Result<Option<usize>> {
        let data = self.data;
        let format = u16_at(data, table)?;
        if kind == if self.positioning { 9 } else { 7 } {
            if format != 1 || depth == 0 {
                return Err(invalid());
            }
            let target = table
                .checked_add(usize::try_from(u32_at(data, table + 4)?).map_err(|_| invalid())?)
                .ok_or_else(invalid)?;
            return self.subtable(
                u16_at(data, table + 2)?,
                target,
                at,
                flags,
                mark_set,
                depth - 1,
            );
        }
        if (!self.positioning && (kind == 5 || kind == 6))
            || (self.positioning && (kind == 7 || kind == 8))
        {
            return self.context(
                table,
                at,
                flags,
                mark_set,
                kind == if self.positioning { 8 } else { 6 },
                depth,
            );
        }
        if self.coverage_query && matches!(kind, 1..=3 | 8) && self.glyphs.len() != 1 {
            return Ok(None);
        }
        let Some(index) = covered(data, offset(data, table, table + 2)?, self.glyphs[at].id)?
        else {
            return Ok(None);
        };
        if self.coverage_query && matches!(kind, 1..=3) {
            return Ok(Some(at + 1));
        }
        if self.positioning {
            match kind {
                1 => {
                    let value_format = u16_at(data, table + 4)?;
                    let size = 2 * value_format.count_ones() as usize;
                    let field = match format {
                        1 => table + 6,
                        2 if index < usize::from(u16_at(data, table + 6)?) => {
                            table + 8 + size * index
                        }
                        _ => return Err(invalid()),
                    };
                    self.value(field, value_format, at)?;
                    return Ok(Some(at + 1));
                }
                2 => {
                    let Some(positions) = self.match_sequence(
                        at + 1,
                        false,
                        (flags, mark_set),
                        &[Match::Any],
                        false,
                        self.glyphs[at].syllable,
                    )?
                    else {
                        return Ok(None);
                    };
                    let next = positions[0];
                    let first_format = u16_at(data, table + 4)?;
                    let second_format = u16_at(data, table + 6)?;
                    let size =
                        2 * (first_format.count_ones() + second_format.count_ones()) as usize;
                    let field = match format {
                        1 => {
                            if index >= usize::from(u16_at(data, table + 8)?) {
                                return Err(invalid());
                            }
                            let set = offset(data, table, table + 10 + 2 * index)?;
                            let mut matched = None;
                            for index in 0..usize::from(u16_at(data, set)?) {
                                let field = set + 2 + (size + 2) * index;
                                if usize::from(u16_at(data, field)?) == self.glyphs[next].id {
                                    matched = Some(field + 2);
                                    break;
                                }
                            }
                            let Some(field) = matched else {
                                return Ok(None);
                            };
                            field
                        }
                        2 => {
                            let first = usize::from(class(
                                data,
                                offset(data, table, table + 8)?,
                                self.glyphs[at].id,
                            )?);
                            let second = usize::from(class(
                                data,
                                offset(data, table, table + 10)?,
                                self.glyphs[next].id,
                            )?);
                            let columns = usize::from(u16_at(data, table + 14)?);
                            if first >= usize::from(u16_at(data, table + 12)?) || second >= columns
                            {
                                return Err(invalid());
                            }
                            table + 16 + size * (first * columns + second)
                        }
                        _ => return Err(invalid()),
                    };
                    let field = self.value(field, first_format, at)?;
                    self.value(field, second_format, next)?;
                    return Ok(Some(if second_format == 0 { next } else { next + 1 }));
                }
                3 => {
                    if format != 1 || index >= usize::from(u16_at(data, table + 4)?) {
                        return Err(invalid());
                    }
                    let Some(entry) = self.anchor(table, table + 6 + 4 * index)? else {
                        return Ok(None);
                    };
                    let Some(positions) = self.match_sequence(
                        at,
                        true,
                        (flags, mark_set),
                        &[Match::Any],
                        false,
                        self.glyphs[at].syllable,
                    )?
                    else {
                        return Ok(None);
                    };
                    let previous = positions[0];
                    let Some(previous_index) = covered(
                        data,
                        offset(data, table, table + 2)?,
                        self.glyphs[previous].id,
                    )?
                    else {
                        return Ok(None);
                    };
                    if previous_index >= usize::from(u16_at(data, table + 4)?) {
                        return Err(invalid());
                    }
                    let Some(exit) = self.anchor(table, table + 8 + 4 * previous_index)? else {
                        return Ok(None);
                    };
                    if self.rtl {
                        let adjustment = exit[0] + self.glyphs[previous].x_offset;
                        self.glyphs[previous].advance -= adjustment;
                        self.glyphs[previous].x_offset -= adjustment;
                        self.glyphs[at].advance = entry[0] + self.glyphs[at].x_offset;
                    } else {
                        self.glyphs[previous].advance = exit[0] + self.glyphs[previous].x_offset;
                        let adjustment = entry[0] + self.glyphs[at].x_offset;
                        self.glyphs[at].advance -= adjustment;
                        self.glyphs[at].x_offset -= adjustment;
                    }
                    let (child, parent, y_offset) = if flags & 1 != 0 {
                        (previous, at, entry[1] - exit[1])
                    } else {
                        (at, previous, exit[1] - entry[1])
                    };
                    self.reverse_cursive(child, parent, 64);
                    self.glyphs[child].attachment = Some((parent, false));
                    self.glyphs[child].y_offset = y_offset;
                    if self.glyphs[parent]
                        .attachment
                        .is_some_and(|(target, _)| target == child)
                    {
                        self.glyphs[parent].attachment = None;
                        self.glyphs[parent].y_offset = 0;
                    }
                    return Ok(Some(at + 1));
                }
                4..=6 => {
                    if format != 1 {
                        return Err(invalid());
                    }
                    return self.mark(kind, table, index, at, flags, mark_set);
                }
                _ => return Err(unsupported()),
            }
        }
        match kind {
            1 => {
                self.glyphs[at].substituted = true;
                self.glyphs[at].id = match format {
                    1 => usize::from(
                        (self.glyphs[at].id as u16).wrapping_add(u16_at(data, table + 4)?),
                    ),
                    2 if index < usize::from(u16_at(data, table + 4)?) => {
                        usize::from(u16_at(data, table + 6 + 2 * index)?)
                    }
                    _ => return Err(invalid()),
                };
                Ok(Some(at + 1))
            }
            2 | 3 => {
                if format != 1 || index >= usize::from(u16_at(data, table + 4)?) {
                    return Err(invalid());
                }
                let sequence = offset(data, table, table + 6 + 2 * index)?;
                let count = usize::from(u16_at(data, sequence)?);
                if kind == 3 {
                    if count == 0 {
                        return Ok(None);
                    }
                    let mapped = |mask: u64| {
                        self.mask_map
                            .iter()
                            .enumerate()
                            .fold(0u32, |value, (bit, &mapped)| {
                                if mask & (1u64 << bit) != 0 {
                                    value | mapped
                                } else {
                                    value
                                }
                            })
                    };
                    let lookup_mask = mapped(self.rule.mask);
                    if lookup_mask == 0 {
                        return Ok(None);
                    }
                    let mut alternate = (mapped(self.glyphs[at].mask) & lookup_mask)
                        >> lookup_mask.trailing_zeros();
                    if alternate == 255 && self.rule.random {
                        self.random_state = self.random_state.wrapping_mul(48271) % 2147483647;
                        alternate = self.random_state % count as u32 + 1;
                    }
                    if alternate == 0 || alternate as usize > count {
                        return Ok(None);
                    }
                    self.glyphs[at].id =
                        usize::from(u16_at(data, sequence + 2 * alternate as usize)?);
                    self.glyphs[at].substituted = true;
                    return Ok(Some(at + 1));
                }
                if count == 1 {
                    self.glyphs[at].id = usize::from(u16_at(data, sequence + 2)?);
                    self.glyphs[at].substituted = true;
                    return Ok(Some(at + 1));
                }
                let original = self.glyphs[at].clone();
                let guess = if self.glyph_class(at)? == 2 {
                    1
                } else {
                    original.class
                };
                let mut replacement = Vec::with_capacity(count);
                for index in 0..count {
                    let mut glyph = original.clone();
                    if index != 0 {
                        glyph.slot = self.next_slot;
                        self.next_slot += 1;
                    }
                    glyph.id = usize::from(u16_at(data, sequence + 2 + 2 * index)?);
                    glyph.class = guess;
                    glyph.substituted = true;
                    glyph.multiplied = true;
                    glyph.multiple = if original.ligature == 0 {
                        index & 15
                    } else {
                        original.component
                    };
                    replacement.push(glyph);
                }
                self.glyphs.splice(at..at + 1, replacement);
                Ok(Some(at + count))
            }
            4 => {
                if format != 1 || index >= usize::from(u16_at(data, table + 4)?) {
                    return Err(invalid());
                }
                let set = offset(data, table, table + 6 + 2 * index)?;
                for index in 0..usize::from(u16_at(data, set)?) {
                    let ligature = offset(data, set, set + 2 + 2 * index)?;
                    let count = usize::from(u16_at(data, ligature + 2)?);
                    if count == 0 {
                        continue;
                    }
                    if self.coverage_query && count != self.glyphs.len() {
                        continue;
                    }
                    let wanted = (1..count)
                        .map(|index| {
                            Ok(Match::Glyph(usize::from(u16_at(
                                data,
                                ligature + 2 + 2 * index,
                            )?)))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let Some(mut positions) = self.match_sequence(
                        at + 1,
                        false,
                        (flags, mark_set),
                        &wanted,
                        false,
                        self.glyphs[at].syllable,
                    )?
                    else {
                        continue;
                    };
                    positions.insert(0, at);
                    if let Some(next) = self.ligate(
                        at,
                        &positions,
                        usize::from(u16_at(data, ligature)?),
                        flags,
                        mark_set,
                    )? {
                        return Ok(Some(next));
                    }
                }
                Ok(None)
            }
            8 => {
                if format != 1 {
                    return Err(invalid());
                }
                let mut field = table + 4;
                for backward in [true, false] {
                    let count = usize::from(u16_at(data, field)?);
                    field += 2;
                    let wanted = (0..count)
                        .map(|index| Ok(Match::Coverage(offset(data, table, field + 2 * index)?)))
                        .collect::<Result<Vec<_>>>()?;
                    if self.coverage_query {
                        if self.zero_context && count != 0 {
                            return Ok(None);
                        }
                    } else if self
                        .match_sequence(
                            if backward { at } else { at + 1 },
                            backward,
                            (flags, mark_set),
                            &wanted,
                            true,
                            self.glyphs[at].syllable,
                        )?
                        .is_none()
                    {
                        return Ok(None);
                    }
                    field += 2 * count;
                }
                if index >= usize::from(u16_at(data, field)?) {
                    return Err(invalid());
                }
                self.glyphs[at].id = usize::from(u16_at(data, field + 2 + 2 * index)?);
                self.glyphs[at].substituted = true;
                Ok(Some(at + 1))
            }
            _ => Err(invalid()),
        }
    }

    fn apply(&mut self, lookups: BTreeMap<usize, LookupRule>) -> Result<()> {
        for (lookup, rule) in lookups {
            self.rule = rule;
            let list = usize::from(u16_at(self.data, 8)?);
            if lookup >= usize::from(u16_at(self.data, list)?) {
                continue;
            }
            let table = offset(self.data, list, list + 2 + 2 * lookup)?;
            let mut kind = u16_at(self.data, table)?;
            if kind == 7 && !self.positioning && u16_at(self.data, table + 4)? != 0 {
                kind = u16_at(self.data, offset(self.data, table, table + 6)? + 2)?;
            }
            if !self.positioning && kind == 8 {
                for position in (0..self.glyphs.len()).rev() {
                    self.lookup(lookup, position, 64)?;
                }
            } else {
                let mut position = 0;
                let mut operations = 0usize;
                while position < self.glyphs.len() {
                    operations += 1;
                    if operations > 1_000_000 {
                        return Err(invalid());
                    }
                    position = self.lookup(lookup, position, 64)?.unwrap_or(position + 1);
                }
            }
        }
        Ok(())
    }
}

pub(super) fn cluster(
    font: &TrueTypeFont,
    text: &str,
    feature: Option<[u8; 4]>,
) -> Result<Vec<Element>> {
    let mut glyphs = Vec::new();
    let characters = super::autofit_unicode::normalize(font, text)?;
    let script = characters
        .iter()
        .find_map(|&character| super::autofit_unicode::script(character))
        .unwrap_or("DFLT");
    let chosen = chosen_script(font, script)?;
    let myanmar = script == "mymr" && chosen.is_some_and(|tag| tag == *b"mym2");
    let universal = chosen.is_some_and(|tag| {
        tag != *b"DFLT"
            && tag != *b"latn"
            && (USE_SCRIPTS.contains(&script)
                || (INDIC_SCRIPTS.contains(&script) && tag[3] == b'3'))
    });
    let indic = chosen.is_some_and(|tag| {
        tag != *b"DFLT" && tag != *b"latn" && INDIC_SCRIPTS.contains(&script) && tag[3] != b'3'
    });
    for (slot, &character) in characters.iter().enumerate() {
        let unicode_mark = super::autofit_unicode::mark(character);
        let (category, position) = if indic {
            syllables::indic_properties(character as u32)
        } else {
            (
                if script == "khmr" {
                    syllables::khmer_category(character as u32)
                } else if myanmar {
                    syllables::myanmar_category(character as u32)
                } else if universal {
                    syllables::universal_category(character as u32, script)
                } else {
                    0
                },
                0,
            )
        };
        glyphs.push(Element {
            id: font.glyph_index(character)?,
            slot,
            character: character as u32,
            mask: 1 | RANDOM_MASK,
            category,
            position,
            class: if unicode_mark { 3 } else { 1 },
            unicode_mark,
            nonspacing: super::autofit_unicode::nonspacing(character),
            combining: if script == "arab" && character == '\u{655}' {
                22
            } else {
                super::autofit_unicode::combining(character)
            },
            ..Element::default()
        });
    }
    if script == "arab" || (universal && matches!(script, "mong" | "adlm" | "nkoo" | "rohg")) {
        joining_masks(&mut glyphs);
    }
    let rtl = matches!(
        script,
        "adlm" | "arab" | "avst" | "cprt" | "hebr" | "nkoo" | "orkh" | "rohg"
    );
    let mut enabled = vec![
        *b"ccmp", *b"locl", *b"rlig", *b"calt", *b"clig", *b"liga", *b"rclt", *b"abvm", *b"blwm",
        *b"mark", *b"mkmk", *b"curs", *b"dist", *b"kern", *b"rand", *b"trak", *b"Harf", *b"HARF",
        *b"Buzz", *b"BUZZ",
    ];
    if rtl {
        enabled.push(*b"rtla");
    } else {
        enabled.extend([*b"ltra", *b"ltrm"]);
    }
    if script == "khmr" || indic {
        enabled.retain(|tag| tag != b"liga");
    }
    if let Some(feature) = feature {
        enabled.push(feature);
    }
    let mut common_features = enabled.clone();
    common_features.extend([*b"frac", *b"numr", *b"dnom", *b"rand", *b"trak"]);
    if rtl {
        common_features.push(*b"rtlm");
    }
    let mut all_features = common_features.clone();
    all_features.push(*b"rvrn");
    if script == "khmr" {
        all_features.extend_from_slice(KHMER_BASIC);
        all_features.extend_from_slice(PRESENTATION_FEATURES);
    }
    if myanmar {
        all_features.extend_from_slice(MYANMAR_BASIC);
        all_features.extend_from_slice(PRESENTATION_FEATURES);
    }
    if universal {
        all_features.extend_from_slice(USE_PRE);
        all_features.extend_from_slice(USE_BASIC);
        all_features.extend_from_slice(USE_FORMS);
        all_features.extend_from_slice(USE_FINAL);
        all_features.extend([*b"rphf", *b"pref"]);
    }
    if script == "arab" {
        all_features.extend(ARABIC_FORMS.iter().map(|&(tag, _)| tag));
        all_features.extend([*b"stch", *b"mset"]);
    }
    if indic {
        all_features.extend(indic::BASIC.iter().chain(indic::FINAL).map(|&(tag, _)| tag));
    }
    let mut local_masks = vec![(*b"frac", 0), (*b"numr", 0), (*b"dnom", 0), (*b"rtlm", 0)];
    if script == "arab" {
        local_masks.extend_from_slice(ARABIC_FORMS);
    } else if indic {
        local_masks.extend(
            indic::BASIC
                .iter()
                .chain(indic::FINAL)
                .copied()
                .filter(|&(_, mask)| mask != 1),
        );
    } else if universal {
        local_masks.push((*b"rphf", 2));
        local_masks.extend(
            USE_FORMS
                .iter()
                .enumerate()
                .map(|(index, &tag)| (tag, 4 << index)),
        );
    } else if script == "khmr" {
        local_masks.extend(
            KHMER_BASIC
                .iter()
                .enumerate()
                .skip(2)
                .map(|(index, &tag)| (tag, 1 << (index - 1))),
        );
    }
    let mask_map = feature_masks(font, script, &all_features, &local_masks)?;
    let late_enabled = enabled
        .iter()
        .copied()
        .filter(|tag| !PRELIMINARY.contains(tag))
        .collect::<Vec<_>>();
    let late_features = common_features
        .iter()
        .copied()
        .filter(|tag| !PRELIMINARY.contains(tag))
        .collect::<Vec<_>>();
    if font.tables.contains_key(b"GSUB") {
        let data = font.table(b"GSUB")?;
        let mut engine = Engine {
            font,
            data,
            positioning: false,
            coverage_query: false,
            zero_context: false,
            mask_map,
            random_state: 1,
            rtl,
            next_ligature: 1,
            next_slot: glyphs.len(),
            rule: LookupRule::default(),
            glyphs,
        };
        engine.apply(plan(
            data,
            script,
            &[*b"rvrn"],
            &[*b"rvrn"],
            &all_features,
            true,
            &[],
        )?)?;
        if script == "khmr" || myanmar || universal || indic {
            let preliminary = enabled
                .iter()
                .copied()
                .filter(|tag| PRELIMINARY.contains(tag))
                .collect::<Vec<_>>();
            engine.apply(plan(
                data,
                script,
                &preliminary,
                PRELIMINARY,
                &all_features,
                false,
                &[],
            )?)?;
        }
        if script == "arab" {
            engine.arabic(&enabled, &common_features, &all_features)?;
        } else if script == "khmr" {
            engine.khmer(script, &late_enabled, &late_features, &all_features)?;
        } else if myanmar {
            engine.myanmar(script, &late_enabled, &late_features, &all_features)?;
        } else if universal {
            engine.universal(script, &late_enabled, &late_features, &all_features)?;
        } else if indic {
            engine.indic(script, &late_enabled, &late_features, &all_features)?;
        } else {
            engine.apply(plan(
                data,
                script,
                &enabled,
                &common_features,
                &all_features,
                false,
                &[],
            )?)?;
        }
        glyphs = engine.glyphs;
    } else if script == "khmr" || script == "arab" {
        let mut engine = Engine {
            font,
            data: &[],
            positioning: false,
            coverage_query: false,
            zero_context: false,
            mask_map,
            random_state: 1,
            rtl,
            next_ligature: 1,
            next_slot: glyphs.len(),
            rule: LookupRule::default(),
            glyphs,
        };
        if script == "arab" {
            engine.arabic(&enabled, &common_features, &all_features)?;
        } else {
            engine.khmer(script, &late_enabled, &late_features, &all_features)?;
        }
        glyphs = engine.glyphs;
    }
    let hmtx = font.table(b"hmtx")?;
    for glyph in &mut glyphs {
        if glyph.id >= font.glyphs || font.metrics == 0 {
            return Err(invalid());
        }
        glyph.advance = i64::from(u16_at(hmtx, 4 * glyph.id.min(font.metrics - 1))?);
    }
    enabled.push(*b"rvrn");
    let mut positioning_rules = Vec::new();
    if script == "arab" {
        enabled.extend(ARABIC_FORMS.iter().map(|&(tag, _)| tag));
        enabled.extend([*b"stch", *b"mset"]);
        positioning_rules.extend(ARABIC_FORMS.iter().map(|&(tag, mask)| {
            (
                tag,
                LookupRule {
                    mask,
                    auto_zwj: false,
                    ..LookupRule::default()
                },
            )
        }));
        positioning_rules.extend(
            [
                *b"ccmp", *b"locl", *b"rlig", *b"calt", *b"rclt", *b"liga", *b"clig", *b"mset",
            ]
            .map(|tag| {
                (
                    tag,
                    LookupRule {
                        auto_zwj: false,
                        ..LookupRule::default()
                    },
                )
            }),
        );
    }
    if indic {
        enabled.extend(indic::BASIC.iter().chain(indic::FINAL).map(|&(tag, _)| tag));
        positioning_rules.extend(indic::BASIC.iter().chain(indic::FINAL).map(|&(tag, mask)| {
            (
                tag,
                LookupRule {
                    mask,
                    auto_zwj: false,
                    auto_zwnj: false,
                    per_syllable: false,
                    random: false,
                },
            )
        }));
    }
    if script == "khmr" {
        enabled.extend_from_slice(KHMER_BASIC);
        enabled.extend_from_slice(PRESENTATION_FEATURES);
        positioning_rules.extend(KHMER_BASIC.iter().enumerate().map(|(index, &tag)| {
            (
                tag,
                LookupRule {
                    mask: if index < 2 { 1 } else { 1 << (index - 1) },
                    auto_zwj: index < 2,
                    auto_zwnj: index < 2,
                    per_syllable: false,
                    random: false,
                },
            )
        }));
        positioning_rules.extend(PRESENTATION_FEATURES.iter().map(|&tag| {
            (
                tag,
                LookupRule {
                    auto_zwj: false,
                    auto_zwnj: false,
                    ..LookupRule::default()
                },
            )
        }));
    }
    if myanmar {
        enabled.extend_from_slice(MYANMAR_BASIC);
        enabled.extend_from_slice(PRESENTATION_FEATURES);
        positioning_rules.extend(
            MYANMAR_BASIC
                .iter()
                .chain(PRESENTATION_FEATURES)
                .map(|&tag| {
                    (
                        tag,
                        LookupRule {
                            auto_zwj: false,
                            ..LookupRule::default()
                        },
                    )
                }),
        );
    }
    if universal {
        enabled.extend_from_slice(USE_PRE);
        enabled.extend_from_slice(USE_BASIC);
        enabled.extend_from_slice(USE_FORMS);
        enabled.extend_from_slice(USE_FINAL);
        enabled.extend([*b"rphf", *b"pref"]);
        positioning_rules.extend(USE_PRE.iter().map(|&tag| {
            (
                tag,
                LookupRule {
                    auto_zwj: tag != *b"akhn",
                    ..LookupRule::default()
                },
            )
        }));
        positioning_rules.extend(USE_BASIC.iter().chain(USE_FINAL).map(|&tag| {
            (
                tag,
                LookupRule {
                    auto_zwj: false,
                    ..LookupRule::default()
                },
            )
        }));
        positioning_rules.extend(USE_FORMS.iter().enumerate().map(|(index, &tag)| {
            (
                tag,
                LookupRule {
                    mask: 4 << index,
                    ..LookupRule::default()
                },
            )
        }));
        positioning_rules.extend([
            (
                *b"rphf",
                LookupRule {
                    mask: 2,
                    auto_zwj: false,
                    ..LookupRule::default()
                },
            ),
            (
                *b"pref",
                LookupRule {
                    auto_zwj: false,
                    ..LookupRule::default()
                },
            ),
        ]);
    }
    if font.tables.contains_key(b"GPOS") {
        let data = font.table(b"GPOS")?;
        let mut engine = Engine {
            font,
            data,
            positioning: true,
            coverage_query: false,
            zero_context: false,
            mask_map,
            random_state: 1,
            rtl,
            next_ligature: 1,
            next_slot: glyphs.len(),
            rule: LookupRule::default(),
            glyphs,
        };
        if myanmar || universal {
            for index in 0..engine.glyphs.len() {
                if engine.glyph_class(index)? == 3 {
                    engine.glyphs[index].advance = 0;
                    engine.glyphs[index].y_advance = 0;
                }
            }
        }
        engine.apply(plan(
            data,
            script,
            &enabled,
            &all_features,
            &all_features,
            true,
            &positioning_rules,
        )?)?;
        for index in 0..engine.glyphs.len() {
            if script != "khmr"
                && !myanmar
                && !universal
                && !indic
                && engine.glyph_class(index)? == 3
            {
                engine.glyphs[index].advance = 0;
                engine.glyphs[index].y_advance = 0;
            }
        }
        for glyph in &mut engine.glyphs {
            if !glyph.substituted && matches!(glyph.character, 0x200C..=0x200D) {
                glyph.advance = 0;
                glyph.y_advance = 0;
                glyph.x_offset = 0;
                glyph.y_offset = 0;
            }
        }
        for index in 0..engine.glyphs.len() {
            engine.finish_attachment(index, 64);
        }
        glyphs = engine.glyphs;
    } else {
        let mut engine = Engine {
            font,
            data: &[],
            positioning: true,
            coverage_query: false,
            zero_context: false,
            mask_map,
            random_state: 1,
            rtl,
            next_ligature: 1,
            next_slot: glyphs.len(),
            rule: LookupRule::default(),
            glyphs,
        };
        for index in 0..engine.glyphs.len() {
            if script != "khmr" && !indic && engine.glyph_class(index)? == 3 {
                engine.glyphs[index].advance = 0;
                engine.glyphs[index].y_advance = 0;
            }
        }
        if script != "khmr" && !myanmar && !universal && !indic {
            engine.fallback_metric_positions()?;
        }
        for glyph in &mut engine.glyphs {
            if !glyph.substituted && matches!(glyph.character, 0x200C..=0x200D) {
                glyph.advance = 0;
                glyph.y_advance = 0;
                glyph.x_offset = 0;
                glyph.y_offset = 0;
            }
        }
        glyphs = engine.glyphs;
    }
    let invisible = font.glyph_index(' ')?;
    if invisible != 0 {
        for glyph in &mut glyphs {
            if !glyph.substituted && matches!(glyph.character, 0x200C..=0x200D) {
                glyph.id = invisible;
            }
        }
    } else {
        glyphs.retain(|glyph| glyph.substituted || !matches!(glyph.character, 0x200C..=0x200D));
    }
    if rtl {
        glyphs.reverse();
    }
    if script == "arab" {
        arabic::stretch(font, &mut glyphs, rtl)?;
    }
    if feature.is_some() {
        let normal = cluster(font, text, None)?;
        if glyphs
            .iter()
            .map(|glyph| glyph.id)
            .eq(normal.iter().map(|glyph| glyph.id))
        {
            glyphs.clear();
        }
    }
    Ok(glyphs)
}

pub(super) fn single_substitution(font: &TrueTypeFont, lookup: usize, id: usize) -> Result<bool> {
    let data = font.table(b"GSUB")?;
    let glyphs = vec![Element {
        id,
        ..Element::default()
    }];
    Ok(Engine {
        font,
        data,
        positioning: false,
        coverage_query: true,
        zero_context: true,
        mask_map: [0; 24],
        random_state: 1,
        rtl: false,
        next_ligature: 1,
        next_slot: 1,
        rule: LookupRule::default(),
        glyphs,
    }
    .lookup(lookup, 0, 64)?
    .is_some())
}
