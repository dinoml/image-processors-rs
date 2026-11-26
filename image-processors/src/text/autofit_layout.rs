//! OpenType lookup coverage used by the automatic hinter.
//!
//! BASIC is the caption layout policy.  It does not disable the font layout
//! metadata consulted by the reference engine's automatic hinter.  Coverage
//! visits all languages of a script and follows contextual lookup references;
//! it is not the set of glyphs produced by shaping one particular string.

use std::collections::BTreeSet;

use super::{invalid, u16_at, u32_at, Result, TrueTypeFont};

pub(super) fn offset(data: &[u8], base: usize, field: usize) -> Result<usize> {
    base.checked_add(usize::from(u16_at(data, field)?))
        .ok_or_else(invalid)
}

pub(super) fn tags(script: &str, include_default: bool) -> Vec<[u8; 4]> {
    let old = match script {
        "geok" | "khms" | "latb" | "latp" | "none" => None,
        "lao" => Some(*b"lao "),
        "nkoo" => Some(*b"nko "),
        "vaii" => Some(*b"vai "),
        _ => script.as_bytes().try_into().ok(),
    };
    let new = match script {
        "beng" => Some(*b"bng2"),
        "deva" => Some(*b"dev2"),
        "gujr" => Some(*b"gjr2"),
        "guru" => Some(*b"gur2"),
        "knda" => Some(*b"knd2"),
        "mlym" => Some(*b"mlm2"),
        "orya" => Some(*b"ory2"),
        "taml" => Some(*b"tml2"),
        "telu" => Some(*b"tel2"),
        "mymr" => Some(*b"mym2"),
        _ => None,
    };
    let mut result = Vec::new();
    if let Some(new) = new {
        if script != "mymr" {
            let mut newest = new;
            newest[3] = b'3';
            result.push(newest);
        }
        result.push(new);
    }
    if let Some(old) = old {
        result.push(old);
    }
    // af_shaper_get_coverage replaces the third script tag when adding DFLT.
    if include_default {
        result.truncate(2);
        if !result.contains(b"DFLT") {
            result.push(*b"DFLT");
        }
    }
    result
}

fn language_features(
    data: &[u8],
    language: usize,
    features: &mut BTreeSet<usize>,
    include_required: bool,
) -> Result<()> {
    let required = u16_at(data, language + 2)?;
    if include_required && required != 0xffff {
        features.insert(usize::from(required));
    }
    for index in 0..usize::from(u16_at(data, language + 4)?) {
        features.insert(usize::from(u16_at(data, language + 6 + 2 * index)?));
    }
    Ok(())
}

pub(super) fn lookups(
    data: &[u8],
    script: &str,
    include_default: bool,
    feature_filter: Option<&[u8; 4]>,
) -> Result<BTreeSet<usize>> {
    let scripts = usize::from(u16_at(data, 4)?);
    let features = usize::from(u16_at(data, 6)?);
    let wanted = tags(script, include_default);
    let mut selected = BTreeSet::new();
    for index in 0..usize::from(u16_at(data, scripts)?) {
        let record = scripts + 2 + 6 * index;
        let tag: [u8; 4] = data
            .get(record..record + 4)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?;
        if !wanted.contains(&tag) {
            continue;
        }
        let table = offset(data, scripts, record + 4)?;
        let default = u16_at(data, table)?;
        if default != 0 {
            language_features(
                data,
                table + usize::from(default),
                &mut selected,
                feature_filter.is_none(),
            )?;
        }
        for index in 0..usize::from(u16_at(data, table + 2)?) {
            language_features(
                data,
                offset(data, table, table + 8 + 6 * index)?,
                &mut selected,
                feature_filter.is_none(),
            )?;
        }
    }
    let mut result = BTreeSet::new();
    let count = usize::from(u16_at(data, features)?);
    for feature in selected {
        // HarfBuzz ignores out-of-range feature indices.
        if feature >= count {
            continue;
        }
        if let Some(filter) = feature_filter {
            let record = features + 2 + 6 * feature;
            if data.get(record..record + 4) != Some(filter.as_slice()) {
                continue;
            }
        }
        let table = offset(data, features, features + 6 + 6 * feature)?;
        for index in 0..usize::from(u16_at(data, table + 2)?) {
            result.insert(usize::from(u16_at(data, table + 4 + 2 * index)?));
        }
    }
    Ok(result)
}

pub(super) fn coverage(data: &[u8], table: usize) -> Result<Vec<u16>> {
    let mut result = Vec::new();
    let count = usize::from(u16_at(data, table + 2)?);
    match u16_at(data, table)? {
        1 => {
            for index in 0..count {
                result.push(u16_at(data, table + 4 + 2 * index)?);
            }
        }
        2 => {
            for index in 0..count {
                let record = table + 4 + 6 * index;
                let first = u16_at(data, record)?;
                let last = u16_at(data, record + 2)?;
                if first > last {
                    return Err(invalid());
                }
                result.extend(first..=last);
            }
        }
        _ => return Err(invalid()),
    }
    Ok(result)
}

struct Collector<'a> {
    data: &'a [u8],
    list: usize,
    visited: BTreeSet<usize>,
    output: BTreeSet<usize>,
    nesting: usize,
}

impl Collector<'_> {
    fn records(&mut self, records: usize, count: usize) -> Result<()> {
        if self.nesting == 0 {
            return Ok(());
        }
        for index in 0..count {
            self.nesting -= 1;
            self.lookup(usize::from(u16_at(self.data, records + 4 * index + 2)?))?;
            self.nesting += 1;
        }
        Ok(())
    }

    fn lookup(&mut self, index: usize) -> Result<()> {
        if index >= usize::from(u16_at(self.data, self.list)?) || !self.visited.insert(index) {
            return Ok(());
        }
        let lookup = offset(self.data, self.list, self.list + 2 + 2 * index)?;
        let kind = u16_at(self.data, lookup)?;
        for index in 0..usize::from(u16_at(self.data, lookup + 4)?) {
            self.subtable(kind, offset(self.data, lookup, lookup + 6 + 2 * index)?, 0)?;
        }
        Ok(())
    }

    fn context(&mut self, table: usize, chained: bool) -> Result<()> {
        let data = self.data;
        let format = u16_at(data, table)?;
        if format == 3 {
            let mut field = table + 2;
            if chained {
                field += 2 + 2 * usize::from(u16_at(data, field)?);
            }
            let inputs = usize::from(u16_at(data, field)?);
            field += 2;
            let mut substitutions = None;
            if !chained {
                substitutions = Some(usize::from(u16_at(data, field)?));
                field += 2;
            }
            field += 2 * inputs;
            if chained {
                field += 2 + 2 * usize::from(u16_at(data, field)?);
            }
            let count = if let Some(count) = substitutions {
                count
            } else {
                let count = usize::from(u16_at(data, field)?);
                field += 2;
                count
            };
            return self.records(field, count);
        }
        let sets = match (chained, format) {
            (false, 1) | (true, 1) => table + 4,
            (false, 2) => table + 6,
            (true, 2) => table + 10,
            _ => return Err(invalid()),
        };
        for index in 0..usize::from(u16_at(data, sets)?) {
            let relative = u16_at(data, sets + 2 + 2 * index)?;
            if relative == 0 {
                continue;
            }
            let set = table + usize::from(relative);
            for index in 0..usize::from(u16_at(data, set)?) {
                let rule = offset(data, set, set + 2 + 2 * index)?;
                let mut field = rule;
                if chained {
                    field += 2 + 2 * usize::from(u16_at(data, field)?);
                }
                let inputs = usize::from(u16_at(data, field)?);
                if inputs == 0 {
                    return Err(invalid());
                }
                field += 2;
                if !chained {
                    let count = usize::from(u16_at(data, field)?);
                    self.records(field + 2 + 2 * (inputs - 1), count)?;
                } else {
                    field += 2 * (inputs - 1);
                    field += 2 + 2 * usize::from(u16_at(data, field)?);
                    let count = usize::from(u16_at(data, field)?);
                    self.records(field + 2, count)?;
                }
            }
        }
        Ok(())
    }

    fn subtable(&mut self, kind: u16, table: usize, extension_depth: usize) -> Result<()> {
        let data = self.data;
        let format = u16_at(data, table)?;
        match kind {
            1 => {
                let input = coverage(data, offset(data, table, table + 2)?)?;
                match format {
                    1 => {
                        let delta = u16_at(data, table + 4)?;
                        self.output.extend(
                            input
                                .into_iter()
                                .map(|glyph| usize::from(glyph.wrapping_add(delta))),
                        );
                    }
                    2 => {
                        for index in 0..usize::from(u16_at(data, table + 4)?).min(input.len()) {
                            self.output
                                .insert(usize::from(u16_at(data, table + 6 + 2 * index)?));
                        }
                    }
                    _ => return Err(invalid()),
                }
            }
            2 | 3 => {
                if format != 1 {
                    return Err(invalid());
                }
                for index in 0..usize::from(u16_at(data, table + 4)?) {
                    let sequence = offset(data, table, table + 6 + 2 * index)?;
                    for index in 0..usize::from(u16_at(data, sequence)?) {
                        self.output
                            .insert(usize::from(u16_at(data, sequence + 2 + 2 * index)?));
                    }
                }
            }
            4 => {
                if format != 1 {
                    return Err(invalid());
                }
                for index in 0..usize::from(u16_at(data, table + 4)?) {
                    let set = offset(data, table, table + 6 + 2 * index)?;
                    for index in 0..usize::from(u16_at(data, set)?) {
                        self.output.insert(usize::from(u16_at(
                            data,
                            offset(data, set, set + 2 + 2 * index)?,
                        )?));
                    }
                }
            }
            5 | 6 => self.context(table, kind == 6)?,
            7 => {
                if format != 1 || extension_depth >= 8 {
                    return Err(invalid());
                }
                let target = table
                    .checked_add(usize::try_from(u32_at(data, table + 4)?).map_err(|_| invalid())?)
                    .ok_or_else(invalid)?;
                self.subtable(u16_at(data, table + 2)?, target, extension_depth + 1)?;
            }
            8 => {
                if format != 1 {
                    return Err(invalid());
                }
                let mut field = table + 4;
                field += 2 + 2 * usize::from(u16_at(data, field)?);
                field += 2 + 2 * usize::from(u16_at(data, field)?);
                for index in 0..usize::from(u16_at(data, field)?) {
                    self.output
                        .insert(usize::from(u16_at(data, field + 2 + 2 * index)?));
                }
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }
}

pub(super) fn default_glyphs(
    font: &TrueTypeFont,
    script: &str,
    include_default: bool,
) -> Result<BTreeSet<usize>> {
    if !font.tables.contains_key(b"GSUB") {
        return Ok(BTreeSet::new());
    }
    let data = font.table(b"GSUB")?;
    let mut collector = Collector {
        data,
        list: usize::from(u16_at(data, 8)?),
        visited: BTreeSet::new(),
        output: BTreeSet::new(),
        nesting: 64,
    };
    for lookup in lookups(data, script, include_default, None)? {
        collector.visited.clear();
        collector.lookup(lookup)?;
    }
    Ok(collector.output)
}

fn class_glyphs(data: &[u8], table: usize, output: &mut BTreeSet<usize>) -> Result<()> {
    match u16_at(data, table)? {
        1 => {
            let first = usize::from(u16_at(data, table + 2)?);
            let count = usize::from(u16_at(data, table + 4)?);
            // Preserve the inclusive endpoints used by HarfBuzz 8.5.0's
            // ClassDefFormat1_3::collect_coverage, including its trailing one.
            let mut start = 0;
            for index in 0..count {
                if u16_at(data, table + 6 + 2 * index)? != 0 {
                    continue;
                }
                if start != index {
                    output.extend(first + start..=first + index);
                }
                start = index + 1;
            }
            if start != count {
                output.extend(first + start..=first + count);
            }
        }
        2 => {
            for index in 0..usize::from(u16_at(data, table + 2)?) {
                let record = table + 4 + 6 * index;
                if u16_at(data, record + 4)? != 0 {
                    output.extend(
                        usize::from(u16_at(data, record)?)..=usize::from(u16_at(data, record + 2)?),
                    );
                }
            }
        }
        _ => return Err(invalid()),
    }
    Ok(())
}

fn position_inputs(
    data: &[u8],
    kind: u16,
    table: usize,
    output: &mut BTreeSet<usize>,
    depth: usize,
) -> Result<()> {
    if depth == 0 {
        return Ok(());
    }
    let format = u16_at(data, table)?;
    if kind == 9 {
        if format != 1 {
            return Err(invalid());
        }
        let target = table
            .checked_add(usize::try_from(u32_at(data, table + 4)?).map_err(|_| invalid())?)
            .ok_or_else(invalid)?;
        return position_inputs(data, u16_at(data, table + 2)?, target, output, depth - 1);
    }
    if (kind == 7 || kind == 8) && format == 3 {
        let mut field = table + 2;
        if kind == 8 {
            field += 2 + 2 * usize::from(u16_at(data, field)?);
        }
        let count = usize::from(u16_at(data, field)?);
        field += if kind == 7 { 4 } else { 2 };
        for index in 0..count {
            output.extend(
                coverage(data, offset(data, table, field + 2 * index)?)?
                    .into_iter()
                    .map(usize::from),
            );
        }
        return Ok(());
    }
    output.extend(
        coverage(data, offset(data, table, table + 2)?)?
            .into_iter()
            .map(usize::from),
    );
    match kind {
        1 | 3 => {}
        2 if format == 1 => {
            let size = 2 + 2
                * (u16_at(data, table + 4)?.count_ones() + u16_at(data, table + 6)?.count_ones())
                    as usize;
            for index in 0..usize::from(u16_at(data, table + 8)?) {
                let set = offset(data, table, table + 10 + 2 * index)?;
                for index in 0..usize::from(u16_at(data, set)?) {
                    output.insert(usize::from(u16_at(data, set + 2 + size * index)?));
                }
            }
        }
        2 if format == 2 => class_glyphs(data, offset(data, table, table + 10)?, output)?,
        4..=6 if format == 1 => output.extend(
            coverage(data, offset(data, table, table + 4)?)?
                .into_iter()
                .map(usize::from),
        ),
        7 | 8 if format == 1 => {
            for index in 0..usize::from(u16_at(data, table + 4)?) {
                let relative = u16_at(data, table + 6 + 2 * index)?;
                if relative == 0 {
                    continue;
                }
                let set = table + usize::from(relative);
                for index in 0..usize::from(u16_at(data, set)?) {
                    let mut field = offset(data, set, set + 2 + 2 * index)?;
                    if kind == 8 {
                        field += 2 + 2 * usize::from(u16_at(data, field)?);
                    }
                    let count = usize::from(u16_at(data, field)?);
                    if count == 0 {
                        return Err(invalid());
                    }
                    field += if kind == 7 { 4 } else { 2 };
                    for index in 0..count - 1 {
                        output.insert(usize::from(u16_at(data, field + 2 * index)?));
                    }
                }
            }
        }
        7 | 8 if format == 2 => class_glyphs(
            data,
            offset(data, table, table + if kind == 7 { 4 } else { 6 })?,
            output,
        )?,
        _ => return Err(invalid()),
    }
    Ok(())
}

pub(super) fn feature_glyphs(
    font: &TrueTypeFont,
    script: &str,
    feature: &[u8; 4],
    blue_characters: impl Iterator<Item = char>,
) -> Result<BTreeSet<usize>> {
    if !font.tables.contains_key(b"GSUB") {
        return Ok(BTreeSet::new());
    }
    let data = font.table(b"GSUB")?;
    let selected = lookups(data, script, false, Some(feature))?;
    let mut found = false;
    for character in blue_characters {
        let glyph = font.glyph_index(character)?;
        for &lookup in &selected {
            if super::autofit_shaping::single_substitution(font, lookup, glyph)? {
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    if !found {
        return Ok(BTreeSet::new());
    }
    let mut collector = Collector {
        data,
        list: usize::from(u16_at(data, 8)?),
        visited: BTreeSet::new(),
        output: BTreeSet::new(),
        nesting: 64,
    };
    for lookup in selected {
        collector.visited.clear();
        collector.lookup(lookup)?;
    }
    if font.tables.contains_key(b"GPOS") {
        let data = font.table(b"GPOS")?;
        let list = usize::from(u16_at(data, 8)?);
        let mut positioned = BTreeSet::new();
        for lookup in lookups(data, script, false, Some(feature))? {
            if lookup >= usize::from(u16_at(data, list)?) {
                continue;
            }
            let table = offset(data, list, list + 2 + 2 * lookup)?;
            for index in 0..usize::from(u16_at(data, table + 4)?) {
                position_inputs(
                    data,
                    u16_at(data, table)?,
                    offset(data, table, table + 6 + 2 * index)?,
                    &mut positioned,
                    64,
                )?;
            }
        }
        collector.output.retain(|glyph| !positioned.contains(glyph));
    }
    Ok(collector.output)
}
