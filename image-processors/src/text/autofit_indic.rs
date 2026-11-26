//! Legacy Indic font-unit shaping for automatic-hinter samples.

use super::{chosen_script, has_feature, plan, syllables, Element, Engine, LookupRule, Result};

pub(super) const BASIC: &[([u8; 4], u64)] = &[
    (*b"nukt", 1),
    (*b"akhn", 1),
    (*b"rphf", 2),
    (*b"rkrf", 1),
    (*b"pref", 4),
    (*b"blwf", 8),
    (*b"abvf", 16),
    (*b"half", 32),
    (*b"pstf", 64),
    (*b"vatu", 1),
    (*b"cjct", 1),
];
pub(super) const FINAL: &[([u8; 4], u64)] = &[
    (*b"init", 128),
    (*b"pres", 1),
    (*b"abvs", 1),
    (*b"blws", 1),
    (*b"psts", 1),
    (*b"haln", 1),
];

fn is(glyph: &Element, categories: &[u8]) -> bool {
    !glyph.ligated && categories.contains(&glyph.category)
}
fn consonant(glyph: &Element) -> bool {
    is(glyph, &[1, 2, 10, 11, 15, 16, 18])
}
fn formed(glyph: &Element) -> bool {
    glyph.ligated && !glyph.multiplied
}

struct Parameters {
    old: bool,
    virama: usize,
    reph_position: u8,
    reph_mode: u8,
    pre_blwf: bool,
    zero_context: bool,
    masks: [u64; 7],
}

impl Engine<'_> {
    pub(super) fn indic(
        &mut self,
        script: &str,
        enabled: &[[u8; 4]],
        common: &[[u8; 4]],
        all: &[[u8; 4]],
    ) -> Result<()> {
        let (virama, reph_position, reph_mode) = match script {
            "deva" => ('\u{94D}', 10, 0),
            "beng" => ('\u{9CD}', 9, 0),
            "guru" => ('\u{A4D}', 7, 0),
            "gujr" => ('\u{ACD}', 10, 0),
            "orya" => ('\u{B4D}', 5, 0),
            "taml" => ('\u{BCD}', 12, 0),
            "telu" => ('\u{C4D}', 12, 1),
            "knda" => ('\u{CCD}', 12, 0),
            "mlym" => ('\u{D4D}', 5, 2),
            _ => return Err(super::invalid()),
        };
        let old = chosen_script(self.font, script)?.is_some_and(|tag| tag[3] != b'2');
        let mut masks = [0; 7];
        for (index, tag) in [
            *b"rphf", *b"pref", *b"blwf", *b"abvf", *b"half", *b"pstf", *b"init",
        ]
        .into_iter()
        .enumerate()
        {
            if has_feature(self.font, script, tag)? {
                masks[index] = 2 << index;
            }
        }
        let parameters = Parameters {
            old,
            virama: self.font.glyph_index(virama)?,
            reph_position,
            reph_mode,
            pre_blwf: !matches!(script, "telu" | "knda"),
            zero_context: !old && script != "mlym",
            masks,
        };
        let labels = syllables::indic(
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
        if parameters.virama != 0 {
            for index in 0..self.glyphs.len() {
                if self.glyphs[index].position != 4 {
                    continue;
                }
                let id = self.glyphs[index].id;
                let forward = [parameters.virama, id];
                let backward = [id, parameters.virama];
                let mut position = 4;
                for (tag, candidate) in
                    [(*b"blwf", 8), (*b"vatu", 8), (*b"pstf", 11), (*b"pref", 11)]
                {
                    if self.probe(script, tag, &forward, parameters.zero_context, all)?
                        || self.probe(script, tag, &backward, parameters.zero_context, all)?
                    {
                        position = candidate;
                        break;
                    }
                }
                self.glyphs[index].position = position;
            }
        }
        self.insert_dotted_circles(4, 11, Some(14))?;
        let mut start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            if matches!(label & 15, 0..=2 | 4) {
                self.indic_initial(script, &parameters, all, start, end)?;
            }
            start = end;
        }
        for &(tag, mask) in BASIC {
            let rule = LookupRule {
                mask,
                auto_zwj: false,
                auto_zwnj: false,
                per_syllable: true,
                random: false,
            };
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
        start = 0;
        while start < self.glyphs.len() {
            let label = self.glyphs[start].syllable;
            let end = (start + 1..self.glyphs.len())
                .find(|&index| self.glyphs[index].syllable != label)
                .unwrap_or(self.glyphs.len());
            self.indic_final(script, &parameters, start, end);
            start = end;
        }
        let earlier = initial
            .into_iter()
            .chain(BASIC.iter().map(|&(tag, _)| tag))
            .collect::<Vec<_>>();
        let final_enabled = enabled
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(FINAL.iter().map(|&(tag, _)| tag))
            .collect::<Vec<_>>();
        let final_features = common
            .iter()
            .copied()
            .filter(|tag| !earlier.contains(tag))
            .chain(FINAL.iter().map(|&(tag, _)| tag))
            .collect::<Vec<_>>();
        let rules = FINAL
            .iter()
            .map(|&(tag, mask)| {
                (
                    tag,
                    LookupRule {
                        mask,
                        auto_zwj: false,
                        auto_zwnj: false,
                        per_syllable: true,
                        random: false,
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
        )?)?;
        for glyph in &mut self.glyphs {
            glyph.syllable = 0;
        }
        Ok(())
    }

    fn indic_initial(
        &mut self,
        script: &str,
        p: &Parameters,
        all: &[[u8; 4]],
        start: usize,
        end: usize,
    ) -> Result<()> {
        if script == "knda"
            && end - start >= 3
            && is(&self.glyphs[start], &[15])
            && is(&self.glyphs[start + 1], &[4])
            && is(&self.glyphs[start + 2], &[6])
        {
            self.glyphs.swap(start + 1, start + 2);
        }
        let mut base = end;
        let mut limit = start;
        let mut reph = false;
        if p.masks[0] != 0
            && end - start >= 3
            && ((p.reph_mode == 0 && !is(&self.glyphs[start + 2], &[5, 6]))
                || (p.reph_mode == 1 && self.glyphs[start + 2].category == 6))
        {
            let ids = self.glyphs[start..start + 3]
                .iter()
                .map(|glyph| glyph.id)
                .collect::<Vec<_>>();
            if self.probe(script, *b"rphf", &ids[..2], p.zero_context, all)?
                || (p.reph_mode == 1 && self.probe(script, *b"rphf", &ids, p.zero_context, all)?)
            {
                limit += 2;
                while limit < end && is(&self.glyphs[limit], &[5, 6]) {
                    limit += 1;
                }
                base = start;
                reph = true;
            }
        } else if p.reph_mode == 2 && self.glyphs[start].category == 14 {
            limit += 1;
            while limit < end && is(&self.glyphs[limit], &[5, 6]) {
                limit += 1;
            }
            base = start;
            reph = true;
        }
        let mut seen_below = false;
        for index in (limit.min(end - 1)..end).rev() {
            let glyph = &self.glyphs[index];
            if consonant(glyph) {
                base = index;
                if glyph.position != 8 && (glyph.position != 11 || seen_below) {
                    break;
                }
                if glyph.position == 8 {
                    seen_below = true;
                }
            } else if index > start && glyph.category == 6 && self.glyphs[index - 1].category == 4 {
                break;
            }
        }
        if reph && base == start && limit - base <= 2 {
            reph = false;
        }
        for glyph in &mut self.glyphs[start..base] {
            glyph.position = glyph.position.min(3);
        }
        if base < end {
            self.glyphs[base].position = 4;
        }
        if reph {
            self.glyphs[start].position = 1;
        }
        if p.old {
            for index in base + 1..end {
                if self.glyphs[index].category != 4 {
                    continue;
                }
                let mut last = end - 1;
                while last > index
                    && !consonant(&self.glyphs[last])
                    && !(script == "knda" && self.glyphs[last].category == 4)
                {
                    last -= 1;
                }
                if last > index && self.glyphs[last].category != 4 {
                    self.glyphs[index..last + 1].rotate_left(1);
                }
                break;
            }
        }
        let mut last_position = 0;
        for index in start..end {
            if matches!(self.glyphs[index].category, 3..=6 | 12 | 16) {
                self.glyphs[index].position = last_position;
                if self.glyphs[index].category == 4 && last_position == 2 {
                    if let Some(previous) = (start..index)
                        .rev()
                        .find(|&previous| self.glyphs[previous].position != 2)
                    {
                        self.glyphs[index].position = self.glyphs[previous].position;
                    }
                }
            } else if self.glyphs[index].position != 13 {
                if self.glyphs[index].category == 13
                    && index > start
                    && self.glyphs[index - 1].category == 8
                {
                    self.glyphs[index - 1].position = self.glyphs[index].position;
                }
                last_position = self.glyphs[index].position;
            }
        }
        let mut last = base;
        for index in base + 1..end {
            if consonant(&self.glyphs[index]) {
                let position = self.glyphs[index].position;
                for glyph in &mut self.glyphs[last + 1..index] {
                    if glyph.position < 13 {
                        glyph.position = position;
                    }
                }
                last = index;
            } else if matches!(self.glyphs[index].category, 7 | 13) {
                last = index;
            }
        }
        self.glyphs[start..end].sort_by_key(|glyph| glyph.position);
        base = (start..end)
            .find(|&index| self.glyphs[index].position == 4)
            .unwrap_or(end);
        let first = (start..base).find(|&index| self.glyphs[index].position == 2);
        let last = (start..base).rfind(|&index| self.glyphs[index].position == 2);
        if let (Some(first), Some(last)) = (first, last) {
            if first < last {
                self.glyphs[first..last + 1].reverse();
                let mut group = first;
                for index in first..=last {
                    if matches!(self.glyphs[index].category, 7 | 13) {
                        self.glyphs[group..index + 1].reverse();
                        group = index + 1;
                    }
                }
            }
        }
        for glyph in &mut self.glyphs[start..end] {
            if glyph.position != 1 {
                break;
            }
            glyph.mask |= p.masks[0];
        }
        let pre_mask = p.masks[4] | if !p.old && p.pre_blwf { p.masks[2] } else { 0 };
        for glyph in &mut self.glyphs[start..base] {
            glyph.mask |= pre_mask;
        }
        for index in base + 1..end {
            self.glyphs[index].mask |= p.masks[2] | p.masks[3] | p.masks[5];
        }
        if p.old && script == "deva" {
            for index in start..base.saturating_sub(1) {
                if self.glyphs[index].category == 15
                    && self.glyphs[index + 1].category == 4
                    && (index + 2 == base || self.glyphs[index + 2].category != 6)
                {
                    self.glyphs[index].mask |= p.masks[2];
                    self.glyphs[index + 1].mask |= p.masks[2];
                }
            }
        }
        if p.masks[1] != 0 && base + 2 < end {
            for index in base + 1..end - 1 {
                let ids = [self.glyphs[index].id, self.glyphs[index + 1].id];
                if self.probe(script, *b"pref", &ids, p.zero_context, all)? {
                    self.glyphs[index].mask |= p.masks[1];
                    self.glyphs[index + 1].mask |= p.masks[1];
                    break;
                }
            }
        }
        for index in start + 1..end {
            if !is(&self.glyphs[index], &[5, 6]) {
                continue;
            }
            let non_joiner = self.glyphs[index].category == 5;
            let mut previous = index;
            loop {
                previous -= 1;
                if non_joiner {
                    self.glyphs[previous].mask &= !p.masks[4];
                }
                if previous == start || consonant(&self.glyphs[previous]) {
                    break;
                }
            }
        }
        Ok(())
    }

    fn indic_final(&mut self, script: &str, p: &Parameters, start: usize, end: usize) {
        if p.virama != 0 {
            for glyph in &mut self.glyphs[start..end] {
                if glyph.id == p.virama && glyph.ligated && glyph.multiplied {
                    glyph.category = 4;
                    glyph.ligated = false;
                    glyph.multiplied = false;
                }
            }
        }
        let mut try_pref = p.masks[1] != 0;
        let mut base = (start..end)
            .find(|&index| self.glyphs[index].position >= 4)
            .unwrap_or(end);
        if base < end {
            if try_pref && base + 1 < end {
                if let Some(index) =
                    (base + 1..end).find(|&index| self.glyphs[index].mask & p.masks[1] != 0)
                {
                    if !(self.glyphs[index].substituted && formed(&self.glyphs[index])) {
                        base = index;
                        while base < end && is(&self.glyphs[base], &[4]) {
                            base += 1;
                        }
                        if base < end {
                            self.glyphs[base].position = 4;
                        }
                        try_pref = false;
                    }
                }
            }
            if base < end && script == "mlym" {
                let mut index = base + 1;
                while index < end {
                    while index < end && is(&self.glyphs[index], &[5, 6]) {
                        index += 1;
                    }
                    if index == end || !is(&self.glyphs[index], &[4]) {
                        break;
                    }
                    index += 1;
                    while index < end && is(&self.glyphs[index], &[5, 6]) {
                        index += 1;
                    }
                    if index < end
                        && consonant(&self.glyphs[index])
                        && self.glyphs[index].position == 8
                    {
                        base = index;
                        self.glyphs[base].position = 4;
                    }
                    index += 1;
                }
            }
            if base < end && base > start && self.glyphs[base].position > 4 {
                base -= 1;
            }
        }
        if base == end && base > start && is(&self.glyphs[base - 1], &[6]) {
            base -= 1;
        }
        if base < end {
            while base > start && is(&self.glyphs[base], &[3, 4]) {
                base -= 1;
            }
        }
        if end - start > 1 && base > start {
            let mut target = if base == end { base - 2 } else { base - 1 };
            if !matches!(script, "mlym" | "taml") {
                loop {
                    while target > start && !is(&self.glyphs[target], &[7, 13, 4]) {
                        target -= 1;
                    }
                    if is(&self.glyphs[target], &[4]) && self.glyphs[target].position != 2 {
                        if target + 1 < end
                            && self.glyphs[target + 1].category == 6
                            && target > start
                        {
                            target -= 1;
                            continue;
                        }
                    } else {
                        target = start;
                    }
                    break;
                }
            }
            if target > start && self.glyphs[target].position != 2 {
                let mut index = target;
                while index > start {
                    let old = index - 1;
                    if self.glyphs[old].position == 2 {
                        if old < base && base <= target {
                            base -= 1;
                        }
                        self.glyphs[old..target + 1].rotate_left(1);
                        target -= 1;
                    }
                    index -= 1;
                }
            }
        }
        if end - start > 1
            && self.glyphs[start].position == 1
            && ((self.glyphs[start].category == 14) ^ formed(&self.glyphs[start]))
        {
            let explicit = (start + 1..base)
                .find(|&index| is(&self.glyphs[index], &[4]))
                .map(|index| {
                    if index + 1 < base && is(&self.glyphs[index + 1], &[5, 6]) {
                        index + 1
                    } else {
                        index
                    }
                });
            let mut target = None;
            if p.reph_position != 12 {
                target = explicit;
                if target.is_none() && matches!(p.reph_position, 5 | 9) && base < end {
                    let mut index = base;
                    while index + 1 < end
                        && (if p.reph_position == 5 {
                            self.glyphs[index + 1].position <= 5
                        } else {
                            !matches!(self.glyphs[index + 1].position, 11..=13)
                        })
                    {
                        index += 1;
                    }
                    target = Some(index);
                }
            }
            let target = target.or(explicit).unwrap_or_else(|| {
                let mut index = end - 1;
                while index > start && self.glyphs[index].position == 13 {
                    index -= 1;
                }
                if is(&self.glyphs[index], &[4]) {
                    let mut next = base + 1;
                    while next < index {
                        if matches!(self.glyphs[next].category, 7 | 13) {
                            index -= 1;
                        }
                        next += 1;
                    }
                }
                index
            });
            self.glyphs[start..target + 1].rotate_left(1);
            if start < base && base <= target {
                base -= 1;
            }
        }
        if try_pref && base + 1 < end {
            if let Some(old) =
                (base + 1..end).find(|&index| self.glyphs[index].mask & p.masks[1] != 0)
            {
                if formed(&self.glyphs[old]) {
                    let mut target = base;
                    if !matches!(script, "mlym" | "taml") {
                        while target > start && !is(&self.glyphs[target - 1], &[7, 13, 4]) {
                            target -= 1;
                        }
                    }
                    if target > start
                        && is(&self.glyphs[target - 1], &[4])
                        && target < end
                        && is(&self.glyphs[target], &[5, 6])
                    {
                        target += 1;
                    }
                    self.glyphs[target..old + 1].rotate_right(1);
                }
            }
        }
        if self.glyphs[start].position == 2
            && (start == 0
                || matches!(
                    syllables::indic_properties(self.glyphs[start - 1].character).0,
                    10 | 11
                ))
        {
            self.glyphs[start].mask |= p.masks[6];
        }
    }
}
