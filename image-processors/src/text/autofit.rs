//! Native outline analysis for the BASIC grayscale automatic-hinting profile.

use super::{
    gray::Position,
    hinting::{mul_div, Zone},
    invalid,
    truetype::{vector_length, Face, Glyph},
    u16_at, Result, TrueTypeFont,
};

pub(super) fn styles(font: &TrueTypeFont) -> Result<Vec<(u16, bool)>> {
    let mut result = vec![(u16::MAX, false); font.glyphs];
    for (script, specification) in super::autofit_scripts::SCRIPTS.iter().enumerate() {
        let script = u16::try_from(script).map_err(|_| invalid())?;
        if matches!(specification.name, "cyrl" | "grek" | "latn") {
            for (feature_index, feature) in super::autofit_shaping::FEATURES.iter().enumerate() {
                let style =
                    script + 64 * u16::try_from(feature_index + 1).map_err(|_| invalid())?;
                for glyph in super::autofit_layout::feature_glyphs(
                    font,
                    specification.name,
                    feature,
                    specification
                        .blues
                        .iter()
                        .flat_map(|(characters, _)| characters.chars()),
                )? {
                    if let Some(glyph_style) = result.get_mut(glyph) {
                        if glyph_style.0 == u16::MAX {
                            glyph_style.0 = style;
                        }
                    }
                }
            }
        }
        for &(first, last) in specification.ranges {
            for code in first..=last {
                if let Some(character) = char::from_u32(code) {
                    let glyph = font.glyph_index(character)?;
                    if glyph != 0 && result[glyph].0 == u16::MAX {
                        result[glyph].0 = script;
                    }
                }
            }
        }
        for &(first, last) in specification.nonbase {
            for code in first..=last {
                if let Some(character) = char::from_u32(code) {
                    let glyph = font.glyph_index(character)?;
                    if glyph != 0 && result[glyph].0 == script {
                        result[glyph].1 = true;
                    }
                }
            }
        }
    }
    // The Windows reference build uses internal HarfBuzz even with Pillow's
    // BASIC layout.  GSUB-only glyphs (for example Aileron's ligatures) must
    // inherit the script of their lookup instead of falling back to CJK.
    for (script, specification) in super::autofit_scripts::SCRIPTS.iter().enumerate() {
        let script = u16::try_from(script).map_err(|_| invalid())?;
        for glyph in super::autofit_layout::default_glyphs(font, specification.name, false)? {
            if let Some(style) = result.get_mut(glyph) {
                if style.0 == u16::MAX {
                    style.0 = script;
                }
            }
        }
    }
    let latin = super::autofit_scripts::SCRIPTS
        .iter()
        .position(|script| script.name == "latn")
        .ok_or_else(invalid)?;
    for glyph in super::autofit_layout::default_glyphs(font, "latn", true)? {
        if let Some(style) = result.get_mut(glyph) {
            if style.0 == u16::MAX {
                style.0 = u16::try_from(latin).map_err(|_| invalid())?;
            }
        }
    }
    let fallback =
        u16::try_from(super::autofit_scripts::SCRIPTS.len() - 1).map_err(|_| invalid())?;
    for style in &mut result {
        if style.0 == u16::MAX {
            style.0 = fallback;
        }
    }
    Ok(result)
}

pub(super) struct Metrics {
    cjk: bool,
    downward: bool,
    units: i64,
    widths: [Vec<i64>; 2],
    standard: [i64; 2],
    scales: [i64; 2],
    blues: Vec<Blue>,
    fixed: bool,
    italic: bool,
    digits: Vec<usize>,
}

struct Blue {
    flags: u8,
    reference: i64,
    overshoot: i64,
    top: bool,
    adjustment: bool,
    ascender: i64,
    descender: i64,
    fitted_reference: i64,
    fitted_overshoot: i64,
    active: bool,
}

impl Metrics {
    pub fn new(face: &mut Face, style: u16) -> Result<Self> {
        let specification = super::autofit_scripts::SCRIPTS
            .get(usize::from(style & 63))
            .ok_or_else(invalid)?;
        let feature = if style < 64 {
            None
        } else {
            Some(
                *super::autofit_shaping::FEATURES
                    .get(usize::from((style >> 6) - 1))
                    .ok_or_else(invalid)?,
            )
        };
        let cjk = specification.engine == 1;
        let units = i64::from(u16_at(face.font.table(b"head")?, 18)?);
        let mut widths: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
        for sample in super::autofit_unicode::samples(specification.standard) {
            let elements = metric_elements(&face.font, sample, feature)?;
            if elements.len() > 1 {
                continue;
            }
            let Some(&(id, _)) = elements.first() else {
                continue;
            };
            if id == 0 {
                continue;
            }
            let glyph = face.unscaled(id)?;
            if glyph.outline.current.is_empty() {
                break;
            }
            let outline = Outline::new(&glyph.outline, units)?;
            for (axis, widths) in widths.iter_mut().enumerate() {
                let mut segments = if cjk {
                    cjk_segments(&outline, axis, units)?
                } else {
                    outline.segments(axis, units)?
                };
                if cjk {
                    cjk_link(&outline, &mut segments, axis, units, 65536)?;
                } else {
                    outline.link(&mut segments, axis, units, &[])?;
                }
                for (index, segment) in segments.iter().enumerate() {
                    if let Some(partner) = segment.link {
                        if partner > index
                            && segments[partner].link == Some(index)
                            && widths.len() < 16
                        {
                            widths.push((segment.position - segments[partner].position).abs());
                        }
                    }
                }
                quantize_widths(widths, units / 100);
            }
            break;
        }
        let standard = [
            widths[0].first().copied().unwrap_or(50 * units / 2048),
            widths[1].first().copied().unwrap_or(50 * units / 2048),
        ];
        let mut blues = Vec::<Blue>::new();
        for &(characters, flags) in specification.blues {
            let top = flags & 3 != 0;
            let adjustment = flags & 8 != 0;
            let mut flats = Vec::new();
            let mut rounds = Vec::new();
            let mut ascender = 0;
            let mut descender = 0;
            for sample in super::autofit_unicode::samples(characters) {
                let mut extremum = None::<(i64, bool)>;
                for (id, y_offset) in metric_elements(&face.font, sample, feature)? {
                    if id == 0 {
                        continue;
                    }
                    let glyph = face.unscaled(id)?;
                    let zone = &glyph.outline;
                    if zone.current.len() <= 2 {
                        continue;
                    }
                    let mut best = None::<(usize, usize, usize)>;
                    let mut first = 0;
                    for &end in &zone.ends {
                        if end > first {
                            for index in first..=end {
                                let y = zone.current[index].y;
                                let improve = best.is_none_or(|(best, _, _)| {
                                    if top {
                                        y > zone.current[best].y
                                    } else {
                                        y < zone.current[best].y
                                    }
                                });
                                if improve {
                                    best = Some((index, first, end));
                                    if top {
                                        ascender = ascender.max(y + y_offset);
                                    } else {
                                        descender = descender.min(y + y_offset);
                                    }
                                } else if top {
                                    descender = descender.min(y + y_offset);
                                } else {
                                    ascender = ascender.max(y + y_offset);
                                }
                            }
                        }
                        first = end + 1;
                    }
                    let Some((best, first, end)) = best else {
                        continue;
                    };
                    let mut segment = [best; 2];
                    let mut on = [zone.on_curve[best].then_some(best); 2];
                    let mut forward = best;
                    for side in 0..2 {
                        let mut index = best;
                        loop {
                            index = if side == 0 {
                                if index == first {
                                    end
                                } else {
                                    index - 1
                                }
                            } else if index == end {
                                first
                            } else {
                                index + 1
                            };
                            if side == 1 {
                                forward = index;
                            }
                            let distance = (zone.current[index].y - zone.current[best].y).abs();
                            if distance > 5
                                && (zone.current[index].x - zone.current[best].x).abs()
                                    <= 20 * distance
                            {
                                break;
                            }
                            segment[side] = index;
                            if zone.on_curve[index] {
                                on[side] = Some(index);
                                if on[1 - side].is_none() {
                                    on[1 - side] = Some(index);
                                }
                            }
                            if index == best {
                                break;
                            }
                        }
                    }
                    let mut best_y = zone.current[best].y;
                    if flags & 16 != 0 {
                        let Some(sample) =
                            long_blue(zone, best, first..=end, segment, on, forward, units)
                        else {
                            continue;
                        };
                        (best_y, segment, on) = sample;
                    }
                    let flat = if let [Some(first), Some(last)] = on {
                        (zone.current[last].x - zone.current[first].x).abs() > units / 14
                    } else {
                        false
                    };
                    let round = !flat && (!zone.on_curve[segment[0]] || !zone.on_curve[segment[1]]);
                    if round && flags & 4 != 0 {
                        continue;
                    }
                    best_y += y_offset;
                    if extremum
                        .is_none_or(|(value, _)| if top { best_y > value } else { best_y < value })
                    {
                        extremum = Some((best_y, round));
                    }
                }
                if let Some((best_y, round)) = extremum {
                    if round {
                        rounds.push(best_y);
                    } else {
                        flats.push(best_y);
                    }
                }
            }
            if flats.is_empty() && rounds.is_empty() {
                continue;
            }
            flats.sort_unstable();
            rounds.sort_unstable();
            let mut reference = flats
                .get(flats.len() / 2)
                .copied()
                .unwrap_or_else(|| rounds[rounds.len() / 2]);
            let mut overshoot = rounds.get(rounds.len() / 2).copied().unwrap_or(reference);
            if reference != overshoot && top != (overshoot > reference) {
                reference = (reference + overshoot) / 2;
                overshoot = reference;
            }
            blues.push(Blue {
                flags,
                reference,
                overshoot,
                top,
                adjustment,
                ascender,
                descender,
                fitted_reference: 0,
                fitted_overshoot: 0,
                active: false,
            });
        }
        if specification.name == "hani" {
            blues = cjk_blues(face)?;
        }
        let mut order = (0..if cjk { 0 } else { blues.len() }).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            if blues[index].top {
                blues[index].reference
            } else {
                blues[index].overshoot
            }
        });
        for pair in order.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let bottom = if blues[b].top {
                blues[b].overshoot
            } else {
                blues[b].reference
            };
            if blues[a].top {
                blues[a].overshoot = blues[a].overshoot.min(bottom);
            } else {
                blues[a].reference = blues[a].reference.min(bottom);
            }
        }
        let mut scales = [face.interpreter.scale; 2];
        if let Some(blue) = blues.iter().find(|blue| blue.adjustment) {
            let scaled = mul_div(blue.overshoot, scales[1], 65536)?;
            let fitted = (scaled + 40) & !63;
            if scaled != 0 && scaled != fitted {
                let adjusted = mul_div(scales[1], fitted, scaled)?;
                let maximum = blues.iter().fold(units, |maximum, blue| {
                    maximum.max(blue.ascender).max(-blue.descender)
                });
                if mul_div(maximum, adjusted - scales[1], 65536)?.abs() & !127 == 0 {
                    scales[1] = adjusted;
                }
            }
        }
        for blue in &mut blues {
            let distance = mul_div(blue.reference - blue.overshoot, scales[1], 65536)?;
            blue.active = distance.abs() <= 48;
            if blue.active {
                blue.fitted_reference = (mul_div(blue.reference, scales[1], 65536)? + 32) & !63;
                if cjk {
                    let unscaled_distance =
                        mul_div(blue.fitted_reference, 65536, scales[1])? - blue.overshoot;
                    let distance = mul_div(unscaled_distance.abs(), scales[1], 65536)?;
                    let amount = if distance < 32 {
                        0
                    } else {
                        (distance + 32) & !63
                    };
                    blue.fitted_overshoot =
                        blue.fitted_reference - amount * unscaled_distance.signum();
                } else {
                    let amount = if distance.abs() < 32 {
                        0
                    } else if distance.abs() < 48 {
                        32
                    } else {
                        64
                    };
                    blue.fitted_overshoot = blue.fitted_reference - amount * distance.signum();
                }
            }
        }
        for index in 0..blues.len() {
            if blues[index].flags & 2 == 0 || !blues[index].active {
                continue;
            }
            if blues.iter().any(|other| {
                other.flags & 2 == 0
                    && other.active
                    && other.fitted_reference <= blues[index].fitted_overshoot
                    && other.fitted_overshoot >= blues[index].fitted_reference
            }) {
                blues[index].active = false;
            }
        }
        let fixed = face
            .font
            .table(b"post")
            .is_ok_and(|post| post.get(12..16).is_some_and(|value| value != [0, 0, 0, 0]));
        let italic = if let Ok(os2) = face.font.table(b"OS/2") {
            if u16_at(os2, 0)? != 0xffff {
                u16_at(os2, 62)? & 1 != 0
            } else {
                u16_at(face.font.table(b"head")?, 44)? & 2 != 0
            }
        } else {
            u16_at(face.font.table(b"head")?, 44)? & 2 != 0
        };
        let mut digits = Vec::new();
        let mut digit_width = None;
        for character in '0'..='9' {
            let id = face.font.glyph_index(character)?;
            if id == 0 {
                continue;
            }
            let mut bytes = [0; 4];
            let elements = super::autofit_shaping::cluster(
                &face.font,
                character.encode_utf8(&mut bytes),
                feature,
            )?;
            if elements.len() != 1 || elements[0].id == 0 {
                continue;
            }
            let advance = elements[0].advance;
            if digit_width.is_some_and(|width| width != advance) {
                digits.clear();
                break;
            }
            digit_width = Some(advance);
            digits.push(id);
        }
        Ok(Self {
            units,
            widths,
            standard,
            scales,
            blues,
            fixed,
            italic,
            digits,
            cjk,
            downward: specification.downward,
        })
    }

    pub fn apply(&self, face: &mut Face, id: usize, nonbase: bool) -> Result<Glyph> {
        let mut glyph = face.unscaled(id)?;
        let outline = Outline::new(&glyph.outline, self.units)?;
        let original_advance = mul_div(glyph.advance, self.scales[0], 65536)?;
        let mut advance = (original_advance + 32) & !63;
        let mut origin = 0;
        for axis in 0..2 {
            if !self.cjk && (self.blues.is_empty() || axis == 0 && self.italic) {
                for point in &mut glyph.outline.current {
                    if axis == 0 {
                        point.x = mul_div(point.x, self.scales[axis], 65536)?;
                    } else {
                        point.y = mul_div(point.y, self.scales[axis], 65536)?;
                    }
                }
                continue;
            }
            let mut segments = if self.cjk {
                cjk_segments(&outline, axis, self.units)?
            } else {
                outline.segments(axis, self.units)?
            };
            if self.cjk {
                cjk_link(&outline, &mut segments, axis, self.units, self.scales[axis])?;
            } else {
                outline.link(&mut segments, axis, self.units, &self.widths[axis])?;
            }
            let mut edges = edges(
                &outline,
                &mut segments,
                axis,
                self.scales,
                self.standard[axis],
                self.cjk,
                axis == 1 && self.downward,
            )?;
            if axis == 1 && (self.cjk || !nonbase) {
                for edge in &mut edges {
                    let mut distance = mul_div(self.units / 40, self.scales[1], 65536)?.min(32);
                    for blue in &self.blues {
                        let neutral = blue.flags & 4 != 0;
                        if !blue.active
                            || !neutral && blue.top == (edge.direction == outline.major[1])
                        {
                            continue;
                        }
                        if self.cjk {
                            let (design, fitted) = if (edge.design - blue.reference).abs()
                                > (edge.design - blue.overshoot).abs()
                            {
                                (blue.overshoot, blue.fitted_overshoot)
                            } else {
                                (blue.reference, blue.fitted_reference)
                            };
                            let candidate =
                                mul_div((edge.design - design).abs(), self.scales[1], 65536)?;
                            if candidate < distance {
                                distance = candidate;
                                edge.blue = Some((fitted, false));
                            }
                            continue;
                        }
                        let reference_distance =
                            mul_div((edge.design - blue.reference).abs(), self.scales[1], 65536)?;
                        if reference_distance < distance {
                            distance = reference_distance;
                            edge.blue = Some((blue.fitted_reference, neutral));
                        }
                        if edge.round
                            && reference_distance != 0
                            && !neutral
                            && blue.top != (edge.design < blue.reference)
                        {
                            let overshoot_distance = mul_div(
                                (edge.design - blue.overshoot).abs(),
                                self.scales[1],
                                65536,
                            )?;
                            if overshoot_distance < distance {
                                distance = overshoot_distance;
                                edge.blue = Some((blue.fitted_overshoot, false));
                            }
                        }
                    }
                }
            }
            let standard = self.widths[axis]
                .first()
                .map(|&width| mul_div(width, self.scales[axis], 65536))
                .transpose()?;
            if self.cjk {
                fit_cjk(&mut edges, axis, standard)?;
            } else {
                fit_edges(
                    &mut edges,
                    axis,
                    standard,
                    mul_div(self.standard[axis], self.scales[axis], 65536)? < 40,
                    face.interpreter.ppem,
                    axis == 1 && self.downward,
                )?;
            }
            align(
                &outline,
                &mut glyph.outline,
                &segments,
                &edges,
                axis,
                self.scales[axis],
                self.cjk,
            )?;
            if !self.cjk && axis == 0 && edges.len() > 1 {
                let first = &edges[0];
                let last = &edges[edges.len() - 1];
                let old_right = original_advance - last.original;
                let left =
                    first.position - first.original - if first.original < 24 { 8 } else { 0 };
                let right = last.position + old_right + if old_right < 24 { 8 } else { 0 };
                origin = (left + 32) & !63;
                let mut end = (right + 32) & !63;
                if origin >= first.position && first.original > 0 {
                    origin -= 64;
                }
                if end <= last.position && old_right > 0 {
                    end += 64;
                }
                if glyph.advance != 0 && !self.fixed && !self.digits.contains(&id) {
                    advance = (end - origin + 32) & !63;
                }
            }
        }
        glyph.advance = advance;
        glyph.origin = origin;
        Ok(glyph)
    }
}

fn metric_elements(
    font: &TrueTypeFont,
    sample: &str,
    feature: Option<[u8; 4]>,
) -> Result<Vec<(usize, i64)>> {
    Ok(super::autofit_shaping::cluster(font, sample, feature)?
        .into_iter()
        .map(|glyph| (glyph.id, glyph.y_offset))
        .collect())
}

fn quantize_widths(widths: &mut Vec<i64>, threshold: i64) {
    if widths.len() <= 1 {
        return;
    }
    widths.sort_unstable();
    let mut start = 0;
    let mut value = widths[0];
    let mut index = 1;
    while index < widths.len() {
        if widths[index] - value > threshold || index == widths.len() - 1 {
            if widths[index] - value <= threshold && index == widths.len() - 1 {
                index += 1;
            }
            let sum = widths[start..index].iter().sum::<i64>();
            widths[start..index].fill(0);
            widths[start] = sum / index as i64;
            if index < widths.len() - 1 {
                start = index + 1;
                value = widths[start];
            }
        }
        index += 1;
    }
    let mut index = 0;
    widths.retain(|&width| {
        let keep = index == 0 || width != 0;
        index += 1;
        keep
    });
}

fn long_blue(
    zone: &Zone,
    best: usize,
    contour: std::ops::RangeInclusive<usize>,
    mut segment: [usize; 2],
    mut on: [Option<usize>; 2],
    forward: usize,
    units: i64,
) -> Option<(i64, [usize; 2], [Option<usize>; 2])> {
    let (contour_first, contour_last) = (*contour.start(), *contour.end());
    let next = |index| {
        if index == contour_last {
            contour_first
        } else {
            index + 1
        }
    };
    let previous = |index| {
        if index == contour_first {
            contour_last
        } else {
            index - 1
        }
    };
    let mut best_y = zone.current[best].y;
    let threshold = units / 25;
    if (zone.current[segment[1]].x - zone.current[segment[0]].x).abs() < threshold
        && segment[1] as i64 - segment[0] as i64 + 2 <= (contour_last - contour_first) as i64
    {
        let mut before = previous(best);
        while before != best && zone.current[before].x == zone.current[best].x {
            before = previous(before);
        }
        if before == best {
            return None;
        }
        let left_to_right = zone.current[before].x < zone.current[best].x;
        let stop = segment[0];
        let mut first = segment[1];
        let mut last = first;
        let mut hit = false;
        let mut first_on = None;
        let mut last_on = None;
        loop {
            if !hit {
                first = last;
                first_on = zone.on_curve[first].then_some(first);
                last_on = first_on;
                hit = true;
            }
            last = next(last);
            let distance = (zone.current[last].y - zone.current[first].y).abs();
            if (best_y - zone.current[first].y).abs() > units / 4
                || distance > 5
                    && (zone.current[last].x - zone.current[first].x).abs() <= 20 * distance
            {
                hit = false;
            } else {
                if zone.on_curve[last] {
                    last_on = Some(last);
                    if first_on.is_none() {
                        first_on = last_on;
                    }
                }
                if (zone.current[first].x < zone.current[last].x) == left_to_right
                    && (zone.current[last].x - zone.current[first].x).abs() >= threshold
                {
                    loop {
                        last = next(last);
                        let vertical = (zone.current[last].y - zone.current[first].y).abs();
                        // Retain the reference algorithm's `next' and `dist'
                        // operands while extending the accepted segment.
                        if vertical > 5
                            && (zone.current[forward].x - zone.current[first].x).abs()
                                <= 20 * distance
                        {
                            last = previous(last);
                            break;
                        }
                        last_on = Some(last);
                        if zone.on_curve[last] && first_on.is_none() {
                            first_on = Some(last);
                        }
                        if last == stop {
                            break;
                        }
                    }
                    best_y = zone.current[first].y;
                    segment = [first, last];
                    on = [first_on, last_on];
                    break;
                }
            }
            if last == stop {
                break;
            }
        }
    }
    Some((best_y, segment, on))
}

fn cjk_blues(face: &mut Face) -> Result<Vec<Blue>> {
    let mut result = Vec::new();
    for (characters, top) in [
        ("他们你來們到和地对對就席我时時會来為能舰說说这這齊|军同已愿既星是景民照现現理用置要軍那配里開雷露面顾", true),
        ("个为人他以们你來個們到和大对對就我时時有来為要說说|主些因它想意理生當看着置者自著裡过还进進過道還里面", false),
    ] {
        let mut fills = Vec::new(); let mut flats = Vec::new(); let mut fill = true;
        for character in characters.chars() {
            if character == '|' { fill = false; continue; }
            let mut bytes = [0; 4];
            let elements = super::autofit_shaping::cluster(&face.font, character.encode_utf8(&mut bytes), None)?;
            if elements.len() > 1 { continue; }
            let Some(element) = elements.first() else { continue; };
            let id = element.id;
            if id == 0 { continue; }
            let glyph = face.unscaled(id)?;
            let zone = &glyph.outline;
            if zone.current.len() <= 2 { continue; }
            let mut best = None::<i64>; let mut first = 0;
            for &last in &zone.ends {
                if last > first {
                    for point in &zone.current[first..=last] {
                        if best.is_none_or(|best| if top { point.y > best } else { point.y < best }) { best = Some(point.y); }
                    }
                }
                first = last + 1;
            }
            // The original records zero for an outline made solely of
            // single-point contours after rejecting outlines with <=2 points.
            let value = best.unwrap_or(0);
            if fill { fills.push(value); } else { flats.push(value); }
        }
        if fills.is_empty() && flats.is_empty() { continue; }
        fills.sort_unstable(); flats.sort_unstable();
        let mut reference = fills.get(fills.len() / 2).copied().unwrap_or_else(|| flats[flats.len() / 2]);
        let mut overshoot = flats.get(flats.len() / 2).copied().unwrap_or(reference);
        if reference != overshoot && top != (overshoot < reference) { reference = (reference + overshoot) / 2; overshoot = reference; }
        result.push(Blue { flags: u8::from(top), reference, overshoot, top, adjustment: false, ascender: 0, descender: 0, fitted_reference: 0, fitted_overshoot: 0, active: false });
    }
    Ok(result)
}

fn cjk_segments(outline: &Outline, axis: usize, units: i64) -> Result<Vec<Segment>> {
    let mut segments = outline.segments(axis, units)?;
    for segment in &mut segments {
        segment.round = false;
        let mut point = segment.first;
        let mut control = outline.points[point].control;
        while point != segment.last {
            point = outline.points[point].next;
            let next_control = outline.points[point].control;
            if !control && !next_control {
                break;
            }
            if point == segment.last {
                segment.round = true;
            }
            control = next_control;
        }
    }
    Ok(segments)
}

fn cjk_link(
    outline: &Outline,
    segments: &mut [Segment],
    axis: usize,
    units: i64,
    scale: i64,
) -> Result<()> {
    let mut lengths = vec![0; segments.len()];
    let threshold = 8 * units / 2048;
    let distance_threshold = mul_div(192, 65536, scale)?;
    for first in 0..segments.len() {
        if segments[first].direction != outline.major[axis] {
            continue;
        }
        for second in 0..segments.len() {
            if first == second || segments[first].direction + segments[second].direction != 0 {
                continue;
            }
            let distance = segments[second].position - segments[first].position;
            if distance < 0 {
                continue;
            }
            let length = segments[first].maximum.min(segments[second].maximum)
                - segments[first].minimum.max(segments[second].minimum);
            if length < threshold {
                continue;
            }
            for (index, partner) in [(first, second), (second, first)] {
                if distance * 8 < segments[index].score * 9
                    && (distance * 8 < segments[index].score * 7 || lengths[index] < length)
                {
                    segments[index].score = distance;
                    lengths[index] = length;
                    segments[index].link = Some(partner);
                }
            }
        }
    }
    for first in 0..segments.len() {
        let Some(link) = segments[first].link else {
            continue;
        };
        if segments[link].link != Some(first)
            || segments[link].position <= segments[first].position
            || segments[first].score >= distance_threshold
        {
            continue;
        }
        for second in 0..segments.len() {
            if first == second || segments[second].position > segments[first].position {
                continue;
            }
            let Some(other) = segments[second].link else {
                continue;
            };
            if segments[other].link != Some(second)
                || segments[other].position < segments[link].position
            {
                continue;
            }
            if segments[first].position == segments[second].position
                && segments[link].position == segments[other].position
            {
                continue;
            }
            if segments[second].score <= segments[first].score
                || segments[first].score * 4 <= segments[second].score
            {
                continue;
            }
            if lengths[first] >= lengths[second] * 3 {
                for segment in segments.iter_mut() {
                    if segment.link == Some(second) {
                        segment.link = None;
                        segment.serif = Some(link);
                    } else if segment.link == Some(other) {
                        segment.link = None;
                        segment.serif = Some(first);
                    }
                }
            } else {
                segments[first].link = None;
                segments[link].link = None;
                break;
            }
        }
    }
    for index in 0..segments.len() {
        if let Some(partner) = segments[index].link {
            if segments[partner].link != Some(index) {
                segments[index].link = None;
                if segments[partner].score < distance_threshold
                    || segments[index].score < segments[partner].score * 4
                {
                    segments[index].serif = segments[partner].link;
                }
            }
        }
    }
    Ok(())
}

fn cjk_width(width: i64, standard: Option<i64>) -> i64 {
    let mut distance = width.abs();
    if let Some(standard) = standard {
        if (distance - standard).abs() < 40 {
            return if width < 0 {
                -standard.max(48)
            } else {
                standard.max(48)
            };
        }
    }
    if distance < 54 {
        distance += (54 - distance) / 2;
    } else if distance < 192 {
        let fraction = distance & 63;
        distance = (distance & !63)
            + if fraction < 10 {
                fraction
            } else if fraction < 22 {
                10
            } else if fraction < 42 {
                fraction
            } else if fraction < 54 {
                54
            } else {
                fraction
            };
    }
    if width < 0 {
        -distance
    } else {
        distance
    }
}

fn cjk_normal(
    edges: &mut [Edge],
    first: usize,
    second: usize,
    anchor: i64,
    standard: Option<i64>,
) -> i64 {
    let length = cjk_width(edges[second].original - edges[first].original, standard);
    let center = (edges[first].original + edges[second].original) / 2 + anchor;
    let position = center - length / 2;
    let down_first = position & 63;
    let down_second = (position + length) & 63;
    let up_first = 64 - down_first;
    let up_second = 64 - down_second;
    let mut delta = 0;
    if down_first != 0 && down_second != 0 {
        if length <= 64 {
            if down_second < length {
                delta = if up_first <= down_second {
                    up_first
                } else {
                    -down_second
                };
            }
        } else {
            let mut offset = length & 63;
            if offset >= 32 || !(up_first <= offset || down_second <= offset) {
                if offset >= 32 {
                    offset = 0;
                }
                let first = if down_first <= up_first - offset {
                    -down_first
                } else {
                    up_first - offset
                };
                let second = if down_second - offset <= up_second {
                    -(down_second - offset)
                } else {
                    up_second
                };
                delta = if first.abs() <= second.abs() {
                    first
                } else {
                    second
                };
            }
        }
    }
    if edges[first].original < edges[second].original {
        edges[first].position = position + delta;
        edges[second].position = position + delta + length;
    } else {
        edges[first].position = position + delta + length;
        edges[second].position = position + delta;
    }
    delta
}

fn fit_cjk(edges: &mut [Edge], axis: usize, standard: Option<i64>) -> Result<()> {
    let linked = |edges: &mut [Edge], first: usize, second: usize| {
        edges[second].position = edges[first].position
            + cjk_width(edges[second].original - edges[first].original, standard);
    };
    let mut anchor = None;
    let mut delta = 0;
    let mut skipped = 0;
    let mut last = None;
    for index in 0..edges.len() {
        if edges[index].done {
            continue;
        }
        let pair = if let Some((position, _)) = edges[index].blue {
            Some((index, edges[index].link, position))
        } else if let Some(partner) = edges[index].link {
            edges[partner]
                .blue
                .map(|(position, _)| (partner, Some(index), position))
        } else {
            None
        };
        if let Some((first, second, position)) = pair {
            edges[first].position = position;
            edges[first].done = true;
            if let Some(second) = second {
                if edges[second].blue.is_none() {
                    linked(edges, first, second);
                    edges[second].done = true;
                }
            }
            if anchor.is_none() {
                anchor = Some(index);
            }
        }
    }
    for index in 0..edges.len() {
        if edges[index].done {
            continue;
        }
        let Some(partner) = edges[index].link else {
            skipped += 1;
            continue;
        };
        if last.is_some_and(|last| {
            edges[index].position < last + 64 || edges[partner].position < last + 64
        }) {
            skipped += 1;
            continue;
        }
        if edges[partner].blue.is_some() {
            linked(edges, partner, index);
            edges[index].done = true;
            continue;
        }
        if partner < index {
            linked(edges, partner, index);
            edges[index].done = true;
            last = Some(edges[index].position);
            continue;
        }
        if axis == 0 && anchor.is_none() {
            delta = cjk_normal(edges, index, partner, 0, standard);
        } else {
            cjk_normal(edges, index, partner, delta, standard);
        }
        anchor = Some(index);
        edges[index].done = true;
        edges[partner].done = true;
        last = Some(edges[partner].position);
    }
    if axis == 0 && matches!(edges.len(), 6 | 12) {
        let (a, b, c) = if edges.len() == 6 {
            (0, 2, 4)
        } else {
            (1, 5, 9)
        };
        if edges[a].link == Some(a + 1)
            && edges[b].link == Some(b + 1)
            && edges[c].link == Some(c + 1)
            && ((edges[b].original - edges[a].original) - (edges[c].original - edges[b].original))
                .abs()
                < 8
        {
            let delta = edges[c].position - (2 * edges[b].position - edges[a].position);
            edges[c].position -= delta;
            edges[c + 1].position -= delta;
            edges[c].done = true;
            edges[c + 1].done = true;
            if edges.len() == 12 {
                edges[8].position -= delta;
                edges[11].position -= delta;
            }
        }
    }
    if skipped == 0 {
        return Ok(());
    }
    for index in 0..edges.len() {
        if edges[index].done {
            continue;
        }
        if let Some(serif) = edges[index].serif {
            edges[index].position =
                edges[serif].position + edges[index].original - edges[serif].original;
            edges[index].done = true;
            skipped -= 1;
        }
    }
    if skipped == 0 {
        return Ok(());
    }
    for index in 0..edges.len() {
        if edges[index].done {
            continue;
        }
        let before = (0..index).rev().find(|&other| edges[other].done);
        let after = (index + 1..edges.len()).find(|&other| edges[other].done);
        edges[index].position = match (before, after) {
            (Some(before), Some(after)) => {
                if edges[after].design == edges[before].design {
                    edges[before].position
                } else {
                    edges[before].position
                        + mul_div(
                            edges[index].design - edges[before].design,
                            edges[after].position - edges[before].position,
                            edges[after].design - edges[before].design,
                        )?
                }
            }
            (Some(other), None) | (None, Some(other)) => {
                edges[other].position + edges[index].original - edges[other].original
            }
            (None, None) => edges[index].position,
        };
    }
    Ok(())
}

#[derive(Clone)]
pub(super) struct Point {
    pub design: Position,
    pub previous: usize,
    pub next: usize,
    pub incoming: i8,
    pub outgoing: i8,
    pub weak: bool,
    pub control: bool,
    before: usize,
    after: usize,
}

pub(super) struct Outline {
    pub points: Vec<Point>,
    pub contours: Vec<usize>,
    pub major: [i8; 2],
}

#[derive(Clone)]
pub(super) struct Segment {
    pub first: usize,
    pub last: usize,
    pub direction: i8,
    pub position: i64,
    pub delta: i64,
    pub minimum: i64,
    pub maximum: i64,
    pub height: i64,
    pub round: bool,
    pub link: Option<usize>,
    pub serif: Option<usize>,
    pub edge: Option<usize>,
    score: i64,
}

pub(super) struct Edge {
    pub design: i64,
    pub original: i64,
    pub position: i64,
    pub direction: i8,
    pub round: bool,
    pub serif_base: bool,
    pub link: Option<usize>,
    pub serif: Option<usize>,
    pub blue: Option<(i64, bool)>,
    pub done: bool,
    segments: Vec<usize>,
}

pub(super) fn edges(
    outline: &Outline,
    segments: &mut [Segment],
    axis: usize,
    scales: [i64; 2],
    standard: i64,
    cjk: bool,
    downward: bool,
) -> Result<Vec<Edge>> {
    if axis > 1 || scales.iter().any(|&scale| scale <= 0) {
        return Err(invalid());
    }
    let length_threshold = if axis == 0 {
        mul_div(64, 65536, scales[1])?
    } else {
        0
    };
    let width_threshold = mul_div(32, 65536, scales[axis])?;
    let scaled_distance = mul_div(standard / 5, scales[axis], 65536)?;
    let distance = if cjk && scaled_distance <= 16 {
        standard / 5
    } else {
        mul_div(scaled_distance.min(16), 65536, scales[axis])?
    };
    let mut edges = Vec::<Edge>::new();
    for (index, segment) in segments.iter().enumerate() {
        if !cjk
            && (segment.direction == 0
                || segment.height < length_threshold
                || segment.delta > width_threshold
                || segment.serif.is_some() && 2 * segment.height < 3 * length_threshold)
        {
            continue;
        }
        let mut best = None;
        let mut best_distance = 65535;
        for (index, edge) in edges.iter().enumerate() {
            let delta = (edge.design - segment.position).abs();
            if edge.direction != segment.direction
                || delta >= distance
                || cjk && delta >= best_distance
            {
                continue;
            }
            if cjk {
                if let Some(link) = segment.link {
                    if edge.segments.iter().any(|&member| {
                        segments[member].link.is_some_and(|other| {
                            (segments[link].position - segments[other].position).abs() >= distance
                        })
                    }) {
                        continue;
                    }
                }
            }
            best = Some(index);
            best_distance = delta;
            if !cjk {
                break;
            }
        }
        if let Some(edge_index) = best {
            let edge = &mut edges[edge_index];
            edge.segments.push(index);
        } else {
            let position = mul_div(segment.position, scales[axis], 65536)?;
            let insertion = edges
                .iter()
                .position(|edge| {
                    (if downward {
                        edge.design < segment.position
                    } else {
                        edge.design > segment.position
                    }) || edge.design == segment.position
                        && segment.direction != outline.major[axis]
                })
                .unwrap_or(edges.len());
            edges.insert(
                insertion,
                Edge {
                    design: segment.position,
                    original: position,
                    position,
                    direction: segment.direction,
                    round: false,
                    serif_base: false,
                    link: None,
                    serif: None,
                    blue: None,
                    done: false,
                    segments: vec![index],
                },
            );
        }
    }
    for (index, segment) in segments.iter().enumerate() {
        if !cjk && segment.direction == 0 {
            if let Some(edge) = edges
                .iter_mut()
                .find(|edge| (edge.design - segment.position).abs() < distance)
            {
                edge.segments.push(index);
            }
        }
    }
    for (index, edge) in edges.iter().enumerate() {
        for &segment in &edge.segments {
            segments[segment].edge = Some(index);
        }
    }
    for index in 0..edges.len() {
        let mut round = 0;
        let mut straight = 0;
        for member in edges[index].segments.clone() {
            let segment = &segments[member];
            if segment.round {
                round += 1;
            } else {
                straight += 1;
            }
            let serif = segment
                .serif
                .filter(|&other| segments[other].edge.is_some_and(|edge| edge != index));
            let partner = if serif.is_some() {
                serif
            } else {
                segment.link.filter(|&other| segments[other].edge.is_some())
            };
            if let Some(partner) = partner {
                let existing = if serif.is_some() {
                    edges[index].serif
                } else {
                    edges[index].link
                };
                let selected = if let Some(existing) = existing {
                    if (segment.position - segments[partner].position).abs()
                        < (edges[index].design - edges[existing].design).abs()
                    {
                        segments[partner].edge
                    } else {
                        Some(existing)
                    }
                } else {
                    segments[partner].edge
                };
                if serif.is_some() {
                    edges[index].serif = selected;
                    if let Some(selected) = selected {
                        edges[selected].serif_base = true;
                    }
                } else {
                    edges[index].link = selected;
                }
            }
        }
        // Preserve the profile's ordering: processing an edge resets its flags;
        // later serif edges may subsequently mark it as a serif base.
        edges[index].round = round > 0 && round >= straight;
        edges[index].serif_base = false;
        if edges[index].link.is_some() {
            edges[index].serif = None;
        }
    }
    Ok(edges)
}

struct StemStyle {
    extra_light: bool,
    base_round: bool,
    stem_serif: bool,
}

fn stem_width(
    width: i64,
    base_delta: i64,
    axis: usize,
    standard: Option<i64>,
    style: StemStyle,
    ppem: i64,
) -> i64 {
    let StemStyle {
        extra_light,
        base_round,
        stem_serif,
    } = style;
    if extra_light {
        return width;
    }
    let mut distance = width.abs();
    if axis == 1 && stem_serif && distance < 192 {
        return width;
    }
    if base_round {
        if distance < 80 {
            distance = 64;
        }
    } else if distance < 56 {
        distance = 56;
    }
    if let Some(standard) = standard {
        if (distance - standard).abs() < 40 {
            distance = standard.max(48);
        } else if distance < 192 {
            let fraction = distance & 63;
            distance = (distance & !63)
                + if fraction < 10 {
                    fraction
                } else if fraction < 32 {
                    10
                } else if fraction < 54 {
                    54
                } else {
                    fraction
                };
        } else {
            let correction = if (width > 0 && base_delta > 0) || (width < 0 && base_delta < 0) {
                if ppem < 10 {
                    base_delta.abs()
                } else if ppem < 30 {
                    (base_delta * (30 - ppem) / 20).abs()
                } else {
                    0
                }
            } else {
                0
            };
            distance = (distance - correction + 32) & !63;
        }
    }
    if width < 0 {
        -distance
    } else {
        distance
    }
}

pub(super) fn fit_edges(
    edges: &mut [Edge],
    axis: usize,
    standard: Option<i64>,
    extra_light: bool,
    ppem: i64,
    downward: bool,
) -> Result<()> {
    let mut anchor = None::<usize>;
    let mut serifs = 0;
    let linked = |edges: &mut [Edge], base: usize, target: usize| {
        let distance = stem_width(
            edges[target].original - edges[base].original,
            edges[base].position - edges[base].original,
            axis,
            standard,
            StemStyle {
                extra_light,
                base_round: edges[base].round,
                stem_serif: edges[target].serif_base,
            },
            ppem,
        );
        edges[target].position = edges[base].position + distance;
    };
    let center = |original: i64, length: i64| {
        let rounded = (original + 32) & !63;
        let (up, down) = if length <= 64 { (32, 32) } else { (38, 26) };
        if (original - (rounded - up)).abs() < (original - (rounded + down)).abs() {
            rounded - up
        } else {
            rounded + down
        }
    };
    if axis == 1 {
        for index in 0..edges.len() {
            if edges[index].done {
                continue;
            }
            if let Some(partner) = edges[index].link {
                if let (Some((_, neutral)), Some((_, other_neutral))) =
                    (edges[index].blue, edges[partner].blue)
                {
                    if other_neutral {
                        edges[partner].blue = None;
                    } else if neutral {
                        edges[index].blue = None;
                    }
                }
            }
            let pair = if let Some((position, _)) = edges[index].blue {
                Some((index, edges[index].link, position))
            } else if let Some(partner) = edges[index].link {
                edges[partner]
                    .blue
                    .map(|(position, _)| (partner, Some(index), position))
            } else {
                None
            };
            if let Some((base, partner, position)) = pair {
                edges[base].position = position;
                edges[base].done = true;
                if let Some(partner) = partner {
                    if edges[partner].blue.is_none() {
                        linked(edges, base, partner);
                        edges[partner].done = true;
                    }
                }
                if anchor.is_none() {
                    anchor = Some(index);
                }
            }
        }
    }
    for index in 0..edges.len() {
        if edges[index].done {
            continue;
        }
        let Some(partner) = edges[index].link else {
            serifs += 1;
            continue;
        };
        if edges[partner].blue.is_some() {
            linked(edges, partner, index);
            edges[index].done = true;
            continue;
        }
        let original_length = edges[partner].original - edges[index].original;
        let length = stem_width(
            original_length,
            0,
            axis,
            standard,
            StemStyle {
                extra_light,
                base_round: edges[index].round,
                stem_serif: edges[partner].serif_base,
            },
            ppem,
        );
        if let Some(anchor) = anchor {
            let original = edges[anchor].position + edges[index].original - edges[anchor].original;
            let original_center = original + (original_length >> 1);
            if edges[partner].done {
                edges[index].position = edges[partner].position - length;
            } else if length < 96 {
                let center = center(original_center, length);
                edges[index].position = center - length / 2;
                edges[partner].position = center + length / 2;
            } else {
                let first = (original + 32) & !63;
                let second = ((original + original_length + 32) & !63) - length;
                edges[index].position = if (first + (length >> 1) - original_center).abs()
                    < (second + (length >> 1) - original_center).abs()
                {
                    first
                } else {
                    second
                };
                edges[partner].position = edges[index].position + length;
            }
            edges[index].done = true;
            edges[partner].done = true;
            if index > 0
                && (if downward {
                    edges[index].position > edges[index - 1].position
                } else {
                    edges[index].position < edges[index - 1].position
                })
                && (edges[partner].position - edges[index - 1].position).abs() > 16
            {
                edges[index].position = edges[index - 1].position;
            }
        } else {
            edges[index].position = if length < 96 {
                center(edges[index].original + (original_length >> 1), length) - length / 2
            } else {
                (edges[index].original + 32) & !63
            };
            anchor = Some(index);
            edges[index].done = true;
            linked(edges, index, partner);
        }
    }
    if axis == 0 && (edges.len() == 6 || edges.len() == 12) {
        let (first, second, third) = if edges.len() == 6 {
            (0, 2, 4)
        } else {
            (1, 5, 9)
        };
        if ((edges[second].original - edges[first].original)
            - (edges[third].original - edges[second].original))
            .abs()
            < 8
        {
            let delta =
                edges[third].position - (2 * edges[second].position - edges[first].position);
            edges[third].position -= delta;
            edges[third].done = true;
            if let Some(partner) = edges[third].link {
                edges[partner].position -= delta;
                edges[partner].done = true;
            }
            if edges.len() == 12 {
                edges[8].position -= delta;
                edges[11].position -= delta;
            }
        }
    }
    if serifs > 0 || anchor.is_none() {
        for index in 0..edges.len() {
            if edges[index].done {
                continue;
            }
            if let Some(serif) = edges[index]
                .serif
                .filter(|&serif| (edges[serif].original - edges[index].original).abs() < 80)
            {
                edges[index].position =
                    edges[serif].position + edges[index].original - edges[serif].original;
            } else if let Some(anchor) = anchor {
                let before = (0..index).rev().find(|&other| edges[other].done);
                let after = (index + 1..edges.len()).find(|&other| edges[other].done);
                edges[index].position = if let (Some(before), Some(after)) = (before, after) {
                    if edges[before].original == edges[after].original {
                        edges[before].position
                    } else {
                        edges[before].position
                            + mul_div(
                                edges[index].original - edges[before].original,
                                edges[after].position - edges[before].position,
                                edges[after].original - edges[before].original,
                            )?
                    }
                } else {
                    edges[anchor].position
                        + ((edges[index].original - edges[anchor].original + 16) & !31)
                };
            } else {
                edges[index].position = (edges[index].original + 32) & !63;
                anchor = Some(index);
            }
            edges[index].done = true;
            if index > 0
                && (if downward {
                    edges[index].position > edges[index - 1].position
                } else {
                    edges[index].position < edges[index - 1].position
                })
            {
                if let Some(partner) = edges[index].link {
                    if (edges[partner].position - edges[index - 1].position).abs() > 16 {
                        edges[index].position = edges[index - 1].position;
                    }
                }
            }
            if index + 1 < edges.len()
                && edges[index + 1].done
                && (if downward {
                    edges[index].position < edges[index + 1].position
                } else {
                    edges[index].position > edges[index + 1].position
                })
            {
                if let Some(partner) = edges[index].link {
                    if index > 0 && (edges[partner].position - edges[index - 1].position).abs() > 16
                    {
                        edges[index].position = edges[index + 1].position;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn align(
    outline: &Outline,
    zone: &mut Zone,
    segments: &[Segment],
    edges: &[Edge],
    axis: usize,
    scale: i64,
    cjk: bool,
) -> Result<()> {
    if axis > 1 || zone.current.len() != outline.points.len() {
        return Err(invalid());
    }
    let original = outline
        .points
        .iter()
        .map(|point| mul_div(coordinate(point.design, axis), scale, 65536))
        .collect::<Result<Vec<_>>>()?;
    let mut positions = original.clone();
    let mut touched = vec![false; positions.len()];
    let order = if cjk {
        edges
            .iter()
            .flat_map(|edge| edge.segments.iter().copied())
            .collect::<Vec<_>>()
    } else {
        (0..segments.len()).collect()
    };
    for member in order {
        let segment = &segments[member];
        if let Some(edge) = segment.edge {
            let edge = edges.get(edge).ok_or_else(invalid)?;
            let mut point = segment.first;
            loop {
                if cjk {
                    positions[point] += edge.position - edge.original;
                } else {
                    positions[point] = edge.position;
                }
                touched[point] = true;
                if point == segment.last {
                    break;
                }
                point = outline.points[point].next;
            }
        }
    }
    if let (Some(first), Some(last)) = (edges.first(), edges.last()) {
        for (index, point) in outline.points.iter().enumerate() {
            if touched[index] || point.weak {
                continue;
            }
            let design = coordinate(point.design, axis);
            positions[index] = if design <= first.design {
                first.position - (first.original - original[index])
            } else if design >= last.design {
                last.position + original[index] - last.original
            } else {
                let after = edges.partition_point(|edge| edge.design < design);
                if edges[after].design == design {
                    edges[after].position
                } else {
                    let before = &edges[after - 1];
                    let after = &edges[after];
                    let interpolation = mul_div(
                        after.position - before.position,
                        65536,
                        after.design - before.design,
                    )?;
                    before.position + mul_div(design - before.design, interpolation, 65536)?
                }
            };
            touched[index] = true;
        }
    }
    for &first in &outline.contours {
        let end = outline.points[first].previous;
        let references = (first..=end)
            .filter(|&index| touched[index])
            .collect::<Vec<_>>();
        if references.len() == 1 {
            let reference = references[0];
            let delta = positions[reference] - original[reference];
            for index in first..=end {
                if index != reference {
                    positions[index] = original[index] + delta;
                }
            }
        } else if references.len() > 1 {
            for pair in 0..references.len() {
                let a = references[pair];
                let b = references[(pair + 1) % references.len()];
                let (low, high) = if original[a] <= original[b] {
                    (a, b)
                } else {
                    (b, a)
                };
                let scale = if original[low] != original[high] && positions[low] != positions[high]
                {
                    Some(mul_div(
                        positions[high] - positions[low],
                        65536,
                        original[high] - original[low],
                    )?)
                } else {
                    None
                };
                let mut index = outline.points[a].next;
                while index != b {
                    let value = original[index];
                    positions[index] = if value <= original[low] {
                        value + positions[low] - original[low]
                    } else if value >= original[high] {
                        value + positions[high] - original[high]
                    } else if let Some(scale) = scale {
                        positions[low] + mul_div(value - original[low], scale, 65536)?
                    } else {
                        positions[low]
                    };
                    index = outline.points[index].next;
                }
            }
        }
    }
    for (index, position) in positions.into_iter().enumerate() {
        if axis == 0 {
            zone.current[index].x = position;
        } else {
            zone.current[index].y = position;
        }
        zone.touched[index][axis] = touched[index];
    }
    Ok(())
}

fn direction(x: i64, y: i64) -> i8 {
    if x.abs() > 14 * y.abs() {
        if x > 0 {
            1
        } else {
            -1
        }
    } else if y.abs() > 14 * x.abs() {
        if y > 0 {
            2
        } else {
            -2
        }
    } else {
        0
    }
}

fn coordinate(point: Position, axis: usize) -> i64 {
    if axis == 0 {
        point.x
    } else {
        point.y
    }
}

impl Outline {
    pub fn new(zone: &Zone, units: i64) -> Result<Self> {
        if zone.current.len() != zone.on_curve.len() || units <= 0 {
            return Err(invalid());
        }
        let mut points = zone
            .current
            .iter()
            .zip(&zone.on_curve)
            .enumerate()
            .map(|(index, (&design, &on))| Point {
                design,
                previous: index,
                next: index,
                incoming: 0,
                outgoing: 0,
                weak: false,
                control: !on,
                before: index,
                after: index,
            })
            .collect::<Vec<_>>();
        let mut contours = Vec::with_capacity(zone.ends.len());
        let mut first = 0;
        let mut area = 0_i128;
        for &end in &zone.ends {
            if end < first || end >= points.len() {
                return Err(invalid());
            }
            contours.push(first);
            for index in first..=end {
                let previous = if index == first { end } else { index - 1 };
                let next = if index == end { first } else { index + 1 };
                points[index].previous = previous;
                points[index].next = next;
                let p = points[previous].design;
                let q = points[index].design;
                area += i128::from(q.y - p.y) * i128::from(q.x + p.x);
            }
            first = end + 1;
        }
        if first != points.len() {
            return Err(invalid());
        }
        let major = if area > 0 { [-2, 1] } else { [2, -1] };
        let near = 20 * units / 2048;
        for &start in &contours {
            let mut first = start;
            let mut previous = points[first].previous;
            while previous != start {
                let p = points[first].design;
                let q = points[previous].design;
                if (p.x - q.x).abs() + (p.y - q.y).abs() >= 2 * near - 1 {
                    break;
                }
                first = previous;
                previous = points[first].previous;
            }
            let mut current = first;
            let mut next = first;
            let mut accumulated = Position::default();
            points[current].after = first;
            points[first].before = current;
            loop {
                let point = next;
                next = points[point].next;
                accumulated.x += points[next].design.x - points[point].design.x;
                accumulated.y += points[next].design.y - points[point].design.y;
                if accumulated.x.abs() + accumulated.y.abs() < near {
                    points[next].weak = true;
                } else {
                    points[current].after = next;
                    points[next].before = current;
                    let outgoing = direction(accumulated.x, accumulated.y);
                    points[current].outgoing = outgoing;
                    current = points[current].next;
                    while current != next {
                        points[current].incoming = outgoing;
                        points[current].outgoing = outgoing;
                        current = points[current].next;
                    }
                    points[next].incoming = outgoing;
                    points[current].after = first;
                    points[first].before = current;
                    accumulated = Position::default();
                }
                if next == first {
                    break;
                }
            }
        }
        for index in 0..points.len() {
            if points[index].weak || points[index].incoming != 0 || points[index].outgoing != 0 {
                continue;
            }
            let before = points[index].before;
            let after = points[index].after;
            let incoming = Position {
                x: points[index].design.x - points[before].design.x,
                y: points[index].design.y - points[before].design.y,
            };
            let outgoing = Position {
                x: points[after].design.x - points[index].design.x,
                y: points[after].design.y - points[index].design.y,
            };
            if (incoming.x ^ outgoing.x) >= 0 && (incoming.y ^ outgoing.y) >= 0 {
                points[index].weak = true;
                points[before].after = after;
                points[after].before = before;
            }
        }
        for index in 0..points.len() {
            if points[index].weak {
                continue;
            }
            let incoming = points[index].incoming;
            let outgoing = points[index].outgoing;
            if points[index].control
                || incoming == -outgoing && incoming != outgoing
                || incoming == outgoing && outgoing != 0
            {
                points[index].weak = true;
            } else if incoming == 0 && outgoing == 0 {
                let before = points[index].before;
                let after = points[index].after;
                let a = Position {
                    x: points[index].design.x - points[before].design.x,
                    y: points[index].design.y - points[before].design.y,
                };
                let b = Position {
                    x: points[after].design.x - points[index].design.x,
                    y: points[after].design.y - points[index].design.y,
                };
                let diagonal = vector_length(a.x + b.x, a.y + b.y)?;
                if vector_length(a.x, a.y)? + vector_length(b.x, b.y)? - diagonal < diagonal >> 4 {
                    points[index].weak = true;
                    points[before].after = after;
                    points[after].before = before;
                }
            }
        }
        Ok(Self {
            points,
            contours,
            major,
        })
    }

    pub fn segments(&self, axis: usize, units: i64) -> Result<Vec<Segment>> {
        if axis > 1 {
            return Err(invalid());
        }
        let mut segments = Vec::<Segment>::new();
        for &contour in &self.contours {
            let mut start = contour;
            let previous = self.points[start].previous;
            if self.points[previous].outgoing.abs() == self.major[axis].abs()
                && self.points[start].outgoing.abs() == self.major[axis].abs()
            {
                loop {
                    start = self.points[start].previous;
                    if self.points[start].outgoing.abs() != self.major[axis].abs() {
                        start = self.points[start].next;
                        break;
                    }
                    if start == contour {
                        break;
                    }
                }
            }
            let mut point = start;
            let mut previous_segment: Option<(usize, Bounds)> = None;
            loop {
                let direction = self.points[point].outgoing;
                if direction.abs() == self.major[axis].abs() || self.points[point].previous == point
                {
                    if segments.len() > 1000 {
                        return Ok(Vec::new());
                    }
                    let first = point;
                    let mut bounds = Bounds::new(&self.points[point], axis);
                    if self.points[point].previous != point {
                        loop {
                            point = self.points[point].next;
                            bounds.include(&self.points[point], axis);
                            if self.points[point].outgoing != direction || point == start {
                                break;
                            }
                        }
                    }
                    let mut segment = bounds.segment(first, point, direction, units / 14);
                    if let Some((index, old_bounds)) = previous_segment {
                        if first == segments[index].last {
                            let mut merged = bounds;
                            if self.points[segments[index].last].incoming
                                == self.points[point].incoming
                            {
                                merged.merge(old_bounds);
                                segment = merged.segment(
                                    segments[index].first,
                                    point,
                                    segments[index].direction,
                                    units / 14,
                                );
                            } else if old_bounds.maximum - old_bounds.minimum
                                > bounds.maximum - bounds.minimum
                            {
                                merged = old_bounds;
                                merged.min_position = merged.min_position.min(bounds.min_position);
                                merged.max_position = merged.max_position.max(bounds.max_position);
                                segment = segments[index].clone();
                                segment.last = point;
                                segment.position = (merged.min_position + merged.max_position) >> 1;
                                segment.delta = (merged.max_position - merged.min_position) >> 1;
                            } else {
                                merged.min_position =
                                    merged.min_position.min(old_bounds.min_position);
                                merged.max_position =
                                    merged.max_position.max(old_bounds.max_position);
                                segment = merged.segment(first, point, direction, units / 14);
                            }
                            segments[index] = segment;
                            previous_segment = Some((index, merged));
                        } else {
                            let index = segments.len();
                            segments.push(segment);
                            previous_segment = Some((index, bounds));
                        }
                    } else {
                        let index = segments.len();
                        segments.push(segment);
                        previous_segment = Some((index, bounds));
                    }
                    if point == start {
                        break;
                    }
                    // The last point of one segment can start the next segment.
                    if self.points[point].outgoing.abs() == self.major[axis].abs() {
                        continue;
                    }
                }
                point = self.points[point].next;
                if point == start {
                    break;
                }
            }
        }
        for segment in &mut segments {
            let first = coordinate(self.points[segment.first].design, 1 - axis);
            let last = coordinate(self.points[segment.last].design, 1 - axis);
            let previous = coordinate(
                self.points[self.points[segment.first].previous].design,
                1 - axis,
            );
            let next = coordinate(self.points[self.points[segment.last].next].design, 1 - axis);
            if first < last {
                if previous < first {
                    segment.height += (first - previous) >> 1;
                }
                if next > last {
                    segment.height += (next - last) >> 1;
                }
            } else {
                if previous > first {
                    segment.height += (previous - first) >> 1;
                }
                if next < last {
                    segment.height += (last - next) >> 1;
                }
            }
        }
        Ok(segments)
    }

    pub fn link(
        &self,
        segments: &mut [Segment],
        axis: usize,
        units: i64,
        widths: &[i64],
    ) -> Result<()> {
        if axis > 1 {
            return Err(invalid());
        }
        let maximum_width = widths.last().copied().unwrap_or(0);
        let threshold = (8 * units / 2048).max(1);
        for first in 0..segments.len() {
            if segments[first].direction != self.major[axis] {
                continue;
            }
            for second in 0..segments.len() {
                let a = &segments[first];
                let b = &segments[second];
                if a.direction + b.direction != 0 || b.position <= a.position {
                    continue;
                }
                let overlap = a.maximum.min(b.maximum) - a.minimum.max(b.minimum);
                if overlap < threshold {
                    continue;
                }
                let distance = b.position - a.position;
                let demerit = if maximum_width > 0 {
                    let delta = (distance << 10) / maximum_width - 1024;
                    if delta > 10000 {
                        32000
                    } else if delta > 0 {
                        delta * delta / 3000
                    } else {
                        0
                    }
                } else {
                    distance
                };
                let score = demerit + (6000 * units / 2048) / overlap;
                if score < segments[first].score {
                    segments[first].score = score;
                    segments[first].link = Some(second);
                }
                if score < segments[second].score {
                    segments[second].score = score;
                    segments[second].link = Some(first);
                }
            }
        }
        for index in 0..segments.len() {
            if let Some(partner) = segments[index].link {
                if segments[partner].link != Some(index) {
                    segments[index].serif = segments[partner].link;
                    segments[index].link = None;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min_position: i64,
    max_position: i64,
    minimum: i64,
    maximum: i64,
    min_control: bool,
    max_control: bool,
    min_on: i64,
    max_on: i64,
}

impl Bounds {
    fn new(point: &Point, axis: usize) -> Self {
        let u = coordinate(point.design, axis);
        let v = coordinate(point.design, 1 - axis);
        Self {
            min_position: u,
            max_position: u,
            minimum: v,
            maximum: v,
            min_control: point.control,
            max_control: point.control,
            min_on: if point.control { 32000 } else { v },
            max_on: if point.control { -32000 } else { v },
        }
    }

    fn include(&mut self, point: &Point, axis: usize) {
        let u = coordinate(point.design, axis);
        let v = coordinate(point.design, 1 - axis);
        self.min_position = self.min_position.min(u);
        self.max_position = self.max_position.max(u);
        if v < self.minimum {
            self.minimum = v;
            self.min_control = point.control;
        }
        if v > self.maximum {
            self.maximum = v;
            self.max_control = point.control;
        }
        if !point.control {
            self.min_on = self.min_on.min(v);
            self.max_on = self.max_on.max(v);
        }
    }

    fn merge(&mut self, other: Self) {
        self.min_position = self.min_position.min(other.min_position);
        self.max_position = self.max_position.max(other.max_position);
        if other.minimum < self.minimum {
            self.minimum = other.minimum;
            self.min_control = other.min_control;
        }
        if other.maximum > self.maximum {
            self.maximum = other.maximum;
            self.max_control = other.max_control;
        }
        self.min_on = self.min_on.min(other.min_on);
        self.max_on = self.max_on.max(other.max_on);
    }

    fn segment(self, first: usize, last: usize, direction: i8, flat_threshold: i64) -> Segment {
        Segment {
            first,
            last,
            direction,
            position: (self.min_position + self.max_position) >> 1,
            delta: (self.max_position - self.min_position) >> 1,
            minimum: self.minimum,
            maximum: self.maximum,
            height: self.maximum - self.minimum,
            round: (self.min_control || self.max_control)
                && self.max_on - self.min_on < flat_threshold,
            link: None,
            serif: None,
            edge: None,
            score: 32000,
        }
    }
}
