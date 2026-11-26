//! Adobe-style character-space hint maps for the native CFF grayscale path.

use super::{
    cff::{Command, Glyph, Point, Private, Stem},
    gray::{Position, Segment},
    hinting::mul_div,
    invalid, Result,
};

const ONE: i64 = 65536;
const COUNTER: i64 = ONE / 2;
fn round(value: i64) -> i64 {
    (value + ONE / 2) & !(ONE - 1)
}
fn multiply(a: i64, b: i64) -> Result<i64> {
    mul_div(a, b, ONE)
}

#[derive(Clone)]
struct Edge {
    design: i64,
    device: i64,
    scale: i64,
    pair: bool,
    top: bool,
    locked: bool,
    synthetic: bool,
    stem: usize,
}

#[derive(Clone)]
struct Map {
    scale: i64,
    edges: Vec<Edge>,
}

impl Map {
    fn new(scale: i64) -> Self {
        Self {
            scale,
            edges: Vec::new(),
        }
    }

    fn map(&self, coordinate: i64) -> Result<i64> {
        if self.edges.is_empty() {
            return multiply(coordinate, self.scale);
        }
        let index = self
            .edges
            .partition_point(|edge| edge.design <= coordinate)
            .saturating_sub(1);
        let edge = &self.edges[index];
        Ok(edge.device
            + multiply(
                coordinate - edge.design,
                if coordinate < self.edges[0].design {
                    self.scale
                } else {
                    edge.scale
                },
            )?)
    }

    fn insert(
        &mut self,
        mut lower: Edge,
        mut upper: Option<Edge>,
        initial: Option<&Map>,
    ) -> Result<()> {
        if upper
            .as_ref()
            .is_some_and(|upper| upper.design < lower.design)
        {
            return Ok(());
        }
        let index = self
            .edges
            .partition_point(|edge| edge.design < lower.design);
        if let Some(existing) = self.edges.get(index) {
            if existing.design == lower.design
                || upper
                    .as_ref()
                    .is_some_and(|upper| existing.design <= upper.design)
                || existing.pair && existing.top
            {
                return Ok(());
            }
        }
        if !lower.locked {
            if let Some(initial) = initial {
                if let Some(upper) = &mut upper {
                    let half = (upper.design - lower.design) / 2;
                    let middle = initial.map(lower.design + half)?;
                    let half = multiply(half, self.scale)?;
                    lower.device = middle - half;
                    upper.device = middle + half;
                } else {
                    lower.device = initial.map(lower.design)?;
                }
            }
        }
        if index > 0 && lower.device < self.edges[index - 1].device {
            return Ok(());
        }
        if let Some(existing) = self.edges.get(index) {
            if upper.as_ref().map_or(lower.device, |upper| upper.device) > existing.device {
                return Ok(());
            }
        }
        if self.edges.len() + 1 + usize::from(upper.is_some()) > 192 {
            return Ok(());
        }
        self.edges.insert(index, lower);
        if let Some(upper) = upper {
            self.edges.insert(index + 1, upper);
        }
        Ok(())
    }

    fn adjust(&mut self) -> Result<()> {
        let mut saved = Vec::new();
        let mut first = 0;
        while first < self.edges.len() {
            let last = first + usize::from(self.edges[first].pair);
            if last >= self.edges.len() {
                return Err(invalid());
            }
            if !self.edges[first].locked {
                let lower = self.edges[first].device;
                let upper = self.edges[last].device;
                let fraction_lower = lower & (ONE - 1);
                let fraction_upper = upper & (ONE - 1);
                let down = (-fraction_lower).max(-fraction_upper);
                let up = (if fraction_lower == 0 {
                    0
                } else {
                    ONE - fraction_lower
                })
                .min(if fraction_upper == 0 {
                    0
                } else {
                    ONE - fraction_upper
                });
                let can_up = last + 1 == self.edges.len()
                    || self.edges[last + 1].device >= upper + up + COUNTER;
                let can_down = first == 0 || self.edges[first - 1].device <= lower + down - COUNTER;
                let (movement, save) = match (can_up, can_down) {
                    (true, true) => (if -down < up { down } else { up }, false),
                    (true, false) => (up, false),
                    (false, true) => (down, up < -down),
                    (false, false) => (0, true),
                };
                if save && last + 1 < self.edges.len() && !self.edges[last + 1].locked {
                    saved.push((last, up - movement));
                }
                self.edges[first].device += movement;
                if first != last {
                    self.edges[last].device += movement;
                }
            }
            if first > 0 && self.edges[first].design != self.edges[first - 1].design {
                self.edges[first - 1].scale = mul_div(
                    self.edges[first].device - self.edges[first - 1].device,
                    ONE,
                    self.edges[first].design - self.edges[first - 1].design,
                )?;
            }
            if last != first && self.edges[last].design != self.edges[first].design {
                self.edges[first].scale = mul_div(
                    self.edges[last].device - self.edges[first].device,
                    ONE,
                    self.edges[last].design - self.edges[first].design,
                )?;
            }
            first = last + 1;
        }
        for (last, movement) in saved.into_iter().rev() {
            if self.edges[last + 1].device >= self.edges[last].device + movement + COUNTER {
                self.edges[last].device += movement;
                if self.edges[last].pair {
                    self.edges[last - 1].device += movement;
                }
            }
        }
        Ok(())
    }
}

struct BlueZone {
    lower: i64,
    upper: i64,
    flat: i64,
    device: i64,
    bottom: bool,
}
struct Blues {
    zones: Vec<BlueZone>,
    synthetic: Vec<Edge>,
    shift: i64,
    fuzz: i64,
    suppress: bool,
}

impl Blues {
    fn new(private: &Private<'_>, scale: i64) -> Result<Self> {
        let value = |operator, default| {
            private
                .values
                .get(&operator)
                .and_then(|values| values.first())
                .copied()
                .unwrap_or(default)
        };
        let delta = |operator| -> Vec<i64> {
            let mut position = 0;
            private
                .values
                .get(&operator)
                .into_iter()
                .flatten()
                .map(|value| {
                    position += value >> 16;
                    position * ONE
                })
                .collect()
        };
        let values = delta(6);
        let others = delta(7);
        let family = delta(8);
        let family_others = delta(9);
        if [&values, &others, &family, &family_others]
            .iter()
            .any(|values| values.len() % 2 != 0)
        {
            return Err(invalid());
        }
        let shift = (value(0x0c0a, 7 * ONE) >> 16) * ONE;
        let fuzz = (value(0x0c0b, ONE) >> 16) * ONE;
        let mut result = Self {
            zones: Vec::new(),
            synthetic: Vec::new(),
            shift,
            fuzz,
            suppress: false,
        };
        if value(0x0c11, 0) / ONE == 1
            && (values.is_empty()
                || values.len() == 4
                    && values[0] < -120 * ONE
                    && values[1] < -120 * ONE
                    && values[2] > 880 * ONE
                    && values[3] > 880 * ONE)
        {
            for (design, top) in [(-120 * ONE - 1, false), (880 * ONE + 1, true)] {
                result.synthetic.push(Edge {
                    design,
                    device: round(multiply(design, scale)?) + if top { COUNTER } else { -COUNTER },
                    scale,
                    pair: false,
                    top,
                    locked: true,
                    synthetic: true,
                    stem: 0,
                });
            }
            return Ok(result);
        }
        let mut maximum = 0;
        for (other, values) in [(false, &values), (true, &others)] {
            for (index, pair) in values.chunks_exact(2).enumerate() {
                if pair[1] < pair[0] {
                    continue;
                }
                maximum = maximum.max(pair[1] - pair[0]);
                let bottom = other || index == 0;
                result.zones.push(BlueZone {
                    lower: pair[0],
                    upper: pair[1],
                    flat: if bottom { pair[1] } else { pair[0] },
                    device: 0,
                    bottom,
                });
            }
        }
        let pixel = mul_div(ONE, ONE, scale)?;
        for zone in &mut result.zones {
            let original = zone.flat;
            let mut difference = i64::from(i32::MAX);
            if zone.bottom {
                for pair in family_others.chunks_exact(2) {
                    let distance = (original - pair[1]).abs();
                    if distance < difference && distance < pixel {
                        difference = distance;
                        zone.flat = pair[1];
                        if distance == 0 {
                            break;
                        }
                    }
                }
                if family.len() >= 2 {
                    let distance = (original - family[1]).abs();
                    if distance < difference && distance < pixel {
                        zone.flat = family[1];
                    }
                }
            } else {
                for pair in family.chunks_exact(2).skip(1) {
                    let distance = (original - pair[0]).abs();
                    if distance < difference && distance < pixel {
                        difference = distance;
                        zone.flat = pair[0];
                        if distance == 0 {
                            break;
                        }
                    }
                }
            }
        }
        let mut blue_scale = value(0x0c09, 2597);
        if maximum > 0 {
            blue_scale = blue_scale.min(mul_div(ONE, ONE, maximum)?);
        }
        let boost = if scale < blue_scale {
            result.suppress = true;
            (39322 - mul_div(39322, scale, blue_scale)?).min(32767)
        } else {
            0
        };
        for zone in &mut result.zones {
            zone.device =
                round(multiply(zone.flat, scale)? + if zone.bottom { -boost } else { boost });
        }
        Ok(result)
    }

    fn capture(&self, lower: &mut Option<Edge>, upper: &mut Option<Edge>) -> bool {
        for zone in &self.zones {
            let selected = if zone.bottom {
                lower.as_ref()
            } else {
                upper.as_ref()
            };
            if let Some(edge) = selected {
                if edge.design < zone.lower - self.fuzz || edge.design > zone.upper + self.fuzz {
                    continue;
                }
                let target = if self.suppress {
                    zone.device
                } else if zone.bottom && zone.upper - edge.design >= self.shift {
                    round(edge.device).min(zone.device - ONE)
                } else if !zone.bottom && edge.design - zone.lower >= self.shift {
                    round(edge.device).max(zone.device + ONE)
                } else {
                    round(edge.device)
                };
                let movement = target - edge.device;
                for edge in [lower.as_mut(), upper.as_mut()].into_iter().flatten() {
                    edge.device += movement;
                    edge.locked = true;
                }
                return true;
            }
        }
        false
    }
}

struct Hints<'a> {
    scale: i64,
    stems: &'a [Stem],
    used: Vec<Option<[i64; 2]>>,
    blues: Blues,
    initial: Option<Map>,
}

impl<'a> Hints<'a> {
    fn pair(&self, index: usize) -> Result<(Option<Edge>, Option<Edge>)> {
        let stem = self.stems.get(index).ok_or_else(invalid)?;
        let (lower, upper, pair) = if stem.width == -21 * ONE {
            (Some(stem.start + stem.width), None, false)
        } else if stem.width == -20 * ONE {
            (None, Some(stem.start), false)
        } else if stem.width < 0 {
            (Some(stem.start + stem.width), Some(stem.start), true)
        } else {
            (Some(stem.start), Some(stem.start + stem.width), true)
        };
        let edge = |design: Option<i64>, top: bool| -> Result<Option<Edge>> {
            design
                .map(|design| {
                    Ok(Edge {
                        design,
                        device: if let Some(used) = self.used[index] {
                            used[usize::from(top)]
                        } else {
                            multiply(design, self.scale)?
                        },
                        scale: self.scale,
                        pair,
                        top,
                        locked: self.used[index].is_some(),
                        synthetic: false,
                        stem: index,
                    })
                })
                .transpose()
        };
        Ok((edge(lower, false)?, edge(upper, true)?))
    }

    fn build(&mut self, mask: &[u8], initial: bool) -> Result<Map> {
        if self.stems.len() > 96 {
            return Err(invalid());
        }
        if !initial && self.initial.is_none() {
            let all = vec![255; self.stems.len().div_ceil(8)];
            self.initial = Some(self.build(&all, true)?);
        }
        let mut map = Map::new(self.scale);
        for edge in &self.blues.synthetic {
            map.insert(edge.clone(), None, self.initial.as_ref())?;
        }
        let mut remaining = Vec::new();
        for (index, stem) in self.stems.iter().enumerate() {
            if stem.vertical
                || mask
                    .get(index / 8)
                    .is_none_or(|value| value & (128 >> (index & 7)) == 0)
            {
                continue;
            }
            let (mut lower, mut upper) = self.pair(index)?;
            let locked = lower.as_ref().is_some_and(|edge| edge.locked)
                || upper.as_ref().is_some_and(|edge| edge.locked);
            if locked || self.blues.capture(&mut lower, &mut upper) {
                if let Some(lower) = lower {
                    map.insert(lower, upper, self.initial.as_ref())?;
                } else if let Some(upper) = upper {
                    map.insert(upper, None, self.initial.as_ref())?;
                }
            } else {
                remaining.push(index);
            }
        }
        if initial {
            if map.edges.is_empty()
                || map.edges[0].design > 0
                || map.edges.last().is_some_and(|edge| edge.design < 0)
            {
                map.insert(
                    Edge {
                        design: 0,
                        device: 0,
                        scale: self.scale,
                        pair: false,
                        top: false,
                        locked: true,
                        synthetic: true,
                        stem: 0,
                    },
                    None,
                    None,
                )?;
            }
        } else {
            for index in remaining {
                let (lower, upper) = self.pair(index)?;
                if let Some(lower) = lower {
                    map.insert(lower, upper, self.initial.as_ref())?;
                } else if let Some(upper) = upper {
                    map.insert(upper, None, self.initial.as_ref())?;
                }
            }
        }
        map.adjust()?;
        if !initial {
            for edge in &map.edges {
                if edge.synthetic {
                    continue;
                }
                let entry = self.used[edge.stem].get_or_insert([0, 0]);
                entry[usize::from(edge.top)] = edge.device;
            }
        }
        Ok(map)
    }
}

pub(super) fn outline(glyph: &Glyph, private: &Private<'_>, scale: i64) -> Result<Vec<Segment>> {
    if scale <= 0 {
        return Err(invalid());
    }
    let mut hints = Hints {
        scale,
        stems: &glyph.stems,
        used: vec![None; glyph.stems.len()],
        blues: Blues::new(private, scale)?,
        initial: None,
    };
    let mut mask = vec![255; glyph.stems.len().div_ceil(8)];
    let mut map = hints.build(&mask, false)?;
    let mut first_map = map.clone();
    let mut changed = false;
    let mut masks = glyph.masks.iter().peekable();
    let mut output = Vec::new();
    let mut start = Point::default();
    let mut current = start;
    let mut queued = None::<Command>;
    let mut open = false;
    let mut pending_move = false;
    let point = |point: Point, map: &Map| -> Result<Position> {
        Ok(Position {
            x: multiply(point.x, scale)? >> 10,
            y: map.map(point.y)? >> 10,
        })
    };
    let flush = |command: Command, map: &Map, output: &mut Vec<Segment>| -> Result<()> {
        match command {
            Command::Line(end) => output.push(Segment::Line(point(end, map)?)),
            Command::Curve(a, b, end) => output.push(Segment::Cubic(
                point(a, map)?,
                point(b, map)?,
                point(end, map)?,
            )),
            _ => return Err(invalid()),
        }
        Ok(())
    };
    for (index, &command) in glyph.commands.iter().enumerate() {
        while masks.peek().is_some_and(|entry| entry.command == index) {
            let entry = masks.next().ok_or_else(invalid)?;
            if entry.counter {
                hints.build(&entry.bits, false)?;
            } else {
                mask.clone_from(&entry.bits);
                changed = true;
            }
        }
        match command {
            Command::Move(position) => {
                if open {
                    if let Some(command) = queued.take() {
                        flush(command, &map, &mut output)?;
                    }
                    output.push(Segment::Line(point(start, &first_map)?));
                }
                if changed {
                    map = hints.build(&mask, false)?;
                    changed = false;
                }
                first_map = map.clone();
                start = position;
                current = position;
                pending_move = true;
                open = false;
            }
            Command::Line(end) | Command::Curve(_, _, end) => {
                if matches!(command, Command::Line(_))
                    && current.x == end.x
                    && current.y == end.y
                    && !changed
                {
                    continue;
                }
                if pending_move {
                    output.push(Segment::Move(point(current, &map)?));
                    pending_move = false;
                    open = true;
                }
                if let Some(previous) = queued.take() {
                    flush(previous, &map, &mut output)?;
                }
                if changed {
                    map = hints.build(&mask, false)?;
                    changed = false;
                }
                queued = Some(command);
                current = end;
            }
            Command::Close => {
                if open {
                    if let Some(previous) = queued.take() {
                        let closing_map = if matches!(previous, Command::Line(end) if end.x == start.x && end.y == start.y)
                        {
                            &first_map
                        } else {
                            &map
                        };
                        flush(previous, closing_map, &mut output)?;
                    }
                    output.push(Segment::Line(point(start, &first_map)?));
                }
                pending_move = true;
                open = false;
                current = start;
            }
        }
    }
    if let Some(command) = queued {
        flush(command, &map, &mut output)?;
        output.push(Segment::Line(point(start, &first_map)?));
    }
    Ok(output)
}
