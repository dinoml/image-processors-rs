//! Native 2D/3D Cellpose Euler dynamics, density basins and mask quality control.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::{elements, invalid, Result};

/// Planar flow components and cell-probability logits on a 2D image or 3D volume.
#[derive(Clone, Debug)]
pub struct CellposeField {
    /// YX or ZYX dimensions.
    pub shape: Vec<usize>,
    /// One contiguous plane per spatial flow component, in shape-axis order.
    pub flows: Vec<f32>,
    /// Unactivated cell-probability logits.
    pub cell_probability: Vec<f32>,
}

/// Flow-to-instance dynamics and filtering options.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CellposeMaskOptions {
    /// Euler steps; zero selects the diameter-dependent default in predict.
    pub iterations: usize,
    /// Minimum cell-probability logit, with strict greater-than selection.
    pub cellprob_threshold: f32,
    /// Maximum mean squared flow error; nonpositive disables 2D QC.
    pub flow_threshold: f64,
    /// Remove labels smaller than this many pixels after filling holes.
    pub min_size: usize,
    /// Remove labels larger than this fraction of all pixels/voxels.
    pub max_size_fraction: f64,
}

impl Default for CellposeMaskOptions {
    fn default() -> Self {
        Self {
            iterations: 0,
            cellprob_threshold: 0.0,
            flow_threshold: 0.4,
            min_size: 15,
            max_size_fraction: 0.4,
        }
    }
}

fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut point = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        point[axis] = index % shape[axis];
        index /= shape[axis];
    }
    point
}

fn flat(point: &[usize], shape: &[usize]) -> usize {
    point
        .iter()
        .zip(shape)
        .fold(0, |index, (&position, &extent)| index * extent + position)
}

fn neighbor(point: &[usize], delta: &[isize], shape: &[usize]) -> Option<usize> {
    let mut index = 0;
    for ((&p, &d), &size) in point.iter().zip(delta).zip(shape) {
        let p = p.checked_add_signed(d)?;
        if p >= size {
            return None;
        }
        index = index * size + p;
    }
    Some(index)
}

fn offsets(dimensions: usize, radius: isize, corners: bool) -> Vec<Vec<isize>> {
    let mut result = vec![vec![]];
    for _ in 0..dimensions {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                (-radius..=radius).map(move |d| {
                    let mut next = prefix.clone();
                    next.push(d);
                    next
                })
            })
            .collect();
    }
    if !corners {
        result.retain(|d| d.iter().map(|x| x.abs()).sum::<isize>() == 1);
    }
    result
}

/// Reconstruct nonnegative integer instance labels without Python dependencies.
///
/// Coordinates and fields follow upstream grid_sample(align_corners=false),
/// including zero-padding interpolation and normalized-coordinate clamping.
/// QC compares diffusion-derived unit flows to the network field divided by five.
/// # Errors
/// Rejects incompatible storage, nonfinite fields, invalid options and overflow.
pub fn reconstruct_masks(field: &CellposeField, options: &CellposeMaskOptions) -> Result<Vec<u32>> {
    let shape = &field.shape;
    if !(2..=3).contains(&shape.len()) {
        return Err(invalid("mask field must be YX or ZYX"));
    }
    let count = elements(shape)?;
    let dimensions = shape.len();
    if field.flows.len() != elements(&[dimensions, count])?
        || field.cell_probability.len() != count
        || field
            .flows
            .iter()
            .chain(&field.cell_probability)
            .any(|v| !v.is_finite())
        || !options.cellprob_threshold.is_finite()
        || !options.flow_threshold.is_finite()
        || !options.max_size_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.max_size_fraction)
    {
        return Err(invalid("invalid flow field or reconstruction options"));
    }
    let indices = (0..count)
        .filter(|&i| field.cell_probability[i] > options.cellprob_threshold)
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Ok(vec![0; count]);
    }
    let mut positions = indices
        .iter()
        .map(|&i| {
            coordinates(i, shape)
                .iter()
                .zip(shape)
                .map(|(&p, &size)| {
                    if size == 1 {
                        0.0
                    } else {
                        (p as f32 / (size - 1) as f32) * 2.0 - 1.0
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let iterations = if options.iterations == 0 {
        200
    } else {
        options.iterations
    };
    for _ in 0..iterations {
        for point in &mut positions {
            let sample = point
                .iter()
                .zip(shape)
                .map(|(&p, &size)| ((p + 1.0) * size as f32 - 1.0) * 0.5)
                .collect::<Vec<_>>();
            let low = sample
                .iter()
                .map(|p| p.floor() as isize)
                .collect::<Vec<_>>();
            let fraction = sample
                .iter()
                .zip(&low)
                .map(|(&p, &l)| p - l as f32)
                .collect::<Vec<_>>();
            let mut update = vec![0.0f32; dimensions];
            for corner in 0..(1 << dimensions) {
                let mut source = 0;
                let mut weight = 1.0;
                let mut valid = true;
                for axis in 0..dimensions {
                    let high = (corner >> (dimensions - 1 - axis)) & 1;
                    let p = low[axis] + high as isize;
                    if p < 0 || p >= shape[axis] as isize {
                        valid = false;
                        break;
                    }
                    source = source * shape[axis] + p as usize;
                    weight *= if high == 0 {
                        1.0 - fraction[axis]
                    } else {
                        fraction[axis]
                    };
                }
                if valid && field.cell_probability[source] > options.cellprob_threshold {
                    for axis in 0..dimensions {
                        let factor = if shape[axis] == 1 {
                            0.0
                        } else {
                            2.0 / (shape[axis] - 1) as f32
                        };
                        update[axis] +=
                            (field.flows[axis * count + source] / 5.0) * factor * weight;
                    }
                }
            }
            for axis in 0..dimensions {
                point[axis] = (point[axis] + update[axis]).clamp(-1.0, 1.0);
            }
        }
    }
    let padded = shape
        .iter()
        .map(|n| {
            n.checked_add(40)
                .ok_or_else(|| invalid("histogram overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut histogram = vec![0usize; elements(&padded)?];
    let mut destinations = Vec::with_capacity(indices.len());
    for point in &positions {
        let point = point
            .iter()
            .zip(shape)
            .map(|(&p, &size)| (((p + 1.0) * 0.5 * (size - 1) as f32) as usize).min(size - 1) + 20)
            .collect::<Vec<_>>();
        let destination = flat(&point, &padded);
        histogram[destination] += 1;
        destinations.push(destination);
    }
    let maxima = offsets(dimensions, 2, true);
    let mut seeds = (0..histogram.len())
        .filter(|&i| histogram[i] > 10)
        .filter(|&i| {
            let p = coordinates(i, &padded);
            maxima
                .iter()
                .all(|d| neighbor(&p, d, &padded).is_none_or(|j| histogram[j] <= histogram[i]))
        })
        .collect::<Vec<_>>();
    seeds.sort_by_key(|&i| (histogram[i], i));
    let mut basin = vec![0u32; histogram.len()];
    let adjacent = offsets(dimensions, 1, true);
    for (seed_number, &seed) in seeds.iter().enumerate() {
        let label =
            u32::try_from(seed_number + 1).map_err(|_| invalid("too many instance labels"))?;
        let center = coordinates(seed, &padded);
        let mut grown = std::collections::BTreeSet::from([seed]);
        for _ in 0..5 {
            let mut next = std::collections::BTreeSet::new();
            for &index in &grown {
                let p = coordinates(index, &padded);
                for d in &adjacent {
                    if let Some(j) = neighbor(&p, d, &padded) {
                        let q = coordinates(j, &padded);
                        if histogram[j] > 2
                            && q.iter().zip(&center).all(|(&a, &b)| a.abs_diff(b) <= 5)
                        {
                            next.insert(j);
                        }
                    }
                }
            }
            grown = next;
        }
        for index in grown {
            basin[index] = basin[index].max(label);
        }
    }
    let mut masks = vec![0u32; count];
    for (&index, &destination) in indices.iter().zip(&destinations) {
        masks[index] = basin[destination];
    }
    let mut sizes = BTreeMap::<u32, usize>::new();
    for &label in &masks {
        if label != 0 {
            *sizes.entry(label).or_default() += 1;
        }
    }
    let maximum = count as f64 * options.max_size_fraction;
    for label in &mut masks {
        if sizes.get(label).is_some_and(|&n| n as f64 > maximum) {
            *label = 0;
        }
    }
    renumber(&mut masks);
    if dimensions == 2 && options.flow_threshold > 0.0 {
        quality_control(&mut masks, field, options.flow_threshold)?;
    }
    if options.min_size > 0 {
        fill_and_filter(&mut masks, shape, options.min_size)?;
    }
    Ok(masks)
}

fn renumber(masks: &mut [u32]) {
    let mut labels = masks
        .iter()
        .copied()
        .filter(|&n| n > 0)
        .map(|n| (n, 0u32))
        .collect::<BTreeMap<_, _>>();
    for (index, value) in labels.values_mut().enumerate() {
        *value = (index + 1) as u32;
    }
    for label in masks {
        if *label > 0 {
            *label = labels[label];
        }
    }
}

fn quality_control(masks: &mut [u32], field: &CellposeField, threshold: f64) -> Result<()> {
    let (height, width) = (field.shape[0], field.shape[1]);
    let mut pixels = BTreeMap::<u32, Vec<usize>>::new();
    for (i, &label) in masks.iter().enumerate() {
        if label > 0 {
            pixels.entry(label).or_default().push(i);
        }
    }
    let maximum = pixels
        .values()
        .map(|p| {
            let ymin = p.iter().map(|i| i / width).min().unwrap_or(0);
            let ymax = p.iter().map(|i| i / width).max().unwrap_or(0);
            let xmin = p.iter().map(|i| i % width).min().unwrap_or(0);
            let xmax = p.iter().map(|i| i % width).max().unwrap_or(0);
            2 * (ymax - ymin + xmax - xmin + 4)
        })
        .max()
        .unwrap_or(0);
    for points in pixels.values() {
        let ymin = points.iter().map(|i| i / width).min().unwrap_or(0);
        let ymax = points.iter().map(|i| i / width).max().unwrap_or(0);
        let xmin = points.iter().map(|i| i % width).min().unwrap_or(0);
        let xmax = points.iter().map(|i| i % width).max().unwrap_or(0);
        let local = [ymax - ymin + 3, xmax - xmin + 3];
        let mean_y = (points.iter().map(|i| i / width).sum::<usize>() as f64 / points.len() as f64)
            .round_ties_even();
        let mean_x = (points.iter().map(|i| i % width).sum::<usize>() as f64 / points.len() as f64)
            .round_ties_even();
        let center = *points
            .iter()
            .min_by(|&&a, &&b| {
                let distance = |i: usize| (i / width) as f64 - mean_y;
                let xdistance = |i: usize| (i % width) as f64 - mean_x;
                (distance(a).powi(2) + xdistance(a).powi(2))
                    .total_cmp(&(distance(b).powi(2) + xdistance(b).powi(2)))
            })
            .ok_or_else(|| invalid("empty labeled mask"))?;
        let center = (center / width - ymin + 1) * local[1] + center % width - xmin + 1;
        let points = points
            .iter()
            .map(|&p| (p, (p / width - ymin + 1) * local[1] + p % width - xmin + 1))
            .collect::<Vec<_>>();
        let mut heat = vec![0.0f64; elements(&local)?];
        let mut member = vec![false; heat.len()];
        for &(_, p) in &points {
            member[p] = true;
        }
        let reduced = (height + 2) as f64 * (width + 2) as f64 > 4e7;
        for _ in 0..maximum {
            heat[center] += 1.0;
            let mut next = vec![0.0; heat.len()];
            for &(_, p) in &points {
                let mut sum = 0.0f64;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let q = p
                            .checked_add_signed(dy * local[1] as isize + dx)
                            .ok_or_else(|| invalid("diffusion index overflow"))?;
                        if member[q] {
                            sum += heat[q];
                            if reduced {
                                sum = f64::from(sum as f32);
                            }
                        }
                    }
                }
                next[p] = if reduced {
                    f64::from((sum / 9.0) as f32)
                } else {
                    sum / 9.0
                };
            }
            heat = next;
        }
        let mut error = 0.0f64;
        for &(original, p) in &points {
            let dy = heat[p + local[1]] - heat[p - local[1]];
            let dx = heat[p + 1] - heat[p - 1];
            let norm = (dy * dy + dx * dx).sqrt() + 1e-60;
            error += (dy / norm - f64::from(field.flows[original] / 5.0)).powi(2)
                + (dx / norm - f64::from(field.flows[height * width + original] / 5.0)).powi(2);
        }
        if error / points.len() as f64 > threshold {
            for &(original, _) in &points {
                masks[original] = 0;
            }
        }
    }
    renumber(masks);
    Ok(())
}

pub(super) fn fill_and_filter(masks: &mut [u32], shape: &[usize], min_size: usize) -> Result<()> {
    let count = elements(shape)?;
    if masks.len() != count {
        return Err(invalid("mask geometry differs"));
    }
    let mut labels = BTreeMap::<u32, Vec<usize>>::new();
    for (i, &label) in masks.iter().enumerate() {
        if label > 0 {
            labels.entry(label).or_default().push(i);
        }
    }
    let adjacency = offsets(shape.len(), 1, false);
    for (&label, points) in &labels {
        if points.len() < min_size {
            for &p in points {
                masks[p] = 0;
            }
            continue;
        }
        let mut low = shape.to_vec();
        let mut high = vec![0; shape.len()];
        for &p in points {
            for (axis, c) in coordinates(p, shape).into_iter().enumerate() {
                low[axis] = low[axis].min(c);
                high[axis] = high[axis].max(c);
            }
        }
        let box_shape = low
            .iter()
            .zip(&high)
            .map(|(&a, &b)| b - a + 3)
            .collect::<Vec<_>>();
        let mut inside = vec![false; elements(&box_shape)?];
        for &p in points {
            let p = coordinates(p, shape)
                .iter()
                .zip(&low)
                .map(|(&a, &b)| a - b + 1)
                .collect::<Vec<_>>();
            inside[flat(&p, &box_shape)] = true;
        }
        let mut exterior = vec![false; inside.len()];
        let mut queue = VecDeque::from([0usize]);
        exterior[0] = true;
        while let Some(index) = queue.pop_front() {
            let p = coordinates(index, &box_shape);
            for delta in &adjacency {
                if let Some(j) = neighbor(&p, delta, &box_shape) {
                    if !inside[j] && !exterior[j] {
                        exterior[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        for (i, &outside) in exterior.iter().enumerate() {
            if !outside {
                let p = coordinates(i, &box_shape)
                    .iter()
                    .zip(&low)
                    .map(|(&a, &b)| a + b - 1)
                    .collect::<Vec<_>>();
                masks[flat(&p, shape)] = label;
            }
        }
    }
    renumber(masks);
    Ok(())
}

#[cfg(test)]
mod mask_filter_original_reference {
    use super::fill_and_filter;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        shape: Vec<usize>,
        min_size: usize,
        input: Vec<u32>,
        expected: Vec<u32>,
    }

    #[test]
    fn matches_unmodified_original_hole_and_size_filter() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/cellpose_mask_filter_original.json"
        ))
        .expect("valid original Cellpose hole and size filter fixtures");
        let mut failures = Vec::new();
        for case in fixture.cases {
            let mut actual = case.input;
            if let Err(error) = fill_and_filter(&mut actual, &case.shape, case.min_size) {
                failures.push(format!("{}: unexpected error {error}", case.name));
                continue;
            }
            assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
            if let Some((index, (a, expected))) = actual
                .iter()
                .zip(&case.expected)
                .enumerate()
                .find(|(_, (a, b))| a != b)
            {
                failures.push(format!(
                    "{}: index {index}, actual {a}, original {expected}",
                    case.name
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
