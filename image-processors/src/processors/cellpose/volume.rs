//! Orthogonal-plane Cellpose volume prediction and adjacent-slice instance stitching.

use super::{
    dynamics, elements, invalid, normalize_planes, predict, reconstruct_masks, resize_planes,
    CellposeError, CellposeField, CellposeImage, CellposeOptions, CellposePrediction, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Floating-point microscopy volume in CZYX order, with three channels.
#[derive(Clone, Debug)]
pub struct CellposeVolume {
    /// ZYX dimensions.
    pub shape: [usize; 3],
    /// Three contiguous volume planes; extra channels are dropped, absent channels zero-filled.
    pub pixels: Vec<f32>,
}

impl CellposeVolume {
    /// Convert a grayscale volume or a multichannel volume with explicit axis identities.
    /// # Errors
    /// Rejects invalid axes, storage or nonfinite intensity values.
    pub fn from_array(
        values: &[f32],
        shape: &[usize],
        z_axis: usize,
        channel_axis: Option<usize>,
    ) -> Result<Self> {
        if !(3..=4).contains(&shape.len())
            || z_axis >= shape.len()
            || elements(shape)? != values.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid volume geometry/storage"));
        }
        let channel = if shape.len() == 4 {
            Some(channel_axis.ok_or_else(|| invalid("4D volume needs a channel axis"))?)
        } else {
            None
        };
        if channel.is_some_and(|c| c >= shape.len() || c == z_axis) {
            return Err(invalid("invalid volume channel axis"));
        }
        let spatial = (0..shape.len())
            .filter(|&a| a != z_axis && Some(a) != channel)
            .collect::<Vec<_>>();
        let output = [shape[z_axis], shape[spatial[0]], shape[spatial[1]]];
        let mut pixels = vec![0.0; elements(&[3, output[0], output[1], output[2]])?];
        let channels = channel.map_or(1, |c| shape[c]).min(3);
        for c in 0..channels {
            for z in 0..output[0] {
                for y in 0..output[1] {
                    for x in 0..output[2] {
                        let mut point = vec![0; shape.len()];
                        point[z_axis] = z;
                        point[spatial[0]] = y;
                        point[spatial[1]] = x;
                        if let Some(axis) = channel {
                            point[axis] = c;
                        }
                        let source = point.iter().zip(shape).fold(0, |i, (&p, &n)| i * n + p);
                        pixels[((c * output[0] + z) * output[1] + y) * output[2] + x] =
                            values[source];
                    }
                }
            }
        }
        Ok(Self {
            shape: output,
            pixels,
        })
    }
}

/// Volume inference options over the same 2D tile predictor.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CellposeVolumeOptions {
    /// Common normalization, tiling and mask options.
    pub inference: CellposeOptions,
    /// Z resampling relative to XY spacing.
    pub anisotropy: f64,
    /// Gaussian flow smoothing standard deviations along ZYX; zero disables an axis.
    pub flow_smooth: [f64; 3],
}
impl Default for CellposeVolumeOptions {
    fn default() -> Self {
        Self {
            inference: CellposeOptions::default(),
            anisotropy: 1.0,
            flow_smooth: [0.0; 3],
        }
    }
}

fn resize_volume(
    values: &[f32],
    channels: usize,
    source: [usize; 3],
    target: [usize; 3],
) -> Result<Vec<f32>> {
    if values.len() != elements(&[channels, source[0], source[1], source[2]])? {
        return Err(invalid("volume resize storage differs"));
    }
    elements(&target)?;
    let mut xy = Vec::with_capacity(elements(&[channels, source[0], target[1], target[2]])?);
    for plane in values.chunks_exact(source[1] * source[2]) {
        xy.extend(resize_planes(
            plane,
            1,
            [source[1], source[2]],
            [target[1], target[2]],
        )?);
    }
    if source[0] == target[0] {
        return Ok(xy);
    }
    // Upstream resize_image_3d performs the XY pass, then treats Z as a resize axis.
    let mut output = vec![0.0; elements(&[channels, target[0], target[1], target[2]])?];
    for c in 0..channels {
        for y in 0..target[1] {
            let mut zx = Vec::with_capacity(source[0] * target[2]);
            for z in 0..source[0] {
                zx.extend_from_slice(
                    &xy[((c * source[0] + z) * target[1] + y) * target[2]
                        ..((c * source[0] + z) * target[1] + y + 1) * target[2]],
                );
            }
            let zx = resize_planes(&zx, 1, [source[0], target[2]], [target[0], target[2]])?;
            for z in 0..target[0] {
                output[((c * target[0] + z) * target[1] + y) * target[2]
                    ..((c * target[0] + z) * target[1] + y + 1) * target[2]]
                    .copy_from_slice(&zx[z * target[2]..(z + 1) * target[2]]);
            }
        }
    }
    Ok(output)
}

/// Evaluate YX, ZX and ZY planes and combine their spatial flow components.
/// Normalization is shared across the input volume, before anisotropic resampling.
/// # Errors
/// Returns malformed geometry/options and propagates callback or processing failures.
pub fn predict_volume<E: From<CellposeError>>(
    volume: &CellposeVolume,
    tile_size: usize,
    batch_size: usize,
    options: &CellposeVolumeOptions,
    mut evaluate: impl FnMut(&[f32], usize, usize) -> std::result::Result<Vec<f32>, E>,
) -> std::result::Result<CellposePrediction, E> {
    if !options.anisotropy.is_finite() || options.anisotropy <= 0.0 {
        return Err(invalid("anisotropy must be finite and positive").into());
    }
    let scale = match options.inference.diameter {
        Some(d) if d.is_finite() && d > 0.0 => 30.0 / d,
        None => 1.0,
        _ => return Err(invalid("invalid diameter").into()),
    };
    let original = volume.shape;
    let shape = [
        (original[0] as f64 * options.anisotropy * scale) as usize,
        (original[1] as f64 * scale) as usize,
        (original[2] as f64 * scale) as usize,
    ];
    let mut pixels = volume.pixels.clone();
    normalize_planes(&mut pixels, &original, &options.inference.normalization)?;
    pixels = resize_volume(&pixels, 3, original, shape)?;
    let count = elements(&shape)?;
    let mut maps = vec![0.0; elements(&[4, shape[0], shape[1], shape[2]])?];
    let mut plane_options = options.inference.clone();
    plane_options.normalization.normalize = false;
    plane_options.normalization.lowhigh = None;
    plane_options.normalization.invert = false;
    plane_options.normalization.tile_norm_blocksize = 0;
    plane_options.normalization.smooth_radius = 0.0;
    plane_options.normalization.sharpen_radius = 0.0;
    plane_options.diameter = None;
    plane_options.resample = false;
    plane_options.compute_masks = false;
    for (outer, row, column) in [(0usize, 1usize, 2usize), (1, 0, 2), (2, 0, 1)] {
        for index in 0..shape[outer] {
            let (height, width) = (shape[row], shape[column]);
            let mut plane = vec![0.0; elements(&[3, height, width])?];
            for c in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let mut p = [0; 3];
                        p[outer] = index;
                        p[row] = y;
                        p[column] = x;
                        plane[(c * height + y) * width + x] =
                            pixels[c * count + (p[0] * shape[1] + p[1]) * shape[2] + p[2]];
                    }
                }
            }
            let result = predict(
                &CellposeImage {
                    height,
                    width,
                    pixels: plane,
                },
                tile_size,
                batch_size,
                &plane_options,
                &mut evaluate,
            )?;
            for y in 0..height {
                for x in 0..width {
                    let mut p = [0; 3];
                    p[outer] = index;
                    p[row] = y;
                    p[column] = x;
                    let target = (p[0] * shape[1] + p[1]) * shape[2] + p[2];
                    let source = y * width + x;
                    maps[row * count + target] += result.field.flows[source];
                    maps[column * count + target] += result.field.flows[height * width + source];
                    maps[3 * count + target] += result.field.cell_probability[source];
                }
            }
        }
    }
    let shape = if options.inference.resample {
        maps = resize_volume(&maps, 4, shape, original)?;
        original
    } else {
        shape
    };
    let count = elements(&shape)?;
    for (axis, &sigma) in options.flow_smooth.iter().enumerate() {
        if sigma == 0.0 {
            continue;
        }
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(invalid("invalid flow smoothing sigma").into());
        }
        // Reuse the shared reflected Gaussian filter for one axis at a time.
        let trailing = shape[axis + 1..].iter().product::<usize>();
        let leading = shape[..axis].iter().product::<usize>();
        for c in 0..3 {
            for prefix in 0..leading {
                let start = c * count + prefix * shape[axis] * trailing;
                let values = maps[start..start + shape[axis] * trailing]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>();
                let smoothed = crate::transforms::gaussian_filter_reflect_f64(
                    &values,
                    crate::transforms::ImageSize {
                        height: shape[axis],
                        width: 1,
                    },
                    trailing,
                    [sigma, 0.0],
                    4.0,
                )
                .map_err(|e| invalid(e.to_string()))?;
                for (out, value) in maps[start..start + smoothed.len()].iter_mut().zip(smoothed) {
                    *out = value as f32;
                }
            }
        }
    }
    let field = CellposeField {
        shape: shape.to_vec(),
        flows: maps[..3 * count].to_vec(),
        cell_probability: maps[3 * count..].to_vec(),
    };
    let mut mask_options = options.inference.masks.clone();
    if mask_options.iterations == 0 {
        mask_options.iterations = (200.0
            / if options.inference.resample {
                scale
            } else {
                1.0
            }) as usize;
    }
    mask_options.min_size = 0;
    let mut masks = if options.inference.compute_masks {
        reconstruct_masks(&field, &mask_options)?
    } else {
        Vec::new()
    };
    if options.inference.compute_masks && shape != original {
        let mut restored = vec![0; elements(&original)?];
        for z in 0..original[0] {
            for y in 0..original[1] {
                for x in 0..original[2] {
                    restored[(z * original[1] + y) * original[2] + x] =
                        masks[((z * shape[0] / original[0]) * shape[1]
                            + y * shape[1] / original[1])
                            * shape[2]
                            + x * shape[2] / original[2]];
                }
            }
        }
        masks = restored;
    }
    if options.inference.compute_masks {
        dynamics::fill_and_filter(&mut masks, &original, options.inference.masks.min_size)?;
    }
    Ok(CellposePrediction {
        field,
        masks,
        mask_shape: original.to_vec(),
        style: [0.0; 256],
    })
}

/// Stitch independently reconstructed adjacent slices by their maximum intersection-over-union.
/// # Errors
/// Rejects inconsistent planes, invalid thresholds and label overflow.
pub fn stitch_masks(slices: &mut [Vec<u32>], threshold: f64) -> Result<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(invalid("invalid stitch threshold"));
    }
    let length = slices.first().map_or(0, Vec::len);
    if slices.iter().any(|slice| slice.len() != length) {
        return Err(invalid("stitched slice shapes differ"));
    }
    let mut mmax = slices
        .first()
        .and_then(|p| p.iter().max())
        .copied()
        .unwrap_or(0);
    // Upstream changes its empty-slice policy only after a nonempty IoU
    // matrix, even when that matrix has no positive overlaps.
    let mut had_iou = false;
    for z in 1..slices.len() {
        let previous_max = slices[z - 1].iter().copied().max().unwrap_or(0);
        let current_max = slices[z].iter().copied().max().unwrap_or(0);
        if previous_max == 0 || current_max == 0 {
            if !had_iou {
                // Preserve original labels/gaps and reset the allocation base.
                mmax = current_max;
            } else {
                let base = mmax;
                mmax = mmax
                    .checked_add(current_max)
                    .ok_or_else(|| invalid("stitch label overflow"))?;
                for label in &mut slices[z] {
                    if *label > 0 {
                        *label = base + *label;
                    }
                }
            }
            continue;
        }
        let mut previous = BTreeMap::<u32, usize>::new();
        let mut current = BTreeMap::<u32, usize>::new();
        let mut overlap = BTreeMap::<(u32, u32), usize>::new();
        for (&a, &b) in slices[z - 1].iter().zip(&slices[z]) {
            if a > 0 {
                *previous.entry(a).or_default() += 1;
            }
            if b > 0 {
                *current.entry(b).or_default() += 1;
            }
            if a > 0 && b > 0 {
                *overlap.entry((b, a)).or_default() += 1;
            }
        }
        let mut scores = BTreeMap::new();
        for (&(b, a), &n) in &overlap {
            let score = n as f64 / (previous[&a] + current[&b] - n) as f64;
            if score >= threshold {
                scores.insert((b, a), score);
            }
        }
        let mut column_maxima = BTreeMap::<u32, f64>::new();
        for (&(_, previous), &score) in &scores {
            let maximum = column_maxima.entry(previous).or_default();
            *maximum = maximum.max(score);
        }
        // Apply per-column filtering BEFORE selecting row maxima. Sorted
        // previous labels also preserve NumPy argmax's first-index tie rule.
        let mut matched = BTreeMap::<u32, (u32, f64)>::new();
        for (&(current, previous), &score) in &scores {
            if score < column_maxima[&previous] {
                continue;
            }
            let best = matched.entry(current).or_insert((previous, score));
            if score > best.1 {
                *best = (previous, score);
            }
        }
        // The original dense IoU matrix has a row for EVERY label through
        // current_max. Absent labels consume fresh IDs as well.
        let matched_count = u32::try_from(matched.len())
            .map_err(|_| invalid("stitch label overflow"))?;
        let next_max = mmax
            .checked_add(current_max - matched_count)
            .ok_or_else(|| invalid("stitch label overflow"))?;
        let mut mapping = BTreeMap::new();
        for &label in current.keys() {
            let target = if let Some(&(previous, _)) = matched.get(&label) {
                previous
            } else {
                let preceding_matches = u32::try_from(matched.range(..=label).count())
                    .map_err(|_| invalid("stitch label overflow"))?;
                mmax + (label - preceding_matches)
            };
            mapping.insert(label, target);
        }
        for label in &mut slices[z] {
            if *label > 0 {
                *label = mapping[label];
            }
        }
        mmax = next_max;
        had_iou = true;
    }
    Ok(())
}

/// Predict a slice stack and optionally stitch labels into a volume.
/// With stitching, normalization uses the configured norm3D policy and small
/// instances are filtered after stitching, rather than being erased per slice.
/// # Errors
/// Rejects inconsistent stack geometry or propagates predictor/processing failures.
pub fn predict_stack<E: From<CellposeError>>(
    images: &[CellposeImage],
    tile_size: usize,
    batch_size: usize,
    options: &CellposeOptions,
    stitch_threshold: f64,
    mut evaluate: impl FnMut(&[f32], usize, usize) -> std::result::Result<Vec<f32>, E>,
) -> std::result::Result<Vec<CellposePrediction>, E> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if stitch_threshold <= 0.0 || images.len() == 1 {
        return images
            .iter()
            .map(|image| predict(image, tile_size, batch_size, options, &mut evaluate))
            .collect();
    }
    let (height, width) = (images[0].height, images[0].width);
    let area = elements(&[height, width])?;
    if images.iter().any(|image| {
        image.height != height || image.width != width || image.pixels.len() != 3 * area
    }) {
        return Err(invalid("stitching requires equally sized three-channel images").into());
    }
    let mut pixels = Vec::with_capacity(elements(&[3, images.len(), height, width])?);
    for c in 0..3 {
        for image in images {
            pixels.extend_from_slice(&image.pixels[c * area..(c + 1) * area]);
        }
    }
    normalize_planes(
        &mut pixels,
        &[images.len(), height, width],
        &options.normalization,
    )?;
    let mut plane_options = options.clone();
    plane_options.normalization = super::CellposeNormalization {
        normalize: false,
        ..Default::default()
    };
    plane_options.masks.min_size = 0;
    let mut predictions = Vec::with_capacity(images.len());
    for z in 0..images.len() {
        let mut plane = Vec::with_capacity(3 * area);
        for c in 0..3 {
            plane.extend_from_slice(
                &pixels[(c * images.len() + z) * area..(c * images.len() + z + 1) * area],
            );
        }
        predictions.push(predict(
            &CellposeImage {
                height,
                width,
                pixels: plane,
            },
            tile_size,
            batch_size,
            &plane_options,
            &mut evaluate,
        )?);
    }
    if options.compute_masks {
        let mut masks = predictions
            .iter_mut()
            .map(|p| std::mem::take(&mut p.masks))
            .collect::<Vec<_>>();
        stitch_masks(&mut masks, stitch_threshold)?;
        let mut volume = masks.into_iter().flatten().collect::<Vec<_>>();
        dynamics::fill_and_filter(
            &mut volume,
            &[images.len(), height, width],
            options.masks.min_size,
        )?;
        for (prediction, mask) in predictions.iter_mut().zip(volume.chunks_exact(area)) {
            prediction.masks = mask.to_vec();
        }
    }
    Ok(predictions)
}
