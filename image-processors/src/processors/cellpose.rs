//! Cellpose 4 inference geometry and floating microscopy image processing.
//!
//! Mathematics follows MouseLand/cellpose a54cb48849b7e225a81e8e43dcb042d42427f543,
//! transforms.py, core.py and dynamics.py. All image operations are native Rust.

mod dynamics;
mod normalization;
mod volume;

pub use dynamics::{reconstruct_masks, CellposeField, CellposeMaskOptions};
pub use volume::{
    predict_stack, predict_volume, stitch_masks, CellposeVolume, CellposeVolumeOptions,
};

use image_resize_kernels::{resize_f32_opencv, ImageSize, OpenCvInterpolation};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Invalid Cellpose geometry, options or callback output.
#[derive(Debug, Error)]
#[error("Cellpose processing: {0}")]
pub struct CellposeError(pub String);

/// Shared Cellpose processing result.
pub type Result<T> = std::result::Result<T, CellposeError>;

fn invalid(message: impl Into<String>) -> CellposeError {
    CellposeError(message.into())
}

/// Finite microscopy pixels in planar CHW order; intensity units are retained.
#[derive(Clone, Debug)]
pub struct CellposeImage {
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
    /// Exactly three planes after dropping extra channels and zero-padding missing channels.
    pub pixels: Vec<f32>,
}

impl CellposeImage {
    /// Convert grayscale, HWC or CHW arrays without byte conversion or RGB replication.
    /// With no channel axis, the upstream first/last-axis size heuristic is used.
    /// # Errors
    /// Rejects nonfinite values, invalid shapes, axes and overflowing storage.
    pub fn from_array(
        values: &[f32],
        shape: &[usize],
        channel_axis: Option<usize>,
    ) -> Result<Self> {
        let count = elements(shape)?;
        if count != values.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "invalid input storage or nonfinite microscopy pixel",
            ));
        }
        let (height, width, channels, axis) = match shape {
            [h, w] => (*h, *w, 1, 2),
            [a, b, c] => match channel_axis.unwrap_or(if a >= c { 2 } else { 0 }) {
                0 => (*b, *c, *a, 0),
                1 => (*a, *c, *b, 1),
                2 => (*a, *b, *c, 2),
                _ => return Err(invalid("channel axis exceeds image rank")),
            },
            _ => return Err(invalid("use a 2D grayscale or 3D multichannel image")),
        };
        let mut pixels = vec![0.0; elements(&[3, height, width])?];
        for c in 0..channels.min(3) {
            for y in 0..height {
                for x in 0..width {
                    let source = if shape.len() == 2 {
                        y * width + x
                    } else {
                        match axis {
                            0 => (c * height + y) * width + x,
                            1 => (y * channels + c) * width + x,
                            _ => (y * width + x) * channels + c,
                        }
                    };
                    pixels[(c * height + y) * width + x] = values[source];
                }
            }
        }
        Ok(Self {
            height,
            width,
            pixels,
        })
    }
}

/// Cellpose percentile normalization options, applied in original intensity units.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CellposeNormalization {
    /// Enable percentile normalization.
    pub normalize: bool,
    /// Lower and upper percentile; defaults to 1 and 99.
    pub percentile: [f64; 2],
    /// Optional global or per-channel explicit low/high bounds.
    pub lowhigh: Option<Vec<[f64; 2]>>,
    /// Invert normalized nonconstant channels.
    pub invert: bool,
    /// Normalize an entire stack together, rather than each Z plane separately.
    #[serde(rename = "norm3D")]
    pub norm_3d: bool,
    /// Local percentile block size; zero selects global percentile normalization.
    pub tile_norm_blocksize: usize,
    /// Extra smoothing of local normalization grids for volume normalization.
    #[serde(rename = "tile_norm_smooth3D")]
    pub tile_norm_smooth_3d: f64,
    /// Circular Gaussian smoothing radius; zero disables smoothing.
    pub smooth_radius: f32,
    /// Circular Gaussian surround subtraction radius; zero disables sharpening.
    pub sharpen_radius: f32,
}

impl Default for CellposeNormalization {
    fn default() -> Self {
        Self {
            normalize: true,
            percentile: [1.0, 99.0],
            lowhigh: None,
            invert: false,
            norm_3d: true,
            tile_norm_blocksize: 0,
            tile_norm_smooth_3d: 1.0,
            smooth_radius: 0.0,
            sharpen_radius: 0.0,
        }
    }
}

/// Native tiling and reconstruction request options.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CellposeOptions {
    /// Optional inference tile side, within the selected predictor's compiled capacity.
    #[serde(alias = "bsize")]
    pub tile_size: Option<usize>,
    /// Normalization in original image space.
    pub normalization: CellposeNormalization,
    /// Desired object diameter; scales input by 30 / diameter.
    pub diameter: Option<f64>,
    /// Restore floating flow maps before reconstructing masks.
    pub resample: bool,
    /// Use overlapping flipped tile augmentation with flow sign correction.
    pub augment: bool,
    /// Fractional tile overlap, clamped to [0.05, 0.5].
    pub tile_overlap: f64,
    /// Run instance reconstruction; false returns flow maps alone.
    pub compute_masks: bool,
    /// Flow dynamics and filtering options.
    pub masks: CellposeMaskOptions,
}

impl Default for CellposeOptions {
    fn default() -> Self {
        Self {
            tile_size: None,
            normalization: CellposeNormalization::default(),
            diameter: None,
            resample: true,
            augment: false,
            tile_overlap: 0.1,
            compute_masks: true,
            masks: CellposeMaskOptions::default(),
        }
    }
}

/// Complete planar flow prediction and optionally reconstructed integer labels.
#[derive(Clone, Debug)]
pub struct CellposePrediction {
    /// Flow/cell-probability field, possibly at scaled resolution when resample=false.
    pub field: CellposeField,
    /// Integer labels at original resolution, empty when compute_masks=false.
    pub masks: Vec<u32>,
    /// Original dimensions corresponding to masks.
    pub mask_shape: Vec<usize>,
    /// Compatibility output, unused by Cellpose 4 inference.
    pub style: [f32; 256],
}

/// Normalize each planar channel in place, retaining FP32 storage and FP64 percentile ranks.
/// # Errors
/// Rejects invalid percentile/bounds or nonfinite values.
pub fn normalize_planes(
    pixels: &mut [f32],
    shape: &[usize],
    options: &CellposeNormalization,
) -> Result<()> {
    if !(2..=3).contains(&shape.len()) {
        return Err(invalid("normalization requires YX or ZYX geometry"));
    }
    let count = elements(shape)?;
    if pixels.len() != elements(&[3, count])? || pixels.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "normalization requires finite three-channel planes",
        ));
    }
    let [lower, upper] = options.percentile;
    if !(0.0 <= lower && lower < upper && upper <= 100.0) {
        return Err(invalid("invalid normalization percentile range"));
    }
    if let Some(bounds) = &options.lowhigh {
        if bounds.len() != 1 && bounds.len() != 3 {
            return Err(invalid("lowhigh needs one or three pairs"));
        }
    }
    if options.lowhigh.is_none() {
        normalization::smooth_sharpen(
            pixels,
            shape,
            options.smooth_radius,
            options.sharpen_radius,
        )?;
        if options.tile_norm_blocksize > 0 {
            return normalization::local_percentiles(pixels, shape, options);
        }
    }
    for (c, plane) in pixels.chunks_exact_mut(count).enumerate() {
        let explicit = options
            .lowhigh
            .as_ref()
            .map(|bounds| bounds[if bounds.len() == 1 { 0 } else { c }]);
        let normalization_shape = if explicit.is_none() && shape.len() == 3 && !options.norm_3d {
            &shape[1..]
        } else {
            shape
        };
        let chunk_count = elements(normalization_shape)?;
        // normalize_img tracks cgood across the entire channel, not per slice.
        // Constant slices retain their input until this channel-wide decision.
        let mut good = false;
        for chunk in plane.chunks_exact_mut(chunk_count) {
            let (low, high) = if let Some([low, high]) = explicit {
                if !low.is_finite() || !high.is_finite() || low == high {
                    return Err(invalid("invalid lowhigh bounds"));
                }
                good = true;
                (low, high)
            } else if options.normalize {
                if chunk.iter().all(|value| *value == chunk[0]) {
                    continue;
                }
                // Upstream marks nonconstant input good even when normalize99
                // zeros a sub-1e-3 percentile range. Inversion still applies.
                good = true;
                let strides = if chunk_count > 224usize.pow(3) {
                    normalization_shape
                        .iter()
                        .enumerate()
                        .map(|(axis, &extent)| {
                            (extent
                                / if normalization_shape.len() == 3 && axis == 0 {
                                    50
                                } else {
                                    224
                                })
                            .max(1)
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![1; normalization_shape.len()]
                };
                let mut sample = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &value)| {
                        let mut rest = i;
                        for axis in (0..normalization_shape.len()).rev() {
                            if !(rest % normalization_shape[axis]).is_multiple_of(strides[axis]) {
                                return None;
                            }
                            rest /= normalization_shape[axis];
                        }
                        Some(value)
                    })
                    .collect::<Vec<_>>();
                sample.sort_by(f32::total_cmp);
                (percentile(&sample, lower), percentile(&sample, upper))
            } else {
                continue;
            };
            if explicit.is_none() && high - low <= 1e-3 {
                chunk.fill(0.0);
                continue;
            }
            for value in chunk {
                *value = (f64::from(*value) - low) as f32;
                *value = (f64::from(*value) / (high - low)) as f32;
            }
        }
        if !good && options.normalize {
            plane.fill(0.0);
        } else if good && options.invert {
            for value in plane {
                *value = 1.0 - *value;
            }
        }
    }
    Ok(())
}

fn percentile(sorted: &[f32], percent: f64) -> f64 {
    let rank = percent / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    f64::from(sorted[lower])
        + (f64::from(sorted[upper]) - f64::from(sorted[lower])) * (rank - lower as f64)
}

fn elements(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| {
            if extent == 0 {
                None
            } else {
                count.checked_mul(extent)
            }
        })
        .ok_or_else(|| invalid("empty or overflowing image geometry"))
}

pub(crate) fn resize_planes(
    values: &[f32],
    channels: usize,
    source: [usize; 2],
    target: [usize; 2],
) -> Result<Vec<f32>> {
    let count = elements(&source)?;
    if values.len() != elements(&[channels, source[0], source[1]])? {
        return Err(invalid("resize geometry differs"));
    }
    let mut output = Vec::with_capacity(elements(&[channels, target[0], target[1]])?);
    for plane in values.chunks_exact(count) {
        output.extend(
            resize_f32_opencv(
                plane,
                ImageSize {
                    height: source[0],
                    width: source[1],
                },
                1,
                ImageSize {
                    height: target[0],
                    width: target[1],
                },
                OpenCvInterpolation::Linear,
            )
            .map_err(|e| invalid(e.to_string()))?,
        );
    }
    Ok(output)
}

fn starts(length: usize, tile: usize, overlap: f64, augment: bool) -> Vec<usize> {
    let count = if augment {
        ((2.0 * length as f64 / tile as f64).ceil() as usize).max(2)
    } else if length <= tile {
        1
    } else {
        ((1.0 + 2.0 * overlap) * length as f64 / tile as f64).ceil() as usize
    };
    let step = if count == 1 {
        0.0
    } else {
        (length - tile) as f64 / (count - 1) as f64
    };
    (0..count)
        .map(|i| {
            if i + 1 == count {
                length - tile
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect()
}

/// Run shared tiling around a batch callback, then reconstruct at original resolution.
/// The callback receives NCHW FP32 pixels and returns three planar FP32 maps per tile.
/// Model precision conversion, execution device and readback belong to the callback.
/// # Errors
/// Returns geometry/processing failures and propagates the callback's typed errors.
pub fn predict<E: From<CellposeError>>(
    image: &CellposeImage,
    tile_size: usize,
    batch_size: usize,
    options: &CellposeOptions,
    mut evaluate: impl FnMut(&[f32], usize, usize) -> std::result::Result<Vec<f32>, E>,
) -> std::result::Result<CellposePrediction, E> {
    let tile_size = options.tile_size.unwrap_or(tile_size);
    if tile_size == 0 || batch_size == 0 || !options.tile_overlap.is_finite() {
        return Err(invalid("invalid tile/batch size or overlap").into());
    }
    let original = [image.height, image.width];
    let mut pixels = image.pixels.clone();
    normalize_planes(&mut pixels, &original, &options.normalization)?;
    let scale = match options.diameter {
        Some(d) if d.is_finite() && d > 0.0 => 30.0 / d,
        None => 1.0,
        _ => return Err(invalid("diameter must be finite and positive").into()),
    };
    let resized = [
        (image.height as f64 * scale) as usize,
        (image.width as f64 * scale) as usize,
    ];
    elements(&resized)?;
    pixels = resize_planes(&pixels, 3, original, resized)?;
    let pad = |length: usize| {
        let extra = if length >= tile_size {
            length.div_ceil(16) * 16 - length
        } else {
            tile_size.saturating_sub(length + 16)
        };
        (8 + extra / 2, 8 + extra - extra / 2)
    };
    let (top, bottom) = pad(resized[0]);
    let (left, right) = pad(resized[1]);
    let padded = [resized[0] + top + bottom, resized[1] + left + right];
    let area = elements(&padded)?;
    let ys = starts(
        padded[0],
        tile_size,
        options.tile_overlap.clamp(0.05, 0.5),
        options.augment,
    );
    let xs = starts(
        padded[1],
        tile_size,
        options.tile_overlap.clamp(0.05, 0.5),
        options.augment,
    );
    let tile_area = elements(&[tile_size, tile_size])?;
    let mut sum = vec![0.0f32; elements(&[3, padded[0], padded[1]])?];
    let mut counts = vec![0.0f64; area];
    let mask_side = tile_size.max(224);
    let start = mask_side / 2 - tile_size / 2;
    let taper = (0..tile_size)
        .map(|i| {
            let distance = ((i + start) as f64 - (mask_side - 1) as f64 / 2.0).abs();
            1.0 / (1.0 + ((distance - (mask_side as f64 / 2.0 - 20.0)) / 7.5).exp())
        })
        .collect::<Vec<_>>();
    let tiles = ys
        .len()
        .checked_mul(xs.len())
        .ok_or_else(|| invalid("tile count overflow"))?;
    for offset in (0..tiles).step_by(batch_size) {
        let batch = batch_size.min(tiles - offset);
        let mut input = vec![0.0f32; elements(&[batch, 3, tile_size, tile_size])?];
        for n in 0..batch {
            let (j, i) = ((offset + n) / xs.len(), (offset + n) % xs.len());
            let fy = options.augment && !i.is_multiple_of(2);
            let fx = options.augment && !j.is_multiple_of(2);
            for c in 0..3 {
                for y in 0..tile_size {
                    for x in 0..tile_size {
                        let py = ys[j] + if fy { tile_size - 1 - y } else { y };
                        let px = xs[i] + if fx { tile_size - 1 - x } else { x };
                        if py >= top
                            && py < top + resized[0]
                            && px >= left
                            && px < left + resized[1]
                        {
                            input[(n * 3 + c) * tile_area + y * tile_size + x] =
                                pixels[(c * resized[0] + py - top) * resized[1] + px - left];
                        }
                    }
                }
            }
        }
        let output = evaluate(&input, batch, tile_size)?;
        if output.len() != input.len() {
            return Err(invalid("tile callback must return N x 3 x H x W").into());
        }
        for n in 0..batch {
            let (j, i) = ((offset + n) / xs.len(), (offset + n) % xs.len());
            let fy = options.augment && !i.is_multiple_of(2);
            let fx = options.augment && !j.is_multiple_of(2);
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let destination = (ys[j] + y) * padded[1] + xs[i] + x;
                    let weight = taper[y] * taper[x];
                    counts[destination] += weight;
                    for c in 0..3 {
                        let sy = if fy { tile_size - 1 - y } else { y };
                        let sx = if fx { tile_size - 1 - x } else { x };
                        let sign = if (c == 0 && fy) || (c == 1 && fx) {
                            -1.0
                        } else {
                            1.0
                        };
                        sum[c * area + destination] = (f64::from(sum[c * area + destination])
                            + f64::from(output[(n * 3 + c) * tile_area + sy * tile_size + sx])
                                * sign
                                * weight)
                            as f32;
                    }
                }
            }
        }
    }
    let mut maps = vec![0.0; elements(&[3, resized[0], resized[1]])?];
    for c in 0..3 {
        for y in 0..resized[0] {
            for x in 0..resized[1] {
                let source = (y + top) * padded[1] + x + left;
                maps[(c * resized[0] + y) * resized[1] + x] =
                    (f64::from(sum[c * area + source]) / counts[source]) as f32;
            }
        }
    }
    let shape = if options.resample {
        maps = resize_planes(&maps, 3, resized, original)?;
        original
    } else {
        resized
    };
    let count = elements(&shape)?;
    let field = CellposeField {
        shape: shape.to_vec(),
        flows: maps[..2 * count].to_vec(),
        cell_probability: maps[2 * count..].to_vec(),
    };
    let mut mask_options = options.masks.clone();
    // resize_and_compute_masks filters/fills after integer-label restoration.
    mask_options.min_size = 0;
    if mask_options.iterations == 0 {
        mask_options.iterations = (200.0 / if options.resample { scale } else { 1.0 }) as usize;
    }
    let mut masks = if options.compute_masks {
        reconstruct_masks(&field, &mask_options)?
    } else {
        Vec::new()
    };
    if options.compute_masks && shape != original {
        let mut restored = vec![0; elements(&original)?];
        for y in 0..original[0] {
            for x in 0..original[1] {
                restored[y * original[1] + x] =
                    masks[(y * shape[0] / original[0]) * shape[1] + x * shape[1] / original[1]];
            }
        }
        masks = restored;
    }
    if options.compute_masks {
        dynamics::fill_and_filter(&mut masks, &original, options.masks.min_size)?;
    }
    Ok(CellposePrediction {
        field,
        masks,
        mask_shape: original.to_vec(),
        style: [0.0; 256],
    })
}
