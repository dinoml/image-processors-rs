//! OpenCV byte-image interpolation and scale-dependent resize policy.
//!
//! Source: OpenCV 4.10.0, commit 71d3237a093b60a27601c20e9ee6c3e52154e8b1,
//! modules/imgproc/src/resize.cpp: interpolateCubic, interpolateLanczos4,
//! computeResizeAreaTab, ResizeAreaFastVec and the scalar separable byte paths.

use rayon::prelude::*;

trait FloatSample: Copy + Send + Sync {
    const ZERO: Self;
    fn finite(self) -> bool;
    fn add(self, other: Self) -> Self;
    fn weighted(self, weight: f32) -> Self;
}
impl FloatSample for f32 {
    const ZERO: Self = 0.0;
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn weighted(self, weight: f32) -> Self {
        self * weight
    }
}
impl FloatSample for f64 {
    const ZERO: Self = 0.0;
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn weighted(self, weight: f32) -> Self {
        self * f64::from(weight)
    }
}

/// Resize finite interleaved FP32 values without byte quantization.
///
/// # Errors
/// Reports invalid dimensions, storage, channels, overflow or nonfinite values.
pub fn resize_f32_opencv(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    interpolation: OpenCvInterpolation,
) -> Result<Vec<f32>, ResizeError> {
    resize_float(values, source, channels, target, interpolation)
}

/// Resize finite interleaved FP64 values with floating interpolation coefficients.
///
/// # Errors
/// Reports invalid dimensions, storage, channels, overflow or nonfinite values.
pub fn resize_f64_opencv(
    values: &[f64],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    interpolation: OpenCvInterpolation,
) -> Result<Vec<f64>, ResizeError> {
    resize_float(values, source, channels, target, interpolation)
}

fn float_axis(
    input: usize,
    output: usize,
    mode: OpenCvInterpolation,
    horizontal: bool,
    area_reduction: bool,
) -> Vec<Vec<(usize, f32)>> {
    if mode == OpenCvInterpolation::Area && area_reduction {
        return area_axis(input, output);
    }
    let scale = axis_scale(input, output);
    (0..output)
        .map(|index| {
            if mode == OpenCvInterpolation::Nearest {
                return vec![(
                    ((index as f64 * scale).floor() as usize).min(input - 1),
                    1.0,
                )];
            }
            let (mut origin, mut fraction) = if mode == OpenCvInterpolation::Area {
                let origin = (index as f64 * scale).floor() as i64;
                let fraction = ((index + 1) as f64
                    - (origin + 1) as f64 * output as f64 / input as f64)
                    as f32;
                (
                    origin,
                    if fraction <= 0.0 {
                        0.0
                    } else {
                        fraction - fraction.floor()
                    },
                )
            } else {
                let coordinate = ((index as f64 + 0.5) * scale - 0.5) as f32;
                let origin = coordinate.floor() as i64;
                (origin, coordinate - origin as f32)
            };
            let (offset, count) = match mode {
                OpenCvInterpolation::Cubic => (-1, 4),
                OpenCvInterpolation::Lanczos4 => (-3, 8),
                _ => (0, 2),
            };
            if horizontal && count == 2 && (origin < 0 || origin >= input as i64 - 1) {
                origin = origin.clamp(0, input as i64 - 1);
                fraction = 0.0;
            }
            let weights = match mode {
                OpenCvInterpolation::Cubic => cubic_coefficients(fraction),
                OpenCvInterpolation::Lanczos4 => lanczos4_coefficients(fraction),
                _ => [1.0 - fraction, fraction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            };
            (0..count)
                .map(|tap| {
                    (
                        (origin + offset + tap as i64).clamp(0, input as i64 - 1) as usize,
                        weights[tap],
                    )
                })
                .collect()
        })
        .collect()
}

fn resize_float<T: FloatSample>(
    values: &[T],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    interpolation: OpenCvInterpolation,
) -> Result<Vec<T>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    super::validate_image_data(values.len(), source, channels)?;
    let count = super::image_len(target, channels)?;
    if let Some(index) = values.iter().position(|value| !value.finite()) {
        return Err(ResizeError::NonFiniteValue { index });
    }
    if source == target {
        return Ok(values.to_vec());
    }
    let mode = if interpolation == OpenCvInterpolation::Linear
        && source.width.is_multiple_of(target.width)
        && source.height.is_multiple_of(target.height)
        && source.width / target.width == 2
        && source.height / target.height == 2
    {
        OpenCvInterpolation::Area
    } else {
        interpolation
    };
    if mode == OpenCvInterpolation::Area
        && source.width.is_multiple_of(target.width)
        && source.height.is_multiple_of(target.height)
    {
        let sx = source.width / target.width;
        let sy = source.height / target.height;
        let mut result = vec![T::ZERO; count];
        for y in 0..target.height {
            for x in 0..target.width {
                for c in 0..channels {
                    let mut value = T::ZERO;
                    for dy in 0..sy {
                        for dx in 0..sx {
                            value = value.add(
                                values[((y * sy + dy) * source.width + x * sx + dx) * channels + c],
                            );
                        }
                    }
                    result[(y * target.width + x) * channels + c] =
                        value.weighted(1.0 / (sx * sy) as f32);
                }
            }
        }
        return Ok(result);
    }
    let area_reduction = source.width >= target.width && source.height >= target.height;
    let xs = float_axis(source.width, target.width, mode, true, area_reduction);
    let ys = float_axis(source.height, target.height, mode, false, area_reduction);
    let intermediate_len = source
        .height
        .checked_mul(target.width)
        .and_then(|v| v.checked_mul(channels))
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let mut intermediate = vec![T::ZERO; intermediate_len];
    for y in 0..source.height {
        for (x, weights) in xs.iter().enumerate() {
            for c in 0..channels {
                let mut sum = T::ZERO;
                for &(sx, weight) in weights {
                    sum = sum.add(values[(y * source.width + sx) * channels + c].weighted(weight));
                }
                intermediate[(y * target.width + x) * channels + c] = sum;
            }
        }
    }
    let mut result = vec![T::ZERO; count];
    for (y, weights) in ys.iter().enumerate() {
        for x in 0..target.width {
            for c in 0..channels {
                let mut sum = T::ZERO;
                for &(sy, weight) in weights {
                    sum = sum
                        .add(intermediate[(sy * target.width + x) * channels + c].weighted(weight));
                }
                result[(y * target.width + x) * channels + c] = sum;
            }
        }
    }
    Ok(result)
}

use super::{ImageSize, ResizeCrop, ResizeError, ResizeFilter};

/// OpenCV interpolation choices for interleaved byte images.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenCvInterpolation {
    /// INTER_NEAREST, using floor-scaled source coordinates.
    Nearest,
    /// INTER_LINEAR, preserving the existing OpenCvLinear byte contract.
    Linear,
    /// INTER_CUBIC, with A=-0.75 and quantized separable coefficients.
    Cubic,
    /// INTER_AREA, including OpenCV's enlargement fallback.
    Area,
    /// INTER_LANCZOS4, with eight taps on each spatial axis.
    Lanczos4,
}

pub(super) fn enlargement_filter(filter: ResizeFilter) -> OpenCvInterpolation {
    match filter {
        ResizeFilter::Nearest => OpenCvInterpolation::Nearest,
        ResizeFilter::Bilinear => OpenCvInterpolation::Linear,
        ResizeFilter::Bicubic => OpenCvInterpolation::Cubic,
        ResizeFilter::Lanczos => OpenCvInterpolation::Lanczos4,
    }
}

/// Resize with INTER_AREA when neither axis grows; otherwise use `enlargement`.
/// Existing profiles keep their independent filter and rounding semantics.
///
/// # Errors
/// Reports invalid storage, channel counts, dimensions or size overflow.
pub fn resize_u8_opencv_scale_dependent(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    enlargement: OpenCvInterpolation,
) -> Result<Vec<u8>, ResizeError> {
    let interpolation = if target.height <= source.height && target.width <= source.width {
        OpenCvInterpolation::Area
    } else {
        enlargement
    };
    resize_u8_opencv(values, source, channels, target, interpolation)
}

/// Resize an interleaved byte image with an explicit OpenCV interpolation mode.
/// Channels 1, 3 and 4 are supported. Rescaling and normalization can follow
/// interpolation without changing the byte-rounding boundary.
///
/// # Errors
/// Reports invalid storage, channel counts, dimensions or size overflow.
pub fn resize_u8_opencv(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    interpolation: OpenCvInterpolation,
) -> Result<Vec<u8>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    super::validate_image_data(values.len(), source, channels)?;
    for (dimension, value) in [
        ("source height", source.height),
        ("source width", source.width),
        ("target height", target.height),
        ("target width", target.width),
    ] {
        if value > i32::MAX as usize {
            return Err(ResizeError::DimensionTooLarge {
                dimension,
                value,
                max: i32::MAX as usize,
            });
        }
    }
    let len = super::image_len(target, channels)?;
    if source == target {
        return Ok(values.to_vec());
    }
    match interpolation {
        OpenCvInterpolation::Linear => {
            super::resize_u8_opencv_linear(values, source, channels, target, ResizeFilter::Bilinear)
        }
        OpenCvInterpolation::Area
            if target.height <= source.height && target.width <= source.width =>
        {
            Ok(resize_area(values, source, channels, target, len))
        }
        OpenCvInterpolation::Nearest => {
            let sx = axis_scale(source.width, target.width);
            let sy = axis_scale(source.height, target.height);
            let mut output = vec![0; len];
            output
                .par_chunks_exact_mut(target.width * channels)
                .enumerate()
                .for_each(|(y, row)| {
                    let source_y = ((y as f64 * sy).floor() as usize).min(source.height - 1);
                    for x in 0..target.width {
                        let source_x = ((x as f64 * sx).floor() as usize).min(source.width - 1);
                        let start = (source_y * source.width + source_x) * channels;
                        row[x * channels..(x + 1) * channels]
                            .copy_from_slice(&values[start..start + channels]);
                    }
                });
            Ok(output)
        }
        OpenCvInterpolation::Area | OpenCvInterpolation::Cubic | OpenCvInterpolation::Lanczos4 => {
            Ok(resize_separable(
                values,
                source,
                channels,
                target,
                interpolation,
                len,
            ))
        }
    }
}

/// Resize with an explicit OpenCV mode, then crop the requested window.
///
/// # Errors
/// Reports invalid resize/crop geometry, storage, channel counts or overflow.
pub fn resize_u8_opencv_with_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    interpolation: OpenCvInterpolation,
) -> Result<Vec<u8>, ResizeError> {
    let resized = resize_u8_opencv(values, source, channels, crop.resized(), interpolation)?;
    super::crop_image_data(
        &resized,
        crop.resized(),
        channels,
        crop.origin_x(),
        crop.origin_y(),
        crop.target(),
    )
}

fn axis_scale(input: usize, output: usize) -> f64 {
    // cv::resize first computes target/source; hal::resize takes its reciprocal.
    1.0 / (output as f64 / input as f64)
}

struct FixedAxis {
    positions: [usize; 8],
    weights: [i32; 8],
    taps: usize,
}

fn fixed_axis(
    input: usize,
    output: usize,
    mode: OpenCvInterpolation,
    horizontal: bool,
) -> Vec<FixedAxis> {
    let scale = axis_scale(input, output);
    let inverse = output as f64 / input as f64;
    (0..output)
        .map(|index| {
            let (mut origin, mut fraction) = if mode == OpenCvInterpolation::Area {
                let origin = (index as f64 * scale).floor() as i64;
                let fraction = ((index + 1) as f64 - (origin + 1) as f64 * inverse) as f32;
                (
                    origin,
                    if fraction <= 0.0 {
                        0.0
                    } else {
                        fraction - fraction.floor()
                    },
                )
            } else {
                let coordinate = ((index as f64 + 0.5) * scale - 0.5) as f32;
                let origin = coordinate.floor() as i64;
                (origin, coordinate - origin as f32)
            };
            let (offset, taps) = match mode {
                OpenCvInterpolation::Cubic => (-1, 4),
                OpenCvInterpolation::Lanczos4 => (-3, 8),
                _ => (0, 2),
            };
            if horizontal && taps == 2 && (origin < 0 || origin >= input as i64 - 1) {
                origin = origin.clamp(0, input as i64 - 1);
                fraction = 0.0;
            }
            let coefficients = match mode {
                OpenCvInterpolation::Cubic => cubic_coefficients(fraction),
                OpenCvInterpolation::Lanczos4 => lanczos4_coefficients(fraction),
                _ => [1.0 - fraction, fraction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            };
            FixedAxis {
                positions: std::array::from_fn(|tap| {
                    (origin + offset + tap as i64).clamp(0, input as i64 - 1) as usize
                }),
                weights: coefficients.map(|value| {
                    (value * 2048.0)
                        .round_ties_even()
                        .clamp(i16::MIN as f32, i16::MAX as f32) as i32
                }),
                taps,
            }
        })
        .collect()
}

fn cubic_coefficients(x: f32) -> [f32; 8] {
    let a = -0.75_f32;
    let first = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
    let second = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let third = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [
        first,
        second,
        third,
        1.0 - first - second - third,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn lanczos4_coefficients(x: f32) -> [f32; 8] {
    let s45 = std::f64::consts::FRAC_1_SQRT_2;
    let rotations = [
        (1.0, 0.0),
        (-s45, -s45),
        (0.0, 1.0),
        (s45, -s45),
        (-1.0, 0.0),
        (s45, s45),
        (0.0, -1.0),
        (-s45, s45),
    ];
    let angle = -f64::from(x + 3.0) * std::f64::consts::PI * 0.25;
    let sine = angle.sin();
    let cosine = angle.cos();
    let mut coefficients = [0.0_f32; 8];
    let mut sum = 0.0_f32;
    for (tap, (sin_rotation, cos_rotation)) in rotations.into_iter().enumerate() {
        let distance = x + 3.0 - tap as f32;
        coefficients[tap] = if distance.abs() >= 1e-6 {
            let y = -f64::from(distance) * std::f64::consts::PI * 0.25;
            ((sin_rotation * sine + cos_rotation * cosine) / (y * y)) as f32
        } else {
            1e30
        };
        sum += coefficients[tap];
    }
    let inverse_sum = 1.0 / sum;
    coefficients.map(|value| value * inverse_sum)
}

fn resize_separable(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    mode: OpenCvInterpolation,
    len: usize,
) -> Vec<u8> {
    let columns = fixed_axis(source.width, target.width, mode, true);
    let rows = fixed_axis(source.height, target.height, mode, false);
    let mut output = vec![0; len];
    output
        .par_chunks_exact_mut(target.width * channels)
        .enumerate()
        .for_each(|(y, row)| {
            let vertical = &rows[y];
            for (x, horizontal) in columns.iter().enumerate() {
                for channel in 0..channels {
                    let mut intermediate = [0_i64; 8];
                    for (tap_y, value) in intermediate.iter_mut().enumerate().take(vertical.taps) {
                        for tap_x in 0..horizontal.taps {
                            let source_offset = (vertical.positions[tap_y] * source.width
                                + horizontal.positions[tap_x])
                                * channels
                                + channel;
                            *value += i64::from(values[source_offset])
                                * i64::from(horizontal.weights[tap_x]);
                        }
                    }
                    let value = if vertical.taps == 2 {
                        // Preserve byte-linear staged shifts in INTER_AREA enlargement.
                        let top = (i64::from(vertical.weights[0]) * (intermediate[0] >> 4)) >> 16;
                        let bottom =
                            (i64::from(vertical.weights[1]) * (intermediate[1] >> 4)) >> 16;
                        (top + bottom + 2) >> 2
                    } else {
                        let sum: i64 = intermediate
                            .iter()
                            .zip(vertical.weights)
                            .take(vertical.taps)
                            .map(|(value, weight)| *value * i64::from(weight))
                            .sum();
                        (sum + (1 << 21)) >> 22
                    };
                    row[x * channels + channel] = value.clamp(0, 255) as u8;
                }
            }
        });
    output
}

fn area_axis(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = axis_scale(input, output);
    (0..output)
        .map(|index| {
            let first = index as f64 * scale;
            let last = first + scale;
            let cell_width = scale.min(input as f64 - first);
            let end = (last.floor() as usize).min(input - 1);
            let start = (first.ceil() as usize).min(end);
            let mut weights = Vec::new();
            if start as f64 - first > 1e-3 {
                weights.push((start - 1, ((start as f64 - first) / cell_width) as f32));
            }
            for source in start..end {
                weights.push((source, (1.0 / cell_width) as f32));
            }
            if last - end as f64 > 1e-3 {
                weights.push((
                    end,
                    ((last - end as f64).min(1.0).min(cell_width) / cell_width) as f32,
                ));
            }
            weights
        })
        .collect()
}

fn resize_area(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    len: usize,
) -> Vec<u8> {
    let mut output = vec![0; len];
    let sx = axis_scale(source.width, target.width);
    let sy = axis_scale(source.height, target.height);
    let integer_x = sx.round_ties_even() as usize;
    let integer_y = sy.round_ties_even() as usize;
    if (sx - integer_x as f64).abs() < f64::EPSILON && (sy - integer_y as f64).abs() < f64::EPSILON
    {
        let area = integer_x * integer_y;
        let scale = 1.0_f32 / area as f32;
        output
            .par_chunks_exact_mut(target.width * channels)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..target.width {
                    for channel in 0..channels {
                        let mut sum = 0_u64;
                        for yy in 0..integer_y {
                            for xx in 0..integer_x {
                                sum += u64::from(
                                    values[((y * integer_y + yy) * source.width
                                        + x * integer_x
                                        + xx)
                                        * channels
                                        + channel],
                                );
                            }
                        }
                        row[x * channels + channel] = if integer_x == 2 && integer_y == 2 {
                            ((sum + 2) >> 2) as u8
                        } else {
                            (sum as f32 * scale).round_ties_even().clamp(0.0, 255.0) as u8
                        };
                    }
                }
            });
    } else {
        let columns = area_axis(source.width, target.width);
        let rows = area_axis(source.height, target.height);
        output
            .par_chunks_exact_mut(target.width * channels)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, column) in columns.iter().enumerate() {
                    for channel in 0..channels {
                        let mut sum = 0.0_f32;
                        for &(source_y, beta) in &rows[y] {
                            let mut horizontal = 0.0_f32;
                            for &(source_x, alpha) in column {
                                horizontal += f32::from(
                                    values
                                        [(source_y * source.width + source_x) * channels + channel],
                                ) * alpha;
                            }
                            sum += horizontal * beta;
                        }
                        row[x * channels + channel] = sum.round_ties_even().clamp(0.0, 255.0) as u8;
                    }
                }
            });
    }
    output
}
