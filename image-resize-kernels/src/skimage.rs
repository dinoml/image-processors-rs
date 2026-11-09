//! FP64 order-one scikit-image resize with zero grid-constant boundaries.

use super::{ImageSize, ResizeError, image_len, usize_to_u32, validate_image_data};

/// Resizes interleaved FP64 pixels using scikit-image's default order-one constant mode.
///
/// Matches `skimage.transform.resize(..., order=1, mode="constant", cval=0)`:
/// downsampled axes receive a Gaussian with sigma `(scale - 1) / 2`, truncated
/// at four sigma, followed by half-pixel grid-constant interpolation. Clipping
/// follows scikit-image's input-range policy, including its zero-fill exception.
/// Values are not rescaled or quantized by this function.
///
/// # Errors
/// Returns an error for invalid geometry, channels, lengths, or nonfinite values.
pub fn resize_f64_skimage_bilinear_constant(
    values: &[f64],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<f64>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    usize_to_u32(source.height, "source height")?;
    usize_to_u32(source.width, "source width")?;
    usize_to_u32(target.height, "target height")?;
    usize_to_u32(target.width, "target width")?;
    validate_image_data(values.len(), source, channels)?;
    let output_len = image_len(target, channels)?;
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ResizeError::NonFiniteValue { index });
    }
    let scale_y = source.height as f64 / target.height as f64;
    let scale_x = source.width as f64 / target.width as f64;
    // scipy.ndimage.gaussian_filter processes axes in order and stores each
    // intermediate in the input dtype, which is FP64 for uint8 skimage inputs.
    let filtered_y = gaussian_axis(
        values,
        source,
        channels,
        (scale_y - 1.0).max(0.0) / 2.0,
        true,
    );
    let filtered = gaussian_axis(
        &filtered_y,
        source,
        channels,
        (scale_x - 1.0).max(0.0) / 2.0,
        false,
    );
    let mut output = vec![0.0; output_len];
    for y in 0..target.height {
        let position_y = (y as f64 + 0.5) * scale_y - 0.5;
        let y0 = position_y.floor() as isize;
        let fraction_y = position_y - y0 as f64;
        for x in 0..target.width {
            let position_x = (x as f64 + 0.5) * scale_x - 0.5;
            let x0 = position_x.floor() as isize;
            let fraction_x = position_x - x0 as f64;
            for channel in 0..channels {
                let mut value = 0.0;
                for (dy, weight_y) in [(0, 1.0 - fraction_y), (1, fraction_y)] {
                    for (dx, weight_x) in [(0, 1.0 - fraction_x), (1, fraction_x)] {
                        value += sample(&filtered, source, channels, y0 + dy, x0 + dx, channel)
                            * weight_y
                            * weight_x;
                    }
                }
                output[(y * target.width + x) * channels + channel] = value;
            }
        }
    }
    let mut minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let output_minimum = output.iter().copied().fold(f64::INFINITY, f64::min);
    let output_maximum = output.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(minimum..=maximum).contains(&0.0) && output_minimum <= 0.0 && output_maximum >= 0.0 {
        minimum = minimum.min(0.0);
        maximum = maximum.max(0.0);
    }
    for value in &mut output {
        *value = value.clamp(minimum, maximum);
    }
    Ok(output)
}

fn sample(
    values: &[f64],
    size: ImageSize,
    channels: usize,
    y: isize,
    x: isize,
    channel: usize,
) -> f64 {
    if y < 0 || x < 0 || y as usize >= size.height || x as usize >= size.width {
        0.0
    } else {
        values[((y as usize * size.width + x as usize) * channels) + channel]
    }
}

fn gaussian_axis(
    values: &[f64],
    size: ImageSize,
    channels: usize,
    sigma: f64,
    vertical: bool,
) -> Vec<f64> {
    // scipy.ndimage.gaussian_filter skips axes at this threshold.
    if sigma <= 1e-15 {
        return values.to_vec();
    }
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut weights = (-radius..=radius)
        .map(|offset| (-0.5 / (sigma * sigma) * offset as f64 * offset as f64).exp())
        .collect::<Vec<_>>();
    let sum: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    let mut output = vec![0.0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            for channel in 0..channels {
                let index = (y * size.width + x) * channels + channel;
                let mut value = values[index] * weights[radius as usize];
                // ndimage's symmetric correlation accumulates paired taps.
                for offset in (1..=radius).rev() {
                    let (dy, dx) = if vertical { (offset, 0) } else { (0, offset) };
                    let before = sample(
                        values,
                        size,
                        channels,
                        y as isize - dy,
                        x as isize - dx,
                        channel,
                    );
                    let after = sample(
                        values,
                        size,
                        channels,
                        y as isize + dy,
                        x as isize + dx,
                        channel,
                    );
                    value += (before + after) * weights[(radius - offset) as usize];
                }
                output[index] = value;
            }
        }
    }
    output
}
