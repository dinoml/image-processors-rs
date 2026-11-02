//! Pillow whole-image integer reduction followed by fractional-box resampling.

use super::*;

/// Resize a byte image with Pillow's optional integer reduction before resampling.
/// The final source box retains the original extent divided by the reduction
/// factors, including partial edge blocks and intermediate byte rounding.
///
/// # Errors
/// Reports invalid storage, dimensions, reduction gaps or allocation-size overflow.
pub fn resize_u8_pillow_reducing_gap(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
    reducing_gap: f64,
) -> Result<Vec<u8>, ResizeError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    validate_image_data(values.len(), source, channels)?;
    image_len(target, channels)?;
    if !reducing_gap.is_finite() || reducing_gap < 1.0 {
        return Err(ResizeError::InvalidScaleFactor(reducing_gap as f32));
    }
    let xscale = ((source.width as f64 / target.width as f64 / reducing_gap) as usize).max(1);
    let yscale = ((source.height as f64 / target.height as f64 / reducing_gap) as usize).max(1);
    if filter == ResizeFilter::Nearest {
        return resize_u8_pillow(values, source, channels, target, filter);
    }
    if xscale == 1 && yscale == 1 {
        if source.height > source.width.saturating_mul(100) && target.height < source.height {
            let transposed_source = ImageSize {
                height: source.width,
                width: source.height,
            };
            let transposed_target = ImageSize {
                height: target.width,
                width: target.height,
            };
            let pixels = resize_u8_pillow(
                &transpose(values, source, channels),
                transposed_source,
                channels,
                transposed_target,
                filter,
            )?;
            return Ok(transpose(&pixels, transposed_target, channels));
        }
        return resize_u8_pillow(values, source, channels, target, filter);
    }
    let reduced = ImageSize::new(
        source.height.div_ceil(yscale),
        source.width.div_ceil(xscale),
    )?;
    let mut pixels = vec![0; image_len(reduced, channels)?];
    for y in 0..reduced.height {
        let height = yscale.min(source.height - y * yscale);
        for x in 0..reduced.width {
            let width = xscale.min(source.width - x * xscale);
            let count = width
                .checked_mul(height)
                .ok_or(ResizeError::ImageSizeOverflow)?;
            let count = u32::try_from(count).map_err(|_| ResizeError::ImageSizeOverflow)?;
            // ImagingReduce uses a float32 reciprocal and fixed-point byte output.
            let multiplier = (4294967296.0_f32 / (256.0 * count as f32)) as u32;
            for channel in 0..channels {
                let mut sum = count / 2;
                for dy in 0..height {
                    for dx in 0..width {
                        sum = sum.wrapping_add(u32::from(
                            values[((y * yscale + dy) * source.width + x * xscale + dx) * channels
                                + channel],
                        ));
                    }
                }
                pixels[(y * reduced.width + x) * channels + channel] =
                    (sum.wrapping_mul(multiplier) >> 24) as u8;
            }
        }
    }
    // Pillow's C resampler receives single-precision box coordinates.
    let width_extent = f64::from((source.width as f64 / xscale as f64) as f32);
    let height_extent = f64::from((source.height as f64 / yscale as f64) as f32);
    let mut source = reduced;
    let mut target_size = target;
    let vertical_first =
        source.height > source.width.saturating_mul(100) && target.height < source.height;
    if vertical_first {
        pixels = transpose(&pixels, source, channels);
        source = ImageSize {
            height: reduced.width,
            width: reduced.height,
        };
        target_size = ImageSize {
            height: target.width,
            width: target.height,
        };
    }
    let (horizontal_extent, vertical_extent) = if vertical_first {
        (height_extent, width_extent)
    } else {
        (width_extent, height_extent)
    };
    let horizontal = PillowFixedCoefficients::new_window_extent(
        source.width,
        target_size.width,
        0,
        target_size.width,
        filter,
        horizontal_extent,
    )?;
    let vertical = PillowFixedCoefficients::new_window_extent(
        source.height,
        target_size.height,
        0,
        target_size.height,
        filter,
        vertical_extent,
    )?;
    let mut intermediate = vec![
        0;
        image_len(
            ImageSize {
                height: source.height,
                width: target_size.width
            },
            channels
        )?
    ];
    let mut output = vec![0; image_len(target_size, channels)?];
    pillow_fixed_passes(
        &pixels,
        source,
        channels,
        target_size,
        &horizontal,
        &vertical,
        &mut intermediate,
        &mut output,
    );
    Ok(if vertical_first {
        transpose(&output, target_size, channels)
    } else {
        output
    })
}

fn transpose(values: &[u8], size: ImageSize, channels: usize) -> Vec<u8> {
    let mut output = vec![0; values.len()];
    for y in 0..size.height {
        for x in 0..size.width {
            output[(x * size.height + y) * channels..(x * size.height + y + 1) * channels]
                .copy_from_slice(
                    &values[(y * size.width + x) * channels..(y * size.width + x + 1) * channels],
                );
        }
    }
    output
}
