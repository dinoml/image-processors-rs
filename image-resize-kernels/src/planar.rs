use crate::{resize_f32_torchvision, ImageSize, ResizeError, ResizeFilter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorInterpolation {
    Nearest,
    NearestExact,
    Bilinear,
    Bicubic,
    Area,
}

pub fn resize_f32_planes(
    values: &[f32],
    planes: usize,
    source: (usize, usize),
    target: (usize, usize),
    filter: TensorInterpolation,
    antialias: bool,
) -> Result<Vec<f32>, ResizeError> {
    ImageSize::new(source.0, source.1)?;
    ImageSize::new(target.0, target.1)?;
    let output_len = planes
        .checked_mul(target.0)
        .and_then(|n| n.checked_mul(target.1))
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let (sh, sw) = source;
    let (th, tw) = target;
    if sh == 0
        || sw == 0
        || th == 0
        || tw == 0
        || planes.checked_mul(sh).and_then(|n| n.checked_mul(sw)) != Some(values.len())
    {
        return Err(ResizeError::InvalidBufferLength {
            expected: planes
                .checked_mul(sh)
                .and_then(|n| n.checked_mul(sw))
                .ok_or(ResizeError::ImageSizeOverflow)?,
            actual: values.len(),
        });
    }
    if source == target {
        return Ok(values.to_vec());
    }
    if antialias
        && matches!(
            filter,
            TensorInterpolation::Bilinear | TensorInterpolation::Bicubic
        )
    {
        let kernel = if filter == TensorInterpolation::Bilinear {
            ResizeFilter::Bilinear
        } else {
            ResizeFilter::Bicubic
        };
        let from = ImageSize::new(sh, sw)?;
        let to = ImageSize::new(th, tw)?;
        let mut output = Vec::with_capacity(planes * th * tw);
        for plane in values.chunks_exact(sh * sw) {
            output.extend(resize_f32_torchvision(plane, from, 1, to, kernel)?);
        }
        return Ok(output);
    }
    let mut output = vec![0.0; output_len];
    for plane in 0..planes {
        let input = &values[plane * sh * sw..(plane + 1) * sh * sw];
        let at = |y: isize, x: isize| {
            input[y.clamp(0, sh as isize - 1) as usize * sw + x.clamp(0, sw as isize - 1) as usize]
        };
        for y in 0..th {
            for x in 0..tw {
                let fy = (y as f64 + 0.5) * sh as f64 / th as f64 - 0.5;
                let fx = (x as f64 + 0.5) * sw as f64 / tw as f64 - 0.5;
                let value = match filter {
                    TensorInterpolation::Nearest => {
                        at((y * sh / th) as isize, (x * sw / tw) as isize)
                    }
                    TensorInterpolation::NearestExact => {
                        at((fy + 0.5).floor() as isize, (fx + 0.5).floor() as isize)
                    }
                    TensorInterpolation::Bilinear => {
                        let (fy, fx) = (fy.max(0.0), fx.max(0.0));
                        let (iy, ix) = (fy.floor() as isize, fx.floor() as isize);
                        let (dy, dx) = ((fy - iy as f64) as f32, (fx - ix as f64) as f32);
                        (at(iy, ix) * (1.0 - dx) + at(iy, ix + 1) * dx) * (1.0 - dy)
                            + (at(iy + 1, ix) * (1.0 - dx) + at(iy + 1, ix + 1) * dx) * dy
                    }
                    TensorInterpolation::Bicubic => {
                        let (iy, ix) = (fy.floor() as isize, fx.floor() as isize);
                        let mut value = 0.0;
                        for j in -1..=2 {
                            for i in -1..=2 {
                                value += at(iy + j, ix + i)
                                    * cubic(fy - (iy + j) as f64)
                                    * cubic(fx - (ix + i) as f64);
                            }
                        }
                        value
                    }
                    TensorInterpolation::Area => {
                        let (top, bottom) = (y * sh / th, ((y + 1) * sh).div_ceil(th));
                        let (left, right) = (x * sw / tw, ((x + 1) * sw).div_ceil(tw));
                        let mut sum = 0.0;
                        for row in top..bottom {
                            for col in left..right {
                                sum += input[row * sw + col];
                            }
                        }
                        sum / ((bottom - top) * (right - left)) as f32
                    }
                };
                output[(plane * th + y) * tw + x] = value;
            }
        }
    }
    Ok(output)
}

fn cubic(distance: f64) -> f32 {
    let x = distance.abs();
    let a = -0.75;
    if x <= 1.0 {
        ((a + 2.0) * x.powi(3) - (a + 3.0) * x.powi(2) + 1.0) as f32
    } else if x < 2.0 {
        (a * x.powi(3) - 5.0 * a * x.powi(2) + 8.0 * a * x - 4.0 * a) as f32
    } else {
        0.0
    }
}
