//! Grayscale document skew search and same-canvas page straightening.

use super::DocumentError;
use crate::{
    media::{ImageFrame, PixelFormat},
    transforms::{self, CanvasFill, ImageSize, OpenCvInterpolation, Padding},
};

fn sum_f32(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return values.iter().fold(-0.0f32, |sum, &value| sum + value);
    }
    if values.len() <= 128 {
        let mut lanes = [0.0f32; 8];
        lanes.copy_from_slice(&values[..8]);
        let stop = values.len() / 8 * 8;
        for block in values[8..stop].chunks_exact(8) {
            for (lane, &value) in lanes.iter_mut().zip(block) {
                *lane += value;
            }
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for &value in &values[stop..] {
            total += value;
        }
        total
    } else {
        let middle = (values.len() / 2) & !7;
        sum_f32(&values[..middle]) + sum_f32(&values[middle..])
    }
}

fn rotation(width: usize, height: usize, degrees: f64) -> [[f64; 3]; 2] {
    let radians = degrees.to_radians();
    let (sine, cosine) = radians.sin_cos();
    let (x, y) = (width as f64 / 2.0, height as f64 / 2.0);
    [
        [cosine, sine, (1.0 - cosine) * x - sine * y],
        [-sine, cosine, sine * x + (1.0 - cosine) * y],
    ]
}

/// Estimate document skew with the original coarse/fine horizontal-ink search.
///
/// The long-side thumbnail is 800 pixels. Affine interpolation and resize
/// arithmetic reuse existing Rust owners; no native OpenCV is loaded.
///
/// # Errors
/// Reports non-grayscale input, invalid geometry or shared resampling failures.
pub fn estimate_document_skew(frame: &ImageFrame) -> Result<f64, DocumentError> {
    if frame.pixel_format() != PixelFormat::Luma8 {
        return Err(DocumentError::InvalidPage("skew search requires Luma8"));
    }
    let scale = 800.0 / frame.width().max(frame.height()) as f64;
    let size = ImageSize::new(
        (frame.height() as f64 * scale).round_ties_even().max(1.0) as usize,
        (frame.width() as f64 * scale).round_ties_even().max(1.0) as usize,
    )?;
    let thumbnail = transforms::resize_frame_opencv(frame, size, OpenCvInterpolation::Area)?;
    let ink = thumbnail
        .data()
        .iter()
        .map(|&value| (255.0 - f32::from(value)) / 255.0)
        .collect::<Vec<_>>();
    if (sum_f32(&ink) / ink.len() as f32) < 0.003 {
        return Ok(0.0);
    }
    let sharpness = |angle: f64| -> Result<f32, DocumentError> {
        let rotated = transforms::warp_affine_f32(
            &ink,
            size,
            size,
            rotation(size.width, size.height, angle),
            &[0.0],
        )?;
        let profile = rotated
            .chunks_exact(size.width)
            .map(sum_f32)
            .collect::<Vec<_>>();
        let differences = profile
            .windows(2)
            .map(|pair| {
                let difference = pair[1] - pair[0];
                difference * difference
            })
            .collect::<Vec<_>>();
        Ok(sum_f32(&differences))
    };
    let mut best = -5.0;
    let mut score = f32::NEG_INFINITY;
    for index in 0..=20 {
        let angle = -5.0 + index as f64 * 0.5;
        let candidate = sharpness(angle)?;
        if candidate > score {
            best = angle;
            score = candidate;
        }
    }
    let start = best - 0.5;
    let step = (start + 0.1) - start;
    score = f32::NEG_INFINITY;
    for index in 0..=10 {
        let angle = start + index as f64 * step;
        let candidate = sharpness(angle)?;
        if candidate > score {
            best = angle;
            score = candidate;
        }
    }
    Ok(best)
}

/// Straighten a page on its original canvas, replicating source edge pixels.
///
/// # Errors
/// Reports invalid grayscale input, angles outside the skew search envelope,
/// coordinate overflow or shared transform failures.
pub fn rotate_document_page(frame: &ImageFrame, degrees: f64) -> Result<ImageFrame, DocumentError> {
    if frame.pixel_format() != PixelFormat::Luma8 || !degrees.is_finite() || degrees.abs() > 5.5 {
        return Err(DocumentError::InvalidPage(
            "invalid document deskew request",
        ));
    }
    let size = ImageSize::new(frame.height(), frame.width())?;
    let inverse = rotation(frame.width(), frame.height(), -degrees);
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (x, y) in [
        (0.0, 0.0),
        ((frame.width() - 1) as f64, 0.0),
        (0.0, (frame.height() - 1) as f64),
        ((frame.width() - 1) as f64, (frame.height() - 1) as f64),
    ] {
        let xx = inverse[0][0] * x + inverse[0][1] * y + inverse[0][2];
        let yy = inverse[1][0] * x + inverse[1][1] * y + inverse[1][2];
        bounds[0] = bounds[0].min(xx);
        bounds[1] = bounds[1].min(yy);
        bounds[2] = bounds[2].max(xx);
        bounds[3] = bounds[3].max(yy);
    }
    let left = (-bounds[0]).ceil().max(0.0) as usize + 1;
    let top = (-bounds[1]).ceil().max(0.0) as usize + 1;
    let right = (bounds[2] - (frame.width() - 1) as f64).ceil().max(0.0) as usize + 1;
    let bottom = (bounds[3] - (frame.height() - 1) as f64).ceil().max(0.0) as usize + 1;
    let padded = transforms::pad_frame_with_canvas_fill(
        frame,
        Padding::new(top, right, bottom, left),
        CanvasFill::ImageEdges,
    )?;
    let mut forward = rotation(frame.width(), frame.height(), degrees);
    for row in &mut forward {
        row[2] -= row[0] * left as f64 + row[1] * top as f64;
    }
    let bytes = transforms::warp_affine_u8(
        padded.data(),
        ImageSize::new(padded.height(), padded.width())?,
        size,
        forward,
        &[0],
    )?;
    Ok(ImageFrame::new(
        frame.width(),
        frame.height(),
        PixelFormat::Luma8,
        bytes,
    )?)
}

/// Map straightened line boxes back to the upright page, rounded to tenths.
pub fn unrotate_document_box(bounds: [f64; 4], degrees: f64, size: [usize; 2]) -> [f64; 4] {
    let matrix = rotation(size[0], size[1], -degrees);
    let mut result = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (x, y) in [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
    ] {
        let xx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
        let yy = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
        result[0] = result[0].min(xx);
        result[1] = result[1].min(yy);
        result[2] = result[2].max(xx);
        result[3] = result[3].max(yy);
    }
    for (index, value) in result.iter_mut().enumerate() {
        *value = ((*value).clamp(0.0, size[index % 2] as f64) * 10.0).round_ties_even() / 10.0;
    }
    result
}
