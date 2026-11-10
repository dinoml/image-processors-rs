//! Shared edge-padded DARK finite differences with source-selected inversion precision.

pub(super) enum HessianPrecision {
    Float32,
    Float64,
}

/// The caller supplies checked nonempty NCHW geometry and its logarithmic maps.
/// Flattened edge padding preserves the upstream negative-coordinate sentinel.
pub(super) fn dark_refine_keypoint(
    maps: &[f32],
    shape: [usize; 4],
    person: usize,
    channel: usize,
    position: [isize; 2],
    precision: HessianPrecision,
) -> [f32; 2] {
    let [persons, channels, height, width] = shape;
    let [x, y] = position;
    let at = |dy: isize, dx: isize| {
        let padded_width = width + 2;
        let padded_plane = (height + 2) * padded_width;
        let index = ((person * channels + channel) * padded_plane) as i128
            + (x as i128 + 1)
            + (y as i128 + 1) * padded_width as i128
            + dx as i128
            + dy as i128 * padded_width as i128;
        let padded_len = persons as i128 * channels as i128 * padded_plane as i128;
        let index = index.rem_euclid(padded_len) as usize;
        let plane = index / padded_plane;
        let within = index % padded_plane;
        let row = (within / padded_width).saturating_sub(1).min(height - 1);
        let column = (within % padded_width).saturating_sub(1).min(width - 1);
        maps[(plane * height + row) * width + column]
    };
    let c = at(0, 0);
    let gx = 0.5 * (at(0, 1) - at(0, -1));
    let gy = 0.5 * (at(1, 0) - at(-1, 0));
    let xx = at(0, 1) - 2.0 * c + at(0, -1);
    let yy = at(1, 0) - 2.0 * c + at(-1, 0);
    let xy = 0.5 * (at(1, 1) - at(0, 1) - at(1, 0) + c + c - at(0, -1) - at(-1, 0) + at(-1, -1));
    match precision {
        HessianPrecision::Float32 => {
            let xx = xx + f32::EPSILON;
            let yy = yy + f32::EPSILON;
            let determinant = xx * yy - xy * xy;
            [
                x as f32 - (yy * gx - xy * gy) / determinant,
                y as f32 - (-xy * gx + xx * gy) / determinant,
            ]
        }
        HessianPrecision::Float64 => {
            // NumPy promotes `hessian + eps * eye(2)` to FP64, then casts the
            // in-place coordinate subtraction back to FP32.
            let xx = f64::from(xx) + f64::from(f32::EPSILON);
            let yy = f64::from(yy) + f64::from(f32::EPSILON);
            let xy = f64::from(xy);
            let determinant = xx * yy - xy * xy;
            let dx = (yy / determinant) * f64::from(gx) + (-xy / determinant) * f64::from(gy);
            let dy = (-xy / determinant) * f64::from(gx) + (xx / determinant) * f64::from(gy);
            [(x as f64 - dx) as f32, (y as f64 - dy) as f32]
        }
    }
}
