//! Host dense inverse-search optical flow, with the DIS ULTRAFAST policy.
//!
//! Mathematics and stage ordering follow OpenCV's modules/video/src/dis_flow.cpp:
//! https://github.com/opencv/opencv/blob/4.x/modules/video/src/dis_flow.cpp
//! Area pyramids, replicated warp borders, integer Sobel derivatives, FP32
//! structure tensors, mean-normalized patch search, eight propagation stripes
//! and residual-weighted densification are preserved. ULTRAFAST has no
//! variational-refinement stage. The implementation is independent of OpenCV.

use super::{
    resize_f32_opencv, resize_frame_opencv, sobel_xy_f64, ImageSize, OpenCvInterpolation,
    SpatialFilterError, TransformError,
};
use crate::media::{ImageFrame, PixelFormat};

/// Failure to prepare or estimate dense inverse-search optical flow.
#[derive(Debug, thiserror::Error)]
pub enum OpticalFlowError {
    /// Input must be same-size byte grayscale frames large enough for a patch.
    #[error("DIS requires same-size grayscale frames with a valid patch pyramid")]
    Geometry,
    /// A scale cannot be represented by the pyramid geometry.
    #[error("DIS finest scale must be between zero and thirty")]
    Scale,
    /// Shared resizing or geometry operation failed.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Shared spatial derivative operation failed.
    #[error(transparent)]
    Filter(#[from] SpatialFilterError),
}

/// Portable host DIS with ULTRAFAST's patch stride and iteration policy.
///
/// A call starts from zero flow, as `DIS.calc(previous, current, None)` does.
/// The requested finest scale is subject to the upstream small-image fallback.
/// This host preprocessing operation can accompany either CPU or GPU inference.
#[derive(Clone, Copy, Debug)]
pub struct DisUltrafast {
    /// Finest requested pyramid scale; the ULTRAFAST preset normally uses two.
    pub finest_scale: u32,
}

impl Default for DisUltrafast {
    fn default() -> Self {
        Self { finest_scale: 2 }
    }
}

impl DisUltrafast {
    /// Estimate row-major [dx, dy] displacement vectors in original pixel units.
    ///
    /// Preserves two directional propagation passes, six inverse-compositional
    /// iterations per pass, and the source's post-update SSD stopping rule.
    ///
    /// # Errors
    /// Rejects incompatible frames, invalid scales or unrepresentable pyramid
    /// dimensions, and propagates shared resize/gradient failures.
    pub fn calculate(
        &self,
        previous: &ImageFrame,
        current: &ImageFrame,
    ) -> Result<Vec<[f32; 2]>, OpticalFlowError> {
        if self.finest_scale > 30 {
            return Err(OpticalFlowError::Scale);
        }
        let size = ImageSize::new(previous.height(), previous.width())?;
        if previous.pixel_format() != PixelFormat::Luma8
            || current.pixel_format() != PixelFormat::Luma8
            || current.width() != size.width
            || current.height() != size.height
            || size.width.max(size.height) > i32::MAX as usize
            || size.width.min(size.height) < 8
        {
            return Err(OpticalFlowError::Geometry);
        }
        let mut patch = 8_usize;
        // The first term rounds the movement search range, the second limits
        // the pyramid to levels containing a complete patch. C++ casts truncate.
        let movement = ((size.width.max(size.height) as f64 / 32.0).log2() + 0.5) as i32;
        let geometry = ((size.width.min(size.height) / patch) as f64).log2() as i32;
        let mut coarsest = movement.min(geometry);
        if coarsest < 0 {
            return Err(OpticalFlowError::Geometry);
        }
        let mut finest = self.finest_scale as usize;
        if (coarsest as usize) < finest {
            patch = if matches!(finest, 3 | 4) { 12 } else { 8 };
            coarsest = ((2.0_f32 * size.width as f32 / (5 * patch) as f32)
                .log2()
                .floor() as i32)
                .max(0);
            finest = (coarsest as usize).saturating_sub(match self.finest_scale {
                3 => 4,
                4 => 5,
                _ => 2,
            });
        }
        let first_size = ImageSize::new(size.height >> finest, size.width >> finest)?;
        let mut previous_pyramid = vec![resize_frame_opencv(
            previous,
            first_size,
            OpenCvInterpolation::Area,
        )?];
        let mut current_pyramid = vec![resize_frame_opencv(
            current,
            first_size,
            OpenCvInterpolation::Area,
        )?];
        for _ in finest..coarsest as usize {
            let index = previous_pyramid.len() - 1;
            let smaller = ImageSize::new(
                previous_pyramid[index].height() / 2,
                previous_pyramid[index].width() / 2,
            )?;
            previous_pyramid.push(resize_frame_opencv(
                &previous_pyramid[index],
                smaller,
                OpenCvInterpolation::Area,
            )?);
            current_pyramid.push(resize_frame_opencv(
                &current_pyramid[index],
                smaller,
                OpenCvInterpolation::Area,
            )?);
        }
        let last = previous_pyramid.len() - 1;
        let mut flow = vec![[0.0_f32; 2]; previous_pyramid[last].data().len()];
        for index in (0..=last).rev() {
            let level = PatchLevel::new(&previous_pyramid[index], &current_pyramid[index], patch)?;
            flow = level.search_and_densify(&flow);
            if index > 0 {
                let target = ImageSize::new(
                    previous_pyramid[index - 1].height(),
                    previous_pyramid[index - 1].width(),
                )?;
                flow = resize_flow(&flow, level.size, target, 2.0)?;
            }
        }
        resize_flow(&flow, first_size, size, (1_u32 << finest) as f32)
    }
}

fn resize_flow(
    flow: &[[f32; 2]],
    source: ImageSize,
    target: ImageSize,
    factor: f32,
) -> Result<Vec<[f32; 2]>, OpticalFlowError> {
    // Reuse the general scalar-plane resize contract for each component, as
    // calcUV does between pyramid levels. No two-channel image adapter is needed.
    let horizontal = flow.iter().map(|vector| vector[0]).collect::<Vec<_>>();
    let vertical = flow.iter().map(|vector| vector[1]).collect::<Vec<_>>();
    let horizontal =
        resize_f32_opencv(&horizontal, source, 1, target, OpenCvInterpolation::Linear)?;
    let vertical = resize_f32_opencv(&vertical, source, 1, target, OpenCvInterpolation::Linear)?;
    Ok(horizontal
        .into_iter()
        .zip(vertical)
        .map(|(x, y)| [x * factor, y * factor])
        .collect())
}

struct PatchLevel<'a> {
    previous: &'a [u8],
    current: &'a [u8],
    size: ImageSize,
    patch: usize,
    columns: usize,
    rows: usize,
    gradient_x: Vec<f32>,
    gradient_y: Vec<f32>,
    // xx, yy, xy, sum(x), sum(y), in sparse row-major order.
    tensors: Vec<[f32; 5]>,
}

impl<'a> PatchLevel<'a> {
    fn new(
        previous: &'a ImageFrame,
        current: &'a ImageFrame,
        patch: usize,
    ) -> Result<Self, OpticalFlowError> {
        let size = ImageSize::new(previous.height(), previous.width())?;
        if size.width < patch || size.height < patch {
            return Err(OpticalFlowError::Geometry);
        }
        // Byte three-tap Sobel results are integer-valued and exactly representable
        // in both FP64 and FP32. Reuse the shared reflect-101 derivative owner.
        let values = previous
            .data()
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let (gradient_x, gradient_y) = sobel_xy_f64(&values, size)?;
        let mut level = Self {
            previous: previous.data(),
            current: current.data(),
            size,
            patch,
            columns: 1 + (size.width - patch) / 4,
            rows: 1 + (size.height - patch) / 4,
            gradient_x: gradient_x.into_iter().map(|value| value as f32).collect(),
            gradient_y: gradient_y.into_iter().map(|value| value as f32).collect(),
            tensors: Vec::new(),
        };
        level.tensors = level.structure_tensors();
        Ok(level)
    }

    fn structure_tensors(&self) -> Vec<[f32; 5]> {
        // Retain the source's horizontal/vertical sliding-sum order instead of
        // independently summing each patch in a different floating-point order.
        let mut horizontal = vec![[0.0; 5]; self.size.height * self.columns];
        for y in 0..self.size.height {
            let terms = |x: usize| {
                let index = y * self.size.width + x;
                let gx = self.gradient_x[index];
                let gy = self.gradient_y[index];
                [gx * gx, gy * gy, gx * gy, gx, gy]
            };
            let mut sums = [0.0_f32; 5];
            for x in 0..self.patch {
                let next = terms(x);
                for axis in 0..5 {
                    sums[axis] += next[axis];
                }
            }
            horizontal[y * self.columns] = sums;
            for x in self.patch..self.size.width {
                let next = terms(x);
                let old = terms(x - self.patch);
                for axis in 0..5 {
                    sums[axis] += next[axis] - old[axis];
                }
                if (x - self.patch + 1).is_multiple_of(4) {
                    horizontal[y * self.columns + (x - self.patch + 1) / 4] = sums;
                }
            }
        }
        let mut tensors = vec![[0.0; 5]; self.rows * self.columns];
        for x in 0..self.columns {
            let mut sums = [0.0_f32; 5];
            for y in 0..self.patch {
                for axis in 0..5 {
                    sums[axis] += horizontal[y * self.columns + x][axis];
                }
            }
            tensors[x] = sums;
            for y in self.patch..self.size.height {
                for axis in 0..5 {
                    sums[axis] += horizontal[y * self.columns + x][axis]
                        - horizontal[(y - self.patch) * self.columns + x][axis];
                }
                if (y - self.patch + 1).is_multiple_of(4) {
                    tensors[(y - self.patch + 1) / 4 * self.columns + x] = sums;
                }
            }
        }
        tensors
    }

    fn patch_distance(&self, x: usize, y: usize, flow: [f32; 2], gradient: bool) -> [f32; 3] {
        // Equivalent to the source's 16-pixel BORDER_REPLICATE buffer, including
        // its patch-origin clamp before bilinear interpolation.
        let padded_x = (x as f32 + flow[0] + 16.0)
            .clamp(17.0 - self.patch as f32, self.size.width as f32 + 15.0);
        let padded_y = (y as f32 + flow[1] + 16.0)
            .clamp(17.0 - self.patch as f32, self.size.height as f32 + 15.0);
        let ix = padded_x.floor() as isize - 16;
        let iy = padded_y.floor() as isize - 16;
        let fx = padded_x - padded_x.floor();
        let fy = padded_y - padded_y.floor();
        let weights = [
            (1.0 - fy) * (1.0 - fx),
            (1.0 - fy) * fx,
            fy * (1.0 - fx),
            fy * fx,
        ];
        let sample = |sx: isize, sy: isize| {
            f32::from(
                self.current[sy.clamp(0, self.size.height as isize - 1) as usize * self.size.width
                    + sx.clamp(0, self.size.width as isize - 1) as usize],
            )
        };
        let mut sum = 0.0_f32;
        let mut squared = 0.0_f32;
        let mut dx = 0.0_f32;
        let mut dy = 0.0_f32;
        for py in 0..self.patch {
            for px in 0..self.patch {
                let index = (y + py) * self.size.width + x + px;
                let sx = ix + px as isize;
                let sy = iy + py as isize;
                let diff = weights[0] * sample(sx, sy)
                    + weights[1] * sample(sx + 1, sy)
                    + weights[2] * sample(sx, sy + 1)
                    + weights[3] * sample(sx + 1, sy + 1)
                    - f32::from(self.previous[index]);
                sum += diff;
                squared += diff * diff;
                if gradient {
                    dx += diff * self.gradient_x[index];
                    dy += diff * self.gradient_y[index];
                }
            }
        }
        let count = (self.patch * self.patch) as f32;
        let tensor = self.tensors[(y / 4) * self.columns + x / 4];
        [
            squared - sum * sum / count,
            dx - sum * tensor[3] / count,
            dy - sum * tensor[4] / count,
        ]
    }

    fn search_and_densify(&self, initial: &[[f32; 2]]) -> Vec<[f32; 2]> {
        let mut sparse = vec![[0.0_f32; 2]; self.rows * self.columns];
        let stripe_height = self.rows.div_ceil(8);
        for stripe in 0..8 {
            let first = (stripe * stripe_height).min(self.rows);
            let end = ((stripe + 1) * stripe_height).min(self.rows);
            for reverse in [false, true] {
                for row_index in 0..end - first {
                    let row = if reverse {
                        end - row_index - 1
                    } else {
                        first + row_index
                    };
                    let y = row * 4;
                    for column_index in 0..self.columns {
                        let column = if reverse {
                            self.columns - column_index - 1
                        } else {
                            column_index
                        };
                        let x = column * 4;
                        let index = row * self.columns + column;
                        let mut best = if reverse {
                            sparse[index]
                        } else {
                            initial[(y + self.patch / 2) * self.size.width + x + self.patch / 2]
                        };
                        let mut distance = self.patch_distance(x, y, best, false)[0];
                        let mut candidates = [None; 2];
                        if column_index > 0 {
                            candidates[0] =
                                Some(sparse[if reverse { index + 1 } else { index - 1 }]);
                        }
                        if row_index > 0 {
                            candidates[1] = Some(
                                sparse[if reverse {
                                    index + self.columns
                                } else {
                                    index - self.columns
                                }],
                            );
                        }
                        for candidate in candidates.into_iter().flatten() {
                            let cost = self.patch_distance(x, y, candidate, false)[0];
                            if cost < distance {
                                best = candidate;
                                distance = cost;
                            }
                        }
                        let [xx, yy, xy, _, _] = self.tensors[index];
                        let determinant = xx * yy - xy * xy;
                        let determinant = if determinant.abs() < 0.001 {
                            0.001
                        } else {
                            determinant
                        };
                        let inverse = [yy / determinant, -xy / determinant, xx / determinant];
                        let mut estimate = best;
                        let mut previous_ssd = 1e10_f32;
                        for _ in 0..6 {
                            let [ssd, dx, dy] = self.patch_distance(x, y, estimate, true);
                            estimate[0] -= inverse[0] * dx + inverse[1] * dy;
                            estimate[1] -= inverse[1] * dx + inverse[2] * dy;
                            if ssd >= previous_ssd {
                                break;
                            }
                            previous_ssd = ssd;
                        }
                        // Source norm promotes to double after FP32 subtraction.
                        let delta_x = f64::from(estimate[0] - best[0]);
                        let delta_y = f64::from(estimate[1] - best[1]);
                        sparse[index] = if (delta_x * delta_x + delta_y * delta_y).sqrt()
                            <= self.patch as f64
                        {
                            estimate
                        } else {
                            best
                        };
                    }
                }
            }
        }
        let mut dense = vec![[0.0; 2]; initial.len()];
        for y in 0..self.size.height {
            let first_row = if y < self.patch {
                0
            } else {
                ((y - self.patch) / 4 + 1).min(self.rows - 1)
            };
            let last_row = (y / 4).min(self.rows - 1);
            for x in 0..self.size.width {
                let first_column = if x < self.patch {
                    0
                } else {
                    ((x - self.patch) / 4 + 1).min(self.columns - 1)
                };
                let last_column = (x / 4).min(self.columns - 1);
                let mut sum = [0.0_f32; 2];
                let mut sum_weights = 0.0_f32;
                for row in first_row..=last_row {
                    for column in first_column..=last_column {
                        let vector = sparse[row * self.columns + column];
                        let sx =
                            (x as f32 + vector[0]).clamp(0.0, self.size.width as f32 - 1.0 - 0.001);
                        let sy = (y as f32 + vector[1])
                            .clamp(0.0, self.size.height as f32 - 1.0 - 0.001);
                        let left = sx as usize;
                        let top = sy as usize;
                        let fx = sx - left as f32;
                        let fy = sy - top as f32;
                        let value = fx
                            * fy
                            * f32::from(self.current[(top + 1) * self.size.width + left + 1])
                            + (1.0 - fx)
                                * fy
                                * f32::from(self.current[(top + 1) * self.size.width + left])
                            + fx * (1.0 - fy)
                                * f32::from(self.current[top * self.size.width + left + 1])
                            + (1.0 - fx)
                                * (1.0 - fy)
                                * f32::from(self.current[top * self.size.width + left]);
                        let difference = value - f32::from(self.previous[y * self.size.width + x]);
                        let weight = 1.0 / difference.abs().max(1.0);
                        sum[0] += weight * vector[0];
                        sum[1] += weight * vector[1];
                        sum_weights += weight;
                    }
                }
                dense[y * self.size.width + x] = [sum[0] / sum_weights, sum[1] / sum_weights];
            }
        }
        dense
    }
}
