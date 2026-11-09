//! Control-image transforms using the shared Rust resize and spatial-filter owners.

use super::{
    composite_alpha_to_rgb, crop_frame, dilate_f32, gaussian_blur_f32, gaussian_blur_u8,
    pad_frame_to_multiple_with_canvas_fill, remap_bilinear_u8, resize_f32_opencv,
    resize_f64_opencv, resize_frame_opencv, resize_frame_opencv_scale_dependent, CanvasFill,
    ImageCropBox, ImageSize, OpenCvInterpolation, TransformError,
};
use crate::{ImageFrame, PixelFormat};

fn invalid(message: impl Into<String>) -> TransformError {
    TransformError::ResizeKernel {
        message: message.into(),
    }
}

fn size(frame: &ImageFrame) -> ImageSize {
    ImageSize {
        height: frame.height(),
        width: frame.width(),
    }
}

fn rgb(frame: &ImageFrame) -> Result<ImageFrame, TransformError> {
    let pixels = composite_alpha_to_rgb(frame.data(), frame.pixel_format(), [255; 3])?;
    ImageFrame::new(frame.width(), frame.height(), PixelFormat::Rgb8, pixels)
        .map_err(|error| invalid(error.to_string()))
}

fn frame(size: ImageSize, pixels: Vec<u8>) -> Result<ImageFrame, TransformError> {
    ImageFrame::new(size.width, size.height, PixelFormat::Rgb8, pixels)
        .map_err(|error| invalid(error.to_string()))
}

fn crop(image: &ImageFrame, content: ImageSize) -> Result<ImageFrame, TransformError> {
    crop_frame(
        image,
        ImageCropBox::new(0, 0, content.width, content.height),
    )
}

/// Aspect-preserving short-edge resize followed by bottom/right edge padding to 64.
/// Returns the padded RGB image and its unpadded content size. Resolution zero
/// bypasses both resize and padding, following ControlNet auxiliary utilities.
///
/// # Errors
/// Reports invalid or overflowing dimensions and shared pixel/resize failures.
pub fn auxiliary_resize_pad(
    image: &ImageFrame,
    resolution: usize,
    enlargement: OpenCvInterpolation,
) -> Result<(ImageFrame, ImageSize), TransformError> {
    let image = rgb(image)?;
    if resolution == 0 {
        let content = size(&image);
        return Ok((image, content));
    }
    let scale = resolution as f64 / image.height().min(image.width()) as f64;
    let height = (image.height() as f64 * scale).round_ties_even();
    let width = (image.width() as f64 * scale).round_ties_even();
    if !height.is_finite()
        || !width.is_finite()
        || height < 1.0
        || width < 1.0
        || height >= usize::MAX as f64
        || width >= usize::MAX as f64
    {
        return Err(invalid("auxiliary resize dimensions overflow"));
    }
    let content = ImageSize {
        height: height as usize,
        width: width as usize,
    };
    let resized = resize_frame_opencv_scale_dependent(&image, content, enlargement)?;
    Ok((
        pad_frame_to_multiple_with_canvas_fill(
            &resized,
            ImageSize {
                height: 64,
                width: 64,
            },
            CanvasFill::ImageEdges,
        )?,
        content,
    ))
}

/// Remove every foreground component smaller than `minimum_size` in a boolean
/// row-major array of one through three dimensions. Connectivity is the maximum
/// number of changed axes in a neighboring offset, as in scikit-image's
/// `generate_binary_structure`. No largest-island exception is applied.
///
/// # Errors
/// Rejects invalid shape, storage, overflow or connectivity outside 1..=rank.
pub fn remove_small_objects(
    mask: &[bool],
    shape: &[usize],
    minimum_size: usize,
    connectivity: usize,
) -> Result<Vec<bool>, TransformError> {
    if shape.is_empty()
        || shape.len() > 3
        || connectivity == 0
        || connectivity > shape.len()
        || shape.contains(&0)
        || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(mask.len())
    {
        return Err(invalid("invalid small-object shape or connectivity"));
    }
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    let mut offsets = Vec::new();
    for code in 0..3usize.pow(shape.len() as u32) {
        let mut digits = code;
        let mut offset = vec![0isize; shape.len()];
        for value in &mut offset {
            *value = (digits % 3) as isize - 1;
            digits /= 3;
        }
        let changed = offset.iter().filter(|&&value| value != 0).count();
        if changed > 0 && changed <= connectivity {
            offsets.push(offset);
        }
    }
    let mut visited = vec![false; mask.len()];
    let mut result = mask.to_vec();
    let mut pending = Vec::new();
    let mut component = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        pending.push(start);
        visited[start] = true;
        component.clear();
        while let Some(index) = pending.pop() {
            component.push(index);
            for offset in &offsets {
                let mut neighbor = 0usize;
                let mut inside = true;
                for axis in 0..shape.len() {
                    let coordinate = index / strides[axis] % shape[axis];
                    if let Some(value) = coordinate
                        .checked_add_signed(offset[axis])
                        .filter(|&value| value < shape[axis])
                    {
                        neighbor += value * strides[axis];
                    } else {
                        inside = false;
                        break;
                    }
                }
                if inside && mask[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        if component.len() < minimum_size {
            for &index in &component {
                result[index] = false;
            }
        }
    }
    Ok(result)
}

/// Standard line extraction with FP32 Gaussian subtraction and median scaling.
///
/// # Errors
/// Reports invalid Gaussian parameters or shared processing failures.
pub fn standard_lineart(
    image: &ImageFrame,
    resolution: usize,
    sigma: f64,
    threshold: f32,
) -> Result<ImageFrame, TransformError> {
    let (image, content) = auxiliary_resize_pad(image, resolution, OpenCvInterpolation::Cubic)?;
    let values = image
        .data()
        .iter()
        .map(|&v| f32::from(v))
        .collect::<Vec<_>>();
    let blurred = gaussian_blur_f32(&values, size(&image), 3, sigma, None)
        .map_err(|error| invalid(error.to_string()))?;
    let intensity = blurred
        .chunks_exact(3)
        .zip(values.chunks_exact(3))
        .map(|(g, x)| {
            (g[0] - x[0])
                .min(g[1] - x[1])
                .min(g[2] - x[2])
                .clamp(0.0, 255.0)
        })
        .collect::<Vec<_>>();
    let mut selected = intensity
        .iter()
        .copied()
        .filter(|&v| v > threshold)
        .collect::<Vec<_>>();
    selected.sort_unstable_by(f32::total_cmp);
    // Python max(16, median(empty)) keeps its first argument when median is NaN.
    let median = if selected.is_empty() {
        16.0
    } else if selected.len() % 2 == 0 {
        (selected[selected.len() / 2 - 1] + selected[selected.len() / 2]) / 2.0
    } else {
        selected[selected.len() / 2]
    };
    let denominator = median.max(16.0);
    let pixels = intensity
        .into_iter()
        .flat_map(|v| [((v / denominator) * 127.0).clamp(0.0, 255.0) as u8; 3])
        .collect();
    crop(&frame(size(&image), pixels)?, content)
}

/// Threshold the minimum RGB channel at 127 using the basic scribble polarity.
/// # Errors
/// Reports resize, padding or pixel-format failures.
pub fn scribble_binary(
    image: &ImageFrame,
    resolution: usize,
) -> Result<ImageFrame, TransformError> {
    let (image, content) = auxiliary_resize_pad(image, resolution, OpenCvInterpolation::Area)?;
    let pixels = image
        .data()
        .chunks_exact(3)
        .flat_map(|p| {
            [if *p.iter().min().unwrap_or(&255) < 127 {
                0
            } else {
                255
            }; 3]
        })
        .collect();
    crop(&frame(size(&image), pixels)?, content)
}

/// Extended difference-of-Gaussian scribbles with source byte truncation.
/// # Errors
/// Reports shared spatial-filter or image-processing failures.
pub fn scribble_xdog(
    image: &ImageFrame,
    resolution: usize,
    threshold: f32,
) -> Result<ImageFrame, TransformError> {
    let (image, content) = auxiliary_resize_pad(image, resolution, OpenCvInterpolation::Cubic)?;
    let values = image
        .data()
        .iter()
        .map(|&v| f32::from(v))
        .collect::<Vec<_>>();
    let first = gaussian_blur_f32(&values, size(&image), 3, 0.5, None)
        .map_err(|e| invalid(e.to_string()))?;
    let second = gaussian_blur_f32(&values, size(&image), 3, 5.0, None)
        .map_err(|e| invalid(e.to_string()))?;
    let pixels = first
        .chunks_exact(3)
        .zip(second.chunks_exact(3))
        .flat_map(|(a, b)| {
            let dog =
                (255.0 - (b[0] - a[0]).min(b[1] - a[1]).min(b[2] - a[2])).clamp(0.0, 255.0) as u8;
            // The source expression remains uint8 and wraps before comparison.
            [if f32::from((255u8 - dog).wrapping_mul(2)) > threshold {
                255
            } else {
                0
            }; 3]
        })
        .collect();
    crop(&frame(size(&image), pixels)?, content)
}

fn pyramid_axis(index: usize, length: usize) -> [(usize, u32); 3] {
    let center = index / 2;
    if index % 2 == 1 {
        [(center, 4), ((center + 1).min(length - 1), 4), (center, 0)]
    } else {
        let previous = if center == 0 {
            1.min(length - 1)
        } else {
            center - 1
        };
        [
            (previous, 1),
            (center, 6),
            ((center + 1).min(length - 1), 1),
        ]
    }
}

/// Double both byte-image axes with the separable [1,4,6,4,1] pyramid kernel.
/// The expanded-grid reflect-101 boundary and single final integer rounding
/// match the source pyrUp operation, including its last column and row.
/// # Errors
/// Reports overflowing output geometry or invalid pixel storage.
pub fn pyramid_expand_u8(image: &ImageFrame) -> Result<ImageFrame, TransformError> {
    let width = image
        .width()
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let height = image
        .height()
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let channels = image.channels();
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut pixels = vec![0; len];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let mut sum = 0u32;
                for (sy, wy) in pyramid_axis(y, image.height()) {
                    for (sx, wx) in pyramid_axis(x, image.width()) {
                        sum +=
                            u32::from(image.data()[(sy * image.width() + sx) * channels + channel])
                                * wx
                                * wy;
                    }
                }
                pixels[(y * width + x) * channels + channel] = ((sum + 32) >> 6) as u8;
            }
        }
    }
    ImageFrame::new(width, height, image.pixel_format(), pixels).map_err(|e| invalid(e.to_string()))
}

/// Downsample to rounded multiples of 64, then apply repeated pyramid expansion.
/// # Errors
/// Reports invalid iteration count, collapsed geometry or shared resize failures.
pub fn tile_preprocess(image: &ImageFrame, iterations: u32) -> Result<ImageFrame, TransformError> {
    let image = rgb(image)?;
    let divisor = 1usize
        .checked_shl(iterations)
        .ok_or_else(|| invalid("tile iteration count overflows"))?;
    let target = ImageSize {
        height: ((image.height() as f64 / 64.0).round_ties_even() as usize * 64) / divisor,
        width: ((image.width() as f64 / 64.0).round_ties_even() as usize * 64) / divisor,
    };
    let mut output = resize_frame_opencv(&image, target, OpenCvInterpolation::Area)?;
    for _ in 0..iterations {
        output = pyramid_expand_u8(&output)?;
    }
    Ok(output)
}

/// Cubic coarse palette followed by nearest-neighbor expansion.
/// # Errors
/// Reports zero resolution, collapsed palette dimensions or resize failures.
pub fn color_palette(image: &ImageFrame, resolution: usize) -> Result<ImageFrame, TransformError> {
    let image = rgb(image)?;
    let scale = resolution as f64 / image.height().min(image.width()) as f64;
    let target = ImageSize {
        height: (image.height() as f64 * scale).round_ties_even() as usize,
        width: (image.width() as f64 * scale).round_ties_even() as usize,
    };
    let image = resize_frame_opencv(&image, target, OpenCvInterpolation::Area)?;
    let palette = resize_frame_opencv(
        &image,
        ImageSize {
            height: target.height / 64,
            width: target.width / 64,
        },
        OpenCvInterpolation::Cubic,
    )?;
    resize_frame_opencv(&palette, target, OpenCvInterpolation::Nearest)
}

fn reflect101(mut index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let last = length as isize - 1;
    while index < 0 || index > last {
        index = if index < 0 { -index } else { 2 * last - index };
    }
    index as usize
}

fn box_mean(values: &[f32], shape: ImageSize, channels: usize, radius: usize) -> Vec<f32> {
    let support = 2 * radius + 1;
    let mut rows = vec![0.0f64; values.len()];
    let mut output = vec![0.0; values.len()];
    for y in 0..shape.height {
        for c in 0..channels {
            let mut sum = 0.0f64;
            for offset in -(radius as isize)..=radius as isize {
                sum += f64::from(
                    values[(y * shape.width + reflect101(offset, shape.width)) * channels + c],
                );
            }
            for x in 0..shape.width {
                rows[(y * shape.width + x) * channels + c] = sum;
                let old = reflect101(x as isize - radius as isize, shape.width);
                let new = reflect101(x as isize + radius as isize + 1, shape.width);
                sum += f64::from(values[(y * shape.width + new) * channels + c])
                    - f64::from(values[(y * shape.width + old) * channels + c]);
            }
        }
    }
    let denominator = (support * support) as f64;
    for x in 0..shape.width {
        for c in 0..channels {
            let mut sum = 0.0f64;
            for offset in -(radius as isize)..=radius as isize {
                sum += rows[(reflect101(offset, shape.height) * shape.width + x) * channels + c];
            }
            for y in 0..shape.height {
                output[(y * shape.width + x) * channels + c] = (sum / denominator) as f32;
                let old = reflect101(y as isize - radius as isize, shape.height);
                let new = reflect101(y as isize + radius as isize + 1, shape.height);
                sum += rows[(new * shape.width + x) * channels + c]
                    - rows[(old * shape.width + x) * channels + c];
            }
        }
    }
    output
}

/// Color fast guided filtering with nearest reduction and linear coefficient
/// expansion. Uses the source FP32 covariance/cofactor arithmetic and truncating
/// uint8 output; no external image-processing library is linked.
/// # Errors
/// Reports nonpositive epsilon/scale, invalid radius or shared resize failures.
pub fn fast_guided_filter_rgb(
    image: &ImageFrame,
    radius: usize,
    epsilon: f64,
    scale: f64,
) -> Result<ImageFrame, TransformError> {
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || !scale.is_finite()
        || scale < 1.0
        || radius > isize::MAX as usize / 4
    {
        return Err(invalid("invalid guided-filter parameters"));
    }
    let image = rgb(image)?;
    let source = size(&image);
    let small = ImageSize {
        height: (source.height as f64 / scale) as usize,
        width: (source.width as f64 / scale) as usize,
    };
    let values = image
        .data()
        .iter()
        .map(|&v| (1.0f32 / 255.0) * f32::from(v))
        .collect::<Vec<_>>();
    let guide = resize_f32_opencv(&values, source, 3, small, OpenCvInterpolation::Nearest)?;
    let radius = (radius as f64 / scale) as usize;
    let mean = box_mean(&guide, small, 3, radius);
    let mut products = Vec::with_capacity(guide.len() * 2);
    for p in guide.chunks_exact(3) {
        products.extend_from_slice(&[
            p[0] * p[0],
            p[0] * p[1],
            p[0] * p[2],
            p[1] * p[1],
            p[1] * p[2],
            p[2] * p[2],
        ]);
    }
    let moments = box_mean(&products, small, 6, radius);
    let mut inverse = Vec::with_capacity(products.len());
    for (m, v) in mean.chunks_exact(3).zip(moments.chunks_exact(6)) {
        let rr = v[0] - m[0] * m[0] + epsilon as f32;
        let rg = v[1] - m[0] * m[1];
        let rb = v[2] - m[0] * m[2];
        let gg = v[3] - m[1] * m[1] + epsilon as f32;
        let gb = v[4] - m[1] * m[2];
        let bb = v[5] - m[2] * m[2] + epsilon as f32;
        let cofactors = [
            gg * bb - gb * gb,
            gb * rb - rg * bb,
            rg * gb - gg * rb,
            rr * bb - rb * rb,
            rb * rg - rr * gb,
            rr * gg - rg * rg,
        ];
        let determinant = cofactors[0] * rr + cofactors[1] * rg + cofactors[2] * rb;
        inverse.extend(cofactors.map(|v| v / determinant));
    }
    let mut output = vec![0.0; values.len()];
    for channel in 0..3 {
        let plane = guide
            .chunks_exact(3)
            .map(|v| v[channel])
            .collect::<Vec<_>>();
        let pmean = box_mean(&plane, small, 1, radius);
        let cross = guide
            .chunks_exact(3)
            .zip(&plane)
            .flat_map(|(v, &p)| [v[0] * p, v[1] * p, v[2] * p])
            .collect::<Vec<_>>();
        let cross = box_mean(&cross, small, 3, radius);
        let mut coefficients = Vec::with_capacity(plane.len() * 4);
        for index in 0..plane.len() {
            let m = &mean[index * 3..index * 3 + 3];
            let v = &inverse[index * 6..index * 6 + 6];
            let cov = [
                cross[index * 3] - m[0] * pmean[index],
                cross[index * 3 + 1] - m[1] * pmean[index],
                cross[index * 3 + 2] - m[2] * pmean[index],
            ];
            let a = [
                v[0] * cov[0] + v[1] * cov[1] + v[2] * cov[2],
                v[1] * cov[0] + v[3] * cov[1] + v[4] * cov[2],
                v[2] * cov[0] + v[4] * cov[1] + v[5] * cov[2],
            ];
            coefficients.extend_from_slice(&[
                a[0],
                a[1],
                a[2],
                pmean[index] - a[0] * m[0] - a[1] * m[1] - a[2] * m[2],
            ]);
        }
        let coefficients = box_mean(&coefficients, small, 4, radius);
        let coefficients =
            resize_f32_opencv(&coefficients, small, 4, source, OpenCvInterpolation::Linear)?;
        for (index, (a, p)) in coefficients
            .chunks_exact(4)
            .zip(values.chunks_exact(3))
            .enumerate()
        {
            output[index * 3 + channel] = a[0] * p[0] + a[1] * p[1] + a[2] * p[2] + a[3];
        }
    }
    // NumPy astype(uint8) truncates and wraps before its redundant uint8 clip.
    let pixels = output
        .into_iter()
        .map(|v| ((255.0 * v).trunc() as i64).rem_euclid(256) as u8)
        .collect();
    frame(source, pixels)
}

/// TTPlanet simple tile: area reduction, cubic expansion, then byte Gaussian blur.
/// # Errors
/// Reports invalid scale/blur or shared image-processing failures.
pub fn tile_preprocess_simple(
    image: &ImageFrame,
    scale: f64,
    blur: f64,
) -> Result<ImageFrame, TransformError> {
    let image = rgb(image)?;
    let resized = tile_resize(&image, scale)?;
    byte_blur(&resized, blur)
}

/// TTPlanet guided tile: byte Gaussian blur, fast guided filtering, then resizing.
/// # Errors
/// Reports invalid parameters or shared image-processing failures.
pub fn tile_preprocess_guided(
    image: &ImageFrame,
    scale: f64,
    blur: f64,
    radius: usize,
    epsilon: f64,
) -> Result<ImageFrame, TransformError> {
    let image = rgb(image)?;
    let blurred = byte_blur(&image, blur)?;
    // Upstream filters BGR. Reversing channels retains its FP32 cofactor order.
    let bgr = frame(
        size(&blurred),
        blurred
            .data()
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect(),
    )?;
    let filtered = fast_guided_filter_rgb(&bgr, radius, epsilon, scale)?;
    let resized = tile_resize(&filtered, scale)?;
    frame(
        size(&resized),
        resized
            .data()
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect(),
    )
}

fn tile_resize(image: &ImageFrame, scale: f64) -> Result<ImageFrame, TransformError> {
    if !scale.is_finite() || scale < 1.0 {
        return Err(invalid("tile scale must be finite and at least one"));
    }
    let small = ImageSize {
        height: (image.height() as f64 / scale) as usize,
        width: (image.width() as f64 / scale) as usize,
    };
    let down = resize_frame_opencv(image, small, OpenCvInterpolation::Area)?;
    resize_frame_opencv(&down, size(image), OpenCvInterpolation::Cubic)
}

fn byte_blur(image: &ImageFrame, blur: f64) -> Result<ImageFrame, TransformError> {
    if !blur.is_finite() || blur < 1.0 || blur >= isize::MAX as f64 / 4.0 {
        return Err(invalid("invalid tile blur strength"));
    }
    let kernel = (blur as usize) | 1;
    let pixels = gaussian_blur_u8(
        image.data(),
        size(image),
        3,
        blur / 2.0,
        Some(ImageSize {
            height: kernel,
            width: kernel,
        }),
    )
    .map_err(|e| invalid(e.to_string()))?;
    frame(size(image), pixels)
}

/// Remap content using independent normalized bicubic noise fields. The caller
/// supplies a persistent uniform generator, allowing NumPy-compatible seeded
/// streams as well as fresh streams without a model-local random implementation.
/// # Errors
/// Reports invalid field scale, nonfinite random samples or resize/remap errors.
pub fn content_shuffle(
    image: &ImageFrame,
    resolution: usize,
    field_size: usize,
    random: &mut impl FnMut() -> f64,
) -> Result<ImageFrame, TransformError> {
    if field_size == 0 {
        return Err(invalid("shuffle field scale must be positive"));
    }
    let (image, content) = auxiliary_resize_pad(image, resolution, OpenCvInterpolation::Cubic)?;
    let shape = size(&image);
    let noise_size = ImageSize {
        height: shape.height / field_size + 2,
        width: shape.width / field_size + 2,
    };
    let expanded = ImageSize {
        height: shape
            .height
            .checked_add(
                field_size
                    .checked_mul(2)
                    .ok_or(TransformError::ImageSizeOverflow)?,
            )
            .ok_or(TransformError::ImageSizeOverflow)?,
        width: shape
            .width
            .checked_add(field_size * 2)
            .ok_or(TransformError::ImageSizeOverflow)?,
    };
    let mut fields = Vec::with_capacity(2);
    for _ in 0..2 {
        let noise = (0..noise_size.height * noise_size.width)
            .map(|_| random())
            .collect::<Vec<_>>();
        if noise
            .iter()
            .any(|v| !v.is_finite() || !(0.0..1.0).contains(v))
        {
            return Err(invalid("shuffle random values must be in [0,1)"));
        }
        let noise = resize_f64_opencv(&noise, noise_size, 1, expanded, OpenCvInterpolation::Cubic)?;
        let mut field = Vec::with_capacity(shape.height * shape.width);
        for row in 0..shape.height {
            field.extend_from_slice(
                &noise[(row + field_size) * expanded.width + field_size
                    ..(row + field_size) * expanded.width + field_size + shape.width],
            );
        }
        let minimum = field.iter().copied().fold(f64::INFINITY, f64::min);
        for v in &mut field {
            *v -= minimum;
        }
        let maximum = field.iter().copied().fold(0.0, f64::max);
        for v in &mut field {
            *v /= maximum;
        }
        fields.push(field);
    }
    let coordinates = fields[0]
        .iter()
        .zip(&fields[1])
        .map(|(&x, &y)| {
            [
                (x * (shape.width - 1) as f64) as f32,
                (y * (shape.height - 1) as f64) as f32,
            ]
        })
        .collect::<Vec<_>>();
    let pixels = remap_bilinear_u8(image.data(), shape, shape, &coordinates, &[0; 3])
        .map_err(|e| invalid(e.to_string()))?;
    crop(&frame(shape, pixels)?, content)
}

/// Directional nonmaximum suppression, Gaussian blur and byte thresholding used
/// by auxiliary HED/PiDiNet scribbles. RGB channels are treated independently,
/// preserving the wrapper's second call on already repeated-channel images.
/// # Errors
/// Reports unsupported pixel formats or shared spatial-filter failures.
pub fn scribble_edges(image: &ImageFrame) -> Result<ImageFrame, TransformError> {
    let image = rgb(image)?;
    let mut planes = Vec::with_capacity(3);
    for channel in 0..3 {
        let values = image
            .data()
            .chunks_exact(3)
            .map(|v| f32::from(v[channel]))
            .collect::<Vec<_>>();
        let blurred = gaussian_blur_f32(&values, size(&image), 1, 3.0, None)
            .map_err(|e| invalid(e.to_string()))?;
        let mut suppressed = vec![0.0f32; values.len()];
        for footprint in [
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
        ] {
            let maximum = dilate_f32(
                &blurred,
                size(&image),
                &footprint,
                ImageSize {
                    height: 3,
                    width: 3,
                },
            )
            .map_err(|e| invalid(e.to_string()))?;
            for ((out, &value), max) in suppressed.iter_mut().zip(&blurred).zip(maximum) {
                if value == max {
                    *out = value;
                }
            }
        }
        let mask = suppressed
            .into_iter()
            .map(|v| if v > 127.0 { 255u8 } else { 0 })
            .collect::<Vec<_>>();
        let blurred = gaussian_blur_u8(&mask, size(&image), 1, 3.0, None)
            .map_err(|e| invalid(e.to_string()))?;
        planes.push(
            blurred
                .into_iter()
                .map(|v| if v > 4 { 255 } else { 0 })
                .collect::<Vec<u8>>(),
        );
    }
    frame(
        size(&image),
        (0..image.width() * image.height())
            .flat_map(|i| [planes[0][i], planes[1][i], planes[2][i]])
            .collect(),
    )
}

/// NumPy default_rng's scalar-seeded PCG64 and 53-bit uniform conversion.
/// Seed expansion follows NumPy SeedSequence's four-word entropy pool. This
/// intentionally implements XSL-RR PCG64 rather than the distinct PCG64DXSM.
#[derive(Clone, Debug)]
pub struct NumpyPcg64 {
    state: u128,
    increment: u128,
}

impl NumpyPcg64 {
    /// Construct a reproducible stream from a nonnegative scalar seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let entropy = [seed as u32, (seed >> 32) as u32];
        let mut hash = 0x43b0d7e5u32;
        let hashmix = |value: u32, hash: &mut u32| {
            let value = value ^ *hash;
            *hash = hash.wrapping_mul(0x931e8875);
            let value = value.wrapping_mul(*hash);
            value ^ (value >> 16)
        };
        let mut pool = [0u32; 4];
        for (index, value) in pool.iter_mut().enumerate() {
            *value = hashmix(entropy.get(index).copied().unwrap_or(0), &mut hash);
        }
        for source in 0..4 {
            for destination in 0..4 {
                if source != destination {
                    let mixed = hashmix(pool[source], &mut hash);
                    let value = pool[destination]
                        .wrapping_mul(0xca01f9dd)
                        .wrapping_sub(mixed.wrapping_mul(0x4973f715));
                    pool[destination] = value ^ (value >> 16);
                }
            }
        }
        let mut generated = [0u32; 8];
        hash = 0x8b51f9dd;
        for (index, value) in generated.iter_mut().enumerate() {
            let data = pool[index % 4] ^ hash;
            hash = hash.wrapping_mul(0x58f38ded);
            let data = data.wrapping_mul(hash);
            *value = data ^ (data >> 16);
        }
        let word = |index: usize| {
            u64::from(generated[index * 2]) | (u64::from(generated[index * 2 + 1]) << 32)
        };
        let initial = (u128::from(word(0)) << 64) | u128::from(word(1));
        let sequence = (u128::from(word(2)) << 64) | u128::from(word(3));
        let mut random = Self {
            state: 0,
            increment: (sequence << 1) | 1,
        };
        random.step();
        random.state = random.state.wrapping_add(initial);
        random.step();
        random
    }

    fn step(&mut self) {
        const MULTIPLIER: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// Draw the next FP64 uniform value in [0,1), preserving NumPy's bit stream.
    pub fn uniform(&mut self) -> f64 {
        self.step();
        let value = ((self.state >> 64) as u64 ^ self.state as u64)
            .rotate_right((self.state >> 122) as u32);
        (value >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}
