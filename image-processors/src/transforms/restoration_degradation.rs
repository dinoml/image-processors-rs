//! Floating synthetic SISR preparation from KAIR main_test_srmd/utils_sisr.

use super::{crop_planar_f32, ImageCropBox, ImageSize, TransformError};

/// Failure preparing floating restoration pixels.
#[derive(Debug, thiserror::Error)]
pub enum DegradationError {
    /// Invalid image geometry or storage.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Shared resize failure.
    #[error(transparent)]
    Resize(#[from] image_resize_kernels::ResizeError),
    /// Kernel storage or coefficients are invalid.
    #[error("provide a finite kernel with positive dimensions and matching storage")]
    InvalidKernel,
    /// A noise standard deviation is not finite and nonnegative.
    #[error("noise standard deviation must be finite and nonnegative")]
    InvalidNoiseLevel,
}

/// Source-compatible blur/downsample/noise settings for one image.
#[derive(Clone, Copy, Debug)]
pub struct SrmdDegradationOptions<'a> {
    /// Positive integer downsampling factor; modcrop is performed first.
    pub scale: usize,
    /// Row-major convolution coefficients, without normalization or substitution.
    pub kernel: &'a [f64],
    /// Kernel height and width.
    pub kernel_size: ImageSize,
    /// Image-noise standard deviation in byte units, independent of model noise.
    pub noise_level: f64,
    /// NumPy RandomState's scalar uint32 seed. KAIR resets it to zero per image.
    pub seed: u32,
}

/// Modcrop, wrap convolution, KAIR antialiased bicubic and seeded image noise.
///
/// Input and output use planar unit-range FP32 pixels; output values may exceed
/// that range. The caller performs byte-to-FP32 conversion before this function
/// and converts to model precision only after it. Noise is sampled in source
/// HWC order and rounded back to FP32, matching NumPy's in-place addition.
///
/// # Errors
/// Reports invalid scale, geometry, storage, coefficients or noise settings.
pub fn prepare_srmd_f32_planes(
    values: &[f32],
    size: ImageSize,
    channels: usize,
    options: SrmdDegradationOptions<'_>,
) -> Result<(Vec<f32>, ImageSize), DegradationError> {
    if options.scale == 0 {
        return Err(TransformError::InvalidScaleFactor(options.scale).into());
    }
    let cropped = ImageSize::new(
        size.height - size.height % options.scale,
        size.width - size.width % options.scale,
    )?;
    let pixels = crop_planar_f32(
        values,
        size,
        channels,
        ImageCropBox {
            x_min: 0,
            y_min: 0,
            x_max: cropped.width,
            y_max: cropped.height,
        },
    )?;
    let blurred = convolve_wrap_f32_planes(
        &pixels,
        cropped,
        channels,
        options.kernel,
        options.kernel_size,
    )?;
    let mut output = image_resize_kernels::downsample_f32_kair_bicubic_planes(
        &blurred,
        image_resize_kernels::ImageSize::new(cropped.height, cropped.width)?,
        channels,
        options.scale,
    )?;
    let output_size = ImageSize::new(
        cropped.height / options.scale,
        cropped.width / options.scale,
    )?;
    add_numpy_legacy_gaussian_noise_planes(
        &mut output,
        output_size,
        channels,
        options.noise_level / 255.0,
        options.seed,
    )?;
    Ok((output, output_size))
}

/// SciPy ndimage.convolve with wrap boundaries, FP64 coefficients/accumulation
/// and FP32 storage. Coefficients retain convolution's reversal, including for
/// asymmetric kernels. Channels are independent and are never spatially mixed.
///
/// # Errors
/// Reports invalid geometry, storage or nonfinite kernel/input values.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "geometry is bounded for signed wrap indexing; SciPy stores the FP64 sum as FP32"
)]
pub fn convolve_wrap_f32_planes(
    values: &[f32],
    size: ImageSize,
    channels: usize,
    kernel: &[f64],
    kernel_size: ImageSize,
) -> Result<Vec<f32>, DegradationError> {
    let length = plane_len(size, channels)?;
    if values.len() != length {
        return Err(TransformError::InvalidBufferLength {
            expected: length,
            actual: values.len(),
        }
        .into());
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(TransformError::NonFiniteRestorationPixel { index }.into());
    }
    if kernel_size.height == 0
        || kernel_size.width == 0
        || kernel_size.height.checked_mul(kernel_size.width) != Some(kernel.len())
        || kernel.iter().any(|weight| !weight.is_finite())
    {
        return Err(DegradationError::InvalidKernel);
    }
    if [
        size.height,
        size.width,
        kernel_size.height,
        kernel_size.width,
    ]
    .iter()
    .any(|extent| *extent > (isize::MAX as usize) / 4)
    {
        return Err(TransformError::ImageSizeOverflow.into());
    }
    let pixels = size.height * size.width;
    let mut output = vec![0.0_f32; length];
    for channel in 0..channels {
        for y in 0..size.height {
            for x in 0..size.width {
                let mut sum = 0.0_f64;
                for ky in 0..kernel_size.height {
                    let sy = (y as isize + ky as isize - ((kernel_size.height - 1) / 2) as isize)
                        .rem_euclid(size.height as isize) as usize;
                    for kx in 0..kernel_size.width {
                        // SciPy reverses the contiguous coefficients before its
                        // correlation loop and omits taps at DBL_EPSILON or below.
                        let weight = kernel[(kernel_size.height - 1 - ky) * kernel_size.width
                            + kernel_size.width
                            - 1
                            - kx];
                        if weight.abs() <= f64::EPSILON {
                            continue;
                        }
                        let sx = (x as isize + kx as isize - ((kernel_size.width - 1) / 2) as isize)
                            .rem_euclid(size.width as isize)
                            as usize;
                        sum += f64::from(values[channel * pixels + sy * size.width + sx]) * weight;
                    }
                }
                output[channel * pixels + y * size.width + x] = sum as f32;
            }
        }
    }
    Ok(output)
}

/// Add NumPy legacy RandomState Gaussian noise to planar FP32 pixels.
///
/// Uses scalar-seeded MT19937, the 53-bit uniform mapping and cached polar
/// Gaussian algorithm in NumPy random/src/legacy/legacy-distributions.c.
/// Draw order is HWC even though storage is planar. Each call resets the stream;
/// zero deviation still consumes the same draws. There is no clipping.
///
/// # Errors
/// Reports invalid storage/geometry or nonfinite/negative standard deviation.
#[expect(
    clippy::cast_possible_truncation,
    reason = "NumPy's in-place sum rounds FP64 noise plus FP32 pixels to FP32"
)]
pub fn add_numpy_legacy_gaussian_noise_planes(
    values: &mut [f32],
    size: ImageSize,
    channels: usize,
    standard_deviation: f64,
    seed: u32,
) -> Result<(), DegradationError> {
    let length = plane_len(size, channels)?;
    if values.len() != length {
        return Err(TransformError::InvalidBufferLength {
            expected: length,
            actual: values.len(),
        }
        .into());
    }
    if !standard_deviation.is_finite() || standard_deviation < 0.0 {
        return Err(DegradationError::InvalidNoiseLevel);
    }
    let mut random = NumpyLegacyNormal::new(seed);
    let pixels = size.height * size.width;
    for pixel in 0..pixels {
        for channel in 0..channels {
            let value = &mut values[channel * pixels + pixel];
            *value = (f64::from(*value) + standard_deviation * random.normal()) as f32;
        }
    }
    Ok(())
}

fn plane_len(size: ImageSize, channels: usize) -> Result<usize, TransformError> {
    ImageSize::new(size.height, size.width)?;
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }
    size.height
        .checked_mul(size.width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or(TransformError::ImageSizeOverflow)
}

struct NumpyLegacyNormal {
    key: [u32; 624],
    position: usize,
    cached: Option<f64>,
}

impl NumpyLegacyNormal {
    fn new(seed: u32) -> Self {
        let mut key = [0_u32; 624];
        key[0] = seed;
        for index in 1..624_u32 {
            let previous = key[index as usize - 1];
            key[index as usize] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index);
        }
        Self {
            key,
            position: 624,
            cached: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.position == 624 {
            for index in 0..624 {
                let combined =
                    (self.key[index] & 0x8000_0000) | (self.key[(index + 1) % 624] & 0x7fff_ffff);
                self.key[index] = self.key[(index + 397) % 624]
                    ^ (combined >> 1)
                    ^ if combined & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.position = 0;
        }
        let mut value = self.key[self.position];
        self.position += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn uniform(&mut self) -> f64 {
        let high = self.next_u32() >> 5;
        let low = self.next_u32() >> 6;
        (f64::from(high) * 67_108_864.0 + f64::from(low)) / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.cached.take() {
            return value;
        }
        loop {
            let first = 2.0 * self.uniform() - 1.0;
            let second = 2.0 * self.uniform() - 1.0;
            let radius = first * first + second * second;
            if radius >= 1.0 || radius == 0.0 {
                continue;
            }
            let factor = (-2.0 * radius.ln() / radius).sqrt();
            self.cached = Some(factor * first);
            return factor * second;
        }
    }
}
