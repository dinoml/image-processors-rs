//! Lossless integer restoration images and floating pixel preparation.

use super::{grayscale_opencv, new_frame, ImageSize, TransformError};
use crate::media::{FrameTiming, ImageFrame, PixelFormat};

/// Decode-compatible byte grayscale normalization, dividing in FP64 before FP32.
///
/// Already-decoded luma is preserved; RGB input uses the shared integer luma
/// conversion. The division boundary follows NumPy `np.float32(img / 255.)`.
///
/// # Errors
/// Reports invalid decoded image storage or grayscale conversion failures.
#[expect(
    clippy::cast_possible_truncation,
    reason = "upstream explicitly converts the FP64 quotient to FP32"
)]
pub fn prepare_grayscale_u8(image: &ImageFrame) -> Result<Vec<f32>, TransformError> {
    let gray = grayscale_opencv(image)?;
    Ok(gray
        .data()
        .iter()
        .map(|value| (f64::from(*value) / 255.0) as f32)
        .collect())
}

/// Clamp unit-range grayscale predictions and quantize using ties-to-even rounding.
///
/// # Errors
/// Rejects incompatible image storage, nonfinite predictions or invalid dimensions.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite clamped predictions round to the byte interval"
)]
pub fn finish_grayscale_u8(
    values: &[f32],
    size: ImageSize,
    timing: FrameTiming,
) -> Result<ImageFrame, TransformError> {
    let expected = restoration_image_len(size.height, size.width, 1)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(TransformError::NonFiniteRestorationPixel { index });
    }
    let data = values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
        .collect();
    new_frame(size.width, size.height, PixelFormat::Luma8, data, timing)
}

/// Interleaved decoded pixel order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestorationPixelFormat {
    /// One luminance channel.
    Gray,
    /// Red, green, blue.
    Rgb,
    /// Blue, green, red.
    Bgr,
    /// Red, green, blue, alpha.
    Rgba,
    /// Blue, green, red, alpha.
    Bgra,
}
impl RestorationPixelFormat {
    /// Number of channels in one pixel.
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }
}

/// Decoded integer storage, without reducing sixteen-bit precision.
#[derive(Clone, Debug)]
pub enum RestorationPixels {
    /// Unsigned byte pixels.
    U8(Vec<u8>),
    /// Unsigned sixteen-bit pixels.
    U16(Vec<u16>),
}
impl RestorationPixels {
    /// Stored component count.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
        }
    }
    /// Whether storage contains no components.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn value(&self, index: usize) -> f32 {
        match self {
            Self::U8(values) => f32::from(values[index]),
            Self::U16(values) => f32::from(values[index]),
        }
    }
}

/// Owned integer restoration image.
#[derive(Clone, Debug)]
pub struct RestorationImage {
    /// Pixel width.
    pub width: usize,
    /// Pixel height.
    pub height: usize,
    /// Interleaved channel convention.
    pub format: RestorationPixelFormat,
    /// Integer samples.
    pub data: RestorationPixels,
}
impl RestorationImage {
    /// Borrow image storage without copying samples.
    pub const fn view(&self) -> RestorationImageView<'_> {
        RestorationImageView {
            width: self.width,
            height: self.height,
            format: self.format,
            data: &self.data,
        }
    }
}

/// Borrowed decoded restoration image.
#[derive(Clone, Copy, Debug)]
pub struct RestorationImageView<'a> {
    /// Pixel width.
    pub width: usize,
    /// Pixel height.
    pub height: usize,
    /// Interleaved channel convention.
    pub format: RestorationPixelFormat,
    /// Integer samples.
    pub data: &'a RestorationPixels,
}

/// Normalized RGB planes and a separately preserved alpha plane.
pub struct PreparedRestorationPixels {
    /// Three RGB planes, each height times width.
    pub rgb: Vec<f32>,
    /// Normalized alpha, when supplied.
    pub alpha: Option<Vec<f32>>,
    /// Source range selected from the actual pixel maximum.
    pub range: f32,
}

/// Checked interleaved/planar image storage length.
/// # Errors
/// Rejects empty dimensions and extent overflow.
pub fn restoration_image_len(
    height: usize,
    width: usize,
    channels: usize,
) -> Result<usize, TransformError> {
    if height == 0 || width == 0 || channels == 0 {
        return Err(TransformError::InvalidSize { height, width });
    }
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(TransformError::ImageSizeOverflow)
}

/// Prepare normalized RGB and alpha, preserving the RealESRGAN range heuristic.
/// # Errors
/// Rejects invalid storage or overflowing geometry.
pub fn prepare_restoration_pixels(
    image: RestorationImageView<'_>,
) -> Result<PreparedRestorationPixels, TransformError> {
    let channels = image.format.channels();
    let pixels = restoration_image_len(image.height, image.width, 1)?;
    let expected = restoration_image_len(image.height, image.width, channels)?;
    if image.data.len() != expected {
        return Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: image.data.len(),
        });
    }
    let range = restoration_source_range(image);
    let mut rgb = vec![0.0; restoration_image_len(image.height, image.width, 3)?];
    let mut alpha = if channels == 4 {
        Some(vec![0.0; pixels])
    } else {
        None
    };
    for index in 0..pixels {
        for channel in 0..3 {
            let source_channel = match image.format {
                RestorationPixelFormat::Gray => 0,
                RestorationPixelFormat::Bgr | RestorationPixelFormat::Bgra => 2 - channel,
                _ => channel,
            };
            rgb[channel * pixels + index] =
                image.data.value(index * channels + source_channel) / range;
        }
        if let Some(alpha) = &mut alpha {
            alpha[index] = image.data.value(index * channels + 3) / range;
        }
    }
    Ok(PreparedRestorationPixels { rgb, alpha, range })
}

/// Clamp RGB planes and convert with explicit floating grayscale coefficients.
/// # Errors
/// Rejects storage that does not contain three equal planes.
pub fn restoration_grayscale(
    values: &[f32],
    coefficients: [f32; 3],
) -> Result<Vec<f32>, TransformError> {
    if !values.len().is_multiple_of(3) {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 3,
            actual: values.len(),
        });
    }
    let pixels = values.len() / 3;
    Ok((0..pixels)
        .map(|index| {
            values[index].clamp(0.0, 1.0) * coefficients[0]
                + values[pixels + index].clamp(0.0, 1.0) * coefficients[1]
                + values[2 * pixels + index].clamp(0.0, 1.0) * coefficients[2]
        })
        .collect())
}

/// Assemble restored RGB/alpha in source order and round once to source range.
/// # Errors
/// Rejects invalid planes or overflowing dimensions.
pub fn finish_restoration_pixels(
    rgb: &[f32],
    alpha: Option<&[f32]>,
    height: usize,
    width: usize,
    format: RestorationPixelFormat,
    range: f32,
    gray_coefficients: [f32; 3],
) -> Result<RestorationImage, TransformError> {
    let pixels = restoration_image_len(height, width, 1)?;
    if rgb.len() != restoration_image_len(height, width, 3)?
        || alpha.is_some_and(|values| values.len() != pixels)
        || (format.channels() == 4) != alpha.is_some()
        || (range != 255.0 && range != 65535.0)
    {
        return Err(TransformError::InvalidChannelDataLength {
            channels: format.channels(),
            actual: rgb.len(),
        });
    }
    let gray = if format == RestorationPixelFormat::Gray {
        Some(restoration_grayscale(rgb, gray_coefficients)?)
    } else {
        None
    };
    if let Some(index) = rgb
        .iter()
        .chain(alpha.into_iter().flatten())
        .position(|value| !value.is_finite())
    {
        return Err(TransformError::NonFiniteRestorationPixel { index });
    }
    let mut values = Vec::with_capacity(restoration_image_len(height, width, format.channels())?);
    for index in 0..pixels {
        if let Some(gray) = &gray {
            values.push(gray[index]);
        } else {
            for channel in 0..3 {
                let channel = if matches!(
                    format,
                    RestorationPixelFormat::Bgr | RestorationPixelFormat::Bgra
                ) {
                    2 - channel
                } else {
                    channel
                };
                values.push(rgb[channel * pixels + index].clamp(0.0, 1.0));
            }
        }
        if let Some(alpha) = alpha {
            values.push(alpha[index]);
        }
    }
    let data = if range == 65535.0 {
        RestorationPixels::U16(
            values
                .into_iter()
                .map(|value| (value * range).round_ties_even() as i64 as u16)
                .collect(),
        )
    } else {
        RestorationPixels::U8(
            values
                .into_iter()
                .map(|value| (value * range).round_ties_even() as i64 as u8)
                .collect(),
        )
    };
    Ok(RestorationImage {
        height,
        width,
        format,
        data,
    })
}

/// Convert source channel order to RGB, retaining integer sample precision.
/// # Errors
/// Rejects invalid pixel storage or extent overflow.
pub fn restoration_to_rgb(
    image: RestorationImageView<'_>,
) -> Result<RestorationImage, TransformError> {
    let channels = image.format.channels();
    if image.data.len() != restoration_image_len(image.height, image.width, channels)? {
        return Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: image.data.len(),
        });
    }
    let count = restoration_image_len(image.height, image.width, 3)?;
    let index = |component: usize| {
        let channel = if channels == 1 {
            0
        } else if matches!(
            image.format,
            RestorationPixelFormat::Bgr | RestorationPixelFormat::Bgra
        ) {
            2 - component % 3
        } else {
            component % 3
        };
        component / 3 * channels + channel
    };
    let data = match image.data {
        RestorationPixels::U8(values) => RestorationPixels::U8(
            (0..count)
                .map(|component| values[index(component)])
                .collect(),
        ),
        RestorationPixels::U16(values) => RestorationPixels::U16(
            (0..count)
                .map(|component| values[index(component)])
                .collect(),
        ),
    };
    Ok(RestorationImage {
        width: image.width,
        height: image.height,
        format: RestorationPixelFormat::Rgb,
        data,
    })
}

/// Source face-helper RGB values, including its FP64 sixteen-bit range scaling.
/// # Errors
/// Rejects malformed storage.
pub fn restoration_rgb_byte_values(
    image: RestorationImageView<'_>,
) -> Result<Vec<f64>, TransformError> {
    let high = restoration_source_range(image) == 65535.0;
    let image = restoration_to_rgb(image)?;
    Ok((0..image.data.len())
        .map(|index| {
            let value = f64::from(image.data.value(index));
            if high {
                value / 65535.0 * 255.0
            } else {
                value
            }
        })
        .collect())
}

/// Source range selected before dropping alpha or changing channel order.
pub fn restoration_source_range(image: RestorationImageView<'_>) -> f32 {
    if (0..image.data.len()).any(|index| image.data.value(index) > 256.0) {
        65535.0
    } else {
        255.0
    }
}

/// Normalize interleaved RGB samples into selected-order CHW floating features.
///
/// Division uses FP64, followed by FP32 conversion and FP32 normalization.
/// Use [`normalize_rgb_f64_with_options`] to keep normalization in FP64 until
/// the final output conversion, as NumPy preprocessing does.
/// # Errors
/// Rejects invalid extents, storage or zero standard deviations.
pub fn normalize_rgb_f64(
    values: &[f64],
    height: usize,
    width: usize,
    divisor: f64,
    mean: [f32; 3],
    std: [f32; 3],
    bgr: bool,
) -> Result<Vec<f32>, TransformError> {
    normalize_rgb_f64_with_options(
        values,
        height,
        width,
        divisor,
        RgbNormalization::Float32 { mean, std },
        bgr,
    )
}

/// Arithmetic boundary for channel normalization of FP64 RGB samples.
///
/// Both modes return FP32 CHW features. Channel statistics retain the selected
/// arithmetic precision, rather than being rounded through another mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RgbNormalization {
    /// Divide in FP64, narrow to FP32, then normalize in FP32.
    Float32 {
        /// Per-channel means in output RGB/BGR order.
        mean: [f32; 3],
        /// Per-channel standard deviations in output RGB/BGR order.
        std: [f32; 3],
    },
    /// Divide and normalize in FP64, narrowing only the completed feature.
    Float64 {
        /// Per-channel means in output RGB/BGR order.
        mean: [f64; 3],
        /// Per-channel standard deviations in output RGB/BGR order.
        std: [f64; 3],
    },
}

/// Normalize interleaved FP64 RGB samples with an explicit arithmetic boundary.
///
/// The operation order is division, mean subtraction, standard-deviation
/// division, and CHW packing. `bgr` selects reversed source channel order.
/// No intermediate pixel quantization or additional rescaling is performed.
///
/// # Errors
/// Rejects invalid extents, storage, divisor or zero standard deviations.
pub fn normalize_rgb_f64_with_options(
    values: &[f64],
    height: usize,
    width: usize,
    divisor: f64,
    normalization: RgbNormalization,
    bgr: bool,
) -> Result<Vec<f32>, TransformError> {
    let length = restoration_image_len(height, width, 3)?;
    let zero_std = match normalization {
        RgbNormalization::Float32 { std, .. } => std.contains(&0.0),
        RgbNormalization::Float64 { std, .. } => std.contains(&0.0),
    };
    if values.len() != length || zero_std || !divisor.is_finite() || divisor == 0.0 {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 3,
            actual: values.len(),
        });
    }
    let pixels = length / 3;
    let mut result = vec![0.0; length];
    for channel in 0..3 {
        for index in 0..pixels {
            let value = values[index * 3 + if bgr { 2 - channel } else { channel }] / divisor;
            result[channel * pixels + index] = match normalization {
                RgbNormalization::Float32 { mean, std } => {
                    // img2tensor narrows before tensor normalization.
                    (value as f32 - mean[channel]) / std[channel]
                }
                RgbNormalization::Float64 { mean, std } => {
                    // NumPy retains the FP64 array until constructing its tensor.
                    ((value - mean[channel]) / std[channel]) as f32
                }
            };
        }
    }
    Ok(result)
}

/// Convert signed RGB graph output to a rounded byte RGB image.
/// # Errors
/// Rejects invalid planes or nonfinite values.
pub fn finish_signed_restoration_rgb(
    values: &[f32],
    height: usize,
    width: usize,
) -> Result<RestorationImage, TransformError> {
    let unit: Vec<f32> = values
        .iter()
        .map(|value| (value.clamp(-1.0, 1.0) + 1.0) / 2.0)
        .collect();
    finish_restoration_pixels(
        &unit,
        None,
        height,
        width,
        RestorationPixelFormat::Rgb,
        255.0,
        [0.299, 0.587, 0.114],
    )
}

/// Composite a byte RGB face with FP64 alpha while retaining background alpha/depth.
/// # Errors
/// Rejects mismatched geometry, storage or nonfinite mask values.
pub struct RestorationComposite {
    image: RestorationImage,
    values: Vec<f64>,
}
impl RestorationComposite {
    /// Retain floating precision across all faces; quantization happens once.
    /// # Errors
    /// Rejects invalid source storage.
    pub fn new(mut image: RestorationImage) -> Result<Self, TransformError> {
        if image.format == RestorationPixelFormat::Gray {
            image = restoration_to_rgb(image.view())?;
        }
        if image.data.len()
            != restoration_image_len(image.height, image.width, image.format.channels())?
        {
            return Err(TransformError::InvalidChannelDataLength {
                channels: image.format.channels(),
                actual: image.data.len(),
            });
        }
        let values = (0..image.data.len())
            .map(|index| f64::from(image.data.value(index)))
            .collect();
        Ok(Self { image, values })
    }
    /// Composite a byte face using a full-size FP64 mask, preserving background alpha.
    /// # Errors
    /// Rejects malformed face/mask storage or nonfinite masks.
    pub fn add_face(&mut self, face: &[u8], mask: &[f64]) -> Result<(), TransformError> {
        let pixels = restoration_image_len(self.image.height, self.image.width, 1)?;
        if face.len() != restoration_image_len(self.image.height, self.image.width, 3)?
            || mask.len() != pixels
        {
            return Err(TransformError::InvalidChannelDataLength {
                channels: 3,
                actual: face.len(),
            });
        }
        if let Some(index) = mask.iter().position(|value| !value.is_finite()) {
            return Err(TransformError::NonFiniteRestorationPixel { index });
        }
        let channels = self.image.format.channels();
        for index in 0..pixels {
            for channel in 0..3 {
                let source = if matches!(
                    self.image.format,
                    RestorationPixelFormat::Bgr | RestorationPixelFormat::Bgra
                ) {
                    2 - channel
                } else {
                    channel
                };
                let offset = index * channels + channel;
                self.values[offset] = mask[index] * f64::from(face[index * 3 + source])
                    + (1.0 - mask[index]) * self.values[offset];
            }
        }
        Ok(())
    }
    /// Select depth from the final values and truncate once as FaceRestoreHelper does.
    pub fn finish(mut self) -> RestorationImage {
        self.image.data = if self.values.iter().any(|value| *value > 256.0) {
            RestorationPixels::U16(
                self.values
                    .into_iter()
                    .map(|value| value as i64 as u16)
                    .collect(),
            )
        } else {
            RestorationPixels::U8(
                self.values
                    .into_iter()
                    .map(|value| value as i64 as u8)
                    .collect(),
            )
        };
        self.image
    }
}
