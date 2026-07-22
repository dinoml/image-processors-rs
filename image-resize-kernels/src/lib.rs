//! Resize kernels for processor parity and performance.
//!
//! The crate owns resize backend choices behind small Rust types so model
//! processors can request a profile without depending on third-party backend
//! semantics directly.

use half::f16;
use thiserror::Error;

/// Image dimensions in height-width order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageSize {
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
}

impl ImageSize {
    /// Creates an image size with positive dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    pub fn new(height: usize, width: usize) -> Result<Self, ResizeError> {
        if height == 0 || width == 0 {
            return Err(ResizeError::InvalidSize { height, width });
        }
        Ok(Self { height, width })
    }
}

/// Crop geometry within a logical resized image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizeCrop {
    resized: ImageSize,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
}

impl ResizeCrop {
    /// Creates crop geometry for a window inside a logical resized image.
    ///
    /// # Errors
    ///
    /// Returns an error when the crop window exceeds the resized image.
    pub fn new(
        resized: ImageSize,
        origin_x: usize,
        origin_y: usize,
        target: ImageSize,
    ) -> Result<Self, ResizeError> {
        validate_crop_window(resized, origin_x, origin_y, target)?;
        Ok(Self {
            resized,
            origin_x,
            origin_y,
            target,
        })
    }

    /// Returns the logical full resized image size.
    pub fn resized(self) -> ImageSize {
        self.resized
    }

    /// Returns the crop origin x coordinate.
    pub fn origin_x(self) -> usize {
        self.origin_x
    }

    /// Returns the crop origin y coordinate.
    pub fn origin_y(self) -> usize {
        self.origin_y
    }

    /// Returns the crop target size.
    pub fn target(self) -> ImageSize {
        self.target
    }
}

/// Resize resampling filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResizeFilter {
    /// Nearest-neighbor filtering.
    Nearest,
    /// Bilinear filtering.
    Bilinear,
    /// Bicubic filtering.
    Bicubic,
    /// Lanczos filtering.
    Lanczos,
}

/// Resize implementation profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResizeProfile {
    /// Fast SIMD-capable resize profile.
    Fast,
    /// Pillow-compatible resize profile used by Transformers image processors.
    ///
    /// The profile uses Pillow's fixed-point coefficients and rounds to `u8`
    /// after each separable pass.
    Pillow,
    /// Torchvision tensor-resize profile used by canonical Transformers processors.
    ///
    /// This profile quantizes normalized filter weights at Torchvision's dynamic
    /// signed-16-bit precision and rounds each separable `u8` pass.
    Torchvision,
}

/// Output layout for direct resized `f32` image writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32ImageLayout {
    /// Channels, height, width order.
    Chw,
    /// Height, width, channels order.
    Hwc,
}

/// Reusable state for repeated resize calls.
///
/// The workspace preserves allocation capacity and cached coefficient tables
/// between calls. It is intended for serial same-shape batch processing; callers
/// doing parallel processing should give each worker its own workspace.
#[derive(Clone, Debug, Default)]
pub struct ResizeWorkspace {
    horizontal: Option<CachedCoefficients>,
    vertical: Option<CachedCoefficients>,
    u8_scratch: Vec<u8>,
}

impl ResizeWorkspace {
    /// Creates an empty resize workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears cached resize state while keeping the workspace allocation alive.
    pub fn clear(&mut self) {
        self.horizontal = None;
        self.vertical = None;
        self.u8_scratch.clear();
    }

    /// Returns the number of cached coefficient tables.
    pub fn coefficient_cache_len(&self) -> usize {
        usize::from(self.horizontal.is_some()) + usize::from(self.vertical.is_some())
    }

    /// Returns the reusable `u8` scratch buffer capacity.
    pub fn u8_scratch_capacity(&self) -> usize {
        self.u8_scratch.capacity()
    }

    fn crop_parts(
        &mut self,
        horizontal: CoefficientsKey,
        vertical: CoefficientsKey,
    ) -> Result<ResizeWorkspaceParts<'_>, ResizeError> {
        Self::ensure_coefficients(&mut self.horizontal, horizontal)?;
        Self::ensure_coefficients(&mut self.vertical, vertical)?;

        let Self {
            horizontal,
            vertical,
            u8_scratch,
        } = self;
        let Some(horizontal) = horizontal.as_ref() else {
            return Err(ResizeError::WorkspaceInvariant);
        };
        let Some(vertical) = vertical.as_ref() else {
            return Err(ResizeError::WorkspaceInvariant);
        };

        Ok(ResizeWorkspaceParts {
            horizontal: &horizontal.coefficients,
            vertical: &vertical.coefficients,
            u8_scratch,
        })
    }

    fn ensure_coefficients(
        slot: &mut Option<CachedCoefficients>,
        key: CoefficientsKey,
    ) -> Result<(), ResizeError> {
        if slot.as_ref().is_some_and(|cached| cached.key == key) {
            return Ok(());
        }

        *slot = Some(CachedCoefficients {
            key,
            coefficients: PillowFixedCoefficients::new_window(
                key.input,
                key.output,
                key.output_start,
                key.output_len,
                key.filter,
            )?,
        });
        Ok(())
    }
}

struct ResizeWorkspaceParts<'a> {
    horizontal: &'a PillowFixedCoefficients,
    vertical: &'a PillowFixedCoefficients,
    u8_scratch: &'a mut Vec<u8>,
}

/// Resizes borrowed interleaved `u8` image data.
///
/// Supported channel counts are 1, 3, and 4.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, channel count, or
/// selected backend operation is invalid.
pub fn resize_u8(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
    profile: ResizeProfile,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    match profile {
        ResizeProfile::Fast => resize_u8_fast(values, source, channels, target, filter),
        ResizeProfile::Pillow => resize_u8_pillow(values, source, channels, target, filter),
        ResizeProfile::Torchvision => {
            resize_u8_torchvision(values, source, channels, target, filter)
        }
    }
}

/// Rescales and resizes interleaved `u8` image data as Torchvision `f32` tensors.
///
/// Multiplication by `scale` happens before interpolation, preserving the
/// rescale-before-resize ordering used by DINOv3 image processors.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, channel count, scale,
/// or resize operation is invalid.
pub fn resize_u8_to_f32_torchvision(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
    scale: f32,
) -> Result<Vec<f32>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    if !scale.is_finite() {
        return Err(ResizeError::InvalidScaleFactor(scale));
    }
    let values = values
        .iter()
        .map(|&value| f32::from(value) * scale)
        .collect::<Vec<_>>();
    resize_f32_torchvision_interleaved(&values, source, channels, target, filter)
}

/// Resizes interleaved finite `f32` image values with Torchvision semantics.
///
/// # Errors
///
/// Returns an error when the value count, dimensions, channel count, input
/// finiteness, or resize operation is invalid.
pub fn resize_f32_torchvision(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<f32>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ResizeError::NonFiniteValue { index });
    }
    resize_f32_torchvision_interleaved(values, source, channels, target, filter)
}

/// Resizes owned interleaved `u8` image data.
///
/// Supported channel counts are 1, 3, and 4.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, channel count, or
/// selected backend operation is invalid.
pub fn resize_u8_owned(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
    profile: ResizeProfile,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    match profile {
        ResizeProfile::Fast => resize_u8_fast_owned(values, source, channels, target, filter),
        ResizeProfile::Pillow => resize_u8_pillow(&values, source, channels, target, filter),
        ResizeProfile::Torchvision => {
            resize_u8_torchvision(&values, source, channels, target, filter)
        }
    }
}

/// Resizes borrowed interleaved `u8` image data and returns a crop window from
/// the resized image.
///
/// `crop` describes the logical full resize size and the target window inside
/// that resized image.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, crop window, channel
/// count, or selected backend operation is invalid.
pub fn resize_u8_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    match profile {
        ResizeProfile::Fast => resize_u8_full_then_crop(values, source, channels, crop, filter),
        ResizeProfile::Pillow => resize_u8_pillow_crop(values, source, channels, crop, filter),
        ResizeProfile::Torchvision => {
            resize_u8_torchvision_crop(values, source, channels, crop, filter)
        }
    }
}

/// Resizes owned interleaved `u8` image data and returns a crop window from the
/// resized image.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, crop window, channel
/// count, or selected backend operation is invalid.
pub fn resize_u8_crop_owned(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    match profile {
        ResizeProfile::Fast => {
            let resized_values =
                resize_u8_fast_owned(values, source, channels, crop.resized, filter)?;
            crop_image_data(
                &resized_values,
                crop.resized,
                channels,
                crop.origin_x,
                crop.origin_y,
                crop.target,
            )
        }
        ResizeProfile::Pillow => resize_u8_pillow_crop(&values, source, channels, crop, filter),
        ResizeProfile::Torchvision => {
            resize_u8_torchvision_crop(&values, source, channels, crop, filter)
        }
    }
}

/// Resizes borrowed interleaved `u8` image data and reuses cached workspace
/// state where the selected profile supports it.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, crop window, channel
/// count, or selected backend operation is invalid.
pub fn resize_u8_crop_with_workspace(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
    workspace: &mut ResizeWorkspace,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    match profile {
        ResizeProfile::Fast => resize_u8_full_then_crop(values, source, channels, crop, filter),
        ResizeProfile::Pillow => {
            resize_u8_pillow_crop_with_workspace(values, source, channels, crop, filter, workspace)
        }
        ResizeProfile::Torchvision => {
            resize_u8_torchvision_crop(values, source, channels, crop, filter)
        }
    }
}

/// Resizes owned interleaved `u8` image data and reuses cached workspace state
/// where the selected profile supports it.
///
/// # Errors
///
/// Returns an error when the buffer length, dimensions, crop window, channel
/// count, or selected backend operation is invalid.
pub fn resize_u8_crop_owned_with_workspace(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
    workspace: &mut ResizeWorkspace,
) -> Result<Vec<u8>, ResizeError> {
    resize_u8_crop_with_workspace(&values, source, channels, crop, filter, profile, workspace)
}

/// Resizes borrowed interleaved `u8` image data, returns a crop window from the
/// resized image, and writes transformed `f32` values directly into `output`.
///
/// This path preserves the Pillow-compatible intermediate `u8` rounding before
/// applying `value * multiplier[channel] + offset[channel]`.
///
/// # Errors
///
/// Returns an error when the input buffer, output buffer, crop, channel count,
/// transform stats, or selected profile is invalid.
#[expect(
    clippy::too_many_arguments,
    reason = "low-level resize primitive keeps call-site state explicit"
)]
pub fn resize_u8_crop_into_f32_with_workspace(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
    output: &mut [f32],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
    workspace: &mut ResizeWorkspace,
) -> Result<(), ResizeError> {
    resize_u8_crop_into_transformed_with_workspace(
        values, source, channels, crop, filter, profile, output, layout, multiplier, offset,
        workspace,
    )
}

/// Resizes borrowed interleaved `u8` image data, returns a crop window from the
/// resized image, and writes transformed `f16` values directly into `output`.
///
/// The transform is evaluated in `f32` before each value is rounded to `f16`.
/// Pillow-compatible intermediate `u8` rounding is unchanged.
///
/// # Errors
///
/// Returns an error when the input buffer, output buffer, crop, channel count,
/// transform stats, or selected profile is invalid.
#[expect(
    clippy::too_many_arguments,
    reason = "low-level resize primitive keeps call-site state explicit"
)]
pub fn resize_u8_crop_into_f16_with_workspace(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
    output: &mut [f16],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
    workspace: &mut ResizeWorkspace,
) -> Result<(), ResizeError> {
    resize_u8_crop_into_transformed_with_workspace(
        values, source, channels, crop, filter, profile, output, layout, multiplier, offset,
        workspace,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "low-level resize primitive keeps call-site state explicit"
)]
fn resize_u8_crop_into_transformed_with_workspace<T: TransformedOutput>(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    profile: ResizeProfile,
    output: &mut [T],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
    workspace: &mut ResizeWorkspace,
) -> Result<(), ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    validate_f32_output(output.len(), crop.target, channels, multiplier, offset)?;
    match profile {
        ResizeProfile::Pillow => resize_u8_pillow_crop_into_with_workspace(
            values, source, channels, crop, filter, output, layout, multiplier, offset, workspace,
        ),
        ResizeProfile::Fast => {
            let resized = resize_u8_full_then_crop(values, source, channels, crop, filter)?;
            write_interleaved_u8_to_output(
                &resized,
                crop.target,
                channels,
                output,
                layout,
                multiplier,
                offset,
            )
        }
        ResizeProfile::Torchvision => {
            let resized = resize_u8_torchvision_crop(values, source, channels, crop, filter)?;
            write_interleaved_u8_to_output(
                &resized,
                crop.target,
                channels,
                output,
                layout,
                multiplier,
                offset,
            )
        }
    }
}

fn resize_u8_fast(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    let source_width = usize_to_u32(source.width, "source width")?;
    let source_height = usize_to_u32(source.height, "source height")?;
    let pixel_type = fast_resize_pixel_type(channels)?;
    let source_image =
        fast_image_resize::images::ImageRef::new(source_width, source_height, values, pixel_type)
            .map_err(|error| backend_error(ResizeProfile::Fast, error))?;
    resize_fast_image(&source_image, target, pixel_type, filter)
}

fn resize_u8_fast_owned(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    let source_width = usize_to_u32(source.width, "source width")?;
    let source_height = usize_to_u32(source.height, "source height")?;
    let pixel_type = fast_resize_pixel_type(channels)?;
    let source_image = fast_image_resize::images::Image::from_vec_u8(
        source_width,
        source_height,
        values,
        pixel_type,
    )
    .map_err(|error| backend_error(ResizeProfile::Fast, error))?;
    resize_fast_image(&source_image, target, pixel_type, filter)
}

fn resize_fast_image(
    source_image: &impl fast_image_resize::IntoImageView,
    target: ImageSize,
    pixel_type: fast_image_resize::PixelType,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    let target_width = usize_to_u32(target.width, "target width")?;
    let target_height = usize_to_u32(target.height, "target height")?;
    let mut destination =
        fast_image_resize::images::Image::new(target_width, target_height, pixel_type);
    let options = fast_image_resize::ResizeOptions::new()
        .resize_alg(fast_resize_algorithm(filter))
        .use_alpha(false);
    let mut resizer = fast_image_resize::Resizer::new();
    resizer
        .resize(source_image, &mut destination, Some(&options))
        .map_err(|error| backend_error(ResizeProfile::Fast, error))?;
    Ok(destination.into_vec())
}

fn resize_u8_full_then_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    let resized_values = resize_u8_fast(values, source, channels, crop.resized, filter)?;
    crop_image_data(
        &resized_values,
        crop.resized,
        channels,
        crop.origin_x,
        crop.origin_y,
        crop.target,
    )
}

fn resize_u8_torchvision(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    if source == target {
        return Ok(values.to_vec());
    }
    if filter == ResizeFilter::Nearest {
        return resize_u8_torchvision_nearest(values, source, channels, target);
    }
    resize_u8_torchvision_fixed(values, source, channels, target, filter)
}

fn resize_u8_torchvision_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    let resized = resize_u8_torchvision(values, source, channels, crop.resized, filter)?;
    crop_image_data(
        &resized,
        crop.resized,
        channels,
        crop.origin_x,
        crop.origin_y,
        crop.target,
    )
}

fn resize_f32_torchvision_interleaved(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<f32>, ResizeError> {
    if source == target {
        return Ok(values.to_vec());
    }
    if filter == ResizeFilter::Nearest {
        let mut output = vec![0.0_f32; image_len(target, channels)?];
        for y in 0..target.height {
            let source_y = torchvision_nearest_exact_index(y, source.height, target.height);
            for x in 0..target.width {
                let source_x = torchvision_nearest_exact_index(x, source.width, target.width);
                let source_start = (source_y * source.width + source_x) * channels;
                let target_start = (y * target.width + x) * channels;
                output[target_start..target_start + channels]
                    .copy_from_slice(&values[source_start..source_start + channels]);
            }
        }
        return Ok(output);
    }

    let horizontal = Coefficients::new(source.width, target.width, filter)?;
    let vertical = Coefficients::new(source.height, target.height, filter)?;
    let output_len = image_len(target, channels)?;
    let intermediate_len = source
        .height
        .checked_mul(target.width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let mut intermediate = vec![0.0_f32; intermediate_len];
    let mut output = vec![0.0_f32; output_len];

    for y in 0..source.height {
        for x in 0..target.width {
            let weights = horizontal.weights(x);
            let start = horizontal.start(x);
            for channel in 0..channels {
                let mut value = 0.0_f32;
                for (offset, &weight) in weights.iter().enumerate() {
                    let source_index = (y * source.width + start + offset) * channels + channel;
                    value += values[source_index] * weight;
                }
                intermediate[(y * target.width + x) * channels + channel] = value;
            }
        }
    }

    for y in 0..target.height {
        let weights = vertical.weights(y);
        let start = vertical.start(y);
        for x in 0..target.width {
            for channel in 0..channels {
                let mut value = 0.0_f32;
                for (offset, &weight) in weights.iter().enumerate() {
                    let intermediate_index =
                        ((start + offset) * target.width + x) * channels + channel;
                    value += intermediate[intermediate_index] * weight;
                }
                output[(y * target.width + x) * channels + channel] = value;
            }
        }
    }

    Ok(output)
}

fn resize_u8_torchvision_nearest(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<u8>, ResizeError> {
    let target_len = image_len(target, channels)?;
    let mut output = vec![0; target_len];
    for y in 0..target.height {
        let source_y = torchvision_nearest_exact_index(y, source.height, target.height);
        for x in 0..target.width {
            let source_x = torchvision_nearest_exact_index(x, source.width, target.width);
            let source_start = (source_y * source.width + source_x) * channels;
            let target_start = (y * target.width + x) * channels;
            output[target_start..target_start + channels]
                .copy_from_slice(&values[source_start..source_start + channels]);
        }
    }
    Ok(output)
}

fn torchvision_nearest_exact_index(output: usize, input_size: usize, output_size: usize) -> usize {
    let scale = input_size as f32 / output_size as f32;
    (((output as f32 + 0.5) * scale).floor() as usize).min(input_size - 1)
}

fn resize_u8_torchvision_fixed(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    let horizontal = TorchvisionFixedCoefficients::new(source.width, target.width, filter)?;
    let vertical = TorchvisionFixedCoefficients::new(source.height, target.height, filter)?;
    let intermediate_size = ImageSize {
        height: source.height,
        width: target.width,
    };
    let mut intermediate = vec![0; image_len(intermediate_size, channels)?];
    let mut output = vec![0; image_len(target, channels)?];

    for y in 0..source.height {
        for x in 0..target.width {
            let weights = horizontal.weights(x);
            let source_x = horizontal.start(x);
            for channel in 0..channels {
                let mut sum = horizontal.rounding_bias();
                for (offset, &weight) in weights.iter().enumerate() {
                    let input = values[(y * source.width + source_x + offset) * channels + channel];
                    sum += i32::from(input) * i32::from(weight);
                }
                intermediate[(y * target.width + x) * channels + channel] =
                    torchvision_clip_u8(sum, horizontal.precision());
            }
        }
    }

    for y in 0..target.height {
        let weights = vertical.weights(y);
        let source_y = vertical.start(y);
        for x in 0..target.width {
            for channel in 0..channels {
                let mut sum = vertical.rounding_bias();
                for (offset, &weight) in weights.iter().enumerate() {
                    let input =
                        intermediate[((source_y + offset) * target.width + x) * channels + channel];
                    sum += i32::from(input) * i32::from(weight);
                }
                output[(y * target.width + x) * channels + channel] =
                    torchvision_clip_u8(sum, vertical.precision());
            }
        }
    }

    Ok(output)
}

fn torchvision_clip_u8(value: i32, precision: u32) -> u8 {
    (value >> precision).clamp(0, 255) as u8
}

#[derive(Clone, Debug)]
struct TorchvisionFixedCoefficients {
    starts: Vec<usize>,
    lengths: Vec<usize>,
    weights: Vec<i16>,
    window: usize,
    precision: u32,
}

impl TorchvisionFixedCoefficients {
    fn new(input: usize, output: usize, filter: ResizeFilter) -> Result<Self, ResizeError> {
        if input == 0 || output == 0 {
            return Err(ResizeError::InvalidSize {
                height: input,
                width: output,
            });
        }

        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = filter_support(filter) * filter_scale;
        let window = (support.ceil() as usize)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(ResizeError::ImageSizeOverflow)?;
        let coefficient_count = output
            .checked_mul(window)
            .ok_or(ResizeError::ImageSizeOverflow)?;
        let mut starts = Vec::with_capacity(output);
        let mut lengths = Vec::with_capacity(output);
        let mut normalized_weights = vec![0.0_f64; coefficient_count];
        let mut max_weight = 0.0_f64;

        for out in 0..output {
            let center = scale * (out as f64 + 0.5);
            let start = (center - support + 0.5).max(0.0) as usize;
            let end = (center + support + 0.5).min(input as f64) as usize;
            let length = end.saturating_sub(start).min(window);
            starts.push(start);
            lengths.push(length);

            let offset = out * window;
            let mut total = 0.0_f64;
            for index in 0..length {
                let x = (start + index) as f64;
                let weight = pillow_filter_weight(filter, (x - center + 0.5) / filter_scale);
                normalized_weights[offset + index] = weight;
                total += weight;
            }
            if total != 0.0 {
                for index in 0..length {
                    normalized_weights[offset + index] /= total;
                    max_weight = max_weight.max(normalized_weights[offset + index]);
                }
            }
        }

        let mut precision = 0;
        while precision < 22 {
            let next = (0.5 + max_weight * ((1_u32 << (precision + 1)) as f64)) as i32;
            if next >= 1_i32 << 15 {
                break;
            }
            precision += 1;
        }
        let scale = (1_u32 << precision) as f64;
        let weights = normalized_weights
            .into_iter()
            .map(|weight| {
                let scaled = weight * scale;
                if scaled < 0.0 {
                    (scaled - 0.5) as i16
                } else {
                    (scaled + 0.5) as i16
                }
            })
            .collect();

        Ok(Self {
            starts,
            lengths,
            weights,
            window,
            precision,
        })
    }

    fn start(&self, output: usize) -> usize {
        self.starts[output]
    }

    fn weights(&self, output: usize) -> &[i16] {
        let start = output * self.window;
        &self.weights[start..start + self.lengths[output]]
    }

    fn precision(&self) -> u32 {
        self.precision
    }

    fn rounding_bias(&self) -> i32 {
        1_i32 << (self.precision - 1)
    }
}

fn resize_u8_pillow(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    match filter {
        ResizeFilter::Nearest => resize_u8_nearest(values, source, channels, target),
        ResizeFilter::Bilinear | ResizeFilter::Bicubic | ResizeFilter::Lanczos => {
            resize_u8_pillow_fixed(values, source, channels, target, filter)
        }
    }
}

fn resize_u8_pillow_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    match filter {
        ResizeFilter::Nearest => resize_u8_nearest_crop(values, source, channels, crop),
        ResizeFilter::Bilinear | ResizeFilter::Bicubic | ResizeFilter::Lanczos => {
            let resized = resize_u8_pillow_fixed(values, source, channels, crop.resized, filter)?;
            crop_image_data(
                &resized,
                crop.resized,
                channels,
                crop.origin_x,
                crop.origin_y,
                crop.target,
            )
        }
    }
}

fn resize_u8_pillow_crop_with_workspace(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    workspace: &mut ResizeWorkspace,
) -> Result<Vec<u8>, ResizeError> {
    match filter {
        ResizeFilter::Nearest => resize_u8_nearest_crop(values, source, channels, crop),
        ResizeFilter::Bilinear | ResizeFilter::Bicubic | ResizeFilter::Lanczos => {
            let resized = resize_u8_pillow_fixed_with_workspace(
                values,
                source,
                channels,
                crop.resized,
                filter,
                workspace,
            )?;
            crop_image_data(
                &resized,
                crop.resized,
                channels,
                crop.origin_x,
                crop.origin_y,
                crop.target,
            )
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps resize, tensor, and transform parameters explicit"
)]
fn resize_u8_pillow_crop_into_with_workspace<T: TransformedOutput>(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    filter: ResizeFilter,
    output: &mut [T],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
    workspace: &mut ResizeWorkspace,
) -> Result<(), ResizeError> {
    if filter == ResizeFilter::Nearest {
        return write_nearest_crop_into(
            values, source, channels, crop, output, layout, multiplier, offset,
        );
    }

    let horizontal_key = CoefficientsKey::new(
        source.width,
        crop.resized.width,
        crop.origin_x,
        crop.target.width,
        filter,
    );
    let vertical_key = CoefficientsKey::new(
        source.height,
        crop.resized.height,
        crop.origin_y,
        crop.target.height,
        filter,
    );
    let parts = workspace.crop_parts(horizontal_key, vertical_key)?;
    let Some(source_rows) = parts.vertical.input_window() else {
        return Err(ResizeError::WorkspaceInvariant);
    };
    if source_rows.is_empty() {
        return Err(ResizeError::WorkspaceInvariant);
    }
    let source_row_start = source_rows.start;
    let intermediate_len = image_len(
        ImageSize {
            height: source_rows.len(),
            width: crop.target.width,
        },
        channels,
    )?;
    parts.u8_scratch.resize(intermediate_len, 0);
    let intermediate = parts.u8_scratch.as_mut_slice();

    if channels == 3 {
        for (intermediate_y, source_y) in source_rows.clone().enumerate() {
            for x in 0..crop.target.width {
                let weights = parts.horizontal.weights(x);
                let source_x = parts.horizontal.start(x);
                let mut sums = [PILLOW_ROUNDING_BIAS; 3];
                for (coefficient_offset, &weight) in weights.iter().enumerate() {
                    let input_index = (source_y * source.width + source_x + coefficient_offset) * 3;
                    let weight = i64::from(weight);
                    sums[0] += i64::from(values[input_index]) * weight;
                    sums[1] += i64::from(values[input_index + 1]) * weight;
                    sums[2] += i64::from(values[input_index + 2]) * weight;
                }
                let intermediate_index = (intermediate_y * crop.target.width + x) * 3;
                intermediate[intermediate_index] = pillow_clip_u8(sums[0]);
                intermediate[intermediate_index + 1] = pillow_clip_u8(sums[1]);
                intermediate[intermediate_index + 2] = pillow_clip_u8(sums[2]);
            }
        }
    } else {
        for (intermediate_y, source_y) in source_rows.enumerate() {
            for x in 0..crop.target.width {
                let weights = parts.horizontal.weights(x);
                let source_x = parts.horizontal.start(x);
                for channel in 0..channels {
                    let mut sum = PILLOW_ROUNDING_BIAS;
                    for (coefficient_offset, &weight) in weights.iter().enumerate() {
                        let input =
                            values[(source_y * source.width + source_x + coefficient_offset)
                                * channels
                                + channel];
                        sum += i64::from(input) * i64::from(weight);
                    }
                    intermediate[(intermediate_y * crop.target.width + x) * channels + channel] =
                        pillow_clip_u8(sum);
                }
            }
        }
    }

    if channels == 3 {
        let pixels = crop.target.height * crop.target.width;
        for y in 0..crop.target.height {
            let weights = parts.vertical.weights(y);
            let source_y = parts.vertical.start(y);
            for x in 0..crop.target.width {
                let mut sums = [PILLOW_ROUNDING_BIAS; 3];
                for (coefficient_offset, &weight) in weights.iter().enumerate() {
                    let intermediate_y = source_y + coefficient_offset - source_row_start;
                    let input_index = (intermediate_y * crop.target.width + x) * 3;
                    let weight = i64::from(weight);
                    sums[0] += i64::from(intermediate[input_index]) * weight;
                    sums[1] += i64::from(intermediate[input_index + 1]) * weight;
                    sums[2] += i64::from(intermediate[input_index + 2]) * weight;
                }
                let values = [
                    pillow_clip_u8(sums[0]),
                    pillow_clip_u8(sums[1]),
                    pillow_clip_u8(sums[2]),
                ];
                let pixel = y * crop.target.width + x;
                match layout {
                    F32ImageLayout::Chw => {
                        output[pixel] =
                            T::from_f32(transform_u8_parts(values[0], multiplier[0], offset[0]));
                        output[pixels + pixel] =
                            T::from_f32(transform_u8_parts(values[1], multiplier[1], offset[1]));
                        output[2 * pixels + pixel] =
                            T::from_f32(transform_u8_parts(values[2], multiplier[2], offset[2]));
                    }
                    F32ImageLayout::Hwc => {
                        let destination = pixel * 3;
                        output[destination] =
                            T::from_f32(transform_u8_parts(values[0], multiplier[0], offset[0]));
                        output[destination + 1] =
                            T::from_f32(transform_u8_parts(values[1], multiplier[1], offset[1]));
                        output[destination + 2] =
                            T::from_f32(transform_u8_parts(values[2], multiplier[2], offset[2]));
                    }
                }
            }
        }
    } else {
        for y in 0..crop.target.height {
            let weights = parts.vertical.weights(y);
            let source_y = parts.vertical.start(y);
            for x in 0..crop.target.width {
                for channel in 0..channels {
                    let mut sum = PILLOW_ROUNDING_BIAS;
                    for (coefficient_offset, &weight) in weights.iter().enumerate() {
                        let intermediate_y = source_y + coefficient_offset - source_row_start;
                        let input = intermediate
                            [(intermediate_y * crop.target.width + x) * channels + channel];
                        sum += i64::from(input) * i64::from(weight);
                    }
                    let value = pillow_clip_u8(sum);
                    let destination = f32_destination(crop.target, channels, layout, x, y, channel);
                    output[destination] =
                        T::from_f32(transform_u8(value, channel, multiplier, offset));
                }
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps resize, tensor, and transform parameters explicit"
)]
fn write_nearest_crop_into<T: TransformedOutput>(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
    output: &mut [T],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
) -> Result<(), ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    let y_scale = source.height as f64 / crop.resized.height as f64;
    let x_scale = source.width as f64 / crop.resized.width as f64;

    for y in 0..crop.target.height {
        let resized_y = crop.origin_y + y;
        let source_y = ((resized_y as f64 + 0.5) * y_scale)
            .floor()
            .min((source.height - 1) as f64) as usize;
        for x in 0..crop.target.width {
            let resized_x = crop.origin_x + x;
            let source_x = ((resized_x as f64 + 0.5) * x_scale)
                .floor()
                .min((source.width - 1) as f64) as usize;
            let source_start = (source_y * source.width + source_x) * channels;
            for channel in 0..channels {
                let destination = f32_destination(crop.target, channels, layout, x, y, channel);
                output[destination] = T::from_f32(transform_u8(
                    values[source_start + channel],
                    channel,
                    multiplier,
                    offset,
                ));
            }
        }
    }

    Ok(())
}

const PILLOW_PRECISION_BITS: u32 = 22;
const PILLOW_ROUNDING_BIAS: i64 = 1_i64 << (PILLOW_PRECISION_BITS - 1);

fn resize_u8_pillow_fixed(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    let horizontal = PillowFixedCoefficients::new(source.width, target.width, filter)?;
    let vertical = PillowFixedCoefficients::new(source.height, target.height, filter)?;
    let mut intermediate = vec![
        0;
        image_len(
            ImageSize {
                height: source.height,
                width: target.width
            },
            channels
        )?
    ];
    let mut output = vec![0; image_len(target, channels)?];

    for y in 0..source.height {
        for x in 0..target.width {
            let weights = horizontal.weights(x);
            let source_x = horizontal.start(x);
            for channel in 0..channels {
                let mut sum = PILLOW_ROUNDING_BIAS;
                for (offset, &weight) in weights.iter().enumerate() {
                    let input = values[(y * source.width + source_x + offset) * channels + channel];
                    sum += i64::from(input) * i64::from(weight);
                }
                intermediate[(y * target.width + x) * channels + channel] = pillow_clip_u8(sum);
            }
        }
    }

    for y in 0..target.height {
        let weights = vertical.weights(y);
        let source_y = vertical.start(y);
        for x in 0..target.width {
            for channel in 0..channels {
                let mut sum = PILLOW_ROUNDING_BIAS;
                for (offset, &weight) in weights.iter().enumerate() {
                    let input =
                        intermediate[((source_y + offset) * target.width + x) * channels + channel];
                    sum += i64::from(input) * i64::from(weight);
                }
                output[(y * target.width + x) * channels + channel] = pillow_clip_u8(sum);
            }
        }
    }

    Ok(output)
}

fn resize_u8_pillow_fixed_with_workspace(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    filter: ResizeFilter,
    workspace: &mut ResizeWorkspace,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    let horizontal_key = CoefficientsKey::new(source.width, target.width, 0, target.width, filter);
    let vertical_key = CoefficientsKey::new(source.height, target.height, 0, target.height, filter);
    let parts = workspace.crop_parts(horizontal_key, vertical_key)?;
    let intermediate_len = image_len(
        ImageSize {
            height: source.height,
            width: target.width,
        },
        channels,
    )?;
    parts.u8_scratch.resize(intermediate_len, 0);
    let intermediate = parts.u8_scratch.as_mut_slice();
    let mut output = vec![0; image_len(target, channels)?];

    for y in 0..source.height {
        for x in 0..target.width {
            let weights = parts.horizontal.weights(x);
            let source_x = parts.horizontal.start(x);
            for channel in 0..channels {
                let mut sum = PILLOW_ROUNDING_BIAS;
                for (offset, &weight) in weights.iter().enumerate() {
                    let input = values[(y * source.width + source_x + offset) * channels + channel];
                    sum += i64::from(input) * i64::from(weight);
                }
                intermediate[(y * target.width + x) * channels + channel] = pillow_clip_u8(sum);
            }
        }
    }

    for y in 0..target.height {
        let weights = parts.vertical.weights(y);
        let source_y = parts.vertical.start(y);
        for x in 0..target.width {
            for channel in 0..channels {
                let mut sum = PILLOW_ROUNDING_BIAS;
                for (offset, &weight) in weights.iter().enumerate() {
                    let input =
                        intermediate[((source_y + offset) * target.width + x) * channels + channel];
                    sum += i64::from(input) * i64::from(weight);
                }
                output[(y * target.width + x) * channels + channel] = pillow_clip_u8(sum);
            }
        }
    }

    Ok(output)
}

fn pillow_clip_u8(value: i64) -> u8 {
    (value >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

#[derive(Clone, Debug)]
struct PillowFixedCoefficients {
    starts: Vec<usize>,
    lengths: Vec<usize>,
    weights: Vec<i32>,
    window: usize,
}

impl PillowFixedCoefficients {
    fn new(input: usize, output: usize, filter: ResizeFilter) -> Result<Self, ResizeError> {
        Self::new_window(input, output, 0, output, filter)
    }

    fn new_window(
        input: usize,
        output: usize,
        output_start: usize,
        output_len: usize,
        filter: ResizeFilter,
    ) -> Result<Self, ResizeError> {
        if input == 0 || output == 0 {
            return Err(ResizeError::InvalidSize {
                height: input,
                width: output,
            });
        }
        if output_start
            .checked_add(output_len)
            .is_none_or(|end| end > output)
        {
            return Err(ResizeError::CropWindowOutOfBounds {
                resized_height: 1,
                resized_width: output,
                origin_x: output_start,
                origin_y: 0,
                target_height: 1,
                target_width: output_len,
            });
        }

        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = filter_support(filter) * filter_scale;
        let window = (support.ceil() as usize)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(ResizeError::ImageSizeOverflow)?;
        let coefficient_count = output_len
            .checked_mul(window)
            .ok_or(ResizeError::ImageSizeOverflow)?;
        let mut starts = Vec::with_capacity(output_len);
        let mut lengths = Vec::with_capacity(output_len);
        let mut weights = vec![0_i32; coefficient_count];
        let mut normalized = Vec::with_capacity(window);

        for local_output in 0..output_len {
            let out = output_start + local_output;
            let center = (out as f64 + 0.5) * scale;
            let mut start = (center - support + 0.5).trunc() as isize;
            let mut end = (center + support + 0.5).trunc() as isize;
            start = start.clamp(0, input as isize);
            end = end.clamp(0, input as isize);
            let start = start as usize;
            let length = (end as usize).saturating_sub(start);
            starts.push(start);
            lengths.push(length);

            let offset = local_output * window;
            normalized.clear();
            let mut sum = 0.0_f64;
            for index in 0..length {
                let x = (start + index) as f64;
                let weight = pillow_filter_weight(filter, (x - center + 0.5) / filter_scale);
                normalized.push(weight);
                sum += weight;
            }
            if sum != 0.0 {
                for (index, &weight) in normalized.iter().enumerate() {
                    let scaled = weight / sum * ((1_u64 << PILLOW_PRECISION_BITS) as f64);
                    weights[offset + index] = if scaled < 0.0 {
                        (scaled - 0.5) as i32
                    } else {
                        (scaled + 0.5) as i32
                    };
                }
            }
        }

        Ok(Self {
            starts,
            lengths,
            weights,
            window,
        })
    }

    fn start(&self, output: usize) -> usize {
        self.starts[output]
    }

    fn weights(&self, output: usize) -> &[i32] {
        let start = output * self.window;
        &self.weights[start..start + self.lengths[output]]
    }

    fn input_window(&self) -> Option<std::ops::Range<usize>> {
        let start = self.starts.iter().copied().min()?;
        let end = self
            .starts
            .iter()
            .zip(&self.lengths)
            .map(|(&start, &length)| start + length)
            .max()?;
        Some(start..end)
    }
}

fn pillow_filter_weight(filter: ResizeFilter, x: f64) -> f64 {
    let x = x.abs();
    match filter {
        ResizeFilter::Nearest => f64::from(x < 0.5),
        ResizeFilter::Bilinear => {
            if x < 1.0 {
                1.0 - x
            } else {
                0.0
            }
        }
        ResizeFilter::Bicubic => {
            if x < 1.0 {
                ((1.5 * x - 2.5) * x * x) + 1.0
            } else if x < 2.0 {
                (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0
            } else {
                0.0
            }
        }
        ResizeFilter::Lanczos => {
            if x == 0.0 {
                1.0
            } else if x < 3.0 {
                let value = std::f64::consts::PI * x;
                let scaled = value / 3.0;
                (value.sin() / value) * (scaled.sin() / scaled)
            } else {
                0.0
            }
        }
    }
}

fn resize_u8_nearest(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<u8>, ResizeError> {
    let mut output = vec![0; image_len(target, channels)?];
    let y_scale = source.height as f64 / target.height as f64;
    let x_scale = source.width as f64 / target.width as f64;

    for y in 0..target.height {
        let source_y = ((y as f64 + 0.5) * y_scale)
            .floor()
            .min((source.height - 1) as f64) as usize;
        for x in 0..target.width {
            let source_x = ((x as f64 + 0.5) * x_scale)
                .floor()
                .min((source.width - 1) as f64) as usize;
            let source_start = (source_y * source.width + source_x) * channels;
            let target_start = (y * target.width + x) * channels;
            output[target_start..target_start + channels]
                .copy_from_slice(&values[source_start..source_start + channels]);
        }
    }
    Ok(output)
}

fn resize_u8_nearest_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: ResizeCrop,
) -> Result<Vec<u8>, ResizeError> {
    let mut output = vec![0; image_len(crop.target, channels)?];
    let y_scale = source.height as f64 / crop.resized.height as f64;
    let x_scale = source.width as f64 / crop.resized.width as f64;

    for y in 0..crop.target.height {
        let resized_y = crop.origin_y + y;
        let source_y = ((resized_y as f64 + 0.5) * y_scale)
            .floor()
            .min((source.height - 1) as f64) as usize;
        for x in 0..crop.target.width {
            let resized_x = crop.origin_x + x;
            let source_x = ((resized_x as f64 + 0.5) * x_scale)
                .floor()
                .min((source.width - 1) as f64) as usize;
            let source_start = (source_y * source.width + source_x) * channels;
            let target_start = (y * crop.target.width + x) * channels;
            output[target_start..target_start + channels]
                .copy_from_slice(&values[source_start..source_start + channels]);
        }
    }
    Ok(output)
}

fn validate_f32_output(
    actual_len: usize,
    target: ImageSize,
    channels: usize,
    multiplier: &[f32],
    offset: &[f32],
) -> Result<(), ResizeError> {
    let expected = image_len(target, channels)?;
    if actual_len != expected {
        return Err(ResizeError::InvalidOutputLength {
            expected,
            actual: actual_len,
        });
    }
    if multiplier.len() != channels || offset.len() != channels {
        return Err(ResizeError::InvalidTransformLength {
            channels,
            multiplier: multiplier.len(),
            offset: offset.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn write_interleaved_u8_to_f32(
    values: &[u8],
    target: ImageSize,
    channels: usize,
    output: &mut [f32],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
) -> Result<(), ResizeError> {
    write_interleaved_u8_to_output(values, target, channels, output, layout, multiplier, offset)
}

fn write_interleaved_u8_to_output<T: TransformedOutput>(
    values: &[u8],
    target: ImageSize,
    channels: usize,
    output: &mut [T],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
) -> Result<(), ResizeError> {
    validate_image_data(values.len(), target, channels)?;
    if channels == 3 {
        write_interleaved_rgb_u8_to_output(values, target, output, layout, multiplier, offset);
        return Ok(());
    }

    for (pixel_index, pixel) in values.chunks_exact(channels).enumerate() {
        let y = pixel_index / target.width;
        let x = pixel_index % target.width;
        for (channel, value) in pixel.iter().copied().enumerate() {
            let destination = f32_destination(target, channels, layout, x, y, channel);
            output[destination] = T::from_f32(transform_u8(value, channel, multiplier, offset));
        }
    }
    Ok(())
}

fn write_interleaved_rgb_u8_to_output<T: TransformedOutput>(
    values: &[u8],
    target: ImageSize,
    output: &mut [T],
    layout: F32ImageLayout,
    multiplier: &[f32],
    offset: &[f32],
) {
    debug_assert_eq!(multiplier.len(), 3);
    debug_assert_eq!(offset.len(), 3);
    let red_multiplier = multiplier[0];
    let green_multiplier = multiplier[1];
    let blue_multiplier = multiplier[2];
    let red_offset = offset[0];
    let green_offset = offset[1];
    let blue_offset = offset[2];

    match layout {
        F32ImageLayout::Chw => {
            let pixels = target.height * target.width;
            for (pixel_index, pixel) in values.chunks_exact(3).enumerate() {
                output[pixel_index] =
                    T::from_f32(transform_u8_parts(pixel[0], red_multiplier, red_offset));
                output[pixels + pixel_index] =
                    T::from_f32(transform_u8_parts(pixel[1], green_multiplier, green_offset));
                output[2 * pixels + pixel_index] =
                    T::from_f32(transform_u8_parts(pixel[2], blue_multiplier, blue_offset));
            }
        }
        F32ImageLayout::Hwc => {
            for (pixel_index, pixel) in values.chunks_exact(3).enumerate() {
                let output_index = pixel_index * 3;
                output[output_index] =
                    T::from_f32(transform_u8_parts(pixel[0], red_multiplier, red_offset));
                output[output_index + 1] =
                    T::from_f32(transform_u8_parts(pixel[1], green_multiplier, green_offset));
                output[output_index + 2] =
                    T::from_f32(transform_u8_parts(pixel[2], blue_multiplier, blue_offset));
            }
        }
    }
}

trait TransformedOutput: Sized {
    fn from_f32(value: f32) -> Self;
}

impl TransformedOutput for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl TransformedOutput for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
}

fn f32_destination(
    target: ImageSize,
    channels: usize,
    layout: F32ImageLayout,
    x: usize,
    y: usize,
    channel: usize,
) -> usize {
    match layout {
        F32ImageLayout::Chw => channel * target.height * target.width + y * target.width + x,
        F32ImageLayout::Hwc => (y * target.width + x) * channels + channel,
    }
}

fn transform_u8(value: u8, channel: usize, multiplier: &[f32], offset: &[f32]) -> f32 {
    transform_u8_parts(value, multiplier[channel], offset[channel])
}

fn transform_u8_parts(value: u8, multiplier: f32, offset: f32) -> f32 {
    value as f32 * multiplier + offset
}

#[derive(Clone, Debug)]
struct CachedCoefficients {
    key: CoefficientsKey,
    coefficients: PillowFixedCoefficients,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoefficientsKey {
    input: usize,
    output: usize,
    output_start: usize,
    output_len: usize,
    filter: ResizeFilter,
}

impl CoefficientsKey {
    fn new(
        input: usize,
        output: usize,
        output_start: usize,
        output_len: usize,
        filter: ResizeFilter,
    ) -> Self {
        Self {
            input,
            output,
            output_start,
            output_len,
            filter,
        }
    }
}

#[derive(Clone, Debug)]
struct Coefficients {
    starts: Vec<usize>,
    lengths: Vec<usize>,
    weights: Vec<f32>,
    window: usize,
}

impl Coefficients {
    fn new(input: usize, output: usize, filter: ResizeFilter) -> Result<Self, ResizeError> {
        Self::new_window(input, output, 0, output, filter)
    }

    fn new_window(
        input: usize,
        output: usize,
        output_start: usize,
        output_len: usize,
        filter: ResizeFilter,
    ) -> Result<Self, ResizeError> {
        if input == 0 || output == 0 {
            return Err(ResizeError::InvalidSize {
                height: input,
                width: output,
            });
        }
        if output_start
            .checked_add(output_len)
            .is_none_or(|end| end > output)
        {
            return Err(ResizeError::CropWindowOutOfBounds {
                resized_height: 1,
                resized_width: output,
                origin_x: output_start,
                origin_y: 0,
                target_height: 1,
                target_width: output_len,
            });
        }

        let filter_support = filter_support(filter);
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = filter_support * filter_scale;
        let window = (support * 2.0).ceil() as usize + 3;
        let mut starts = Vec::with_capacity(output_len);
        let mut lengths = Vec::with_capacity(output_len);
        let mut weights = vec![0.0; output_len * window];

        for local_out in 0..output_len {
            let out = output_start + local_out;
            let center = (out as f64 + 0.5) * scale;
            let mut start = (center - support + 0.5).floor() as isize;
            let mut end = (center + support + 0.5).floor() as isize;
            start = start.clamp(0, input as isize);
            end = end.clamp(0, input as isize);
            if end <= start {
                end = (start + 1).min(input as isize);
            }

            let start = start as usize;
            let len = (end as usize).saturating_sub(start);
            starts.push(start);
            lengths.push(len);

            let mut sum = 0.0;
            for index in 0..len {
                let input_position = start + index;
                let x = ((input_position as f64 - center + 0.5) / filter_scale) as f32;
                let weight = filter_weight(filter, x);
                weights[local_out * window + index] = weight;
                sum += f64::from(weight);
            }

            if sum != 0.0 {
                let scale = (1.0 / sum) as f32;
                for index in 0..len {
                    weights[local_out * window + index] *= scale;
                }
            }
        }

        Ok(Self {
            starts,
            lengths,
            weights,
            window,
        })
    }

    fn start(&self, output: usize) -> usize {
        self.starts[output]
    }

    fn weights(&self, output: usize) -> &[f32] {
        let start = output * self.window;
        &self.weights[start..start + self.lengths[output]]
    }
}

fn crop_image_data(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
) -> Result<Vec<u8>, ResizeError> {
    validate_image_data(values.len(), source, channels)?;
    validate_crop_window(source, origin_x, origin_y, target)?;
    let row_len = target
        .width
        .checked_mul(channels)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let mut output = Vec::with_capacity(image_len(target, channels)?);
    for y in 0..target.height {
        let source_start = ((origin_y + y) * source.width + origin_x)
            .checked_mul(channels)
            .ok_or(ResizeError::ImageSizeOverflow)?;
        let source_end = source_start
            .checked_add(row_len)
            .ok_or(ResizeError::ImageSizeOverflow)?;
        output.extend_from_slice(&values[source_start..source_end]);
    }
    Ok(output)
}

fn filter_support(filter: ResizeFilter) -> f64 {
    match filter {
        ResizeFilter::Nearest => 0.5,
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::Bicubic => 2.0,
        ResizeFilter::Lanczos => 3.0,
    }
}

fn filter_weight(filter: ResizeFilter, x: f32) -> f32 {
    let x = x.abs();
    match filter {
        ResizeFilter::Nearest => {
            if x < 0.5 {
                1.0
            } else {
                0.0
            }
        }
        ResizeFilter::Bilinear => {
            if x < 1.0 {
                1.0 - x
            } else {
                0.0
            }
        }
        ResizeFilter::Bicubic => {
            if x < 1.0 {
                ((1.5 * x - 2.5) * x * x) + 1.0
            } else if x < 2.0 {
                (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0
            } else {
                0.0
            }
        }
        ResizeFilter::Lanczos => {
            if x == 0.0 {
                1.0
            } else if x < 3.0 {
                sinc(x) * sinc(x / 3.0)
            } else {
                0.0
            }
        }
    }
}

fn sinc(x: f32) -> f32 {
    let x = std::f32::consts::PI * x;
    x.sin() / x
}

fn fast_resize_pixel_type(channels: usize) -> Result<fast_image_resize::PixelType, ResizeError> {
    match channels {
        1 => Ok(fast_image_resize::PixelType::U8),
        3 => Ok(fast_image_resize::PixelType::U8x3),
        4 => Ok(fast_image_resize::PixelType::U8x4),
        _ => Err(ResizeError::UnsupportedChannels(channels)),
    }
}

fn fast_resize_algorithm(filter: ResizeFilter) -> fast_image_resize::ResizeAlg {
    match filter {
        ResizeFilter::Nearest => fast_image_resize::ResizeAlg::Nearest,
        ResizeFilter::Bilinear => {
            fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)
        }
        ResizeFilter::Bicubic => {
            fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom)
        }
        ResizeFilter::Lanczos => {
            fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3)
        }
    }
}

fn validate_image_data(actual: usize, size: ImageSize, channels: usize) -> Result<(), ResizeError> {
    let expected = image_len(size, channels)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ResizeError::InvalidBufferLength { expected, actual })
    }
}

fn validate_crop_window(
    resized: ImageSize,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
) -> Result<(), ResizeError> {
    let end_x = origin_x
        .checked_add(target.width)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    let end_y = origin_y
        .checked_add(target.height)
        .ok_or(ResizeError::ImageSizeOverflow)?;
    if end_x <= resized.width && end_y <= resized.height {
        Ok(())
    } else {
        Err(ResizeError::CropWindowOutOfBounds {
            resized_height: resized.height,
            resized_width: resized.width,
            origin_x,
            origin_y,
            target_height: target.height,
            target_width: target.width,
        })
    }
}

fn image_len(size: ImageSize, channels: usize) -> Result<usize, ResizeError> {
    if !matches!(channels, 1 | 3 | 4) {
        return Err(ResizeError::UnsupportedChannels(channels));
    }
    checked_area(size.height, size.width)?
        .checked_mul(channels)
        .ok_or(ResizeError::ImageSizeOverflow)
}

fn checked_area(height: usize, width: usize) -> Result<usize, ResizeError> {
    height
        .checked_mul(width)
        .ok_or(ResizeError::ImageSizeOverflow)
}

fn usize_to_u32(value: usize, dimension: &'static str) -> Result<u32, ResizeError> {
    u32::try_from(value).map_err(|_| ResizeError::DimensionTooLarge {
        dimension,
        value,
        max: u32::MAX as usize,
    })
}

fn backend_error(profile: ResizeProfile, error: impl std::fmt::Display) -> ResizeError {
    ResizeError::Backend {
        profile,
        message: error.to_string(),
    }
}

/// Errors returned by resize kernels.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum ResizeError {
    /// Image dimensions were zero.
    #[error("height and width must be positive, got height={height}, width={width}")]
    InvalidSize {
        /// Image height.
        height: usize,
        /// Image width.
        width: usize,
    },
    /// Image buffer length did not match dimensions and channels.
    #[error("invalid image buffer length: expected {expected} elements, got {actual}")]
    InvalidBufferLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Output tensor buffer length did not match target dimensions and channels.
    #[error("invalid output buffer length: expected {expected} values, got {actual}")]
    InvalidOutputLength {
        /// Expected float length.
        expected: usize,
        /// Actual float length.
        actual: usize,
    },
    /// Transform vectors did not match the channel count.
    #[error(
        "invalid transform length: channels={channels}, multiplier={multiplier}, offset={offset}"
    )]
    InvalidTransformLength {
        /// Expected channel count.
        channels: usize,
        /// Multiplier length.
        multiplier: usize,
        /// Offset length.
        offset: usize,
    },
    /// A floating-point scale was not finite.
    #[error("scale factor must be finite, got {0}")]
    InvalidScaleFactor(f32),
    /// An input floating-point value was not finite.
    #[error("input value at index {index} must be finite")]
    NonFiniteValue {
        /// Invalid value index.
        index: usize,
    },
    /// Image dimensions or buffer length overflowed `usize`.
    #[error("image size overflows usize")]
    ImageSizeOverflow,
    /// A dimension exceeded a backend-supported maximum.
    #[error("{dimension} dimension {value} exceeds maximum supported value {max}")]
    DimensionTooLarge {
        /// Dimension name.
        dimension: &'static str,
        /// Actual dimension value.
        value: usize,
        /// Maximum supported value.
        max: usize,
    },
    /// The channel count is unsupported.
    #[error("unsupported channel count {0}")]
    UnsupportedChannels(usize),
    /// The requested crop window was outside the resized image.
    #[error(
        "crop window origin=({origin_x}, {origin_y}) size={target_width}x{target_height} exceeds resized image {resized_width}x{resized_height}"
    )]
    CropWindowOutOfBounds {
        /// Resized image height.
        resized_height: usize,
        /// Resized image width.
        resized_width: usize,
        /// Crop origin x coordinate.
        origin_x: usize,
        /// Crop origin y coordinate.
        origin_y: usize,
        /// Crop target height.
        target_height: usize,
        /// Crop target width.
        target_width: usize,
    },
    /// A resize backend failed.
    #[error("{profile:?} resize backend failed: {message}")]
    Backend {
        /// Selected resize profile.
        profile: ResizeProfile,
        /// Backend error message.
        message: String,
    },
    /// Internal workspace cache state was inconsistent.
    #[error("resize workspace cache state was inconsistent")]
    WorkspaceInvariant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_u8_torchvision_matches_audited_bilinear_tensor_fixture() {
        let source = ImageSize::new(3, 4).expect("source dimensions should be valid");
        let target = ImageSize::new(2, 3).expect("target dimensions should be valid");
        let values = deterministic_rgb(source);

        let resized = resize_u8(
            &values,
            source,
            3,
            target,
            ResizeFilter::Bilinear,
            ResizeProfile::Torchvision,
        )
        .expect("Torchvision bilinear resize should succeed");

        assert_eq!(
            resized,
            [
                92, 129, 99, 159, 116, 153, 112, 101, 138, 167, 157, 126, 154, 111, 148, 139, 176,
                101,
            ]
        );
    }

    #[test]
    fn resize_u8_torchvision_matches_audited_bicubic_tensor_fixture() {
        let source = ImageSize::new(3, 4).expect("source dimensions should be valid");
        let target = ImageSize::new(2, 3).expect("target dimensions should be valid");
        let values = deterministic_rgb(source);

        let resized = resize_u8(
            &values,
            source,
            3,
            target,
            ResizeFilter::Bicubic,
            ResizeProfile::Torchvision,
        )
        .expect("Torchvision bicubic resize should succeed");

        assert_eq!(
            resized,
            [
                81, 131, 93, 164, 117, 157, 108, 93, 138, 175, 160, 122, 152, 105, 152, 137, 187,
                98,
            ]
        );
    }

    #[test]
    fn resize_u8_torchvision_nearest_matches_audited_nearest_exact_fixture() {
        let source = ImageSize::new(8, 1).expect("source dimensions should be valid");
        let target = ImageSize::new(10, 1).expect("target dimensions should be valid");

        let resized = resize_u8(
            &[0, 1, 2, 3, 4, 5, 6, 7],
            source,
            1,
            target,
            ResizeFilter::Nearest,
            ResizeProfile::Torchvision,
        )
        .expect("Torchvision nearest-exact resize should succeed");

        assert_eq!(resized, [0, 1, 2, 2, 3, 4, 5, 6, 6, 7]);
    }

    #[test]
    fn resize_u8_to_f32_torchvision_nearest_exact_preserves_non_unit_scale() {
        let source = ImageSize::new(2, 2).expect("source dimensions should be valid");
        let target = ImageSize::new(3, 3).expect("target dimensions should be valid");

        let resized = resize_u8_to_f32_torchvision(
            &[0, 64, 128, 255],
            source,
            1,
            target,
            ResizeFilter::Nearest,
            1.0 / 255.0,
        )
        .expect("Torchvision nearest resize should succeed");

        assert_eq!(
            resized,
            [
                0.0,
                64.0 / 255.0,
                64.0 / 255.0,
                128.0 / 255.0,
                1.0,
                1.0,
                128.0 / 255.0,
                1.0,
                1.0,
            ]
        );
    }

    #[test]
    fn resize_u8_pillow_profile_matches_pillow_clip_fixture_prefix() {
        let source = ImageSize::new(17, 13).unwrap();
        let target = ImageSize::new(292, 224).unwrap();
        let values = deterministic_rgb(source);

        let resized = resize_u8_owned(
            values,
            source,
            3,
            target,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();

        let crop_y = (target.height - 224) / 2;
        let crop_prefix = crop_y * target.width * 3;

        assert_eq!(
            &resized[crop_prefix..crop_prefix + 12],
            &[135, 189, 235, 135, 189, 234, 134, 189, 234, 134, 188, 232]
        );
    }

    #[test]
    fn resize_u8_pillow_profile_preserves_constant_image() {
        let source = ImageSize::new(11, 7).unwrap();
        let target = ImageSize::new(23, 19).unwrap();
        let values = vec![137; source.width * source.height * 3];

        let resized = resize_u8(
            &values,
            source,
            3,
            target,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();

        assert!(resized.iter().all(|value| *value == 137));
    }

    #[test]
    fn resize_u8_crop_pillow_matches_full_resize_then_crop() {
        let source = ImageSize::new(17, 13).unwrap();
        let resized_size = ImageSize::new(292, 224).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_rgb(source);
        let crop_origin_x = 0;
        let crop_origin_y = (resized_size.height - target.height) / 2;

        let full = resize_u8(
            &values,
            source,
            3,
            resized_size,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let expected =
            crop_image_data(&full, resized_size, 3, crop_origin_x, crop_origin_y, target).unwrap();
        let crop = ResizeCrop::new(resized_size, crop_origin_x, crop_origin_y, target).unwrap();
        let cropped = resize_u8_crop(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();

        assert_eq!(cropped, expected);
    }

    #[test]
    fn resize_u8_crop_pillow_matches_full_grayscale_resize_then_crop() {
        let source = ImageSize::new(101, 79).unwrap();
        let resized_size = ImageSize::new(224, 277).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_gray(source);
        let crop_origin_x = (resized_size.width - target.width) / 2;
        let crop_origin_y = 0;

        let full = resize_u8(
            &values,
            source,
            1,
            resized_size,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let expected =
            crop_image_data(&full, resized_size, 1, crop_origin_x, crop_origin_y, target).unwrap();
        let crop = ResizeCrop::new(resized_size, crop_origin_x, crop_origin_y, target).unwrap();
        let cropped = resize_u8_crop(
            &values,
            source,
            1,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();

        assert_eq!(cropped, expected);
    }

    #[test]
    fn resize_workspace_reuses_crop_coefficients() {
        let source = ImageSize::new(101, 79).unwrap();
        let resized_size = ImageSize::new(224, 277).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_gray(source);
        let crop_origin_x = (resized_size.width - target.width) / 2;
        let crop = ResizeCrop::new(resized_size, crop_origin_x, 0, target).unwrap();
        let mut workspace = ResizeWorkspace::new();

        let expected = resize_u8_crop(
            &values,
            source,
            1,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let resized = resize_u8_crop_with_workspace(
            &values,
            source,
            1,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(resized, expected);
        assert_eq!(workspace.coefficient_cache_len(), 2);
    }

    #[test]
    fn resize_workspace_reuses_rgb_horizontal_scratch() {
        let source = ImageSize::new(101, 79).unwrap();
        let resized_size = ImageSize::new(224, 277).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_rgb(source);
        let crop_origin_x = (resized_size.width - target.width) / 2;
        let crop = ResizeCrop::new(resized_size, crop_origin_x, 0, target).unwrap();
        let mut workspace = ResizeWorkspace::new();

        let expected = resize_u8_crop(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let resized = resize_u8_crop_with_workspace(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(resized, expected);
        assert!(workspace.u8_scratch_capacity() >= 3 * target.width);
    }

    #[test]
    fn pillow_fixed_window_coefficients_match_the_full_table_slice() {
        const INPUT: usize = 79;
        const OUTPUT: usize = 277;
        const OUTPUT_START: usize = 23;
        const OUTPUT_LEN: usize = 41;

        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Bilinear,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos,
        ] {
            let full = PillowFixedCoefficients::new(INPUT, OUTPUT, filter).unwrap();
            let window = PillowFixedCoefficients::new_window(
                INPUT,
                OUTPUT,
                OUTPUT_START,
                OUTPUT_LEN,
                filter,
            )
            .unwrap();

            for local_output in 0..OUTPUT_LEN {
                let full_output = OUTPUT_START + local_output;
                assert_eq!(window.start(local_output), full.start(full_output));
                assert_eq!(window.weights(local_output), full.weights(full_output));
            }
        }
    }

    #[test]
    fn pillow_direct_crop_matches_full_resize_reference_across_filters_layouts_and_sizes() {
        let geometries = [
            (
                ImageSize::new(47, 83).unwrap(),
                ResizeCrop::new(
                    ImageSize::new(73, 129).unwrap(),
                    23,
                    17,
                    ImageSize::new(41, 59).unwrap(),
                )
                .unwrap(),
                3,
            ),
            (
                ImageSize::new(91, 137).unwrap(),
                ResizeCrop::new(
                    ImageSize::new(63, 95).unwrap(),
                    19,
                    11,
                    ImageSize::new(41, 57).unwrap(),
                )
                .unwrap(),
                1,
            ),
            (
                ImageSize::new(79, 43).unwrap(),
                ResizeCrop::new(
                    ImageSize::new(61, 107).unwrap(),
                    31,
                    7,
                    ImageSize::new(37, 53).unwrap(),
                )
                .unwrap(),
                4,
            ),
            (
                ImageSize::new(350, 146).unwrap(),
                ResizeCrop::new(
                    ImageSize::new(536, 224).unwrap(),
                    0,
                    156,
                    ImageSize::new(224, 224).unwrap(),
                )
                .unwrap(),
                3,
            ),
        ];

        for (source, crop, channels) in geometries {
            for filter in [
                ResizeFilter::Nearest,
                ResizeFilter::Bilinear,
                ResizeFilter::Bicubic,
                ResizeFilter::Lanczos,
            ] {
                for layout in [F32ImageLayout::Chw, F32ImageLayout::Hwc] {
                    assert_direct_pillow_crop_parity(source, crop, channels, filter, layout);
                }
            }
        }
    }

    #[test]
    fn direct_f16_output_matches_rounding_the_f32_output_bit_for_bit() {
        let source = ImageSize::new(47, 83).unwrap();
        let crop = ResizeCrop::new(
            ImageSize::new(73, 129).unwrap(),
            23,
            17,
            ImageSize::new(41, 59).unwrap(),
        )
        .unwrap();
        let values = deterministic_rgb(source);
        let multiplier = [0.003_125, 0.004_375, 0.005_625];
        let offset = [-0.75, -0.5, -0.25];

        for profile in [
            ResizeProfile::Fast,
            ResizeProfile::Pillow,
            ResizeProfile::Torchvision,
        ] {
            for filter in [
                ResizeFilter::Nearest,
                ResizeFilter::Bilinear,
                ResizeFilter::Bicubic,
                ResizeFilter::Lanczos,
            ] {
                for layout in [F32ImageLayout::Chw, F32ImageLayout::Hwc] {
                    let mut f32_workspace = ResizeWorkspace::new();
                    let mut f16_workspace = ResizeWorkspace::new();
                    let mut f32_output = vec![0.0; crop.target.height * crop.target.width * 3];
                    let mut f16_output = vec![f16::ZERO; f32_output.len()];
                    resize_u8_crop_into_f32_with_workspace(
                        &values,
                        source,
                        3,
                        crop,
                        filter,
                        profile,
                        &mut f32_output,
                        layout,
                        &multiplier,
                        &offset,
                        &mut f32_workspace,
                    )
                    .unwrap();
                    resize_u8_crop_into_f16_with_workspace(
                        &values,
                        source,
                        3,
                        crop,
                        filter,
                        profile,
                        &mut f16_output,
                        layout,
                        &multiplier,
                        &offset,
                        &mut f16_workspace,
                    )
                    .unwrap();
                    let expected = f32_output
                        .into_iter()
                        .map(f16::from_f32)
                        .collect::<Vec<_>>();

                    assert_eq!(
                        f16_output, expected,
                        "f16 output diverged for profile={profile:?}, filter={filter:?}, layout={layout:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn pillow_direct_tall_crop_only_filters_rows_used_by_the_vertical_window() {
        let source = ImageSize::new(350, 146).unwrap();
        let crop = ResizeCrop::new(
            ImageSize::new(536, 224).unwrap(),
            0,
            156,
            ImageSize::new(224, 224).unwrap(),
        )
        .unwrap();
        let values = deterministic_rgb(source);
        let mut output = vec![0.0; crop.target.height * crop.target.width * 3];
        let mut workspace = ResizeWorkspace::new();

        resize_u8_crop_into_f32_with_workspace(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut output,
            F32ImageLayout::Hwc,
            &[1.0; 3],
            &[0.0; 3],
            &mut workspace,
        )
        .unwrap();

        let source_rows = PillowFixedCoefficients::new_window(
            source.height,
            crop.resized.height,
            crop.origin_y,
            crop.target.height,
            ResizeFilter::Bicubic,
        )
        .unwrap()
        .input_window()
        .unwrap();
        assert!(source_rows.len() < source.height);
        assert_eq!(
            workspace.u8_scratch.len(),
            source_rows.len() * crop.target.width * 3
        );
    }

    #[test]
    fn resize_crop_into_f32_matches_u8_then_transform() {
        let source = ImageSize::new(101, 79).unwrap();
        let resized_size = ImageSize::new(224, 277).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_rgb(source);
        let crop_origin_x = (resized_size.width - target.width) / 2;
        let crop = ResizeCrop::new(resized_size, crop_origin_x, 0, target).unwrap();
        let multiplier = [1.0 / 255.0 / 0.25, 1.0 / 255.0 / 0.5, 1.0 / 255.0 / 0.75];
        let offset = [-0.1 / 0.25, -0.2 / 0.5, -0.3 / 0.75];
        let mut workspace = ResizeWorkspace::new();
        let mut direct = vec![0.0; target.height * target.width * 3];

        resize_u8_crop_into_f32_with_workspace(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut direct,
            F32ImageLayout::Chw,
            &multiplier,
            &offset,
            &mut workspace,
        )
        .unwrap();
        let resized = resize_u8_crop(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let mut expected = vec![0.0; direct.len()];
        write_interleaved_u8_to_f32(
            &resized,
            target,
            3,
            &mut expected,
            F32ImageLayout::Chw,
            &multiplier,
            &offset,
        )
        .unwrap();

        assert_eq!(direct, expected);
    }

    #[test]
    fn resize_crop_into_f32_matches_u8_then_transform_for_downscaled_chw() {
        let source = ImageSize::new(480, 640).unwrap();
        let resized_size = ImageSize::new(224, 299).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_rgb(source);
        let crop = ResizeCrop::new(resized_size, 37, 0, target).unwrap();
        let multiplier = [1.0 / 255.0 / 0.25, 1.0 / 255.0 / 0.5, 1.0 / 255.0 / 0.75];
        let offset = [-0.1 / 0.25, -0.2 / 0.5, -0.3 / 0.75];
        let mut workspace = ResizeWorkspace::new();
        let mut direct = vec![0.0; target.height * target.width * 3];

        resize_u8_crop_into_f32_with_workspace(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut direct,
            F32ImageLayout::Chw,
            &multiplier,
            &offset,
            &mut workspace,
        )
        .unwrap();
        let resized = resize_u8_crop(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let mut expected = vec![0.0; direct.len()];
        write_interleaved_u8_to_f32(
            &resized,
            target,
            3,
            &mut expected,
            F32ImageLayout::Chw,
            &multiplier,
            &offset,
        )
        .unwrap();

        assert_eq!(direct, expected);
    }

    #[test]
    fn resize_crop_into_f32_matches_u8_then_transform_for_hwc() {
        let source = ImageSize::new(101, 79).unwrap();
        let resized_size = ImageSize::new(224, 277).unwrap();
        let target = ImageSize::new(224, 224).unwrap();
        let values = deterministic_rgb(source);
        let crop_origin_x = (resized_size.width - target.width) / 2;
        let crop = ResizeCrop::new(resized_size, crop_origin_x, 0, target).unwrap();
        let multiplier = [1.0 / 255.0 / 0.25, 1.0 / 255.0 / 0.5, 1.0 / 255.0 / 0.75];
        let offset = [-0.1 / 0.25, -0.2 / 0.5, -0.3 / 0.75];
        let mut workspace = ResizeWorkspace::new();
        let mut direct = vec![0.0; target.height * target.width * 3];

        resize_u8_crop_into_f32_with_workspace(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut direct,
            F32ImageLayout::Hwc,
            &multiplier,
            &offset,
            &mut workspace,
        )
        .unwrap();
        let resized = resize_u8_crop(
            &values,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let mut expected = vec![0.0; direct.len()];
        write_interleaved_u8_to_f32(
            &resized,
            target,
            3,
            &mut expected,
            F32ImageLayout::Hwc,
            &multiplier,
            &offset,
        )
        .unwrap();

        assert_eq!(direct, expected);
    }

    #[test]
    fn resize_u8_fast_rejects_unsupported_channels() {
        let err = resize_u8(
            &[0, 1],
            ImageSize::new(1, 1).unwrap(),
            2,
            ImageSize::new(1, 1).unwrap(),
            ResizeFilter::Bilinear,
            ResizeProfile::Fast,
        )
        .unwrap_err();

        assert_eq!(err, ResizeError::UnsupportedChannels(2));
    }

    fn assert_direct_pillow_crop_parity(
        source: ImageSize,
        crop: ResizeCrop,
        channels: usize,
        filter: ResizeFilter,
        layout: F32ImageLayout,
    ) {
        let values = deterministic_interleaved(source, channels);
        let multiplier = (0..channels)
            .map(|channel| 0.003_125 + channel as f32 * 0.001_25)
            .collect::<Vec<_>>();
        let offset = (0..channels)
            .map(|channel| -0.75 + channel as f32 * 0.25)
            .collect::<Vec<_>>();
        let mut workspace = ResizeWorkspace::new();
        let mut direct = vec![0.0; crop.target.height * crop.target.width * channels];

        resize_u8_crop_into_f32_with_workspace(
            &values,
            source,
            channels,
            crop,
            filter,
            ResizeProfile::Pillow,
            &mut direct,
            layout,
            &multiplier,
            &offset,
            &mut workspace,
        )
        .unwrap();

        let full = resize_u8(
            &values,
            source,
            channels,
            crop.resized,
            filter,
            ResizeProfile::Pillow,
        )
        .unwrap();
        let resized = crop_image_data(
            &full,
            crop.resized,
            channels,
            crop.origin_x,
            crop.origin_y,
            crop.target,
        )
        .unwrap();
        let mut expected = vec![0.0; direct.len()];
        write_interleaved_u8_to_f32(
            &resized,
            crop.target,
            channels,
            &mut expected,
            layout,
            &multiplier,
            &offset,
        )
        .unwrap();

        assert_eq!(
            direct, expected,
            "direct Pillow crop diverged for source={source:?}, crop={crop:?}, channels={channels}, filter={filter:?}, layout={layout:?}"
        );
    }

    fn deterministic_interleaved(size: ImageSize, channels: usize) -> Vec<u8> {
        (0..size.width * size.height * channels)
            .map(|index| ((index as u32 * 37 + 17) % 256) as u8)
            .collect()
    }

    fn deterministic_rgb(size: ImageSize) -> Vec<u8> {
        deterministic_interleaved(size, 3)
    }

    fn deterministic_gray(size: ImageSize) -> Vec<u8> {
        (0..size.width * size.height)
            .map(|index| ((index as u32 * 29 + 101) % 256) as u8)
            .collect()
    }
}
