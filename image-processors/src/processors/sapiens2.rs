//! Sapiens2 FP32 pose crops and UDP DARK heatmap decoding.
//!
//! Source: Transformers models/sapiens2/image_processing_sapiens2.py. Crop
//! sampling retains floating pixels and chooses bilinear when downscaling and
//! bicubic when upscaling. It does not use VitPose's integer-quantized affine path.

use image_resize_kernels::{
    gaussian_blur_zero_f32_planes, grid_sample_f32_planes, GridInterpolation,
};
use thiserror::Error;

use super::{ImageGptPreprocessingPolicy, TaskVisionPoseBox};
use crate::tensor::{Layout, Tensor, TensorData, TensorError};
use crate::transforms::{
    centered_padding, convert_frame_pixel_format, pad_frame, resize_f32_bilinear_interleaved,
    resize_frame_with_decision, ImageSize, ResizeMode, TransformError,
};
use crate::{ImageFrame, ImageProcessor, ImageProcessorConfig, ImageProcessorError};

/// Errors from crop geometry, heatmap decoding or shared preprocessing.
#[derive(Debug, Error)]
pub enum Sapiens2ProcessorError {
    /// Invalid tensor, geometry or decoder settings.
    #[error("invalid Sapiens2 processor input: {0}")]
    Invalid(&'static str),
    /// Shared preprocessing failed.
    #[error(transparent)]
    Image(#[from] ImageProcessorError),
    /// Shared grid sampling failed.
    #[error(transparent)]
    Sampling(#[from] image_resize_kernels::ResizeError),
    /// Tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Image geometry was invalid.
    #[error(transparent)]
    Geometry(#[from] TransformError),
    /// Shared center cropping failed.
    #[error(transparent)]
    Preset(#[from] super::encoder_presets::PresetProcessorError),
}

/// A decoded person with confidence-filtered keypoints in image coordinates.
#[derive(Debug, Clone)]
pub struct Sapiens2Pose {
    /// XY positions, optionally rescaled to a requested output image size.
    pub keypoints: Vec<[f32; 2]>,
    /// Unblurred heatmap peak confidences.
    pub scores: Vec<f32>,
    /// Original keypoint channel identifiers.
    pub labels: Vec<usize>,
    /// XYXY bounding box in the same coordinate system as the keypoints.
    pub bbox: [f32; 4],
}

/// Configured whole-image Sapiens2 preprocessing, including optional symmetric padding.
pub struct Sapiens2ImageProcessor {
    config: ImageProcessorConfig,
    rescale_factor: f64,
    processor: ImageProcessor,
    size: ImageSize,
    do_pad: bool,
    center_crop: Option<ImageSize>,
}

impl Sapiens2ImageProcessor {
    /// Create a processor from the selected checkpoint settings.
    ///
    /// # Errors
    /// Reports invalid geometry or shared tensor conversion settings.
    pub fn new(
        config: ImageProcessorConfig,
        size: ImageSize,
        do_pad: bool,
        center_crop: Option<ImageSize>,
    ) -> Result<Self, Sapiens2ProcessorError> {
        ImageSize::new(size.height, size.width)?;
        if let Some(crop) = center_crop {
            ImageSize::new(crop.height, crop.width)?;
        }
        ImageProcessor::new(config.clone())?;
        let rescale_factor = f64::from(config.rescale_factor);
        let mut tensor_config = config.clone();
        tensor_config.do_resize = false;
        tensor_config.do_rescale = false;
        tensor_config.do_normalize = false;
        tensor_config.pixel_format = Some(crate::PixelFormat::Rgb8);
        tensor_config.output_layout = Layout::NCHW;
        Ok(Self {
            config,
            rescale_factor,
            processor: ImageProcessor::new(tensor_config)?,
            size,
            do_pad,
            center_crop,
        })
    }

    /// Retain a JSON rescale factor's precision until fused normalization.
    /// The shared tensor stage validates the factor during preprocessing.
    pub fn with_rescale_factor(mut self, factor: f64) -> Self {
        self.rescale_factor = factor;
        self
    }

    /// Resize, optionally pad/crop, rescale and normalize decoded images.
    ///
    /// # Errors
    /// Reports incompatible image shapes or shared preprocessing failures.
    pub fn preprocess(&self, images: &[ImageFrame]) -> Result<Tensor, Sapiens2ProcessorError> {
        let mut prepared = Vec::with_capacity(images.len());
        for image in images {
            let mut image = convert_frame_pixel_format(image, crate::PixelFormat::Rgb8)?;
            if self.config.do_resize {
                let target = if self.do_pad {
                    sapiens2_resize_size(ImageSize::new(image.height(), image.width())?, self.size)?
                } else {
                    self.size
                };
                image = resize_frame_with_decision(
                    &image,
                    target,
                    self.config.resize_decision(),
                    ResizeMode::Default,
                )?;
                if self.do_pad {
                    image = pad_frame(&image, centered_padding(target, self.size)?, &[0])?;
                }
            }
            if let Some(crop) = self.center_crop {
                image = super::encoder_presets::center_crop_with_padding_rounding(
                    &image, crop, false, false,
                )?;
            }
            prepared.push(image);
        }
        Ok(ImageGptPreprocessingPolicy::Torchvision.finish_tensor(
            self.processor.preprocess_images(&prepared)?,
            self.config.do_rescale.then_some(self.rescale_factor),
            self.config.do_normalize.then_some((
                self.config.image_mean.as_slice(),
                self.config.image_std.as_slice(),
            )),
            None,
        )?)
    }
}

/// Sapiens2 crop processor using shared RGB conversion and grid kernels.
pub struct Sapiens2PoseProcessor {
    config: ImageProcessorConfig,
    rescale_factor: f64,
    source_processor: ImageProcessor,
    size: ImageSize,
    center_crop: Option<ImageSize>,
}

impl Sapiens2PoseProcessor {
    /// Create a processor from checkpoint-selected scaling and normalization.
    /// The target must have two or more pixels along each axis (UDP endpoints).
    ///
    /// # Errors
    /// Reports invalid sizes or shared normalization configuration.
    pub fn new(
        config: ImageProcessorConfig,
        size: ImageSize,
    ) -> Result<Self, Sapiens2ProcessorError> {
        if size.height < 2 || size.width < 2 {
            return Err(Sapiens2ProcessorError::Invalid(
                "crop size must exceed one pixel",
            ));
        }
        // Validate the selected normalization in its existing owner.
        ImageProcessor::new(config.clone())?;
        if config.do_normalize
            && (![1, 3].contains(&config.image_mean.len())
                || ![1, 3].contains(&config.image_std.len()))
        {
            return Err(Sapiens2ProcessorError::Invalid(
                "RGB normalization requires one or three statistics",
            ));
        }
        let rescale_factor = f64::from(config.rescale_factor);
        let mut source = config.clone();
        source.do_resize = false;
        source.do_normalize = false;
        source.do_rescale = false;
        source.do_binarize = false;
        source.pixel_format = Some(crate::PixelFormat::Rgb8);
        source.output_layout = Layout::NCHW;
        Ok(Self {
            config,
            rescale_factor,
            source_processor: ImageProcessor::new(source)?,
            size,
            center_crop: None,
        })
    }

    /// Retain a JSON rescale factor's precision until fused normalization.
    /// The shared tensor stage validates the factor during preprocessing.
    pub fn with_rescale_factor(mut self, factor: f64) -> Self {
        self.rescale_factor = factor;
        self
    }

    /// Select the optional center crop applied to FP32 person crops before normalization.
    ///
    /// # Errors
    /// Reports invalid crop dimensions.
    pub fn with_center_crop(
        mut self,
        center_crop: Option<ImageSize>,
    ) -> Result<Self, Sapiens2ProcessorError> {
        if let Some(size) = center_crop {
            ImageSize::new(size.height, size.width)?;
        }
        self.center_crop = center_crop;
        Ok(self)
    }

    /// Crop persons from one decoded image into normalized NCHW FP32 pixels.
    ///
    /// # Errors
    /// Reports empty boxes, degenerate images, overflow or preprocessing failures.
    pub fn preprocess(
        &self,
        image: &ImageFrame,
        boxes: &[TaskVisionPoseBox],
        flip: bool,
    ) -> Result<Tensor, Sapiens2ProcessorError> {
        if boxes.is_empty() || image.width() < 2 || image.height() < 2 {
            return Err(Sapiens2ProcessorError::Invalid(
                "nonempty boxes and a nondegenerate image are required",
            ));
        }
        let source = self.source_processor.preprocess_image(image)?;
        let TensorData::F32(values) = source.data() else {
            return Err(Sapiens2ProcessorError::Invalid(
                "source pixels must be FP32",
            ));
        };
        let target_pixels = self
            .size
            .height
            .checked_mul(self.size.width)
            .ok_or(Sapiens2ProcessorError::Invalid("crop size overflow"))?;
        let length = boxes
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_mul(target_pixels))
            .ok_or(Sapiens2ProcessorError::Invalid("crop batch overflow"))?;
        let mut output = Vec::with_capacity(length);
        for &bbox in boxes {
            let (center, scale) = crop_geometry(bbox, self.size);
            let sx = (self.size.width - 1) as f32 / scale[0];
            let sy = (self.size.height - 1) as f32 / scale[1];
            let mut grid = Vec::with_capacity(target_pixels);
            for y in 0..self.size.height {
                for x in 0..self.size.width {
                    let ix = x as f32 / sx + center[0] - 0.5 * scale[0];
                    let iy = y as f32 / sy + center[1] - 0.5 * scale[1];
                    grid.push([
                        2.0 * ix / (image.width() - 1) as f32 - 1.0,
                        2.0 * iy / (image.height() - 1) as f32 - 1.0,
                    ]);
                }
            }
            let interpolation = if sx.min(sy) < 1.0 {
                GridInterpolation::Bilinear
            } else {
                GridInterpolation::Bicubic
            };
            let crop = grid_sample_f32_planes(
                values,
                image_resize_kernels::ImageSize::new(image.height(), image.width())?,
                3,
                &grid,
                image_resize_kernels::ImageSize::new(self.size.height, self.size.width)?,
                interpolation,
            )?;
            if let Some(size) = self.center_crop {
                for plane in crop.chunks_exact(target_pixels) {
                    output.extend(super::encoder_presets::center_crop_interleaved_f32(
                        plane, self.size, 1, size, false, false,
                    )?);
                }
            } else {
                output.extend(crop);
            }
        }
        let output_size = self.center_crop.unwrap_or(self.size);
        if flip {
            for row in output.chunks_exact_mut(output_size.width) {
                row.reverse();
            }
        }
        Ok(ImageGptPreprocessingPolicy::Torchvision.finish_tensor(
            Tensor::new(
                TensorData::F32(output),
                vec![boxes.len(), 3, output_size.height, output_size.width],
                Layout::NCHW,
            )?,
            self.config.do_rescale.then_some(self.rescale_factor),
            self.config.do_normalize.then_some((
                self.config.image_mean.as_slice(),
                self.config.image_std.as_slice(),
            )),
            None,
        )?)
    }

    /// Decode NCHW heatmaps with FP32 UDP DARK refinement and source remapping.
    /// `flipped` supplies raw flipped-input predictions and the mirrored channel pairs.
    ///
    /// # Errors
    /// Reports inconsistent shapes, invalid kernel size or output scaling settings.
    #[allow(clippy::too_many_arguments)] // Keep the independent decode inputs explicit.
    pub fn postprocess(
        &self,
        heatmaps: &Tensor,
        flipped: Option<(&Tensor, &[[usize; 2]])>,
        boxes: &[TaskVisionPoseBox],
        kernel: usize,
        threshold: Option<f32>,
        source_size: Option<ImageSize>,
        target_size: Option<ImageSize>,
    ) -> Result<Vec<Sapiens2Pose>, Sapiens2ProcessorError> {
        if heatmaps.shape().len() != 4
            || heatmaps.shape()[0] != boxes.len()
            || heatmaps.layout() != Layout::NCHW
        {
            return Err(Sapiens2ProcessorError::Invalid(
                "heatmaps must have shape [persons,keypoints,height,width]",
            ));
        }
        if kernel <= 1 || kernel.is_multiple_of(2) || kernel > isize::MAX as usize {
            return Err(Sapiens2ProcessorError::Invalid(
                "blur kernel must be odd and greater than one",
            ));
        }
        if target_size.is_some() && source_size.is_none() {
            return Err(Sapiens2ProcessorError::Invalid(
                "target scaling requires source size",
            ));
        }
        let TensorData::F32(original) = heatmaps.data() else {
            return Err(Sapiens2ProcessorError::Invalid(
                "DARK heatmaps must be FP32",
            ));
        };
        let mut maps = original.clone();
        if let Some((flipped, pairs)) = flipped {
            let TensorData::F32(values) = flipped.data() else {
                return Err(Sapiens2ProcessorError::Invalid(
                    "flipped heatmaps must be FP32",
                ));
            };
            if flipped.shape() != heatmaps.shape() || flipped.layout() != Layout::NCHW {
                return Err(Sapiens2ProcessorError::Invalid(
                    "flipped heatmaps differ in shape",
                ));
            }
            let shape = heatmaps.shape();
            let mut channels = (0..shape[1]).collect::<Vec<_>>();
            for &[left, right] in pairs {
                if left >= shape[1] || right >= shape[1] {
                    return Err(Sapiens2ProcessorError::Invalid(
                        "flip pair exceeds heatmap channels",
                    ));
                }
                channels[left] = right;
                channels[right] = left;
            }
            for person in 0..shape[0] {
                for (channel, &source) in channels.iter().enumerate() {
                    for y in 0..shape[2] {
                        for x in 0..shape[3] {
                            let index =
                                ((person * shape[1] + channel) * shape[2] + y) * shape[3] + x;
                            let source = ((person * shape[1] + source) * shape[2] + y) * shape[3]
                                + shape[3]
                                - 1
                                - x;
                            maps[index] = (maps[index] + values[source]) / 2.0;
                        }
                    }
                }
            }
        }
        let shape = heatmaps.shape();
        let (channels, height, width) = (shape[1], shape[2], shape[3]);
        if height < 2 || width < 2 || channels == 0 {
            return Err(Sapiens2ProcessorError::Invalid(
                "heatmap dimensions must exceed one",
            ));
        }
        let sigma = (0.3 * ((kernel - 1) as f64 * 0.5 - 1.0) + 0.8) as f32;
        let mut blurred = gaussian_blur_zero_f32_planes(
            &maps,
            image_resize_kernels::ImageSize::new(height, width)?,
            boxes.len() * channels,
            kernel,
            sigma,
        )?;
        for (source, destination) in maps
            .chunks_exact(height * width)
            .zip(blurred.chunks_exact_mut(height * width))
        {
            let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let blurred_max = destination
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let gain = if blurred_max > 0.0 {
                maximum / blurred_max
            } else {
                1.0
            };
            for value in destination {
                *value = (*value * gain).clamp(1e-3, 50.0).ln();
            }
        }
        let mut results = Vec::with_capacity(boxes.len());
        for (person, &bbox) in boxes.iter().enumerate() {
            let (center, scale) = crop_geometry(bbox, self.size);
            let output_scale = match (source_size, target_size) {
                (Some(source), Some(target)) => [
                    target.width as f32 / source.width as f32,
                    target.height as f32 / source.height as f32,
                ],
                _ => [1.0, 1.0],
            };
            let mut result = Sapiens2Pose {
                keypoints: Vec::new(),
                scores: Vec::new(),
                labels: Vec::new(),
                bbox: [
                    bbox.top_left_x() * output_scale[0],
                    bbox.top_left_y() * output_scale[1],
                    (bbox.top_left_x() + bbox.width()) * output_scale[0],
                    (bbox.top_left_y() + bbox.height()) * output_scale[1],
                ],
            };
            for channel in 0..channels {
                let plane = &maps[(person * channels + channel) * height * width
                    ..(person * channels + channel + 1) * height * width];
                let (mut peak, mut score) = (0, plane[0]);
                for (index, &value) in plane.iter().enumerate().skip(1) {
                    if value > score {
                        peak = index;
                        score = value;
                    }
                }
                if threshold.is_some_and(|threshold| {
                    score.partial_cmp(&threshold) != Some(std::cmp::Ordering::Greater)
                }) {
                    continue;
                }
                let (x, y) = if score > 0.0 {
                    ((peak % width) as isize, (peak / width) as isize)
                } else {
                    (-1, -1)
                };
                let [px, py] = super::pose::dark_refine_keypoint(
                    &blurred,
                    [boxes.len(), channels, height, width],
                    person,
                    channel,
                    [x, y],
                    super::pose::HessianPrecision::Float32,
                );
                result.keypoints.push([
                    (px / (width - 1) as f32 * scale[0] + center[0] - 0.5 * scale[0])
                        * output_scale[0],
                    (py / (height - 1) as f32 * scale[1] + center[1] - 0.5 * scale[1])
                        * output_scale[1],
                ]);
                result.scores.push(score);
                result.labels.push(channel);
            }
            results.push(result);
        }
        Ok(results)
    }
}

fn crop_geometry(bbox: TaskVisionPoseBox, size: ImageSize) -> ([f32; 2], [f32; 2]) {
    let center = [
        bbox.top_left_x() + bbox.width() / 2.0,
        bbox.top_left_y() + bbox.height() / 2.0,
    ];
    let width = bbox.width() * 1.25;
    let height = bbox.height() * 1.25;
    let aspect = size.width as f32 / size.height as f32;
    let scale = if width > height * aspect {
        [width, width / aspect]
    } else {
        [height * aspect, height]
    };
    (center, scale)
}

/// Aspect-preserving resize with both enlargement and reduction, using the
/// FP64 scale-then-truncate policy of Transformers get_image_size_for_max_height_width.
///
/// # Errors
/// Returns an error for invalid dimensions or a result smaller than one pixel.
pub fn sapiens2_resize_size(
    source: ImageSize,
    target: ImageSize,
) -> Result<ImageSize, TransformError> {
    ImageSize::new(source.height, source.width)?;
    ImageSize::new(target.height, target.width)?;
    let scale = (target.height as f64 / source.height as f64)
        .min(target.width as f64 / source.width as f64);
    ImageSize::new(
        (source.height as f64 * scale) as usize,
        (source.width as f64 * scale) as usize,
    )
}

/// A decoded byte image or a CHW/HWC image tensor used for matting compositing.
#[derive(Debug, Clone)]
pub enum Sapiens2MattingBackground {
    /// Decoded byte pixels, converted to unit-range RGB.
    Image(ImageFrame),
    /// Floating pixels retain their range; integer pixels scale by their dtype maximum.
    /// One-channel tensors broadcast over RGB; RGB tensors retain their channels.
    Tensor(Tensor),
}

impl From<ImageFrame> for Sapiens2MattingBackground {
    fn from(image: ImageFrame) -> Self {
        Self::Image(image)
    }
}

impl From<Tensor> for Sapiens2MattingBackground {
    fn from(tensor: Tensor) -> Self {
        Self::Tensor(tensor)
    }
}

/// Composite a premultiplied FP32 RGB foreground and alpha over an image background.
/// Backgrounds are converted to unit floating pixels before non-antialiased
/// bilinear resize. Output quantization follows Torchvision's scale=true uint8 conversion.
///
/// # Errors
/// Reports inconsistent CHW tensor shapes or shared conversion/resize failures.
pub fn composite_sapiens2_matting(
    foreground: &Tensor,
    alpha: &Tensor,
    background: &Sapiens2MattingBackground,
) -> Result<Tensor, Sapiens2ProcessorError> {
    if foreground.layout() != Layout::CHW
        || alpha.layout() != Layout::CHW
        || foreground.shape().len() != 3
        || alpha.shape().len() != 3
        || foreground.shape()[0] != 3
        || alpha.shape()[0] != 1
        || foreground.shape()[1..] != alpha.shape()[1..]
    {
        return Err(Sapiens2ProcessorError::Invalid(
            "matting requires matching RGB/alpha CHW maps",
        ));
    }
    let (TensorData::F32(foreground_values), TensorData::F32(alpha_values)) =
        (foreground.data(), alpha.data())
    else {
        return Err(Sapiens2ProcessorError::Invalid("matting maps must be FP32"));
    };
    let background = match background {
        Sapiens2MattingBackground::Image(image) => {
            ImageProcessor::new(ImageProcessorConfig::default())?.preprocess_image(image)?
        }
        Sapiens2MattingBackground::Tensor(tensor) => {
            if tensor.shape().len() != 3
                || !matches!(tensor.layout(), Layout::CHW | Layout::HWC)
                || ![1, 3].contains(&tensor.channels())
            {
                return Err(Sapiens2ProcessorError::Invalid(
                    "background tensor must be a one-channel or RGB CHW/HWC image",
                ));
            }
            let scale = match tensor.data() {
                TensorData::F32(_)
                | TensorData::F16(_)
                | TensorData::BF16(_)
                | TensorData::Bool(_) => 1.0,
                TensorData::U8(_) => 1.0 / f32::from(u8::MAX),
                TensorData::I8(_) => 1.0 / f32::from(i8::MAX),
                TensorData::I32(_) => 1.0 / i32::MAX as f32,
                TensorData::I64(_) => 1.0 / i64::MAX as f32,
                _ => {
                    return Err(Sapiens2ProcessorError::Invalid(
                        "quantized backgrounds are unsupported",
                    ));
                }
            };
            tensor.to_layout(Layout::CHW)?.rescale(scale)?
        }
    };
    let TensorData::F32(background_values) = background.data() else {
        return Err(Sapiens2ProcessorError::Invalid(
            "background must convert to FP32",
        ));
    };
    let source = ImageSize::new(background.height(), background.width())?;
    let target = ImageSize::new(foreground.height(), foreground.width())?;
    let mut resized = Vec::new();
    let source_pixels = source.height * source.width;
    for channel in 0..3 {
        let offset = (channel % background.channels()) * source_pixels;
        let plane = &background_values[offset..offset + source_pixels];
        resized.extend(resize_f32_bilinear_interleaved(
            plane, source, 1, target, false,
        )?);
    }
    let pixels = target.height * target.width;
    let values = foreground_values
        .iter()
        .zip(resized)
        .enumerate()
        .map(|(index, (&foreground, background))| {
            let value =
                (foreground + (1.0 - alpha_values[index % pixels]) * background).clamp(0.0, 1.0);
            (value * (256.0 - 1e-3)) as u8
        })
        .collect::<Vec<_>>();
    Ok(Tensor::new(
        TensorData::U8(values),
        foreground.shape().to_vec(),
        Layout::CHW,
    )?)
}
