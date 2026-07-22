//! Diffusers Marigold dense-prediction image processor.

use std::borrow::Cow;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{
    batched_layout_for_image_layout, checked_spatial_pad_mul, resize_decision_for_parts,
    DepthMapU16, DEFAULT_RESCALE_FACTOR, DIFFUSION_VAE_IMAGE_MEAN, DIFFUSION_VAE_IMAGE_STD,
};
use crate::image::{BatchExecution, ImageProcessorConfig, ImageProcessorError};
use crate::media::{
    load_image_from_path_with_backend, ImageDecodeBackend, ImageFrame, PixelFormat,
};
use crate::output::{ProcessorMetadataName, ProcessorMetadataValue, ProcessorOutput};
use crate::postprocess::{
    post_process_dense_map, post_process_depth_map, DenseMapPostprocessOutput,
    DepthMapPostprocessOutput,
};
use crate::recipe::{RecipeDenseMapTask, RecipeDepthUnit};
use crate::tensor::{ImageLayout, Layout, Tensor, TensorData, TensorLeadingAxis};
use crate::transforms::{
    convert_frame_pixel_format, pad_to_multiple_plan, ImageSize, Padding, ResizeDecision,
    ResizeFilter, ResizeMode, ResizeParity, TransformError,
};

const MARIGOLD_VAE_SCALE_FACTOR: usize = 8;

/// Diffusers Marigold preprocessing output.
#[derive(Clone, Debug, PartialEq)]
pub struct MarigoldPreprocessOutput {
    pixel_values: Tensor,
    padding: Padding,
    original_resolution: ImageSize,
    reshaped_input_size: ImageSize,
}

impl MarigoldPreprocessOutput {
    /// Returns the padded image tensor.
    pub fn pixel_values(&self) -> &Tensor {
        &self.pixel_values
    }

    /// Returns the bottom/right replicate padding applied after optional resize.
    pub fn padding(&self) -> Padding {
        self.padding
    }

    /// Returns the original decoded image resolution before optional resize.
    pub fn original_resolution(&self) -> ImageSize {
        self.original_resolution
    }

    /// Returns the image resolution after optional max-edge resize and before padding.
    pub fn reshaped_input_size(&self) -> ImageSize {
        self.reshaped_input_size
    }

    /// Consumes this output into tensor, padding, original size, and pre-pad size.
    pub fn into_parts(self) -> (Tensor, Padding, ImageSize, ImageSize) {
        (
            self.pixel_values,
            self.padding,
            self.original_resolution,
            self.reshaped_input_size,
        )
    }
}

/// Configuration for Diffusers Marigold dense-prediction preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MarigoldImageProcessorConfig {
    /// VAE scale factor used for bottom/right replicate padding alignment.
    pub vae_scale_factor: usize,
    /// Whether image values should be normalized from `[0, 1]` to `[-1, 1]`.
    pub do_normalize: bool,
    /// Whether canonical floating-point inputs should be range-checked.
    pub do_range_check: bool,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used for optional max-edge tensor resize.
    pub resample: ResizeFilter,
    /// Resize parity policy retained for catalog consistency.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl Default for MarigoldImageProcessorConfig {
    fn default() -> Self {
        Self {
            vae_scale_factor: MARIGOLD_VAE_SCALE_FACTOR,
            do_normalize: true,
            do_range_check: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl MarigoldImageProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the canonical tensor-stage config.
    ///
    /// Marigold resolves max-edge tensor resize and bottom/right replicate
    /// padding in the wrapper before returning `pixel_values`.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: self.do_normalize,
            image_mean: DIFFUSION_VAE_IMAGE_MEAN.to_vec(),
            image_std: DIFFUSION_VAE_IMAGE_STD.to_vec(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }
}

/// Diffusers Marigold image processor.
#[derive(Clone, Debug)]
pub struct MarigoldImageProcessor {
    config: MarigoldImageProcessorConfig,
}

impl MarigoldImageProcessor {
    /// Creates a processor from a validated Marigold configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when Marigold dimensions or resize settings are invalid.
    pub fn new(config: MarigoldImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_marigold_config(&config)?;
        Ok(Self { config })
    }

    /// Returns this processor's Marigold configuration.
    pub fn config(&self) -> &MarigoldImageProcessorConfig {
        &self.config
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, Marigold preprocessing, or
    /// tensor construction fails.
    pub fn open(
        &self,
        path: impl AsRef<Path>,
        processing_resolution: Option<usize>,
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_with_processing_resolution(&image, processing_resolution)
    }

    /// Loads and preprocesses image paths as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or preprocessing fails.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        processing_resolution: Option<usize>,
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images_with_processing_resolution(&images, processing_resolution)
    }

    /// Preprocesses one decoded image without max-edge resizing.
    ///
    /// # Errors
    ///
    /// Returns an error when canonical loading, normalization, padding, or
    /// tensor construction fails.
    pub fn preprocess_image(
        &self,
        image: &ImageFrame,
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        self.preprocess_image_with_processing_resolution(image, None)
    }

    /// Preprocesses one decoded image with optional max-edge resizing.
    ///
    /// `processing_resolution` mirrors Diffusers: `None` or `Some(0)` skips
    /// resizing; positive values resize the longest edge before VAE alignment.
    ///
    /// # Errors
    ///
    /// Returns an error when canonical loading, max-edge resizing, padding, or
    /// tensor construction fails.
    pub fn preprocess_image_with_processing_resolution(
        &self,
        image: &ImageFrame,
        processing_resolution: Option<usize>,
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        self.preprocess_images_with_processing_resolution(
            std::slice::from_ref(image),
            processing_resolution,
        )
    }

    /// Preprocesses decoded images as a batch tensor without max-edge resizing.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, input resolutions differ, or
    /// preprocessing fails.
    pub fn preprocess_images(
        &self,
        images: &[ImageFrame],
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        self.preprocess_images_with_processing_resolution(images, None)
    }

    /// Preprocesses decoded images as a batch tensor with optional max-edge resizing.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, input resolutions differ, or
    /// preprocessing fails.
    pub fn preprocess_images_with_processing_resolution(
        &self,
        images: &[ImageFrame],
        processing_resolution: Option<usize>,
    ) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
        marigold_preprocess_images(&self.config, images, processing_resolution)
    }

    /// Preprocesses one decoded image as typed processor output.
    ///
    /// The output contains `pixel_values`, `original_sizes`,
    /// `reshaped_input_sizes`, and a processor-specific `padding` shape
    /// metadata entry in top-right-bottom-left order.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
        processing_resolution: Option<usize>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_with_processing_resolution(image, processing_resolution)
            .map(marigold_processor_output)
    }

    /// Preprocesses decoded images as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
        processing_resolution: Option<usize>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images_with_processing_resolution(images, processing_resolution)
            .map(marigold_processor_output)
    }

    /// Restores a depth tensor by removing Marigold padding and resizing back.
    ///
    /// # Errors
    ///
    /// Returns an error when tensor layout, padding, channel count, or dense
    /// values are invalid.
    pub fn postprocess_depth(
        &self,
        tensor: &Tensor,
        preprocess: &MarigoldPreprocessOutput,
        unit: RecipeDepthUnit,
    ) -> Result<Vec<DepthMapPostprocessOutput>, ImageProcessorError> {
        let samples = marigold_unpadded_interleaved_samples(tensor, 1, preprocess.padding)?;
        samples
            .into_iter()
            .map(|sample| {
                post_process_depth_map(
                    &sample.values,
                    sample.size,
                    unit,
                    Some(preprocess.original_resolution),
                )
                .map_err(ImageProcessorError::Transform)
            })
            .collect()
    }

    /// Restores dense image-like maps by removing Marigold padding and resizing back.
    ///
    /// # Errors
    ///
    /// Returns an error when tensor layout, padding, channel count, or dense
    /// values are invalid.
    pub fn postprocess_dense_map(
        &self,
        tensor: &Tensor,
        task: RecipeDenseMapTask,
        channels: usize,
        preprocess: &MarigoldPreprocessOutput,
    ) -> Result<Vec<DenseMapPostprocessOutput>, ImageProcessorError> {
        let samples = marigold_unpadded_interleaved_samples(tensor, channels, preprocess.padding)?;
        samples
            .into_iter()
            .map(|sample| {
                post_process_dense_map(
                    &sample.values,
                    sample.size,
                    channels,
                    task,
                    Some(preprocess.original_resolution),
                )
                .map_err(ImageProcessorError::Transform)
            })
            .collect()
    }

    /// Restores Marigold surface-normal outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when postprocessing fails.
    pub fn postprocess_normals(
        &self,
        tensor: &Tensor,
        preprocess: &MarigoldPreprocessOutput,
    ) -> Result<Vec<DenseMapPostprocessOutput>, ImageProcessorError> {
        self.postprocess_dense_map(tensor, RecipeDenseMapTask::SurfaceNormal, 3, preprocess)
    }

    /// Restores Marigold intrinsic-image outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when postprocessing fails.
    pub fn postprocess_intrinsics(
        &self,
        tensor: &Tensor,
        preprocess: &MarigoldPreprocessOutput,
    ) -> Result<Vec<DenseMapPostprocessOutput>, ImageProcessorError> {
        self.postprocess_dense_map(tensor, RecipeDenseMapTask::IntrinsicImage, 3, preprocess)
    }

    /// Restores Marigold uncertainty outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when postprocessing fails or the channel count is not
    /// one or three.
    pub fn postprocess_uncertainty(
        &self,
        tensor: &Tensor,
        preprocess: &MarigoldPreprocessOutput,
    ) -> Result<Vec<DenseMapPostprocessOutput>, ImageProcessorError> {
        let channels = tensor.channels();
        if !matches!(channels, 1 | 3) {
            return Err(ImageProcessorError::UnsupportedProcessorOption {
                field: "uncertainty_channels",
            });
        }
        self.postprocess_dense_map(
            tensor,
            RecipeDenseMapTask::Uncertainty,
            channels,
            preprocess,
        )
    }

    /// Exports a normalized depth map as 16-bit depth values.
    ///
    /// # Errors
    ///
    /// Returns an error when the value range is invalid or output construction fails.
    pub fn export_depth_to_16bit(
        depth: &DepthMapPostprocessOutput,
        val_min: f32,
        val_max: f32,
    ) -> Result<DepthMapU16, ImageProcessorError> {
        marigold_export_depth_to_16bit(depth, val_min, val_max)
    }

    /// Visualizes a normalized depth map with Marigold's custom Spectral map.
    ///
    /// # Errors
    ///
    /// Returns an error when the value range is invalid or output construction fails.
    pub fn visualize_depth(
        depth: &DepthMapPostprocessOutput,
        val_min: f32,
        val_max: f32,
    ) -> Result<ImageFrame, ImageProcessorError> {
        marigold_visualize_depth(depth, val_min, val_max)
    }

    /// Visualizes a surface-normal map.
    ///
    /// # Errors
    ///
    /// Returns an error when channels or output construction are invalid.
    pub fn visualize_normals(
        normals: &DenseMapPostprocessOutput,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
    ) -> Result<ImageFrame, ImageProcessorError> {
        marigold_visualize_normals(normals, [flip_x, flip_y, flip_z])
    }

    /// Visualizes one RGB intrinsic-image map in sRGB space.
    ///
    /// # Errors
    ///
    /// Returns an error when channels or output construction are invalid.
    pub fn visualize_intrinsic_rgb(
        intrinsic: &DenseMapPostprocessOutput,
    ) -> Result<ImageFrame, ImageProcessorError> {
        marigold_visualize_unit_rgb(intrinsic)
    }

    /// Visualizes one uncertainty map with percentile saturation.
    ///
    /// # Errors
    ///
    /// Returns an error when channels, percentile, values, or output construction are invalid.
    pub fn visualize_uncertainty(
        uncertainty: &DenseMapPostprocessOutput,
        saturation_percentile: f32,
    ) -> Result<ImageFrame, ImageProcessorError> {
        marigold_visualize_uncertainty(uncertainty, saturation_percentile)
    }
}

fn validate_marigold_config(
    config: &MarigoldImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    if config.vae_scale_factor == 0 {
        return Err(TransformError::InvalidScaleFactor(config.vae_scale_factor).into());
    }
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct MarigoldCanonicalSample {
    values: Vec<f32>,
    size: ImageSize,
}

#[derive(Clone, Debug)]
struct MarigoldUnpaddedSample {
    size: ImageSize,
    values: Vec<f32>,
}

fn marigold_processor_output(preprocess: MarigoldPreprocessOutput) -> ProcessorOutput {
    let batch = preprocess
        .pixel_values
        .shape()
        .first()
        .copied()
        .unwrap_or(1);
    let mut output = ProcessorOutput::from_pixel_values(preprocess.pixel_values);
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(vec![preprocess.original_resolution; batch]),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(vec![preprocess.reshaped_input_size; batch]),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("padding"),
        ProcessorMetadataValue::Shape(vec![
            preprocess.padding.top,
            preprocess.padding.right,
            preprocess.padding.bottom,
            preprocess.padding.left,
        ]),
    );
    output
}

fn marigold_preprocess_images(
    config: &MarigoldImageProcessorConfig,
    images: &[ImageFrame],
    processing_resolution: Option<usize>,
) -> Result<MarigoldPreprocessOutput, ImageProcessorError> {
    if images.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut samples = Vec::with_capacity(images.len());
    let mut original_resolution = None;
    for image in images {
        let sample = marigold_canonical_sample(image, config)?;
        if let Some(expected) = original_resolution {
            if sample.size != expected {
                return Err(ImageProcessorError::IncompatibleBatchShapes);
            }
        } else {
            original_resolution = Some(sample.size);
        }
        samples.push(sample);
    }

    let original_resolution = original_resolution.ok_or(ImageProcessorError::EmptyBatch)?;
    let reshaped_input_size = marigold_processing_size(original_resolution, processing_resolution)?;
    let multiples = ImageSize::new(config.vae_scale_factor, config.vae_scale_factor)?;
    let pad_plan = pad_to_multiple_plan(reshaped_input_size, multiples)?;
    let values_len = checked_spatial_pad_mul(samples.len(), PixelFormat::Rgb8.channels())
        .and_then(|value| checked_spatial_pad_mul(value, pad_plan.padded_size.height))
        .and_then(|value| checked_spatial_pad_mul(value, pad_plan.padded_size.width))?;
    let mut values = vec![0.0; values_len];

    for (batch, sample) in samples.into_iter().enumerate() {
        let resized = marigold_resize_nchw_sample(
            sample.values,
            sample.size,
            reshaped_input_size,
            PixelFormat::Rgb8.channels(),
            config.resample,
        )?;
        marigold_copy_replicate_nchw_sample(
            &resized,
            &mut values,
            batch,
            PixelFormat::Rgb8.channels(),
            reshaped_input_size,
            pad_plan.padded_size,
        );
    }

    let tensor = Tensor::new(
        TensorData::F32(values),
        vec![
            images.len(),
            PixelFormat::Rgb8.channels(),
            pad_plan.padded_size.height,
            pad_plan.padded_size.width,
        ],
        Layout::NCHW,
    )
    .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
    .map_err(ImageProcessorError::Tensor)?;
    let tensor = match config.output_layout {
        ImageLayout::ChannelsHeightWidth => tensor,
        ImageLayout::HeightWidthChannels => tensor.to_layout(Layout::NHWC)?,
    };

    Ok(MarigoldPreprocessOutput {
        pixel_values: tensor,
        padding: pad_plan.padding,
        original_resolution,
        reshaped_input_size,
    })
}

fn marigold_canonical_sample(
    image: &ImageFrame,
    config: &MarigoldImageProcessorConfig,
) -> Result<MarigoldCanonicalSample, ImageProcessorError> {
    let rgb = if image.pixel_format() == PixelFormat::Rgb8 {
        Cow::Borrowed(image)
    } else {
        Cow::Owned(convert_frame_pixel_format(image, PixelFormat::Rgb8)?)
    };
    let size = ImageSize::new(rgb.height(), rgb.width())?;
    let channels = PixelFormat::Rgb8.channels();
    let values_len = checked_spatial_pad_mul(channels, size.height)
        .and_then(|value| checked_spatial_pad_mul(value, size.width))?;
    let mut values = Vec::with_capacity(values_len);

    for channel in 0..channels {
        for y in 0..size.height {
            for x in 0..size.width {
                let index = (y * size.width + x) * channels + channel;
                values.push(f32::from(rgb.data()[index]) * DEFAULT_RESCALE_FACTOR);
            }
        }
    }

    if config.do_range_check {
        for value in &values {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(TransformError::InvalidImageValue(*value).into());
            }
        }
    }

    if config.do_normalize {
        for value in &mut values {
            *value = 2.0 * *value - 1.0;
        }
    }

    Ok(MarigoldCanonicalSample { values, size })
}

fn marigold_processing_size(
    source: ImageSize,
    processing_resolution: Option<usize>,
) -> Result<ImageSize, ImageProcessorError> {
    let Some(processing_resolution) = processing_resolution.filter(|value| *value > 0) else {
        return Ok(source);
    };
    let max_orig = source.height.max(source.width);
    let height = source
        .height
        .checked_mul(processing_resolution)
        .ok_or(TransformError::ImageSizeOverflow)?
        / max_orig;
    let width = source
        .width
        .checked_mul(processing_resolution)
        .ok_or(TransformError::ImageSizeOverflow)?
        / max_orig;
    ImageSize::new(height, width).map_err(ImageProcessorError::Transform)
}

fn marigold_resize_nchw_sample(
    values: Vec<f32>,
    source_size: ImageSize,
    target_size: ImageSize,
    channels: usize,
    filter: ResizeFilter,
) -> Result<Vec<f32>, ImageProcessorError> {
    if source_size == target_size {
        return Ok(values);
    }
    let source_pixels = checked_spatial_pad_mul(source_size.height, source_size.width)?;
    if values.len() != source_pixels * channels {
        return Err(TransformError::InvalidBufferLength {
            expected: source_pixels * channels,
            actual: values.len(),
        }
        .into());
    }

    let target_pixels = checked_spatial_pad_mul(target_size.height, target_size.width)?;
    let mut resized = Vec::with_capacity(target_pixels * channels);
    for channel in 0..channels {
        let start = channel * source_pixels;
        let plane = &values[start..start + source_pixels];
        let resized_plane = match filter {
            ResizeFilter::Bilinear => {
                crate::transforms::resize_f32_image_bilinear(plane, source_size, target_size)?
            }
            ResizeFilter::Nearest => marigold_resize_f32_nearest(plane, source_size, target_size)?,
            ResizeFilter::Bicubic | ResizeFilter::Lanczos => {
                return Err(ImageProcessorError::UnsupportedProcessorOption { field: "resample" });
            }
        };
        resized.extend(resized_plane);
    }
    Ok(resized)
}

fn marigold_resize_f32_nearest(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, ImageProcessorError> {
    let expected = checked_spatial_pad_mul(source_size.height, source_size.width)?;
    if values.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: values.len(),
        }
        .into());
    }
    let target_pixels = checked_spatial_pad_mul(target_size.height, target_size.width)?;
    let mut output = Vec::with_capacity(target_pixels);
    for y in 0..target_size.height {
        let source_y = y * source_size.height / target_size.height;
        for x in 0..target_size.width {
            let source_x = x * source_size.width / target_size.width;
            let value = values[source_y * source_size.width + source_x];
            if !value.is_finite() {
                return Err(TransformError::InvalidImageValue(value).into());
            }
            output.push(value);
        }
    }
    Ok(output)
}

fn marigold_copy_replicate_nchw_sample(
    source: &[f32],
    target: &mut [f32],
    batch: usize,
    channels: usize,
    source_size: ImageSize,
    target_size: ImageSize,
) {
    let source_pixels = source_size.height * source_size.width;
    for channel in 0..channels {
        let source_channel_offset = channel * source_pixels;
        for y in 0..target_size.height {
            let source_y = y.min(source_size.height - 1);
            for x in 0..target_size.width {
                let source_x = x.min(source_size.width - 1);
                let target_index = (((batch * channels + channel) * target_size.height + y)
                    * target_size.width)
                    + x;
                let source_index = source_channel_offset + source_y * source_size.width + source_x;
                target[target_index] = source[source_index];
            }
        }
    }
}

fn marigold_unpadded_interleaved_samples(
    tensor: &Tensor,
    channels: usize,
    padding: Padding,
) -> Result<Vec<MarigoldUnpaddedSample>, ImageProcessorError> {
    if channels == 0 {
        return Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: tensor.data().len(),
        }
        .into());
    }
    if padding.top != 0 || padding.left != 0 {
        return Err(ImageProcessorError::UnsupportedProcessorOption { field: "padding" });
    }

    let tensor = marigold_tensor_as_nchw(tensor)?;
    if tensor.channels() != channels {
        return Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: tensor.channels(),
        }
        .into());
    }
    if padding.bottom >= tensor.height() || padding.right >= tensor.width() {
        return Err(TransformError::CropTooLarge {
            source_size: ImageSize::new(tensor.height(), tensor.width())?,
            target_size: ImageSize::new(
                tensor.height().saturating_sub(padding.bottom),
                tensor.width().saturating_sub(padding.right),
            )
            .unwrap_or(ImageSize {
                height: 1,
                width: 1,
            }),
        }
        .into());
    }

    let unpadded_size = ImageSize::new(
        tensor.height() - padding.bottom,
        tensor.width() - padding.right,
    )?;
    let batch = tensor.shape()[0];
    let values = tensor.data().to_vec::<f32>();
    let padded_pixels = tensor.height() * tensor.width();
    let unpadded_pixels = checked_spatial_pad_mul(unpadded_size.height, unpadded_size.width)?;
    let mut samples = Vec::with_capacity(batch);

    for sample_index in 0..batch {
        let mut sample_values = Vec::with_capacity(unpadded_pixels * channels);
        for y in 0..unpadded_size.height {
            for x in 0..unpadded_size.width {
                for channel in 0..channels {
                    let index = ((sample_index * channels + channel) * padded_pixels)
                        + y * tensor.width()
                        + x;
                    sample_values.push(values[index]);
                }
            }
        }
        samples.push(MarigoldUnpaddedSample {
            size: unpadded_size,
            values: sample_values,
        });
    }

    Ok(samples)
}

fn marigold_tensor_as_nchw(tensor: &Tensor) -> Result<Tensor, ImageProcessorError> {
    match tensor.layout() {
        Layout::NCHW => Ok(tensor.clone()),
        Layout::NHWC => tensor
            .to_layout(Layout::NCHW)
            .map_err(ImageProcessorError::Tensor),
        Layout::CHW => Tensor::new(
            tensor.data().clone(),
            vec![1, tensor.channels(), tensor.height(), tensor.width()],
            Layout::NCHW,
        )
        .map_err(ImageProcessorError::Tensor),
        Layout::HWC => {
            let chw = tensor.to_layout(Layout::CHW)?;
            Tensor::new(
                chw.data().clone(),
                vec![1, chw.channels(), chw.height(), chw.width()],
                Layout::NCHW,
            )
            .map_err(ImageProcessorError::Tensor)
        }
        layout => Err(ImageProcessorError::UnsupportedLayout(layout)),
    }
}

fn marigold_export_depth_to_16bit(
    depth: &DepthMapPostprocessOutput,
    val_min: f32,
    val_max: f32,
) -> Result<DepthMapU16, ImageProcessorError> {
    validate_marigold_value_range(val_min, val_max)?;
    let values = depth
        .values()
        .iter()
        .copied()
        .map(|value| normalized_range_value(value, val_min, val_max) * u16::MAX as f32)
        .map(|value| value as u16)
        .collect::<Vec<_>>();
    DepthMapU16::new(depth.size().width, depth.size().height, values)
}

fn marigold_visualize_depth(
    depth: &DepthMapPostprocessOutput,
    val_min: f32,
    val_max: f32,
) -> Result<ImageFrame, ImageProcessorError> {
    validate_marigold_value_range(val_min, val_max)?;
    let mut data = Vec::with_capacity(depth.values().len() * PixelFormat::Rgb8.channels());
    for value in depth.values().iter().copied() {
        data.extend_from_slice(&marigold_spectral_rgb(normalized_range_value(
            value, val_min, val_max,
        )));
    }
    ImageFrame::new(
        depth.size().width,
        depth.size().height,
        PixelFormat::Rgb8,
        data,
    )
    .map_err(ImageProcessorError::Media)
}

fn marigold_visualize_normals(
    normals: &DenseMapPostprocessOutput,
    flips: [bool; 3],
) -> Result<ImageFrame, ImageProcessorError> {
    if normals.channels() != 3 {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 3,
            actual: normals.channels(),
        }
        .into());
    }
    let mut data = Vec::with_capacity(normals.values().len());
    for pixel in normals.values().chunks_exact(3) {
        for (channel, value) in pixel.iter().copied().enumerate() {
            let value = if flips[channel] { -value } else { value };
            data.push(float_to_u8((value + 1.0) * 0.5 * 255.0)?);
        }
    }
    ImageFrame::new(
        normals.size().width,
        normals.size().height,
        PixelFormat::Rgb8,
        data,
    )
    .map_err(ImageProcessorError::Media)
}

fn marigold_visualize_unit_rgb(
    intrinsic: &DenseMapPostprocessOutput,
) -> Result<ImageFrame, ImageProcessorError> {
    if intrinsic.channels() != 3 {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 3,
            actual: intrinsic.channels(),
        }
        .into());
    }
    let data = intrinsic
        .values()
        .iter()
        .copied()
        .map(|value| float_to_u8(value * 255.0))
        .collect::<Result<Vec<_>, _>>()?;
    ImageFrame::new(
        intrinsic.size().width,
        intrinsic.size().height,
        PixelFormat::Rgb8,
        data,
    )
    .map_err(ImageProcessorError::Media)
}

fn marigold_visualize_uncertainty(
    uncertainty: &DenseMapPostprocessOutput,
    saturation_percentile: f32,
) -> Result<ImageFrame, ImageProcessorError> {
    if !saturation_percentile.is_finite() || !(0.0..=100.0).contains(&saturation_percentile) {
        return Err(TransformError::InvalidFloatScaleFactor(saturation_percentile).into());
    }
    if !matches!(uncertainty.channels(), 1 | 3) {
        return Err(TransformError::InvalidChannelDataLength {
            channels: 1,
            actual: uncertainty.channels(),
        }
        .into());
    }
    for value in uncertainty.values().iter().copied() {
        if !value.is_finite() || value < 0.0 {
            return Err(TransformError::InvalidImageValue(value).into());
        }
    }
    let saturation = percentile_linear(uncertainty.values(), saturation_percentile)?;
    if saturation <= 0.0 {
        return Err(TransformError::InvalidFloatScaleFactor(saturation).into());
    }
    let data = uncertainty
        .values()
        .iter()
        .copied()
        .map(|value| float_to_u8((value * 255.0 / saturation).clamp(0.0, 255.0)))
        .collect::<Result<Vec<_>, _>>()?;
    let format = if uncertainty.channels() == 1 {
        PixelFormat::Luma8
    } else {
        PixelFormat::Rgb8
    };
    ImageFrame::new(
        uncertainty.size().width,
        uncertainty.size().height,
        format,
        data,
    )
    .map_err(ImageProcessorError::Media)
}

fn validate_marigold_value_range(val_min: f32, val_max: f32) -> Result<(), ImageProcessorError> {
    if val_min.is_finite() && val_max.is_finite() && val_max > val_min {
        Ok(())
    } else {
        Err(TransformError::InvalidFloatScaleFactor(val_max - val_min).into())
    }
}

fn normalized_range_value(value: f32, val_min: f32, val_max: f32) -> f32 {
    ((value - val_min) / (val_max - val_min)).clamp(0.0, 1.0)
}

fn float_to_u8(value: f32) -> Result<u8, ImageProcessorError> {
    if !value.is_finite() {
        return Err(TransformError::InvalidImageValue(value).into());
    }
    Ok(value.clamp(0.0, 255.0) as u8)
}

fn percentile_linear(values: &[f32], percentile: f32) -> Result<f32, ImageProcessorError> {
    if values.is_empty() {
        return Err(TransformError::InvalidBufferLength {
            expected: 1,
            actual: 0,
        }
        .into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let rank = f64::from(percentile) / 100.0 * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        return Ok(sorted[low]);
    }
    let weight = (rank - low as f64) as f32;
    Ok(sorted[low] * (1.0 - weight) + sorted[high] * weight)
}

const MARIGOLD_SPECTRAL: [[f32; 3]; 11] = [
    [0.61960787, 0.003921569, 0.25882354],
    [0.8352941, 0.24313726, 0.30980393],
    [0.95686275, 0.42745098, 0.2627451],
    [0.99215686, 0.68235296, 0.38039216],
    [0.99607843, 0.8784314, 0.54509807],
    [1.0, 1.0, 0.7490196],
    [0.9019608, 0.9607843, 0.59607846],
    [0.67058825, 0.8666667, 0.6431373],
    [0.4, 0.7607843, 0.64705884],
    [0.19607843, 0.53333336, 0.7411765],
    [0.36862746, 0.30980393, 0.63529414],
];

fn marigold_spectral_rgb(value: f32) -> [u8; 3] {
    let value = value.clamp(0.0, 1.0);
    let position = value * (MARIGOLD_SPECTRAL.len() - 1) as f32;
    let left = position.floor() as usize;
    let right = (left + 1).min(MARIGOLD_SPECTRAL.len() - 1);
    let weight = position - left as f32;
    let mut rgb = [0; 3];
    for (channel, byte) in rgb.iter_mut().enumerate() {
        let left_color = MARIGOLD_SPECTRAL[left][channel];
        let right_color = MARIGOLD_SPECTRAL[right][channel];
        *byte = ((left_color * (1.0 - weight) + right_color * weight) * 255.0) as u8;
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_f32_values_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "expected {actual} to be within {tolerance} of {expected}"
            );
        }
    }
    #[test]
    fn marigold_processor_normalizes_and_replicate_pads_to_vae_multiple() {
        let processor = MarigoldImageProcessor::new(MarigoldImageProcessorConfig {
            vae_scale_factor: 4,
            ..Default::default()
        })
        .unwrap();
        let frame = ImageFrame::new(
            3,
            2,
            PixelFormat::Rgb8,
            vec![
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
            ],
        )
        .unwrap();

        let output = processor.preprocess_image(&frame).unwrap();

        assert_eq!(output.pixel_values().shape(), [1, 3, 4, 4]);
        assert_eq!(output.pixel_values().layout(), Layout::NCHW);
        assert_eq!(output.padding(), Padding::new(0, 1, 2, 0));
        assert_eq!(output.original_resolution(), ImageSize::new(2, 3).unwrap());
        assert_eq!(output.reshaped_input_size(), ImageSize::new(2, 3).unwrap());
        let values = output.pixel_values().data().to_vec::<f32>();
        assert_f32_values_close(
            &values[..4],
            &[-1.0, -0.764_705_9, -0.529_411_8, -0.529_411_8],
            1e-6,
        );
        assert_f32_values_close(
            &values[12..16],
            &[-0.294_117_63, -0.058_823_526, 0.176_470_64, 0.176_470_64],
            1e-6,
        );

        let typed = processor
            .preprocess_image_output(&frame, Some(4))
            .expect("typed Marigold output should build");
        assert_eq!(typed.pixel_values().unwrap().shape(), [1, 3, 4, 4]);
        assert_eq!(
            typed.reshaped_input_sizes().unwrap(),
            &[ImageSize::new(2, 4).unwrap()]
        );
        assert!(matches!(
            typed.metadata_value(&ProcessorMetadataName::other("padding")),
            Some(ProcessorMetadataValue::Shape(shape)) if shape.as_slice() == [0, 0, 2, 0]
        ));
    }

    #[test]
    fn marigold_processor_restores_depth_normals_and_uncertainty_outputs() {
        let processor = MarigoldImageProcessor::new(MarigoldImageProcessorConfig {
            vae_scale_factor: 4,
            ..Default::default()
        })
        .unwrap();
        let frame = ImageFrame::new(3, 2, PixelFormat::Rgb8, vec![0; 3 * 2 * 3]).unwrap();
        let preprocess = processor.preprocess_image(&frame).unwrap();

        let depth_tensor = Tensor::new(
            TensorData::F32(vec![
                0.0, 0.2, 0.4, 9.0, //
                0.6, 0.8, 1.0, 9.0, //
                9.0, 9.0, 9.0, 9.0, //
                9.0, 9.0, 9.0, 9.0,
            ]),
            vec![1, 1, 4, 4],
            Layout::NCHW,
        )
        .unwrap();
        let depth = processor
            .postprocess_depth(&depth_tensor, &preprocess, RecipeDepthUnit::Relative)
            .unwrap();

        assert_eq!(depth[0].size(), ImageSize::new(2, 3).unwrap());
        assert_f32_values_close(depth[0].values(), &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 1e-6);
        let exported = MarigoldImageProcessor::export_depth_to_16bit(&depth[0], 0.0, 1.0).unwrap();
        assert_eq!(exported.size(), ImageSize::new(2, 3).unwrap());
        assert_eq!(exported.data()[0], 0);
        assert_eq!(exported.data()[5], u16::MAX);
        let depth_visual = MarigoldImageProcessor::visualize_depth(&depth[0], 0.0, 1.0).unwrap();
        assert_eq!(depth_visual.pixel_format(), PixelFormat::Rgb8);
        assert_eq!(depth_visual.width(), 3);
        assert_eq!(depth_visual.height(), 2);

        let normals_tensor = Tensor::new(
            TensorData::F32(vec![
                -1.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, -1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ]),
            vec![1, 3, 4, 4],
            Layout::NCHW,
        )
        .unwrap();
        let normals = processor
            .postprocess_normals(&normals_tensor, &preprocess)
            .unwrap();

        assert_eq!(normals[0].task(), RecipeDenseMapTask::SurfaceNormal);
        assert_eq!(normals[0].channels(), 3);
        assert_f32_values_close(&normals[0].values()[..3], &[-1.0, 0.0, 1.0], 1e-6);
        let normal_visual =
            MarigoldImageProcessor::visualize_normals(&normals[0], false, false, false).unwrap();
        assert_eq!(&normal_visual.data()[..3], &[0, 127, 255]);

        let uncertainty = processor
            .postprocess_uncertainty(&depth_tensor, &preprocess)
            .unwrap();
        let uncertainty_visual =
            MarigoldImageProcessor::visualize_uncertainty(&uncertainty[0], 100.0).unwrap();
        assert_eq!(uncertainty_visual.pixel_format(), PixelFormat::Luma8);
        assert_eq!(&uncertainty_visual.data()[..3], &[0, 51, 102]);
    }
}
