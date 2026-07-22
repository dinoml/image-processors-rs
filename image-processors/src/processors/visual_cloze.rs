//! Diffusers VisualCloze image-grid processor.

use serde::{Deserialize, Serialize};

use super::{
    resize_decision_for_parts, validate_positive_dimension, validate_vae_config,
    video_layout_for_image_layout, VaeImageProcessor, VaeImageProcessorConfig,
    VISUAL_CLOZE_RESOLUTION, VISUAL_CLOZE_VAE_LATENT_CHANNELS, VISUAL_CLOZE_VAE_SCALE_FACTOR,
};
use crate::image::{
    BatchExecution, ImageProcessor, ImageProcessorConfig, ImageProcessorError,
    ImageProcessorOptions,
};
use crate::media::{ImageDecodeBackend, ImageFrame, PixelFormat};
use crate::tensor::{ImageLayout, Layout, Tensor, TensorData, VideoLayout};
use crate::transforms::{
    convert_frame_pixel_format, crop_frame, nested_image_grid_metadata, resize_center_crop_plan,
    resize_frame_with_decision, ImageSize, NestedImageGridMetadata, ResizeDecision, ResizeFilter,
    ResizeMode, ResizeParity, ResizeRounding, TransformError,
};

/// Nested tensor rows returned by Diffusers VisualCloze preprocessing.
pub type VisualClozeNestedTensors = Vec<Vec<Tensor>>;

/// Consumed parts for Diffusers VisualCloze preprocessing output.
pub type VisualClozePreprocessParts = (
    VisualClozeNestedTensors,
    VisualClozeNestedTensors,
    Vec<usize>,
    NestedImageGridMetadata,
);

/// Diffusers VisualCloze preprocessing output.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualClozePreprocessOutput {
    images: VisualClozeNestedTensors,
    masks: VisualClozeNestedTensors,
    target_position: Vec<usize>,
    metadata: NestedImageGridMetadata,
}

impl VisualClozePreprocessOutput {
    /// Returns processed image tensors in row-column order.
    pub fn images(&self) -> &[Vec<Tensor>] {
        &self.images
    }

    /// Returns generated mask tensors in row-column order.
    pub fn masks(&self) -> &[Vec<Tensor>] {
        &self.masks
    }

    /// Returns the upstream-style query-row target flags.
    pub fn target_position(&self) -> &[usize] {
        &self.target_position
    }

    /// Returns validated nested image-grid metadata.
    pub fn metadata(&self) -> &NestedImageGridMetadata {
        &self.metadata
    }

    /// Consumes this output into nested image tensors, nested masks, target
    /// flags, and grid metadata.
    pub fn into_parts(self) -> VisualClozePreprocessParts {
        (self.images, self.masks, self.target_position, self.metadata)
    }
}

/// Configuration for Diffusers VisualCloze image-grid preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VisualClozeProcessorConfig {
    /// Target per-cell area baseline used before nesting images.
    pub resolution: usize,
    /// Whether images should be resized to VAE-scale multiples during tensor conversion.
    pub do_resize: bool,
    /// VAE scale factor used by the underlying VAE processor.
    pub vae_scale_factor: usize,
    /// Channel count that represents VisualCloze VAE latent tensors.
    pub vae_latent_channels: usize,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used by VAE-style preprocessing.
    pub resample: ResizeFilter,
    /// Resize parity policy used by VAE-style preprocessing.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
    /// Optional pixel-format conversion before tensor conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Whether image values should be normalized from `[0, 1]` to `[-1, 1]`.
    pub do_normalize: bool,
    /// Whether image values should be thresholded to `0.0` or `1.0`.
    pub do_binarize: bool,
}

impl Default for VisualClozeProcessorConfig {
    fn default() -> Self {
        Self {
            resolution: VISUAL_CLOZE_RESOLUTION,
            do_resize: true,
            vae_scale_factor: VISUAL_CLOZE_VAE_SCALE_FACTOR,
            vae_latent_channels: VISUAL_CLOZE_VAE_LATENT_CHANNELS,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Lanczos,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: None,
            do_normalize: true,
            do_binarize: false,
        }
    }
}

impl VisualClozeProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the video output axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        video_layout_for_image_layout(self.output_layout)
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this config into the underlying VAE processor config.
    pub fn vae_image_processor_config(&self) -> VaeImageProcessorConfig {
        self.vae_image_processor_config_inner()
    }

    /// Converts this family config into the generic image processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        self.vae_image_processor_config_inner()
            .image_processor_config()
    }

    pub(super) fn vae_image_processor_config_inner(&self) -> VaeImageProcessorConfig {
        VaeImageProcessorConfig {
            do_resize: self.do_resize,
            vae_scale_factor: self.vae_scale_factor,
            vae_latent_channels: self.vae_latent_channels,
            output_layout: self.output_layout,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: self.pixel_format,
            do_normalize: self.do_normalize,
            do_binarize: self.do_binarize,
        }
    }
}

/// Diffusers VisualCloze image processor.
#[derive(Clone, Debug)]
pub struct VisualClozeProcessor {
    config: VisualClozeProcessorConfig,
    processor: VaeImageProcessor,
}

impl VisualClozeProcessor {
    /// Creates a processor from a validated VisualCloze configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when VisualCloze dimensions are invalid or the derived
    /// VAE image processor config is invalid.
    pub fn new(config: VisualClozeProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_visual_cloze_config(&config)?;
        let processor = VaeImageProcessor::new(config.vae_image_processor_config_inner())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's VisualCloze configuration.
    pub fn config(&self) -> &VisualClozeProcessorConfig {
        &self.config
    }

    /// Returns the VAE-style processor used for VisualCloze tensor conversion.
    pub fn vae_processor(&self) -> &VaeImageProcessor {
        &self.processor
    }

    /// Returns the underlying generic image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        self.processor.image_processor()
    }

    /// Returns the VisualCloze layout prompt for a row-column grid.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero or the cell count
    /// overflows.
    pub fn layout_prompt(rows: usize, columns: usize) -> Result<String, ImageProcessorError> {
        validate_positive_dimension("rows", rows)?;
        validate_positive_dimension("columns", columns)?;
        let cells = rows
            .checked_mul(columns)
            .ok_or(TransformError::ImageSizeOverflow)?;
        Ok(format!(
            "A grid layout with {rows} rows and {columns} columns, displaying {cells} images arranged side by side."
        ))
    }

    /// Computes the per-row VisualCloze resize target for a source image.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are invalid or arithmetic overflows.
    pub fn target_size_for_image(
        &self,
        image: &ImageFrame,
    ) -> Result<ImageSize, ImageProcessorError> {
        visual_cloze_target_size_for_image_size(
            ImageSize::new(image.height(), image.width())?,
            self.config.resolution,
            self.config.vae_scale_factor,
        )
    }

    /// Applies VisualCloze's VAE resize-to-cover and center-crop helper.
    ///
    /// The returned frame is RGB, matching the upstream helper's RGB canvas.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, cropping, RGB conversion, or frame
    /// construction fails.
    pub fn resize_center_crop_image(
        &self,
        image: &ImageFrame,
        target: ImageSize,
    ) -> Result<ImageFrame, ImageProcessorError> {
        visual_cloze_resize_center_crop(image, target, self.config.resample)
    }

    /// Preprocesses a nested VisualCloze image grid.
    ///
    /// The final row marks target cells with `None`; non-target cells contain
    /// decoded images. Output image tensors and masks preserve row-column
    /// nesting, matching the upstream processor.
    ///
    /// # Errors
    ///
    /// Returns an error when the grid is empty, row lengths are inconsistent,
    /// a row has no source image for sizing, resizing fails, or tensor
    /// conversion fails.
    pub fn preprocess_image_grid(
        &self,
        input_images: &[Vec<Option<ImageFrame>>],
    ) -> Result<VisualClozePreprocessOutput, ImageProcessorError> {
        visual_cloze_preprocess_grid(&self.config, &self.processor, input_images)
    }

    /// Preprocesses one image for VisualCloze's upsampling stage.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are invalid, resizing fails, or tensor
    /// conversion fails.
    pub fn preprocess_image_upsampling(
        &self,
        image: &ImageFrame,
        target: ImageSize,
    ) -> Result<VisualClozePreprocessOutput, ImageProcessorError> {
        let resized = resize_frame_with_decision(
            image,
            target,
            self.config.resize_decision(),
            ResizeMode::Default,
        )
        .map_err(ImageProcessorError::Transform)?;
        let tensor = self.processor.preprocess_image_with_options(
            &resized,
            ImageProcessorOptions {
                height: Some(target.height),
                width: Some(target.width),
                resize_mode: None,
            },
        )?;
        let mask = visual_cloze_upsampling_mask_tensor(target)?;
        let metadata = nested_image_grid_metadata(vec![vec![target]], vec![vec![true]])
            .map_err(ImageProcessorError::Transform)?;
        Ok(VisualClozePreprocessOutput {
            images: vec![vec![tensor]],
            masks: vec![vec![mask]],
            target_position: vec![1],
            metadata,
        })
    }
}

pub(super) fn validate_visual_cloze_config(
    config: &VisualClozeProcessorConfig,
) -> Result<(), ImageProcessorError> {
    validate_positive_dimension("resolution", config.resolution)?;
    validate_vae_config(&config.vae_image_processor_config_inner())
}

fn visual_cloze_preprocess_grid(
    config: &VisualClozeProcessorConfig,
    processor: &VaeImageProcessor,
    input_images: &[Vec<Option<ImageFrame>>],
) -> Result<VisualClozePreprocessOutput, ImageProcessorError> {
    let columns = visual_cloze_validate_grid(input_images)?;
    let rows = input_images.len();
    let query_row = rows - 1;
    let mut processed_frames: Vec<Vec<ImageFrame>> = Vec::with_capacity(rows);
    let mut resize_sizes = Vec::with_capacity(rows);
    let mut target_position = Vec::with_capacity(columns);

    for (row_index, row) in input_images.iter().enumerate() {
        let resize_size = row
            .iter()
            .find_map(|cell| {
                cell.as_ref()
                    .map(|image| ImageSize::new(image.height(), image.width()))
            })
            .transpose()?
            .map(|size| {
                visual_cloze_target_size_for_image_size(
                    size,
                    config.resolution,
                    config.vae_scale_factor,
                )
            })
            .transpose()?;
        resize_sizes.push(resize_size);

        let fallback = ImageSize::new(config.resolution, config.resolution)?;
        let target = resize_size.unwrap_or(fallback);
        let mut processed_row = Vec::with_capacity(columns);
        for cell in row {
            match cell {
                Some(image) => {
                    processed_row.push(visual_cloze_resize_center_crop(
                        image,
                        target,
                        config.resample,
                    )?);
                    if row_index == query_row {
                        target_position.push(0);
                    }
                }
                None => {
                    processed_row.push(visual_cloze_blank_frame(target)?);
                    if row_index == query_row {
                        target_position.push(1);
                    }
                }
            }
        }
        processed_frames.push(processed_row);
    }

    if target_position.len() > 1 && target_position.iter().copied().sum::<usize>() > 1 {
        let mut new_width = resize_sizes[query_row]
            .map(|size| size.width)
            .unwrap_or(config.resolution);
        for row in &mut processed_frames {
            for image in row {
                let source = ImageSize::new(image.height(), image.width())?;
                let new_height =
                    ((source.height as f64) * (new_width as f64 / source.width as f64)) as usize;
                new_width = (new_width / 16) * 16;
                let new_height = (new_height / 16) * 16;
                let target = ImageSize::new(new_width, new_height)?;
                *image = visual_cloze_resize_center_crop(image, target, config.resample)?;
            }
        }
    }

    let image_sizes = processed_frames
        .iter()
        .map(|row| {
            row.iter()
                .map(|image| ImageSize::new(image.height(), image.width()))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_mask = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| row == query_row && target_position[column] == 1)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let metadata = nested_image_grid_metadata(image_sizes, target_mask)
        .map_err(ImageProcessorError::Transform)?;

    let mut images = Vec::with_capacity(rows);
    for row in &processed_frames {
        let mut tensor_row = Vec::with_capacity(columns);
        for image in row {
            tensor_row.push(processor.preprocess_image(image)?);
        }
        images.push(tensor_row);
    }

    let mut masks = Vec::with_capacity(rows);
    for (row_index, row) in images.iter().enumerate() {
        let first = row.first().ok_or(ImageProcessorError::EmptyBatch)?;
        let mask_size = ImageSize::new(first.height(), first.width())?;
        let mut mask_row = Vec::with_capacity(columns);
        for &position in target_position.iter().take(columns) {
            let fill = if row_index == query_row {
                position as i64
            } else {
                0
            };
            mask_row.push(visual_cloze_mask_tensor(mask_size, fill)?);
        }
        masks.push(mask_row);
    }

    Ok(VisualClozePreprocessOutput {
        images,
        masks,
        target_position,
        metadata,
    })
}

fn visual_cloze_validate_grid(
    input_images: &[Vec<Option<ImageFrame>>],
) -> Result<usize, ImageProcessorError> {
    let columns = input_images
        .first()
        .map(Vec::len)
        .ok_or(TransformError::EmptyNestedImageGrid)?;
    if columns == 0 {
        return Err(TransformError::EmptyNestedImageGrid.into());
    }
    for row in input_images {
        if row.len() != columns {
            return Err(TransformError::IncompatibleFrameSize {
                frame: "visual_cloze row",
                expected: ImageSize::new(1, columns)?,
                actual: ImageSize::new(1, row.len())?,
            }
            .into());
        }
    }
    if !input_images
        .last()
        .is_some_and(|row| row.iter().any(Option::is_none))
    {
        return Err(TransformError::EmptyNestedImageGrid.into());
    }
    Ok(columns)
}

fn visual_cloze_target_size_for_image_size(
    source: ImageSize,
    resolution: usize,
    vae_scale_factor: usize,
) -> Result<ImageSize, ImageProcessorError> {
    validate_positive_dimension("resolution", resolution)?;
    validate_positive_dimension("vae_scale_factor", vae_scale_factor)?;
    let divisible = vae_scale_factor
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let target_area = resolution
        .checked_mul(resolution)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let aspect_ratio = source.width as f64 / source.height as f64;
    let new_height = ((target_area as f64 / aspect_ratio).sqrt()) as usize;
    let new_width = ((new_height as f64) * aspect_ratio) as usize;
    ImageSize::new(
        (new_height / divisible).max(1) * divisible,
        (new_width / divisible).max(1) * divisible,
    )
    .map_err(ImageProcessorError::Transform)
}

fn visual_cloze_resize_center_crop(
    image: &ImageFrame,
    target: ImageSize,
    resample: ResizeFilter,
) -> Result<ImageFrame, ImageProcessorError> {
    let rgb = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
    let plan = resize_center_crop_plan(
        ImageSize::new(rgb.height(), rgb.width())?,
        target,
        ResizeRounding::Floor,
    )?;
    let resized = resize_frame_with_decision(
        &rgb,
        plan.resized_size,
        ResizeDecision::compatibility(resample),
        ResizeMode::Default,
    )?;
    crop_frame(&resized, plan.crop_box).map_err(ImageProcessorError::Transform)
}

fn visual_cloze_blank_frame(target: ImageSize) -> Result<ImageFrame, ImageProcessorError> {
    let len = target
        .height
        .checked_mul(target.width)
        .and_then(|pixels| pixels.checked_mul(PixelFormat::Rgb8.channels()))
        .ok_or(TransformError::ImageSizeOverflow)?;
    Ok(ImageFrame::new(
        target.width,
        target.height,
        PixelFormat::Rgb8,
        vec![0; len],
    )?)
}

fn visual_cloze_mask_tensor(size: ImageSize, fill: i64) -> Result<Tensor, ImageProcessorError> {
    let pixels = size
        .height
        .checked_mul(size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    Tensor::new(
        TensorData::I64(vec![fill; pixels]),
        vec![1, 1, size.height, size.width],
        Layout::NCHW,
    )
    .map_err(ImageProcessorError::Tensor)
}

fn visual_cloze_upsampling_mask_tensor(size: ImageSize) -> Result<Tensor, ImageProcessorError> {
    let pixels = size
        .height
        .checked_mul(size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    Tensor::new(
        TensorData::F32(vec![1.0; pixels]),
        vec![1, 1, size.height, size.width],
        Layout::NCHW,
    )
    .map_err(ImageProcessorError::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_cloze_processor_builds_nested_images_masks_and_metadata() {
        let processor = VisualClozeProcessor::new(VisualClozeProcessorConfig {
            resolution: 64,
            ..Default::default()
        })
        .unwrap();
        let context = ImageFrame::new(6, 4, PixelFormat::Rgb8, vec![64; 6 * 4 * 3]).unwrap();
        let context_second = ImageFrame::new(4, 4, PixelFormat::Rgb8, vec![96; 4 * 4 * 3]).unwrap();
        let query = ImageFrame::new(4, 6, PixelFormat::Rgb8, vec![128; 4 * 6 * 3]).unwrap();

        assert_eq!(
            processor.target_size_for_image(&context).unwrap(),
            ImageSize::new(32, 64).unwrap()
        );
        assert_eq!(
            VisualClozeProcessor::layout_prompt(2, 2).unwrap(),
            "A grid layout with 2 rows and 2 columns, displaying 4 images arranged side by side."
        );

        let output = processor
            .preprocess_image_grid(&[
                vec![Some(context), Some(context_second)],
                vec![Some(query), None],
            ])
            .unwrap();

        assert_eq!(output.target_position(), &[0, 1]);
        assert_eq!(output.metadata().rows, 2);
        assert_eq!(output.metadata().columns, 2);
        assert_eq!(
            output.metadata().target_mask,
            vec![vec![false, false], vec![false, true]]
        );
        assert_eq!(
            output.metadata().image_sizes,
            vec![
                vec![ImageSize::new(32, 64).unwrap(); 2],
                vec![ImageSize::new(64, 32).unwrap(); 2],
            ]
        );
        assert_eq!(output.images()[0][0].shape(), [1, 3, 32, 64]);
        assert_eq!(output.images()[1][0].shape(), [1, 3, 64, 32]);
        assert_eq!(output.masks()[0][0].shape(), [1, 1, 32, 64]);
        assert_eq!(output.masks()[1][1].shape(), [1, 1, 64, 32]);
        assert!(output.masks()[0][1]
            .data()
            .to_vec::<f32>()
            .iter()
            .all(|value| value.abs() < f32::EPSILON));
        assert!(output.masks()[1][1]
            .data()
            .to_vec::<f32>()
            .iter()
            .all(|value| (*value - 1.0).abs() < f32::EPSILON));

        let upsampling = processor
            .preprocess_image_upsampling(
                &ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![255; 2 * 2 * 3]).unwrap(),
                ImageSize::new(32, 32).unwrap(),
            )
            .unwrap();
        assert_eq!(upsampling.images()[0][0].shape(), [1, 3, 32, 32]);
        assert_eq!(upsampling.masks()[0][0].shape(), [1, 1, 32, 32]);
        assert!(matches!(
            upsampling.masks()[0][0].data(),
            TensorData::F32(_)
        ));
        assert_eq!(upsampling.target_position(), &[1]);
    }
}
