//! Nested VLM processor implementations.

use super::*;

/// Idefics3 image processor with split-image frames and attention masks.
#[derive(Clone, Debug)]
pub struct Idefics3ImageProcessor {
    config: Idefics3ImageProcessorConfig,
    processor: ImageProcessor,
}

impl Idefics3ImageProcessor {
    /// Creates a processor from a validated Idefics3 configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when split geometry or the derived generic processor
    /// config is invalid.
    pub fn new(config: Idefics3ImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_idefics3_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &Idefics3ImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic frame tensor processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, split-frame preparation, tensor
    /// conversion, or nested padding fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as Idefics3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, split-frame preparation, tensor
    /// conversion, or nested padding fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, split-frame
    /// preparation fails, or nested padding fails.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images(&images)
    }

    /// Loads image paths and preprocesses them as Idefics3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, split-frame
    /// preparation fails, or nested padding fails.
    pub fn open_batch_output<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images_output(&images)
    }

    /// Preprocesses one decoded image frame as an Idefics3 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when split-frame preparation, tensor conversion, or
    /// nested padding fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        preprocess_idefics3_samples(
            &self.config,
            &self.processor,
            &[std::slice::from_ref(image)],
        )
        .map(|batch| batch.pixel_values)
    }

    /// Preprocesses decoded image frames as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, split-frame preparation fails,
    /// tensor conversion fails, or nested padding fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        let samples = images.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        preprocess_idefics3_samples(&self.config, &self.processor, &samples)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses nested image samples as an Idefics3 tensor.
    ///
    /// The outer slice is the batch and each inner vector is the original
    /// images attached to that sample, matching Transformers' nested VLM input.
    ///
    /// # Errors
    ///
    /// Returns an error when no actual images are provided, split-frame
    /// preparation fails, tensor conversion fails, or nested padding fails.
    pub fn preprocess_image_samples(
        &self,
        samples: &[Vec<ImageFrame>],
    ) -> Result<Tensor, ImageProcessorError> {
        let sample_refs = nested_sample_refs(samples);
        preprocess_idefics3_samples(&self.config, &self.processor, &sample_refs)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses one decoded image frame as Idefics3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when split-frame preparation, tensor conversion, or
    /// nested padding fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_idefics3_samples_output(
            &self.config,
            &self.processor,
            &[std::slice::from_ref(image)],
        )
    }

    /// Preprocesses decoded image frames as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, split-frame preparation fails,
    /// tensor conversion fails, or nested padding fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let samples = images.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        preprocess_idefics3_samples_output(&self.config, &self.processor, &samples)
    }

    /// Preprocesses nested image samples as Idefics3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when no actual images are provided, split-frame
    /// preparation fails, tensor conversion fails, or nested padding fails.
    pub fn preprocess_image_samples_output(
        &self,
        samples: &[Vec<ImageFrame>],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let sample_refs = nested_sample_refs(samples);
        preprocess_idefics3_samples_output(&self.config, &self.processor, &sample_refs)
    }
}

/// Gemma3 image processor with optional pan-and-scan crop expansion.
#[derive(Clone, Debug)]
pub struct Gemma3ImageProcessor {
    config: Gemma3ImageProcessorConfig,
    processor: ImageProcessor,
}

impl Gemma3ImageProcessor {
    /// Creates a processor from a validated Gemma3 configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when pan-and-scan geometry or the derived generic
    /// processor config is invalid.
    pub fn new(config: Gemma3ImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_gemma3_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &Gemma3ImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic frame tensor processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, pan-and-scan preparation, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as Gemma3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, pan-and-scan preparation, or tensor
    /// conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as a flat Gemma3 frame batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, pan-and-scan
    /// preparation fails, or tensor conversion fails.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images(&images)
    }

    /// Loads image paths and preprocesses them as Gemma3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, pan-and-scan
    /// preparation fails, or tensor conversion fails.
    pub fn open_batch_output<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images_output(&images)
    }

    /// Preprocesses one decoded image frame as a Gemma3 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when pan-and-scan preparation or tensor conversion
    /// fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        preprocess_gemma3_images(&self.config, &self.processor, std::slice::from_ref(image))
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses decoded image frames as a flat Gemma3 frame batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, pan-and-scan preparation, or
    /// tensor conversion fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        preprocess_gemma3_images(&self.config, &self.processor, images)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses one decoded image frame as Gemma3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when pan-and-scan preparation or tensor conversion
    /// fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_gemma3_images_output(&self.config, &self.processor, std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as Gemma3 processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, pan-and-scan preparation, or
    /// tensor conversion fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_gemma3_images_output(&self.config, &self.processor, images)
    }
}

/// Mllama image processor with tiled-canvas frames and aspect-ratio metadata.
#[derive(Clone, Debug)]
pub struct MllamaImageProcessor {
    config: MllamaImageProcessorConfig,
    processor: ImageProcessor,
}

impl MllamaImageProcessor {
    /// Creates a processor from a validated Mllama configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when tiled geometry or the derived generic processor
    /// config is invalid.
    pub fn new(config: MllamaImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_mllama_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &MllamaImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic tile tensor processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, tiled-canvas preparation, tensor
    /// conversion, or nested packing fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as Mllama processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, tiled-canvas preparation, tensor
    /// conversion, or nested packing fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, tiled-canvas
    /// preparation fails, tensor conversion fails, or nested packing fails.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images(&images)
    }

    /// Loads image paths and preprocesses them as Mllama processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, tiled-canvas
    /// preparation fails, tensor conversion fails, or nested packing fails.
    pub fn open_batch_output<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_images_output(&images)
    }

    /// Preprocesses one decoded image frame as an Mllama tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when tiled-canvas preparation, tensor conversion, or
    /// nested packing fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        preprocess_mllama_samples(
            &self.config,
            &self.processor,
            &[std::slice::from_ref(image)],
        )
        .map(|batch| batch.pixel_values)
    }

    /// Preprocesses decoded image frames as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, tiled-canvas preparation
    /// fails, tensor conversion fails, or nested packing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        let samples = images.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        preprocess_mllama_samples(&self.config, &self.processor, &samples)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses nested image samples as an Mllama tensor.
    ///
    /// The outer slice is the batch and each inner vector is the original
    /// images attached to that sample, matching Transformers' nested VLM input.
    ///
    /// # Errors
    ///
    /// Returns an error when no actual images are provided, tiled-canvas
    /// preparation fails, tensor conversion fails, or nested packing fails.
    pub fn preprocess_image_samples(
        &self,
        samples: &[Vec<ImageFrame>],
    ) -> Result<Tensor, ImageProcessorError> {
        let sample_refs = nested_sample_refs(samples);
        preprocess_mllama_samples(&self.config, &self.processor, &sample_refs)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses one decoded image frame as Mllama processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when tiled-canvas preparation, tensor conversion, or
    /// nested packing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_mllama_samples_output(
            &self.config,
            &self.processor,
            &[std::slice::from_ref(image)],
        )
    }

    /// Preprocesses decoded image frames as one image per batch sample.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, tiled-canvas preparation
    /// fails, tensor conversion fails, or nested packing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let samples = images.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        preprocess_mllama_samples_output(&self.config, &self.processor, &samples)
    }

    /// Preprocesses nested image samples as Mllama processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when no actual images are provided, tiled-canvas
    /// preparation fails, tensor conversion fails, or nested packing fails.
    pub fn preprocess_image_samples_output(
        &self,
        samples: &[Vec<ImageFrame>],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let sample_refs = nested_sample_refs(samples);
        preprocess_mllama_samples_output(&self.config, &self.processor, &sample_refs)
    }
}

fn validate_idefics3_config(
    config: &Idefics3ImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    if config.longest_edge == 0 {
        return Err(TransformError::InvalidScaleFactor(config.longest_edge).into());
    }
    if config.max_image_size == 0 {
        return Err(TransformError::InvalidScaleFactor(config.max_image_size).into());
    }
    ImageSize::new(config.max_image_size, config.max_image_size)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn validate_gemma3_config(config: &Gemma3ImageProcessorConfig) -> Result<(), ImageProcessorError> {
    ImageSize::new(config.size.height, config.size.width)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    aspect_ratio_crop_plan(config.size, config.pan_and_scan_options())?;
    Ok(())
}

fn validate_mllama_config(config: &MllamaImageProcessorConfig) -> Result<(), ImageProcessorError> {
    if config.tile_size == 0 {
        return Err(TransformError::InvalidScaleFactor(config.tile_size).into());
    }
    if config.max_image_tiles == 0 {
        return Err(TransformError::InvalidScaleFactor(config.max_image_tiles).into());
    }
    if !config.do_resize {
        return Err(ImageProcessorError::UnsupportedProcessorOption { field: "do_resize" });
    }
    if !config.do_pad {
        return Err(ImageProcessorError::UnsupportedProcessorOption { field: "do_pad" });
    }
    ImageSize::new(config.tile_size, config.tile_size)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct Idefics3PreparedFrames {
    plan: SplitImagePlan,
    frames: Vec<ImageFrame>,
}

#[derive(Clone, Debug)]
struct Idefics3PreprocessBatch {
    pixel_values: Tensor,
    pixel_attention_mask: Option<Tensor>,
    original_sizes: Vec<ImageSize>,
    reshaped_sizes: Vec<ImageSize>,
    frame_counts: Vec<usize>,
    rows: Vec<Vec<usize>>,
    columns: Vec<Vec<usize>>,
}

fn preprocess_idefics3_samples_output(
    config: &Idefics3ImageProcessorConfig,
    processor: &ImageProcessor,
    samples: &[&[ImageFrame]],
) -> Result<ProcessorOutput, ImageProcessorError> {
    let batch = preprocess_idefics3_samples(config, processor, samples)?;
    let mut output = ProcessorOutput::from_pixel_values(batch.pixel_values);
    if let Some(mask) = batch.pixel_attention_mask {
        output.insert_tensor(ProcessorTensorName::other("pixel_attention_mask"), mask);
    }
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(batch.original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(batch.reshaped_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ImagePatchCounts,
        ProcessorMetadataValue::Counts(batch.frame_counts),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("rows"),
        ProcessorMetadataValue::NestedCounts(batch.rows),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("cols"),
        ProcessorMetadataValue::NestedCounts(batch.columns),
    );
    Ok(output)
}

fn preprocess_idefics3_samples(
    config: &Idefics3ImageProcessorConfig,
    processor: &ImageProcessor,
    samples: &[&[ImageFrame]],
) -> Result<Idefics3PreprocessBatch, ImageProcessorError> {
    if samples.is_empty() || !samples.iter().any(|sample| !sample.is_empty()) {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut sample_plans = Vec::with_capacity(samples.len());
    let mut sample_tensors = Vec::with_capacity(samples.len());
    let mut sample_frame_sizes = Vec::with_capacity(samples.len());
    let mut original_sizes = Vec::new();
    let mut reshaped_sizes = Vec::new();
    let mut frame_counts = Vec::new();

    for sample in samples {
        let mut plans = Vec::with_capacity(sample.len());
        let mut tensors = Vec::new();
        let mut frame_sizes = Vec::new();

        for image in *sample {
            let prepared = prepare_idefics3_frames(config, image)?;
            original_sizes.push(prepared.plan.original_size);
            reshaped_sizes.push(prepared.plan.vision_encoder_size);
            frame_counts.push(prepared.plan.frame_count);
            plans.push(prepared.plan);

            for frame in prepared.frames {
                let tensor = processor.preprocess_image(&frame)?;
                frame_sizes.push(ImageSize::new(tensor.height(), tensor.width())?);
                tensors.push(tensor);
            }
        }

        sample_plans.push(plans);
        sample_tensors.push(tensors);
        sample_frame_sizes.push(frame_sizes);
    }

    let split_metadata = split_image_batch_metadata(&sample_plans)?;
    let padding_plan = nested_frame_batch_padding_plan(&sample_frame_sizes)?;
    if !config.do_pad {
        validate_unpadded_nested_frame_shapes(&sample_frame_sizes, padding_plan.target_size)?;
    }

    let pixel_values = padded_nested_frame_tensor(
        config.output_layout,
        &sample_tensors,
        padding_plan.max_frames_per_sample,
        padding_plan.target_size,
    )?;
    let pixel_attention_mask = if config.do_pad {
        Some(nested_frame_pixel_attention_mask_tensor(
            config.output_layout,
            &padding_plan.pixel_attention_mask,
            padding_plan.target_size,
        )?)
    } else {
        None
    };

    Ok(Idefics3PreprocessBatch {
        pixel_values,
        pixel_attention_mask,
        original_sizes,
        reshaped_sizes,
        frame_counts,
        rows: split_metadata.rows,
        columns: split_metadata.columns,
    })
}

fn prepare_idefics3_frames(
    config: &Idefics3ImageProcessorConfig,
    image: &ImageFrame,
) -> Result<Idefics3PreparedFrames, ImageProcessorError> {
    let original_size = ImageSize::new(image.height(), image.width())?;
    let mut resized_size = original_size;
    let mut working = image.clone();

    if config.do_resize {
        resized_size = split_image_resize_size(original_size, config.longest_edge)?;
        working = resize_idefics3_frame(config, &working, resized_size)?;
    }

    if config.do_image_splitting {
        let vision_encoder_size = split_image_encoder_size(resized_size, config.max_image_size)?;
        if ImageSize::new(working.height(), working.width())? != vision_encoder_size {
            working = resize_idefics3_frame(config, &working, vision_encoder_size)?;
        }
        let (frames, rows, columns) = split_idefics3_frame(config, &working)?;
        let frame_count = frames.len();
        let plan = SplitImagePlan {
            original_size,
            resized_size,
            vision_encoder_size,
            max_image_size: config.max_image_size,
            rows,
            columns,
            frame_count,
        };
        return Ok(Idefics3PreparedFrames { plan, frames });
    }

    let vision_encoder_size = square_size(config.max_image_size);
    let frame = resize_idefics3_frame(config, &working, vision_encoder_size)?;
    let plan = SplitImagePlan {
        original_size,
        resized_size,
        vision_encoder_size,
        max_image_size: config.max_image_size,
        rows: 0,
        columns: 0,
        frame_count: 1,
    };
    Ok(Idefics3PreparedFrames {
        plan,
        frames: vec![frame],
    })
}

fn split_idefics3_frame(
    config: &Idefics3ImageProcessorConfig,
    frame: &ImageFrame,
) -> Result<(Vec<ImageFrame>, usize, usize), ImageProcessorError> {
    let size = ImageSize::new(frame.height(), frame.width())?;
    if size.height <= config.max_image_size && size.width <= config.max_image_size {
        return Ok((vec![frame.clone()], 0, 0));
    }

    let rows = size.height.div_ceil(config.max_image_size);
    let columns = size.width.div_ceil(config.max_image_size);
    let crop_height = size.height.div_ceil(rows);
    let crop_width = size.width.div_ceil(columns);
    let mut frames = Vec::with_capacity(
        rows.checked_mul(columns)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(vlm_shape_overflow)?,
    );

    for row in 0..rows {
        for column in 0..columns {
            let x_min = column
                .checked_mul(crop_width)
                .ok_or_else(vlm_shape_overflow)?;
            let y_min = row
                .checked_mul(crop_height)
                .ok_or_else(vlm_shape_overflow)?;
            let x_max = x_min.saturating_add(crop_width).min(size.width);
            let y_max = y_min.saturating_add(crop_height).min(size.height);
            frames.push(crop_frame(
                frame,
                ImageCropBox::new(x_min, y_min, x_max, y_max),
            )?);
        }
    }

    frames.push(resize_idefics3_frame(
        config,
        frame,
        square_size(config.max_image_size),
    )?);
    Ok((frames, rows, columns))
}

fn resize_idefics3_frame(
    config: &Idefics3ImageProcessorConfig,
    frame: &ImageFrame,
    target: ImageSize,
) -> Result<ImageFrame, ImageProcessorError> {
    resize_frame_with_decision(frame, target, config.resize_decision(), ResizeMode::Default)
        .map_err(ImageProcessorError::Transform)
}

fn padded_nested_frame_tensor(
    output_layout: ImageLayout,
    sample_tensors: &[Vec<Tensor>],
    max_frames_per_sample: usize,
    target_size: ImageSize,
) -> Result<Tensor, ImageProcessorError> {
    let first = sample_tensors
        .iter()
        .flat_map(|sample| sample.iter())
        .next()
        .ok_or(ImageProcessorError::EmptyBatch)?;
    let expected_layout = batched_layout_for_image_layout(output_layout);
    if first.layout() != expected_layout || first.batch() != Some(1) {
        return Err(ImageProcessorError::UnsupportedLayout(first.layout()));
    }

    let channels = first.channels();
    let sample_count = sample_tensors.len();
    let frame_slots = checked_vlm_mul(sample_count, max_frames_per_sample)?;
    let pixels = checked_vlm_mul(target_size.height, target_size.width)?;
    let values_len = checked_vlm_mul(checked_vlm_mul(frame_slots, channels)?, pixels)?;
    let mut values = vec![0.0; values_len];

    for (sample_index, sample) in sample_tensors.iter().enumerate() {
        for (frame_index, tensor) in sample.iter().enumerate() {
            if tensor.layout() != expected_layout
                || tensor.batch() != Some(1)
                || tensor.channels() != channels
            {
                return Err(ImageProcessorError::IncompatibleBatchShapes);
            }
            let source_size = ImageSize::new(tensor.height(), tensor.width())?;
            validate_padding_target(target_size, source_size)?;
            let source_values = tensor.data().to_vec::<f32>();

            match output_layout {
                ImageLayout::ChannelsHeightWidth => copy_nested_frame_nchw_tensor(
                    &source_values,
                    &mut values,
                    NestedFrameTensorCopy {
                        sample_index,
                        frame_index,
                        max_frames_per_sample,
                        channels,
                        source_size,
                        target_size,
                    },
                ),
                ImageLayout::HeightWidthChannels => copy_nested_frame_nhwc_tensor(
                    &source_values,
                    &mut values,
                    NestedFrameTensorCopy {
                        sample_index,
                        frame_index,
                        max_frames_per_sample,
                        channels,
                        source_size,
                        target_size,
                    },
                ),
            }
        }
    }

    let shape = match output_layout {
        ImageLayout::ChannelsHeightWidth => vec![
            sample_count,
            max_frames_per_sample,
            channels,
            target_size.height,
            target_size.width,
        ],
        ImageLayout::HeightWidthChannels => vec![
            sample_count,
            max_frames_per_sample,
            target_size.height,
            target_size.width,
            channels,
        ],
    };
    Tensor::new(
        TensorData::F32(values),
        shape,
        patch_batched_layout_for_image_layout(output_layout),
    )
    .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
    .map_err(ImageProcessorError::Tensor)
}

#[derive(Clone, Copy, Debug)]
struct NestedFrameTensorCopy {
    sample_index: usize,
    frame_index: usize,
    max_frames_per_sample: usize,
    channels: usize,
    source_size: ImageSize,
    target_size: ImageSize,
}

fn copy_nested_frame_nchw_tensor(source: &[f32], target: &mut [f32], copy: NestedFrameTensorCopy) {
    for channel in 0..copy.channels {
        let source_channel_offset = channel * copy.source_size.height * copy.source_size.width;
        let target_channel_offset =
            (((copy.sample_index * copy.max_frames_per_sample + copy.frame_index) * copy.channels
                + channel)
                * copy.target_size.height)
                * copy.target_size.width;
        for row in 0..copy.source_size.height {
            let source_start = source_channel_offset + row * copy.source_size.width;
            let target_start = target_channel_offset + row * copy.target_size.width;
            target[target_start..target_start + copy.source_size.width]
                .copy_from_slice(&source[source_start..source_start + copy.source_size.width]);
        }
    }
}

fn copy_nested_frame_nhwc_tensor(source: &[f32], target: &mut [f32], copy: NestedFrameTensorCopy) {
    let source_row_values = copy.source_size.width * copy.channels;
    for row in 0..copy.source_size.height {
        let source_start = row * source_row_values;
        let target_start = (((copy.sample_index * copy.max_frames_per_sample + copy.frame_index)
            * copy.target_size.height
            + row)
            * copy.target_size.width)
            * copy.channels;
        target[target_start..target_start + source_row_values]
            .copy_from_slice(&source[source_start..source_start + source_row_values]);
    }
}

fn nested_frame_pixel_attention_mask_tensor(
    output_layout: ImageLayout,
    mask: &[Vec<Vec<bool>>],
    target_size: ImageSize,
) -> Result<Tensor, ImageProcessorError> {
    let sample_count = mask.len();
    let max_frames_per_sample = mask
        .first()
        .map(Vec::len)
        .ok_or(ImageProcessorError::EmptyBatch)?;
    let pixels = checked_vlm_mul(target_size.height, target_size.width)?;
    let values_len = checked_vlm_mul(
        checked_vlm_mul(sample_count, max_frames_per_sample)?,
        pixels,
    )?;
    let mut values = Vec::with_capacity(values_len);
    for sample in mask {
        if sample.len() != max_frames_per_sample {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        for frame in sample {
            if frame.len() != pixels {
                return Err(ImageProcessorError::IncompatibleBatchShapes);
            }
            values.extend(frame.iter().map(|value| u8::from(*value)));
        }
    }

    let (shape, layout) = match output_layout {
        ImageLayout::ChannelsHeightWidth => (
            vec![
                sample_count,
                max_frames_per_sample,
                1,
                target_size.height,
                target_size.width,
            ],
            Layout::NPCHW,
        ),
        ImageLayout::HeightWidthChannels => (
            vec![
                sample_count,
                max_frames_per_sample,
                target_size.height,
                target_size.width,
                1,
            ],
            Layout::NPHWC,
        ),
    };
    Tensor::new(TensorData::Bool(values), shape, layout)
        .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
        .map_err(ImageProcessorError::Tensor)
}

fn validate_unpadded_nested_frame_shapes(
    frame_sizes: &[Vec<ImageSize>],
    target_size: ImageSize,
) -> Result<(), ImageProcessorError> {
    let max_frames = frame_sizes
        .iter()
        .map(Vec::len)
        .max()
        .ok_or(ImageProcessorError::EmptyBatch)?;
    for sample in frame_sizes {
        if sample.len() != max_frames || sample.iter().any(|size| *size != target_size) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
    }
    Ok(())
}

fn nested_sample_refs(samples: &[Vec<ImageFrame>]) -> Vec<&[ImageFrame]> {
    samples.iter().map(Vec::as_slice).collect()
}

#[derive(Clone, Debug)]
struct Gemma3PreparedFrames {
    plan: AspectRatioCropPlan,
    frames: Vec<ImageFrame>,
    frame_sizes: Vec<ImageSize>,
}

#[derive(Clone, Debug)]
struct Gemma3PreprocessBatch {
    pixel_values: Tensor,
    original_sizes: Vec<ImageSize>,
    frame_sizes: Vec<ImageSize>,
    num_crops: Vec<usize>,
}

fn preprocess_gemma3_images_output(
    config: &Gemma3ImageProcessorConfig,
    processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<ProcessorOutput, ImageProcessorError> {
    let batch = preprocess_gemma3_images(config, processor, images)?;
    let mut output = ProcessorOutput::from_pixel_values(batch.pixel_values);
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(batch.original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(batch.frame_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("num_crops"),
        ProcessorMetadataValue::Counts(batch.num_crops),
    );
    Ok(output)
}

fn preprocess_gemma3_images(
    config: &Gemma3ImageProcessorConfig,
    processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<Gemma3PreprocessBatch, ImageProcessorError> {
    if images.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut frames = Vec::with_capacity(images.len());
    let mut original_sizes = Vec::with_capacity(images.len());
    let mut frame_sizes = Vec::new();
    let mut num_crops = Vec::with_capacity(images.len());

    for image in images {
        let prepared = prepare_gemma3_frames(config, image)?;
        original_sizes.push(prepared.plan.original_size);
        num_crops.push(prepared.plan.crop_count());
        frame_sizes.extend(prepared.frame_sizes);
        frames.extend(prepared.frames);
    }

    let pixel_values = processor.preprocess_images(&frames)?;

    Ok(Gemma3PreprocessBatch {
        pixel_values,
        original_sizes,
        frame_sizes,
        num_crops,
    })
}

fn prepare_gemma3_frames(
    config: &Gemma3ImageProcessorConfig,
    image: &ImageFrame,
) -> Result<Gemma3PreparedFrames, ImageProcessorError> {
    let original_size = ImageSize::new(image.height(), image.width())?;
    let inactive_plan = || AspectRatioCropPlan {
        original_size,
        crop_rows: 0,
        crop_columns: 0,
        crop_size: None,
        crops: Vec::new(),
    };

    if !config.do_pan_and_scan {
        let frame_size = if config.do_resize {
            config.size
        } else {
            original_size
        };
        return Ok(Gemma3PreparedFrames {
            plan: inactive_plan(),
            frames: vec![image.clone()],
            frame_sizes: vec![frame_size],
        });
    }

    let working = if image.pixel_format() == config.pixel_format {
        image.clone()
    } else {
        convert_frame_pixel_format(image, config.pixel_format)?
    };
    let plan = aspect_ratio_crop_plan(original_size, config.pan_and_scan_options())?;
    let mut frames = Vec::with_capacity(1 + plan.crops.len());
    let mut frame_sizes = Vec::with_capacity(1 + plan.crops.len());
    frames.push(working.clone());
    frame_sizes.push(if config.do_resize {
        config.size
    } else {
        original_size
    });

    for crop in &plan.crops {
        let x_max = crop
            .origin_x
            .checked_add(crop.size.width)
            .ok_or_else(vlm_shape_overflow)?;
        let y_max = crop
            .origin_y
            .checked_add(crop.size.height)
            .ok_or_else(vlm_shape_overflow)?;
        frames.push(crop_frame(
            &working,
            ImageCropBox::new(crop.origin_x, crop.origin_y, x_max, y_max),
        )?);
        frame_sizes.push(if config.do_resize {
            config.size
        } else {
            crop.size
        });
    }

    Ok(Gemma3PreparedFrames {
        plan,
        frames,
        frame_sizes,
    })
}

#[derive(Clone, Debug)]
struct MllamaPreparedFrames {
    plan: TiledCanvasPlan,
    frames: Vec<ImageFrame>,
}

#[derive(Clone, Debug)]
struct MllamaPreprocessBatch {
    pixel_values: Tensor,
    original_sizes: Vec<ImageSize>,
    reshaped_sizes: Vec<ImageSize>,
    canvas_sizes: Vec<ImageSize>,
    tile_counts: Vec<usize>,
    num_tiles: Vec<Vec<usize>>,
    aspect_ratio_ids: Vec<Vec<usize>>,
    aspect_ratio_mask: Vec<Vec<Vec<bool>>>,
}

fn preprocess_mllama_samples_output(
    config: &MllamaImageProcessorConfig,
    processor: &ImageProcessor,
    samples: &[&[ImageFrame]],
) -> Result<ProcessorOutput, ImageProcessorError> {
    let batch = preprocess_mllama_samples(config, processor, samples)?;
    let mut output = ProcessorOutput::from_pixel_values(batch.pixel_values);
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(batch.original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(batch.reshaped_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ImagePatchCounts,
        ProcessorMetadataValue::Counts(batch.tile_counts),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("canvas_sizes"),
        ProcessorMetadataValue::ImageSizes(batch.canvas_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("num_tiles"),
        ProcessorMetadataValue::NestedCounts(batch.num_tiles),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("aspect_ratio_ids"),
        ProcessorMetadataValue::NestedCounts(batch.aspect_ratio_ids),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("aspect_ratio_mask"),
        ProcessorMetadataValue::NestedBoolMask(batch.aspect_ratio_mask),
    );
    Ok(output)
}

fn preprocess_mllama_samples(
    config: &MllamaImageProcessorConfig,
    processor: &ImageProcessor,
    samples: &[&[ImageFrame]],
) -> Result<MllamaPreprocessBatch, ImageProcessorError> {
    if samples.is_empty() || !samples.iter().any(|sample| !sample.is_empty()) {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut sample_grids = Vec::with_capacity(samples.len());
    let mut sample_tensors = Vec::with_capacity(samples.len());
    let mut original_sizes = Vec::new();
    let mut reshaped_sizes = Vec::new();
    let mut canvas_sizes = Vec::new();
    let mut tile_counts = Vec::new();

    for sample in samples {
        let mut grids = Vec::with_capacity(sample.len());
        let mut image_tensors = Vec::with_capacity(sample.len());
        for image in *sample {
            let prepared = prepare_mllama_frames(config, image)?;
            let mut tensors = Vec::with_capacity(prepared.frames.len());
            for frame in prepared.frames {
                tensors.push(processor.preprocess_image(&frame)?);
            }
            tile_counts.push(prepared.plan.grid.tile_count()?);
            original_sizes.push(prepared.plan.original_size);
            reshaped_sizes.push(prepared.plan.resized_size);
            canvas_sizes.push(prepared.plan.canvas_size);
            grids.push(prepared.plan.grid);
            image_tensors.push(tensors);
        }
        sample_grids.push(grids);
        sample_tensors.push(image_tensors);
    }

    let metadata = tiled_canvas_batch_metadata(&sample_grids, config.max_image_tiles)?;
    let pixel_values = padded_tiled_canvas_tensor(
        config.output_layout,
        &sample_tensors,
        metadata.max_images_per_sample,
        config.max_image_tiles,
        config.tile_size,
    )?;

    Ok(MllamaPreprocessBatch {
        pixel_values,
        original_sizes,
        reshaped_sizes,
        canvas_sizes,
        tile_counts,
        num_tiles: metadata.num_tiles,
        aspect_ratio_ids: metadata.aspect_ratio_ids,
        aspect_ratio_mask: metadata.aspect_ratio_mask,
    })
}

fn prepare_mllama_frames(
    config: &MllamaImageProcessorConfig,
    image: &ImageFrame,
) -> Result<MllamaPreparedFrames, ImageProcessorError> {
    let original_size = ImageSize::new(image.height(), image.width())?;
    let plan = tiled_canvas_plan(original_size, config.tile_size, config.max_image_tiles)?;
    let mut working = if image.pixel_format() == config.pixel_format {
        image.clone()
    } else {
        convert_frame_pixel_format(image, config.pixel_format)?
    };

    if config.do_resize {
        working = resize_frame_with_decision(
            &working,
            plan.resized_size,
            config.resize_decision(),
            ResizeMode::Default,
        )?;
    }
    if config.do_pad {
        working = pad_frame(&working, plan.padding, &[0])?;
    }

    let frames = split_mllama_tiles(config, &working, plan.grid)?;
    Ok(MllamaPreparedFrames { plan, frames })
}

fn split_mllama_tiles(
    config: &MllamaImageProcessorConfig,
    frame: &ImageFrame,
    grid: TiledCanvasGrid,
) -> Result<Vec<ImageFrame>, ImageProcessorError> {
    let mut frames = Vec::with_capacity(grid.tile_count()?);
    for row in 0..grid.rows {
        let y_min = row
            .checked_mul(config.tile_size)
            .ok_or_else(vlm_shape_overflow)?;
        for column in 0..grid.columns {
            let x_min = column
                .checked_mul(config.tile_size)
                .ok_or_else(vlm_shape_overflow)?;
            frames.push(crop_frame(
                frame,
                ImageCropBox::new(
                    x_min,
                    y_min,
                    x_min
                        .checked_add(config.tile_size)
                        .ok_or_else(vlm_shape_overflow)?,
                    y_min
                        .checked_add(config.tile_size)
                        .ok_or_else(vlm_shape_overflow)?,
                ),
            )?);
        }
    }
    Ok(frames)
}

fn padded_tiled_canvas_tensor(
    output_layout: ImageLayout,
    sample_tensors: &[Vec<Vec<Tensor>>],
    max_images_per_sample: usize,
    max_tiles_per_image: usize,
    tile_edge: usize,
) -> Result<Tensor, ImageProcessorError> {
    let first = sample_tensors
        .iter()
        .flat_map(|sample| sample.iter())
        .flat_map(|image| image.iter())
        .next()
        .ok_or(ImageProcessorError::EmptyBatch)?;
    let expected_layout = batched_layout_for_image_layout(output_layout);
    if first.layout() != expected_layout || first.batch() != Some(1) {
        return Err(ImageProcessorError::UnsupportedLayout(first.layout()));
    }

    let channels = first.channels();
    let tile_size = ImageSize::new(first.height(), first.width())?;
    if tile_size.height != tile_edge || tile_size.width != tile_edge {
        return Err(ImageProcessorError::IncompatibleBatchShapes);
    }

    let sample_count = sample_tensors.len();
    let tile_len = checked_vlm_mul(checked_vlm_mul(tile_edge, tile_edge)?, channels)?;
    let values_len = checked_vlm_mul(
        checked_vlm_mul(
            checked_vlm_mul(sample_count, max_images_per_sample)?,
            max_tiles_per_image,
        )?,
        tile_len,
    )?;
    let mut values = vec![0.0; values_len];

    for (sample_index, sample) in sample_tensors.iter().enumerate() {
        for (image_index, image) in sample.iter().enumerate() {
            if image.len() > max_tiles_per_image || image_index >= max_images_per_sample {
                return Err(ImageProcessorError::IncompatibleBatchShapes);
            }
            for (tile_index, tensor) in image.iter().enumerate() {
                if tensor.layout() != expected_layout
                    || tensor.batch() != Some(1)
                    || tensor.channels() != channels
                    || tensor.height() != tile_edge
                    || tensor.width() != tile_edge
                {
                    return Err(ImageProcessorError::IncompatibleBatchShapes);
                }
                let source_values = tensor.data().to_vec::<f32>();
                if source_values.len() != tile_len {
                    return Err(TransformError::InvalidBufferLength {
                        expected: tile_len,
                        actual: source_values.len(),
                    }
                    .into());
                }

                let slot = checked_vlm_mul(sample_index, max_images_per_sample)?
                    .checked_add(image_index)
                    .ok_or_else(vlm_shape_overflow)?;
                let tile_slot = checked_vlm_mul(slot, max_tiles_per_image)?
                    .checked_add(tile_index)
                    .ok_or_else(vlm_shape_overflow)?;
                let output_start = checked_vlm_mul(tile_slot, tile_len)?;
                let output_end = output_start
                    .checked_add(tile_len)
                    .ok_or_else(vlm_shape_overflow)?;
                values[output_start..output_end].copy_from_slice(&source_values);
            }
        }
    }

    let shape = match output_layout {
        ImageLayout::ChannelsHeightWidth => vec![
            sample_count,
            max_images_per_sample,
            max_tiles_per_image,
            channels,
            tile_edge,
            tile_edge,
        ],
        ImageLayout::HeightWidthChannels => vec![
            sample_count,
            max_images_per_sample,
            max_tiles_per_image,
            tile_edge,
            tile_edge,
            channels,
        ],
    };
    Tensor::new(
        TensorData::F32(values),
        shape,
        tiled_batch_layout_for_image_layout(output_layout),
    )
    .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
    .map_err(ImageProcessorError::Tensor)
}
