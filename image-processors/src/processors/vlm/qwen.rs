//! Qwen/VLM smart-resize and patch-flattening processor.

use super::*;

/// Qwen/VLM image processor with smart-resize preprocessing.
#[derive(Clone, Debug)]
pub struct QwenVlImageProcessor {
    config: QwenVlImageProcessorConfig,
    processor: ImageProcessor,
    patch_processor: ImageProcessor,
}

impl QwenVlImageProcessor {
    /// Creates a processor from a validated Qwen/VLM configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when patch geometry or derived generic image processor configs are invalid.
    pub fn new(config: QwenVlImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_qwen_patch_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        let patch_processor = ImageProcessor::new(config.patch_image_processor_config())?;
        Ok(Self {
            config,
            processor,
            patch_processor,
        })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &QwenVlImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, resizing, or tensor conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as Qwen/VLM processor output.
    ///
    /// The output contains patch-flattened `pixel_values`, `original_sizes`,
    /// `reshaped_input_sizes`, and `image_grid_thw`.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, smart-resize sizing, resizing, or tensor conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as Qwen/VLM processor output.
    ///
    /// The output concatenates patch rows from each image into `pixel_values`
    /// and records one `image_grid_thw` entry per image.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, smart-resize
    /// sizing fails, or tensor conversion fails.
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

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when smart-resize sizing, resizing, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame as Qwen/VLM processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when smart-resize sizing, resizing, or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_output_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded image frames as Qwen/VLM processor output.
    ///
    /// The returned `pixel_values` tensor uses `NC` layout. Its rows are the
    /// concatenated patch rows from every image in input order, while metadata
    /// keeps the original size, reshaped size, and grid for each image.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, smart-resize sizing fails,
    /// patch flattening fails, or tensor conversion fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_qwen_frames_output(&self.config, &self.patch_processor, images.iter())
    }

    /// Preprocesses an image sequence as Qwen/VLM processor output.
    ///
    /// Each decoded sequence frame contributes one entry to `image_grid_thw`.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, smart-resize sizing fails,
    /// patch flattening fails, or tensor conversion fails.
    pub fn preprocess_image_sequence_output(
        &self,
        sequence: &ImageSequence,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_qwen_frames_output(
            &self.config,
            &self.patch_processor,
            sequence.frames().iter(),
        )
    }

    /// Preprocesses a video clip as Qwen/VLM processor output.
    ///
    /// Video frames are grouped by `temporal_patch_size` and contribute one
    /// clip-level entry to `image_grid_thw`.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, smart-resize sizing fails,
    /// patch flattening fails, or tensor conversion fails.
    pub fn preprocess_video_output(
        &self,
        video: &VideoClip,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_qwen_video_output(&self.config, &self.patch_processor, video)
    }

    /// Preprocesses one decoded image frame with per-call options.
    ///
    /// Per-call dimensions override smart-resize only when both `height` and
    /// `width` are provided.
    ///
    /// # Errors
    ///
    /// Returns an error when smart-resize sizing, resizing, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let target = smart_resize_target_for_image(image, self.config.resize_limits, options)?;
        self.processor
            .preprocess_image_with_options(image, resize_options_for_target(target, options))
    }

    /// Preprocesses one decoded image frame with per-call options as Qwen/VLM output.
    ///
    /// # Errors
    ///
    /// Returns an error when smart-resize sizing, resizing, patch flattening, or
    /// tensor conversion fails.
    pub fn preprocess_image_output_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let target = smart_resize_target_for_image(image, self.config.resize_limits, options)?;
        let image_tensor = self
            .patch_processor
            .preprocess_image_with_options(image, resize_options_for_target(target, options))?;
        let pixel_values =
            flatten_image_patches(&image_tensor, patch_flatten_spec(&self.config), target)?;
        let mut output = ProcessorOutput::from_pixel_values(pixel_values);
        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(vec![original_size]),
        );
        output.insert_metadata(
            ProcessorMetadataName::ReshapedInputSizes,
            ProcessorMetadataValue::ImageSizes(vec![target]),
        );
        output.insert_metadata(
            ProcessorMetadataName::ImageGridThw,
            ProcessorMetadataValue::GridThw(vec![flattened_patch_grid_thw(
                target,
                patch_flatten_spec(&self.config),
            )?]),
        );
        Ok(output)
    }
}

fn preprocess_qwen_frames_output<'a>(
    config: &QwenVlImageProcessorConfig,
    patch_processor: &ImageProcessor,
    images: impl IntoIterator<Item = &'a ImageFrame>,
) -> Result<ProcessorOutput, ImageProcessorError> {
    let mut original_sizes = Vec::new();
    let mut reshaped_sizes = Vec::new();
    let mut grids = Vec::new();
    let mut values = Vec::new();
    let mut patch_rows = 0usize;
    let mut feature_dim = None;

    for image in images {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let target = smart_resize_target_for_image(
            image,
            config.resize_limits,
            ImageProcessorOptions::default(),
        )?;
        let image_tensor = patch_processor.preprocess_image_with_options(
            image,
            resize_options_for_target(target, ImageProcessorOptions::default()),
        )?;
        let flattened = flatten_image_patches(&image_tensor, patch_flatten_spec(config), target)?;
        let shape = flattened.shape();
        let rows = shape[0];
        let sample_dim = shape[1];
        if feature_dim.is_some_and(|dim| dim != sample_dim) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        feature_dim = Some(sample_dim);
        patch_rows = checked_vlm_add(patch_rows, rows)?;

        let TensorData::F32(patch_values) = flattened.data() else {
            return Err(ImageProcessorError::ExpectedDataType("f32"));
        };
        values.extend_from_slice(patch_values);
        original_sizes.push(original_size);
        reshaped_sizes.push(target);
        grids.push(flattened_patch_grid_thw(
            target,
            patch_flatten_spec(config),
        )?);
    }

    let feature_dim = feature_dim.ok_or(ImageProcessorError::EmptyBatch)?;
    let mut output = ProcessorOutput::from_pixel_values(
        Tensor::new(
            TensorData::F32(values),
            vec![patch_rows, feature_dim],
            Layout::NC,
        )
        .map_err(ImageProcessorError::Tensor)?,
    );
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(reshaped_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ImageGridThw,
        ProcessorMetadataValue::GridThw(grids),
    );
    Ok(output)
}

fn preprocess_qwen_video_output(
    config: &QwenVlImageProcessorConfig,
    patch_processor: &ImageProcessor,
    video: &VideoClip,
) -> Result<ProcessorOutput, ImageProcessorError> {
    let first = video
        .frames()
        .first()
        .ok_or(ImageProcessorError::EmptyBatch)?;
    let target = smart_resize_target_for_image(
        first.image(),
        config.resize_limits,
        ImageProcessorOptions::default(),
    )?;
    let mut original_sizes = Vec::with_capacity(video.len());
    let reshaped_sizes = vec![target; video.len()];
    let mut tensors = Vec::with_capacity(video.len());

    for frame in video.frames() {
        let image = frame.image();
        original_sizes.push(ImageSize::new(image.height(), image.width())?);
        tensors.push(patch_processor.preprocess_image_with_options(
            image,
            resize_options_for_target(target, ImageProcessorOptions::default()),
        )?);
    }

    let pixel_values = flatten_temporal_patches(&tensors, patch_flatten_spec(config), target)?;
    let mut output = ProcessorOutput::from_pixel_values(pixel_values);
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(reshaped_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ImageGridThw,
        ProcessorMetadataValue::GridThw(vec![flattened_temporal_patch_grid_thw(
            video.len(),
            target,
            patch_flatten_spec(config),
        )?]),
    );
    Ok(output)
}

fn smart_resize_target_for_image(
    image: &ImageFrame,
    resize_limits: ResizeLimits,
    options: ImageProcessorOptions,
) -> Result<ImageSize, ImageProcessorError> {
    match (options.height, options.width) {
        (Some(height), Some(width)) => Ok(ImageSize::new(height, width)?),
        _ => Ok(ImageSize::new(image.height(), image.width())?.smart_resize(resize_limits)?),
    }
}

#[derive(Clone, Copy, Debug)]
struct PatchFlattenSpec {
    source_layout: Layout,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
}

fn patch_flatten_spec(config: &QwenVlImageProcessorConfig) -> PatchFlattenSpec {
    PatchFlattenSpec {
        source_layout: Layout::NCHW,
        patch_size: config.patch_size,
        temporal_patch_size: config.temporal_patch_size,
        merge_size: config.merge_size,
    }
}

fn flattened_patch_grid_thw(
    target: ImageSize,
    spec: PatchFlattenSpec,
) -> Result<[usize; 3], ImageProcessorError> {
    recipe_patch_stage(spec)
        .image_grid_thw(target)
        .map_err(recipe_patch_error_to_processor_error)
}

fn flattened_temporal_patch_grid_thw(
    frame_count: usize,
    target: ImageSize,
    spec: PatchFlattenSpec,
) -> Result<[usize; 3], ImageProcessorError> {
    recipe_patch_stage(spec)
        .temporal_grid_thw(frame_count, target)
        .map_err(recipe_patch_error_to_processor_error)
}

fn validate_qwen_patch_config(
    config: &QwenVlImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    validate_patch_flatten_config(patch_flatten_spec(config), config.resize_limits.factor)
}

fn validate_patch_flatten_config(
    spec: PatchFlattenSpec,
    resize_factor: usize,
) -> Result<(), ImageProcessorError> {
    validate_positive_dimension("patch_size", spec.patch_size)?;
    validate_positive_dimension("temporal_patch_size", spec.temporal_patch_size)?;
    validate_positive_dimension("merge_size", spec.merge_size)?;

    let expected_factor = spec
        .patch_size
        .checked_mul(spec.merge_size)
        .ok_or_else(vlm_shape_overflow)?;
    if resize_factor != expected_factor {
        return Err(ImageProcessorError::InvalidPatchGeometry {
            resize_factor,
            patch_size: spec.patch_size,
            merge_size: spec.merge_size,
        });
    }
    Ok(())
}

pub(in crate::processors) fn validate_positive_dimension(
    field: &'static str,
    value: usize,
) -> Result<(), ImageProcessorError> {
    if value == 0 {
        return Err(ImageProcessorError::InvalidConfiguredDimension { field, value });
    }
    Ok(())
}

fn recipe_patch_stage(spec: PatchFlattenSpec) -> RecipePatchStage {
    RecipePatchStage::flatten(
        spec.source_layout,
        spec.patch_size,
        spec.temporal_patch_size,
        spec.merge_size,
    )
}

fn recipe_patch_error_to_processor_error(error: RecipePatchError) -> ImageProcessorError {
    match error {
        RecipePatchError::InvalidDimension { field, value } => {
            ImageProcessorError::InvalidConfiguredDimension { field, value }
        }
        RecipePatchError::InvalidTarget {
            target_size,
            patch_size,
            merge_size,
        } => ImageProcessorError::InvalidPatchTarget {
            target_size,
            patch_size,
            merge_size,
        },
        RecipePatchError::EmptyFrameCount => ImageProcessorError::EmptyBatch,
        RecipePatchError::UnsupportedSourceLayout { actual, .. } => {
            ImageProcessorError::UnsupportedLayout(actual)
        }
        RecipePatchError::InvalidSourceShape { .. }
        | RecipePatchError::IncompatibleSourceShape { .. } => {
            ImageProcessorError::IncompatibleBatchShapes
        }
        RecipePatchError::ExpectedDataType { expected, .. } => match expected {
            DType::F32 => ImageProcessorError::ExpectedDataType("f32"),
            DType::F16 => ImageProcessorError::ExpectedDataType("f16"),
            DType::BF16 => ImageProcessorError::ExpectedDataType("bf16"),
            DType::U8 => ImageProcessorError::ExpectedDataType("u8"),
            DType::I8 => ImageProcessorError::ExpectedDataType("i8"),
            DType::QuantizedU8 => ImageProcessorError::ExpectedDataType("quantized u8"),
            DType::QuantizedI8 => ImageProcessorError::ExpectedDataType("quantized i8"),
            DType::PackedU4 => ImageProcessorError::ExpectedDataType("packed u4"),
            DType::PackedI4 => ImageProcessorError::ExpectedDataType("packed i4"),
            DType::I32 => ImageProcessorError::ExpectedDataType("i32"),
            DType::I64 => ImageProcessorError::ExpectedDataType("i64"),
            DType::Bool => ImageProcessorError::ExpectedDataType("bool"),
        },
        RecipePatchError::Tensor(error) => ImageProcessorError::Tensor(error),
        RecipePatchError::GeometryOverflow => vlm_shape_overflow(),
    }
}

fn flatten_image_patches(
    tensor: &Tensor,
    spec: PatchFlattenSpec,
    target: ImageSize,
) -> Result<Tensor, ImageProcessorError> {
    recipe_patch_stage(spec)
        .flatten_image_tensor(tensor, target)
        .map_err(recipe_patch_error_to_processor_error)
}

fn flatten_temporal_patches(
    tensors: &[Tensor],
    spec: PatchFlattenSpec,
    target: ImageSize,
) -> Result<Tensor, ImageProcessorError> {
    recipe_patch_stage(spec)
        .flatten_temporal_tensors(tensors, target)
        .map_err(recipe_patch_error_to_processor_error)
}
