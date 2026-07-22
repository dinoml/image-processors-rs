//! LLaVA-NeXT and Pixtral processor implementations.

use super::*;

/// LLaVA-NeXT AnyRes image processor.
#[derive(Clone, Debug)]
pub struct LlavaNextImageProcessor {
    config: LlavaNextImageProcessorConfig,
    patch_processor: ImageProcessor,
}

impl LlavaNextImageProcessor {
    /// Creates a processor from a validated LLaVA-NeXT configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when AnyRes candidates or the derived patch-frame
    /// processor config are invalid.
    pub fn new(config: LlavaNextImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_llava_next_config(&config)?;
        let patch_processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self {
            config,
            patch_processor,
        })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &LlavaNextImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic patch-frame image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.patch_processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, AnyRes patch extraction, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as LLaVA-NeXT processor output.
    ///
    /// The output contains 5D patch-batched `pixel_values`, `original_sizes`,
    /// `reshaped_input_sizes`, `image_patch_counts`, and selected AnyRes sizes.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, AnyRes patch extraction, or tensor
    /// conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as a patch-batched tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, AnyRes patch
    /// extraction fails, or unpadded patch counts are incompatible.
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

    /// Loads image paths and preprocesses them as LLaVA-NeXT processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, AnyRes patch
    /// extraction fails, or unpadded patch counts are incompatible.
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

    /// Preprocesses one decoded image frame as a 5D patch-batched tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when AnyRes patch extraction or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        preprocess_llava_next_images(
            &self.config,
            &self.patch_processor,
            std::slice::from_ref(image),
        )
        .map(|batch| batch.pixel_values)
    }

    /// Preprocesses decoded image frames as a 5D patch-batched tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, AnyRes patch extraction fails,
    /// or unpadded patch counts are incompatible.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        preprocess_llava_next_images(&self.config, &self.patch_processor, images)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses one decoded image frame as LLaVA-NeXT processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when AnyRes patch extraction or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_llava_next_images_output(
            &self.config,
            &self.patch_processor,
            std::slice::from_ref(image),
        )
    }

    /// Preprocesses decoded image frames as LLaVA-NeXT processor output.
    ///
    /// The returned `pixel_values` tensor uses `NPCHW` or `NPHWC` layout. The
    /// patch axis is padded to the largest patch count when `do_pad` is true.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, AnyRes patch extraction fails,
    /// or unpadded patch counts are incompatible.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_llava_next_images_output(&self.config, &self.patch_processor, images)
    }
}

/// Pixtral image processor with patch-aligned resizing and spatial batch padding.
#[derive(Clone, Debug)]
pub struct PixtralImageProcessor {
    config: PixtralImageProcessorConfig,
    processor: ImageProcessor,
}

impl PixtralImageProcessor {
    /// Creates a processor from a validated Pixtral configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when patch geometry or the derived generic image
    /// processor config is invalid.
    pub fn new(config: PixtralImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_pixtral_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &PixtralImageProcessorConfig {
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
    /// Returns an error when loading, patch-aligned resizing, tensor
    /// conversion, or spatial padding fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses an image as Pixtral processor output.
    ///
    /// The output contains spatially padded `pixel_values`, `original_sizes`,
    /// `reshaped_input_sizes`, `image_grid_thw`, and `image_patch_counts`.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, patch-aligned resizing, tensor
    /// conversion, or spatial padding fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as a spatially padded batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, resizing fails,
    /// or tensor-space padding fails.
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

    /// Loads image paths and preprocesses them as Pixtral processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, resizing fails,
    /// or tensor-space padding fails.
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

    /// Preprocesses one decoded image frame as a spatially padded batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when patch-aligned resizing, tensor conversion, or
    /// spatial padding fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        preprocess_pixtral_images(&self.config, &self.processor, std::slice::from_ref(image))
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses decoded image frames as a spatially padded batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, resizing fails, or
    /// tensor-space padding fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        preprocess_pixtral_images(&self.config, &self.processor, images)
            .map(|batch| batch.pixel_values)
    }

    /// Preprocesses one decoded image frame as Pixtral processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when patch-aligned resizing, tensor conversion, or
    /// spatial padding fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_pixtral_images_output(&self.config, &self.processor, std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as Pixtral processor output.
    ///
    /// The returned `pixel_values` tensor uses `NCHW` or `NHWC` layout and is
    /// padded spatially to the largest patch-aligned resized size in the batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, resizing fails, or
    /// tensor-space padding fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_pixtral_images_output(&self.config, &self.processor, images)
    }
}

fn validate_llava_next_config(
    config: &LlavaNextImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    ImageSize::new(config.size.height, config.size.width)?;
    ImageSize::new(config.crop_size.height, config.crop_size.width)?;
    if config.image_grid_pinpoints.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates.into());
    }
    for candidate in &config.image_grid_pinpoints {
        ImageSize::new(candidate.height, candidate.width)?;
    }
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn validate_pixtral_config(
    config: &PixtralImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    ImageSize::new(config.max_size.height, config.max_size.width)?;
    ImageSize::new(config.patch_size.height, config.patch_size.width)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct LlavaNextPreprocessBatch {
    pixel_values: Tensor,
    original_sizes: Vec<ImageSize>,
    reshaped_sizes: Vec<ImageSize>,
    selected_sizes: Vec<ImageSize>,
    patch_counts: Vec<usize>,
}

fn preprocess_llava_next_images_output(
    config: &LlavaNextImageProcessorConfig,
    patch_processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<ProcessorOutput, ImageProcessorError> {
    let batch = preprocess_llava_next_images(config, patch_processor, images)?;
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
        ProcessorMetadataValue::Counts(batch.patch_counts),
    );
    output.insert_metadata(
        ProcessorMetadataName::other("selected_sizes"),
        ProcessorMetadataValue::ImageSizes(batch.selected_sizes),
    );
    Ok(output)
}

fn preprocess_llava_next_images(
    config: &LlavaNextImageProcessorConfig,
    patch_processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<LlavaNextPreprocessBatch, ImageProcessorError> {
    if images.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut plans = Vec::with_capacity(images.len());
    let mut tensors = Vec::with_capacity(images.len());
    let mut original_sizes = Vec::with_capacity(images.len());
    let mut reshaped_sizes = Vec::with_capacity(images.len());
    let mut selected_sizes = Vec::with_capacity(images.len());

    for image in images {
        let patches = patch_grid_image_patches_with_decision(
            image,
            &config.image_grid_pinpoints,
            config.size,
            config.patch_size(),
            config.resize_decision(),
        )?;
        let tensor = patch_processor.preprocess_images(&patches.frames)?;
        original_sizes.push(patches.plan.original_size);
        reshaped_sizes.push(patches.plan.resized_size);
        selected_sizes.push(patches.plan.selected_size);
        plans.push(patches.plan);
        tensors.push(tensor);
    }

    let batch_plan = patch_grid_batch_plan(&plans)?;
    if !config.do_pad && batch_plan.padding_patches.iter().any(|count| *count != 0) {
        return Err(ImageProcessorError::IncompatibleBatchShapes);
    }

    let pixel_values = padded_patch_grid_tensor(config, &tensors, &plans, batch_plan.max_patches)?;

    Ok(LlavaNextPreprocessBatch {
        pixel_values,
        original_sizes,
        reshaped_sizes,
        selected_sizes,
        patch_counts: batch_plan.patch_counts,
    })
}

fn padded_patch_grid_tensor(
    config: &LlavaNextImageProcessorConfig,
    tensors: &[Tensor],
    plans: &[PatchGridPlan],
    max_patches: usize,
) -> Result<Tensor, ImageProcessorError> {
    let first = tensors.first().ok_or(ImageProcessorError::EmptyBatch)?;
    let expected_layout = config.patch_output_layout();
    if first.layout() != expected_layout {
        return Err(ImageProcessorError::UnsupportedLayout(first.layout()));
    }

    let channels = first.channels();
    let height = first.height();
    let width = first.width();
    let item_len = checked_vlm_mul(checked_vlm_mul(channels, height)?, width)?;
    let batch_len = checked_vlm_mul(tensors.len(), max_patches)?;
    let output_len = checked_vlm_mul(batch_len, item_len)?;
    let mut values = vec![0.0; output_len];

    for (image_index, (tensor, plan)) in tensors.iter().zip(plans).enumerate() {
        if tensor.layout() != expected_layout
            || tensor.channels() != channels
            || tensor.height() != height
            || tensor.width() != width
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }

        let patch_count = plan.image_patch_count()?;
        if tensor.batch() != Some(patch_count) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        let expected_values = checked_vlm_mul(patch_count, item_len)?;
        let TensorData::F32(patch_values) = tensor.data() else {
            return Err(ImageProcessorError::ExpectedDataType("f32"));
        };
        if patch_values.len() != expected_values {
            return Err(TransformError::InvalidBufferLength {
                expected: expected_values,
                actual: patch_values.len(),
            }
            .into());
        }

        let output_start = checked_vlm_mul(checked_vlm_mul(image_index, max_patches)?, item_len)?;
        let output_end = output_start
            .checked_add(expected_values)
            .ok_or_else(vlm_shape_overflow)?;
        values[output_start..output_end].copy_from_slice(patch_values);
    }

    let shape = match config.patched_batch_layout() {
        Layout::NPCHW => vec![tensors.len(), max_patches, channels, height, width],
        Layout::NPHWC => vec![tensors.len(), max_patches, height, width, channels],
        layout => return Err(ImageProcessorError::UnsupportedLayout(layout)),
    };
    Tensor::new(
        TensorData::F32(values),
        shape,
        config.patched_batch_layout(),
    )
    .map_err(ImageProcessorError::Tensor)
}

#[derive(Clone, Debug)]
struct PixtralPreprocessBatch {
    pixel_values: Tensor,
    original_sizes: Vec<ImageSize>,
    reshaped_sizes: Vec<ImageSize>,
    grids: Vec<[usize; 3]>,
    patch_counts: Vec<usize>,
}

fn preprocess_pixtral_images_output(
    config: &PixtralImageProcessorConfig,
    processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<ProcessorOutput, ImageProcessorError> {
    let batch = preprocess_pixtral_images(config, processor, images)?;
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
        ProcessorMetadataName::ImageGridThw,
        ProcessorMetadataValue::GridThw(batch.grids),
    );
    output.insert_metadata(
        ProcessorMetadataName::ImagePatchCounts,
        ProcessorMetadataValue::Counts(batch.patch_counts),
    );
    Ok(output)
}

fn preprocess_pixtral_images(
    config: &PixtralImageProcessorConfig,
    processor: &ImageProcessor,
    images: &[ImageFrame],
) -> Result<PixtralPreprocessBatch, ImageProcessorError> {
    if images.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut tensors = Vec::with_capacity(images.len());
    let mut original_sizes = Vec::with_capacity(images.len());
    let mut reshaped_sizes = Vec::with_capacity(images.len());
    let mut plans = Vec::with_capacity(images.len());

    for image in images {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let plan = patch_aligned_resize_plan(original_size, config.max_size, config.patch_size)?;
        let tensor = processor.preprocess_image_with_options(
            image,
            resize_options_for_target(plan.resized_size, ImageProcessorOptions::default()),
        )?;
        original_sizes.push(original_size);
        reshaped_sizes.push(plan.resized_size);
        plans.push(plan);
        tensors.push(tensor);
    }

    let batch_plan = spatial_batch_padding_plan(&reshaped_sizes)?;
    let pixel_values = pad_spatial_tensors(tensors, batch_plan.target_size, config.output_layout)?;
    let grids = plans.iter().map(patch_aligned_grid_thw).collect::<Vec<_>>();
    let patch_counts = plans
        .iter()
        .map(PatchAlignedResizePlan::patch_count)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PixtralPreprocessBatch {
        pixel_values,
        original_sizes,
        reshaped_sizes,
        grids,
        patch_counts,
    })
}

fn patch_aligned_grid_thw(plan: &PatchAlignedResizePlan) -> [usize; 3] {
    [1, plan.patch_rows, plan.patch_columns]
}
