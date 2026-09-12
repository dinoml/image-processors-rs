//! DETR and SAM processor support.

use super::*;

/// Aspect-preserving resize target with an optional longest-edge cap.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ShortestEdgeResizeConfig {
    /// Target size for the shorter image edge.
    pub shortest_edge: usize,
    /// Optional maximum size for the longer image edge.
    #[serde(default)]
    pub longest_edge: Option<usize>,
}

impl ShortestEdgeResizeConfig {
    /// Creates a shortest-edge resize target.
    pub fn new(shortest_edge: usize, longest_edge: Option<usize>) -> Option<Self> {
        if shortest_edge == 0 || longest_edge == Some(0) {
            return None;
        }
        Some(Self {
            shortest_edge,
            longest_edge,
        })
    }
}

impl Default for ShortestEdgeResizeConfig {
    fn default() -> Self {
        Self {
            shortest_edge: 800,
            longest_edge: Some(1333),
        }
    }
}

/// Configuration for DETR-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DetrImageProcessorConfig {
    /// Fixed target image size used by generic image-processor conversion.
    pub image_size: ImageSize,
    /// Shortest-edge target used by typed DETR output APIs.
    #[serde(default)]
    pub resize_size: ShortestEdgeResizeConfig,
    /// Whether typed DETR outputs should pad resized images into a batch tensor.
    #[serde(default = "default_true")]
    pub do_pad: bool,
    /// Explicit typed-output padding target. When unset, batches pad to their
    /// largest resized height and width.
    #[serde(default)]
    pub pad_size: Option<ImageSize>,
    /// Rounds the typed-output padding target up to this size divisor.
    #[serde(default)]
    pub pad_to_multiple: Option<usize>,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl Default for DetrImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_size: square_size(800),
            resize_size: ShortestEdgeResizeConfig::default(),
            do_pad: true,
            pad_size: None,
            pad_to_multiple: None,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl DetrImageProcessorConfig {
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

    /// Converts this family config into the generic image processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        fixed_image_processor_config(FixedImageProcessorSpec {
            image_size: self.image_size,
            output_layout: self.output_layout,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &IMAGENET_MEAN,
            image_std: &IMAGENET_STD,
        })
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let recipe = image_normalization_recipe(ImageNormalizationRecipeSpec {
            id: "transformers.detr_image_processor",
            input: ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages: vec![
                ProcessorRecipeStage::ConvertPixelFormat {
                    format: PixelFormat::Rgb8,
                },
                ProcessorRecipeStage::Resize {
                    resize: RecipeResizeStage::shortest_edge(
                        self.resize_size.shortest_edge,
                        self.resize_size.longest_edge,
                        self.resample,
                        self.resize_parity,
                    ),
                },
            ],
            output_layout: self.output_layout,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: &IMAGENET_MEAN,
            image_std: &IMAGENET_STD,
        })?;

        recipe.with_postprocess(vec![ProcessorRecipePostprocess::ObjectDetection(
            RecipeObjectDetectionPostprocess::new(
                "logits",
                "pred_boxes",
                None,
                0.5,
                RecipeImageSizeSource::CallerProvided,
            ),
        )])
    }
}

/// Configuration for SAM-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SamImageProcessorConfig {
    /// Target image size.
    pub image_size: ImageSize,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
}

impl Default for SamImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_size: square_size(1024),
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl SamImageProcessorConfig {
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

    /// Converts this family config into the generic image processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        fixed_image_processor_config(FixedImageProcessorSpec {
            image_size: self.image_size,
            output_layout: self.output_layout,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            do_rescale: false,
            rescale_factor: 1.0,
            do_normalize: true,
            image_mean: &SAM_IMAGE_MEAN,
            image_std: &SAM_IMAGE_STD,
        })
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let recipe = image_normalization_recipe(ImageNormalizationRecipeSpec {
            id: "transformers.sam_image_processor",
            input: ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages: vec![
                ProcessorRecipeStage::ConvertPixelFormat {
                    format: PixelFormat::Rgb8,
                },
                ProcessorRecipeStage::Resize {
                    resize: RecipeResizeStage::longest_edge(
                        self.image_size.height.max(self.image_size.width),
                        self.resample,
                        self.resize_parity,
                    ),
                },
            ],
            output_layout: self.output_layout,
            do_rescale: false,
            rescale_factor: 1.0,
            do_normalize: true,
            image_mean: &SAM_IMAGE_MEAN,
            image_std: &SAM_IMAGE_STD,
        })?;

        recipe.with_postprocess(vec![ProcessorRecipePostprocess::BinaryMasks(
            RecipeBinaryMaskPostprocess::new(
                "pred_masks",
                RecipeImageSizeSource::OriginalSizes,
                RecipeImageSizeSource::ReshapedInputSizes,
                0.0,
            ),
        )])
    }
}

impl DetrImageProcessor {
    /// Loads and preprocesses media as DETR processor output.
    ///
    /// The output contains `pixel_values`, `pixel_mask`, `original_sizes`, and
    /// `reshaped_input_sizes`.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as DETR processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or preprocessing fails.
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

    /// Preprocesses one decoded image frame as DETR processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, padding, normalization, or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images_output(std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as DETR processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        preprocess_detr_images_output(&self.config, images)
    }

    /// Converts batched DETR logits and normalized boxes into final predictions.
    ///
    /// `logits` are flattened as
    /// `[batch, num_queries, num_labels_with_background]`. `pred_boxes` are
    /// flattened as `[batch, num_queries]` and must use normalized center
    /// format. The last label is treated as DETR's no-object background class.
    ///
    /// # Errors
    ///
    /// Returns an error when target sizes are empty, flattened batch lengths are
    /// inconsistent, or object-detection postprocessing inputs are invalid.
    pub fn post_process_object_detection(
        &self,
        logits: &[f32],
        pred_boxes: &[DetectionCenterBox],
        num_labels_with_background: usize,
        target_sizes: &[ImageSize],
        score_threshold: f32,
    ) -> Result<Vec<Vec<ObjectDetectionPrediction>>, ImageProcessorError> {
        let batch = target_sizes.len();
        if batch == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }
        if !pred_boxes.len().is_multiple_of(batch) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }

        let expected_logits = pred_boxes
            .len()
            .checked_mul(num_labels_with_background)
            .ok_or(ImageProcessorError::Tensor(
                crate::tensor::TensorError::ShapeElementCountOverflow,
            ))?;
        if logits.len() != expected_logits {
            return Err(ImageProcessorError::Transform(
                TransformError::InvalidBufferLength {
                    expected: expected_logits,
                    actual: logits.len(),
                },
            ));
        }

        let num_queries = pred_boxes.len() / batch;
        let logits_per_sample = num_queries.checked_mul(num_labels_with_background).ok_or(
            ImageProcessorError::Tensor(crate::tensor::TensorError::ShapeElementCountOverflow),
        )?;

        target_sizes
            .iter()
            .copied()
            .enumerate()
            .map(|(sample_index, target_size)| {
                let box_start = sample_index * num_queries;
                let logits_start = sample_index * logits_per_sample;
                post_process_detr_object_detection(
                    &logits[logits_start..logits_start + logits_per_sample],
                    &pred_boxes[box_start..box_start + num_queries],
                    num_labels_with_background,
                    target_size,
                    score_threshold,
                )
                .map_err(ImageProcessorError::Transform)
            })
            .collect()
    }
}

impl SamImageProcessor {
    /// Loads and preprocesses an image as SAM processor output.
    ///
    /// The output contains `pixel_values`, `original_sizes`, and
    /// `reshaped_input_sizes`.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, resizing, padding, or tensor conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image_output(&image)
    }

    /// Loads image paths and preprocesses them as SAM processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
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

    /// Preprocesses one decoded image frame as SAM processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, padding, normalization, or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images_output(std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as SAM processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or shapes are incompatible.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let mut original_sizes = Vec::with_capacity(images.len());
        let mut reshaped_sizes = Vec::with_capacity(images.len());
        let mut prepared = Vec::with_capacity(images.len());
        let tensor_processor = tensor_only_processor(self.config.image_processor_config())?;
        for image in images {
            let (frame, original_size, reshaped_size) = prepare_sam_frame(&self.config, image)?;
            prepared.push(tensor_processor.preprocess_image(&frame)?);
            original_sizes.push(original_size);
            reshaped_sizes.push(reshaped_size);
        }

        let mut output = ProcessorOutput::from_pixel_values(pad_spatial_tensors(
            prepared,
            self.config.image_size,
            self.config.output_layout,
        )?);
        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(original_sizes),
        );
        output.insert_metadata(
            ProcessorMetadataName::ReshapedInputSizes,
            ProcessorMetadataValue::ImageSizes(reshaped_sizes),
        );
        Ok(output)
    }

    /// Restores batched SAM mask logits to original image sizes.
    ///
    /// `mask_logits` are flattened as
    /// `[batch, masks_per_image, mask_size.height, mask_size.width]`.
    /// `original_sizes` and `reshaped_input_sizes` should normally come from
    /// this processor's preprocessing output metadata. The configured
    /// `image_size` is used as SAM's padded encoder canvas size.
    ///
    /// # Errors
    ///
    /// Returns an error when sizes are empty or inconsistent, no masks are
    /// requested, the flattened mask length is invalid, or SAM mask
    /// postprocessing inputs are invalid.
    pub fn post_process_masks(
        &self,
        mask_logits: &[f32],
        mask_size: ImageSize,
        masks_per_image: usize,
        original_sizes: &[ImageSize],
        reshaped_input_sizes: &[ImageSize],
        mask_threshold: f32,
    ) -> Result<Vec<BinaryMaskPostprocessOutput>, ImageProcessorError> {
        let batch = original_sizes.len();
        if batch == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }
        if reshaped_input_sizes.len() != batch {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if masks_per_image == 0 {
            return Err(ImageProcessorError::Transform(
                TransformError::EmptyMaskBatch,
            ));
        }

        let mask_pixels =
            mask_size
                .height
                .checked_mul(mask_size.width)
                .ok_or(ImageProcessorError::Transform(
                    TransformError::ImageSizeOverflow,
                ))?;
        let masks_per_sample =
            masks_per_image
                .checked_mul(mask_pixels)
                .ok_or(ImageProcessorError::Transform(
                    TransformError::ImageSizeOverflow,
                ))?;
        let expected_logits =
            batch
                .checked_mul(masks_per_sample)
                .ok_or(ImageProcessorError::Transform(
                    TransformError::ImageSizeOverflow,
                ))?;
        if mask_logits.len() != expected_logits {
            return Err(ImageProcessorError::Transform(
                TransformError::InvalidBufferLength {
                    expected: expected_logits,
                    actual: mask_logits.len(),
                },
            ));
        }

        let mut outputs = Vec::with_capacity(batch);
        for (sample_index, (&original_size, &reshaped_input_size)) in
            original_sizes.iter().zip(reshaped_input_sizes).enumerate()
        {
            let sample_start = sample_index * masks_per_sample;
            let mut masks = Vec::with_capacity(masks_per_image);
            for mask_index in 0..masks_per_image {
                let start = sample_start + mask_index * mask_pixels;
                masks.push(post_process_binary_mask(
                    &mask_logits[start..start + mask_pixels],
                    mask_size,
                    self.config.image_size,
                    reshaped_input_size,
                    original_size,
                    mask_threshold,
                )?);
            }
            outputs.push(BinaryMaskPostprocessOutput::new(original_size, masks));
        }

        Ok(outputs)
    }
}

fn preprocess_detr_images_output(
    config: &DetrImageProcessorConfig,
    images: &[ImageFrame],
) -> Result<ProcessorOutput, ImageProcessorError> {
    if images.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let mut resized = Vec::with_capacity(images.len());
    let mut original_sizes = Vec::with_capacity(images.len());
    let mut reshaped_sizes = Vec::with_capacity(images.len());
    for image in images {
        let (frame, original_size, reshaped_size) = prepare_detr_frame(config, image)?;
        resized.push(frame);
        original_sizes.push(original_size);
        reshaped_sizes.push(reshaped_size);
    }

    let tensor_processor = tensor_only_processor(config.image_processor_config())?;
    let (tensor, pixel_mask) = if config.do_pad {
        let target = padded_batch_target_for_sizes(
            &reshaped_sizes,
            config.pad_size,
            config.pad_to_multiple,
        )?;
        // DETR pads after normalization, so invisible pixels are tensor zeros.
        let tensors = resized
            .iter()
            .map(|frame| tensor_processor.preprocess_images(std::slice::from_ref(frame)))
            .collect::<Result<Vec<_>, _>>()?;
        (
            pad_spatial_tensors(tensors, target, config.output_layout)?,
            pixel_mask_for_valid_sizes(&reshaped_sizes, target)?,
        )
    } else {
        let tensor = tensor_processor.preprocess_images(&resized)?;
        let pixel_mask = full_pixel_mask_for(&tensor)?;
        (tensor, pixel_mask)
    };

    let mut output = ProcessorOutput::from_pixel_values(tensor);
    output.insert_tensor(ProcessorTensorName::PixelMask, pixel_mask);
    output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(original_sizes),
    );
    output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(reshaped_sizes),
    );
    Ok(output)
}

fn prepare_detr_frame(
    config: &DetrImageProcessorConfig,
    image: &ImageFrame,
) -> Result<(ImageFrame, ImageSize, ImageSize), ImageProcessorError> {
    let original_size = ImageSize::new(image.height(), image.width())?;
    let reshaped_size = shortest_edge_resize_output_size(original_size, config.resize_size)?;
    let resized = resize_frame_with_decision(
        image,
        reshaped_size,
        config.resize_decision(),
        ResizeMode::Default,
    )?;
    Ok((resized, original_size, reshaped_size))
}

pub(super) fn shortest_edge_resize_output_size(
    original_size: ImageSize,
    config: ShortestEdgeResizeConfig,
) -> Result<ImageSize, ImageProcessorError> {
    let fallback =
        shortest_edge_resize_size(original_size, config.shortest_edge, config.longest_edge)?;
    let shortest = original_size.height.min(original_size.width);
    let longest = original_size.height.max(original_size.width);
    let Some(cap) = config.longest_edge else {
        return Ok(fallback);
    };
    if longest as f64 / shortest as f64 * config.shortest_edge as f64 <= cap as f64 {
        return Ok(fallback);
    }
    // Transformers retains the unrounded capped size for the long axis.
    // Recomputing it from the rounded short axis shifts document pixels.
    let raw_short = cap as f64 * shortest as f64 / longest as f64;
    let short = raw_short.round_ties_even() as usize;
    if shortest == short {
        return Ok(original_size);
    }
    let long = (raw_short * longest as f64 / shortest as f64) as usize;
    let (height, width) = if original_size.width < original_size.height {
        (long, short)
    } else {
        (short, long)
    };
    Ok(ImageSize::new(height.max(1), width.max(1))?)
}

fn padded_batch_target_for_sizes(
    sizes: &[ImageSize],
    requested_target: Option<ImageSize>,
    pad_to_multiple: Option<usize>,
) -> Result<ImageSize, ImageProcessorError> {
    let first = sizes
        .first()
        .copied()
        .ok_or(ImageProcessorError::EmptyBatch)?;
    let mut target = requested_target.unwrap_or_else(|| {
        sizes.iter().copied().fold(first, |target, size| ImageSize {
            height: target.height.max(size.height),
            width: target.width.max(size.width),
        })
    });

    if let Some(multiple) = pad_to_multiple {
        target = round_up_size_to_multiple(target, multiple)?;
    }

    for size in sizes {
        validate_padding_target(target, *size)?;
    }
    Ok(target)
}

fn pixel_mask_for_valid_sizes(
    sizes: &[ImageSize],
    target: ImageSize,
) -> Result<Tensor, ImageProcessorError> {
    let items = sizes.len();
    let elements = items
        .checked_mul(target.height)
        .and_then(|value| value.checked_mul(target.width))
        .ok_or(ImageProcessorError::Tensor(
            crate::tensor::TensorError::ShapeElementCountOverflow,
        ))?;
    let mut values = Vec::with_capacity(elements);

    for size in sizes {
        validate_padding_target(target, *size)?;
        for row in 0..target.height {
            for column in 0..target.width {
                values.push(u8::from(row < size.height && column < size.width));
            }
        }
    }

    Tensor::new(
        TensorData::Bool(values),
        vec![items, 1, target.height, target.width],
        Layout::NCHW,
    )
    .map_err(ImageProcessorError::Tensor)
}

pub(super) fn validate_padding_target(
    target_size: ImageSize,
    image_size: ImageSize,
) -> Result<(), ImageProcessorError> {
    if target_size.height < image_size.height || target_size.width < image_size.width {
        return Err(ImageProcessorError::InvalidPaddingTarget {
            target_size,
            image_size,
        });
    }
    Ok(())
}

fn round_up_size_to_multiple(
    size: ImageSize,
    factor: usize,
) -> Result<ImageSize, ImageProcessorError> {
    if factor == 0 {
        return Err(ImageProcessorError::Transform(
            crate::transforms::TransformError::InvalidScaleFactor(factor),
        ));
    }

    Ok(ImageSize {
        height: round_up_to_multiple(size.height, factor)?,
        width: round_up_to_multiple(size.width, factor)?,
    })
}

fn round_up_to_multiple(value: usize, factor: usize) -> Result<usize, ImageProcessorError> {
    let remainder = value % factor;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(factor - remainder)
        .ok_or(ImageProcessorError::Transform(
            crate::transforms::TransformError::ImageSizeOverflow,
        ))
}

fn full_pixel_mask_for(pixel_values: &Tensor) -> Result<Tensor, ImageProcessorError> {
    let leading_axis = pixel_values
        .leading_axis()
        .unwrap_or(TensorLeadingAxis::Batch);
    let items = pixel_values.shape().first().copied().unwrap_or(1);
    let elements = items
        .checked_mul(pixel_values.height())
        .and_then(|value| value.checked_mul(pixel_values.width()))
        .ok_or(ImageProcessorError::Tensor(
            crate::tensor::TensorError::ShapeElementCountOverflow,
        ))?;
    Tensor::new(
        TensorData::Bool(vec![1; elements]),
        vec![items, 1, pixel_values.height(), pixel_values.width()],
        Layout::NCHW,
    )
    .and_then(|tensor| tensor.with_leading_axis(leading_axis))
    .map_err(ImageProcessorError::Tensor)
}

fn prepare_sam_frame(
    config: &SamImageProcessorConfig,
    image: &ImageFrame,
) -> Result<(ImageFrame, ImageSize, ImageSize), ImageProcessorError> {
    let original_size = ImageSize::new(image.height(), image.width())?;
    let reshaped_size = resize_to_fit_size(original_size, config.image_size);
    let resized = resize_frame(image, reshaped_size, config.resample, ResizeMode::Default)?;
    Ok((resized, original_size, reshaped_size))
}

pub(super) fn pad_spatial_tensors(
    tensors: Vec<Tensor>,
    target: ImageSize,
    output_layout: ImageLayout,
) -> Result<Tensor, ImageProcessorError> {
    let first = tensors.first().ok_or(ImageProcessorError::EmptyBatch)?;
    let items = tensors.len();
    let channels = first.channels();
    let layout = batched_layout_for_image_layout(output_layout);
    let elements = checked_spatial_pad_mul(items, target.height)
        .and_then(|value| checked_spatial_pad_mul(value, target.width))
        .and_then(|value| checked_spatial_pad_mul(value, channels))?;
    let mut values = vec![0.0; elements];

    for (batch, tensor) in tensors.into_iter().enumerate() {
        let (data, shape, tensor_layout, _) = tensor.into_parts();
        if tensor_layout != layout || shape.first().copied() != Some(1) {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if shape[tensor_layout.channel_axis()] != channels {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }

        let source_size = ImageSize::new(
            shape[tensor_layout.height_axis()],
            shape[tensor_layout.width_axis()],
        )?;
        validate_padding_target(target, source_size)?;
        let source = match data {
            TensorData::F32(values) => values,
            other => other.to_vec::<f32>(),
        };

        match output_layout {
            ImageLayout::ChannelsHeightWidth => {
                copy_spatial_nchw_tensor(&source, &mut values, batch, channels, source_size, target)
            }
            ImageLayout::HeightWidthChannels => {
                copy_spatial_nhwc_tensor(&source, &mut values, batch, channels, source_size, target)
            }
        }
    }

    let shape = match output_layout {
        ImageLayout::ChannelsHeightWidth => vec![items, channels, target.height, target.width],
        ImageLayout::HeightWidthChannels => vec![items, target.height, target.width, channels],
    };

    Tensor::new(TensorData::F32(values), shape, layout)
        .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
        .map_err(ImageProcessorError::Tensor)
}

fn copy_spatial_nchw_tensor(
    source: &[f32],
    target_values: &mut [f32],
    batch: usize,
    channels: usize,
    source_size: ImageSize,
    target_size: ImageSize,
) {
    for channel in 0..channels {
        let source_channel_offset = channel * source_size.height * source_size.width;
        let target_channel_offset =
            ((batch * channels + channel) * target_size.height) * target_size.width;
        for row in 0..source_size.height {
            let source_start = source_channel_offset + row * source_size.width;
            let target_start = target_channel_offset + row * target_size.width;
            target_values[target_start..target_start + source_size.width]
                .copy_from_slice(&source[source_start..source_start + source_size.width]);
        }
    }
}

fn copy_spatial_nhwc_tensor(
    source: &[f32],
    target_values: &mut [f32],
    batch: usize,
    channels: usize,
    source_size: ImageSize,
    target_size: ImageSize,
) {
    let source_row_values = source_size.width * channels;
    for row in 0..source_size.height {
        let source_start = row * source_row_values;
        let target_start = ((batch * target_size.height + row) * target_size.width) * channels;
        target_values[target_start..target_start + source_row_values]
            .copy_from_slice(&source[source_start..source_start + source_row_values]);
    }
}

pub(super) fn checked_spatial_pad_mul(
    left: usize,
    right: usize,
) -> Result<usize, ImageProcessorError> {
    left.checked_mul(right).ok_or(ImageProcessorError::Tensor(
        crate::tensor::TensorError::ShapeElementCountOverflow,
    ))
}
