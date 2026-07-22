//! Video processor families.

use super::*;

/// Configuration for VideoMAE-style video preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VideoMaeImageProcessorConfig {
    /// Shortest-edge resize target used before optional center cropping.
    pub size: usize,
    /// Center-crop target used when `do_center_crop` is enabled.
    pub crop_size: ImageSize,
    /// Whether frames should be resized before tensor conversion.
    pub do_resize: bool,
    /// Whether resized frames should be center-cropped.
    pub do_center_crop: bool,
    /// Device-agnostic video output axis order.
    pub output_layout: VideoLayout,
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
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel or scalar normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel or scalar normalization standard deviations.
    pub image_std: Vec<f32>,
}

impl Default for VideoMaeImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: 224,
            crop_size: square_size(224),
            do_resize: true,
            do_center_crop: true,
            output_layout: VideoLayout::FramesChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl VideoMaeImageProcessorConfig {
    /// Returns the video output axis order.
    pub fn output_video_layout(&self) -> VideoLayout {
        self.output_layout
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the generic tensor-stage processor
    /// config used after per-frame VideoMAE geometry has been applied.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: ResizeParity::Resampling,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: Layout::from(self.output_layout),
        }
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid normalization or
    /// output-layout settings.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(3);
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        });
        if self.do_rescale {
            stages.push(ProcessorRecipeStage::Rescale {
                factor: self.rescale_factor,
            });
        }
        if self.do_normalize {
            stages.push(ProcessorRecipeStage::Normalize {
                mean: self.image_mean.clone(),
                std: self.image_std.clone(),
            });
        }

        ProcessorRecipe::new(
            "transformers.videomae_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput {
                layout: Layout::from(self.output_layout),
                leading_axis: TensorLeadingAxis::Frames,
            },
        )
    }
}

/// Configuration for ViViT-style video preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VivitImageProcessorConfig {
    /// Shortest-edge resize target used before optional center cropping.
    pub size: usize,
    /// Center-crop target used when `do_center_crop` is enabled.
    pub crop_size: ImageSize,
    /// Whether frames should be resized before tensor conversion.
    pub do_resize: bool,
    /// Whether resized frames should be center-cropped.
    pub do_center_crop: bool,
    /// Device-agnostic video output axis order.
    pub output_layout: VideoLayout,
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
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether rescaling should shift values down by one before normalization.
    pub offset: bool,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel or scalar normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel or scalar normalization standard deviations.
    pub image_std: Vec<f32>,
}

impl Default for VivitImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: 256,
            crop_size: square_size(224),
            do_resize: true,
            do_center_crop: true,
            output_layout: VideoLayout::FramesChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            do_rescale: true,
            rescale_factor: VIVIT_RESCALE_FACTOR,
            offset: true,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl VivitImageProcessorConfig {
    /// Returns the video output axis order.
    pub fn output_video_layout(&self) -> VideoLayout {
        self.output_layout
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the generic tensor-stage processor
    /// config used after per-frame ViViT geometry has been applied.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        let (do_normalize, image_mean, image_std) = self.tensor_stage_normalization();
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: ResizeParity::Resampling,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_binarize: false,
            output_layout: Layout::from(self.output_layout),
        }
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid normalization or
    /// output-layout settings.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let (do_normalize, image_mean, image_std) = self.tensor_stage_normalization();
        let mut stages = Vec::with_capacity(3);
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: PixelFormat::Rgb8,
        });
        if self.do_rescale {
            stages.push(ProcessorRecipeStage::Rescale {
                factor: self.rescale_factor,
            });
        }
        if do_normalize {
            stages.push(ProcessorRecipeStage::Normalize {
                mean: image_mean,
                std: image_std,
            });
        }

        ProcessorRecipe::new(
            "transformers.vivit_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput {
                layout: Layout::from(self.output_layout),
                leading_axis: TensorLeadingAxis::Frames,
            },
        )
    }

    fn tensor_stage_normalization(&self) -> (bool, Vec<f32>, Vec<f32>) {
        if !self.offset {
            return (
                self.do_normalize,
                self.image_mean.clone(),
                self.image_std.clone(),
            );
        }

        if self.do_normalize {
            let image_mean = self.image_mean.iter().map(|mean| mean + 1.0).collect();
            return (true, image_mean, self.image_std.clone());
        }

        (true, vec![1.0], vec![1.0])
    }
}

/// VideoMAE video processor for decoded frame clips.
#[derive(Clone, Debug)]
pub struct VideoMaeImageProcessor {
    config: VideoMaeImageProcessorConfig,
    processor: ImageProcessor,
}

impl VideoMaeImageProcessor {
    /// Creates a processor from a validated VideoMAE configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor-stage image processor config is invalid.
    pub fn new(config: VideoMaeImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &VideoMaeImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses media from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, VideoMAE geometry, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        self.open_source(MediaSource::path(path.as_ref()))
    }

    /// Loads and preprocesses a media source.
    ///
    /// Images are treated as one-frame clips. Image sequences and videos are
    /// preprocessed in decoded frame order.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, VideoMAE geometry, or tensor
    /// conversion fails.
    pub fn open_source(&self, source: MediaSource) -> Result<Tensor, ImageProcessorError> {
        match DefaultMediaLoader.load_with_remote_options_and_image_decode_backend(
            source,
            &Default::default(),
            self.config.decode_backend,
        )? {
            LoadedMedia::Image(image) => self.preprocess_frames(std::slice::from_ref(&image)),
            LoadedMedia::ImageSequence(sequence) => self.preprocess_image_sequence(&sequence),
            LoadedMedia::Video(video) => self.preprocess_video(&video),
        }
    }

    /// Loads image paths and preprocesses them as a single decoded frame clip.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, loading fails, VideoMAE
    /// geometry fails, or tensor shapes are incompatible.
    pub fn open_frames<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_frames(&images)
    }

    /// Preprocesses one decoded image frame as a one-frame clip.
    ///
    /// # Errors
    ///
    /// Returns an error when VideoMAE geometry or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_frames(std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, VideoMAE geometry fails,
    /// or tensor shapes are incompatible.
    pub fn preprocess_frames(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let prepared = images
            .iter()
            .map(|image| video_mae_prepare_frame(&self.config, image))
            .collect::<Result<Vec<_>, _>>()?;
        self.processor
            .preprocess_images(&prepared)?
            .with_leading_axis(TensorLeadingAxis::Frames)
            .map_err(ImageProcessorError::Tensor)
    }

    /// Preprocesses an image sequence as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, VideoMAE geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_image_sequence(
        &self,
        sequence: &ImageSequence,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_frames(sequence.frames())
    }

    /// Preprocesses an image sequence as a `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, VideoMAE geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_image_sequence_output(
        &self,
        sequence: &ImageSequence,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_sequence(sequence)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses a video clip as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, VideoMAE geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_video(&self, video: &VideoClip) -> Result<Tensor, ImageProcessorError> {
        if video.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let images = video
            .frames()
            .iter()
            .map(|frame| frame.image().clone())
            .collect::<Vec<_>>();
        self.preprocess_frames(&images)
    }

    /// Preprocesses a video clip as a `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when VideoMAE geometry or tensor conversion fails.
    pub fn preprocess_video_output(
        &self,
        video: &VideoClip,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_video(video)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded video clips as a batched VideoMAE tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, VideoMAE
    /// geometry fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos(&self, videos: &[VideoClip]) -> Result<Tensor, ImageProcessorError> {
        if videos.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let tensors = videos
            .iter()
            .map(|video| self.preprocess_video(video))
            .collect::<Result<Vec<_>, _>>()?;
        stack_frame_tensors_as_video_batch(tensors)
    }

    /// Preprocesses decoded video clips as `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, VideoMAE
    /// geometry fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos_output(
        &self,
        videos: &[VideoClip],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_videos(videos)
            .map(ProcessorOutput::from_pixel_values)
    }
}

/// ViViT video processor for decoded frame clips.
#[derive(Clone, Debug)]
pub struct VivitImageProcessor {
    config: VivitImageProcessorConfig,
    processor: ImageProcessor,
}

impl VivitImageProcessor {
    /// Creates a processor from a validated ViViT configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when offset/rescale settings are incompatible or the
    /// tensor-stage image processor config is invalid.
    pub fn new(config: VivitImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_vivit_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &VivitImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses media from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, ViViT geometry, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        self.open_source(MediaSource::path(path.as_ref()))
    }

    /// Loads and preprocesses a media source.
    ///
    /// Images are treated as one-frame clips. Image sequences and videos are
    /// preprocessed in decoded frame order.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, ViViT geometry, or tensor
    /// conversion fails.
    pub fn open_source(&self, source: MediaSource) -> Result<Tensor, ImageProcessorError> {
        match DefaultMediaLoader.load_with_remote_options_and_image_decode_backend(
            source,
            &Default::default(),
            self.config.decode_backend,
        )? {
            LoadedMedia::Image(image) => self.preprocess_frames(std::slice::from_ref(&image)),
            LoadedMedia::ImageSequence(sequence) => self.preprocess_image_sequence(&sequence),
            LoadedMedia::Video(video) => self.preprocess_video(&video),
        }
    }

    /// Loads image paths and preprocesses them as a single decoded frame clip.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, loading fails, ViViT
    /// geometry fails, or tensor shapes are incompatible.
    pub fn open_frames<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        let images = paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend))
            .collect::<Result<Vec<_>, _>>()?;
        self.preprocess_frames(&images)
    }

    /// Preprocesses one decoded image frame as a one-frame clip.
    ///
    /// # Errors
    ///
    /// Returns an error when ViViT geometry or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_frames(std::slice::from_ref(image))
    }

    /// Preprocesses decoded image frames as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, ViViT geometry fails, or
    /// tensor shapes are incompatible.
    pub fn preprocess_frames(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let prepared = images
            .iter()
            .map(|image| vivit_prepare_frame(&self.config, image))
            .collect::<Result<Vec<_>, _>>()?;
        self.processor
            .preprocess_images(&prepared)?
            .with_leading_axis(TensorLeadingAxis::Frames)
            .map_err(ImageProcessorError::Tensor)
    }

    /// Preprocesses an image sequence as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, ViViT geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_image_sequence(
        &self,
        sequence: &ImageSequence,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_frames(sequence.frames())
    }

    /// Preprocesses an image sequence as a `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, ViViT geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_image_sequence_output(
        &self,
        sequence: &ImageSequence,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_sequence(sequence)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses a video clip as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, ViViT geometry fails, or
    /// tensor conversion fails.
    pub fn preprocess_video(&self, video: &VideoClip) -> Result<Tensor, ImageProcessorError> {
        if video.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let images = video
            .frames()
            .iter()
            .map(|frame| frame.image().clone())
            .collect::<Vec<_>>();
        self.preprocess_frames(&images)
    }

    /// Preprocesses a video clip as a `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when ViViT geometry or tensor conversion fails.
    pub fn preprocess_video_output(
        &self,
        video: &VideoClip,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_video(video)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded video clips as a batched ViViT tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, ViViT
    /// geometry fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos(&self, videos: &[VideoClip]) -> Result<Tensor, ImageProcessorError> {
        if videos.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let tensors = videos
            .iter()
            .map(|video| self.preprocess_video(video))
            .collect::<Result<Vec<_>, _>>()?;
        stack_frame_tensors_as_video_batch(tensors)
    }

    /// Preprocesses decoded video clips as `pixel_values` processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, ViViT
    /// geometry fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos_output(
        &self,
        videos: &[VideoClip],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_videos(videos)
            .map(ProcessorOutput::from_pixel_values)
    }
}

fn validate_vivit_config(config: &VivitImageProcessorConfig) -> Result<(), ImageProcessorError> {
    if config.offset && !config.do_rescale {
        return Err(ImageProcessorError::UnsupportedProcessorOption { field: "offset" });
    }
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn video_mae_prepare_frame(
    config: &VideoMaeImageProcessorConfig,
    image: &ImageFrame,
) -> Result<ImageFrame, ImageProcessorError> {
    prepare_video_shortest_edge_frame(
        image,
        config.size,
        config.crop_size,
        config.do_resize,
        config.do_center_crop,
        config.resize_decision(),
    )
}

fn vivit_prepare_frame(
    config: &VivitImageProcessorConfig,
    image: &ImageFrame,
) -> Result<ImageFrame, ImageProcessorError> {
    prepare_video_shortest_edge_frame(
        image,
        config.size,
        config.crop_size,
        config.do_resize,
        config.do_center_crop,
        config.resize_decision(),
    )
}
