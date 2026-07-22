//! Diffusers-oriented processor families.

use super::*;

/// Configuration for Diffusers BLIP image conditioning.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BlipImageProcessorConfig {
    /// Exact resize target used before optional center cropping.
    pub size: ImageSize,
    /// Whether images should be resized to `size`.
    pub do_resize: bool,
    /// Whether images should be center-cropped to `size`.
    pub do_center_crop: bool,
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
    /// Optional pixel-format conversion before tensor conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Whether image values should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied to pixel values when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether per-channel normalization should be applied.
    pub do_normalize: bool,
    /// Per-channel normalization means.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviations.
    pub image_std: Vec<f32>,
}

impl Default for BlipImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: square_size(BLIP_IMAGE_SIZE),
            do_resize: true,
            do_center_crop: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: Some(PixelFormat::Rgb8),
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: CLIP_IMAGE_MEAN.to_vec(),
            image_std: CLIP_IMAGE_STD.to_vec(),
        }
    }
}

impl BlipImageProcessorConfig {
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
        ImageProcessorConfig {
            do_resize: self.do_resize,
            height: Some(self.size.height),
            width: Some(self.size.width),
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: self.pixel_format,
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }
}

/// Configuration for Diffusers Flux2 image conditioning.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Flux2ImageProcessorConfig {
    /// Whether images should be resized to Flux2 VAE-scale multiples.
    pub do_resize: bool,
    /// VAE scale factor used to round image dimensions down.
    pub vae_scale_factor: usize,
    /// Channel count that represents Flux2 VAE latent tensors.
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

impl Default for Flux2ImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: FLUX2_VAE_SCALE_FACTOR,
            vae_latent_channels: FLUX2_VAE_LATENT_CHANNELS,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Lanczos,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: Some(PixelFormat::Rgb8),
            do_normalize: true,
            do_binarize: false,
        }
    }
}

impl Flux2ImageProcessorConfig {
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
        self.vae_image_processor_config().image_processor_config()
    }

    fn vae_image_processor_config(&self) -> VaeImageProcessorConfig {
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

/// Configuration for Diffusers HunyuanVideo 1.5 image/video conditioning.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HunyuanVideo15ImageProcessorConfig {
    /// Whether images and video frames should be resized to VAE-scale multiples.
    pub do_resize: bool,
    /// VAE scale factor used by HunyuanVideo 1.5.
    pub vae_scale_factor: usize,
    /// Channel count that represents HunyuanVideo 1.5 VAE latent tensors.
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

impl Default for HunyuanVideo15ImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: HUNYUAN_VIDEO_15_VAE_SCALE_FACTOR,
            vae_latent_channels: HUNYUAN_VIDEO_15_VAE_LATENT_CHANNELS,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Lanczos,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: Some(PixelFormat::Rgb8),
            do_normalize: true,
            do_binarize: false,
        }
    }
}

impl HunyuanVideo15ImageProcessorConfig {
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

    fn vae_image_processor_config_inner(&self) -> VaeImageProcessorConfig {
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

/// Configuration for Diffusers JoyImage edit image conditioning.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct JoyImageEditImageProcessorConfig {
    /// Whether input images should be bucket-resized and center-cropped.
    pub do_resize: bool,
    /// VAE scale factor retained for upstream config compatibility.
    pub vae_scale_factor: usize,
    /// Base bucket table size. The audited upstream implementation defines 1024.
    pub basesize: usize,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used for cover resize before center crop.
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
    /// Optional pixel-format conversion before tensor conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Whether image values should be normalized from `[0, 1]` to `[-1, 1]`.
    pub do_normalize: bool,
    /// Whether image values should be thresholded to `0.0` or `1.0`.
    pub do_binarize: bool,
}

impl Default for JoyImageEditImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: 8,
            basesize: JOY_IMAGE_BASE_SIZE,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: None,
            do_normalize: true,
            do_binarize: false,
        }
    }
}

impl JoyImageEditImageProcessorConfig {
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

    /// Returns the audited bucket table for this configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `basesize` is not supported by the audited
    /// JoyImage processor table.
    pub fn bucket_candidates(&self) -> Result<&'static [ImageSize], ImageProcessorError> {
        joy_image_bucket_candidates(self.basesize)
    }

    /// Selects the JoyImage bucket closest to `source_size`.
    ///
    /// # Errors
    ///
    /// Returns an error when the source dimensions are invalid or the base
    /// bucket table is unsupported.
    pub fn target_size_for_size(
        &self,
        source_size: ImageSize,
    ) -> Result<ImageSize, ImageProcessorError> {
        if !self.do_resize {
            return Ok(source_size);
        }
        select_aspect_ratio_bucket(source_size, self.bucket_candidates()?)
            .map_err(ImageProcessorError::Transform)
    }

    /// Selects the JoyImage bucket closest to a decoded frame.
    ///
    /// # Errors
    ///
    /// Returns an error when the image dimensions are invalid or the base
    /// bucket table is unsupported.
    pub fn target_size_for_image(
        &self,
        image: &ImageFrame,
    ) -> Result<ImageSize, ImageProcessorError> {
        self.target_size_for_size(ImageSize::new(image.height(), image.width())?)
    }

    /// Converts this family config into the tensor-stage processor config.
    ///
    /// JoyImage resolves bucket geometry in the wrapper before invoking this
    /// generic tensor conversion stage.
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
            pixel_format: self.pixel_format,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: self.do_normalize,
            image_mean: DIFFUSION_VAE_IMAGE_MEAN.to_vec(),
            image_std: DIFFUSION_VAE_IMAGE_STD.to_vec(),
            do_binarize: self.do_binarize,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }
}

/// Configuration for Diffusers Wan Animate image conditioning.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WanAnimateImageProcessorConfig {
    /// Whether input images should be resized to Wan spatial multiples.
    pub do_resize: bool,
    /// VAE scale factor used with `spatial_patch_size` for target multiples.
    pub vae_scale_factor: usize,
    /// Channel count that represents Wan VAE latent tensors.
    pub vae_latent_channels: usize,
    /// Spatial patch height and width used by the diffusion transformer.
    pub spatial_patch_size: ImageSize,
    /// Device-agnostic image output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used for fit-inside resize before canvas fill.
    pub resample: ResizeFilter,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
    /// Optional pixel-format conversion before tensor conversion.
    pub pixel_format: Option<PixelFormat>,
    /// Canvas fill used by Wan `resize_mode="fill"` preprocessing.
    pub fill: CanvasFill,
    /// Whether image values should be normalized from `[0, 1]` to `[-1, 1]`.
    pub do_normalize: bool,
    /// Whether image values should be thresholded to `0.0` or `1.0`.
    pub do_binarize: bool,
}

impl Default for WanAnimateImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: 8,
            vae_latent_channels: 16,
            spatial_patch_size: ImageSize {
                height: 2,
                width: 2,
            },
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Lanczos,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: Some(PixelFormat::Rgb8),
            fill: CanvasFill::Constant(vec![0]),
            do_normalize: true,
            do_binarize: false,
        }
    }
}

impl WanAnimateImageProcessorConfig {
    /// Returns the image output axis order.
    pub fn output_image_layout(&self) -> ImageLayout {
        self.output_layout
    }

    /// Returns the video output axis order implied by `output_layout`.
    pub fn output_video_layout(&self) -> VideoLayout {
        video_layout_for_image_layout(self.output_layout)
    }

    /// Selects the Wan rounded target for a source or requested size.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured multiples are invalid or rounding
    /// would produce a zero-sized target.
    pub fn target_size_for_size(
        &self,
        source_size: ImageSize,
    ) -> Result<ImageSize, ImageProcessorError> {
        if !self.do_resize {
            return Ok(source_size);
        }
        wan_animate_target_size_for_size(self, source_size)
    }

    /// Selects the Wan rounded target for a decoded frame.
    ///
    /// # Errors
    ///
    /// Returns an error when the image dimensions or configured multiples are
    /// invalid.
    pub fn target_size_for_image(
        &self,
        image: &ImageFrame,
    ) -> Result<ImageSize, ImageProcessorError> {
        self.target_size_for_size(ImageSize::new(image.height(), image.width())?)
    }

    /// Converts this family config into the tensor-stage processor config.
    ///
    /// Wan resolves resize-fill geometry in the wrapper before invoking this
    /// generic tensor conversion stage.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: false,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: self.pixel_format,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: self.do_normalize,
            image_mean: DIFFUSION_VAE_IMAGE_MEAN.to_vec(),
            image_std: DIFFUSION_VAE_IMAGE_STD.to_vec(),
            do_binarize: self.do_binarize,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }
}

/// Configuration for Diffusers LTX2 HDR video processing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Ltx2VideoHdrProcessorConfig {
    /// Whether the base VAE stage rounds both frame dimensions down to
    /// `vae_scale_factor` multiples before HDR resize-and-pad geometry.
    #[serde(default = "default_true")]
    pub do_resize: bool,
    /// VAE spatial scale factor retained for upstream config compatibility.
    pub vae_scale_factor: usize,
    /// Device-agnostic video output axis order used for reference conditioning.
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
}

impl Default for Ltx2VideoHdrProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: 32,
            output_layout: VideoLayout::FramesChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
        }
    }
}

impl Ltx2VideoHdrProcessorConfig {
    /// Returns the video output axis order.
    pub fn output_video_layout(&self) -> VideoLayout {
        self.output_layout
    }

    /// Returns the resize decision implied by `resample`.
    pub fn resize_decision(&self) -> ResizeDecision {
        resize_decision_for_parts(self.resample, self.resize_parity)
    }

    /// Converts this family config into the base VAE image processor config.
    pub fn vae_image_processor_config(&self) -> VaeImageProcessorConfig {
        VaeImageProcessorConfig {
            do_resize: self.do_resize,
            vae_scale_factor: self.vae_scale_factor,
            vae_latent_channels: 4,
            output_layout: match self.output_layout {
                VideoLayout::FramesChannelsHeightWidth => ImageLayout::ChannelsHeightWidth,
                VideoLayout::FramesHeightWidthChannels => ImageLayout::HeightWidthChannels,
            },
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(PixelFormat::Rgb8),
            do_normalize: true,
            do_binarize: false,
        }
    }

    /// Converts this family config into the generic processor config used by
    /// the base VAE preprocessing stage.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        self.vae_image_processor_config().image_processor_config()
    }

    /// Converts this config into the shared LTX2 LogC3 postprocess recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the static recipe id or output name is invalid.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        ProcessorRecipe::logc3_hdr_video_postprocess(
            "diffusers.ltx2_video_hdr_postprocess",
            "decoded_video",
        )
    }
}

/// Diffusers LTX2 HDR video processor for decoded reference clips and outputs.
#[derive(Clone, Debug)]
pub struct Ltx2VideoHdrProcessor {
    config: Ltx2VideoHdrProcessorConfig,
    processor: VaeImageProcessor,
}

impl Ltx2VideoHdrProcessor {
    /// Creates a processor from a validated LTX2 HDR configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the VAE scale factor, resize policy, or
    /// tensor-stage image processor config is invalid.
    pub fn new(config: Ltx2VideoHdrProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_ltx2_video_hdr_config(&config)?;
        let processor = VaeImageProcessor::new(config.vae_image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's family configuration.
    pub fn config(&self) -> &Ltx2VideoHdrProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        self.processor.image_processor()
    }

    /// Returns the base VAE processor used for the first reference-video stage.
    pub fn vae_processor(&self) -> &VaeImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses media from a filesystem path as LTX2 HDR reference input.
    ///
    /// Images are treated as one-frame clips. Image sequences and videos are
    /// preprocessed in decoded frame order.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, LTX2 geometry, or tensor
    /// conversion fails.
    pub fn open_reference_video_hdr(
        &self,
        path: impl AsRef<Path>,
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        self.open_reference_source_hdr(MediaSource::path(path.as_ref()), target_size)
    }

    /// Loads and preprocesses a media source as LTX2 HDR reference input.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, LTX2 geometry, or tensor
    /// conversion fails.
    pub fn open_reference_source_hdr(
        &self,
        source: MediaSource,
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        match DefaultMediaLoader.load_with_remote_options_and_image_decode_backend(
            source,
            &Default::default(),
            self.config.decode_backend,
        )? {
            LoadedMedia::Image(image) => {
                self.preprocess_reference_frames_hdr(std::slice::from_ref(&image), target_size)
            }
            LoadedMedia::ImageSequence(sequence) => {
                self.preprocess_reference_image_sequence_hdr(&sequence, target_size)
            }
            LoadedMedia::Video(video) => self.preprocess_reference_video_hdr(&video, target_size),
        }
    }

    /// Preprocesses one decoded image frame as a one-frame LTX2 HDR reference clip.
    ///
    /// # Errors
    ///
    /// Returns an error when LTX2 geometry or tensor conversion fails.
    pub fn preprocess_reference_image_hdr(
        &self,
        image: &ImageFrame,
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_reference_frames_hdr(std::slice::from_ref(image), target_size)
    }

    /// Preprocesses decoded frames as an LTX2 HDR reference clip.
    ///
    /// The frame geometry follows upstream LTX2 reference conditioning. The
    /// default first rounds both dimensions down to VAE-scale multiples using
    /// Pillow-compatible resizing and normalizes to `[-1, 1]`. A second f32
    /// bilinear stage fits inside `target_size`, then pads the bottom and right
    /// by reflection when valid or edge extension otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, geometry fails, or tensor
    /// conversion fails.
    pub fn preprocess_reference_frames_hdr(
        &self,
        images: &[ImageFrame],
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let vae_preprocessed = self.processor.preprocess_images(images)?;
        ltx2_resize_and_pad_f32_tensor(&vae_preprocessed, target_size, self.config.output_layout)
    }

    /// Preprocesses an image sequence as an LTX2 HDR reference clip.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, geometry fails, or tensor
    /// conversion fails.
    pub fn preprocess_reference_image_sequence_hdr(
        &self,
        sequence: &ImageSequence,
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_reference_frames_hdr(sequence.frames(), target_size)
    }

    /// Preprocesses a decoded video clip as LTX2 HDR reference input.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, geometry fails, or tensor
    /// conversion fails.
    pub fn preprocess_reference_video_hdr(
        &self,
        video: &VideoClip,
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        if video.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let images = video
            .frames()
            .iter()
            .map(|frame| frame.image().clone())
            .collect::<Vec<_>>();
        self.preprocess_reference_frames_hdr(&images, target_size)
    }

    /// Preprocesses decoded video clips as batched LTX2 HDR reference input.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, geometry
    /// fails, or processed clip shapes are incompatible.
    pub fn preprocess_reference_videos_hdr(
        &self,
        videos: &[VideoClip],
        target_size: ImageSize,
    ) -> Result<Tensor, ImageProcessorError> {
        if videos.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let tensors = videos
            .iter()
            .map(|video| self.preprocess_reference_video_hdr(video, target_size))
            .collect::<Result<Vec<_>, _>>()?;
        stack_frame_tensors_as_video_batch(tensors)
    }

    /// Restores VAE-decoded LogC3 output to linear HDR `BFHWC` tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not a supported video layout, values
    /// are non-finite, or output tensor construction fails.
    pub fn postprocess_hdr_video(&self, tensor: &Tensor) -> Result<Tensor, ImageProcessorError> {
        post_process_logc3_hdr_video_tensor(tensor)
            .map_err(image_processor_error_from_tensor_postprocess)
    }
}

/// Diffusers BLIP image processor.
#[derive(Clone, Debug)]
pub struct BlipImageProcessor {
    config: BlipImageProcessorConfig,
    processor: ImageProcessor,
    postprocessor: VaeImageProcessor,
}

impl BlipImageProcessor {
    /// Creates a processor from a validated BLIP configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when BLIP dimensions, resize policy, normalization
    /// settings, or postprocess settings are invalid.
    pub fn new(config: BlipImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_blip_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        let postprocessor = VaeImageProcessor::new(VaeImageProcessorConfig {
            do_resize: false,
            pixel_format: Some(PixelFormat::Rgb8),
            do_normalize: true,
            ..Default::default()
        })?;
        Ok(Self {
            config,
            processor,
            postprocessor,
        })
    }

    /// Returns this processor's BLIP configuration.
    pub fn config(&self) -> &BlipImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying tensor-stage image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, BLIP preprocessing, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Loads and preprocesses image paths as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or preprocessing fails.
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

    /// Preprocesses one decoded image.
    ///
    /// # Errors
    ///
    /// Returns an error when BLIP resize, optional center crop,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image with per-call size options.
    ///
    /// # Errors
    ///
    /// Returns an error when BLIP resize, optional center crop,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let target = blip_target_size(&self.config, options)?;
        let input = blip_prepare_frame(&self.config, image, target)?;
        self.processor
            .preprocess_image_with_options(input.as_ref(), blip_preprocess_options(target))
    }

    /// Preprocesses decoded images as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_images_with_options(images, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded images as a batch tensor with per-call size options.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        let target = blip_target_size(&self.config, options)?;
        if self.config.do_center_crop && !self.config.do_resize {
            let prepared = images
                .iter()
                .map(|image| blip_prepare_frame(&self.config, image, target).map(Cow::into_owned))
                .collect::<Result<Vec<_>, _>>()?;
            return self
                .processor
                .preprocess_images_with_options(&prepared, blip_preprocess_options(target));
        }

        self.processor
            .preprocess_images_with_options(images, blip_preprocess_options(target))
    }

    /// Preprocesses one decoded image as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image(image)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded images as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images(images)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Postprocesses a BLIP sample tensor.
    ///
    /// `Tensor`, `Array`, and `Frames` correspond to upstream `pt`, `np`, and
    /// `pil` outputs respectively.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor layout, denormalization flags, or
    /// frame construction are invalid.
    pub fn postprocess(
        &self,
        tensor: &Tensor,
        output_type: VaeOutputType,
        do_denormalize: Option<&[bool]>,
    ) -> Result<VaePostprocessOutput, ImageProcessorError> {
        self.postprocessor
            .postprocess(tensor, output_type, do_denormalize)
    }
}

/// Diffusers Flux2 image processor.
#[derive(Clone, Debug)]
pub struct Flux2ImageProcessor {
    config: Flux2ImageProcessorConfig,
    processor: VaeImageProcessor,
}

impl Flux2ImageProcessor {
    /// Creates a processor from a validated Flux2 configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when Flux2 VAE dimensions are invalid or the derived
    /// generic image processor config is invalid.
    pub fn new(config: Flux2ImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_flux2_config(&config)?;
        let processor = VaeImageProcessor::new(config.vae_image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's Flux2 configuration.
    pub fn config(&self) -> &Flux2ImageProcessorConfig {
        &self.config
    }

    /// Returns the VAE-style processor used for Flux2 tensor conversion.
    pub fn vae_processor(&self) -> &VaeImageProcessor {
        &self.processor
    }

    /// Returns the underlying generic image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        self.processor.image_processor()
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, Flux2 VAE resize planning, or
    /// tensor conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        self.processor.open(path)
    }

    /// Loads and preprocesses image paths as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or preprocessing fails.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        self.processor.open_batch(paths)
    }

    /// Preprocesses one decoded reference image.
    ///
    /// # Errors
    ///
    /// Returns an error when Flux2 VAE resize planning, resizing,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.processor.preprocess_image(image)
    }

    /// Preprocesses one decoded reference image with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when Flux2 VAE resize planning, resizing,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.processor.preprocess_image_with_options(image, options)
    }

    /// Preprocesses decoded reference images as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.processor.preprocess_images(images)
    }

    /// Preprocesses decoded reference images as a batch tensor with options.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.processor
            .preprocess_images_with_options(images, options)
    }

    /// Preprocesses one decoded reference image as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.processor.preprocess_image_output(image)
    }

    /// Preprocesses decoded reference images as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.processor.preprocess_images_output(images)
    }

    /// Validates Flux2 reference-image minimum side and aspect-ratio limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the image is smaller than Flux2's minimum side
    /// length or exceeds Flux2's maximum aspect ratio.
    pub fn check_image_input(&self, image: &ImageFrame) -> Result<(), ImageProcessorError> {
        flux2_check_image_input(image, FLUX2_MIN_SIDE_LENGTH, FLUX2_MAX_ASPECT_RATIO)
    }

    /// Resizes an image to the default Flux2 maximum area when needed.
    ///
    /// # Errors
    ///
    /// Returns an error when area-limit planning, resizing, or frame
    /// construction fails.
    pub fn resize_if_exceeds_default_area(
        &self,
        image: &ImageFrame,
    ) -> Result<ImageFrame, ImageProcessorError> {
        self.resize_if_exceeds_area(image, FLUX2_DEFAULT_MAX_AREA)
    }

    /// Resizes an image to `target_area` when its pixel area is larger.
    ///
    /// # Errors
    ///
    /// Returns an error when `target_area` is zero, dimensions overflow,
    /// resizing fails, or output frame construction fails.
    pub fn resize_if_exceeds_area(
        &self,
        image: &ImageFrame,
        target_area: usize,
    ) -> Result<ImageFrame, ImageProcessorError> {
        flux2_resize_if_exceeds_area(image, target_area)
    }

    /// Applies Flux2's center crop utility to an already-sized image.
    ///
    /// # Errors
    ///
    /// Returns an error when `target` is invalid, larger than the source, or
    /// output frame construction fails.
    pub fn resize_and_crop_image(
        &self,
        image: &ImageFrame,
        target: ImageSize,
    ) -> Result<ImageFrame, ImageProcessorError> {
        center_crop_frame(image, target).map_err(ImageProcessorError::Transform)
    }

    /// Concatenates Flux2 reference images horizontally on a white RGB canvas.
    ///
    /// A single image is returned as a clone, matching the upstream helper.
    ///
    /// # Errors
    ///
    /// Returns an error when no images are provided, dimensions overflow, RGB
    /// conversion fails, or output frame construction fails.
    pub fn concatenate_images(
        &self,
        images: &[ImageFrame],
    ) -> Result<ImageFrame, ImageProcessorError> {
        flux2_concatenate_images(images)
    }
}

/// Diffusers HunyuanVideo 1.5 image/video processor.
#[derive(Clone, Debug)]
pub struct HunyuanVideo15ImageProcessor {
    config: HunyuanVideo15ImageProcessorConfig,
    processor: VaeImageProcessor,
}

impl HunyuanVideo15ImageProcessor {
    /// Creates a processor from a validated HunyuanVideo 1.5 configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when HunyuanVideo VAE dimensions are invalid or the
    /// derived generic image processor config is invalid.
    pub fn new(config: HunyuanVideo15ImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_hunyuan_video_15_config(&config)?;
        let processor = VaeImageProcessor::new(config.vae_image_processor_config_inner())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's HunyuanVideo 1.5 configuration.
    pub fn config(&self) -> &HunyuanVideo15ImageProcessorConfig {
        &self.config
    }

    /// Returns the VAE-style processor used for tensor conversion.
    pub fn vae_processor(&self) -> &VaeImageProcessor {
        &self.processor
    }

    /// Returns the underlying generic image processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        self.processor.image_processor()
    }

    /// Returns HunyuanVideo 1.5 candidate spatial buckets for `target_size`.
    ///
    /// # Errors
    ///
    /// Returns an error when `target_size`, the configured scale factor, or
    /// generated bucket dimensions are invalid.
    pub fn bucket_candidates(
        &self,
        target_size: usize,
    ) -> Result<Vec<ImageSize>, ImageProcessorError> {
        hunyuan_video_15_bucket_candidates(
            target_size,
            self.config.vae_scale_factor,
            HUNYUAN_VIDEO_15_MAX_BUCKET_RATIO,
        )
    }

    /// Selects the default HunyuanVideo 1.5 height and width for a source size.
    ///
    /// # Errors
    ///
    /// Returns an error when source or bucket dimensions are invalid.
    pub fn calculate_default_height_width(
        &self,
        height: usize,
        width: usize,
        target_size: usize,
    ) -> Result<ImageSize, ImageProcessorError> {
        hunyuan_video_15_default_height_width(
            ImageSize::new(height, width)?,
            target_size,
            self.config.vae_scale_factor,
        )
    }

    /// Preprocesses one decoded image with VAE-style tensor conversion.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE resize planning, resizing, normalization, or
    /// tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.processor.preprocess_image(image)
    }

    /// Preprocesses one decoded image with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE resize planning, resizing, normalization, or
    /// tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.processor.preprocess_image_with_options(image, options)
    }

    /// Preprocesses decoded frames as one frame-leading video tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame list is empty, preprocessing fails, or
    /// tensor layout conversion fails.
    pub fn preprocess_frames(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        self.processor
            .preprocess_images(images)?
            .with_leading_axis(TensorLeadingAxis::Frames)
            .map_err(ImageProcessorError::Tensor)
    }

    /// Preprocesses a decoded video clip as a frame-leading tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty or preprocessing fails.
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

    /// Preprocesses decoded video clips as a batched video tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, or shapes
    /// are incompatible.
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
}

/// Diffusers JoyImage edit image processor.
#[derive(Clone, Debug)]
pub struct JoyImageEditImageProcessor {
    config: JoyImageEditImageProcessorConfig,
    processor: ImageProcessor,
}

impl JoyImageEditImageProcessor {
    /// Creates a processor from a validated JoyImage edit configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when JoyImage dimensions are invalid, the bucket base
    /// is unsupported, or the derived generic image processor config is invalid.
    pub fn new(config: JoyImageEditImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_joy_image_edit_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's JoyImage edit configuration.
    pub fn config(&self) -> &JoyImageEditImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic tensor-stage processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, JoyImage geometry, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when JoyImage bucket selection, resizing, cropping,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame with per-call dimensions.
    ///
    /// When both `height` and `width` are provided, JoyImage selects the
    /// nearest bucket from those requested dimensions, matching upstream.
    ///
    /// # Errors
    ///
    /// Returns an error when JoyImage bucket selection, resizing, cropping,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let prepared = joy_image_prepare_frame(&self.config, image, options)?;
        self.processor.preprocess_image(&prepared)
    }

    /// Preprocesses decoded image frames as a batch tensor.
    ///
    /// The default JoyImage bucket is derived from the first image, matching
    /// Diffusers list-input behavior for image preprocessors.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_images_with_options(images, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded image frames as a batch tensor with per-call dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let first = images.first().ok_or(ImageProcessorError::EmptyBatch)?;
        let target = joy_image_target_size(&self.config, first, options)?;
        let prepared = images
            .iter()
            .map(|image| joy_image_resize_center_crop_to_target(&self.config, image, target))
            .collect::<Result<Vec<_>, _>>()?;
        self.processor.preprocess_images(&prepared)
    }

    /// Applies JoyImage bucket resize and center crop without tensor conversion.
    ///
    /// # Errors
    ///
    /// Returns an error when bucket selection, resizing, cropping, or output
    /// frame validation fails.
    pub fn resize_center_crop_image(
        &self,
        image: &ImageFrame,
    ) -> Result<ImageFrame, ImageProcessorError> {
        self.resize_center_crop_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Applies JoyImage bucket resize and center crop with per-call dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when bucket selection, resizing, cropping, or output
    /// frame validation fails.
    pub fn resize_center_crop_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageFrame, ImageProcessorError> {
        joy_image_prepare_frame(&self.config, image, options)
    }

    /// Preprocesses one decoded image frame as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image(image)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded image frames as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images(images)
            .map(ProcessorOutput::from_pixel_values)
    }
}

/// Diffusers Wan Animate image processor.
#[derive(Clone, Debug)]
pub struct WanAnimateImageProcessor {
    config: WanAnimateImageProcessorConfig,
    processor: ImageProcessor,
}

impl WanAnimateImageProcessor {
    /// Creates a processor from a validated Wan Animate configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when Wan dimensions are invalid or the derived generic
    /// image processor config is invalid.
    pub fn new(config: WanAnimateImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_wan_animate_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's Wan Animate configuration.
    pub fn config(&self) -> &WanAnimateImageProcessorConfig {
        &self.config
    }

    /// Returns the underlying generic tensor-stage processor.
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.processor
    }

    /// Loads and preprocesses an image from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, Wan geometry, or tensor
    /// conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        let image = load_image_from_path_with_backend(path, self.config.decode_backend)?;
        self.preprocess_image(&image)
    }

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when Wan target rounding, resize-fill geometry,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame with per-call dimensions.
    ///
    /// Per-call dimensions are rounded down to multiples of
    /// `vae_scale_factor * spatial_patch_size`, matching upstream Wan.
    ///
    /// # Errors
    ///
    /// Returns an error when Wan target rounding, resize-fill geometry,
    /// normalization, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let prepared = wan_animate_prepare_frame(&self.config, image, options)?;
        self.processor.preprocess_image(&prepared)
    }

    /// Preprocesses decoded image frames as a batch tensor.
    ///
    /// The default Wan target is derived from the first image, matching
    /// Diffusers list-input behavior for image preprocessors.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_images_with_options(images, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded image frames as a batch tensor with per-call dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let first = images.first().ok_or(ImageProcessorError::EmptyBatch)?;
        let target = wan_animate_target_size(&self.config, first, options)?;
        let prepared = images
            .iter()
            .map(|image| wan_animate_resize_fill_to_target(&self.config, image, target))
            .collect::<Result<Vec<_>, _>>()?;
        self.processor.preprocess_images(&prepared)
    }

    /// Applies Wan resize-fill geometry without tensor conversion.
    ///
    /// # Errors
    ///
    /// Returns an error when target rounding, resizing, filling, or output
    /// frame validation fails.
    pub fn resize_fill_image(&self, image: &ImageFrame) -> Result<ImageFrame, ImageProcessorError> {
        self.resize_fill_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Applies Wan resize-fill geometry with per-call dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when target rounding, resizing, filling, or output
    /// frame validation fails.
    pub fn resize_fill_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageFrame, ImageProcessorError> {
        wan_animate_prepare_frame(&self.config, image, options)
    }

    /// Preprocesses one decoded image frame as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image(image)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded image frames as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing fails.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images(images)
            .map(ProcessorOutput::from_pixel_values)
    }
}

fn validate_blip_config(config: &BlipImageProcessorConfig) -> Result<(), ImageProcessorError> {
    ImageSize::new(config.size.height, config.size.width)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn blip_target_size(
    config: &BlipImageProcessorConfig,
    options: ImageProcessorOptions,
) -> Result<ImageSize, ImageProcessorError> {
    ImageSize::new(
        options.height.unwrap_or(config.size.height),
        options.width.unwrap_or(config.size.width),
    )
    .map_err(ImageProcessorError::Transform)
}

fn blip_preprocess_options(target: ImageSize) -> ImageProcessorOptions {
    ImageProcessorOptions {
        height: Some(target.height),
        width: Some(target.width),
        resize_mode: Some(ResizeMode::Default),
    }
}

fn blip_prepare_frame<'a>(
    config: &BlipImageProcessorConfig,
    image: &'a ImageFrame,
    target: ImageSize,
) -> Result<Cow<'a, ImageFrame>, ImageProcessorError> {
    if config.do_center_crop && !config.do_resize {
        return center_crop_frame(image, target)
            .map(Cow::Owned)
            .map_err(ImageProcessorError::Transform);
    }
    Ok(Cow::Borrowed(image))
}

fn validate_flux2_config(config: &Flux2ImageProcessorConfig) -> Result<(), ImageProcessorError> {
    validate_vae_config(&config.vae_image_processor_config())
}

fn validate_hunyuan_video_15_config(
    config: &HunyuanVideo15ImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    validate_vae_config(&config.vae_image_processor_config_inner())
}

fn validate_joy_image_edit_config(
    config: &JoyImageEditImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    if config.vae_scale_factor == 0 {
        return Err(TransformError::InvalidScaleFactor(config.vae_scale_factor).into());
    }
    joy_image_bucket_candidates(config.basesize)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn validate_wan_animate_config(
    config: &WanAnimateImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    if config.vae_scale_factor == 0 {
        return Err(TransformError::InvalidScaleFactor(config.vae_scale_factor).into());
    }
    if config.vae_latent_channels == 0 {
        return Err(ImageProcessorError::InvalidVaeLatentChannels(
            config.vae_latent_channels,
        ));
    }
    ImageSize::new(
        config.spatial_patch_size.height,
        config.spatial_patch_size.width,
    )?;
    let multiples = wan_animate_multiples(config)?;
    if multiples.height == 0 || multiples.width == 0 {
        return Err(TransformError::InvalidScaleFactor(config.vae_scale_factor).into());
    }
    Ok(())
}

fn flux2_check_image_input(
    image: &ImageFrame,
    min_side_length: usize,
    max_aspect_ratio: f64,
) -> Result<(), ImageProcessorError> {
    let image_size = ImageSize::new(image.height(), image.width())?;
    validate_image_size_constraints(
        image_size,
        ImageSizeConstraints {
            max_aspect_ratio,
            min_side_length,
        },
    )
    .map_err(ImageProcessorError::Transform)
}

fn flux2_resize_if_exceeds_area(
    image: &ImageFrame,
    target_area: usize,
) -> Result<ImageFrame, ImageProcessorError> {
    let plan = area_resize_plan(ImageSize::new(image.height(), image.width())?, target_area)?;
    if !plan.should_resize() {
        return Ok(image.clone());
    }
    resize_frame_with_decision(
        image,
        plan.resized_size,
        ResizeDecision::compatibility(ResizeFilter::Lanczos),
        ResizeMode::Default,
    )
    .map_err(ImageProcessorError::Transform)
}

fn flux2_concatenate_images(images: &[ImageFrame]) -> Result<ImageFrame, ImageProcessorError> {
    if images.len() == 1 {
        return Ok(images[0].clone());
    }
    concatenate_frames_horizontally_rgb(images, [255, 255, 255])
        .map_err(ImageProcessorError::Transform)
}

fn hunyuan_video_15_default_height_width(
    source: ImageSize,
    target_size: usize,
    patch_size: usize,
) -> Result<ImageSize, ImageProcessorError> {
    let candidates = hunyuan_video_15_bucket_candidates(
        target_size,
        patch_size,
        HUNYUAN_VIDEO_15_MAX_BUCKET_RATIO,
    )?;
    hunyuan_video_15_closest_ratio(source, &candidates)
}

fn hunyuan_video_15_bucket_candidates(
    base_size: usize,
    patch_size: usize,
    max_ratio: f64,
) -> Result<Vec<ImageSize>, ImageProcessorError> {
    validate_positive_dimension("base_size", base_size)?;
    validate_positive_dimension("patch_size", patch_size)?;
    if !max_ratio.is_finite() || max_ratio < 1.0 {
        return Err(TransformError::InvalidAspectRatio(max_ratio).into());
    }

    let patches_per_side = base_size as f64 / patch_size as f64;
    let num_patches = python_round_positive_to_usize(patches_per_side * patches_per_side)?;
    if num_patches == 0 {
        return Err(TransformError::EmptyResolutionCandidates.into());
    }

    let mut crop_size_list = Vec::new();
    let mut width_patches = num_patches;
    let mut height_patches = 1usize;
    while width_patches > 0 {
        let ratio =
            width_patches.max(height_patches) as f64 / width_patches.min(height_patches) as f64;
        if ratio <= max_ratio {
            crop_size_list.push(ImageSize::new(
                width_patches
                    .checked_mul(patch_size)
                    .ok_or(TransformError::ImageSizeOverflow)?,
                height_patches
                    .checked_mul(patch_size)
                    .ok_or(TransformError::ImageSizeOverflow)?,
            )?);
        }

        if height_patches
            .checked_add(1)
            .and_then(|value| value.checked_mul(width_patches))
            .is_some_and(|value| value <= num_patches)
        {
            height_patches += 1;
        } else {
            width_patches -= 1;
        }
    }

    if crop_size_list.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates.into());
    }
    Ok(crop_size_list)
}

fn hunyuan_video_15_closest_ratio(
    source: ImageSize,
    candidates: &[ImageSize],
) -> Result<ImageSize, ImageProcessorError> {
    if candidates.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates.into());
    }
    let aspect_ratio = source.height as f64 / source.width as f64;
    let mut best: Option<(ImageSize, f64)> = None;
    for &candidate in candidates {
        let candidate_ratio =
            round_to_decimal_places(candidate.height as f64 / candidate.width as f64, 5);
        let diff = candidate_ratio - aspect_ratio;
        let directional_match = if aspect_ratio >= 1.0 {
            diff <= 0.0
        } else {
            diff >= 0.0
        };
        if directional_match && best.is_none_or(|(_, best_diff)| diff.abs() < best_diff.abs()) {
            best = Some((candidate, diff));
        }
    }

    best.map(|(size, _)| size)
        .ok_or_else(|| TransformError::EmptyResolutionCandidates.into())
}

fn python_round_positive_to_usize(value: f64) -> Result<usize, ImageProcessorError> {
    if !value.is_finite() || value < 0.0 {
        return Err(TransformError::InvalidFloatScaleFactor(value as f32).into());
    }
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else {
        let floor_as_u128 = floor as u128;
        if floor_as_u128.is_multiple_of(2) {
            floor
        } else {
            floor + 1.0
        }
    };
    if rounded > usize::MAX as f64 {
        return Err(TransformError::ImageSizeOverflow.into());
    }
    Ok(rounded as usize)
}

fn round_to_decimal_places(value: f64, places: i32) -> f64 {
    let scale = 10_f64.powi(places);
    (value * scale).round() / scale
}

fn validate_ltx2_video_hdr_config(
    config: &Ltx2VideoHdrProcessorConfig,
) -> Result<(), ImageProcessorError> {
    validate_vae_config(&config.vae_image_processor_config())
}

fn joy_image_bucket_candidates(
    basesize: usize,
) -> Result<&'static [ImageSize], ImageProcessorError> {
    match basesize {
        JOY_IMAGE_BASE_SIZE => Ok(&JOY_IMAGE_BUCKETS_1024),
        _ => Err(ImageProcessorError::UnsupportedProcessorOption { field: "basesize" }),
    }
}

fn joy_image_target_size(
    config: &JoyImageEditImageProcessorConfig,
    image: &ImageFrame,
    options: ImageProcessorOptions,
) -> Result<ImageSize, ImageProcessorError> {
    let source_size = match (options.height, options.width) {
        (Some(height), Some(width)) => ImageSize::new(height, width)?,
        _ => ImageSize::new(image.height(), image.width())?,
    };
    config.target_size_for_size(source_size)
}

fn joy_image_prepare_frame(
    config: &JoyImageEditImageProcessorConfig,
    image: &ImageFrame,
    options: ImageProcessorOptions,
) -> Result<ImageFrame, ImageProcessorError> {
    let target = joy_image_target_size(config, image, options)?;
    joy_image_resize_center_crop_to_target(config, image, target)
}

fn joy_image_resize_center_crop_to_target(
    config: &JoyImageEditImageProcessorConfig,
    image: &ImageFrame,
    target: ImageSize,
) -> Result<ImageFrame, ImageProcessorError> {
    if !config.do_resize {
        return Ok(image.clone());
    }
    resize_center_crop_frame_with_decision(
        image,
        target,
        config.resize_decision(),
        ResizeRounding::Ceil,
    )
    .map_err(ImageProcessorError::Transform)
}

fn wan_animate_multiples(
    config: &WanAnimateImageProcessorConfig,
) -> Result<ImageSize, ImageProcessorError> {
    ImageSize::new(
        config
            .vae_scale_factor
            .checked_mul(config.spatial_patch_size.height)
            .ok_or(TransformError::ImageSizeOverflow)?,
        config
            .vae_scale_factor
            .checked_mul(config.spatial_patch_size.width)
            .ok_or(TransformError::ImageSizeOverflow)?,
    )
    .map_err(ImageProcessorError::Transform)
}

fn wan_animate_target_size_for_size(
    config: &WanAnimateImageProcessorConfig,
    source_size: ImageSize,
) -> Result<ImageSize, ImageProcessorError> {
    let multiples = wan_animate_multiples(config)?;
    ImageSize::new(
        source_size.height - source_size.height % multiples.height,
        source_size.width - source_size.width % multiples.width,
    )
    .map_err(ImageProcessorError::Transform)
}

fn wan_animate_target_size(
    config: &WanAnimateImageProcessorConfig,
    image: &ImageFrame,
    options: ImageProcessorOptions,
) -> Result<ImageSize, ImageProcessorError> {
    let source_size = ImageSize::new(
        options.height.unwrap_or_else(|| image.height()),
        options.width.unwrap_or_else(|| image.width()),
    )?;
    config.target_size_for_size(source_size)
}

fn wan_animate_prepare_frame(
    config: &WanAnimateImageProcessorConfig,
    image: &ImageFrame,
    options: ImageProcessorOptions,
) -> Result<ImageFrame, ImageProcessorError> {
    let target = wan_animate_target_size(config, image, options)?;
    wan_animate_resize_fill_to_target(config, image, target)
}

fn wan_animate_resize_fill_to_target(
    config: &WanAnimateImageProcessorConfig,
    image: &ImageFrame,
    target: ImageSize,
) -> Result<ImageFrame, ImageProcessorError> {
    if !config.do_resize {
        return Ok(image.clone());
    }
    let rgb = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
    let plan = resize_fill_plan(ImageSize::new(rgb.height(), rgb.width())?, target)?;
    let resized = resize_frame_with_decision(
        &rgb,
        plan.resized_size,
        ResizeDecision::compatibility(config.resample),
        ResizeMode::Default,
    )?;
    pad_frame_with_canvas_fill(&resized, plan.padding, config.fill.clone())
        .map_err(ImageProcessorError::Transform)
}

fn ltx2_resize_and_pad_f32_tensor(
    tensor: &Tensor,
    target_size: ImageSize,
    output_layout: VideoLayout,
) -> Result<Tensor, ImageProcessorError> {
    let nchw = match tensor.layout() {
        Layout::NCHW => tensor.clone(),
        Layout::NHWC => tensor.to_layout(Layout::NCHW)?,
        layout => return Err(ImageProcessorError::UnsupportedLayout(layout)),
    };
    let [frames, channels, source_height, source_width] = nchw.shape() else {
        return Err(ImageProcessorError::UnsupportedLayout(nchw.layout()));
    };
    let source_size = ImageSize::new(*source_height, *source_width)?;
    let resized_size = ltx2_hdr_resize_size(source_size, target_size)?;
    if resized_size.height > target_size.height || resized_size.width > target_size.width {
        return Err(ImageProcessorError::InvalidPaddingTarget {
            target_size,
            image_size: resized_size,
        });
    }

    let source_pixels = ltx2_checked_area(source_size)?;
    let target_pixels = ltx2_checked_area(target_size)?;
    let plane_count = frames
        .checked_mul(*channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let output_len = plane_count
        .checked_mul(target_pixels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let TensorData::F32(values) = nchw.data() else {
        return Err(ImageProcessorError::ExpectedDataType("f32"));
    };
    let pad_bottom = target_size.height - resized_size.height;
    let pad_right = target_size.width - resized_size.width;
    let reflect = pad_bottom < resized_size.height && pad_right < resized_size.width;
    let mut output = Vec::with_capacity(output_len);

    for plane in values.chunks_exact(source_pixels) {
        let resized =
            crate::transforms::resize_f32_image_bilinear(plane, source_size, resized_size)?;
        for output_y in 0..target_size.height {
            let source_y = ltx2_pad_coordinate(output_y, resized_size.height, reflect);
            let row_start = source_y
                .checked_mul(resized_size.width)
                .ok_or(TransformError::ImageSizeOverflow)?;
            for output_x in 0..target_size.width {
                let source_x = ltx2_pad_coordinate(output_x, resized_size.width, reflect);
                output.push(resized[row_start + source_x]);
            }
        }
    }

    let tensor = Tensor::new(
        TensorData::F32(output),
        vec![*frames, *channels, target_size.height, target_size.width],
        Layout::NCHW,
    )?;
    let tensor = match output_layout {
        VideoLayout::FramesChannelsHeightWidth => tensor,
        VideoLayout::FramesHeightWidthChannels => tensor.to_layout(Layout::NHWC)?,
    };
    tensor
        .with_leading_axis(TensorLeadingAxis::Frames)
        .map_err(ImageProcessorError::Tensor)
}

fn ltx2_hdr_resize_size(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<ImageSize, ImageProcessorError> {
    if target_size.height >= source_size.height && target_size.width >= source_size.width {
        return Ok(source_size);
    }

    let scale = (target_size.height as f64 / source_size.height as f64)
        .min(target_size.width as f64 / source_size.width as f64);
    ImageSize::new(
        python_round_positive_to_usize(source_size.height as f64 * scale)?,
        python_round_positive_to_usize(source_size.width as f64 * scale)?,
    )
    .map_err(ImageProcessorError::Transform)
}

fn ltx2_checked_area(size: ImageSize) -> Result<usize, ImageProcessorError> {
    size.height
        .checked_mul(size.width)
        .ok_or_else(|| TransformError::ImageSizeOverflow.into())
}

fn ltx2_pad_coordinate(coordinate: usize, extent: usize, reflect: bool) -> usize {
    if coordinate < extent {
        coordinate
    } else if reflect {
        extent - 2 - (coordinate - extent)
    } else {
        extent - 1
    }
}
