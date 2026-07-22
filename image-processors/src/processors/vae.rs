//! Diffusers VAE and LDM3D processor support.

use ::image::{imageops::FilterType as ImageResizeFilter, ImageBuffer, Luma};

use super::*;

/// Rust output kind for VAE postprocessing.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VaeOutputType {
    /// Return the input tensor unchanged.
    Latent,
    /// Return a tensor in the input layout after optional denormalization.
    Tensor,
    /// Return an NHWC, HWC, or BFHWC tensor after optional denormalization.
    Array,
    /// Return owned image frames after optional denormalization.
    Frames,
}

/// VAE postprocess output.
#[derive(Clone, Debug, PartialEq)]
pub enum VaePostprocessOutput {
    /// Tensor output for `latent`, tensor, or array modes.
    Tensor(Tensor),
    /// Image-frame output for frame mode. Batched video outputs are flattened
    /// in batch-major, frame-major order.
    Frames(Vec<ImageFrame>),
}

impl VaePostprocessOutput {
    /// Returns the tensor output when this value contains one.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::Frames(_) => None,
        }
    }

    /// Returns the frame outputs when this value contains frames.
    pub fn as_frames(&self) -> Option<&[ImageFrame]> {
        match self {
            Self::Tensor(_) => None,
            Self::Frames(frames) => Some(frames),
        }
    }

    /// Consumes this value and returns its tensor, when present.
    pub fn into_tensor(self) -> Option<Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::Frames(_) => None,
        }
    }

    /// Consumes this value and returns its frames, when present.
    pub fn into_frames(self) -> Option<Vec<ImageFrame>> {
        match self {
            Self::Tensor(_) => None,
            Self::Frames(frames) => Some(frames),
        }
    }
}

/// Original-canvas data needed for crop-aware inpaint postprocessing.
#[derive(Clone, Debug, PartialEq)]
pub struct InpaintOverlayContext {
    /// Crop box used before inpaint model generation.
    pub crop_box: ImageCropBox,
    /// Original image frame in decoded pixel space.
    pub original_image: ImageFrame,
    /// Original mask frame in decoded pixel space.
    pub original_mask: ImageFrame,
}

/// Diffusers inpaint preprocessing output.
#[derive(Clone, Debug, PartialEq)]
pub struct InpaintPreprocessOutput {
    pixel_values: Tensor,
    mask: Tensor,
    overlay_context: Option<InpaintOverlayContext>,
}

impl InpaintPreprocessOutput {
    /// Returns the image tensor to feed to the inpaint pipeline.
    pub fn pixel_values(&self) -> &Tensor {
        &self.pixel_values
    }

    /// Returns the binarized grayscale mask tensor.
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }

    /// Returns the optional crop-aware overlay context.
    pub fn overlay_context(&self) -> Option<&InpaintOverlayContext> {
        self.overlay_context.as_ref()
    }

    /// Consumes this output into image tensor, mask tensor, and overlay context.
    pub fn into_parts(self) -> (Tensor, Tensor, Option<InpaintOverlayContext>) {
        (self.pixel_values, self.mask, self.overlay_context)
    }
}

/// A 16-bit single-channel depth map used by Diffusers LDM3D.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMapU16 {
    width: usize,
    height: usize,
    data: Vec<u16>,
}

impl DepthMapU16 {
    /// Creates a depth map from row-major `u16` values.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are zero, dimensions overflow, or the
    /// value count does not match `width * height`.
    pub fn new(width: usize, height: usize, data: Vec<u16>) -> Result<Self, ImageProcessorError> {
        let size = ImageSize::new(height, width)?;
        let expected = size
            .height
            .checked_mul(size.width)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if data.len() != expected {
            return Err(TransformError::InvalidBufferLength {
                expected,
                actual: data.len(),
            }
            .into());
        }

        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns the width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the height-width size.
    pub fn size(&self) -> ImageSize {
        ImageSize {
            height: self.height,
            width: self.width,
        }
    }

    /// Returns row-major 16-bit depth values.
    pub fn data(&self) -> &[u16] {
        &self.data
    }

    /// Consumes this map into width, height, and row-major values.
    pub fn into_parts(self) -> (usize, usize, Vec<u16>) {
        (self.width, self.height, self.data)
    }
}

/// Diffusers LDM3D preprocessing output.
#[derive(Clone, Debug, PartialEq)]
pub struct Ldm3dPreprocessOutput {
    rgb: Tensor,
    depth: Tensor,
}

impl Ldm3dPreprocessOutput {
    /// Returns the RGB tensor.
    pub fn rgb(&self) -> &Tensor {
        &self.rgb
    }

    /// Returns the depth tensor.
    pub fn depth(&self) -> &Tensor {
        &self.depth
    }

    /// Consumes this output into RGB and depth tensors.
    pub fn into_parts(self) -> (Tensor, Tensor) {
        (self.rgb, self.depth)
    }
}

/// Diffusers LDM3D postprocessing output.
#[derive(Clone, Debug, PartialEq)]
pub struct Ldm3dPostprocessOutput {
    rgb: Vec<ImageFrame>,
    depth: Vec<DepthMapU16>,
}

impl Ldm3dPostprocessOutput {
    /// Returns the RGB image frames.
    pub fn rgb(&self) -> &[ImageFrame] {
        &self.rgb
    }

    /// Returns the 16-bit depth maps.
    pub fn depth(&self) -> &[DepthMapU16] {
        &self.depth
    }

    /// Consumes this output into RGB frames and depth maps.
    pub fn into_parts(self) -> (Vec<ImageFrame>, Vec<DepthMapU16>) {
        (self.rgb, self.depth)
    }
}

/// Configuration for Diffusers VAE-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VaeImageProcessorConfig {
    /// Whether images should be resized to VAE-scale multiples.
    pub do_resize: bool,
    /// VAE scale factor used to round image dimensions down.
    pub vae_scale_factor: usize,
    /// Channel count that represents latent tensors.
    pub vae_latent_channels: usize,
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
    /// Whether image values should be normalized from `[0, 1]` to `[-1, 1]`.
    pub do_normalize: bool,
    /// Whether image values should be thresholded to `0.0` or `1.0`.
    pub do_binarize: bool,
}

impl Default for VaeImageProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            vae_scale_factor: 8,
            vae_latent_channels: 4,
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

impl VaeImageProcessorConfig {
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

    /// Converts this family config into a reusable processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid resize,
    /// normalization, binarization, or output-layout settings.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(5);
        if let Some(format) = self.pixel_format {
            stages.push(ProcessorRecipeStage::ConvertPixelFormat { format });
        }
        if self.do_resize {
            stages.push(ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::dynamic(
                    ResizeMode::Default,
                    self.resample,
                    self.resize_parity,
                ),
            });
        }
        stages.push(ProcessorRecipeStage::Rescale {
            factor: DEFAULT_RESCALE_FACTOR,
        });
        if self.do_normalize {
            stages.push(ProcessorRecipeStage::Normalize {
                mean: DIFFUSION_VAE_IMAGE_MEAN.to_vec(),
                std: DIFFUSION_VAE_IMAGE_STD.to_vec(),
            });
        }
        if self.do_binarize {
            stages.push(ProcessorRecipeStage::Binarize);
        }

        ProcessorRecipe::new(
            "diffusers.vae_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}

/// Diffusers VAE image processor.
#[derive(Clone, Debug)]
pub struct VaeImageProcessor {
    config: VaeImageProcessorConfig,
    processor: ImageProcessor,
}

impl VaeImageProcessor {
    /// Creates a processor from a validated VAE configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE dimensions are invalid or the derived generic
    /// image processor config is invalid.
    pub fn new(config: VaeImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_vae_config(&config)?;
        let processor = ImageProcessor::new(config.image_processor_config())?;
        Ok(Self { config, processor })
    }

    /// Returns this processor's VAE configuration.
    pub fn config(&self) -> &VaeImageProcessorConfig {
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
    /// Returns an error when loading, decoding, VAE resize planning, or tensor
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

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE resize planning, resizing, normalization, or
    /// tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame with per-call options.
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
        let options = self.options_for_vae_resize(image, options)?;
        self.processor.preprocess_image_with_options(image, options)
    }

    /// Preprocesses decoded image frames as a batch tensor.
    ///
    /// The default VAE size is derived from the first image, matching
    /// Diffusers list-input behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or preprocessing fails.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_images_with_options(images, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded image frames as a batch tensor with options.
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
        let options = self.options_for_vae_resize(first, options)?;
        self.processor
            .preprocess_images_with_options(images, options)
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

    /// Preprocesses an image and mask for Diffusers-style inpaint pipelines.
    ///
    /// The image follows this VAE processor's RGB/grayscale, resize, rescale,
    /// and normalization settings. The mask is converted to grayscale,
    /// rescaled to `[0, 1]`, and binarized without `[-1, 1]` normalization.
    ///
    /// # Errors
    ///
    /// Returns an error when image and mask dimensions differ, VAE resize
    /// planning fails, crop planning fails, or image or mask preprocessing
    /// fails.
    pub fn preprocess_inpaint(
        &self,
        image: &ImageFrame,
        mask: &ImageFrame,
    ) -> Result<InpaintPreprocessOutput, ImageProcessorError> {
        self.preprocess_inpaint_with_options(image, mask, ImageProcessorOptions::default(), None)
    }

    /// Preprocesses an image and mask with optional padding-mask crop.
    ///
    /// When `padding_mask_crop` is `Some`, the non-zero mask bounds are padded,
    /// adjusted to the VAE target aspect ratio, cropped from both image and
    /// mask, and resized with fill semantics. The returned overlay context can
    /// be passed to [`VaeImageProcessor::postprocess_inpaint`] to paste
    /// generated frames back onto the original image canvas.
    ///
    /// # Errors
    ///
    /// Returns an error when image and mask dimensions differ, crop or resize
    /// dimensions are invalid, output tensors cannot be built, or crop-aware
    /// frame construction fails.
    pub fn preprocess_inpaint_with_options(
        &self,
        image: &ImageFrame,
        mask: &ImageFrame,
        options: ImageProcessorOptions,
        padding_mask_crop: Option<usize>,
    ) -> Result<InpaintPreprocessOutput, ImageProcessorError> {
        let original_size = ImageSize::new(image.height(), image.width())?;
        let mask_size = ImageSize::new(mask.height(), mask.width())?;
        if mask_size != original_size {
            return Err(TransformError::IncompatibleFrameSize {
                frame: "mask",
                expected: original_size,
                actual: mask_size,
            }
            .into());
        }

        let mask_processor = self.inpaint_mask_processor()?;
        let target_size = self.target_size_for_vae_resize(image, options)?;
        let (image_input, mask_input, preprocess_options, overlay_context) =
            if let Some(pad) = padding_mask_crop {
                let (x_min, y_min, x_max, y_max) = crop_region(mask, target_size, pad)?;
                let crop_box = ImageCropBox::new(x_min, y_min, x_max, y_max);
                let cropped_image = crop_frame(image, crop_box)?;
                let cropped_mask = crop_frame(mask, crop_box)?;
                let overlay_context = Some(InpaintOverlayContext {
                    crop_box,
                    original_image: image.clone(),
                    original_mask: mask.clone(),
                });
                (
                    Cow::Owned(cropped_image),
                    Cow::Owned(cropped_mask),
                    ImageProcessorOptions {
                        height: Some(target_size.height),
                        width: Some(target_size.width),
                        resize_mode: Some(ResizeMode::Fill),
                    },
                    overlay_context,
                )
            } else {
                (
                    Cow::Borrowed(image),
                    Cow::Borrowed(mask),
                    self.options_for_vae_resize(image, options)?,
                    None,
                )
            };

        Ok(InpaintPreprocessOutput {
            pixel_values: self
                .processor
                .preprocess_image_with_options(image_input.as_ref(), preprocess_options)?,
            mask: mask_processor
                .preprocess_image_with_options(mask_input.as_ref(), preprocess_options)?,
            overlay_context,
        })
    }

    /// Postprocesses a VAE tensor into a Rust output type.
    ///
    /// `Latent` returns the tensor unchanged. `Tensor` returns the input layout
    /// after optional denormalization. `Array` returns HWC/NHWC/BFHWC tensor
    /// data. `Frames` returns owned image frames with byte values rounded from
    /// clamped `[0, 1]` values.
    ///
    /// # Errors
    ///
    /// Returns an error when denormalization flags do not match the sample
    /// count, the tensor layout cannot be converted to an image layout, channel
    /// count is unsupported, or frame construction fails.
    pub fn postprocess(
        &self,
        tensor: &Tensor,
        output_type: VaeOutputType,
        do_denormalize: Option<&[bool]>,
    ) -> Result<VaePostprocessOutput, ImageProcessorError> {
        match output_type {
            VaeOutputType::Latent => Ok(VaePostprocessOutput::Tensor(tensor.clone())),
            VaeOutputType::Tensor => {
                let tensor =
                    denormalize_vae_tensor(tensor, self.config.do_normalize, do_denormalize)?;
                Ok(VaePostprocessOutput::Tensor(tensor))
            }
            VaeOutputType::Array => {
                let tensor =
                    denormalize_vae_tensor(tensor, self.config.do_normalize, do_denormalize)?;
                Ok(VaePostprocessOutput::Tensor(channel_last_array_tensor(
                    &tensor,
                )?))
            }
            VaeOutputType::Frames => {
                let tensor =
                    denormalize_vae_tensor(tensor, self.config.do_normalize, do_denormalize)?;
                let clips = post_process_video_tensor(&tensor)
                    .map_err(image_processor_error_from_tensor_postprocess)?;
                Ok(VaePostprocessOutput::Frames(
                    clips.into_iter().flatten().collect(),
                ))
            }
        }
    }

    /// Postprocesses generated inpaint output and applies crop-aware overlay.
    ///
    /// The tensor is decoded through [`VaeOutputType::Frames`]. If the
    /// preprocessing output contains overlay context, each generated frame is
    /// pasted into the saved crop box and composited with the original image
    /// using the saved mask.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE frame postprocessing fails or crop-aware
    /// overlay composition fails.
    pub fn postprocess_inpaint(
        &self,
        tensor: &Tensor,
        preprocess: &InpaintPreprocessOutput,
        do_denormalize: Option<&[bool]>,
    ) -> Result<Vec<ImageFrame>, ImageProcessorError> {
        let VaePostprocessOutput::Frames(frames) =
            self.postprocess(tensor, VaeOutputType::Frames, do_denormalize)?
        else {
            return Err(ImageProcessorError::UnsupportedLayout(tensor.layout()));
        };

        let Some(context) = preprocess.overlay_context() else {
            return Ok(frames);
        };

        frames
            .iter()
            .map(|frame| {
                inpaint_overlay(
                    &context.original_image,
                    frame,
                    &context.original_mask,
                    Some(context.crop_box),
                )
                .map_err(ImageProcessorError::Transform)
            })
            .collect()
    }

    fn options_for_vae_resize(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageProcessorOptions, ImageProcessorError> {
        if !self.config.do_resize {
            return Ok(options);
        }

        let resized_size = self.target_size_for_vae_resize(image, options)?;

        Ok(ImageProcessorOptions {
            height: Some(resized_size.height),
            width: Some(resized_size.width),
            resize_mode: options.resize_mode,
        })
    }

    fn target_size_for_vae_resize(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageSize, ImageProcessorError> {
        let requested_size = ImageSize::new(
            options.height.unwrap_or_else(|| image.height()),
            options.width.unwrap_or_else(|| image.width()),
        )?;
        if !self.config.do_resize {
            return Ok(requested_size);
        }

        let plan = scale_factor_resize_plan(
            ImageSize::new(image.height(), image.width())?,
            Some(requested_size),
            self.config.vae_scale_factor,
        )?;
        Ok(plan.resized_size)
    }

    fn inpaint_mask_processor(&self) -> Result<ImageProcessor, ImageProcessorError> {
        let mut config = self.config.clone();
        config.pixel_format = Some(PixelFormat::Luma8);
        config.do_normalize = false;
        config.do_binarize = true;
        ImageProcessor::new(config.image_processor_config())
    }
}

/// Diffusers LDM3D VAE image processor.
#[derive(Clone, Debug)]
pub struct VaeImageProcessorLdm3d {
    processor: VaeImageProcessor,
}

impl VaeImageProcessorLdm3d {
    /// Creates an LDM3D processor from a validated VAE configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when VAE dimensions are invalid or the derived generic
    /// image processor config is invalid.
    pub fn new(config: VaeImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        Ok(Self {
            processor: VaeImageProcessor::new(config)?,
        })
    }

    /// Returns this processor's VAE configuration.
    pub fn config(&self) -> &VaeImageProcessorConfig {
        self.processor.config()
    }

    /// Returns the underlying VAE image processor.
    pub fn vae_processor(&self) -> &VaeImageProcessor {
        &self.processor
    }

    /// Preprocesses one RGB frame and one 16-bit depth map.
    ///
    /// # Errors
    ///
    /// Returns an error when RGB or depth preprocessing fails or when paired
    /// batch shapes are incompatible.
    pub fn preprocess(
        &self,
        rgb: &ImageFrame,
        depth: &DepthMapU16,
    ) -> Result<Ldm3dPreprocessOutput, ImageProcessorError> {
        self.preprocess_with_options(rgb, depth, ImageProcessorOptions::default())
    }

    /// Preprocesses one RGB/depth pair with per-call size options.
    ///
    /// Diffusers LDM3D uses direct resize for both RGB and depth inputs. The
    /// height and width overrides are honored, while non-default resize modes
    /// are treated as direct resize.
    ///
    /// # Errors
    ///
    /// Returns an error when RGB or depth preprocessing fails or when output
    /// tensors cannot be built.
    pub fn preprocess_with_options(
        &self,
        rgb: &ImageFrame,
        depth: &DepthMapU16,
        options: ImageProcessorOptions,
    ) -> Result<Ldm3dPreprocessOutput, ImageProcessorError> {
        self.preprocess_batch_with_options(
            std::slice::from_ref(rgb),
            std::slice::from_ref(depth),
            options,
        )
    }

    /// Preprocesses RGB/depth pairs as batch tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when either batch is empty, batch lengths differ, RGB
    /// preprocessing fails, or depth preprocessing fails.
    pub fn preprocess_batch(
        &self,
        rgb: &[ImageFrame],
        depth: &[DepthMapU16],
    ) -> Result<Ldm3dPreprocessOutput, ImageProcessorError> {
        self.preprocess_batch_with_options(rgb, depth, ImageProcessorOptions::default())
    }

    /// Preprocesses RGB/depth pairs as batch tensors with size options.
    ///
    /// The first RGB frame selects the default VAE-scale target size, matching
    /// Diffusers list-input behavior for VAE processors.
    ///
    /// # Errors
    ///
    /// Returns an error when either batch is empty, batch lengths differ, RGB
    /// preprocessing fails, or depth preprocessing fails.
    pub fn preprocess_batch_with_options(
        &self,
        rgb: &[ImageFrame],
        depth: &[DepthMapU16],
        options: ImageProcessorOptions,
    ) -> Result<Ldm3dPreprocessOutput, ImageProcessorError> {
        if rgb.is_empty() || depth.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }
        if rgb.len() != depth.len() {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }

        let options = ImageProcessorOptions {
            resize_mode: Some(ResizeMode::Default),
            ..options
        };
        let first = rgb.first().ok_or(ImageProcessorError::EmptyBatch)?;
        let target = if self.config().do_resize {
            self.processor.target_size_for_vae_resize(first, options)?
        } else {
            ImageSize::new(first.height(), first.width())?
        };
        let rgb_options = self.processor.options_for_vae_resize(first, options)?;
        let rgb = self
            .processor
            .image_processor()
            .preprocess_images_with_options(rgb, rgb_options)?;
        let depth = ldm3d_depth_batch_tensor(depth, target, self.config())?;

        Ok(Ldm3dPreprocessOutput { rgb, depth })
    }

    /// Postprocesses a generated LDM3D tensor into RGB frames and depth maps.
    ///
    /// The tensor must contain either four channels (`RGB + scalar depth`) or
    /// six channels (`RGB + RGB-like depth`). RGB values are materialized as
    /// `Rgb8` frames and depth values as 16-bit row-major maps.
    ///
    /// # Errors
    ///
    /// Returns an error when denormalization flags are invalid, the tensor
    /// layout is unsupported, channel count is not 4 or 6, or output frame
    /// construction fails.
    pub fn postprocess(
        &self,
        tensor: &Tensor,
        do_denormalize: Option<&[bool]>,
    ) -> Result<Ldm3dPostprocessOutput, ImageProcessorError> {
        let tensor = denormalize_vae_tensor(tensor, self.config().do_normalize, do_denormalize)?;
        let tensor = channel_last_array_tensor(&tensor)?;
        ldm3d_tensor_to_output(&tensor)
    }
}

pub(super) fn validate_vae_config(
    config: &VaeImageProcessorConfig,
) -> Result<(), ImageProcessorError> {
    if config.vae_scale_factor == 0 {
        return Err(TransformError::InvalidScaleFactor(config.vae_scale_factor).into());
    }
    if config.vae_latent_channels == 0 {
        return Err(ImageProcessorError::InvalidVaeLatentChannels(
            config.vae_latent_channels,
        ));
    }
    ResizeDecision::new(config.resample, config.resize_parity)?;
    Ok(())
}

fn denormalize_vae_tensor(
    tensor: &Tensor,
    default_do_denormalize: bool,
    do_denormalize: Option<&[bool]>,
) -> Result<Tensor, ImageProcessorError> {
    let sample_count = tensor_sample_count(tensor)?;
    let Some(flags) = do_denormalize else {
        return if default_do_denormalize {
            Ok(tensor.denormalize())
        } else {
            Ok(tensor.clone())
        };
    };

    if flags.len() != sample_count {
        return Err(ImageProcessorError::InvalidDenormalizeFlags {
            expected: sample_count,
            actual: flags.len(),
        });
    }
    if flags.iter().all(|flag| *flag) {
        return Ok(tensor.denormalize());
    }
    if flags.iter().all(|flag| !*flag) {
        return Ok(tensor.clone());
    }

    let values = Vec::<f32>::from(tensor.data());
    let item_len = values
        .len()
        .checked_div(sample_count)
        .ok_or(ImageProcessorError::IncompatibleBatchShapes)?;
    let mut output = Vec::with_capacity(values.len());
    for (sample, flag) in values.chunks_exact(item_len).zip(flags.iter().copied()) {
        if flag {
            output.extend(
                sample
                    .iter()
                    .copied()
                    .map(|value| (value * 0.5 + 0.5).clamp(0.0, 1.0)),
            );
        } else {
            output.extend_from_slice(sample);
        }
    }

    let leading_axis = tensor.leading_axis();
    let tensor = Tensor::new(
        TensorData::F32(output),
        tensor.shape().to_vec(),
        tensor.layout(),
    )?;
    preserve_leading_axis(tensor, leading_axis)
}

fn channel_last_array_tensor(tensor: &Tensor) -> Result<Tensor, ImageProcessorError> {
    match tensor.layout() {
        Layout::NCHW => tensor
            .to_layout(Layout::NHWC)
            .map_err(ImageProcessorError::Tensor),
        Layout::CHW => tensor
            .to_layout(Layout::HWC)
            .map_err(ImageProcessorError::Tensor),
        Layout::BFCHW => tensor
            .to_layout(Layout::BFHWC)
            .map_err(ImageProcessorError::Tensor),
        Layout::NHWC | Layout::HWC | Layout::BFHWC => Ok(tensor.clone()),
        Layout::NC | Layout::NPCHW | Layout::NPHWC | Layout::NIPCHW | Layout::NIPHWC => {
            Err(ImageProcessorError::UnsupportedLayout(tensor.layout()))
        }
    }
}

fn tensor_sample_count(tensor: &Tensor) -> Result<usize, ImageProcessorError> {
    match tensor.layout() {
        Layout::NCHW | Layout::NHWC | Layout::BFCHW | Layout::BFHWC => Ok(tensor.shape()[0]),
        Layout::CHW | Layout::HWC => Ok(1),
        Layout::NC | Layout::NPCHW | Layout::NPHWC | Layout::NIPCHW | Layout::NIPHWC => {
            Err(ImageProcessorError::UnsupportedLayout(tensor.layout()))
        }
    }
}

fn preserve_leading_axis(
    tensor: Tensor,
    leading_axis: Option<TensorLeadingAxis>,
) -> Result<Tensor, ImageProcessorError> {
    match leading_axis {
        Some(leading_axis) => tensor
            .with_leading_axis(leading_axis)
            .map_err(ImageProcessorError::Tensor),
        None => Ok(tensor),
    }
}

fn ldm3d_depth_batch_tensor(
    depth: &[DepthMapU16],
    target: ImageSize,
    config: &VaeImageProcessorConfig,
) -> Result<Tensor, ImageProcessorError> {
    let item_len = target
        .height
        .checked_mul(target.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut values = Vec::with_capacity(
        depth
            .len()
            .checked_mul(item_len)
            .ok_or(TransformError::ImageSizeOverflow)?,
    );

    for map in depth {
        let prepared = if config.do_resize {
            Cow::Owned(resize_depth_map(map, target, config.resample)?)
        } else {
            validate_ldm3d_depth_size(map, target)?;
            Cow::Borrowed(map)
        };
        values.extend(
            prepared
                .data()
                .iter()
                .copied()
                .map(|value| value as f32 / u16::MAX as f32),
        );
    }

    if config.do_normalize {
        for value in &mut values {
            *value = *value * 2.0 - 1.0;
        }
    }
    if config.do_binarize {
        for value in &mut values {
            *value = if *value < 0.5 { 0.0 } else { 1.0 };
        }
    }

    let shape = match config.output_layout {
        ImageLayout::ChannelsHeightWidth => vec![depth.len(), 1, target.height, target.width],
        ImageLayout::HeightWidthChannels => vec![depth.len(), target.height, target.width, 1],
    };
    Tensor::new(
        TensorData::F32(values),
        shape,
        batched_layout_for_image_layout(config.output_layout),
    )
    .map_err(ImageProcessorError::Tensor)
}

fn validate_ldm3d_depth_size(
    depth: &DepthMapU16,
    expected: ImageSize,
) -> Result<(), ImageProcessorError> {
    let actual = depth.size();
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::IncompatibleFrameSize {
            frame: "depth",
            expected,
            actual,
        }
        .into())
    }
}

fn resize_depth_map(
    depth: &DepthMapU16,
    target: ImageSize,
    filter: ResizeFilter,
) -> Result<DepthMapU16, ImageProcessorError> {
    if depth.size() == target {
        return Ok(depth.clone());
    }

    let source = depth.size();
    let source_width = u32_dimension(source.width, "source width")?;
    let source_height = u32_dimension(source.height, "source height")?;
    let target_width = u32_dimension(target.width, "target width")?;
    let target_height = u32_dimension(target.height, "target height")?;
    let buffer = ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(
        source_width,
        source_height,
        depth.data().to_vec(),
    )
    .ok_or(ImageProcessorError::Transform(
        TransformError::InvalidBufferLength {
            expected: source
                .height
                .checked_mul(source.width)
                .ok_or(TransformError::ImageSizeOverflow)?,
            actual: depth.data().len(),
        },
    ))?;
    let resized = ::image::imageops::resize(
        &buffer,
        target_width,
        target_height,
        image_filter_for_resize(filter),
    );
    DepthMapU16::new(target.width, target.height, resized.into_raw())
}

fn image_filter_for_resize(filter: ResizeFilter) -> ImageResizeFilter {
    match filter {
        ResizeFilter::Nearest => ImageResizeFilter::Nearest,
        ResizeFilter::Bilinear => ImageResizeFilter::Triangle,
        ResizeFilter::Bicubic => ImageResizeFilter::CatmullRom,
        ResizeFilter::Lanczos => ImageResizeFilter::Lanczos3,
    }
}

fn u32_dimension(value: usize, dimension: &'static str) -> Result<u32, ImageProcessorError> {
    u32::try_from(value)
        .map_err(|_| TransformError::DimensionTooLarge {
            dimension,
            value,
            max: u32::MAX as usize,
        })
        .map_err(ImageProcessorError::Transform)
}

fn ldm3d_tensor_to_output(tensor: &Tensor) -> Result<Ldm3dPostprocessOutput, ImageProcessorError> {
    let values = Vec::<f32>::from(tensor.data());
    match tensor.layout() {
        Layout::NHWC => {
            let shape = tensor.shape();
            let batch = shape[0];
            let height = shape[1];
            let width = shape[2];
            let channels = shape[3];
            let item_len = height
                .checked_mul(width)
                .and_then(|value| value.checked_mul(channels))
                .ok_or(TransformError::ImageSizeOverflow)?;
            let mut rgb = Vec::with_capacity(batch);
            let mut depth = Vec::with_capacity(batch);
            for sample in values.chunks_exact(item_len).take(batch) {
                let (rgb_frame, depth_map) =
                    ldm3d_sample_to_output(sample, height, width, channels)?;
                rgb.push(rgb_frame);
                depth.push(depth_map);
            }
            Ok(Ldm3dPostprocessOutput { rgb, depth })
        }
        Layout::HWC => {
            let shape = tensor.shape();
            let (rgb, depth) = ldm3d_sample_to_output(&values, shape[0], shape[1], shape[2])?;
            Ok(Ldm3dPostprocessOutput {
                rgb: vec![rgb],
                depth: vec![depth],
            })
        }
        Layout::NCHW
        | Layout::CHW
        | Layout::NC
        | Layout::NPCHW
        | Layout::NPHWC
        | Layout::BFCHW
        | Layout::BFHWC
        | Layout::NIPCHW
        | Layout::NIPHWC => Err(ImageProcessorError::UnsupportedLayout(tensor.layout())),
    }
}

fn ldm3d_sample_to_output(
    sample: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<(ImageFrame, DepthMapU16), ImageProcessorError> {
    if channels != 4 && channels != 6 {
        return Err(TransformError::UnsupportedChannels(channels).into());
    }

    let pixels = height
        .checked_mul(width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let expected = pixels
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    if sample.len() != expected {
        return Err(TransformError::InvalidBufferLength {
            expected,
            actual: sample.len(),
        }
        .into());
    }

    let mut rgb = Vec::with_capacity(
        pixels
            .checked_mul(PixelFormat::Rgb8.channels())
            .ok_or(TransformError::ImageSizeOverflow)?,
    );
    let mut depth = Vec::with_capacity(pixels);
    for pixel in 0..pixels {
        let offset = pixel
            .checked_mul(channels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        rgb.push(unit_f32_to_u8(sample[offset]));
        rgb.push(unit_f32_to_u8(sample[offset + 1]));
        rgb.push(unit_f32_to_u8(sample[offset + 2]));
        if channels == 4 {
            depth.push(unit_f32_to_u16(sample[offset + 3]));
        } else {
            let high = unit_f32_to_u8(sample[offset + 4]);
            let low = unit_f32_to_u8(sample[offset + 5]);
            depth.push(u16::from(high) * 256 + u16::from(low));
        }
    }

    let rgb = ImageFrame::new(width, height, PixelFormat::Rgb8, rgb)
        .map_err(ImageProcessorError::Media)?;
    let depth = DepthMapU16::new(width, height, depth)?;
    Ok((rgb, depth))
}

fn unit_f32_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn unit_f32_to_u16(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * u16::MAX as f32) as u16
}
