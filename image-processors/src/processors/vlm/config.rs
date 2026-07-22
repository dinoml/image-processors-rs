//! VLM family configuration types.

use super::*;

/// Configuration for Qwen/VLM-style image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QwenVlImageProcessorConfig {
    /// Smart-resize pixel and factor limits.
    pub resize_limits: ResizeLimits,
    /// Spatial patch size used by Qwen-style vision encoders.
    pub patch_size: usize,
    /// Temporal patch size used by Qwen-style vision encoders.
    pub temporal_patch_size: usize,
    /// Spatial merge size between the vision encoder and language model.
    pub merge_size: usize,
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
    /// Pixel format used before tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for QwenVlImageProcessorConfig {
    fn default() -> Self {
        Self {
            resize_limits: ResizeLimits {
                factor: QWEN_VL_PATCH_SIZE * QWEN_VL_MERGE_SIZE,
                min_pixels: 56 * 56,
                max_pixels: QWEN_VL_PATCH_SIZE * QWEN_VL_PATCH_SIZE * 4 * 1280,
            },
            patch_size: QWEN_VL_PATCH_SIZE,
            temporal_patch_size: QWEN_VL_TEMPORAL_PATCH_SIZE,
            merge_size: QWEN_VL_MERGE_SIZE,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: CLIP_IMAGE_MEAN.to_vec(),
            image_std: CLIP_IMAGE_STD.to_vec(),
        }
    }
}

impl QwenVlImageProcessorConfig {
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
            do_resize: true,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// The recipe records the smart-resize limits and the patch-flattening
    /// geometry used by Qwen/VLM processor outputs. Lowering returns the
    /// generic image config consumed before patch flattening.
    ///
    /// # Errors
    ///
    /// Returns an error when smart-resize limits, patch geometry,
    /// normalization statistics, or output layout are invalid.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(5);
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
        });
        stages.push(ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::smart(
                self.resize_limits,
                ResizeMode::Default,
                self.resample,
                self.resize_parity,
            ),
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
        stages.push(ProcessorRecipeStage::PatchFlatten {
            patch: RecipePatchStage::flatten(
                Layout::NCHW,
                self.patch_size,
                self.temporal_patch_size,
                self.merge_size,
            ),
        });

        ProcessorRecipe::new(
            "transformers.qwen_vl_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput {
                layout: Layout::NC,
                leading_axis: TensorLeadingAxis::Batch,
            },
        )
    }

    pub(crate) fn patch_image_processor_config(&self) -> ImageProcessorConfig {
        let mut config = self.image_processor_config();
        config.output_layout = Layout::NCHW;
        config
    }
}

/// Configuration for LLaVA-NeXT AnyRes image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LlavaNextImageProcessorConfig {
    /// Resize target used for the base image before optional center cropping.
    pub size: ImageSize,
    /// Center-crop target and square tile edge used for patch frames.
    pub crop_size: ImageSize,
    /// Candidate AnyRes canvas sizes used for high-resolution patch extraction.
    pub image_grid_pinpoints: Vec<ImageSize>,
    /// Whether patch frames should be center-cropped after resizing.
    pub do_center_crop: bool,
    /// Whether image batches should pad patch frames to the largest patch count.
    pub do_pad: bool,
    /// Device-agnostic patch-frame output axis order.
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
    /// Pixel format used before tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for LlavaNextImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: square_size(LLAVA_NEXT_PATCH_SIZE),
            crop_size: square_size(LLAVA_NEXT_PATCH_SIZE),
            image_grid_pinpoints: LLAVA_NEXT_DEFAULT_GRID_PINPOINTS.to_vec(),
            do_center_crop: true,
            do_pad: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: CLIP_IMAGE_MEAN.to_vec(),
            image_std: CLIP_IMAGE_STD.to_vec(),
        }
    }
}

impl LlavaNextImageProcessorConfig {
    /// Returns the patch-frame output axis order.
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

    /// Converts this family config into the generic patch-frame processor config.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: true,
            height: Some(self.target_size().height),
            width: Some(self.target_size().width),
            resize_mode: self.resize_mode(),
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable AnyRes patch-frame recipe.
    ///
    /// The recipe records AnyRes canvas selection and patch extraction before
    /// the generic processor stages used for each base and patch frame.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid resize,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(5);
        stages.push(ProcessorRecipeStage::PatchGrid {
            patch_grid: RecipePatchGridStage::new(
                self.image_grid_pinpoints.clone(),
                self.size,
                self.patch_size(),
            ),
        });
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
        });
        stages.push(ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::fixed(
                self.target_size(),
                self.resize_mode(),
                self.resample,
                self.resize_parity,
            ),
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
            "transformers.llava_next_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }

    fn target_size(&self) -> ImageSize {
        if self.do_center_crop {
            self.crop_size
        } else {
            self.size
        }
    }

    fn resize_mode(&self) -> ResizeMode {
        if self.do_center_crop {
            ResizeMode::Crop
        } else {
            ResizeMode::Default
        }
    }

    pub(crate) fn patch_size(&self) -> usize {
        self.target_size().height
    }

    pub(super) fn patch_output_layout(&self) -> Layout {
        batched_layout_for_image_layout(self.output_layout)
    }

    pub(super) fn patched_batch_layout(&self) -> Layout {
        match self.output_layout {
            ImageLayout::ChannelsHeightWidth => Layout::NPCHW,
            ImageLayout::HeightWidthChannels => Layout::NPHWC,
        }
    }
}

/// Configuration for Pixtral patch-aligned image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PixtralImageProcessorConfig {
    /// Maximum image dimensions before patch alignment.
    pub max_size: ImageSize,
    /// Patch height and width used to align resized images.
    pub patch_size: ImageSize,
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
    /// Pixel format used before tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for PixtralImageProcessorConfig {
    fn default() -> Self {
        Self {
            max_size: square_size(PIXTRAL_MAX_EDGE),
            patch_size: square_size(PIXTRAL_PATCH_SIZE),
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bicubic,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: CLIP_IMAGE_MEAN.to_vec(),
            image_std: CLIP_IMAGE_STD.to_vec(),
        }
    }
}

impl PixtralImageProcessorConfig {
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

    /// Converts this family config into the generic per-image processor config.
    ///
    /// Pixtral resolves the patch-aligned resize target per image before
    /// invoking the generic processor, then pads normalized tensors across the
    /// batch.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: true,
            height: None,
            width: None,
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable processor recipe.
    ///
    /// The recipe lowers to the generic per-image processor. Patch-aligned
    /// target selection and tensor-space batch padding are family-wrapper
    /// behavior layered on top of that generic config.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid resize,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(4);
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
        });
        stages.push(ProcessorRecipeStage::Resize {
            resize: RecipeResizeStage::dynamic(
                ResizeMode::Default,
                self.resample,
                self.resize_parity,
            ),
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
            "transformers.pixtral_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}

/// Configuration for Idefics3 split-image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Idefics3ImageProcessorConfig {
    /// Longest-edge target used before vision-encoder multiple alignment.
    pub longest_edge: usize,
    /// Maximum square image edge used for split frames and global thumbnails.
    pub max_image_size: usize,
    /// Whether images should be resized to `longest_edge` before splitting.
    pub do_resize: bool,
    /// Whether images should be split into local crops plus a global frame.
    pub do_image_splitting: bool,
    /// Whether nested frame batches should be padded and return an attention mask.
    pub do_pad: bool,
    /// Device-agnostic frame output axis order.
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
    /// Pixel format used before tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for Idefics3ImageProcessorConfig {
    fn default() -> Self {
        Self {
            longest_edge: IDEFICS3_DEFAULT_LONGEST_EDGE,
            max_image_size: IDEFICS3_DEFAULT_MAX_IMAGE_EDGE,
            do_resize: true,
            do_image_splitting: true,
            do_pad: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Lanczos,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl Idefics3ImageProcessorConfig {
    /// Returns the frame output axis order.
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

    /// Converts this family config into the generic tensor-stage processor config.
    ///
    /// Idefics3 resolves resizing, image splitting, and nested padding in the
    /// family wrapper. The generic processor handles RGB conversion, rescale,
    /// normalization, and layout conversion for each prepared frame.
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
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable split-image processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid split geometry,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(4);
        if self.do_image_splitting {
            stages.push(ProcessorRecipeStage::ImageSplit {
                split: RecipeImageSplitStage::new(
                    self.longest_edge,
                    self.max_image_size,
                    self.do_resize,
                ),
            });
        }
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
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
            "transformers.idefics3_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}

/// Configuration for Gemma3 pan-and-scan image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Gemma3ImageProcessorConfig {
    /// Fixed output image size used after optional pan-and-scan crop expansion.
    pub size: ImageSize,
    /// Whether frames should be resized to `size`.
    pub do_resize: bool,
    /// Whether elongated images should emit additional pan-and-scan crops.
    pub do_pan_and_scan: bool,
    /// Minimum crop edge length required for pan-and-scan crops.
    pub pan_and_scan_min_crop_size: usize,
    /// Maximum number of crops along the elongated image axis.
    pub pan_and_scan_max_num_crops: usize,
    /// Minimum elongated-to-short aspect ratio required to emit crops.
    pub pan_and_scan_min_ratio_to_activate: f64,
    /// Device-agnostic frame output axis order.
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
    /// Pixel format used before pan-and-scan and tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for Gemma3ImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: square_size(GEMMA3_IMAGE_SIZE),
            do_resize: true,
            do_pan_and_scan: false,
            pan_and_scan_min_crop_size: GEMMA3_PAN_AND_SCAN_MIN_CROP_SIZE,
            pan_and_scan_max_num_crops: GEMMA3_PAN_AND_SCAN_MAX_NUM_CROPS,
            pan_and_scan_min_ratio_to_activate: GEMMA3_PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl Gemma3ImageProcessorConfig {
    /// Returns the frame output axis order.
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

    /// Returns the pan-and-scan crop selection options.
    pub fn pan_and_scan_options(&self) -> AspectRatioCropOptions {
        AspectRatioCropOptions {
            min_crop_size: self.pan_and_scan_min_crop_size,
            max_num_crops: self.pan_and_scan_max_num_crops,
            min_ratio_to_activate: self.pan_and_scan_min_ratio_to_activate,
        }
    }

    /// Converts this family config into the generic tensor-stage processor config.
    ///
    /// Gemma3 resolves optional pan-and-scan crop expansion in the family
    /// wrapper. The generic processor handles fixed resizing, rescale,
    /// normalization, and layout conversion for each emitted frame.
    pub fn image_processor_config(&self) -> ImageProcessorConfig {
        ImageProcessorConfig {
            do_resize: self.do_resize,
            height: self.do_resize.then_some(self.size.height),
            width: self.do_resize.then_some(self.size.width),
            resize_mode: ResizeMode::Default,
            resample: self.resample,
            resize_parity: self.resize_parity,
            decode_backend: self.decode_backend,
            batch_execution: self.batch_execution,
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable aspect-ratio crop processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid dimensions,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(5);
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
        });
        if self.do_pan_and_scan {
            stages.push(ProcessorRecipeStage::AspectRatioCrops {
                aspect_ratio_crops: RecipeAspectRatioCropStage::new(self.pan_and_scan_options()),
            });
        }
        if self.do_resize {
            stages.push(ProcessorRecipeStage::Resize {
                resize: RecipeResizeStage::fixed(
                    self.size,
                    ResizeMode::Default,
                    self.resample,
                    self.resize_parity,
                ),
            });
        }
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
            "transformers.gemma3_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}

/// Configuration for Mllama tiled-image preprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MllamaImageProcessorConfig {
    /// Square tile edge used by the vision encoder.
    pub tile_size: usize,
    /// Maximum number of tiles allowed for one original image.
    pub max_image_tiles: usize,
    /// Whether images should be resized onto the selected tiled canvas.
    pub do_resize: bool,
    /// Whether resized images should be padded to the selected tiled canvas.
    pub do_pad: bool,
    /// Device-agnostic tile output axis order.
    pub output_layout: ImageLayout,
    /// Resize filter used before raw canvas padding and tensor conversion.
    pub resample: ResizeFilter,
    /// Resize parity policy used before raw canvas padding and tensor conversion.
    #[serde(default)]
    pub resize_parity: ResizeParity,
    /// Image decode backend.
    #[serde(default)]
    pub decode_backend: ImageDecodeBackend,
    /// Batch processing execution mode.
    #[serde(default)]
    pub batch_execution: BatchExecution,
    /// Pixel format used before resizing and tensor conversion.
    pub pixel_format: PixelFormat,
    /// Whether pixels should be multiplied by `rescale_factor`.
    pub do_rescale: bool,
    /// Scale factor applied when rescaling is enabled.
    pub rescale_factor: f32,
    /// Whether pixels should be normalized after optional rescaling.
    pub do_normalize: bool,
    /// Per-channel normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel normalization standard deviation.
    pub image_std: Vec<f32>,
}

impl Default for MllamaImageProcessorConfig {
    fn default() -> Self {
        Self {
            tile_size: MLLAMA_TILE_SIZE,
            max_image_tiles: MLLAMA_MAX_IMAGE_TILES,
            do_resize: true,
            do_pad: true,
            output_layout: ImageLayout::ChannelsHeightWidth,
            resample: ResizeFilter::Bilinear,
            resize_parity: ResizeParity::Compatibility,
            decode_backend: ImageDecodeBackend::ImageCrate,
            batch_execution: BatchExecution::default(),
            pixel_format: PixelFormat::Rgb8,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: STANDARD_IMAGE_MEAN.to_vec(),
            image_std: STANDARD_IMAGE_STD.to_vec(),
        }
    }
}

impl MllamaImageProcessorConfig {
    /// Returns the tile output axis order.
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

    /// Converts this family config into the generic tensor-stage processor config.
    ///
    /// Mllama resolves tiled-canvas selection, resizing, raw padding, tile
    /// splitting, and nested batch packing in the family wrapper. The generic
    /// processor handles rescale, normalization, and layout conversion for
    /// each prepared tile.
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
            pixel_format: Some(self.pixel_format),
            do_rescale: self.do_rescale,
            rescale_factor: self.rescale_factor,
            do_normalize: self.do_normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            do_binarize: false,
            output_layout: batched_layout_for_image_layout(self.output_layout),
        }
    }

    /// Converts this family config into a reusable tiled-canvas processor recipe.
    ///
    /// # Errors
    ///
    /// Returns an error when the derived recipe has invalid tile geometry,
    /// normalization statistics, or output layout.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut stages = Vec::with_capacity(4);
        stages.push(ProcessorRecipeStage::TiledCanvas {
            tile: RecipeTiledCanvasStage::new(self.tile_size, self.max_image_tiles),
        });
        stages.push(ProcessorRecipeStage::ConvertPixelFormat {
            format: self.pixel_format,
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
            "transformers.mllama_image_processor",
            ProcessorRecipeInput {
                decode_backend: self.decode_backend,
                batch_execution: self.batch_execution,
            },
            stages,
            ProcessorRecipeOutput::batch(batched_layout_for_image_layout(self.output_layout)),
        )
    }
}
