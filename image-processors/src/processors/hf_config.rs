//! Hugging Face preprocessor config parsing.

use super::*;

#[derive(Debug, Deserialize)]
pub(super) struct HfPreprocessorConfig {
    image_processor_type: Option<String>,
    processor_class: Option<String>,
    size: Option<HfSize>,
    crop_size: Option<HfSize>,
    max_image_size: Option<HfSize>,
    resample: Option<serde_json::Value>,
    do_center_crop: Option<bool>,
    do_resize: Option<bool>,
    do_thumbnail: Option<bool>,
    do_align_long_axis: Option<bool>,
    do_image_splitting: Option<bool>,
    do_pan_and_scan: Option<bool>,
    do_pad: Option<bool>,
    pad_size: Option<HfSize>,
    pad_to_multiple: Option<usize>,
    image_grid_pinpoints: Option<Vec<[usize; 2]>>,
    max_image_tiles: Option<usize>,
    pan_and_scan_min_crop_size: Option<usize>,
    pan_and_scan_max_num_crops: Option<usize>,
    pan_and_scan_min_ratio_to_activate: Option<f64>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<HfSize>,
    temporal_patch_size: Option<usize>,
    merge_size: Option<usize>,
    do_rescale: Option<bool>,
    rescale_factor: Option<f32>,
    offset: Option<bool>,
    do_normalize: Option<bool>,
    do_range_check: Option<bool>,
    do_binarize: Option<bool>,
    do_convert_rgb: Option<bool>,
    do_convert_grayscale: Option<bool>,
    vae_scale_factor: Option<usize>,
    vae_latent_channels: Option<usize>,
    resolution: Option<usize>,
    basesize: Option<usize>,
    hdr_transform: Option<String>,
    image_mean: Option<Vec<f32>>,
    image_std: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HfSize {
    Edge(usize),
    Map {
        height: Option<usize>,
        width: Option<usize>,
        shortest_edge: Option<usize>,
        longest_edge: Option<usize>,
        max_height: Option<usize>,
        max_width: Option<usize>,
    },
}

pub(super) fn default_true() -> bool {
    true
}

impl HfPreprocessorConfig {
    pub(super) fn into_family_config(self) -> Result<ProcessorFamilyConfig, ProcessorConfigError> {
        let processor_type = self
            .image_processor_type
            .as_deref()
            .or(self.processor_class.as_deref())
            .ok_or(ProcessorConfigError::MissingProcessorType)?;

        if processor_type.contains("BlipImageProcessor") {
            self.blip_config().map(ProcessorFamilyConfig::Blip)
        } else if processor_type.contains("Flux2ImageProcessor") {
            self.flux2_config().map(ProcessorFamilyConfig::Flux2)
        } else if processor_type.contains("VisualClozeProcessor") {
            self.visual_cloze_config()
                .map(ProcessorFamilyConfig::VisualCloze)
        } else if processor_type.contains("HunyuanVideo15ImageProcessor") {
            self.hunyuan_video_15_config()
                .map(ProcessorFamilyConfig::HunyuanVideo15)
        } else if processor_type.contains("MarigoldImageProcessor") {
            Ok(ProcessorFamilyConfig::Marigold(self.marigold_config()))
        } else if processor_type.contains("JoyImageEditImageProcessor") {
            self.joy_image_edit_config()
                .map(ProcessorFamilyConfig::JoyImageEdit)
        } else if processor_type.contains("WanAnimateImageProcessor") {
            self.wan_animate_config()
                .map(ProcessorFamilyConfig::WanAnimate)
        } else if processor_type.contains("LTX2VideoHDRProcessor") {
            self.ltx2_video_hdr_config()
                .map(ProcessorFamilyConfig::Ltx2VideoHdr)
        } else if processor_type.contains("CLIPImageProcessor") {
            self.clip_config().map(ProcessorFamilyConfig::Clip)
        } else if processor_type.contains("VaeImageProcessorLDM3D")
            || processor_type.contains("VaeImageProcessorLdm3d")
        {
            self.vae_config().map(ProcessorFamilyConfig::VaeLdm3d)
        } else if processor_type.contains("VaeImageProcessor") {
            self.vae_config().map(ProcessorFamilyConfig::Vae)
        } else if processor_type.contains("VideoMAE") {
            self.video_mae_config().map(ProcessorFamilyConfig::VideoMae)
        } else if processor_type.contains("Vivit") {
            self.vivit_config().map(ProcessorFamilyConfig::Vivit)
        } else if processor_type.contains("ViTImageProcessor") {
            self.vit_config().map(ProcessorFamilyConfig::Vit)
        } else if processor_type.contains("LlavaNext") || processor_type.contains("LLaVA-NeXT") {
            self.llava_next_config()
                .map(ProcessorFamilyConfig::LlavaNext)
        } else if processor_type.contains("Pixtral") {
            self.pixtral_config().map(ProcessorFamilyConfig::Pixtral)
        } else if processor_type.contains("Idefics3") {
            self.idefics3_config().map(ProcessorFamilyConfig::Idefics3)
        } else if processor_type.contains("Gemma3") {
            self.gemma3_config().map(ProcessorFamilyConfig::Gemma3)
        } else if processor_type.contains("Mllama") {
            self.mllama_config().map(ProcessorFamilyConfig::Mllama)
        } else if processor_type.contains("Qwen") {
            self.qwen_config().map(ProcessorFamilyConfig::QwenVl)
        } else if processor_type.contains("DetrImageProcessor") {
            self.detr_config().map(ProcessorFamilyConfig::Detr)
        } else if processor_type.contains("SamImageProcessor") {
            self.sam_config().map(ProcessorFamilyConfig::Sam)
        } else if processor_type.contains("Document")
            || processor_type.contains("OCR")
            || processor_type.contains("Donut")
        {
            self.document_ocr_config()
                .map(ProcessorFamilyConfig::DocumentOcr)
        } else {
            Err(ProcessorConfigError::UnsupportedProcessorType(
                processor_type.to_owned(),
            ))
        }
    }

    fn vae_config(&self) -> Result<VaeImageProcessorConfig, ProcessorConfigError> {
        let mut config = VaeImageProcessorConfig::default();
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(vae_latent_channels) = self.vae_latent_channels {
            config.vae_latent_channels = vae_latent_channels;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(false);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => {}
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn blip_config(&self) -> Result<BlipImageProcessorConfig, ProcessorConfigError> {
        let mut config = BlipImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.image_size_or_edge("size", config.size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(true);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => config.pixel_format = None,
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn flux2_config(&self) -> Result<Flux2ImageProcessorConfig, ProcessorConfigError> {
        let mut config = Flux2ImageProcessorConfig::default();
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(vae_latent_channels) = self.vae_latent_channels {
            config.vae_latent_channels = vae_latent_channels;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(true);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => config.pixel_format = None,
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn visual_cloze_config(&self) -> Result<VisualClozeProcessorConfig, ProcessorConfigError> {
        let mut config = VisualClozeProcessorConfig::default();
        if let Some(resolution) = self.resolution {
            config.resolution = resolution;
        }
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(vae_latent_channels) = self.vae_latent_channels {
            config.vae_latent_channels = vae_latent_channels;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(false);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => config.pixel_format = None,
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn hunyuan_video_15_config(
        &self,
    ) -> Result<HunyuanVideo15ImageProcessorConfig, ProcessorConfigError> {
        let mut config = HunyuanVideo15ImageProcessorConfig::default();
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(vae_latent_channels) = self.vae_latent_channels {
            config.vae_latent_channels = vae_latent_channels;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(true);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => config.pixel_format = None,
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn marigold_config(&self) -> MarigoldImageProcessorConfig {
        let mut config = MarigoldImageProcessorConfig::default();
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_range_check) = self.do_range_check {
            config.do_range_check = do_range_check;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        config
    }

    fn joy_image_edit_config(
        &self,
    ) -> Result<JoyImageEditImageProcessorConfig, ProcessorConfigError> {
        let mut config = JoyImageEditImageProcessorConfig::default();
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(basesize) = self.basesize {
            config.basesize = basesize;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(false);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => {}
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn wan_animate_config(&self) -> Result<WanAnimateImageProcessorConfig, ProcessorConfigError> {
        let mut config = WanAnimateImageProcessorConfig::default();
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(vae_latent_channels) = self.vae_latent_channels {
            config.vae_latent_channels = vae_latent_channels;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(do_binarize) = self.do_binarize {
            config.do_binarize = do_binarize;
        }

        let convert_rgb = self.do_convert_rgb.unwrap_or(false);
        let convert_grayscale = self.do_convert_grayscale.unwrap_or(false);
        if convert_rgb && convert_grayscale {
            return Err(ProcessorConfigError::ConflictingPixelFormatConversion);
        }

        match (convert_rgb, convert_grayscale) {
            (true, false) => config.pixel_format = Some(PixelFormat::Rgb8),
            (false, true) => config.pixel_format = Some(PixelFormat::Luma8),
            (false, false) => {}
            (true, true) => return Err(ProcessorConfigError::ConflictingPixelFormatConversion),
        }

        Ok(config)
    }

    fn ltx2_video_hdr_config(&self) -> Result<Ltx2VideoHdrProcessorConfig, ProcessorConfigError> {
        let mut config = Ltx2VideoHdrProcessorConfig::default();
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(vae_scale_factor) = self.vae_scale_factor {
            config.vae_scale_factor = vae_scale_factor;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        if let Some(hdr_transform) = &self.hdr_transform {
            if hdr_transform != "logc3" {
                return Err(ProcessorConfigError::UnsupportedHdrTransform(
                    hdr_transform.clone(),
                ));
            }
        }
        Ok(config)
    }

    fn clip_config(&self) -> Result<ClipImageProcessorConfig, ProcessorConfigError> {
        let mut config = ClipImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.image_size_or_edge("size", config.size)?;
        }
        if let Some(crop_size) = &self.crop_size {
            config.crop_size = crop_size.image_size_or_edge("crop_size", config.crop_size)?;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn vit_config(&self) -> Result<VitImageProcessorConfig, ProcessorConfigError> {
        let mut config = VitImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.image_size_or_edge("size", config.size)?;
        }
        if let Some(crop_size) = &self.crop_size {
            config.crop_size = crop_size.image_size_or_edge("crop_size", config.crop_size)?;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn video_mae_config(&self) -> Result<VideoMaeImageProcessorConfig, ProcessorConfigError> {
        let mut config = VideoMaeImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.square_edge("size", config.size)?;
        }
        if let Some(crop_size) = &self.crop_size {
            config.crop_size = crop_size.image_size_or_edge("crop_size", config.crop_size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn vivit_config(&self) -> Result<VivitImageProcessorConfig, ProcessorConfigError> {
        let mut config = VivitImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.square_edge("size", config.size)?;
        }
        if let Some(crop_size) = &self.crop_size {
            config.crop_size = crop_size.image_size_or_edge("crop_size", config.crop_size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(offset) = self.offset {
            config.offset = offset;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn qwen_config(&self) -> Result<QwenVlImageProcessorConfig, ProcessorConfigError> {
        let mut config = QwenVlImageProcessorConfig::default();
        if let Some(min_pixels) = self.min_pixels {
            config.resize_limits.min_pixels = min_pixels;
        }
        if let Some(max_pixels) = self.max_pixels {
            config.resize_limits.max_pixels = max_pixels;
        }
        if let Some(patch_size) = &self.patch_size {
            config.patch_size = patch_size.square_edge("patch_size", config.patch_size)?;
            config.resize_limits.factor =
                checked_hf_factor(config.patch_size, config.merge_size, "patch_size")?;
        }
        if let Some(temporal_patch_size) = self.temporal_patch_size {
            config.temporal_patch_size = temporal_patch_size;
        }
        if let Some(merge_size) = self.merge_size {
            config.merge_size = merge_size;
            config.resize_limits.factor =
                checked_hf_factor(config.patch_size, config.merge_size, "merge_size")?;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn llava_next_config(&self) -> Result<LlavaNextImageProcessorConfig, ProcessorConfigError> {
        let mut config = LlavaNextImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.image_size_or_edge("size", config.size)?;
        }
        if let Some(crop_size) = &self.crop_size {
            config.crop_size = crop_size.image_size_or_edge("crop_size", config.crop_size)?;
        }
        if let Some(image_grid_pinpoints) = &self.image_grid_pinpoints {
            config.image_grid_pinpoints = image_grid_pinpoints
                .iter()
                .map(|[height, width]| {
                    ImageSize::new(*height, *width).map_err(|_| ProcessorConfigError::InvalidSize {
                        field: "image_grid_pinpoints",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        if let Some(do_center_crop) = self.do_center_crop {
            config.do_center_crop = do_center_crop;
        }
        if let Some(do_pad) = self.do_pad {
            config.do_pad = do_pad;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn pixtral_config(&self) -> Result<PixtralImageProcessorConfig, ProcessorConfigError> {
        let mut config = PixtralImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.max_size = size.image_size_or_edge("size", config.max_size)?;
        }
        if let Some(patch_size) = &self.patch_size {
            config.patch_size = patch_size.image_size_or_edge("patch_size", config.patch_size)?;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn idefics3_config(&self) -> Result<Idefics3ImageProcessorConfig, ProcessorConfigError> {
        let mut config = Idefics3ImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.longest_edge = size.square_edge("size", config.longest_edge)?;
        }
        if let Some(max_image_size) = &self.max_image_size {
            config.max_image_size =
                max_image_size.square_edge("max_image_size", config.max_image_size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_image_splitting) = self.do_image_splitting {
            config.do_image_splitting = do_image_splitting;
        }
        if let Some(do_pad) = self.do_pad {
            config.do_pad = do_pad;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn gemma3_config(&self) -> Result<Gemma3ImageProcessorConfig, ProcessorConfigError> {
        let mut config = Gemma3ImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.size = size.image_size_or_edge("size", config.size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_pan_and_scan) = self.do_pan_and_scan {
            config.do_pan_and_scan = do_pan_and_scan;
        }
        if let Some(min_crop_size) = self.pan_and_scan_min_crop_size {
            config.pan_and_scan_min_crop_size = min_crop_size;
        }
        if let Some(max_num_crops) = self.pan_and_scan_max_num_crops {
            config.pan_and_scan_max_num_crops = max_num_crops;
        }
        if let Some(min_ratio_to_activate) = self.pan_and_scan_min_ratio_to_activate {
            config.pan_and_scan_min_ratio_to_activate = min_ratio_to_activate;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn mllama_config(&self) -> Result<MllamaImageProcessorConfig, ProcessorConfigError> {
        let mut config = MllamaImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.tile_size = size.square_edge("size", config.tile_size)?;
        }
        if let Some(max_image_tiles) = self.max_image_tiles {
            config.max_image_tiles = max_image_tiles;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_pad) = self.do_pad {
            config.do_pad = do_pad;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn detr_config(&self) -> Result<DetrImageProcessorConfig, ProcessorConfigError> {
        let mut config = DetrImageProcessorConfig::default();
        if let Some(size) = &self.size {
            if let Some(resize_size) = size.shortest_edge_resize_config("size")? {
                config.resize_size = resize_size;
            }
            config.image_size = size.image_size_or_edge("size", config.image_size)?;
        }
        if let Some(do_pad) = self.do_pad {
            config.do_pad = do_pad;
        }
        if let Some(pad_size) = &self.pad_size {
            config.pad_size = Some(pad_size.image_size_or_edge("pad_size", config.image_size)?);
        }
        if let Some(pad_to_multiple) = self.pad_to_multiple {
            config.pad_to_multiple = Some(pad_to_multiple);
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn sam_config(&self) -> Result<SamImageProcessorConfig, ProcessorConfigError> {
        let mut config = SamImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.image_size = size.image_size_or_edge("size", config.image_size)?;
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }

    fn document_ocr_config(&self) -> Result<DocumentOcrImageProcessorConfig, ProcessorConfigError> {
        let mut config = DocumentOcrImageProcessorConfig::default();
        if let Some(size) = &self.size {
            config.image_size = size.image_size_or_edge("size", config.image_size)?;
        }
        if let Some(do_resize) = self.do_resize {
            config.do_resize = do_resize;
        }
        if let Some(do_thumbnail) = self.do_thumbnail {
            config.do_thumbnail = do_thumbnail;
        }
        if let Some(do_align_long_axis) = self.do_align_long_axis {
            config.do_align_long_axis = do_align_long_axis;
        }
        if let Some(do_pad) = self.do_pad {
            config.do_pad = do_pad;
        }
        if let Some(do_rescale) = self.do_rescale {
            config.do_rescale = do_rescale;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            config.rescale_factor = rescale_factor;
        }
        if let Some(do_normalize) = self.do_normalize {
            config.do_normalize = do_normalize;
        }
        if let Some(image_mean) = &self.image_mean {
            config.image_mean = image_mean.clone();
        }
        if let Some(image_std) = &self.image_std {
            config.image_std = image_std.clone();
        }
        if let Some(resample) = &self.resample {
            config.resample = parse_hf_resample(resample, config.resample);
        }
        Ok(config)
    }
}

impl HfSize {
    fn square_edge(
        &self,
        field: &'static str,
        default: usize,
    ) -> Result<usize, ProcessorConfigError> {
        let size = self.image_size_or_edge(
            field,
            ImageSize::new(default, default)
                .map_err(|_| ProcessorConfigError::InvalidSize { field })?,
        )?;
        if size.height == size.width {
            Ok(size.height)
        } else {
            Err(ProcessorConfigError::InvalidSize { field })
        }
    }

    fn image_size_or_edge(
        &self,
        field: &'static str,
        default: ImageSize,
    ) -> Result<ImageSize, ProcessorConfigError> {
        match self {
            Self::Edge(edge) => ImageSize::new(*edge, *edge)
                .map_err(|_| ProcessorConfigError::InvalidSize { field }),
            Self::Map {
                height,
                width,
                shortest_edge,
                longest_edge,
                max_height,
                max_width,
            } => {
                if let (Some(height), Some(width)) =
                    ((*height).or(*max_height), (*width).or(*max_width))
                {
                    ImageSize::new(height, width)
                        .map_err(|_| ProcessorConfigError::InvalidSize { field })
                } else if let Some(edge) = (*shortest_edge).or(*longest_edge) {
                    ImageSize::new(edge, edge)
                        .map_err(|_| ProcessorConfigError::InvalidSize { field })
                } else {
                    Ok(default)
                }
            }
        }
    }

    fn shortest_edge_resize_config(
        &self,
        field: &'static str,
    ) -> Result<Option<ShortestEdgeResizeConfig>, ProcessorConfigError> {
        match self {
            Self::Edge(edge) => {
                let config = ShortestEdgeResizeConfig::new(*edge, None)
                    .ok_or(ProcessorConfigError::InvalidSize { field })?;
                Ok(Some(config))
            }
            Self::Map {
                shortest_edge,
                longest_edge,
                ..
            } => {
                let Some(shortest_edge) = (*shortest_edge).or(*longest_edge) else {
                    return Ok(None);
                };
                let config = ShortestEdgeResizeConfig::new(shortest_edge, *longest_edge)
                    .ok_or(ProcessorConfigError::InvalidSize { field })?;
                Ok(Some(config))
            }
        }
    }
}

fn checked_hf_factor(
    patch_size: usize,
    merge_size: usize,
    field: &'static str,
) -> Result<usize, ProcessorConfigError> {
    patch_size
        .checked_mul(merge_size)
        .filter(|factor| *factor > 0)
        .ok_or(ProcessorConfigError::InvalidSize { field })
}

fn parse_hf_resample(value: &serde_json::Value, default: ResizeFilter) -> ResizeFilter {
    match value {
        serde_json::Value::Number(number) => match number.as_u64() {
            Some(0) => ResizeFilter::Nearest,
            Some(1) => ResizeFilter::Lanczos,
            Some(2) => ResizeFilter::Bilinear,
            Some(3) => ResizeFilter::Bicubic,
            _ => default,
        },
        serde_json::Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "nearest" => ResizeFilter::Nearest,
            "lanczos" => ResizeFilter::Lanczos,
            "bilinear" => ResizeFilter::Bilinear,
            "bicubic" => ResizeFilter::Bicubic,
            _ => default,
        },
        _ => default,
    }
}
