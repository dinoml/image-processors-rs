//! DPT geometry composed with shared pixel conversion, normalization and padding.

use super::*;
use crate::image::ImageProcessorPrecision;
use crate::postprocess::post_process_dense_semantic_segmentation;
use crate::transforms::{
    closest_scale_resize_size, pad_to_multiple_plan, SemanticSegmentationPrediction,
};
use image_resize_kernels::{resize_f32_planes, TensorInterpolation};

/// Upstream DPT image-resize implementation.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DptResizeBackend {
    /// PIL byte interpolation, followed by floating rescale and normalization.
    Pil,
    /// Torchvision floating interpolation with antialiasing.
    #[default]
    Torchvision,
}

/// DPT-specific geometry around the general image processor configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DptImageProcessorConfig {
    /// Requested height and width before aspect and multiple adjustments.
    pub size: ImageSize,
    /// Whether decoded pixels are resized.
    pub do_resize: bool,
    /// Optional pixel conversion before geometry; `None` preserves source channels.
    pub pixel_format: Option<PixelFormat>,
    /// Optional center crop after resize and before rescale or normalization.
    pub crop_size: Option<ImageSize>,
    /// Source interpolation filter.
    pub resample: ResizeFilter,
    /// Whether byte values are rescaled.
    pub do_rescale: bool,
    /// Rescale multiplier, retaining PIL's FP64 scalar precision.
    pub rescale_factor: f64,
    /// Whether per-channel normalization is applied.
    pub do_normalize: bool,
    /// Per-channel or scalar normalization mean.
    pub image_mean: Vec<f32>,
    /// Per-channel or scalar normalization standard deviation.
    pub image_std: Vec<f32>,
    /// Channel order of the resulting batched image tensor.
    pub output_layout: ImageLayout,
    /// Select the scale closest to one while preserving the original aspect ratio.
    pub keep_aspect_ratio: bool,
    /// Round each resized axis to this multiple using nearest-even rounding.
    pub ensure_multiple_of: usize,
    /// Center-pad normalized tensors to this divisor, when supplied.
    pub size_divisor: Option<usize>,
    /// Select upstream PIL or Torchvision precision and interpolation semantics.
    pub backend: DptResizeBackend,
}

impl Default for DptImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: ImageSize {
                height: 384,
                width: 384,
            },
            do_resize: true,
            pixel_format: Some(PixelFormat::Rgb8),
            crop_size: None,
            resample: ResizeFilter::Bicubic,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            output_layout: ImageLayout::ChannelsHeightWidth,
            keep_aspect_ratio: false,
            ensure_multiple_of: 1,
            size_divisor: None,
            backend: DptResizeBackend::Torchvision,
        }
    }
}

/// A relative depth map and its upstream min/max-normalized grayscale view.
#[derive(Clone, Debug, PartialEq)]
pub struct DptDepthPrediction {
    /// Spatial extent of both row-major outputs.
    pub size: ImageSize,
    /// Relative depth values after optional bicubic restoration.
    pub predicted_depth: Vec<f32>,
    /// Min/max-normalized grayscale pixels, truncated to bytes as upstream.
    pub depth: Vec<u8>,
}

/// Semantic logits and the per-class byte masks used by image segmentation.
#[derive(Clone, Debug)]
pub struct DptSemanticPrediction {
    /// Restored class indices and raw score planes.
    pub segmentation: SemanticSegmentationPrediction,
    /// Masks for classes present in the prediction, in ascending label order.
    pub masks: std::collections::BTreeMap<i64, Vec<u8>>,
}

/// DPT image processing without model-local geometry or native imaging bindings.
pub struct DptImageProcessor {
    config: DptImageProcessorConfig,
}

impl DptImageProcessor {
    /// Construct a processor with validated generic settings and DPT divisors.
    ///
    /// # Errors
    /// Rejects invalid image settings, layouts, resize multiples or padding divisors.
    pub fn new(config: DptImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        if config.ensure_multiple_of == 0 || config.size_divisor == Some(0) {
            return Err(ImageProcessorError::Transform(
                TransformError::InvalidScaleFactor(0),
            ));
        }
        if !config.rescale_factor.is_finite() {
            return Err(ImageProcessorError::NonFiniteRescaleFactor(
                config.rescale_factor as f32,
            ));
        }
        ImageSize::new(config.size.height, config.size.width)?;
        if let Some(size) = config.crop_size {
            ImageSize::new(size.height, size.width)?;
        }
        Ok(Self { config })
    }

    /// Preprocess one image, padding after normalization with exact floating zeros.
    ///
    /// # Errors
    /// Reports invalid geometry, pixel conversion, interpolation, normalization or padding.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        let mut config = ImageProcessorConfig {
            do_resize: self.config.do_resize,
            pixel_format: self.config.pixel_format,
            resample: self.config.resample,
            do_rescale: false,
            do_normalize: false,
            output_layout: batched_layout_for_image_layout(self.config.output_layout),
            ..ImageProcessorConfig::default()
        };
        if self.config.do_resize {
            let target = closest_scale_resize_size(
                ImageSize::new(image.height(), image.width())?,
                self.config.size,
                self.config.keep_aspect_ratio,
                self.config.ensure_multiple_of,
            )?;
            config.height = Some(target.height);
            config.width = Some(target.width);
        }
        config.resize_parity = match self.config.backend {
            DptResizeBackend::Pil => ResizeParity::Compatibility,
            DptResizeBackend::Torchvision => ResizeParity::Torchvision,
        };
        let precision = match self.config.backend {
            DptResizeBackend::Pil => ImageProcessorPrecision::Byte,
            DptResizeBackend::Torchvision => ImageProcessorPrecision::Float32 {
                rescale_divisor: None,
            },
        };
        let processor = ImageProcessor::new(config)?;
        let mut tensor = processor.preprocess_image_with_precision(
            image,
            ImageProcessorOptions::default(),
            precision,
        )?;
        if let Some(size) = self.config.crop_size {
            let source = ImageSize::new(tensor.height(), tensor.width())?;
            let interleaved = tensor.to_layout(Layout::NHWC)?;
            let channels = interleaved.shape()[3];
            let TensorData::F32(values) = interleaved.data() else {
                return Err(ImageProcessorError::ExpectedDataType("floating pixels"));
            };
            let values = super::encoder_presets::center_crop_interleaved_f32(
                values,
                source,
                channels,
                size,
                self.config.backend == DptResizeBackend::Pil,
                self.config.backend != DptResizeBackend::Pil,
            )?;
            tensor = Tensor::new(
                TensorData::F32(values),
                vec![1, size.height, size.width, channels],
                Layout::NHWC,
            )?
            .to_layout(processor.config().output_layout)?;
        }
        // PIL rescales through FP64 and narrows to FP32 before normalization.
        // Torchvision fuses rescale into normalization statistics in FP32.
        if self.config.backend == DptResizeBackend::Torchvision
            && self.config.do_rescale
            && self.config.do_normalize
        {
            let scale = self.config.rescale_factor as f32;
            let mean = self
                .config
                .image_mean
                .iter()
                .map(|value| value / scale)
                .collect::<Vec<_>>();
            let std = self
                .config
                .image_std
                .iter()
                .map(|value| value / scale)
                .collect::<Vec<_>>();
            tensor = tensor.normalize_channels(&mean, &std)?;
        } else {
            if self.config.do_rescale {
                tensor = match self.config.backend {
                    DptResizeBackend::Pil => tensor.rescale_f64(self.config.rescale_factor)?,
                    DptResizeBackend::Torchvision => {
                        tensor.rescale(self.config.rescale_factor as f32)?
                    }
                };
            }
            if self.config.do_normalize {
                tensor =
                    tensor.normalize_channels(&self.config.image_mean, &self.config.image_std)?;
            }
        }
        if let Some(divisor) = self.config.size_divisor {
            let plan = pad_to_multiple_plan(
                ImageSize::new(tensor.height(), tensor.width())?,
                ImageSize::new(divisor, divisor)?,
            )?;
            return pad_spatial_tensors_with_alignment(
                vec![tensor],
                plan.padded_size,
                processor.config().output_image_layout(),
                SpatialPaddingAlignment::Center,
            );
        }
        Ok(tensor)
    }

    /// Restore one relative depth plane with bicubic half-pixel interpolation.
    ///
    /// `precision` is the model output dtype, including rounding after resize.
    /// No target size retains source resolution. Constant maps produce black views.
    ///
    /// # Errors
    /// Reports invalid storage, dimensions, precision or resize failures.
    pub fn post_process_depth(
        values: &[f32],
        source: ImageSize,
        target: Option<ImageSize>,
        precision: DType,
    ) -> Result<DptDepthPrediction, ImageProcessorError> {
        if !matches!(precision, DType::F32 | DType::F16 | DType::BF16) {
            return Err(ImageProcessorError::ExpectedDataType("floating depth"));
        }
        let round = |value: f32| match precision {
            DType::F16 => half::f16::from_f32(value).to_f32(),
            DType::BF16 => half::bf16::from_f32(value).to_f32(),
            _ => value,
        };
        let size = target.unwrap_or(source);
        let mut values = resize_f32_planes(
            values,
            1,
            (source.height, source.width),
            (size.height, size.width),
            TensorInterpolation::Bicubic,
            false,
        )?;
        for value in &mut values {
            *value = round(*value);
        }
        let depth = Self::normalize_depth(&values, precision)?
            .iter()
            .map(|value| round(value * 255.0) as u8)
            .collect();
        Ok(DptDepthPrediction {
            size,
            predicted_depth: values,
            depth,
        })
    }

    /// Normalize depth to its min/max range, retaining arithmetic precision.
    ///
    /// The unquantized result is also used for MiDaS background masking before
    /// normal-map construction. Constant maps retain upstream NaNs, which cast
    /// to black when producing a byte visualization.
    ///
    /// # Errors
    /// Rejects nonfloating precision.
    pub fn normalize_depth(
        values: &[f32],
        precision: DType,
    ) -> Result<Vec<f32>, ImageProcessorError> {
        if !matches!(precision, DType::F32 | DType::F16 | DType::BF16) {
            return Err(ImageProcessorError::ExpectedDataType("floating depth"));
        }
        let round = |value: f32| match precision {
            DType::F16 => half::f16::from_f32(value).to_f32(),
            DType::BF16 => half::bf16::from_f32(value).to_f32(),
            _ => value,
        };
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = round(max - min);
        Ok(values
            .iter()
            .map(|value| round(round(value - min) / span))
            .collect())
    }

    /// Restore raw semantic logits before selecting class indices.
    ///
    /// # Errors
    /// Reports invalid logits, precision, dimensions or shared interpolation failures.
    pub fn post_process_semantic(
        values: &[f32],
        channels: usize,
        source: ImageSize,
        target: Option<ImageSize>,
        precision: DType,
    ) -> Result<DptSemanticPrediction, crate::postprocess::MulticlassSegmentationError> {
        let segmentation = post_process_dense_semantic_segmentation(
            values,
            channels,
            source,
            target.unwrap_or(source),
            precision,
        )?;
        let masks = Self::semantic_masks(&segmentation.class_ids);
        Ok(DptSemanticPrediction {
            segmentation,
            masks,
        })
    }

    /// Build sorted per-class byte masks from an independently supplied class map.
    pub fn semantic_masks(class_ids: &[i64]) -> std::collections::BTreeMap<i64, Vec<u8>> {
        let labels = class_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        labels
            .into_iter()
            .map(|label| {
                let mask = class_ids
                    .iter()
                    .map(|id| if *id == label { 255 } else { 0 })
                    .collect();
                (label, mask)
            })
            .collect()
    }
}
