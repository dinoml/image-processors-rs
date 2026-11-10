//! Transformers ViTPose box-affine crops and SciPy-profile DARK decoding.

use image_resize_kernels::gaussian_blur_reflect_f32_planes;
use serde::Deserialize;
use thiserror::Error;

use super::pose::{dark_refine_keypoint, HessianPrecision};
use super::task_vision_presets::{vitpose_affine_frame, vitpose_box_geometry};
use super::{
    ImageGptPreprocessingPolicy, TaskVisionImageProcessorConfig, TaskVisionPoseBox,
    TaskVisionProcessorPreset, TaskVisionResize,
};
use crate::tensor::{Layout, Tensor, TensorData, TensorError};
use crate::transforms::{convert_frame_pixel_format, ImageSize, TransformError};
use crate::{ImageFrame, ImageProcessor, ImageProcessorConfig, ImageProcessorError, PixelFormat};

/// Invalid settings, preprocessing or pose-decoding failures.
#[derive(Debug, Error)]
pub enum VitPoseProcessorError {
    /// Invalid processor or heatmap geometry.
    #[error("invalid ViTPose processor input: {0}")]
    Invalid(&'static str),
    /// Preprocessor JSON decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Shared image processing failed.
    #[error(transparent)]
    Image(#[from] ImageProcessorError),
    /// Tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Image geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] TransformError),
    /// Shared Gaussian kernel failed.
    #[error(transparent)]
    Kernel(#[from] image_resize_kernels::ResizeError),
    /// Shared normalization failed.
    #[error(transparent)]
    Normalization(#[from] super::PresetProcessorError),
}

/// Confidence-filtered keypoints from one person heatmap.
#[derive(Debug, Clone)]
pub struct VitPosePose {
    /// XY coordinates restored to the supplied image-box coordinate system.
    pub keypoints: Vec<[f32; 2]>,
    /// Original, unblurred heatmap peak scores.
    pub scores: Vec<f32>,
    /// Original keypoint channel identifiers.
    pub labels: Vec<usize>,
    /// Upstream bbox output: center and normalized scale converted from COCO to XYXY.
    pub bbox: [f32; 4],
}

#[derive(Deserialize)]
struct CropSize {
    height: usize,
    width: usize,
}

#[derive(Deserialize)]
struct Settings {
    image_processor_type: Option<String>,
    size: Option<CropSize>,
    do_affine_transform: Option<bool>,
    normalize_factor: Option<f64>,
    do_rescale: Option<bool>,
    rescale_factor: Option<f64>,
    do_normalize: Option<bool>,
    image_mean: Option<Vec<f32>>,
    image_std: Option<Vec<f32>>,
}

/// Configurable ViTPose processor; image geometry and pixel conversion retain their shared owners.
pub struct VitPoseImageProcessor {
    config: ImageProcessorConfig,
    tensor_processor: ImageProcessor,
    size: ImageSize,
    affine: bool,
    normalize_factor: f64,
    rescale_factor: f64,
    policy: ImageGptPreprocessingPolicy,
}

impl VitPoseImageProcessor {
    /// Configure box-affine preprocessing and FP32 tensor normalization.
    /// Input images without affine preprocessing must already have the desired shape.
    ///
    /// # Errors
    /// Reports invalid crop geometry, scale, layout or normalization settings.
    pub fn new(
        mut config: ImageProcessorConfig,
        size: ImageSize,
        affine: bool,
        normalize_factor: f64,
        policy: ImageGptPreprocessingPolicy,
    ) -> Result<Self, VitPoseProcessorError> {
        if size.height < 2
            || size.width < 2
            || !normalize_factor.is_finite()
            || normalize_factor <= 0.0
        {
            return Err(VitPoseProcessorError::Invalid(
                "crop extents must exceed one and normalize_factor must be positive",
            ));
        }
        config.do_resize = false;
        config.output_layout = Layout::NCHW;
        config.pixel_format = Some(PixelFormat::Rgb8);
        ImageProcessor::new(config.clone())?;
        let rescale_factor = f64::from(config.rescale_factor);
        let mut raw_config = config.clone();
        raw_config.do_rescale = false;
        raw_config.do_normalize = false;
        raw_config.do_binarize = false;
        Ok(Self {
            config,
            tensor_processor: ImageProcessor::new(raw_config)?,
            size,
            affine,
            normalize_factor,
            rescale_factor,
            policy,
        })
    }

    /// Read checkpoint preprocessing settings using the existing VitPose preset defaults.
    /// JSON rescale precision is retained until the shared normalization stage.
    ///
    /// # Errors
    /// Reports malformed JSON, a different processor class or invalid settings.
    pub fn from_hf_preprocessor_json(json: &str) -> Result<Self, VitPoseProcessorError> {
        let settings: Settings = serde_json::from_str(json)?;
        let policy = match settings.image_processor_type.as_deref() {
            Some("VitPoseImageProcessorPil") => ImageGptPreprocessingPolicy::Pil,
            None | Some("VitPoseImageProcessor") | Some("VitPoseImageProcessorFast") => {
                ImageGptPreprocessingPolicy::Torchvision
            }
            _ => {
                return Err(VitPoseProcessorError::Invalid(
                    "expected a ViTPose image processor",
                ));
            }
        };
        let mut config =
            TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::VitPose);
        let size = if let Some(size) = settings.size {
            ImageSize::new(size.height, size.width)?
        } else if let TaskVisionResize::Fixed { size } = config.resize {
            size
        } else {
            return Err(VitPoseProcessorError::Invalid(
                "ViTPose preset requires a fixed crop size",
            ));
        };
        if let Some(value) = settings.do_rescale {
            config.do_rescale = value;
        }
        if let Some(value) = settings.rescale_factor {
            config.rescale_factor = value as f32;
        }
        if let Some(value) = settings.do_normalize {
            config.do_normalize = value;
        }
        if let Some(value) = settings.image_mean {
            config.image_mean = value;
        }
        if let Some(value) = settings.image_std {
            config.image_std = value;
        }
        let mut processor = Self::new(
            config.tensor_processor_config(PixelFormat::Rgb8),
            size,
            settings.do_affine_transform.unwrap_or(true),
            settings.normalize_factor.unwrap_or(200.0),
            policy,
        )?;
        if let Some(factor) = settings.rescale_factor {
            processor.rescale_factor = factor;
        }
        Ok(processor)
    }

    /// Select the upstream PIL or Torchvision rescale/normalization order.
    pub fn with_policy(mut self, policy: ImageGptPreprocessingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Return the checkpoint's person-crop geometry.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Convert one image and its COCO-format boxes into a person batch.
    /// With affine disabled, the image contributes one already-prepared crop.
    ///
    /// # Errors
    /// Reports empty boxes, invalid pixel geometry or shared preprocessing failures.
    pub fn preprocess(
        &self,
        image: &ImageFrame,
        boxes: &[TaskVisionPoseBox],
    ) -> Result<Tensor, VitPoseProcessorError> {
        let image = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        let raw = if self.affine {
            if boxes.is_empty() {
                return Err(VitPoseProcessorError::Invalid(
                    "affine preprocessing requires boxes",
                ));
            }
            let crops = boxes
                .iter()
                .map(|&bbox| vitpose_affine_frame(&image, bbox, self.size, self.normalize_factor))
                .collect::<Result<Vec<_>, _>>()?;
            self.tensor_processor.preprocess_images(&crops)?
        } else {
            self.tensor_processor.preprocess_image(&image)?
        };
        Ok(self.policy.finish_tensor(
            raw,
            self.config.do_rescale.then_some(self.rescale_factor),
            self.config.do_normalize.then_some((
                self.config.image_mean.as_slice(),
                self.config.image_std.as_slice(),
            )),
            None,
        )?)
    }

    /// Decode NCHW FP32 heatmaps with SciPy sigma=0.8 reflect filtering and DARK.
    /// `target_size` retains the upstream `target_sizes` pair order: its first
    /// value scales box X/width and its second scales box Y/height. It is useful
    /// when the supplied boxes are normalized coordinates.
    ///
    /// # Errors
    /// Reports invalid tensor geometry, kernel, target scale or non-FP32 heatmaps.
    pub fn postprocess(
        &self,
        heatmaps: &Tensor,
        boxes: &[TaskVisionPoseBox],
        kernel: usize,
        threshold: Option<f32>,
        target_size: Option<[f64; 2]>,
    ) -> Result<Vec<VitPosePose>, VitPoseProcessorError> {
        let shape = heatmaps.shape();
        if heatmaps.layout() != Layout::NCHW
            || shape.len() != 4
            || shape[0] != boxes.len()
            || boxes.is_empty()
            || shape[1] == 0
            || shape[2] < 2
            || shape[3] < 2
            || kernel == 0
        {
            return Err(VitPoseProcessorError::Invalid(
                "expected nonempty [persons,keypoints,height,width] heatmaps and a positive kernel",
            ));
        }
        if target_size
            .is_some_and(|size| size.iter().any(|value| !value.is_finite() || *value <= 0.0))
        {
            return Err(VitPoseProcessorError::Invalid(
                "target dimensions must be finite and positive",
            ));
        }
        let TensorData::F32(maps) = heatmaps.data() else {
            return Err(VitPoseProcessorError::Invalid(
                "DARK requires FP32 heatmaps",
            ));
        };
        let (persons, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let padded = height
            .checked_add(2)
            .and_then(|h| width.checked_add(2).and_then(|w| h.checked_mul(w)))
            .and_then(|pixels| {
                persons
                    .checked_mul(channels)
                    .and_then(|count| count.checked_mul(pixels))
            });
        if padded.is_none() {
            return Err(VitPoseProcessorError::Invalid(
                "padded heatmap size overflow",
            ));
        }
        // SciPy's radius is floor((kernel - 1)/2), including even user kernels.
        let mut blurred = gaussian_blur_reflect_f32_planes(
            maps,
            image_resize_kernels::ImageSize::new(height, width)?,
            persons * channels,
            2 * ((kernel - 1) / 2) + 1,
            0.8,
        )?;
        for value in &mut blurred {
            *value = value.clamp(0.001, 50.0).ln();
        }
        let mut results = Vec::with_capacity(persons);
        for (person, &bbox) in boxes.iter().enumerate() {
            let factors = target_size.unwrap_or([1.0, 1.0]);
            let (center, scale) = vitpose_box_geometry(
                [
                    f64::from(bbox.top_left_x()) * factors[0],
                    f64::from(bbox.top_left_y()) * factors[1],
                    f64::from(bbox.width()) * factors[0],
                    f64::from(bbox.height()) * factors[1],
                ],
                self.size,
                200.0,
            );
            let mut result = VitPosePose {
                keypoints: Vec::new(),
                scores: Vec::new(),
                labels: Vec::new(),
                bbox: [
                    center[0],
                    center[1],
                    center[0] + scale[0],
                    center[1] + scale[1],
                ],
            };
            let extent = [scale[0] * 200.0, scale[1] * 200.0];
            // transform_preds applies the scale/translation to FP32 coordinate
            // arrays, retaining the rounding between these vector operations.
            let coordinate_scale = [
                extent[0] / (width - 1) as f32,
                extent[1] / (height - 1) as f32,
            ];
            for channel in 0..channels {
                let offset = (person * channels + channel) * height * width;
                let plane = &maps[offset..offset + height * width];
                let (mut peak, mut score) = (0, plane[0]);
                for (index, &value) in plane.iter().enumerate().skip(1) {
                    if value > score || (value.is_nan() && !score.is_nan()) {
                        peak = index;
                        score = value;
                    }
                }
                if threshold.is_some_and(|threshold| {
                    score.partial_cmp(&threshold) != Some(std::cmp::Ordering::Greater)
                }) {
                    continue;
                }
                let position = if score > 0.0 {
                    [(peak % width) as isize, (peak / width) as isize]
                } else {
                    [-1, -1]
                };
                let [px, py] = dark_refine_keypoint(
                    &blurred,
                    [persons, channels, height, width],
                    person,
                    channel,
                    position,
                    HessianPrecision::Float64,
                );
                result.keypoints.push([
                    px * coordinate_scale[0] + center[0] - extent[0] * 0.5,
                    py * coordinate_scale[1] + center[1] - extent[1] * 0.5,
                ]);
                result.scores.push(score);
                result.labels.push(channel);
            }
            results.push(result);
        }
        Ok(results)
    }
}
