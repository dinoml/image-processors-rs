//! EoMT window extraction using the shared resize, crop, pad and tensor owners.

use super::*;
use crate::recipe::{
    RecipeSegmentationPostprocess, RecipeSegmentationStrategy, RecipeSegmentationTask,
};
use crate::transforms::{crop_frame, pad_frame, Padding};

/// EoMT's preprocessing controls, including settings absent from generic task presets.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EomtImageProcessorConfig {
    /// Shared resize, pixel conversion, rescaling, normalization and layout settings.
    pub image: TaskVisionImageProcessorConfig,
    /// Upstream shortest-edge target and optional longest-edge cap.
    pub size: ShortestEdgeResizeConfig,
    /// Extract overlapping windows along each resized image's longer edge.
    pub do_split_image: bool,
    /// Pad each window at the bottom and right before numeric normalization.
    pub do_pad: bool,
}

impl EomtImageProcessorConfig {
    /// Return EoMT's task postprocessing recipe independently of resize enablement.
    ///
    /// # Errors
    /// Reports invalid canvas or recipe options.
    pub fn processor_recipe(&self) -> Result<ProcessorRecipe, RecipeError> {
        let mut descriptor = RecipeSegmentationPostprocess::new(
            RecipeSegmentationTask::Semantic,
            "class_queries_logits",
            "masks_queries_logits",
            None,
        )
        .with_strategy(RecipeSegmentationStrategy::Eomt {
            size: ImageSize {
                height: self.size.shortest_edge,
                width: self.size.longest_edge.unwrap_or(self.size.shortest_edge),
            },
        });
        descriptor.options.score_threshold = 0.8;
        descriptor.options.target_size_source = Some(RecipeImageSizeSource::CallerProvided);
        ProcessorRecipe::new_postprocess_only(
            "transformers.eomt_image_processor",
            ProcessorRecipeInput::default(),
            ProcessorRecipeOutput::batch(Layout::NCHW),
            vec![ProcessorRecipePostprocess::Segmentation(descriptor)],
        )
    }
}

/// EoMT preprocessing with original sizes and flattened `(image, start, end)` offsets.
pub struct EomtImageProcessor {
    config: EomtImageProcessorConfig,
    geometry: TaskVisionImageProcessor,
    tensor: ImageProcessor,
}

impl EomtImageProcessor {
    /// Create a processor using existing shared geometry and numeric conversion.
    ///
    /// # Errors
    /// Reports invalid canvas, normalization or geometry settings.
    pub fn new(config: EomtImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        ImageSize::new(
            config.size.shortest_edge,
            config
                .size
                .longest_edge
                .unwrap_or(config.size.shortest_edge),
        )?;
        let mut image = config.image.clone();
        image.pre_resize_padding = TaskVisionPadding::None;
        image.post_resize_padding = TaskVisionPadding::None;
        let tensor = ImageProcessor::new(image.tensor_processor_config(PixelFormat::Rgb8))?;
        let geometry = TaskVisionImageProcessor::new(image)?;
        Ok(Self {
            config,
            geometry,
            tensor,
        })
    }

    /// Return the effective EoMT preprocessing settings.
    pub fn config(&self) -> &EomtImageProcessorConfig {
        &self.config
    }

    /// Preprocess one image, retaining its windows in model batch order.
    ///
    /// # Errors
    /// Reports resize, crop, padding or tensor conversion failures.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images_output(std::slice::from_ref(image))
    }

    /// Preprocess images into windows ordered by original image and window start.
    ///
    /// `patch_offsets` metadata is a flat sequence of image index, inclusive start
    /// and exclusive end along the resized long axis, matching Transformers.
    ///
    /// # Errors
    /// Reports empty/incompatible batches, arithmetic overflow or frame processing failures.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Upstream computes overlapping starts in double precision and truncates nonnegative values"
    )]
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        let mut frames = Vec::new();
        let mut originals = Vec::with_capacity(images.len());
        let mut resized = Vec::with_capacity(images.len());
        let mut offsets = Vec::new();
        for (image_index, image) in images.iter().enumerate() {
            let (frame, original, reshaped) = self.geometry.prepare_image(image)?;
            originals.push(original);
            resized.push(reshaped);
            if self.config.do_split_image {
                let plan = super::OverlappingImagePlan::new(
                    original,
                    reshaped,
                    self.config.size.shortest_edge,
                )?;
                for crop in plan.crops() {
                    let (start, end) = if frame.height() > frame.width() {
                        (crop.y_min, crop.y_max)
                    } else {
                        (crop.x_min, crop.x_max)
                    };
                    frames.push(crop_frame(&frame, *crop)?);
                    offsets.extend([image_index, start, end]);
                }
            } else {
                frames.push(frame);
            }
        }
        if self.config.do_pad {
            for frame in &mut frames {
                let padding = Padding::new(
                    0,
                    self.config
                        .size
                        .longest_edge
                        .unwrap_or(self.config.size.shortest_edge)
                        .saturating_sub(frame.width()),
                    self.config
                        .size
                        .shortest_edge
                        .saturating_sub(frame.height()),
                    0,
                );
                *frame = pad_frame(frame, padding, &[0])?;
            }
        }
        let mut output =
            ProcessorOutput::from_pixel_values(self.tensor.preprocess_images(&frames)?);
        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(originals),
        );
        output.insert_metadata(
            ProcessorMetadataName::ReshapedInputSizes,
            ProcessorMetadataValue::ImageSizes(resized),
        );
        output.insert_metadata(
            ProcessorMetadataName::Other("eomt_resize".into()),
            ProcessorMetadataValue::Shape(vec![
                self.config.size.shortest_edge,
                self.config.size.longest_edge.unwrap_or(0),
            ]),
        );
        if self.config.do_split_image {
            output.insert_metadata(
                ProcessorMetadataName::Other("patch_offsets".into()),
                ProcessorMetadataValue::Shape(offsets),
            );
        }
        Ok(output)
    }
}
