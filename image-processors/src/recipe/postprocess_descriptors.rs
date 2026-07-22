use super::*;
use serde::{Deserialize, Serialize};

/// A task-specific postprocessing descriptor attached to a processor recipe.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "task", content = "parameters", rename_all = "snake_case")]
pub enum ProcessorRecipePostprocess {
    /// Restore object-detection logits and boxes into image-space predictions.
    ObjectDetection(RecipeObjectDetectionPostprocess),
    /// Restore semantic, instance, or panoptic segmentation logits.
    Segmentation(RecipeSegmentationPostprocess),
    /// Restore low-resolution binary mask logits to original image mask planes.
    BinaryMasks(RecipeBinaryMaskPostprocess),
    /// Restore dense depth-like outputs to image space.
    Depth(RecipeDepthPostprocess),
    /// Restore dense image-like output maps to image space.
    DenseMap(RecipeDenseMapPostprocess),
    /// Apply common video tensor postprocessing transfer functions.
    VideoTensor(RecipeVideoTensorPostprocess),
    /// Restore point, keypoint, matching, or pose coordinates.
    Coordinates(RecipeCoordinatePostprocess),
    /// Greedily decode structured logits into tokenizer-independent token ids.
    TokenSequence(RecipeTokenSequencePostprocess),
    /// Capture a task-labeled output tensor and optional target-size metadata.
    OutputHook(RecipeOutputHookPostprocess),
}

/// Source used to resolve image sizes during postprocessing.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "source", content = "name", rename_all = "snake_case")]
pub enum RecipeImageSizeSource {
    /// Use `original_sizes` metadata emitted by preprocessing.
    OriginalSizes,
    /// Use `reshaped_input_sizes` metadata emitted by preprocessing.
    ReshapedInputSizes,
    /// Use explicit target sizes supplied with the model outputs.
    CallerProvided,
    /// Use a processor-specific metadata field.
    Other(String),
}

/// Object-detection postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecipeObjectDetectionPostprocess {
    /// Model output name containing class logits.
    pub logits_output: String,
    /// Model output name containing normalized center-format boxes.
    pub boxes_output: String,
    /// Number of labels in the logits, including the final background label.
    ///
    /// This is optional because the value belongs to the model head rather than
    /// the image processor config for many detection families.
    pub num_labels_with_background: Option<usize>,
    /// Class-score activation and query-selection policy.
    #[serde(default)]
    pub score_mode: RecipeDetectionScoreMode,
    /// Maximum number of flattened query-class predictions retained before thresholding.
    ///
    /// This is used only by [`RecipeDetectionScoreMode::SigmoidTopK`]. When
    /// absent, that mode retains at most one prediction per query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Minimum foreground score required to keep a prediction.
    pub score_threshold: f32,
    /// Source for target image sizes used to scale boxes.
    pub target_size_source: RecipeImageSizeSource,
}

impl RecipeObjectDetectionPostprocess {
    /// Creates an object-detection postprocessing descriptor.
    pub fn new(
        logits_output: impl Into<String>,
        boxes_output: impl Into<String>,
        num_labels_with_background: Option<usize>,
        score_threshold: f32,
        target_size_source: RecipeImageSizeSource,
    ) -> Self {
        Self {
            logits_output: logits_output.into(),
            boxes_output: boxes_output.into(),
            num_labels_with_background,
            score_mode: RecipeDetectionScoreMode::default(),
            top_k: None,
            score_threshold,
            target_size_source,
        }
    }

    /// Returns this descriptor with an explicit class-score policy.
    pub fn with_score_mode(mut self, score_mode: RecipeDetectionScoreMode) -> Self {
        self.score_mode = score_mode;
        self
    }

    /// Returns this descriptor with a flattened sigmoid top-k limit.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Class-score activation and query-selection policy for object detection.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeDetectionScoreMode {
    /// Apply softmax per query, excluding the final no-object class.
    #[default]
    SoftmaxWithBackground,
    /// Apply sigmoid and retain the best class independently for each query.
    SigmoidBestPerQuery,
    /// Apply sigmoid, rank flattened query-class pairs, then retain top-k entries.
    SigmoidTopK,
}

/// Segmentation postprocessing task kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeSegmentationTask {
    /// Produce one semantic class id per pixel.
    Semantic,
    /// Produce instance segment ids and segment metadata.
    Instance,
    /// Produce panoptic segment ids with optional label fusion.
    Panoptic,
}

/// Upstream query-mask postprocessing contract used by a segmentation descriptor.
///
/// These families share model output names, but their semantic, instance, and
/// panoptic algorithms are not interchangeable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum RecipeSegmentationStrategy {
    /// Generic DETR-style query-mask composition.
    #[default]
    Standard,
    /// MaskFormer postprocessing without a fixed intermediate mask canvas.
    MaskFormer,
    /// Mask2Former postprocessing through its fixed `384 x 384` mask canvas.
    Mask2Former,
    /// OneFormer semantic and panoptic postprocessing.
    ///
    /// OneFormer instance postprocessing additionally requires dataset class
    /// metadata and is rejected by the generic recipe executor.
    OneFormer,
    /// EOMT postprocessing through its configured intermediate mask canvas.
    Eomt {
        /// Intermediate `(shortest_edge, longest_edge)` mask canvas.
        size: ImageSize,
    },
}

/// Threshold and target-size options for segmentation postprocessing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecipeSegmentationPostprocessOptions {
    /// Minimum class score required for retained instance or panoptic queries.
    pub score_threshold: f32,
    /// Mask threshold used for retained segment support.
    pub mask_threshold: f32,
    /// Minimum assigned-area to original-area ratio required to keep a segment.
    pub overlap_mask_area_threshold: f32,
    /// Optional source for output image sizes.
    pub target_size_source: Option<RecipeImageSizeSource>,
}

impl Default for RecipeSegmentationPostprocessOptions {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            mask_threshold: 0.5,
            overlap_mask_area_threshold: 0.8,
            target_size_source: None,
        }
    }
}

/// Segmentation postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecipeSegmentationPostprocess {
    /// Segmentation task to restore.
    pub task: RecipeSegmentationTask,
    /// Model output name containing class logits.
    pub class_logits_output: String,
    /// Model output name containing mask logits.
    pub mask_logits_output: String,
    /// Number of queries, when the executor should not infer it from shapes.
    pub num_queries: Option<usize>,
    /// Number of labels in the logits, including the final background label.
    ///
    /// When absent, the executor infers this model-head property from the
    /// class-logit tensor rather than the image processor configuration.
    pub num_labels_with_background: Option<usize>,
    /// Mask-logit spatial size, when the executor should not infer it.
    pub mask_size: Option<ImageSize>,
    /// Segmentation thresholds and target-size resolution policy.
    #[serde(default)]
    pub options: RecipeSegmentationPostprocessOptions,
    /// Label ids that should share one panoptic segment id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub label_ids_to_fuse: Vec<i64>,
    /// Family-specific query-mask execution strategy.
    #[serde(default)]
    pub strategy: RecipeSegmentationStrategy,
}

impl RecipeSegmentationPostprocess {
    /// Creates a segmentation postprocessing descriptor.
    pub fn new(
        task: RecipeSegmentationTask,
        class_logits_output: impl Into<String>,
        mask_logits_output: impl Into<String>,
        num_labels_with_background: Option<usize>,
    ) -> Self {
        Self {
            task,
            class_logits_output: class_logits_output.into(),
            mask_logits_output: mask_logits_output.into(),
            num_queries: None,
            num_labels_with_background,
            mask_size: None,
            options: RecipeSegmentationPostprocessOptions::default(),
            label_ids_to_fuse: Vec::new(),
            strategy: RecipeSegmentationStrategy::default(),
        }
    }

    /// Returns this descriptor with a family-specific execution strategy.
    pub fn with_strategy(mut self, strategy: RecipeSegmentationStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

/// Binary mask postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecipeBinaryMaskPostprocess {
    /// Model output name containing low-resolution mask logits.
    pub mask_logits_output: String,
    /// Mask-logit spatial size, when the executor should not infer it.
    pub mask_size: Option<ImageSize>,
    /// Number of masks per image, when the executor should not infer it.
    pub masks_per_image: Option<usize>,
    /// Source for original image sizes.
    pub original_size_source: RecipeImageSizeSource,
    /// Source for resized input sizes before padded-canvas restoration.
    pub reshaped_input_size_source: RecipeImageSizeSource,
    /// Threshold used to binarize restored mask logits.
    pub mask_threshold: f32,
}

impl RecipeBinaryMaskPostprocess {
    /// Creates a binary mask postprocessing descriptor.
    pub fn new(
        mask_logits_output: impl Into<String>,
        original_size_source: RecipeImageSizeSource,
        reshaped_input_size_source: RecipeImageSizeSource,
        mask_threshold: f32,
    ) -> Self {
        Self {
            mask_logits_output: mask_logits_output.into(),
            mask_size: None,
            masks_per_image: None,
            original_size_source,
            reshaped_input_size_source,
            mask_threshold,
        }
    }
}

/// Unit semantics for depth-like postprocessing outputs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeDepthUnit {
    /// Metric depth values in meters.
    Meters,
    /// Relative depth values without an absolute metric scale.
    Relative,
    /// Disparity values.
    Disparity,
    /// Inverse-depth values.
    InverseDepth,
    /// Unit-normalized depth values.
    Normalized,
    /// Unit semantics are processor-specific or not yet modeled.
    Unknown,
}

/// Depth-like output postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeDepthPostprocess {
    /// Model output name containing depth-like values.
    pub depth_output: String,
    /// Unit semantics for the restored output.
    pub unit: RecipeDepthUnit,
    /// Optional source for output image sizes.
    pub target_size_source: Option<RecipeImageSizeSource>,
    /// Whether the output should be resized back to target image coordinates.
    pub resize_back: bool,
}

impl RecipeDepthPostprocess {
    /// Creates a depth-like postprocessing descriptor.
    pub fn new(
        depth_output: impl Into<String>,
        unit: RecipeDepthUnit,
        target_size_source: Option<RecipeImageSizeSource>,
        resize_back: bool,
    ) -> Self {
        Self {
            depth_output: depth_output.into(),
            unit,
            target_size_source,
            resize_back,
        }
    }
}

/// Dense image-like postprocessing task kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeDenseMapTask {
    /// Intrinsic image decomposition target maps, such as albedo or shading.
    IntrinsicImage,
    /// Surface-normal vectors represented as per-pixel channels.
    SurfaceNormal,
    /// Uncertainty values represented as a dense map.
    Uncertainty,
    /// Dense map semantics are processor-specific or not yet modeled.
    Other,
}

/// Dense image-like output postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeDenseMapPostprocess {
    /// Dense-map task semantics to attach to the restored output.
    pub task: RecipeDenseMapTask,
    /// Model output name containing dense image-like values.
    pub output: String,
    /// Expected channel count, when the executor should validate it.
    pub channels: Option<usize>,
    /// Optional target names when one tensor stores multiple dense targets per image.
    ///
    /// Outputs are assigned names in row-major batch order with
    /// `sample_index % target_names.len()`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_names: Vec<String>,
    /// Optional source for output image sizes.
    pub target_size_source: Option<RecipeImageSizeSource>,
    /// Whether the output should be resized back to target image coordinates.
    pub resize_back: bool,
}

impl RecipeDenseMapPostprocess {
    /// Creates a dense-map postprocessing descriptor.
    pub fn new(
        task: RecipeDenseMapTask,
        output: impl Into<String>,
        target_size_source: Option<RecipeImageSizeSource>,
        resize_back: bool,
    ) -> Self {
        Self {
            task,
            output: output.into(),
            channels: None,
            target_names: Vec::new(),
            target_size_source,
            resize_back,
        }
    }

    /// Returns this descriptor with an expected channel count.
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Returns this descriptor with ordered dense-map target names.
    pub fn with_target_names(mut self, target_names: Vec<String>) -> Self {
        self.target_names = target_names;
        self
    }
}

/// Video tensor postprocessing transfer function.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeVideoTransferFunction {
    /// Restore Diffusers/LTX-style LogC3 code values to linear HDR values.
    LogC3,
}

/// Video tensor postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeVideoTensorPostprocess {
    /// Model output name containing the decoded video tensor.
    pub output: String,
    /// Transfer function applied to the decoded tensor values.
    pub transfer: RecipeVideoTransferFunction,
}

impl RecipeVideoTensorPostprocess {
    /// Creates a video tensor postprocessing descriptor.
    pub fn new(output: impl Into<String>, transfer: RecipeVideoTransferFunction) -> Self {
        Self {
            output: output.into(),
            transfer,
        }
    }
}

/// Coordinate postprocessing task kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeCoordinateTask {
    /// Image keypoints.
    Keypoints,
    /// Point matches between image pairs.
    Matches,
    /// Pose landmarks or skeleton points.
    Pose,
}

/// Coordinate postprocessing recipe parameters.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeCoordinatePostprocess {
    /// Coordinate task to restore.
    pub task: RecipeCoordinateTask,
    /// Model output name containing coordinates.
    pub coordinates_output: String,
    /// Optional model output name containing coordinate confidence scores.
    pub scores_output: Option<String>,
    /// Source for target image sizes used to scale coordinates.
    pub target_size_source: RecipeImageSizeSource,
}

impl RecipeCoordinatePostprocess {
    /// Creates a coordinate postprocessing descriptor.
    pub fn new(
        task: RecipeCoordinateTask,
        coordinates_output: impl Into<String>,
        scores_output: Option<String>,
        target_size_source: RecipeImageSizeSource,
    ) -> Self {
        Self {
            task,
            coordinates_output: coordinates_output.into(),
            scores_output,
            target_size_source,
        }
    }
}

/// Structured token-sequence task kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeTokenSequenceTask {
    /// OCR recognition output.
    Ocr,
    /// Table structure output.
    TableStructure,
    /// Document text or markup output.
    DocumentText,
}

/// Greedy token-id decoding applied to structured sequence logits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum RecipeTokenSequenceDecoder {
    /// Collapse repeated ids and remove the configured CTC blank id.
    CtcGreedy {
        /// Vocabulary id used for the CTC blank symbol.
        blank_token_id: usize,
    },
    /// Select one id per step, omitting begin tokens and stopping at an end
    /// token after the initial decoding position.
    Greedy {
        /// Optional begin-of-sequence id to omit from results.
        begin_token_id: Option<usize>,
        /// Optional end-of-sequence id. At the initial position it is omitted;
        /// at later positions it terminates decoding.
        end_token_id: Option<usize>,
    },
}

/// Source required to turn decoded token ids into text or schema tokens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeTokenVocabularySource {
    /// The caller owns the tokenizer or task vocabulary outside this image crate.
    #[default]
    External,
}

/// Recipe parameters for tokenizer-independent structured token-id decoding.
///
/// This descriptor deliberately stops at token ids. Mapping those ids to text,
/// HTML, or model-specific schemas requires the external tokenizer or
/// vocabulary selected with the model and is outside image preprocessing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeTokenSequencePostprocess {
    /// Structured task semantics attached to the decoded sequence.
    pub task: RecipeTokenSequenceTask,
    /// Model output name containing `[batch, steps, vocabulary]` logits.
    pub logits_output: String,
    /// Greedy token-id decoding policy.
    pub decoder: RecipeTokenSequenceDecoder,
    /// Vocabulary ownership boundary for text or schema decoding.
    #[serde(default)]
    pub vocabulary_source: RecipeTokenVocabularySource,
}

impl RecipeTokenSequencePostprocess {
    /// Creates a tokenizer-independent token-sequence descriptor.
    pub fn new(
        task: RecipeTokenSequenceTask,
        logits_output: impl Into<String>,
        decoder: RecipeTokenSequenceDecoder,
    ) -> Self {
        Self {
            task,
            logits_output: logits_output.into(),
            decoder,
            vocabulary_source: RecipeTokenVocabularySource::External,
        }
    }
}

/// Structured output-hook task kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeOutputHookTask {
    /// OCR text, token, or recognition outputs.
    Ocr,
    /// Table structure outputs.
    TableStructure,
    /// Page or document layout geometry outputs.
    Layout,
}

/// Recipe parameters for capturing a structured model-output hook.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeOutputHookPostprocess {
    /// Structured output task label.
    pub task: RecipeOutputHookTask,
    /// Model output name containing task-specific data.
    pub output: String,
    /// Optional source for target image sizes used by geometry outputs.
    pub target_size_source: Option<RecipeImageSizeSource>,
}

impl RecipeOutputHookPostprocess {
    /// Creates a structured output-hook postprocessing descriptor.
    pub fn new(
        task: RecipeOutputHookTask,
        output: impl Into<String>,
        target_size_source: Option<RecipeImageSizeSource>,
    ) -> Self {
        Self {
            task,
            output: output.into(),
            target_size_source,
        }
    }
}
