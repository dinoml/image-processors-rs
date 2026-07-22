use super::*;
use thiserror::Error;

/// Additional inputs used when executing recipe postprocessing descriptors.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipePostprocessContext<'a> {
    target_sizes: Option<&'a [ImageSize]>,
}

impl<'a> RecipePostprocessContext<'a> {
    /// Creates an empty recipe postprocess execution context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the caller-provided target sizes, when present.
    pub fn target_sizes(&self) -> Option<&'a [ImageSize]> {
        self.target_sizes
    }

    /// Sets caller-provided target sizes for descriptors that request them.
    pub fn with_target_sizes(mut self, target_sizes: &'a [ImageSize]) -> Self {
        self.target_sizes = Some(target_sizes);
        self
    }
}

/// Errors returned by tensor-to-image-frame postprocessing helpers.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum TensorPostprocessError {
    /// The tensor layout cannot be restored by the requested helper.
    #[error("unsupported tensor postprocess layout {0:?}")]
    UnsupportedLayout(Layout),
    /// Output tensor construction failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Tensor values or frame construction violated image transform invariants.
    #[error(transparent)]
    Transform(#[from] TransformError),
}

/// Binary mask predictions restored to one original image size.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryMaskPostprocessOutput {
    size: ImageSize,
    masks: Vec<Vec<bool>>,
}

impl BinaryMaskPostprocessOutput {
    pub(crate) fn new(size: ImageSize, masks: Vec<Vec<bool>>) -> Self {
        Self { size, masks }
    }

    /// Returns the original image size these masks were restored to.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Returns row-major binary mask planes.
    pub fn masks(&self) -> &[Vec<bool>] {
        &self.masks
    }

    /// Consumes this output into original size and mask planes.
    pub fn into_parts(self) -> (ImageSize, Vec<Vec<bool>>) {
        (self.size, self.masks)
    }
}

/// One post-processed depth-like map.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMapPostprocessOutput {
    size: ImageSize,
    unit: RecipeDepthUnit,
    values: Vec<f32>,
}

impl DepthMapPostprocessOutput {
    pub(super) fn new(size: ImageSize, unit: RecipeDepthUnit, values: Vec<f32>) -> Self {
        Self { size, unit, values }
    }

    /// Returns the spatial dimensions of `values`.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Returns the descriptor-provided depth unit semantics.
    pub fn unit(&self) -> RecipeDepthUnit {
        self.unit
    }

    /// Returns row-major depth-like values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consumes this output into size, unit semantics, and row-major values.
    pub fn into_parts(self) -> (ImageSize, RecipeDepthUnit, Vec<f32>) {
        (self.size, self.unit, self.values)
    }
}

/// One post-processed dense image-like map.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMapPostprocessOutput {
    task: RecipeDenseMapTask,
    target_name: Option<String>,
    size: ImageSize,
    channels: usize,
    values: Vec<f32>,
}

impl DenseMapPostprocessOutput {
    pub(super) fn new(
        task: RecipeDenseMapTask,
        target_name: Option<String>,
        size: ImageSize,
        channels: usize,
        values: Vec<f32>,
    ) -> Self {
        Self {
            task,
            target_name,
            size,
            channels,
            values,
        }
    }

    /// Returns the descriptor-provided dense-map task semantics.
    pub fn task(&self) -> RecipeDenseMapTask {
        self.task
    }

    /// Returns the dense-map target name supplied by the descriptor, if any.
    pub fn target_name(&self) -> Option<&str> {
        self.target_name.as_deref()
    }

    /// Returns the spatial dimensions of `values`.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Returns the number of channels per pixel.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns row-major pixel values with channels interleaved per pixel.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consumes this output into task, target name, size, channel count, and values.
    pub fn into_parts(
        self,
    ) -> (
        RecipeDenseMapTask,
        Option<String>,
        ImageSize,
        usize,
        Vec<f32>,
    ) {
        (
            self.task,
            self.target_name,
            self.size,
            self.channels,
            self.values,
        )
    }
}

/// One restored image-space point in `(x, y)` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinatePoint {
    /// Horizontal coordinate in pixels.
    pub x: f32,
    /// Vertical coordinate in pixels.
    pub y: f32,
}

impl CoordinatePoint {
    /// Creates a coordinate point from finite pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error when either coordinate is non-finite.
    pub fn new(x: f32, y: f32) -> Result<Self, TransformError> {
        validate_coordinate_point([x, y])?;
        Ok(Self { x, y })
    }
}

/// Coordinate predictions restored to one target image size.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinatePostprocessOutput {
    task: RecipeCoordinateTask,
    size: ImageSize,
    points: Vec<CoordinatePoint>,
    scores: Option<Vec<f32>>,
}

impl CoordinatePostprocessOutput {
    pub(super) fn new(
        task: RecipeCoordinateTask,
        size: ImageSize,
        points: Vec<CoordinatePoint>,
        scores: Option<Vec<f32>>,
    ) -> Self {
        Self {
            task,
            size,
            points,
            scores,
        }
    }

    /// Returns the coordinate task semantics attached by the recipe.
    pub fn task(&self) -> RecipeCoordinateTask {
        self.task
    }

    /// Returns the target image size used to restore the coordinates.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Returns restored points in row-major model output order.
    pub fn points(&self) -> &[CoordinatePoint] {
        &self.points
    }

    /// Returns per-point confidence scores when the descriptor supplied them.
    pub fn scores(&self) -> Option<&[f32]> {
        self.scores.as_deref()
    }

    /// Consumes this output into task, target size, points, and optional scores.
    pub fn into_parts(
        self,
    ) -> (
        RecipeCoordinateTask,
        ImageSize,
        Vec<CoordinatePoint>,
        Option<Vec<f32>>,
    ) {
        (self.task, self.size, self.points, self.scores)
    }
}

/// Structured model output attached to recipe metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputHookPostprocessOutput {
    task: RecipeOutputHookTask,
    output_name: String,
    tensor: Tensor,
    target_sizes: Option<Vec<ImageSize>>,
}

impl OutputHookPostprocessOutput {
    pub(super) fn new(
        task: RecipeOutputHookTask,
        output_name: String,
        tensor: Tensor,
        target_sizes: Option<Vec<ImageSize>>,
    ) -> Self {
        Self {
            task,
            output_name,
            tensor,
            target_sizes,
        }
    }

    /// Returns the structured task label attached by the recipe.
    pub fn task(&self) -> RecipeOutputHookTask {
        self.task
    }

    /// Returns the model-output name this hook captured.
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Returns the captured model-output tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Returns target image sizes when the descriptor requested a size source.
    pub fn target_sizes(&self) -> Option<&[ImageSize]> {
        self.target_sizes.as_deref()
    }

    /// Consumes this output into task, output name, tensor, and target sizes.
    pub fn into_parts(self) -> (RecipeOutputHookTask, String, Tensor, Option<Vec<ImageSize>>) {
        (self.task, self.output_name, self.tensor, self.target_sizes)
    }
}

/// One tokenizer-independent structured token sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenSequencePostprocessOutput {
    task: RecipeTokenSequenceTask,
    token_ids: Vec<usize>,
    scores: Vec<f32>,
    vocabulary_source: RecipeTokenVocabularySource,
}

impl TokenSequencePostprocessOutput {
    pub(super) fn new(
        task: RecipeTokenSequenceTask,
        token_ids: Vec<usize>,
        scores: Vec<f32>,
        vocabulary_source: RecipeTokenVocabularySource,
    ) -> Self {
        Self {
            task,
            token_ids,
            scores,
            vocabulary_source,
        }
    }

    /// Returns the structured task semantics attached by the recipe.
    pub fn task(&self) -> RecipeTokenSequenceTask {
        self.task
    }

    /// Returns decoded vocabulary ids in sequence order.
    ///
    /// Converting these ids to text or schema tokens requires the external
    /// tokenizer or vocabulary selected with the model.
    pub fn token_ids(&self) -> &[usize] {
        &self.token_ids
    }

    /// Returns the selected model value for each decoded token id.
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Returns the mean selected model value, or `None` for an empty sequence.
    pub fn mean_score(&self) -> Option<f32> {
        (!self.scores.is_empty())
            .then(|| self.scores.iter().sum::<f32>() / self.scores.len() as f32)
    }

    /// Returns the vocabulary ownership boundary declared by the recipe.
    pub fn vocabulary_source(&self) -> RecipeTokenVocabularySource {
        self.vocabulary_source
    }

    /// Consumes this output into task, token ids, scores, and vocabulary source.
    pub fn into_parts(
        self,
    ) -> (
        RecipeTokenSequenceTask,
        Vec<usize>,
        Vec<f32>,
        RecipeTokenVocabularySource,
    ) {
        (
            self.task,
            self.token_ids,
            self.scores,
            self.vocabulary_source,
        )
    }
}

/// Output produced by executing one recipe postprocessing descriptor.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum RecipePostprocessOutput {
    /// Object-detection predictions grouped by input image.
    ObjectDetection {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored predictions for each batch sample.
        predictions: Vec<Vec<ObjectDetectionPrediction>>,
    },
    /// Semantic segmentation predictions grouped by input image.
    SemanticSegmentation {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored semantic segmentation maps for each batch sample.
        predictions: Vec<SemanticSegmentationPrediction>,
    },
    /// Instance segmentation predictions grouped by input image.
    InstanceSegmentation {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored instance segmentation maps for each batch sample.
        predictions: Vec<SegmentationPostProcessPrediction>,
    },
    /// Panoptic segmentation predictions grouped by input image.
    PanopticSegmentation {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored panoptic segmentation maps for each batch sample.
        predictions: Vec<SegmentationPostProcessPrediction>,
    },
    /// Binary masks grouped by input image.
    BinaryMasks {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored masks for each batch sample.
        outputs: Vec<BinaryMaskPostprocessOutput>,
    },
    /// Depth-like maps grouped by input image.
    Depth {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored depth-like maps for each batch sample.
        outputs: Vec<DepthMapPostprocessOutput>,
    },
    /// Dense image-like maps grouped by input image.
    DenseMap {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored dense image-like maps for each batch sample.
        outputs: Vec<DenseMapPostprocessOutput>,
    },
    /// Video tensor restored by a common transfer function.
    VideoTensor {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Model output name that supplied the decoded video tensor.
        output_name: String,
        /// Transfer function applied to the decoded tensor.
        transfer: RecipeVideoTransferFunction,
        /// Restored video tensor.
        tensor: Tensor,
    },
    /// Coordinate predictions grouped by input image.
    Coordinates {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Restored coordinates for each batch sample.
        outputs: Vec<CoordinatePostprocessOutput>,
    },
    /// Tokenizer-independent structured token sequences grouped by batch item.
    TokenSequences {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Greedily decoded token ids and selected model values.
        outputs: Vec<TokenSequencePostprocessOutput>,
    },
    /// Structured model-output hook.
    OutputHook {
        /// Index of the descriptor in [`ProcessorRecipe::postprocess`].
        descriptor_index: usize,
        /// Captured model output and optional target-size metadata.
        output: OutputHookPostprocessOutput,
    },
}

impl RecipePostprocessOutput {
    /// Returns the recipe postprocessing descriptor index that produced this output.
    pub fn descriptor_index(&self) -> usize {
        match self {
            Self::ObjectDetection {
                descriptor_index, ..
            }
            | Self::SemanticSegmentation {
                descriptor_index, ..
            }
            | Self::InstanceSegmentation {
                descriptor_index, ..
            }
            | Self::PanopticSegmentation {
                descriptor_index, ..
            }
            | Self::BinaryMasks {
                descriptor_index, ..
            }
            | Self::Depth {
                descriptor_index, ..
            }
            | Self::DenseMap {
                descriptor_index, ..
            }
            | Self::VideoTensor {
                descriptor_index, ..
            }
            | Self::Coordinates {
                descriptor_index, ..
            }
            | Self::TokenSequences {
                descriptor_index, ..
            }
            | Self::OutputHook {
                descriptor_index, ..
            } => *descriptor_index,
        }
    }
}

/// Errors returned by recipe-driven postprocessing execution.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum RecipePostprocessError {
    /// A descriptor references a tensor output that is not present.
    #[error("model output tensor `{name}` is missing")]
    MissingTensor {
        /// Missing tensor name.
        name: String,
    },
    /// A descriptor references image-size metadata that is not present.
    #[error("image-size metadata `{name}` is missing")]
    MissingImageSizeMetadata {
        /// Missing metadata name or source.
        name: String,
    },
    /// A descriptor references metadata that is present with the wrong value type.
    #[error("metadata `{name}` is not image-size metadata")]
    InvalidImageSizeMetadata {
        /// Metadata name with an unexpected value type.
        name: String,
    },
    /// The descriptor kind is valid data but this executor does not implement it yet.
    #[error("recipe postprocess task `{task}` is not executable yet")]
    UnsupportedDescriptor {
        /// Unsupported task name.
        task: &'static str,
    },
    /// A family-specific postprocessor requires model or dataset metadata not owned by the recipe.
    #[error("{strategy} {task} requires external metadata: {metadata}")]
    ExternalMetadataRequired {
        /// Query-mask processor strategy.
        strategy: &'static str,
        /// Requested postprocessing task.
        task: &'static str,
        /// Metadata the caller must obtain with the model artifacts.
        metadata: &'static str,
    },
    /// A tensor shape did not match the descriptor's expected layout.
    #[error("tensor `{output}` expected shape {expected}, got {actual:?}")]
    InvalidTensorShape {
        /// Tensor output name.
        output: String,
        /// Expected shape description.
        expected: &'static str,
        /// Actual tensor shape.
        actual: Vec<usize>,
    },
    /// A model-output tensor contained a non-finite value.
    #[error("tensor `{output}` value at index {index} is not finite: {value}")]
    NonFiniteTensorValue {
        /// Tensor output name.
        output: String,
        /// Flat row-major index of the invalid value.
        index: usize,
        /// Invalid tensor value.
        value: f32,
    },
    /// A decoder control token id is outside the model vocabulary dimension.
    #[error(
        "token decoder field {field} id {token_id} is outside tensor `{output}` vocabulary size {vocabulary}"
    )]
    TokenIdOutsideVocabulary {
        /// Tensor output name.
        output: String,
        /// Decoder field containing the invalid id.
        field: &'static str,
        /// Invalid token id.
        token_id: usize,
        /// Vocabulary dimension from the model output tensor.
        vocabulary: usize,
    },
    /// Related model output tensors had incompatible batch or query dimensions.
    #[error("tensor shapes are incompatible: `{left}`={left_shape:?}, `{right}`={right_shape:?}")]
    IncompatibleTensorShapes {
        /// First tensor output name.
        left: String,
        /// First tensor shape.
        left_shape: Vec<usize>,
        /// Second tensor output name.
        right: String,
        /// Second tensor shape.
        right_shape: Vec<usize>,
    },
    /// Image-size metadata did not have one entry per model-output batch item.
    #[error("image-size source `{size_source}` has {actual} entries, expected {expected}")]
    IncompatibleImageSizeBatch {
        /// Image-size source name.
        size_source: String,
        /// Expected number of image-size entries.
        expected: usize,
        /// Actual number of image-size entries.
        actual: usize,
    },
    /// A model-output tensor used the wrong layout for a descriptor.
    #[error("tensor `{output}` expected layout {expected}, got {actual:?}")]
    UnsupportedTensorLayout {
        /// Tensor output name.
        output: String,
        /// Expected layout description.
        expected: &'static str,
        /// Actual tensor layout.
        actual: Layout,
    },
    /// A model-output tensor shape or conversion failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// Task-specific postprocessing failed.
    #[error(transparent)]
    Transform(#[from] TransformError),
}
