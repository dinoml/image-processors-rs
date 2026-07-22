//! Typed processor outputs for model-family preprocessing.

use crate::tensor::Tensor;
use crate::transforms::{
    DetectionAnnotation, ImageSize, NestedImageGridMetadata, NormalizedDetectionAnnotation,
};

/// Standard tensor output names used by model processors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessorTensorName {
    /// Main image or video tensor consumed by a model.
    PixelValues,
    /// Pixel-validity mask used by processors that pad variable-sized inputs.
    PixelMask,
    /// Processor-specific tensor output.
    Other(String),
}

impl ProcessorTensorName {
    /// Creates a processor-specific tensor name.
    pub fn other(name: impl Into<String>) -> Self {
        Self::Other(name.into())
    }

    /// Returns the conventional string name for this tensor output.
    pub fn as_str(&self) -> &str {
        match self {
            Self::PixelValues => "pixel_values",
            Self::PixelMask => "pixel_mask",
            Self::Other(name) => name,
        }
    }
}

/// Standard metadata output names used by model processors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessorMetadataName {
    /// Original decoded image sizes before processor transforms.
    OriginalSizes,
    /// Image sizes after resize and before final padding or tensor conversion.
    ReshapedInputSizes,
    /// Qwen/VLM-style temporal-height-width grid metadata.
    ImageGridThw,
    /// Per-image count of patch frames before batch padding.
    ImagePatchCounts,
    /// Nested image-grid sizes, target mask, and target positions.
    NestedImageGrid,
    /// Detection or segmentation labels.
    Labels,
    /// Processor-specific metadata output.
    Other(String),
}

impl ProcessorMetadataName {
    /// Creates a processor-specific metadata name.
    pub fn other(name: impl Into<String>) -> Self {
        Self::Other(name.into())
    }

    /// Returns the conventional string name for this metadata output.
    pub fn as_str(&self) -> &str {
        match self {
            Self::OriginalSizes => "original_sizes",
            Self::ReshapedInputSizes => "reshaped_input_sizes",
            Self::ImageGridThw => "image_grid_thw",
            Self::ImagePatchCounts => "image_patch_counts",
            Self::NestedImageGrid => "nested_image_grid",
            Self::Labels => "labels",
            Self::Other(name) => name,
        }
    }
}

/// Typed metadata values returned by processor families.
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessorMetadataValue {
    /// Per-image height-width sizes.
    ImageSizes(Vec<ImageSize>),
    /// Per-image temporal-height-width grid entries.
    GridThw(Vec<[usize; 3]>),
    /// Per-image count-like metadata.
    Counts(Vec<usize>),
    /// Nested count-like metadata keyed by batch sample and original image.
    NestedCounts(Vec<Vec<usize>>),
    /// Nested boolean mask metadata keyed by batch sample, image, and tile.
    NestedBoolMask(Vec<Vec<Vec<bool>>>),
    /// Nested image-grid sizes, target mask, and target positions.
    NestedImageGrid(NestedImageGridMetadata),
    /// Per-image detection labels with absolute corner-format boxes.
    DetectionLabels(Vec<DetectionAnnotation>),
    /// Per-image detection labels with normalized center-format boxes.
    NormalizedDetectionLabels(Vec<NormalizedDetectionAnnotation>),
    /// Processor-specific shape-like metadata.
    Shape(Vec<usize>),
}

/// One named tensor output.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessorTensorOutput {
    name: ProcessorTensorName,
    tensor: Tensor,
}

impl ProcessorTensorOutput {
    /// Creates a named tensor output.
    pub fn new(name: ProcessorTensorName, tensor: Tensor) -> Self {
        Self { name, tensor }
    }

    /// Returns the tensor output name.
    pub fn name(&self) -> &ProcessorTensorName {
        &self.name
    }

    /// Returns the tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Consumes this output and returns the tensor.
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

/// One named metadata output.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessorMetadataOutput {
    name: ProcessorMetadataName,
    value: ProcessorMetadataValue,
}

impl ProcessorMetadataOutput {
    /// Creates a named metadata output.
    pub fn new(name: ProcessorMetadataName, value: ProcessorMetadataValue) -> Self {
        Self { name, value }
    }

    /// Returns the metadata output name.
    pub fn name(&self) -> &ProcessorMetadataName {
        &self.name
    }

    /// Returns the metadata value.
    pub fn value(&self) -> &ProcessorMetadataValue {
        &self.value
    }
}

/// Typed output bundle returned by processor-family APIs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProcessorOutput {
    tensors: Vec<ProcessorTensorOutput>,
    metadata: Vec<ProcessorMetadataOutput>,
}

impl ProcessorOutput {
    /// Creates an empty output bundle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an output bundle containing only `pixel_values`.
    pub fn from_pixel_values(pixel_values: Tensor) -> Self {
        let mut output = Self::new();
        output.insert_tensor(ProcessorTensorName::PixelValues, pixel_values);
        output
    }

    /// Returns all named tensor outputs in insertion order.
    pub fn tensors(&self) -> &[ProcessorTensorOutput] {
        &self.tensors
    }

    /// Returns all named metadata outputs in insertion order.
    pub fn metadata(&self) -> &[ProcessorMetadataOutput] {
        &self.metadata
    }

    /// Inserts or replaces a named tensor output.
    pub fn insert_tensor(&mut self, name: ProcessorTensorName, tensor: Tensor) -> Option<Tensor> {
        if let Some(existing) = self.tensors.iter_mut().find(|output| output.name == name) {
            return Some(std::mem::replace(&mut existing.tensor, tensor));
        }

        self.tensors.push(ProcessorTensorOutput::new(name, tensor));
        None
    }

    /// Inserts or replaces a named metadata output.
    pub fn insert_metadata(
        &mut self,
        name: ProcessorMetadataName,
        value: ProcessorMetadataValue,
    ) -> Option<ProcessorMetadataValue> {
        if let Some(existing) = self.metadata.iter_mut().find(|output| output.name == name) {
            return Some(std::mem::replace(&mut existing.value, value));
        }

        self.metadata
            .push(ProcessorMetadataOutput::new(name, value));
        None
    }

    /// Returns a named tensor output.
    pub fn tensor(&self, name: &ProcessorTensorName) -> Option<&Tensor> {
        self.tensors
            .iter()
            .find(|output| output.name == *name)
            .map(ProcessorTensorOutput::tensor)
    }

    /// Returns the `pixel_values` tensor.
    pub fn pixel_values(&self) -> Option<&Tensor> {
        self.tensor(&ProcessorTensorName::PixelValues)
    }

    /// Returns the `pixel_mask` tensor.
    pub fn pixel_mask(&self) -> Option<&Tensor> {
        self.tensor(&ProcessorTensorName::PixelMask)
    }

    /// Returns a named metadata output.
    pub fn metadata_value(&self, name: &ProcessorMetadataName) -> Option<&ProcessorMetadataValue> {
        self.metadata
            .iter()
            .find(|output| output.name == *name)
            .map(ProcessorMetadataOutput::value)
    }

    /// Returns `original_sizes` metadata when present.
    pub fn original_sizes(&self) -> Option<&[ImageSize]> {
        match self.metadata_value(&ProcessorMetadataName::OriginalSizes)? {
            ProcessorMetadataValue::ImageSizes(sizes) => Some(sizes),
            _ => None,
        }
    }

    /// Returns `reshaped_input_sizes` metadata when present.
    pub fn reshaped_input_sizes(&self) -> Option<&[ImageSize]> {
        match self.metadata_value(&ProcessorMetadataName::ReshapedInputSizes)? {
            ProcessorMetadataValue::ImageSizes(sizes) => Some(sizes),
            _ => None,
        }
    }

    /// Returns `image_grid_thw` metadata when present.
    pub fn image_grid_thw(&self) -> Option<&[[usize; 3]]> {
        match self.metadata_value(&ProcessorMetadataName::ImageGridThw)? {
            ProcessorMetadataValue::GridThw(grid) => Some(grid),
            _ => None,
        }
    }

    /// Returns `image_patch_counts` metadata when present.
    pub fn image_patch_counts(&self) -> Option<&[usize]> {
        match self.metadata_value(&ProcessorMetadataName::ImagePatchCounts)? {
            ProcessorMetadataValue::Counts(counts) => Some(counts),
            _ => None,
        }
    }

    /// Returns processor-specific nested count metadata when present.
    pub fn nested_counts(&self, name: &ProcessorMetadataName) -> Option<&[Vec<usize>]> {
        match self.metadata_value(name)? {
            ProcessorMetadataValue::NestedCounts(counts) => Some(counts),
            _ => None,
        }
    }

    /// Returns processor-specific nested boolean-mask metadata when present.
    pub fn nested_bool_mask(&self, name: &ProcessorMetadataName) -> Option<&[Vec<Vec<bool>>]> {
        match self.metadata_value(name)? {
            ProcessorMetadataValue::NestedBoolMask(mask) => Some(mask),
            _ => None,
        }
    }

    /// Returns `nested_image_grid` metadata when present.
    pub fn nested_image_grid(&self) -> Option<&NestedImageGridMetadata> {
        match self.metadata_value(&ProcessorMetadataName::NestedImageGrid)? {
            ProcessorMetadataValue::NestedImageGrid(metadata) => Some(metadata),
            _ => None,
        }
    }

    /// Returns `labels` metadata with absolute corner-format boxes when present.
    pub fn detection_labels(&self) -> Option<&[DetectionAnnotation]> {
        match self.metadata_value(&ProcessorMetadataName::Labels)? {
            ProcessorMetadataValue::DetectionLabels(labels) => Some(labels),
            _ => None,
        }
    }

    /// Returns `labels` metadata with normalized center-format boxes when present.
    pub fn normalized_detection_labels(&self) -> Option<&[NormalizedDetectionAnnotation]> {
        match self.metadata_value(&ProcessorMetadataName::Labels)? {
            ProcessorMetadataValue::NormalizedDetectionLabels(labels) => Some(labels),
            _ => None,
        }
    }

    /// Consumes this output and returns `pixel_values` when present.
    pub fn into_pixel_values(mut self) -> Option<Tensor> {
        let index = self
            .tensors
            .iter()
            .position(|output| output.name == ProcessorTensorName::PixelValues)?;
        Some(self.tensors.swap_remove(index).into_tensor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Layout, TensorData};
    use crate::transforms::{nested_image_grid_metadata, NestedImageGridTarget};

    #[test]
    fn processor_output_replaces_named_tensor() {
        let first = Tensor::new(TensorData::F32(vec![1.0]), vec![1, 1, 1], Layout::CHW).unwrap();
        let second = Tensor::new(TensorData::F32(vec![2.0]), vec![1, 1, 1], Layout::CHW).unwrap();
        let mut output = ProcessorOutput::from_pixel_values(first);

        let previous = output.insert_tensor(ProcessorTensorName::PixelValues, second);

        assert!(previous.is_some());
        assert_eq!(
            output.pixel_values().unwrap().data().to_vec::<f32>(),
            vec![2.0]
        );
    }

    #[test]
    fn processor_output_exposes_typed_metadata() {
        let mut output = ProcessorOutput::new();
        let sizes = vec![ImageSize {
            height: 10,
            width: 20,
        }];

        output.insert_metadata(
            ProcessorMetadataName::OriginalSizes,
            ProcessorMetadataValue::ImageSizes(sizes.clone()),
        );
        output.insert_metadata(
            ProcessorMetadataName::ImagePatchCounts,
            ProcessorMetadataValue::Counts(vec![3]),
        );

        assert_eq!(output.original_sizes(), Some(sizes.as_slice()));
        assert_eq!(output.image_patch_counts(), Some([3].as_slice()));
        assert_eq!(
            ProcessorMetadataName::ImagePatchCounts.as_str(),
            "image_patch_counts"
        );
    }

    #[test]
    fn processor_output_exposes_detection_labels_metadata() {
        let mut output = ProcessorOutput::new();
        let labels = vec![DetectionAnnotation {
            image_id: 7,
            original_size: ImageSize::new(10, 20).unwrap(),
            size: ImageSize::new(10, 20).unwrap(),
            class_labels: Vec::new(),
            boxes: Vec::new(),
            area: Vec::new(),
            iscrowd: Vec::new(),
        }];

        output.insert_metadata(
            ProcessorMetadataName::Labels,
            ProcessorMetadataValue::DetectionLabels(labels.clone()),
        );

        assert_eq!(ProcessorMetadataName::Labels.as_str(), "labels");
        assert_eq!(output.detection_labels(), Some(labels.as_slice()));
    }

    #[test]
    fn processor_output_exposes_nested_image_grid_metadata() {
        let mut output = ProcessorOutput::new();
        let metadata = nested_image_grid_metadata(
            vec![
                vec![
                    ImageSize::new(32, 32).unwrap(),
                    ImageSize::new(32, 32).unwrap(),
                ],
                vec![
                    ImageSize::new(32, 32).unwrap(),
                    ImageSize::new(32, 32).unwrap(),
                ],
            ],
            vec![vec![false, false], vec![false, true]],
        )
        .unwrap();

        output.insert_metadata(
            ProcessorMetadataName::NestedImageGrid,
            ProcessorMetadataValue::NestedImageGrid(metadata.clone()),
        );

        assert_eq!(
            output
                .nested_image_grid()
                .unwrap()
                .target_positions
                .as_slice(),
            &[NestedImageGridTarget::new(1, 1)]
        );
        assert_eq!(
            output
                .metadata_value(&ProcessorMetadataName::NestedImageGrid)
                .unwrap(),
            &ProcessorMetadataValue::NestedImageGrid(metadata)
        );
        assert_eq!(
            ProcessorMetadataName::NestedImageGrid.as_str(),
            "nested_image_grid"
        );
    }
}
