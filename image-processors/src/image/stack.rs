use super::*;

pub(super) fn stack_tensors(
    tensors: Vec<Tensor>,
    leading_axis: TensorLeadingAxis,
) -> Result<Tensor, ImageProcessorError> {
    let shape = tensors[0].shape().to_vec();
    let layout = tensors[0].layout();
    let mut data = Vec::with_capacity(tensors.iter().map(|tensor| tensor.data().len()).sum());

    for tensor in tensors {
        if tensor.layout() != layout || tensor.shape()[1..] != shape[1..] {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        match tensor.data() {
            TensorData::F32(values) => data.extend_from_slice(values),
            _ => return Err(ImageProcessorError::ExpectedDataType("F32")),
        }
    }

    let mut batch_shape = shape;
    batch_shape[0] = data.len() / batch_shape[1..].iter().product::<usize>();
    Tensor::new(TensorData::F32(data), batch_shape, layout)
        .and_then(|tensor| tensor.with_leading_axis(leading_axis))
        .map_err(ImageProcessorError::Tensor)
}

/// Stacks frame-leading clip tensors into one batched video tensor.
pub(crate) fn stack_frame_tensors_as_video_batch(
    tensors: Vec<Tensor>,
) -> Result<Tensor, ImageProcessorError> {
    if tensors.is_empty() {
        return Err(ImageProcessorError::EmptyBatch);
    }

    let shape = tensors[0].shape().to_vec();
    let layout = tensors[0].layout();
    if tensors[0].leading_axis() != Some(TensorLeadingAxis::Frames) {
        return Err(ImageProcessorError::UnsupportedLayout(layout));
    }

    let batch_layout = match layout {
        Layout::NCHW => Layout::BFCHW,
        Layout::NHWC => Layout::BFHWC,
        Layout::NC
        | Layout::CHW
        | Layout::HWC
        | Layout::NPCHW
        | Layout::NPHWC
        | Layout::BFCHW
        | Layout::BFHWC
        | Layout::NIPCHW
        | Layout::NIPHWC => return Err(ImageProcessorError::UnsupportedLayout(layout)),
    };

    let batch = tensors.len();
    let mut data = Vec::with_capacity(tensors.iter().map(|tensor| tensor.data().len()).sum());
    for tensor in tensors {
        if tensor.layout() != layout
            || tensor.leading_axis() != Some(TensorLeadingAxis::Frames)
            || tensor.shape() != shape
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        match tensor.data() {
            TensorData::F32(values) => data.extend_from_slice(values),
            _ => return Err(ImageProcessorError::ExpectedDataType("F32")),
        }
    }

    let batch_shape = vec![batch, shape[0], shape[1], shape[2], shape[3]];
    Tensor::new(TensorData::F32(data), batch_shape, batch_layout)
        .and_then(|tensor| tensor.with_leading_axis(TensorLeadingAxis::Batch))
        .map_err(ImageProcessorError::Tensor)
}
