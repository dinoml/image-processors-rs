//! Tensor data, shapes, dtypes, and layout conversion.

use std::borrow::Cow;

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Tensor memory layout.
#[allow(
    clippy::upper_case_acronyms,
    reason = "tensor layouts use established axis-order abbreviations"
)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Layout {
    /// Row-major matrix with item/patch rows and feature columns.
    NC,
    /// Channel, height, width.
    CHW,
    /// Height, width, channel.
    HWC,
    /// Batch or frame, channel, height, width.
    NCHW,
    /// Batch or frame, height, width, channel.
    NHWC,
    /// Batch, patch, channel, height, width.
    NPCHW,
    /// Batch, patch, height, width, channel.
    NPHWC,
    /// Batch, frames, channel, height, width.
    BFCHW,
    /// Batch, frames, height, width, channel.
    BFHWC,
    /// Batch, image, patch, channel, height, width.
    NIPCHW,
    /// Batch, image, patch, height, width, channel.
    NIPHWC,
}

#[derive(Clone, Copy)]
struct LayoutAxes {
    rank: usize,
    channel: usize,
    height: usize,
    patch: Option<usize>,
    frame: Option<usize>,
    image: Option<usize>,
}

impl LayoutAxes {
    const fn new(
        rank: usize,
        channel: usize,
        height: usize,
        patch: Option<usize>,
        frame: Option<usize>,
        image: Option<usize>,
    ) -> Self {
        Self {
            rank,
            channel,
            height,
            patch,
            frame,
            image,
        }
    }
}

impl Layout {
    const fn axes(self) -> LayoutAxes {
        match self {
            Self::NC => LayoutAxes::new(2, 1, 0, None, None, None),
            Self::CHW => LayoutAxes::new(3, 0, 1, None, None, None),
            Self::HWC => LayoutAxes::new(3, 2, 0, None, None, None),
            Self::NCHW => LayoutAxes::new(4, 1, 2, None, None, None),
            Self::NHWC => LayoutAxes::new(4, 3, 1, None, None, None),
            Self::NPCHW => LayoutAxes::new(5, 2, 3, Some(1), None, None),
            Self::NPHWC => LayoutAxes::new(5, 4, 2, Some(1), None, None),
            Self::BFCHW => LayoutAxes::new(5, 2, 3, None, Some(1), None),
            Self::BFHWC => LayoutAxes::new(5, 4, 2, None, Some(1), None),
            Self::NIPCHW => LayoutAxes::new(6, 3, 4, Some(2), None, Some(1)),
            Self::NIPHWC => LayoutAxes::new(6, 5, 3, Some(2), None, Some(1)),
        }
    }

    const fn supports_leading_axis(self, leading_axis: TensorLeadingAxis) -> bool {
        match leading_axis {
            TensorLeadingAxis::Batch => !matches!(self, Self::NC | Self::CHW | Self::HWC),
            TensorLeadingAxis::Frames => matches!(self, Self::NCHW | Self::NHWC),
        }
    }

    fn is_semantically_compatible_with(self, target: Self) -> bool {
        let source = self.axes();
        let target = target.axes();
        source.rank == target.rank
            && source.patch == target.patch
            && source.frame == target.frame
            && source.image == target.image
    }

    /// Returns the rank required by this layout.
    pub fn rank(self) -> usize {
        self.axes().rank
    }

    /// Returns the channel axis index.
    pub fn channel_axis(self) -> usize {
        self.axes().channel
    }

    /// Returns the height axis index.
    pub fn height_axis(self) -> usize {
        self.axes().height
    }

    /// Returns the width axis index.
    pub fn width_axis(self) -> usize {
        self.height_axis() + 1
    }

    /// Returns the patch axis index for patch-batched tensors.
    pub fn patch_axis(self) -> Option<usize> {
        self.axes().patch
    }

    /// Returns the frame axis index for batched video tensors.
    pub fn frame_axis(self) -> Option<usize> {
        self.axes().frame
    }

    /// Returns the image slot axis index for nested image tensors.
    pub fn image_axis(self) -> Option<usize> {
        self.axes().image
    }
}

/// Device-agnostic axis order for image tensors.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ImageLayout {
    /// Channel, height, width.
    ChannelsHeightWidth,
    /// Height, width, channel.
    HeightWidthChannels,
}

impl ImageLayout {
    /// Returns the generic tensor layout for this image layout.
    pub fn tensor_layout(self) -> Layout {
        match self {
            Self::ChannelsHeightWidth => Layout::CHW,
            Self::HeightWidthChannels => Layout::HWC,
        }
    }

    /// Returns the tensor shape for image dimensions and channels.
    pub fn shape(self, height: usize, width: usize, channels: usize) -> [usize; 3] {
        match self {
            Self::ChannelsHeightWidth => [channels, height, width],
            Self::HeightWidthChannels => [height, width, channels],
        }
    }
}

impl From<ImageLayout> for Layout {
    fn from(layout: ImageLayout) -> Self {
        layout.tensor_layout()
    }
}

/// Device-agnostic axis order for temporal video tensors.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VideoLayout {
    /// Frames, channel, height, width.
    FramesChannelsHeightWidth,
    /// Frames, height, width, channel.
    FramesHeightWidthChannels,
}

impl VideoLayout {
    /// Returns the generic tensor layout for this video layout.
    pub fn tensor_layout(self) -> Layout {
        match self {
            Self::FramesChannelsHeightWidth => Layout::NCHW,
            Self::FramesHeightWidthChannels => Layout::NHWC,
        }
    }

    /// Returns the batched video tensor layout for this frame layout.
    pub fn batched_tensor_layout(self) -> Layout {
        match self {
            Self::FramesChannelsHeightWidth => Layout::BFCHW,
            Self::FramesHeightWidthChannels => Layout::BFHWC,
        }
    }

    /// Returns the semantic meaning of this layout's leading axis.
    pub fn leading_axis(self) -> TensorLeadingAxis {
        TensorLeadingAxis::Frames
    }

    /// Returns the tensor shape for video dimensions and channels.
    pub fn shape(self, frames: usize, height: usize, width: usize, channels: usize) -> [usize; 4] {
        match self {
            Self::FramesChannelsHeightWidth => [frames, channels, height, width],
            Self::FramesHeightWidthChannels => [frames, height, width, channels],
        }
    }

    /// Returns the tensor shape for batched video dimensions and channels.
    pub fn batched_shape(
        self,
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
    ) -> [usize; 5] {
        match self {
            Self::FramesChannelsHeightWidth => [batch, frames, channels, height, width],
            Self::FramesHeightWidthChannels => [batch, frames, height, width, channels],
        }
    }
}

impl From<VideoLayout> for Layout {
    fn from(layout: VideoLayout) -> Self {
        layout.tensor_layout()
    }
}

/// Meaning of the first axis in tensors with a leading sample dimension.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TensorLeadingAxis {
    /// The first axis is a batch dimension.
    Batch,
    /// The first axis is a temporal frame dimension.
    Frames,
}

/// Tensor element dtype.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point.
    F16,
    /// 16-bit bfloat.
    BF16,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Affine-quantized unsigned 8-bit integer.
    QuantizedU8,
    /// Affine-quantized signed 8-bit integer.
    QuantizedI8,
    /// Packed unsigned 4-bit integer.
    PackedU4,
    /// Packed signed 4-bit integer.
    PackedI4,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Boolean storage represented as canonicalized `u8` values.
    Bool,
}

/// Affine quantization parameters for integer tensor storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizationParams {
    scale: f32,
    zero_point: i32,
}

impl QuantizationParams {
    /// Creates quantization parameters with a positive finite scale.
    ///
    /// # Errors
    ///
    /// Returns an error when `scale` is not finite or is not positive.
    pub fn new(scale: f32, zero_point: i32) -> Result<Self, TensorError> {
        if !scale.is_finite() {
            return Err(TensorError::NonFiniteQuantizationScale(scale));
        }
        if scale <= 0.0 {
            return Err(TensorError::NonPositiveQuantizationScale(scale));
        }
        Ok(Self { scale, zero_point })
    }

    /// Returns the scale used to dequantize stored values.
    pub fn scale(self) -> f32 {
        self.scale
    }

    /// Returns the storage zero point.
    pub fn zero_point(self) -> i32 {
        self.zero_point
    }
}

/// Owned tensor element storage.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    /// 32-bit floating point values.
    F32(Vec<f32>),
    /// 16-bit floating point values.
    F16(Vec<f16>),
    /// 16-bit bfloat values.
    BF16(Vec<bf16>),
    /// Unsigned 8-bit integer values.
    U8(Vec<u8>),
    /// Signed 8-bit integer values.
    I8(Vec<i8>),
    /// Affine-quantized unsigned 8-bit values.
    QuantizedU8 {
        /// Raw quantized storage values.
        values: Vec<u8>,
        /// Quantization parameters used for dequantization.
        quantization: QuantizationParams,
    },
    /// Affine-quantized signed 8-bit values.
    QuantizedI8 {
        /// Raw quantized storage values.
        values: Vec<i8>,
        /// Quantization parameters used for dequantization.
        quantization: QuantizationParams,
    },
    /// Packed unsigned 4-bit values, two logical elements per byte.
    PackedU4 {
        /// Packed bytes using low nibble first, then high nibble.
        bytes: Vec<u8>,
        /// Number of logical 4-bit elements.
        elements: usize,
    },
    /// Packed signed 4-bit values, two logical elements per byte.
    PackedI4 {
        /// Packed bytes using two's-complement nibbles, low nibble first.
        bytes: Vec<u8>,
        /// Number of logical 4-bit elements.
        elements: usize,
    },
    /// Signed 32-bit integer values.
    I32(Vec<i32>),
    /// Signed 64-bit integer values.
    I64(Vec<i64>),
    /// Boolean values represented as `0` or `1` bytes.
    Bool(Vec<u8>),
}

/// Borrowed tensor element storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorDataView<'a> {
    /// 32-bit floating point values.
    F32(&'a [f32]),
    /// 16-bit floating point values.
    F16(&'a [f16]),
    /// 16-bit bfloat values.
    BF16(&'a [bf16]),
    /// Unsigned 8-bit integer values.
    U8(&'a [u8]),
    /// Signed 8-bit integer values.
    I8(&'a [i8]),
    /// Affine-quantized unsigned 8-bit values.
    QuantizedU8 {
        /// Raw quantized storage values.
        values: &'a [u8],
        /// Quantization parameters used for dequantization.
        quantization: QuantizationParams,
    },
    /// Affine-quantized signed 8-bit values.
    QuantizedI8 {
        /// Raw quantized storage values.
        values: &'a [i8],
        /// Quantization parameters used for dequantization.
        quantization: QuantizationParams,
    },
    /// Packed unsigned 4-bit values, two logical elements per byte.
    PackedU4 {
        /// Packed bytes using low nibble first, then high nibble.
        bytes: &'a [u8],
        /// Number of logical 4-bit elements.
        elements: usize,
    },
    /// Packed signed 4-bit values, two logical elements per byte.
    PackedI4 {
        /// Packed bytes using two's-complement nibbles, low nibble first.
        bytes: &'a [u8],
        /// Number of logical 4-bit elements.
        elements: usize,
    },
    /// Signed 32-bit integer values.
    I32(&'a [i32]),
    /// Signed 64-bit integer values.
    I64(&'a [i64]),
    /// Boolean values represented as `0` or `1` bytes.
    Bool(&'a [u8]),
}

impl<'a> TensorDataView<'a> {
    /// Returns the dtype for this borrowed storage.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::QuantizedU8 { .. } => DType::QuantizedU8,
            Self::QuantizedI8 { .. } => DType::QuantizedI8,
            Self::PackedU4 { .. } => DType::PackedU4,
            Self::PackedI4 { .. } => DType::PackedI4,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::BF16(values) => values.len(),
            Self::U8(values) | Self::QuantizedU8 { values, .. } | Self::Bool(values) => {
                values.len()
            }
            Self::I8(values) | Self::QuantizedI8 { values, .. } => values.len(),
            Self::PackedU4 { elements, .. } | Self::PackedI4 { elements, .. } => *elements,
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
        }
    }

    /// Returns true when no elements are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies borrowed storage into owned tensor storage.
    pub fn to_owned_data(&self) -> TensorData {
        match self {
            Self::F32(values) => TensorData::F32(values.to_vec()),
            Self::F16(values) => TensorData::F16(values.to_vec()),
            Self::BF16(values) => TensorData::BF16(values.to_vec()),
            Self::U8(values) => TensorData::U8(values.to_vec()),
            Self::I8(values) => TensorData::I8(values.to_vec()),
            Self::QuantizedU8 {
                values,
                quantization,
            } => TensorData::QuantizedU8 {
                values: values.to_vec(),
                quantization: *quantization,
            },
            Self::QuantizedI8 {
                values,
                quantization,
            } => TensorData::QuantizedI8 {
                values: values.to_vec(),
                quantization: *quantization,
            },
            Self::PackedU4 { bytes, elements } => TensorData::PackedU4 {
                bytes: bytes.to_vec(),
                elements: *elements,
            },
            Self::PackedI4 { bytes, elements } => TensorData::PackedI4 {
                bytes: bytes.to_vec(),
                elements: *elements,
            },
            Self::I32(values) => TensorData::I32(values.to_vec()),
            Self::I64(values) => TensorData::I64(values.to_vec()),
            Self::Bool(values) => {
                TensorData::Bool(values.iter().map(|value| u8::from(*value != 0)).collect())
            }
        }
    }
}

impl From<&TensorData> for Vec<f32> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F32(values) => values.clone(),
            TensorData::F16(values) => values.iter().map(|value| f32::from(*value)).collect(),
            TensorData::BF16(values) => values.iter().map(|value| f32::from(*value)).collect(),
            TensorData::U8(values) => values.iter().map(|value| *value as f32).collect(),
            TensorData::I8(values) => values.iter().map(|value| *value as f32).collect(),
            TensorData::QuantizedU8 {
                values,
                quantization,
            } => values
                .iter()
                .map(|value| dequantize(i32::from(*value), *quantization))
                .collect(),
            TensorData::QuantizedI8 {
                values,
                quantization,
            } => values
                .iter()
                .map(|value| dequantize(i32::from(*value), *quantization))
                .collect(),
            TensorData::PackedU4 { bytes, elements } => unpack_u4_values(bytes, *elements)
                .into_iter()
                .map(f32::from)
                .collect(),
            TensorData::PackedI4 { bytes, elements } => unpack_i4_values(bytes, *elements)
                .into_iter()
                .map(f32::from)
                .collect(),
            TensorData::I32(values) => values.iter().map(|value| *value as f32).collect(),
            TensorData::I64(values) => values.iter().map(|value| *value as f32).collect(),
            TensorData::Bool(values) => values
                .iter()
                .map(|value| u8::from(*value != 0) as f32)
                .collect(),
        }
    }
}

impl From<&TensorData> for Vec<f16> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F16(values) => values.clone(),
            _ => Vec::<f32>::from(data)
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        }
    }
}

impl From<&TensorData> for Vec<bf16> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::BF16(values) => values.clone(),
            _ => Vec::<f32>::from(data)
                .into_iter()
                .map(bf16::from_f32)
                .collect(),
        }
    }
}

impl From<&TensorData> for Vec<u8> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F32(values) => values
                .iter()
                .map(|value| value.round().clamp(u8::MIN as f32, u8::MAX as f32) as u8)
                .collect(),
            TensorData::F16(values) => values
                .iter()
                .map(|value| {
                    f32::from(*value)
                        .round()
                        .clamp(u8::MIN as f32, u8::MAX as f32) as u8
                })
                .collect(),
            TensorData::BF16(values) => values
                .iter()
                .map(|value| {
                    f32::from(*value)
                        .round()
                        .clamp(u8::MIN as f32, u8::MAX as f32) as u8
                })
                .collect(),
            TensorData::U8(values) | TensorData::Bool(values) => values.clone(),
            TensorData::I8(values) => values
                .iter()
                .map(|value| i32::from(*value).clamp(u8::MIN as i32, u8::MAX as i32) as u8)
                .collect(),
            TensorData::QuantizedU8 { .. } | TensorData::QuantizedI8 { .. } => {
                Vec::<f32>::from(data)
                    .into_iter()
                    .map(|value| value.round().clamp(u8::MIN as f32, u8::MAX as f32) as u8)
                    .collect()
            }
            TensorData::PackedU4 { bytes, elements } => unpack_u4_values(bytes, *elements),
            TensorData::PackedI4 { bytes, elements } => unpack_i4_values(bytes, *elements)
                .into_iter()
                .map(|value| i32::from(value).clamp(u8::MIN as i32, u8::MAX as i32) as u8)
                .collect(),
            TensorData::I32(values) => values
                .iter()
                .map(|value| (*value).clamp(u8::MIN as i32, u8::MAX as i32) as u8)
                .collect(),
            TensorData::I64(values) => values
                .iter()
                .map(|value| (*value).clamp(u8::MIN as i64, u8::MAX as i64) as u8)
                .collect(),
        }
    }
}

impl From<&TensorData> for Vec<i32> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F32(values) => values.iter().map(|value| *value as i32).collect(),
            TensorData::F16(values) => values
                .iter()
                .map(|value| f32::from(*value) as i32)
                .collect(),
            TensorData::BF16(values) => values
                .iter()
                .map(|value| f32::from(*value) as i32)
                .collect(),
            TensorData::U8(values) => values.iter().map(|value| *value as i32).collect(),
            TensorData::I8(values) => values.iter().map(|value| i32::from(*value)).collect(),
            TensorData::QuantizedU8 { .. } | TensorData::QuantizedI8 { .. } => {
                Vec::<f32>::from(data)
                    .into_iter()
                    .map(|value| value as i32)
                    .collect()
            }
            TensorData::PackedU4 { bytes, elements } => unpack_u4_values(bytes, *elements)
                .into_iter()
                .map(i32::from)
                .collect(),
            TensorData::PackedI4 { bytes, elements } => unpack_i4_values(bytes, *elements)
                .into_iter()
                .map(i32::from)
                .collect(),
            TensorData::I32(values) => values.clone(),
            TensorData::I64(values) => values
                .iter()
                .map(|value| (*value).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                .collect(),
            TensorData::Bool(values) => values.iter().map(|value| i32::from(*value != 0)).collect(),
        }
    }
}

impl From<&TensorData> for Vec<i64> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F32(values) => values.iter().map(|value| *value as i64).collect(),
            TensorData::F16(values) => values
                .iter()
                .map(|value| f32::from(*value) as i64)
                .collect(),
            TensorData::BF16(values) => values
                .iter()
                .map(|value| f32::from(*value) as i64)
                .collect(),
            TensorData::U8(values) => values.iter().map(|value| *value as i64).collect(),
            TensorData::I8(values) => values.iter().map(|value| i64::from(*value)).collect(),
            TensorData::QuantizedU8 { .. } | TensorData::QuantizedI8 { .. } => {
                Vec::<f32>::from(data)
                    .into_iter()
                    .map(|value| value as i64)
                    .collect()
            }
            TensorData::PackedU4 { bytes, elements } => unpack_u4_values(bytes, *elements)
                .into_iter()
                .map(i64::from)
                .collect(),
            TensorData::PackedI4 { bytes, elements } => unpack_i4_values(bytes, *elements)
                .into_iter()
                .map(i64::from)
                .collect(),
            TensorData::I32(values) => values.iter().map(|value| *value as i64).collect(),
            TensorData::I64(values) => values.clone(),
            TensorData::Bool(values) => values.iter().map(|value| i64::from(*value != 0)).collect(),
        }
    }
}

impl From<&TensorData> for Vec<bool> {
    fn from(data: &TensorData) -> Self {
        match data {
            TensorData::F32(values) => values.iter().map(|value| *value != 0.0).collect(),
            TensorData::F16(values) => values.iter().map(|value| *value != f16::ZERO).collect(),
            TensorData::BF16(values) => values.iter().map(|value| *value != bf16::ZERO).collect(),
            TensorData::U8(values) => values.iter().map(|value| *value != 0).collect(),
            TensorData::I8(values) => values.iter().map(|value| *value != 0).collect(),
            TensorData::QuantizedU8 { .. } | TensorData::QuantizedI8 { .. } => {
                Vec::<f32>::from(data)
                    .into_iter()
                    .map(|value| value != 0.0)
                    .collect()
            }
            TensorData::PackedU4 { bytes, elements } => unpack_u4_values(bytes, *elements)
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            TensorData::PackedI4 { bytes, elements } => unpack_i4_values(bytes, *elements)
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            TensorData::I32(values) => values.iter().map(|value| *value != 0).collect(),
            TensorData::I64(values) => values.iter().map(|value| *value != 0).collect(),
            TensorData::Bool(values) => values.iter().map(|value| *value != 0).collect(),
        }
    }
}

impl TensorData {
    /// Returns the dtype for this storage.
    pub fn dtype(&self) -> DType {
        self.as_view().dtype()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    /// Returns true when no elements are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns borrowed storage for this tensor data.
    pub fn as_view(&self) -> TensorDataView<'_> {
        match self {
            Self::F32(values) => TensorDataView::F32(values),
            Self::F16(values) => TensorDataView::F16(values),
            Self::BF16(values) => TensorDataView::BF16(values),
            Self::U8(values) => TensorDataView::U8(values),
            Self::I8(values) => TensorDataView::I8(values),
            Self::QuantizedU8 {
                values,
                quantization,
            } => TensorDataView::QuantizedU8 {
                values,
                quantization: *quantization,
            },
            Self::QuantizedI8 {
                values,
                quantization,
            } => TensorDataView::QuantizedI8 {
                values,
                quantization: *quantization,
            },
            Self::PackedU4 { bytes, elements } => TensorDataView::PackedU4 {
                bytes,
                elements: *elements,
            },
            Self::PackedI4 { bytes, elements } => TensorDataView::PackedI4 {
                bytes,
                elements: *elements,
            },
            Self::I32(values) => TensorDataView::I32(values),
            Self::I64(values) => TensorDataView::I64(values),
            Self::Bool(values) => TensorDataView::Bool(values),
        }
    }

    /// Creates packed unsigned 4-bit storage from unpacked values.
    ///
    /// Values are packed low nibble first, then high nibble. The returned
    /// storage keeps the logical element count so odd element counts round up
    /// to the next byte without changing tensor shape.
    ///
    /// # Errors
    ///
    /// Returns an error when any value is greater than `15`.
    pub fn packed_u4(values: impl IntoIterator<Item = u8>) -> Result<Self, TensorError> {
        let (bytes, elements) = pack_u4_values(values)?;
        Ok(Self::PackedU4 { bytes, elements })
    }

    /// Creates packed signed 4-bit storage from unpacked values.
    ///
    /// Values are encoded as two's-complement nibbles in the range `-8..=7`.
    ///
    /// # Errors
    ///
    /// Returns an error when any value is outside the signed 4-bit range.
    pub fn packed_i4(values: impl IntoIterator<Item = i8>) -> Result<Self, TensorError> {
        let (bytes, elements) = pack_i4_values(values)?;
        Ok(Self::PackedI4 { bytes, elements })
    }

    /// Converts tensor storage into a typed vector.
    pub fn to_vec<T>(&self) -> Vec<T>
    where
        for<'a> Vec<T>: From<&'a TensorData>,
    {
        Vec::<T>::from(self)
    }

    fn canonicalized(self) -> Self {
        match self {
            Self::Bool(values) => Self::Bool(
                values
                    .into_iter()
                    .map(|value| u8::from(value != 0))
                    .collect(),
            ),
            data => data,
        }
    }

    fn map_to<F>(&self, mut transform: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        self.map_indexed_to(|_, value| transform(value))
    }

    fn map_indexed_to<F>(&self, mut transform: F) -> Self
    where
        F: FnMut(usize, f32) -> f32,
    {
        match self {
            Self::F32(values) => {
                Self::F32(map_values_to_f32(values, &mut transform, |value| value))
            }
            Self::F16(values) => Self::F32(map_values_to_f32(values, &mut transform, f32::from)),
            Self::BF16(values) => Self::F32(map_values_to_f32(values, &mut transform, f32::from)),
            Self::U8(values) => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                value as f32
            })),
            Self::I8(values) => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                value as f32
            })),
            Self::QuantizedU8 {
                values,
                quantization,
            } => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                dequantize(i32::from(value), *quantization)
            })),
            Self::QuantizedI8 {
                values,
                quantization,
            } => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                dequantize(i32::from(value), *quantization)
            })),
            Self::PackedU4 { bytes, elements } => Self::F32(
                (0..*elements)
                    .map(|index| transform(index, f32::from(packed_u4_value(bytes, index))))
                    .collect(),
            ),
            Self::PackedI4 { bytes, elements } => Self::F32(
                (0..*elements)
                    .map(|index| transform(index, f32::from(packed_i4_value(bytes, index))))
                    .collect(),
            ),
            Self::I32(values) => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                value as f32
            })),
            Self::I64(values) => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                value as f32
            })),
            Self::Bool(values) => Self::F32(map_values_to_f32(values, &mut transform, |value| {
                u8::from(value != 0) as f32
            })),
        }
    }

    fn gather<F>(&self, output_len: usize, mut source_index: F) -> Self
    where
        F: FnMut(usize) -> usize,
    {
        match self {
            Self::F32(values) => Self::F32(gather_values(values, output_len, &mut source_index)),
            Self::F16(values) => Self::F16(gather_values(values, output_len, &mut source_index)),
            Self::BF16(values) => Self::BF16(gather_values(values, output_len, &mut source_index)),
            Self::U8(values) => Self::U8(gather_values(values, output_len, &mut source_index)),
            Self::I8(values) => Self::I8(gather_values(values, output_len, &mut source_index)),
            Self::QuantizedU8 {
                values,
                quantization,
            } => Self::QuantizedU8 {
                values: gather_values(values, output_len, &mut source_index),
                quantization: *quantization,
            },
            Self::QuantizedI8 {
                values,
                quantization,
            } => Self::QuantizedI8 {
                values: gather_values(values, output_len, &mut source_index),
                quantization: *quantization,
            },
            Self::PackedU4 { bytes, .. } => {
                let (bytes, elements) = pack_nibbles(
                    (0..output_len).map(|idx| packed_u4_value(bytes, source_index(idx))),
                );
                Self::PackedU4 { bytes, elements }
            }
            Self::PackedI4 { bytes, .. } => {
                let (bytes, elements) = pack_nibbles(
                    (0..output_len).map(|idx| encode_i4(packed_i4_value(bytes, source_index(idx)))),
                );
                Self::PackedI4 { bytes, elements }
            }
            Self::I32(values) => Self::I32(gather_values(values, output_len, &mut source_index)),
            Self::I64(values) => Self::I64(gather_values(values, output_len, &mut source_index)),
            Self::Bool(values) => Self::Bool(gather_values(values, output_len, &mut source_index)),
        }
    }
}

fn map_values_to_f32<T: Copy>(
    values: &[T],
    transform: &mut impl FnMut(usize, f32) -> f32,
    into_f32: impl Fn(T) -> f32,
) -> Vec<f32> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| transform(index, into_f32(value)))
        .collect()
}

fn gather_values<T: Copy>(
    values: &[T],
    output_len: usize,
    source_index: &mut impl FnMut(usize) -> usize,
) -> Vec<T> {
    (0..output_len)
        .map(|index| values[source_index(index)])
        .collect()
}

/// Borrowed tensor view with validated shape and layout metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorView<'a> {
    data: TensorDataView<'a>,
    shape: Cow<'a, [usize]>,
    layout: Layout,
    leading_axis: Option<TensorLeadingAxis>,
}

impl<'a> TensorView<'a> {
    /// Creates a tensor view from borrowed data, shape, and layout.
    ///
    /// # Errors
    ///
    /// Returns an error when rank, dimensions, element count, or bool values are invalid.
    pub fn new(
        data: TensorDataView<'a>,
        shape: impl Into<Vec<usize>>,
        layout: Layout,
    ) -> Result<Self, TensorError> {
        validate_data_view(data)?;
        let shape: Cow<'a, [usize]> = Cow::Owned(shape.into());
        validate_tensor_parts(data.len(), shape.as_ref(), layout)?;

        Ok(Self {
            data,
            shape,
            layout,
            leading_axis: default_leading_axis(layout),
        })
    }

    /// Creates an image tensor view from borrowed data and image dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or element count are invalid.
    pub fn image(
        data: TensorDataView<'a>,
        height: usize,
        width: usize,
        channels: usize,
        layout: ImageLayout,
    ) -> Result<Self, TensorError> {
        Self::new(
            data,
            layout.shape(height, width, channels),
            layout.tensor_layout(),
        )
    }

    /// Creates a video tensor view from borrowed data and video dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or element count are invalid.
    pub fn video(
        data: TensorDataView<'a>,
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
        layout: VideoLayout,
    ) -> Result<Self, TensorError> {
        Self::new(
            data,
            layout.shape(frames, height, width, channels),
            layout.tensor_layout(),
        )
        .and_then(|tensor| tensor.with_leading_axis(layout.leading_axis()))
    }

    /// Returns tensor element storage.
    pub fn data(&self) -> TensorDataView<'a> {
        self.data
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns tensor layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the meaning of the leading axis.
    pub fn leading_axis(&self) -> Option<TensorLeadingAxis> {
        self.leading_axis
    }

    /// Returns this view with an explicit leading-axis meaning.
    ///
    /// # Errors
    ///
    /// Returns an error when the layout has no leading batch or frame axis.
    pub fn with_leading_axis(
        mut self,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Self, TensorError> {
        validate_leading_axis(self.layout, leading_axis)?;
        self.leading_axis = Some(leading_axis);
        Ok(self)
    }

    /// Returns tensor dtype.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Returns the batch size for batch tensors.
    pub fn batch(&self) -> Option<usize> {
        tensor_batch(&self.shape, self.leading_axis)
    }

    /// Returns the frame count for temporal tensors.
    pub fn frames(&self) -> Option<usize> {
        tensor_frames(&self.shape, self.layout, self.leading_axis)
    }

    /// Returns the channel count.
    pub fn channels(&self) -> usize {
        self.shape[self.layout.channel_axis()]
    }

    /// Returns the image height.
    pub fn height(&self) -> usize {
        self.shape[self.layout.height_axis()]
    }

    /// Returns the image width.
    pub fn width(&self) -> usize {
        self.shape[self.layout.width_axis()]
    }

    /// Returns the patch count for patch-batched image tensors.
    pub fn patches(&self) -> Option<usize> {
        self.layout.patch_axis().map(|axis| self.shape[axis])
    }

    /// Copies this view into an owned tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if copied data does not satisfy tensor invariants.
    pub fn to_owned_tensor(&self) -> Result<Tensor, TensorError> {
        let tensor = Tensor::new(self.data.to_owned_data(), self.shape.to_vec(), self.layout)?;
        match self.leading_axis {
            Some(leading_axis) => tensor.with_leading_axis(leading_axis),
            None => Ok(tensor),
        }
    }
}

/// Owned tensor with validated shape and layout metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
    layout: Layout,
    leading_axis: Option<TensorLeadingAxis>,
}

impl Tensor {
    /// Creates a tensor from owned data, shape, and layout.
    ///
    /// # Errors
    ///
    /// Returns an error when rank, dimensions, or element count are invalid.
    pub fn new(
        data: TensorData,
        shape: impl Into<Vec<usize>>,
        layout: Layout,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let data = data.canonicalized();
        validate_data(&data)?;
        validate_tensor_parts(data.len(), &shape, layout)?;

        Ok(Self {
            data,
            shape,
            layout,
            leading_axis: default_leading_axis(layout),
        })
    }

    /// Returns a borrowed view over this tensor.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data.as_view(),
            shape: Cow::Borrowed(&self.shape),
            layout: self.layout,
            leading_axis: self.leading_axis,
        }
    }

    fn with_data(&self, data: TensorData) -> Self {
        Self {
            data,
            shape: self.shape.clone(),
            layout: self.layout,
            leading_axis: self.leading_axis,
        }
    }

    /// Returns tensor element storage.
    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// Returns tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns tensor layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Consumes this tensor and returns its validated storage and metadata.
    pub fn into_parts(self) -> (TensorData, Vec<usize>, Layout, Option<TensorLeadingAxis>) {
        (self.data, self.shape, self.layout, self.leading_axis)
    }

    /// Returns the meaning of the leading axis.
    pub fn leading_axis(&self) -> Option<TensorLeadingAxis> {
        self.leading_axis
    }

    /// Returns this tensor with an explicit leading-axis meaning.
    ///
    /// # Errors
    ///
    /// Returns an error when the layout has no leading batch or frame axis.
    pub fn with_leading_axis(
        mut self,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Self, TensorError> {
        validate_leading_axis(self.layout, leading_axis)?;
        self.leading_axis = Some(leading_axis);
        Ok(self)
    }

    /// Returns tensor dtype.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Returns the batch size for batch tensors.
    pub fn batch(&self) -> Option<usize> {
        tensor_batch(&self.shape, self.leading_axis)
    }

    /// Returns the frame count for temporal tensors.
    pub fn frames(&self) -> Option<usize> {
        tensor_frames(&self.shape, self.layout, self.leading_axis)
    }

    /// Returns the channel count.
    pub fn channels(&self) -> usize {
        self.shape[self.layout.channel_axis()]
    }

    /// Returns the image height.
    pub fn height(&self) -> usize {
        self.shape[self.layout.height_axis()]
    }

    /// Returns the image width.
    pub fn width(&self) -> usize {
        self.shape[self.layout.width_axis()]
    }

    /// Returns the patch count for patch-batched image tensors.
    pub fn patches(&self) -> Option<usize> {
        self.layout.patch_axis().map(|axis| self.shape[axis])
    }

    /// Multiplies every tensor value by `scale` and returns `F32` storage.
    ///
    /// # Errors
    ///
    /// Returns an error when `scale` is not finite.
    pub fn rescale(&self, scale: f32) -> Result<Self, TensorError> {
        if !scale.is_finite() {
            return Err(TensorError::NonFiniteScale(scale));
        }

        Ok(self.with_data(self.data.map_to(|value| value * scale)))
    }

    /// Applies per-channel normalization and returns `F32` storage.
    ///
    /// `mean` and `std` may each contain either one scalar value or one value
    /// per channel in this tensor's layout.
    ///
    /// # Errors
    ///
    /// Returns an error when statistics are empty, non-finite, mismatched with
    /// the tensor channel count, or when any standard deviation is not positive.
    pub fn normalize_channels(&self, mean: &[f32], std: &[f32]) -> Result<Self, TensorError> {
        let channels = self.channels();
        validate_channel_stats("mean", mean, channels)?;
        validate_channel_stats("std", std, channels)?;
        validate_positive_channel_std(std)?;

        let channel_axis = self.layout.channel_axis();
        let shape = &self.shape;
        Ok(self.with_data(self.data.map_indexed_to(|index, value| {
            let channel = axis_coordinate(index, shape, channel_axis);
            (value - channel_stat(mean, channel)) / channel_stat(std, channel)
        })))
    }

    /// Maps values from `[0, 1]` to `[-1, 1]`.
    pub fn normalize(&self) -> Self {
        self.with_data(self.data.map_to(|value| 2.0 * value - 1.0))
    }

    /// Maps values from `[-1, 1]` to `[0, 1]`.
    pub fn denormalize(&self) -> Self {
        self.with_data(
            self.data
                .map_to(|value| (value * 0.5 + 0.5).clamp(0.0, 1.0)),
        )
    }

    /// Thresholds values into `0.0` or `1.0`.
    pub fn binarize(&self) -> Self {
        self.with_data(
            self.data
                .map_to(|value| if value < 0.5 { 0.0 } else { 1.0 }),
        )
    }

    /// Converts this tensor to a different layout with the same rank.
    ///
    /// # Errors
    ///
    /// Returns an error when source and target semantic axes differ or shape arithmetic overflows.
    pub fn to_layout(&self, layout: Layout) -> Result<Self, TensorError> {
        if self.layout == layout {
            return Ok(self.clone());
        }

        if !self.layout.is_semantically_compatible_with(layout) {
            return Err(TensorError::UnsupportedLayoutConversion {
                source_layout: self.layout,
                target: layout,
            });
        }

        let source_axes_by_target_axis = channel_axis_permutation(self.layout, layout);
        let shape = source_axes_by_target_axis
            .iter()
            .map(|axis| self.shape[*axis])
            .collect::<Vec<_>>();
        let source_strides = strides(&self.shape)?;
        let target_strides = strides(&shape)?;
        let data = self.data.gather(self.data.len(), |idx| {
            source_index_for_permuted_target(
                idx,
                &shape,
                &target_strides,
                &source_strides,
                &source_axes_by_target_axis,
            )
        });

        Self::new(data, shape, layout).map(|mut tensor| {
            tensor.leading_axis = self.leading_axis;
            tensor
        })
    }
}

fn channel_axis_permutation(source: Layout, target: Layout) -> Vec<usize> {
    let rank = source.rank();
    let source_channel_axis = source.channel_axis();
    let target_channel_axis = target.channel_axis();
    let mut non_channel_axes = (0..rank).filter(|axis| *axis != source_channel_axis);

    (0..rank)
        .map(|target_axis| {
            if target_axis == target_channel_axis {
                source_channel_axis
            } else {
                non_channel_axes
                    .next()
                    .expect("rank leaves enough non-channel axes")
            }
        })
        .collect()
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for (axis, size) in shape.iter().copied().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(size)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
    }
    Ok(strides)
}

fn source_index_for_permuted_target(
    target_index: usize,
    target_shape: &[usize],
    target_strides: &[usize],
    source_strides: &[usize],
    source_axes_by_target_axis: &[usize],
) -> usize {
    source_axes_by_target_axis
        .iter()
        .enumerate()
        .map(|(target_axis, source_axis)| {
            let coordinate = target_index / target_strides[target_axis] % target_shape[target_axis];
            coordinate * source_strides[*source_axis]
        })
        .sum()
}

fn validate_channel_stats(
    field: &'static str,
    values: &[f32],
    channels: usize,
) -> Result<(), TensorError> {
    if values.is_empty() {
        return Err(TensorError::EmptyChannelStats { field });
    }
    if values.len() != 1 && values.len() != channels {
        return Err(TensorError::InvalidChannelStats {
            field,
            channels,
            actual: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteChannelStat {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_positive_channel_std(values: &[f32]) -> Result<(), TensorError> {
    for (index, value) in values.iter().copied().enumerate() {
        if value <= 0.0 {
            return Err(TensorError::NonPositiveChannelStd { index, value });
        }
    }
    Ok(())
}

fn channel_stat(values: &[f32], channel: usize) -> f32 {
    if values.len() == 1 {
        values[0]
    } else {
        values[channel]
    }
}

fn axis_coordinate(index: usize, shape: &[usize], axis: usize) -> usize {
    let stride = shape[axis + 1..].iter().product::<usize>();
    index / stride % shape[axis]
}

fn validate_data(data: &TensorData) -> Result<(), TensorError> {
    validate_data_view(data.as_view())
}

fn validate_data_view(data: TensorDataView<'_>) -> Result<(), TensorError> {
    match data {
        TensorDataView::Bool(values) => {
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !matches!(value, 0 | 1))
            {
                return Err(TensorError::InvalidBoolValue { index, value });
            }
            Ok(())
        }
        TensorDataView::QuantizedU8 { quantization, .. } => validate_quantization_zero_point(
            DType::QuantizedU8,
            quantization,
            u8::MIN as i32,
            u8::MAX as i32,
        ),
        TensorDataView::QuantizedI8 { quantization, .. } => validate_quantization_zero_point(
            DType::QuantizedI8,
            quantization,
            i8::MIN as i32,
            i8::MAX as i32,
        ),
        TensorDataView::PackedU4 { bytes, elements } => {
            validate_packed_byte_count(DType::PackedU4, bytes.len(), elements)
        }
        TensorDataView::PackedI4 { bytes, elements } => {
            validate_packed_byte_count(DType::PackedI4, bytes.len(), elements)
        }
        _ => Ok(()),
    }
}

fn validate_quantization_zero_point(
    dtype: DType,
    quantization: QuantizationParams,
    min: i32,
    max: i32,
) -> Result<(), TensorError> {
    let zero_point = quantization.zero_point();
    if (min..=max).contains(&zero_point) {
        Ok(())
    } else {
        Err(TensorError::QuantizationZeroPointOutOfRange {
            dtype,
            zero_point,
            min,
            max,
        })
    }
}

fn validate_packed_byte_count(
    dtype: DType,
    actual: usize,
    elements: usize,
) -> Result<(), TensorError> {
    let expected = packed_byte_len(elements);
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::InvalidPackedByteCount {
            dtype,
            elements,
            expected,
            actual,
        })
    }
}

fn validate_tensor_parts(
    data_len: usize,
    shape: &[usize],
    layout: Layout,
) -> Result<(), TensorError> {
    if shape.len() != layout.rank() {
        return Err(TensorError::InvalidRank {
            expected: layout.rank(),
            actual: shape.len(),
        });
    }

    if let Some((axis, size)) = shape
        .iter()
        .copied()
        .enumerate()
        .find(|(_, size)| *size == 0)
    {
        return Err(TensorError::InvalidDimension { axis, size });
    }

    let expected = checked_product(shape.iter().copied())?;
    if data_len != expected {
        return Err(TensorError::InvalidElementCount {
            expected,
            actual: data_len,
        });
    }

    Ok(())
}

fn default_leading_axis(layout: Layout) -> Option<TensorLeadingAxis> {
    layout
        .supports_leading_axis(TensorLeadingAxis::Batch)
        .then_some(TensorLeadingAxis::Batch)
}

fn tensor_batch(shape: &[usize], leading_axis: Option<TensorLeadingAxis>) -> Option<usize> {
    (leading_axis == Some(TensorLeadingAxis::Batch)).then_some(shape[0])
}

fn tensor_frames(
    shape: &[usize],
    layout: Layout,
    leading_axis: Option<TensorLeadingAxis>,
) -> Option<usize> {
    layout
        .frame_axis()
        .map(|axis| shape[axis])
        .or_else(|| (leading_axis == Some(TensorLeadingAxis::Frames)).then_some(shape[0]))
}

fn validate_leading_axis(
    layout: Layout,
    leading_axis: TensorLeadingAxis,
) -> Result<(), TensorError> {
    if layout.supports_leading_axis(leading_axis) {
        Ok(())
    } else {
        Err(TensorError::UnsupportedLeadingAxis(layout))
    }
}

fn checked_product(values: impl IntoIterator<Item = usize>) -> Result<usize, TensorError> {
    values
        .into_iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(value))
        .ok_or(TensorError::ShapeElementCountOverflow)
}

fn dequantize(value: i32, quantization: QuantizationParams) -> f32 {
    (value - quantization.zero_point()) as f32 * quantization.scale()
}

fn pack_u4_values(values: impl IntoIterator<Item = u8>) -> Result<(Vec<u8>, usize), TensorError> {
    let mut bytes = Vec::new();
    let mut low_nibble = None;
    let mut elements = 0usize;

    for value in values {
        if value > 0x0f {
            return Err(TensorError::InvalidPackedValue {
                dtype: DType::PackedU4,
                index: elements,
                value: i32::from(value),
                min: 0,
                max: 15,
            });
        }
        push_packed_nibble(&mut bytes, &mut low_nibble, value);
        elements += 1;
    }

    finish_packed_nibbles(&mut bytes, low_nibble);
    Ok((bytes, elements))
}

fn pack_i4_values(values: impl IntoIterator<Item = i8>) -> Result<(Vec<u8>, usize), TensorError> {
    let mut bytes = Vec::new();
    let mut low_nibble = None;
    let mut elements = 0usize;

    for value in values {
        if !(-8..=7).contains(&value) {
            return Err(TensorError::InvalidPackedValue {
                dtype: DType::PackedI4,
                index: elements,
                value: i32::from(value),
                min: -8,
                max: 7,
            });
        }
        push_packed_nibble(&mut bytes, &mut low_nibble, encode_i4(value));
        elements += 1;
    }

    finish_packed_nibbles(&mut bytes, low_nibble);
    Ok((bytes, elements))
}

fn pack_nibbles(values: impl IntoIterator<Item = u8>) -> (Vec<u8>, usize) {
    let mut bytes = Vec::new();
    let mut low_nibble = None;
    let mut elements = 0usize;

    for value in values {
        push_packed_nibble(&mut bytes, &mut low_nibble, value & 0x0f);
        elements += 1;
    }

    finish_packed_nibbles(&mut bytes, low_nibble);
    (bytes, elements)
}

fn push_packed_nibble(bytes: &mut Vec<u8>, low_nibble: &mut Option<u8>, value: u8) {
    if let Some(low) = low_nibble.take() {
        bytes.push(low | ((value & 0x0f) << 4));
    } else {
        *low_nibble = Some(value & 0x0f);
    }
}

fn finish_packed_nibbles(bytes: &mut Vec<u8>, low_nibble: Option<u8>) {
    if let Some(low) = low_nibble {
        bytes.push(low);
    }
}

fn unpack_u4_values(bytes: &[u8], elements: usize) -> Vec<u8> {
    (0..elements)
        .map(|index| packed_u4_value(bytes, index))
        .collect()
}

fn unpack_i4_values(bytes: &[u8], elements: usize) -> Vec<i8> {
    (0..elements)
        .map(|index| packed_i4_value(bytes, index))
        .collect()
}

fn packed_u4_value(bytes: &[u8], index: usize) -> u8 {
    packed_nibble(bytes, index)
}

fn packed_i4_value(bytes: &[u8], index: usize) -> i8 {
    decode_i4(packed_nibble(bytes, index))
}

fn packed_nibble(bytes: &[u8], index: usize) -> u8 {
    let Some(byte) = bytes.get(index / 2) else {
        return 0;
    };
    if index.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

fn encode_i4(value: i8) -> u8 {
    (i16::from(value) & 0x0f) as u8
}

fn decode_i4(value: u8) -> i8 {
    let nibble = value & 0x0f;
    if nibble & 0x08 == 0 {
        nibble as i8
    } else {
        nibble as i8 - 16
    }
}

fn packed_byte_len(elements: usize) -> usize {
    elements / 2 + elements % 2
}

/// Errors returned by tensor construction and layout conversion.
#[derive(Debug, Error, PartialEq)]
pub enum TensorError {
    /// Tensor rank did not match the layout.
    #[error("invalid tensor rank: expected {expected}, got {actual}")]
    InvalidRank {
        /// Expected rank for the layout.
        expected: usize,
        /// Actual shape rank.
        actual: usize,
    },
    /// A tensor dimension was zero.
    #[error("tensor dimension at axis {axis} must be positive, got {size}")]
    InvalidDimension {
        /// Axis containing the invalid dimension.
        axis: usize,
        /// Invalid dimension size.
        size: usize,
    },
    /// Tensor data length did not match the shape.
    #[error("invalid element count: expected {expected}, got {actual}")]
    InvalidElementCount {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// Shape element count overflowed `usize`.
    #[error("tensor shape element count overflows usize")]
    ShapeElementCountOverflow,
    /// A borrowed bool tensor contained a non-canonical value.
    #[error("bool tensor value at index {index} must be 0 or 1, got {value}")]
    InvalidBoolValue {
        /// Index containing the invalid bool value.
        index: usize,
        /// Invalid bool storage value.
        value: u8,
    },
    /// The requested layout conversion is unsupported.
    #[error("unsupported tensor layout conversion {source_layout:?} -> {target:?}")]
    UnsupportedLayoutConversion {
        /// Source layout.
        source_layout: Layout,
        /// Target layout.
        target: Layout,
    },
    /// The layout does not have a leading batch or frame axis.
    #[error("layout {0:?} does not have a leading batch/frame axis")]
    UnsupportedLeadingAxis(Layout),
    /// A tensor rescale factor was not finite.
    #[error("tensor rescale scale must be finite, got {0}")]
    NonFiniteScale(f32),
    /// A quantization scale was not finite.
    #[error("quantization scale must be finite, got {0}")]
    NonFiniteQuantizationScale(f32),
    /// A quantization scale was not positive.
    #[error("quantization scale must be positive, got {0}")]
    NonPositiveQuantizationScale(f32),
    /// A quantization zero point was outside the storage range.
    #[error("{dtype:?} zero point {zero_point} is outside [{min}, {max}]")]
    QuantizationZeroPointOutOfRange {
        /// Quantized storage dtype.
        dtype: DType,
        /// Invalid zero point.
        zero_point: i32,
        /// Minimum valid zero point.
        min: i32,
        /// Maximum valid zero point.
        max: i32,
    },
    /// A value could not be represented in packed storage.
    #[error("{dtype:?} value at index {index} must be in [{min}, {max}], got {value}")]
    InvalidPackedValue {
        /// Packed storage dtype.
        dtype: DType,
        /// Logical element index.
        index: usize,
        /// Invalid value.
        value: i32,
        /// Minimum representable value.
        min: i32,
        /// Maximum representable value.
        max: i32,
    },
    /// Packed byte storage did not match the logical element count.
    #[error("{dtype:?} storage for {elements} elements must have {expected} bytes, got {actual}")]
    InvalidPackedByteCount {
        /// Packed storage dtype.
        dtype: DType,
        /// Logical element count.
        elements: usize,
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
    /// Required per-channel statistics were empty.
    #[error("{field} channel stats cannot be empty")]
    EmptyChannelStats {
        /// Statistic field name.
        field: &'static str,
    },
    /// Per-channel statistics did not match the channel count.
    #[error("{field} channel stats must have length 1 or channel count {channels}, got {actual}")]
    InvalidChannelStats {
        /// Statistic field name.
        field: &'static str,
        /// Expected channel count.
        channels: usize,
        /// Actual statistics length.
        actual: usize,
    },
    /// A per-channel statistic was not finite.
    #[error("{field} channel stat at index {index} must be finite, got {value}")]
    NonFiniteChannelStat {
        /// Statistic field name.
        field: &'static str,
        /// Statistic index.
        index: usize,
        /// Non-finite value.
        value: f32,
    },
    /// A per-channel standard deviation was not positive.
    #[error("channel std at index {index} must be positive, got {value}")]
    NonPositiveChannelStd {
        /// Statistic index.
        index: usize,
        /// Non-positive value.
        value: f32,
    },
}

#[cfg(test)]
mod tests;
