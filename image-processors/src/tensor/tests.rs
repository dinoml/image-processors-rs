use proptest::prelude::*;

use super::*;

#[test]
fn layout_exposes_spatial_axes() {
    assert_eq!(
        (Layout::CHW.height_axis(), Layout::CHW.width_axis()),
        (1, 2)
    );
    assert_eq!(
        (Layout::HWC.height_axis(), Layout::HWC.width_axis()),
        (0, 1)
    );
    assert_eq!(
        (Layout::NCHW.height_axis(), Layout::NCHW.width_axis()),
        (2, 3)
    );
    assert_eq!(
        (Layout::NHWC.height_axis(), Layout::NHWC.width_axis()),
        (1, 2)
    );
    assert_eq!(
        (Layout::NPCHW.height_axis(), Layout::NPCHW.width_axis()),
        (3, 4)
    );
    assert_eq!(
        (Layout::NPHWC.height_axis(), Layout::NPHWC.width_axis()),
        (2, 3)
    );
    assert_eq!(
        (Layout::BFCHW.height_axis(), Layout::BFCHW.width_axis()),
        (3, 4)
    );
    assert_eq!(
        (Layout::BFHWC.height_axis(), Layout::BFHWC.width_axis()),
        (2, 3)
    );
    assert_eq!(
        (Layout::NIPCHW.height_axis(), Layout::NIPCHW.width_axis()),
        (4, 5)
    );
    assert_eq!(
        (Layout::NIPHWC.height_axis(), Layout::NIPHWC.width_axis()),
        (3, 4)
    );
    assert_eq!((Layout::NC.height_axis(), Layout::NC.width_axis()), (0, 1));
}

#[test]
fn tensor_validates_shape_against_layout() {
    let err = Tensor::new(TensorData::F32(vec![0.0; 4]), vec![1, 4], Layout::NCHW).unwrap_err();
    assert_eq!(
        err,
        TensorError::InvalidRank {
            expected: 4,
            actual: 2,
        }
    );

    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 12]),
        vec![1, 3, 2, 2],
        Layout::NCHW,
    )
    .unwrap();
    assert_eq!(tensor.batch(), Some(1));
    assert_eq!(tensor.channels(), 3);
    assert_eq!(tensor.height(), 2);
    assert_eq!(tensor.width(), 2);
    assert_eq!(tensor.patches(), None);

    let patched = Tensor::new(
        TensorData::F32(vec![0.0; 24]),
        vec![1, 2, 3, 2, 2],
        Layout::NPCHW,
    )
    .unwrap();
    assert_eq!(patched.batch(), Some(1));
    assert_eq!(patched.patches(), Some(2));
    assert_eq!(patched.channels(), 3);
    assert_eq!(patched.height(), 2);
    assert_eq!(patched.width(), 2);
    assert_eq!(patched.layout().image_axis(), None);

    let video_batch = Tensor::new(
        TensorData::F32(vec![0.0; 48]),
        vec![2, 2, 3, 2, 2],
        Layout::BFCHW,
    )
    .unwrap();
    assert_eq!(video_batch.batch(), Some(2));
    assert_eq!(video_batch.frames(), Some(2));
    assert_eq!(video_batch.layout().frame_axis(), Some(1));
    assert_eq!(video_batch.patches(), None);
    assert_eq!(video_batch.channels(), 3);
    assert_eq!(video_batch.height(), 2);
    assert_eq!(video_batch.width(), 2);

    let nested = Tensor::new(
        TensorData::F32(vec![0.0; 48]),
        vec![1, 2, 2, 3, 2, 2],
        Layout::NIPCHW,
    )
    .unwrap();
    assert_eq!(nested.batch(), Some(1));
    assert_eq!(nested.layout().image_axis(), Some(1));
    assert_eq!(nested.patches(), Some(2));
    assert_eq!(nested.channels(), 3);
    assert_eq!(nested.height(), 2);
    assert_eq!(nested.width(), 2);

    let matrix = Tensor::new(TensorData::F32(vec![0.0; 6]), vec![2, 3], Layout::NC).unwrap();
    assert_eq!(matrix.batch(), None);
    assert_eq!(matrix.channels(), 3);
    assert_eq!(matrix.height(), 2);
    assert_eq!(matrix.width(), 3);
}

#[test]
fn tensor_converts_patch_batched_channel_layouts() {
    let tensor = Tensor::new(
        TensorData::F32((0..12).map(|value| value as f32).collect()),
        vec![1, 1, 3, 2, 2],
        Layout::NPCHW,
    )
    .unwrap();

    let converted = tensor.to_layout(Layout::NPHWC).unwrap();

    assert_eq!(converted.shape(), [1, 1, 2, 2, 3]);
    assert_eq!(
        converted.data().to_vec::<f32>(),
        vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );
}

#[test]
fn tensor_converts_batched_video_channel_layouts() {
    let tensor = Tensor::new(
        TensorData::F32((0..24).map(|value| value as f32).collect()),
        vec![2, 1, 3, 2, 2],
        Layout::BFCHW,
    )
    .unwrap();

    let converted = tensor.to_layout(Layout::BFHWC).unwrap();

    assert_eq!(converted.shape(), [2, 1, 2, 2, 3]);
    assert_eq!(
        converted.data().to_vec::<f32>(),
        vec![
            0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 12.0, 16.0, 20.0, 13.0,
            17.0, 21.0, 14.0, 18.0, 22.0, 15.0, 19.0, 23.0,
        ]
    );
    assert_eq!(converted.batch(), Some(2));
    assert_eq!(converted.frames(), Some(1));
}

#[test]
fn tensor_rejects_layout_conversion_between_semantic_families() {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 12]),
        vec![1, 1, 3, 2, 2],
        Layout::NPCHW,
    )
    .unwrap();

    let error = tensor.to_layout(Layout::BFHWC).unwrap_err();

    assert_eq!(
        error,
        TensorError::UnsupportedLayoutConversion {
            source_layout: Layout::NPCHW,
            target: Layout::BFHWC,
        }
    );
}

#[test]
fn tensor_converts_nested_patch_batched_channel_layouts() {
    let tensor = Tensor::new(
        TensorData::F32((0..24).map(|value| value as f32).collect()),
        vec![1, 1, 2, 3, 2, 2],
        Layout::NIPCHW,
    )
    .unwrap();

    let converted = tensor.to_layout(Layout::NIPHWC).unwrap();

    assert_eq!(converted.shape(), [1, 1, 2, 2, 2, 3]);
    assert_eq!(
        converted.data().to_vec::<f32>(),
        vec![
            0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 12.0, 16.0, 20.0, 13.0,
            17.0, 21.0, 14.0, 18.0, 22.0, 15.0, 19.0, 23.0,
        ]
    );
}

#[test]
fn tensor_rejects_invalid_dimensions_and_shape_overflow() {
    let err = Tensor::new(TensorData::F32(Vec::new()), vec![1, 0, 1], Layout::CHW).unwrap_err();
    assert_eq!(err, TensorError::InvalidDimension { axis: 1, size: 0 });

    let err = Tensor::new(
        TensorData::F32(Vec::new()),
        vec![usize::MAX, 2, 1],
        Layout::CHW,
    )
    .unwrap_err();
    assert_eq!(err, TensorError::ShapeElementCountOverflow);
}

#[test]
fn tensor_supports_multiple_storage_dtypes() {
    let tensor = Tensor::new(
        TensorData::U8(vec![0, 127, 255]),
        vec![1, 1, 3],
        Layout::HWC,
    )
    .unwrap();

    assert_eq!(tensor.dtype(), DType::U8);
    assert_eq!(tensor.data().to_vec::<f32>(), vec![0.0, 127.0, 255.0]);
    assert_eq!(tensor.data().to_vec::<u8>(), vec![0, 127, 255]);
    assert_eq!(tensor.data().to_vec::<i64>(), vec![0, 127, 255]);

    let half_tensor = Tensor::new(
        TensorData::F16(vec![f16::from_f32(1.5)]),
        vec![1, 1, 1],
        Layout::CHW,
    )
    .unwrap();
    assert_eq!(half_tensor.dtype(), DType::F16);
    assert_eq!(half_tensor.data().to_vec::<f32>(), vec![1.5]);
    assert_eq!(half_tensor.data().to_vec::<f16>(), vec![f16::from_f32(1.5)]);

    let bool_tensor =
        Tensor::new(TensorData::Bool(vec![0, 1]), vec![1, 1, 2], Layout::CHW).unwrap();
    assert_eq!(bool_tensor.data().to_vec::<u8>(), vec![0, 1]);
    assert_eq!(bool_tensor.data().to_vec::<bool>(), vec![false, true]);
}

#[test]
fn tensor_supports_affine_quantized_storage() {
    let quantization = QuantizationParams::new(0.5, 10).unwrap();
    let tensor = Tensor::new(
        TensorData::QuantizedU8 {
            values: vec![10, 12, 14],
            quantization,
        },
        vec![1, 1, 3],
        Layout::HWC,
    )
    .unwrap();

    assert_eq!(tensor.dtype(), DType::QuantizedU8);
    assert_eq!(tensor.data().to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
    assert_eq!(
        tensor.rescale(2.0).unwrap().data().to_vec::<f32>(),
        vec![0.0, 2.0, 4.0]
    );
}

#[test]
fn tensor_supports_packed_unsigned_four_bit_storage() {
    let tensor = Tensor::new(
        TensorData::packed_u4([1, 15, 2]).unwrap(),
        vec![1, 1, 3],
        Layout::HWC,
    )
    .unwrap();

    let chw = tensor.to_layout(Layout::CHW).unwrap();

    assert_eq!(tensor.dtype(), DType::PackedU4);
    assert_eq!(tensor.data().to_vec::<u8>(), vec![1, 15, 2]);
    assert_eq!(chw.shape(), [3, 1, 1]);
    assert_eq!(chw.data().to_vec::<u8>(), vec![1, 15, 2]);
}

#[test]
fn tensor_supports_packed_signed_four_bit_storage() {
    let tensor = Tensor::new(
        TensorData::packed_i4([-8, -1, 0, 7]).unwrap(),
        vec![1, 1, 4],
        Layout::HWC,
    )
    .unwrap();

    assert_eq!(tensor.dtype(), DType::PackedI4);
    assert_eq!(tensor.data().to_vec::<i32>(), vec![-8, -1, 0, 7]);
    assert_eq!(tensor.data().to_vec::<f32>(), vec![-8.0, -1.0, 0.0, 7.0]);
}

#[test]
fn tensor_rejects_invalid_quantized_and_packed_storage() {
    let err = QuantizationParams::new(0.0, 0).unwrap_err();
    assert_eq!(err, TensorError::NonPositiveQuantizationScale(0.0));

    let quantization = QuantizationParams::new(1.0, 300).unwrap();
    let err = Tensor::new(
        TensorData::QuantizedU8 {
            values: vec![0],
            quantization,
        },
        vec![1, 1, 1],
        Layout::HWC,
    )
    .unwrap_err();
    assert_eq!(
        err,
        TensorError::QuantizationZeroPointOutOfRange {
            dtype: DType::QuantizedU8,
            zero_point: 300,
            min: 0,
            max: 255,
        }
    );

    let err = TensorData::packed_u4([16]).unwrap_err();
    assert_eq!(
        err,
        TensorError::InvalidPackedValue {
            dtype: DType::PackedU4,
            index: 0,
            value: 16,
            min: 0,
            max: 15,
        }
    );

    let err = Tensor::new(
        TensorData::PackedU4 {
            bytes: Vec::new(),
            elements: 1,
        },
        vec![1, 1, 1],
        Layout::HWC,
    )
    .unwrap_err();
    assert_eq!(
        err,
        TensorError::InvalidPackedByteCount {
            dtype: DType::PackedU4,
            elements: 1,
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn tensor_rescale_converts_values_to_f32() {
    let tensor = Tensor::new(TensorData::U8(vec![10, 20, 30]), vec![1, 1, 3], Layout::HWC).unwrap();

    let scaled = tensor.rescale(0.5).unwrap();

    assert_eq!(scaled.dtype(), DType::F32);
    assert_eq!(scaled.shape(), [1, 1, 3]);
    assert_eq!(scaled.layout(), Layout::HWC);
    assert_eq!(float32_values(&scaled), &[5.0, 10.0, 15.0]);
}

#[test]
fn tensor_rescale_rejects_non_finite_scale() {
    let tensor = Tensor::new(TensorData::F32(vec![1.0]), vec![1, 1, 1], Layout::HWC).unwrap();

    let err = tensor.rescale(f32::INFINITY).unwrap_err();

    assert!(matches!(
        err,
        TensorError::NonFiniteScale(value) if value.is_infinite()
    ));
}

#[test]
fn tensor_normalize_channels_handles_nhwc() {
    let tensor = Tensor::new(
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![1, 1, 2, 3],
        Layout::NHWC,
    )
    .unwrap();

    let normalized = tensor
        .normalize_channels(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .unwrap();

    assert_eq!(normalized.shape(), [1, 1, 2, 3]);
    assert_eq!(normalized.layout(), Layout::NHWC);
    assert_eq!(float32_values(&normalized), &[0.0, 0.0, 0.0, 3.0, 1.5, 1.0]);
}

#[test]
fn tensor_normalize_channels_handles_nchw() {
    let tensor = Tensor::new(
        TensorData::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        vec![1, 3, 1, 2],
        Layout::NCHW,
    )
    .unwrap()
    .with_leading_axis(TensorLeadingAxis::Frames)
    .unwrap();

    let normalized = tensor
        .normalize_channels(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .unwrap();

    assert_eq!(normalized.shape(), [1, 3, 1, 2]);
    assert_eq!(normalized.layout(), Layout::NCHW);
    assert_eq!(normalized.leading_axis(), Some(TensorLeadingAxis::Frames));
    assert_eq!(float32_values(&normalized), &[0.0, 3.0, 0.0, 1.5, 0.0, 1.0]);
}

#[test]
fn tensor_normalize_channels_accepts_scalar_stats() {
    let tensor = Tensor::new(TensorData::U8(vec![1, 2, 3]), vec![1, 1, 3], Layout::HWC).unwrap();

    let normalized = tensor.normalize_channels(&[1.0], &[2.0]).unwrap();

    assert_eq!(normalized.dtype(), DType::F32);
    assert_eq!(float32_values(&normalized), &[0.0, 0.5, 1.0]);
}

#[test]
fn tensor_normalize_channels_validates_stats() {
    let tensor = Tensor::new(
        TensorData::F32(vec![1.0, 2.0, 3.0]),
        vec![1, 1, 3],
        Layout::HWC,
    )
    .unwrap();

    let err = tensor.normalize_channels(&[], &[1.0]).unwrap_err();
    assert_eq!(err, TensorError::EmptyChannelStats { field: "mean" });

    let err = tensor.normalize_channels(&[0.0, 0.0], &[1.0]).unwrap_err();
    assert_eq!(
        err,
        TensorError::InvalidChannelStats {
            field: "mean",
            channels: 3,
            actual: 2,
        }
    );

    let err = tensor.normalize_channels(&[f32::NAN], &[1.0]).unwrap_err();
    assert!(matches!(
        err,
        TensorError::NonFiniteChannelStat {
            field: "mean",
            index: 0,
            value,
        } if value.is_nan()
    ));

    let err = tensor.normalize_channels(&[0.0], &[0.0]).unwrap_err();
    assert_eq!(
        err,
        TensorError::NonPositiveChannelStd {
            index: 0,
            value: 0.0,
        }
    );
}

#[test]
fn tensor_converts_layouts() {
    let tensor = Tensor::new(
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![1, 2, 3],
        Layout::HWC,
    )
    .unwrap();

    let chw = tensor.to_layout(Layout::CHW).unwrap();

    assert_eq!(chw.shape(), [3, 1, 2]);
    assert_eq!(float32_values(&chw), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn tensor_exposes_batch_or_frame_leading_axis() {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 2 * 3 * 2 * 2]),
        vec![2, 3, 2, 2],
        Layout::NCHW,
    )
    .unwrap();

    let frames = tensor
        .with_leading_axis(TensorLeadingAxis::Frames)
        .unwrap()
        .to_layout(Layout::NHWC)
        .unwrap();

    assert_eq!(frames.batch(), None);
    assert_eq!(frames.frames(), Some(2));
    assert_eq!(frames.leading_axis(), Some(TensorLeadingAxis::Frames));
}

#[test]
fn tensor_rejects_frame_leading_axis_for_explicit_batch_layout() {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0; 12]),
        vec![1, 1, 3, 2, 2],
        Layout::BFCHW,
    )
    .unwrap();

    let error = tensor
        .with_leading_axis(TensorLeadingAxis::Frames)
        .unwrap_err();

    assert_eq!(error, TensorError::UnsupportedLeadingAxis(Layout::BFCHW));
}

#[test]
fn tensor_view_borrows_owned_tensor_storage() {
    let tensor = Tensor::new(
        TensorData::U8(vec![0, 127, 255]),
        vec![1, 1, 3],
        Layout::HWC,
    )
    .unwrap();
    let view = tensor.view();

    let TensorData::U8(owned) = tensor.data() else {
        panic!("expected owned u8 tensor data");
    };
    let TensorDataView::U8(borrowed) = view.data() else {
        panic!("expected borrowed u8 tensor data");
    };

    assert!(std::ptr::eq(borrowed.as_ptr(), owned.as_ptr()));
    assert_eq!(view.shape(), tensor.shape());
    assert_eq!(view.layout(), tensor.layout());
    assert_eq!(view.dtype(), tensor.dtype());
}

#[test]
fn tensor_view_copies_to_owned_tensor() {
    let values = [0, 10, 20, 30, 40, 50];
    let view = TensorView::image(
        TensorDataView::U8(&values),
        1,
        2,
        3,
        ImageLayout::HeightWidthChannels,
    )
    .unwrap();

    let tensor = view.to_owned_tensor().unwrap();

    assert_eq!(tensor.shape(), [1, 2, 3]);
    assert_eq!(tensor.layout(), Layout::HWC);
    assert_eq!(tensor.data().to_vec::<u8>(), values);
}

#[test]
fn video_layout_marks_leading_axis_as_frames() {
    let values = [0; 12];
    let view = TensorView::video(
        TensorDataView::U8(&values),
        2,
        1,
        2,
        3,
        VideoLayout::FramesHeightWidthChannels,
    )
    .unwrap();

    assert_eq!(view.shape(), [2, 1, 2, 3]);
    assert_eq!(view.layout(), Layout::NHWC);
    assert_eq!(view.frames(), Some(2));
    assert_eq!(view.batch(), None);
}

#[test]
fn tensor_view_rejects_non_canonical_bool_storage() {
    let err =
        TensorView::new(TensorDataView::Bool(&[0, 2]), vec![1, 1, 2], Layout::CHW).unwrap_err();

    assert_eq!(err, TensorError::InvalidBoolValue { index: 1, value: 2 });
}

proptest! {
    #[test]
    fn tensor_layout_conversion_roundtrips_nhwc(
        batch in 1usize..4,
        height in 1usize..5,
        width in 1usize..5,
        channels in 1usize..5,
    ) {
        let len = batch * height * width * channels;
        let values = (0..len).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = Tensor::new(
            TensorData::F32(values),
            vec![batch, height, width, channels],
            Layout::NHWC,
        )
        .unwrap();

        let roundtrip = tensor
            .to_layout(Layout::NCHW)
            .unwrap()
            .to_layout(Layout::NHWC)
            .unwrap();

        prop_assert_eq!(roundtrip.shape(), tensor.shape());
        prop_assert_eq!(float32_values(&roundtrip), float32_values(&tensor));
    }
}

fn float32_values(tensor: &Tensor) -> &[f32] {
    match tensor.data() {
        TensorData::F32(values) => values,
        _ => panic!("expected F32 tensor data"),
    }
}
