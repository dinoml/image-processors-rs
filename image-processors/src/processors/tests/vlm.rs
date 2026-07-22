use super::*;

#[test]
fn clip_processor_open_batch_loads_paths_and_stacks_tensor() {
    let first = write_test_png("image_processors_clip_batch_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_clip_batch_second.png", &[0, 255, 0]);
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();

    let tensor = processor.open_batch(&[first, second]).unwrap();

    assert_eq!(tensor.shape(), [2, 3, 224, 224]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.batch(), Some(2));
}

#[test]
fn clip_processor_open_batch_into_recycles_workspace_buffer() {
    let first = write_test_png("image_processors_clip_batch_into_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_clip_batch_into_second.png", &[0, 255, 0]);
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();
    let mut workspace = ImageProcessorWorkspace::new();

    let tensor = processor
        .open_batch_into(&[first, second], &mut workspace)
        .unwrap();
    let output_len = tensor.data().len();

    assert_eq!(tensor.shape(), [2, 3, 224, 224]);
    workspace.recycle_tensor(tensor).unwrap();
    assert!(workspace.f32_capacity() >= output_len);
}

#[test]
fn clip_processor_f16_slice_matches_f32_rounding_across_layouts_and_resize_modes() {
    let first = write_test_png("image_processors_clip_f16_slice_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_clip_f16_slice_second.png", &[0, 255, 0]);

    #[cfg(feature = "parallel")]
    let batch_execution = BatchExecution::Parallel;
    #[cfg(not(feature = "parallel"))]
    let batch_execution = BatchExecution::Serial;

    for (output_layout, do_center_crop) in [
        (ImageLayout::ChannelsHeightWidth, true),
        (ImageLayout::HeightWidthChannels, true),
        (ImageLayout::ChannelsHeightWidth, false),
        (ImageLayout::HeightWidthChannels, false),
    ] {
        let processor = ClipImageProcessor::new(ClipImageProcessorConfig {
            output_layout,
            do_center_crop,
            batch_execution,
            ..Default::default()
        })
        .unwrap();
        let paths = [&first, &second];
        let expected = processor.open_batch(&paths).unwrap();
        let TensorData::F32(expected_values) = expected.data() else {
            panic!("expected float32 reference output");
        };
        let expected_values = expected_values
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let expected_metadata = (
            expected.shape().to_vec(),
            expected.layout(),
            DType::F16,
            expected.leading_axis(),
        );
        let mut output = vec![f16::ZERO; expected_values.len()];
        let mut workspace = ImageProcessorWorkspace::new();

        let view = processor
            .open_batch_f16_into_slice(&paths, &mut output, &mut workspace)
            .unwrap();
        let actual_metadata = (
            view.shape().to_vec(),
            view.layout(),
            view.dtype(),
            view.leading_axis(),
        );
        let actual_values = match view.data() {
            crate::tensor::TensorDataView::F16(values) => values.to_vec(),
            other => panic!("expected float16 view, got {other:?}"),
        };

        assert_eq!(
            (actual_metadata, actual_values),
            (expected_metadata, expected_values),
            "f16 slice output diverged for {output_layout:?}, do_center_crop={do_center_crop}"
        );
    }
}

#[test]
fn clip_processor_owned_f16_output_recycles_workspace_storage() {
    let first = write_test_png("image_processors_clip_f16_owned_first.png", &[255, 0, 0]);
    let second = write_test_png("image_processors_clip_f16_owned_second.png", &[0, 255, 0]);
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();
    let paths = [&first, &second];
    let mut workspace = ImageProcessorWorkspace::new();
    let expected = processor.open_batch(&paths).unwrap();
    let TensorData::F32(expected) = expected.data() else {
        panic!("expected float32 reference output");
    };
    let expected = expected
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let first = processor
        .open_batch_f16_into(&paths, &mut workspace)
        .unwrap();
    let output_len = first.data().len();
    workspace.recycle_tensor(first).unwrap();
    let retained_capacity = workspace.f16_capacity();
    let repeated = processor
        .open_batch_f16_into(&paths, &mut workspace)
        .unwrap();
    let TensorData::F16(repeated_values) = repeated.data() else {
        panic!("expected float16 owned output");
    };

    assert_eq!(
        (
            repeated.dtype(),
            repeated.shape(),
            repeated.layout(),
            retained_capacity >= output_len,
            repeated_values,
        ),
        (
            DType::F16,
            [2, 3, 224, 224].as_slice(),
            Layout::NCHW,
            true,
            &expected,
        )
    );
}

#[test]
fn clip_processor_f16_slice_rejects_an_incorrect_output_length() {
    let path = write_test_png("image_processors_clip_f16_short.png", &[255, 0, 0]);
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();
    let expected_len = 3 * 224 * 224;
    let mut output = vec![f16::ZERO; expected_len - 1];
    let mut workspace = ImageProcessorWorkspace::new();

    let error = processor
        .open_batch_f16_into_slice(&[path], &mut output, &mut workspace)
        .unwrap_err();

    assert!(matches!(
        error,
        ImageProcessorError::Tensor(TensorError::InvalidElementCount {
            expected,
            actual
        }) if expected == expected_len && actual == expected_len - 1
    ));
}

#[test]
fn llava_next_processor_output_extracts_anyres_patches_and_metadata() {
    let frame = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![127; 4 * 2 * 3]).unwrap();
    let processor = llava_next_small_patch_processor();

    let output = processor.preprocess_image_output(&frame).unwrap();
    let pixel_values = output.pixel_values().unwrap();

    assert_eq!(pixel_values.shape(), [1, 3, 3, 2, 2]);
    assert_eq!(pixel_values.layout(), Layout::NPCHW);
    assert_eq!(pixel_values.batch(), Some(1));
    assert_eq!(pixel_values.patches(), Some(3));
    assert_eq!(
        output.original_sizes().unwrap(),
        &[ImageSize {
            height: 2,
            width: 4,
        }]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[ImageSize {
            height: 2,
            width: 4,
        }]
    );
    assert_eq!(output.image_patch_counts().unwrap(), &[3]);
    let Some(ProcessorMetadataValue::ImageSizes(selected_sizes)) =
        output.metadata_value(&ProcessorMetadataName::other("selected_sizes"))
    else {
        panic!("expected selected_sizes metadata");
    };
    assert_eq!(
        selected_sizes,
        &[ImageSize {
            height: 2,
            width: 4,
        }]
    );
}

#[test]
fn llava_next_processor_pads_batch_to_largest_patch_count() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![255; 4 * 2 * 3]).unwrap();
    let square = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![255; 2 * 2 * 3]).unwrap();
    let processor = llava_next_small_patch_processor();

    let output = processor.preprocess_images_output(&[wide, square]).unwrap();
    let pixel_values = output.pixel_values().unwrap();
    let values = pixel_values.data().to_vec::<f32>();
    let padded_patch_offset = (3 + 2) * 3 * 2 * 2;

    assert_eq!(pixel_values.shape(), [2, 3, 3, 2, 2]);
    assert_eq!(output.image_patch_counts().unwrap(), &[3, 2]);
    assert!(values[padded_patch_offset..padded_patch_offset + 12]
        .iter()
        .all(|value| value.abs() < f32::EPSILON));
}

#[test]
fn llava_next_processor_rejects_unpadded_mixed_patch_counts() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![255; 4 * 2 * 3]).unwrap();
    let square = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![255; 2 * 2 * 3]).unwrap();
    let processor = LlavaNextImageProcessor::new(LlavaNextImageProcessorConfig {
        do_pad: false,
        size: square_size(2),
        crop_size: square_size(2),
        image_grid_pinpoints: vec![
            ImageSize {
                height: 2,
                width: 2,
            },
            ImageSize {
                height: 2,
                width: 4,
            },
        ],
        resample: ResizeFilter::Nearest,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap();

    let err = processor.preprocess_images(&[wide, square]).unwrap_err();

    assert!(matches!(err, ImageProcessorError::IncompatibleBatchShapes));
}

#[test]
fn idefics3_processor_output_splits_nested_samples_and_attention_mask() {
    let wide = ImageFrame::new(9, 7, PixelFormat::Rgb8, vec![127; 9 * 7 * 3]).unwrap();
    let tall = ImageFrame::new(4, 11, PixelFormat::Rgb8, vec![63; 4 * 11 * 3]).unwrap();
    let processor = idefics3_small_split_processor();

    let output = processor
        .preprocess_image_samples_output(&[vec![wide], vec![tall]])
        .unwrap();
    let pixel_values = output.pixel_values().unwrap();
    let attention_mask = output
        .tensor(&ProcessorTensorName::other("pixel_attention_mask"))
        .unwrap();

    assert_eq!(pixel_values.shape(), [2, 5, 3, 5, 5]);
    assert_eq!(pixel_values.layout(), Layout::NPCHW);
    assert_eq!(pixel_values.batch(), Some(2));
    assert_eq!(pixel_values.patches(), Some(5));
    assert_eq!(attention_mask.shape(), [2, 5, 1, 5, 5]);
    assert_eq!(attention_mask.layout(), Layout::NPCHW);
    assert_eq!(
        attention_mask
            .data()
            .to_vec::<bool>()
            .iter()
            .filter(|v| **v)
            .count(),
        200
    );
    assert_eq!(output.image_patch_counts().unwrap(), &[5, 3]);
    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("rows")),
        Some([vec![2], vec![2]].as_slice())
    );
    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("cols")),
        Some([vec![2], vec![1]].as_slice())
    );
}

#[test]
fn gemma3_processor_output_emits_original_plus_pan_and_scan_crops() {
    let frame = ImageFrame::new(
        4,
        2,
        PixelFormat::Rgb8,
        vec![
            1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24, 5, 15, 25, 6, 16, 26, 7, 17, 27, 8, 18, 28,
        ],
    )
    .unwrap();
    let processor = gemma3_small_pan_and_scan_processor();

    let output = processor.preprocess_image_output(&frame).unwrap();
    let pixel_values = output.pixel_values().unwrap();

    assert_eq!(pixel_values.shape(), [3, 3, 2, 2]);
    assert_eq!(pixel_values.layout(), Layout::NCHW);
    assert_eq!(
        pixel_values.data().to_vec::<f32>(),
        vec![
            2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 18.0, 22.0, 24.0, 26.0, 28.0, 1.0, 2.0, 5.0, 6.0,
            11.0, 12.0, 15.0, 16.0, 21.0, 22.0, 25.0, 26.0, 3.0, 4.0, 7.0, 8.0, 13.0, 14.0, 17.0,
            18.0, 23.0, 24.0, 27.0, 28.0,
        ]
    );
    assert_eq!(
        output.original_sizes().unwrap(),
        &[ImageSize {
            height: 2,
            width: 4
        }]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[
            ImageSize {
                height: 2,
                width: 2
            },
            ImageSize {
                height: 2,
                width: 2
            },
            ImageSize {
                height: 2,
                width: 2
            },
        ]
    );
    assert_eq!(
        output.metadata_value(&ProcessorMetadataName::other("num_crops")),
        Some(&ProcessorMetadataValue::Counts(vec![2]))
    );
}

#[test]
fn mllama_processor_output_tiles_nested_samples_and_metadata() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![7; 4 * 2 * 3]).unwrap();
    let square = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![9; 2 * 2 * 3]).unwrap();
    let tall = ImageFrame::new(2, 4, PixelFormat::Rgb8, vec![13; 2 * 4 * 3]).unwrap();
    let processor = mllama_small_tile_processor();

    let output = processor
        .preprocess_image_samples_output(&[vec![wide, square], vec![tall]])
        .unwrap();
    let pixel_values = output.pixel_values().unwrap();
    let values = pixel_values.data().to_vec::<f32>();
    let tile_len = 3 * 2 * 2;

    assert_eq!(pixel_values.shape(), [2, 2, 4, 3, 2, 2]);
    assert_eq!(pixel_values.layout(), Layout::NIPCHW);
    assert_eq!(pixel_values.batch(), Some(2));
    assert_eq!(pixel_values.patches(), Some(4));
    assert_eq!(
        output.original_sizes().unwrap(),
        &[
            ImageSize {
                height: 2,
                width: 4
            },
            ImageSize {
                height: 2,
                width: 2
            },
            ImageSize {
                height: 4,
                width: 2
            },
        ]
    );
    assert_eq!(output.image_patch_counts().unwrap(), &[2, 1, 2]);
    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("num_tiles")),
        Some([vec![2, 1], vec![2]].as_slice())
    );
    assert_eq!(
        output.nested_counts(&ProcessorMetadataName::other("aspect_ratio_ids")),
        Some([vec![2, 1], vec![5, 0]].as_slice())
    );
    assert_eq!(
        output.nested_bool_mask(&ProcessorMetadataName::other("aspect_ratio_mask")),
        Some(
            [
                vec![
                    vec![true, true, false, false],
                    vec![true, false, false, false]
                ],
                vec![
                    vec![true, true, false, false],
                    vec![true, false, false, false]
                ],
            ]
            .as_slice()
        )
    );

    assert!(values[0..tile_len].iter().all(|value| *value == 7.0));
    assert!(values[tile_len..tile_len * 2]
        .iter()
        .all(|value| *value == 7.0));
    assert!(values[tile_len * 3..tile_len * 4]
        .iter()
        .all(|value| value.abs() < f32::EPSILON));
    let padded_image_slot = tile_len * 12;
    assert!(values[padded_image_slot..padded_image_slot + tile_len * 4]
        .iter()
        .all(|value| value.abs() < f32::EPSILON));
}

#[test]
fn mllama_processor_rejects_disabled_required_padding() {
    let err = MllamaImageProcessor::new(MllamaImageProcessorConfig {
        do_pad: false,
        ..Default::default()
    })
    .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::UnsupportedProcessorOption { field: "do_pad" }
    ));
}

#[test]
fn qwen_processor_uses_smart_resize_limits() {
    let frame = ImageFrame::new(777, 333, PixelFormat::Rgb8, vec![127; 777 * 333 * 3]).unwrap();
    let processor = QwenVlImageProcessor::new(QwenVlImageProcessorConfig::default()).unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();

    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.height() % 28, 0);
    assert_eq!(tensor.width() % 28, 0);
    assert!(tensor.height() * tensor.width() <= 28 * 28 * 1280);
}

#[test]
fn qwen_processor_output_contains_grid_metadata() {
    let frame = ImageFrame::new(56, 28, PixelFormat::Rgb8, vec![127; 56 * 28 * 3]).unwrap();
    let processor = QwenVlImageProcessor::new(QwenVlImageProcessorConfig::default()).unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();

    assert_eq!(
        output.original_sizes().unwrap(),
        &[ImageSize {
            height: 28,
            width: 56,
        }]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[ImageSize {
            height: 56,
            width: 84,
        }]
    );
    assert_eq!(output.pixel_values().unwrap().shape(), [24, 1176]);
    assert_eq!(output.pixel_values().unwrap().layout(), Layout::NC);
    assert_eq!(output.image_grid_thw().unwrap(), &[[1, 4, 6]]);
}

#[test]
fn qwen_processor_output_flattens_patches_in_qwen_order() {
    let frame = ImageFrame::new(
        4,
        2,
        PixelFormat::Rgb8,
        vec![
            1, 11, 21, 2, 12, 22, 5, 15, 25, 6, 16, 26, 3, 13, 23, 4, 14, 24, 7, 17, 27, 8, 18, 28,
        ],
    )
    .unwrap();
    let processor = QwenVlImageProcessor::new(QwenVlImageProcessorConfig {
        resize_limits: ResizeLimits {
            factor: 2,
            min_pixels: 8,
            max_pixels: 8,
        },
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 1,
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    })
    .unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();
    let values = output.pixel_values().unwrap().data().to_vec::<f32>();

    assert_eq!(output.pixel_values().unwrap().shape(), [2, 24]);
    assert_eq!(output.pixel_values().unwrap().layout(), Layout::NC);
    assert_eq!(output.image_grid_thw().unwrap(), &[[1, 1, 2]],);
    assert_eq!(
        values,
        [
            qwen_expected_patch(
                [1.0, 2.0, 3.0, 4.0],
                [11.0, 12.0, 13.0, 14.0],
                [21.0, 22.0, 23.0, 24.0,]
            ),
            qwen_expected_patch(
                [5.0, 6.0, 7.0, 8.0],
                [15.0, 16.0, 17.0, 18.0],
                [25.0, 26.0, 27.0, 28.0,]
            ),
        ]
        .concat()
    );
}

#[test]
fn qwen_processor_images_output_concatenates_patch_rows_and_metadata() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![127; 4 * 2 * 3]).unwrap();
    let square = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![63; 2 * 2 * 3]).unwrap();
    let processor = qwen_small_patch_processor();

    let output = processor.preprocess_images_output(&[wide, square]).unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [3, 24]);
    assert_eq!(output.pixel_values().unwrap().layout(), Layout::NC);
    assert_eq!(
        output.original_sizes().unwrap(),
        &[
            ImageSize {
                height: 2,
                width: 4,
            },
            ImageSize {
                height: 2,
                width: 2,
            },
        ]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[
            ImageSize {
                height: 2,
                width: 4,
            },
            ImageSize {
                height: 2,
                width: 2,
            },
        ]
    );
    assert_eq!(output.image_grid_thw().unwrap(), &[[1, 1, 2], [1, 1, 1]]);
}

#[test]
fn qwen_processor_image_sequence_output_tracks_each_frame_grid() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![127; 4 * 2 * 3]).unwrap();
    let square = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![63; 2 * 2 * 3]).unwrap();
    let sequence =
        ImageSequence::new(vec![wide, square], crate::media::LoopBehavior::Once).unwrap();
    let processor = qwen_small_patch_processor();

    let output = processor
        .preprocess_image_sequence_output(&sequence)
        .unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [3, 24]);
    assert_eq!(output.image_grid_thw().unwrap(), &[[1, 1, 2], [1, 1, 1]]);
}

#[test]
fn qwen_processor_video_output_groups_frames_by_temporal_patch_size() {
    let first = ImageFrame::new(
        2,
        2,
        PixelFormat::Rgb8,
        vec![1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24],
    )
    .unwrap();
    let second = ImageFrame::new(
        2,
        2,
        PixelFormat::Rgb8,
        vec![101, 111, 121, 102, 112, 122, 103, 113, 123, 104, 114, 124],
    )
    .unwrap();
    let third = ImageFrame::new(
        2,
        2,
        PixelFormat::Rgb8,
        vec![201, 211, 221, 202, 212, 222, 203, 213, 223, 204, 214, 224],
    )
    .unwrap();
    let video = VideoClip::new(
        vec![
            crate::media::VideoFrame::new(first),
            crate::media::VideoFrame::new(second),
            crate::media::VideoFrame::new(third),
        ],
        Some(24.0),
    )
    .unwrap();
    let processor = qwen_small_patch_processor();

    let output = processor.preprocess_video_output(&video).unwrap();
    let values = output.pixel_values().unwrap().data().to_vec::<f32>();

    assert_eq!(output.pixel_values().unwrap().shape(), [2, 24]);
    assert_eq!(output.image_grid_thw().unwrap(), &[[2, 1, 1]]);
    assert_eq!(
        values,
        [
            qwen_expected_temporal_patch(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [11.0, 12.0, 13.0, 14.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [101.0, 102.0, 103.0, 104.0],
                    [111.0, 112.0, 113.0, 114.0],
                    [121.0, 122.0, 123.0, 124.0],
                ],
            ),
            qwen_expected_temporal_patch(
                [
                    [201.0, 202.0, 203.0, 204.0],
                    [211.0, 212.0, 213.0, 214.0],
                    [221.0, 222.0, 223.0, 224.0],
                ],
                [
                    [201.0, 202.0, 203.0, 204.0],
                    [211.0, 212.0, 213.0, 214.0],
                    [221.0, 222.0, 223.0, 224.0],
                ],
            ),
        ]
        .concat()
    );
}

#[test]
fn qwen_processor_rejects_inconsistent_patch_geometry() {
    let err = QwenVlImageProcessor::new(QwenVlImageProcessorConfig {
        patch_size: 7,
        ..Default::default()
    })
    .unwrap_err();

    match err {
        ImageProcessorError::InvalidPatchGeometry {
            resize_factor,
            patch_size,
            merge_size,
        } => {
            assert_eq!(resize_factor, 28);
            assert_eq!(patch_size, 7);
            assert_eq!(merge_size, 2);
        }
        other => panic!("expected invalid patch geometry error, got {other:?}"),
    }
}

#[test]
fn qwen_config_normalizes_with_openai_clip_stats() {
    let generic = QwenVlImageProcessorConfig::default().image_processor_config();

    assert!(generic.do_rescale);
    assert!(generic.do_normalize);
    assert_eq!(generic.image_mean, CLIP_IMAGE_MEAN.to_vec());
    assert_eq!(generic.image_std, CLIP_IMAGE_STD.to_vec());
}

#[test]
fn idefics3_config_normalizes_with_standard_unit_stats() {
    let generic = Idefics3ImageProcessorConfig::default().image_processor_config();

    assert!(generic.do_rescale);
    assert!(generic.do_normalize);
    assert_eq!(generic.image_mean, STANDARD_IMAGE_MEAN.to_vec());
    assert_eq!(generic.image_std, STANDARD_IMAGE_STD.to_vec());
}

#[test]
fn gemma3_config_normalizes_with_standard_unit_stats() {
    let generic = Gemma3ImageProcessorConfig::default().image_processor_config();

    assert!(generic.do_rescale);
    assert!(generic.do_normalize);
    assert_eq!(generic.image_mean, STANDARD_IMAGE_MEAN.to_vec());
    assert_eq!(generic.image_std, STANDARD_IMAGE_STD.to_vec());
}

#[test]
fn mllama_config_normalizes_with_standard_unit_stats() {
    let generic = MllamaImageProcessorConfig::default().image_processor_config();

    assert!(generic.do_rescale);
    assert!(generic.do_normalize);
    assert_eq!(generic.image_mean, STANDARD_IMAGE_MEAN.to_vec());
    assert_eq!(generic.image_std, STANDARD_IMAGE_STD.to_vec());
}

#[test]
fn sam_config_normalizes_raw_pixels_without_rescale() {
    let generic = SamImageProcessorConfig::default().image_processor_config();

    assert!(!generic.do_rescale);
    assert!(generic.do_normalize);
    assert_eq!(generic.image_mean, SAM_IMAGE_MEAN.to_vec());
    assert_eq!(generic.image_std, SAM_IMAGE_STD.to_vec());
}

#[test]
fn fixed_family_configs_build_processors() {
    BlipImageProcessor::new(BlipImageProcessorConfig::default()).unwrap();
    ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();
    VitImageProcessor::new(VitImageProcessorConfig::default()).unwrap();
    DetrImageProcessor::new(DetrImageProcessorConfig::default()).unwrap();
    SamImageProcessor::new(SamImageProcessorConfig::default()).unwrap();
    DocumentOcrImageProcessor::new(DocumentOcrImageProcessorConfig::default()).unwrap();
    VideoMaeImageProcessor::new(VideoMaeImageProcessorConfig::default()).unwrap();
    VivitImageProcessor::new(VivitImageProcessorConfig::default()).unwrap();
    LlavaNextImageProcessor::new(LlavaNextImageProcessorConfig::default()).unwrap();
    PixtralImageProcessor::new(PixtralImageProcessorConfig::default()).unwrap();
    Idefics3ImageProcessor::new(Idefics3ImageProcessorConfig::default()).unwrap();
    Gemma3ImageProcessor::new(Gemma3ImageProcessorConfig::default()).unwrap();
    MllamaImageProcessor::new(MllamaImageProcessorConfig::default()).unwrap();
    Flux2ImageProcessor::new(Flux2ImageProcessorConfig::default()).unwrap();
    VisualClozeProcessor::new(VisualClozeProcessorConfig::default()).unwrap();
    HunyuanVideo15ImageProcessor::new(HunyuanVideo15ImageProcessorConfig::default()).unwrap();
    MarigoldImageProcessor::new(MarigoldImageProcessorConfig::default()).unwrap();
    JoyImageEditImageProcessor::new(JoyImageEditImageProcessorConfig::default()).unwrap();
    WanAnimateImageProcessor::new(WanAnimateImageProcessorConfig::default()).unwrap();
}
