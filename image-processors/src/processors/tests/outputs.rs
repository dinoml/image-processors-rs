use super::*;

#[test]
fn family_configs_default_to_image_decode_and_build_batch_execution() {
    let blip = BlipImageProcessorConfig::default().image_processor_config();
    assert_eq!(blip.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(blip.batch_execution, BatchExecution::default());

    let clip = ClipImageProcessorConfig::default().image_processor_config();
    assert_eq!(clip.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(clip.batch_execution, BatchExecution::default());

    let qwen = QwenVlImageProcessorConfig::default().image_processor_config();
    assert_eq!(qwen.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(qwen.batch_execution, BatchExecution::default());

    let pixtral = PixtralImageProcessorConfig::default().image_processor_config();
    assert_eq!(pixtral.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(pixtral.batch_execution, BatchExecution::default());

    let idefics3 = Idefics3ImageProcessorConfig::default().image_processor_config();
    assert_eq!(idefics3.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(idefics3.batch_execution, BatchExecution::default());

    let gemma3 = Gemma3ImageProcessorConfig::default().image_processor_config();
    assert_eq!(gemma3.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(gemma3.batch_execution, BatchExecution::default());

    let mllama = MllamaImageProcessorConfig::default().image_processor_config();
    assert_eq!(mllama.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(mllama.batch_execution, BatchExecution::default());

    let sam = SamImageProcessorConfig::default().image_processor_config();
    assert_eq!(sam.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(sam.batch_execution, BatchExecution::default());

    let joy = JoyImageEditImageProcessorConfig::default().image_processor_config();
    assert_eq!(joy.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(joy.batch_execution, BatchExecution::default());

    let flux2 = Flux2ImageProcessorConfig::default().image_processor_config();
    assert_eq!(flux2.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(flux2.batch_execution, BatchExecution::default());

    let visual_cloze = VisualClozeProcessorConfig::default().image_processor_config();
    assert_eq!(visual_cloze.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(visual_cloze.batch_execution, BatchExecution::default());

    let hunyuan = HunyuanVideo15ImageProcessorConfig::default().image_processor_config();
    assert_eq!(hunyuan.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(hunyuan.batch_execution, BatchExecution::default());

    let marigold = MarigoldImageProcessorConfig::default().image_processor_config();
    assert_eq!(marigold.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(marigold.batch_execution, BatchExecution::default());

    let wan = WanAnimateImageProcessorConfig::default().image_processor_config();
    assert_eq!(wan.decode_backend, ImageDecodeBackend::ImageCrate);
    assert_eq!(wan.batch_execution, BatchExecution::default());
}

#[test]
fn family_configs_propagate_decode_backend_and_batch_execution_to_generic_config() {
    let config = ClipImageProcessorConfig {
        decode_backend: ImageDecodeBackend::TurboJpeg,
        batch_execution: BatchExecution::Parallel,
        ..Default::default()
    };

    assert_eq!(
        config.image_processor_config().decode_backend,
        ImageDecodeBackend::TurboJpeg
    );
    assert_eq!(
        config.image_processor_config().batch_execution,
        BatchExecution::Parallel
    );
}

#[test]
fn clip_processor_preprocesses_to_default_channels_first_shape() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();

    let tensor = processor.preprocess_image(&frame).unwrap();

    assert_eq!(tensor.shape(), [1, 3, 224, 224]);
    assert_eq!(tensor.layout(), Layout::NCHW);
}

#[test]
fn clip_processor_output_contains_pixel_values() {
    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default()).unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [1, 3, 224, 224]);
}

#[test]
fn shortest_edge_resize_config_applies_longest_edge_cap() {
    let tiny = ImageSize {
        height: 17,
        width: 13,
    };
    assert_eq!(
        shortest_edge_resize_output_size(tiny, ShortestEdgeResizeConfig::default()).unwrap(),
        ImageSize {
            height: 1046,
            width: 800,
        }
    );

    let wide = ImageSize {
        height: 800,
        width: 2000,
    };
    assert_eq!(
        shortest_edge_resize_output_size(wide, ShortestEdgeResizeConfig::default()).unwrap(),
        ImageSize {
            height: 533,
            width: 1332,
        }
    );
}

#[test]
fn detr_processor_output_contains_valid_pixel_mask_and_sizes() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![255; 2 * 3]).unwrap();
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig {
        image_size: ImageSize {
            height: 4,
            width: 4,
        },
        resize_size: ShortestEdgeResizeConfig {
            shortest_edge: 2,
            longest_edge: None,
        },
        pad_size: Some(ImageSize {
            height: 4,
            width: 4,
        }),
        ..Default::default()
    })
    .unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();
    let mask = output.pixel_mask().unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [1, 3, 4, 4]);
    assert_eq!(mask.shape(), [1, 1, 4, 4]);
    assert_eq!(
        mask.data().to_vec::<bool>(),
        vec![
            true, true, true, true, true, true, true, true, false, false, false, false, false,
            false, false, false,
        ]
    );
    assert_eq!(
        output.original_sizes().unwrap(),
        &[ImageSize {
            height: 1,
            width: 2,
        }]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[ImageSize {
            height: 2,
            width: 4,
        }]
    );
}

#[test]
fn detr_processor_batch_output_pads_to_max_with_size_divisor() {
    let wide = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![255; 4 * 2 * 3]).unwrap();
    let tall = ImageFrame::new(2, 4, PixelFormat::Rgb8, vec![127; 2 * 4 * 3]).unwrap();
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig {
        image_size: ImageSize {
            height: 8,
            width: 8,
        },
        resize_size: ShortestEdgeResizeConfig {
            shortest_edge: 4,
            longest_edge: None,
        },
        pad_to_multiple: Some(6),
        ..Default::default()
    })
    .unwrap();

    let output = processor.preprocess_images_output(&[wide, tall]).unwrap();
    let mask = output.pixel_mask().unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [2, 3, 12, 12]);
    assert_eq!(mask.shape(), [2, 1, 12, 12]);
    assert_eq!(
        mask.data().to_vec::<bool>(),
        expected_pixel_mask(
            &[
                ImageSize {
                    height: 4,
                    width: 8,
                },
                ImageSize {
                    height: 8,
                    width: 4,
                },
            ],
            ImageSize {
                height: 12,
                width: 12,
            }
        )
    );
    assert_eq!(
        output.original_sizes().unwrap(),
        &[
            ImageSize {
                height: 2,
                width: 4,
            },
            ImageSize {
                height: 4,
                width: 2,
            },
        ]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[
            ImageSize {
                height: 4,
                width: 8,
            },
            ImageSize {
                height: 8,
                width: 4,
            },
        ]
    );
}

#[test]
fn detr_processor_post_process_object_detection_slices_batched_outputs() {
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig::default()).unwrap();
    let logits = [
        5.0, 1.0, 0.0, //
        0.0, 0.0, 6.0, //
        1.0, 4.0, 0.0, //
        0.1, 0.2, 0.0,
    ];
    let boxes = [
        DetectionCenterBox::new(0.5, 0.5, 0.2, 0.4).unwrap(),
        DetectionCenterBox::new(0.1, 0.1, 0.1, 0.1).unwrap(),
        DetectionCenterBox::new(0.25, 0.75, 0.5, 0.25).unwrap(),
        DetectionCenterBox::new(0.2, 0.2, 0.1, 0.1).unwrap(),
    ];
    let targets = [
        ImageSize {
            height: 10,
            width: 20,
        },
        ImageSize {
            height: 8,
            width: 12,
        },
    ];

    let predictions = processor
        .post_process_object_detection(&logits, &boxes, 3, &targets, 0.8)
        .unwrap();

    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].len(), 1);
    assert_eq!(predictions[1].len(), 1);
    assert_eq!(predictions[0][0].class_label, 0);
    assert!(predictions[0][0].score > 0.9);
    assert_bbox_close(
        predictions[0][0].bbox,
        crate::transforms::DetectionBoundingBox::new(8.0, 3.0, 12.0, 7.0).unwrap(),
    );
    assert_eq!(predictions[1][0].class_label, 1);
    assert!(predictions[1][0].score > 0.9);
    assert_bbox_close(
        predictions[1][0].bbox,
        crate::transforms::DetectionBoundingBox::new(0.0, 5.0, 6.0, 7.0).unwrap(),
    );
}

#[test]
fn detr_processor_post_process_object_detection_rejects_misaligned_box_batch() {
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig::default()).unwrap();
    let boxes = [
        DetectionCenterBox::new(0.5, 0.5, 0.2, 0.4).unwrap(),
        DetectionCenterBox::new(0.1, 0.1, 0.1, 0.1).unwrap(),
        DetectionCenterBox::new(0.25, 0.75, 0.5, 0.25).unwrap(),
    ];
    let targets = [
        ImageSize {
            height: 10,
            width: 20,
        },
        ImageSize {
            height: 8,
            width: 12,
        },
    ];

    let err = processor
        .post_process_object_detection(&[], &boxes, 3, &targets, 0.8)
        .unwrap_err();

    assert!(matches!(err, ImageProcessorError::IncompatibleBatchShapes));
}

#[test]
fn detr_processor_rejects_pad_size_smaller_than_resized_image() {
    let frame = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![255; 4 * 2 * 3]).unwrap();
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig {
        image_size: ImageSize {
            height: 8,
            width: 8,
        },
        resize_size: ShortestEdgeResizeConfig {
            shortest_edge: 4,
            longest_edge: None,
        },
        pad_size: Some(ImageSize {
            height: 4,
            width: 4,
        }),
        ..Default::default()
    })
    .unwrap();

    let err = processor.preprocess_image_output(&frame).unwrap_err();

    match err {
        ImageProcessorError::InvalidPaddingTarget {
            target_size,
            image_size,
        } => {
            assert_eq!(
                target_size,
                ImageSize {
                    height: 4,
                    width: 4,
                }
            );
            assert_eq!(
                image_size,
                ImageSize {
                    height: 4,
                    width: 8,
                }
            );
        }
        other => panic!("expected invalid padding target error, got {other:?}"),
    }
}

#[test]
fn pixtral_processor_output_aligns_to_patch_grid_and_pads_batch() {
    let first = ImageFrame::new(3, 3, PixelFormat::Rgb8, vec![7; 3 * 3 * 3]).unwrap();
    let second = ImageFrame::new(2, 4, PixelFormat::Rgb8, vec![9; 2 * 4 * 3]).unwrap();
    let processor = pixtral_small_patch_processor();

    let output = processor
        .preprocess_images_output(&[first, second])
        .unwrap();
    let tensor = output.pixel_values().unwrap();
    let values = tensor.data().to_vec::<f32>();

    assert_eq!(tensor.shape(), [2, 3, 4, 4]);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(
        output.original_sizes().unwrap(),
        &[
            ImageSize {
                height: 3,
                width: 3
            },
            ImageSize {
                height: 4,
                width: 2
            },
        ]
    );
    assert_eq!(
        output.reshaped_input_sizes().unwrap(),
        &[
            ImageSize {
                height: 4,
                width: 4
            },
            ImageSize {
                height: 4,
                width: 2
            },
        ]
    );
    assert_eq!(output.image_grid_thw().unwrap(), &[[1, 2, 2], [1, 2, 1]]);
    assert_eq!(output.image_patch_counts().unwrap(), &[4, 2]);

    let second_batch_offset = 3 * 4 * 4;
    for channel in 0..3 {
        for row in 0..4 {
            let row_start = second_batch_offset + channel * 4 * 4 + row * 4;
            assert_eq!(&values[row_start..row_start + 2], &[9.0, 9.0]);
            assert_eq!(&values[row_start + 2..row_start + 4], &[0.0, 0.0]);
        }
    }
}

#[test]
fn sam_processor_output_contains_sizes() {
    let frame = ImageFrame::new(4, 2, PixelFormat::Rgb8, vec![127; 4 * 2 * 3]).unwrap();
    let processor = SamImageProcessor::new(SamImageProcessorConfig {
        image_size: ImageSize {
            height: 8,
            width: 8,
        },
        ..Default::default()
    })
    .unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();

    assert_eq!(output.pixel_values().unwrap().shape(), [1, 3, 8, 8]);
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
            height: 4,
            width: 8,
        }]
    );
}

#[test]
fn sam_processor_zero_pads_after_normalization_for_nhwc_output() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![0; 2 * 3]).unwrap();
    let processor = SamImageProcessor::new(SamImageProcessorConfig {
        image_size: ImageSize {
            height: 4,
            width: 4,
        },
        output_layout: ImageLayout::HeightWidthChannels,
        ..Default::default()
    })
    .unwrap();

    let output = processor.preprocess_image_output(&frame).unwrap();
    let tensor = output.pixel_values().unwrap();
    let values = tensor.data().to_vec::<f32>();

    assert_eq!(tensor.shape(), [1, 4, 4, 3]);
    assert_eq!(tensor.layout(), Layout::NHWC);
    assert!((values[0] + SAM_IMAGE_MEAN[0] / SAM_IMAGE_STD[0]).abs() < 1e-6);
    assert!((values[1] + SAM_IMAGE_MEAN[1] / SAM_IMAGE_STD[1]).abs() < 1e-6);
    assert!((values[2] + SAM_IMAGE_MEAN[2] / SAM_IMAGE_STD[2]).abs() < 1e-6);
    assert!(
        values[2 * 4 * 3..]
            .iter()
            .all(|value| value.abs() < f32::EPSILON),
        "SAM padded tensor region should stay zero after normalization"
    );
}

#[test]
fn sam_processor_post_process_masks_restores_batched_logits() {
    let processor = SamImageProcessor::new(SamImageProcessorConfig {
        image_size: ImageSize {
            height: 2,
            width: 2,
        },
        ..Default::default()
    })
    .unwrap();
    let logits = [
        1.0, 3.0, 5.0, 7.0, //
        0.0, 0.0, 0.0, 0.0, //
        3.0, 1.0, 3.0, 1.0, //
        1.0, 3.0, 1.0, 3.0,
    ];
    let original_sizes = [
        ImageSize {
            height: 2,
            width: 2,
        },
        ImageSize {
            height: 2,
            width: 2,
        },
    ];
    let reshaped_sizes = [
        ImageSize {
            height: 1,
            width: 2,
        },
        ImageSize {
            height: 2,
            width: 2,
        },
    ];

    let outputs = processor
        .post_process_masks(
            &logits,
            ImageSize {
                height: 2,
                width: 2,
            },
            2,
            &original_sizes,
            &reshaped_sizes,
            2.0,
        )
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].size(), original_sizes[0]);
    assert_eq!(
        outputs[0].masks(),
        &[
            vec![false, true, false, true],
            vec![false, false, false, false],
        ]
    );
    assert_eq!(outputs[1].size(), original_sizes[1]);
    assert_eq!(
        outputs[1].masks(),
        &[
            vec![true, false, true, false],
            vec![false, true, false, true],
        ]
    );
}

#[test]
fn sam_processor_post_process_masks_rejects_invalid_flattened_length() {
    let processor = SamImageProcessor::new(SamImageProcessorConfig {
        image_size: ImageSize {
            height: 2,
            width: 2,
        },
        ..Default::default()
    })
    .unwrap();
    let sizes = [ImageSize {
        height: 2,
        width: 2,
    }];

    let err = processor
        .post_process_masks(
            &[1.0, 2.0, 3.0],
            ImageSize {
                height: 2,
                width: 2,
            },
            1,
            &sizes,
            &sizes,
            0.0,
        )
        .unwrap_err();

    assert!(matches!(
        err,
        ImageProcessorError::Transform(TransformError::InvalidBufferLength {
            expected: 4,
            actual: 3,
        })
    ));
}
