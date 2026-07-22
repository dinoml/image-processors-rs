use super::*;

#[test]
fn default_size_rounds_down_to_scale_factor() {
    assert_eq!(
        ImageSize::new(513, 777)
            .unwrap()
            .round_down_to_multiple(8)
            .unwrap(),
        ImageSize {
            height: 512,
            width: 776,
        }
    );
}

#[test]
fn smart_resize_keeps_factor_multiple() {
    let size = ImageSize::new(333, 777)
        .unwrap()
        .smart_resize(ResizeLimits {
            factor: 28,
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
        })
        .unwrap();

    assert_eq!(size.height % 28, 0);
    assert_eq!(size.width % 28, 0);
}

#[test]
fn smart_resize_rejects_unachievable_pixel_limits() {
    let err = ImageSize::new(100, 100)
        .unwrap()
        .smart_resize(ResizeLimits {
            factor: 28,
            min_pixels: 1,
            max_pixels: 28 * 28 - 1,
        })
        .unwrap_err();

    assert_eq!(err, TransformError::InvalidSmartResizeParams);
}

#[test]
fn prepare_coco_detection_annotation_filters_and_clips_boxes() {
    let target = prepare_coco_detection_annotation(
        42,
        ImageSize::new(50, 50).unwrap(),
        &[
            CocoObjectAnnotation {
                category_id: 7,
                bbox: [-10.0, 5.0, 30.0, 20.0],
                area: 600.0,
                iscrowd: false,
            },
            CocoObjectAnnotation {
                category_id: 8,
                bbox: [40.0, 40.0, 20.0, 20.0],
                area: 400.0,
                iscrowd: false,
            },
            CocoObjectAnnotation {
                category_id: 9,
                bbox: [60.0, 10.0, 5.0, 5.0],
                area: 25.0,
                iscrowd: false,
            },
            CocoObjectAnnotation {
                category_id: 10,
                bbox: [1.0, 1.0, 2.0, 2.0],
                area: 4.0,
                iscrowd: true,
            },
        ],
    )
    .unwrap();

    assert_eq!(target.image_id, 42);
    assert_eq!(target.original_size, ImageSize::new(50, 50).unwrap());
    assert_eq!(target.size, ImageSize::new(50, 50).unwrap());
    assert_eq!(target.class_labels, vec![7, 8]);
    assert_eq!(
        target.boxes,
        vec![
            DetectionBoundingBox::new(0.0, 5.0, 20.0, 25.0).unwrap(),
            DetectionBoundingBox::new(40.0, 40.0, 50.0, 50.0).unwrap(),
        ]
    );
    assert_eq!(target.area, vec![600.0, 400.0]);
    assert_eq!(target.iscrowd, vec![false, false]);
}

#[test]
fn resize_detection_annotation_scales_boxes_and_area() {
    let target = prepare_coco_detection_annotation(
        1,
        ImageSize::new(100, 200).unwrap(),
        &[CocoObjectAnnotation {
            category_id: 3,
            bbox: [10.0, 20.0, 100.0, 50.0],
            area: 5000.0,
            iscrowd: false,
        }],
    )
    .unwrap();

    let resized = resize_detection_annotation(&target, ImageSize::new(50, 100).unwrap()).unwrap();

    assert_eq!(resized.size, ImageSize::new(50, 100).unwrap());
    assert_eq!(
        resized.boxes,
        vec![DetectionBoundingBox::new(5.0, 10.0, 55.0, 35.0).unwrap()]
    );
    assert_eq!(resized.area, vec![1250.0]);
}

#[test]
fn normalize_detection_annotation_converts_boxes_to_relative_center_format() {
    let target = DetectionAnnotation {
        image_id: 1,
        original_size: ImageSize::new(100, 200).unwrap(),
        size: ImageSize::new(100, 200).unwrap(),
        class_labels: vec![3],
        boxes: vec![DetectionBoundingBox::new(10.0, 20.0, 110.0, 70.0).unwrap()],
        area: vec![5000.0],
        iscrowd: vec![false],
    };

    let normalized = normalize_detection_annotation(&target).unwrap();

    assert_center_box_close(
        normalized.boxes[0],
        DetectionCenterBox::new(0.3, 0.45, 0.5, 0.5).unwrap(),
    );
}

#[test]
fn pad_normalized_detection_annotation_scales_boxes_to_padded_canvas() {
    let normalized = NormalizedDetectionAnnotation {
        image_id: 1,
        original_size: ImageSize::new(100, 200).unwrap(),
        size: ImageSize::new(100, 200).unwrap(),
        class_labels: vec![3],
        boxes: vec![DetectionCenterBox::new(0.3, 0.45, 0.5, 0.5).unwrap()],
        area: vec![5000.0],
        iscrowd: vec![false],
    };

    let padded =
        pad_normalized_detection_annotation(&normalized, ImageSize::new(100, 400).unwrap())
            .unwrap();

    assert_eq!(padded.size, ImageSize::new(100, 400).unwrap());
    assert_center_box_close(
        padded.boxes[0],
        DetectionCenterBox::new(0.15, 0.45, 0.25, 0.5).unwrap(),
    );
}

#[test]
fn normalize_detection_annotation_rejects_inconsistent_lengths() {
    let target = DetectionAnnotation {
        image_id: 1,
        original_size: ImageSize::new(10, 10).unwrap(),
        size: ImageSize::new(10, 10).unwrap(),
        class_labels: vec![1],
        boxes: Vec::new(),
        area: vec![1.0],
        iscrowd: vec![false],
    };

    let err = normalize_detection_annotation(&target).unwrap_err();

    assert_eq!(
        err,
        TransformError::InconsistentDetectionAnnotationLengths {
            class_labels: 1,
            boxes: 0,
            area: 1,
            iscrowd: 1,
        }
    );
}

#[test]
fn prepare_coco_detection_annotation_rejects_non_finite_boxes() {
    let err = prepare_coco_detection_annotation(
        1,
        ImageSize::new(10, 10).unwrap(),
        &[CocoObjectAnnotation {
            category_id: 1,
            bbox: [0.0, f32::NAN, 1.0, 1.0],
            area: 1.0,
            iscrowd: false,
        }],
    )
    .unwrap_err();

    assert!(matches!(err, TransformError::InvalidBoundingBox { .. }));
}

#[test]
fn post_process_object_detection_applies_softmax_threshold_and_box_scaling() {
    let logits = vec![4.0, 1.0, 0.0, 0.0, 0.0, 4.0];
    let boxes = vec![
        DetectionCenterBox::new(0.5, 0.5, 0.4, 0.2).unwrap(),
        DetectionCenterBox::new(0.2, 0.2, 0.1, 0.1).unwrap(),
    ];

    let predictions =
        post_process_object_detection(&logits, &boxes, 3, ImageSize::new(100, 200).unwrap(), 0.5)
            .unwrap();

    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].class_label, 0);
    assert!((predictions[0].score - 0.936_239_54).abs() < 1e-6);
    assert_detection_box_close(
        predictions[0].bbox,
        DetectionBoundingBox::new(60.0, 40.0, 140.0, 60.0).unwrap(),
    );
}

#[test]
fn post_process_object_detection_rejects_missing_background_class() {
    let err = post_process_object_detection(
        &[1.0],
        &[DetectionCenterBox::new(0.5, 0.5, 0.25, 0.25).unwrap()],
        1,
        ImageSize::new(10, 10).unwrap(),
        0.5,
    )
    .unwrap_err();

    assert_eq!(err, TransformError::InvalidClassCount(1));
}

#[test]
fn post_process_semantic_segmentation_combines_masks_and_resizes_scores() {
    let class_logits = vec![10.0, 0.0, -10.0, 0.0, 10.0, -10.0];
    let mask_logits = vec![10.0, -10.0, -10.0, 10.0];

    let prediction = post_process_semantic_segmentation(
        &class_logits,
        &mask_logits,
        2,
        3,
        ImageSize::new(1, 2).unwrap(),
        Some(ImageSize::new(2, 2).unwrap()),
    )
    .unwrap();

    assert_eq!(prediction.size, ImageSize::new(2, 2).unwrap());
    assert_eq!(prediction.num_labels, 2);
    assert_eq!(prediction.class_ids, vec![0, 1, 0, 1]);
    assert_eq!(prediction.scores.len(), 8);
}

#[test]
fn post_process_semantic_segmentation_rejects_inconsistent_mask_shape() {
    let err = post_process_semantic_segmentation(
        &[1.0, 0.0, -1.0],
        &[0.0],
        1,
        3,
        ImageSize::new(1, 2).unwrap(),
        None,
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidBufferLength {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn post_process_instance_segmentation_assigns_weighted_mask_segments() {
    let class_logits = vec![8.0, 0.0, -8.0, 0.0, 8.0, -8.0];
    let mask_logits = vec![8.0, 8.0, -8.0, -8.0, -8.0, -8.0, 8.0, 8.0];

    let prediction = post_process_instance_segmentation(
        &class_logits,
        &mask_logits,
        2,
        3,
        ImageSize::new(2, 2).unwrap(),
        SegmentationPostProcessOptions::default(),
    )
    .unwrap();

    assert_eq!(prediction.size, ImageSize::new(2, 2).unwrap());
    assert_eq!(prediction.segmentation, vec![1, 1, 2, 2]);
    assert_eq!(prediction.segments.len(), 2);
    assert_eq!(prediction.segments[0].id, 1);
    assert_eq!(prediction.segments[0].label_id, 0);
    assert!(!prediction.segments[0].was_fused);
    assert!((prediction.segments[0].score - 0.999_665).abs() < 1e-6);
    assert_eq!(prediction.segments[1].id, 2);
    assert_eq!(prediction.segments[1].label_id, 1);
    assert!(!prediction.segments[1].was_fused);
}

#[test]
fn post_process_panoptic_segmentation_fuses_matching_label_ids() {
    let class_logits = vec![0.0, 0.0, 8.0, -8.0, 0.0, 0.0, 8.0, -8.0];
    let mask_logits = vec![8.0, -8.0, 8.0, -8.0, -8.0, 8.0, -8.0, 8.0];

    let prediction = post_process_panoptic_segmentation(
        &class_logits,
        &mask_logits,
        2,
        4,
        ImageSize::new(2, 2).unwrap(),
        &[2],
        SegmentationPostProcessOptions::default(),
    )
    .unwrap();

    assert_eq!(prediction.segmentation, vec![1, 1, 1, 1]);
    assert_eq!(prediction.segments.len(), 2);
    assert!(prediction
        .segments
        .iter()
        .all(|segment| segment.id == 1 && segment.label_id == 2 && segment.was_fused));
}

#[test]
fn post_process_instance_segmentation_returns_negative_one_when_all_queries_are_background() {
    let prediction = post_process_instance_segmentation(
        &[0.0, 0.0, 8.0],
        &[0.0, 0.0, 0.0, 0.0],
        1,
        3,
        ImageSize::new(2, 2).unwrap(),
        SegmentationPostProcessOptions {
            target_size: Some(ImageSize::new(1, 3).unwrap()),
            ..SegmentationPostProcessOptions::default()
        },
    )
    .unwrap();

    assert_eq!(prediction.size, ImageSize::new(1, 3).unwrap());
    assert_eq!(prediction.segmentation, vec![-1, -1, -1]);
    assert!(prediction.segments.is_empty());
}

#[test]
fn post_process_instance_segmentation_rejects_inconsistent_mask_shape() {
    let err = post_process_instance_segmentation(
        &[1.0, 0.0, -1.0],
        &[0.0],
        1,
        3,
        ImageSize::new(2, 2).unwrap(),
        SegmentationPostProcessOptions::default(),
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::InvalidBufferLength {
            expected: 4,
            actual: 1,
        }
    );
}

#[test]
fn non_max_suppression_keeps_descending_scores_below_iou_threshold() {
    let boxes = vec![
        DetectionBoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
        DetectionBoundingBox::new(1.0, 1.0, 11.0, 11.0).unwrap(),
        DetectionBoundingBox::new(20.0, 20.0, 30.0, 30.0).unwrap(),
    ];
    let iou = detection_box_iou(boxes[0], boxes[1]).unwrap();

    let keep = non_max_suppression(&boxes, &[0.9, 0.8, 0.7], 0.5).unwrap();

    assert!((iou - 0.680_672_3).abs() < 1e-6);
    assert_eq!(keep, vec![0, 2]);
}

#[test]
fn scale_image_point_and_box_use_resized_input_ratios() {
    let original = ImageSize::new(300, 500).unwrap();
    let resized = ImageSize::new(614, 1024).unwrap();
    let point =
        scale_image_point(ImagePoint::new(250.0, 150.0).unwrap(), original, resized).unwrap();
    let bbox = scale_detection_box(
        DetectionBoundingBox::new(50.0, 60.0, 150.0, 180.0).unwrap(),
        original,
        resized,
    )
    .unwrap();

    assert_point_close(point, ImagePoint::new(512.0, 307.0).unwrap());
    assert_detection_box_close(
        bbox,
        DetectionBoundingBox::new(102.4, 122.8, 307.2, 368.4).unwrap(),
    );
}

#[test]
fn normalized_point_grid_places_points_at_cell_centers() {
    let points = normalized_point_grid(2).unwrap();

    assert_eq!(
        points,
        vec![
            ImagePoint::new(0.25, 0.25).unwrap(),
            ImagePoint::new(0.75, 0.25).unwrap(),
            ImagePoint::new(0.25, 0.75).unwrap(),
            ImagePoint::new(0.75, 0.75).unwrap(),
        ]
    );
}

#[test]
fn generate_layered_crop_boxes_matches_layer_overlap_geometry() {
    let boxes = generate_layered_crop_boxes(
        ImageSize::new(100, 200).unwrap(),
        CropGenerationOptions {
            crop_layers: 1,
            overlap_ratio: 0.2,
        },
    )
    .unwrap();

    assert_eq!(
        boxes,
        vec![
            LayeredCropBox {
                x_min: 0,
                y_min: 0,
                x_max: 200,
                y_max: 100,
                layer: 0,
            },
            LayeredCropBox {
                x_min: 0,
                y_min: 0,
                x_max: 110,
                y_max: 60,
                layer: 1,
            },
            LayeredCropBox {
                x_min: 0,
                y_min: 40,
                x_max: 110,
                y_max: 100,
                layer: 1,
            },
            LayeredCropBox {
                x_min: 90,
                y_min: 0,
                x_max: 200,
                y_max: 60,
                layer: 1,
            },
            LayeredCropBox {
                x_min: 90,
                y_min: 40,
                x_max: 200,
                y_max: 100,
                layer: 1,
            },
        ]
    );
}

#[test]
fn binary_mask_to_box_returns_bounds_for_non_empty_masks() {
    let mask = vec![
        false, false, false, false, false, true, false, true, false, false, false, true,
    ];

    let bbox = binary_mask_to_box(&mask, ImageSize::new(3, 4).unwrap()).unwrap();

    assert_eq!(
        bbox,
        Some(DetectionBoundingBox::new(1.0, 1.0, 3.0, 2.0).unwrap())
    );
}

#[test]
fn binary_mask_to_rle_roundtrips_column_major_counts() {
    let size = ImageSize::new(2, 3).unwrap();
    let mask = vec![false, true, false, true, true, false];

    let rle = binary_mask_to_rle(&mask, size).unwrap();
    let decoded = binary_rle_to_mask(&rle).unwrap();

    assert_eq!(rle.counts, vec![1, 3, 2]);
    assert_eq!(decoded, mask);
}

#[test]
fn binarize_mask_and_stability_score_use_strict_thresholds() {
    let values = vec![-1.0, 0.2, 0.7, 1.4];

    let mask = binarize_mask(&values, 0.5).unwrap();
    let stability = mask_stability_score(&values, 0.5, 0.5).unwrap();

    assert_eq!(mask, vec![false, false, true, true]);
    assert!((stability - (1.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn post_process_binary_mask_removes_padding_before_resizing_to_original() {
    let mask = post_process_binary_mask(
        &[1.0, 3.0, 5.0, 7.0],
        ImageSize::new(2, 2).unwrap(),
        ImageSize::new(2, 2).unwrap(),
        ImageSize::new(1, 2).unwrap(),
        ImageSize::new(2, 2).unwrap(),
        2.0,
    )
    .unwrap();

    assert_eq!(mask, vec![false, true, false, true]);
}

#[test]
fn resize_padded_mask_logits_rejects_crop_larger_than_pad_size() {
    let err = resize_padded_mask_logits(
        &[0.0],
        ImageSize::new(1, 1).unwrap(),
        ImageSize::new(2, 2).unwrap(),
        ImageSize::new(3, 2).unwrap(),
        ImageSize::new(3, 2).unwrap(),
    )
    .unwrap_err();

    assert_eq!(
        err,
        TransformError::CropTooLarge {
            source_size: ImageSize::new(2, 2).unwrap(),
            target_size: ImageSize::new(3, 2).unwrap(),
        }
    );
}

#[test]
fn box_near_crop_edge_ignores_original_image_edges() {
    let original = ImageSize::new(6, 6).unwrap();
    let near_inner_crop = box_near_crop_edge(
        DetectionBoundingBox::new(0.0, 1.0, 1.0, 2.0).unwrap(),
        LayeredCropBox {
            x_min: 2,
            y_min: 2,
            x_max: 5,
            y_max: 5,
            layer: 1,
        },
        original,
        0.0,
    )
    .unwrap();
    let near_original_edge = box_near_crop_edge(
        DetectionBoundingBox::new(0.0, 1.0, 1.0, 2.0).unwrap(),
        LayeredCropBox {
            x_min: 0,
            y_min: 0,
            x_max: 6,
            y_max: 6,
            layer: 0,
        },
        original,
        0.0,
    )
    .unwrap();

    assert!(near_inner_crop);
    assert!(!near_original_edge);
}

#[test]
fn pad_crop_mask_places_crop_pixels_on_original_canvas() {
    let mut expected = vec![false; 20];
    expected[7] = true;
    expected[13] = true;

    let padded = pad_crop_mask(
        &[true, false, false, true],
        ImageSize::new(2, 2).unwrap(),
        LayeredCropBox {
            x_min: 2,
            y_min: 1,
            x_max: 4,
            y_max: 3,
            layer: 1,
        },
        ImageSize::new(4, 5).unwrap(),
    )
    .unwrap();

    assert_eq!(padded, expected);
}

#[test]
fn filter_generated_masks_applies_score_threshold_and_crop_edge_filter() {
    let kept_mask = [
        -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
    ];
    let edge_mask = [
        2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0,
    ];
    let low_score_mask = [2.0; 16];
    let mut logits = Vec::new();
    logits.extend_from_slice(&kept_mask);
    logits.extend_from_slice(&edge_mask);
    logits.extend_from_slice(&low_score_mask);

    let filtered = filter_generated_masks(
        &logits,
        ImageSize::new(4, 4).unwrap(),
        &[0.9, 0.95, 0.1],
        LayeredCropBox {
            x_min: 1,
            y_min: 1,
            x_max: 5,
            y_max: 5,
            layer: 1,
        },
        ImageSize::new(6, 6).unwrap(),
        MaskFilterOptions {
            pred_iou_threshold: 0.5,
            stability_score_threshold: 0.0,
            crop_edge_tolerance: 0.0,
            ..MaskFilterOptions::default()
        },
    )
    .unwrap();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].score, 0.9);
    assert_eq!(filtered[0].stability_score, 1.0);
    assert_eq!(
        filtered[0].crop_bbox,
        DetectionBoundingBox::new(1.0, 1.0, 2.0, 2.0).unwrap()
    );
    assert_eq!(
        filtered[0].bbox,
        DetectionBoundingBox::new(2.0, 2.0, 3.0, 3.0).unwrap()
    );
    assert_eq!(filtered[0].rle.size, ImageSize::new(6, 6).unwrap());
}

#[test]
fn post_process_generated_masks_applies_nms_and_decodes_rles() {
    let masks = [
        vec![true, false, false, false],
        vec![false, true, false, false],
        vec![false, false, false, true],
    ];
    let rles = masks
        .iter()
        .map(|mask| binary_mask_to_rle(mask, ImageSize::new(2, 2).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let boxes = vec![
        DetectionBoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
        DetectionBoundingBox::new(1.0, 1.0, 11.0, 11.0).unwrap(),
        DetectionBoundingBox::new(20.0, 20.0, 30.0, 30.0).unwrap(),
    ];

    let predictions = post_process_generated_masks(&rles, &[0.9, 0.8, 0.7], &boxes, 0.5).unwrap();

    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].score, 0.9);
    assert_eq!(predictions[0].mask, masks[0]);
    assert_eq!(predictions[1].score, 0.7);
    assert_eq!(predictions[1].mask, masks[2]);
}
