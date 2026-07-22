use image_processors::postprocess::{
    binary_mask_to_rle, binary_rle_to_mask, post_process_coordinates, post_process_dense_map,
    post_process_depth_map, post_process_image_tensor, post_process_logc3_hdr_video_tensor,
    post_process_object_detection, post_process_recipe_outputs, post_process_video_tensor,
    CoordinatePoint, DetectionBoundingBox, DetectionCenterBox, ImageSize, RecipePostprocessContext,
    RecipePostprocessError, RecipePostprocessOutput,
};
use image_processors::{
    DetrImageProcessorConfig, Layout, PixelFormat, ProcessorMetadataName, ProcessorMetadataValue,
    ProcessorOutput, ProcessorRecipe, ProcessorRecipePostprocess, ProcessorTensorName,
    RecipeCoordinatePostprocess, RecipeCoordinateTask, RecipeDenseMapPostprocess,
    RecipeDenseMapTask, RecipeDepthPostprocess, RecipeDepthUnit, RecipeImageSizeSource,
    RecipeOutputHookPostprocess, RecipeOutputHookTask, RecipeSegmentationPostprocess,
    RecipeSegmentationTask, RecipeTokenSequenceDecoder, RecipeTokenSequencePostprocess,
    RecipeTokenSequenceTask, RecipeTokenVocabularySource, RecipeVideoTensorPostprocess,
    RecipeVideoTransferFunction, SamImageProcessorConfig, TaskVisionImageProcessor,
    TaskVisionImageProcessorConfig, TaskVisionProcessorPreset, Tensor, TensorData,
    TensorLeadingAxis,
};

fn segmentation_recipe(segmentation: RecipeSegmentationPostprocess) -> ProcessorRecipe {
    DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::Segmentation(segmentation)])
        .expect("segmentation descriptor should be valid")
}

fn segmentation_model_outputs(
    class_logits: Vec<f32>,
    class_shape: Vec<usize>,
    mask_logits: Vec<f32>,
    mask_shape: Vec<usize>,
) -> ProcessorOutput {
    segmentation_model_outputs_named(
        "class_logits",
        class_logits,
        class_shape,
        "mask_logits",
        mask_logits,
        mask_shape,
    )
}

fn segmentation_model_outputs_named(
    class_output: &str,
    class_logits: Vec<f32>,
    class_shape: Vec<usize>,
    mask_output: &str,
    mask_logits: Vec<f32>,
    mask_shape: Vec<usize>,
) -> ProcessorOutput {
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other(class_output),
        Tensor::new(
            TensorData::F32(class_logits),
            class_shape.clone(),
            test_layout(class_shape.len()),
        )
        .expect("class-logits tensor should be valid"),
    );
    model_outputs.insert_tensor(
        ProcessorTensorName::other(mask_output),
        Tensor::new(
            TensorData::F32(mask_logits),
            mask_shape.clone(),
            test_layout(mask_shape.len()),
        )
        .expect("mask-logits tensor should be valid"),
    );
    model_outputs
}

fn test_layout(rank: usize) -> Layout {
    match rank {
        2 => Layout::NC,
        3 => Layout::CHW,
        4 => Layout::NCHW,
        other => panic!("test tensors do not use rank {other}"),
    }
}

#[test]
fn postprocess_module_restores_channel_first_image_tensor() {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0, 1.0, 0.25, 0.5, 0.75, 1.0]),
        vec![1, 3, 1, 2],
        Layout::NCHW,
    )
    .expect("image tensor should be valid");

    let frames = post_process_image_tensor(&tensor).expect("image tensor should restore");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].pixel_format(), PixelFormat::Rgb8);
    assert_eq!(frames[0].data(), &[0, 64, 191, 255, 128, 255]);
}

#[test]
fn postprocess_module_restores_channel_last_video_tensor() {
    let tensor = Tensor::new(
        TensorData::F32(vec![
            0.0, 0.5, 1.0, 1.0, 0.0, 0.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0, 2.0, 0.0, 0.75,
            0.125, 0.875, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ]),
        vec![2, 2, 1, 2, 3],
        Layout::BFHWC,
    )
    .expect("video tensor should be valid");

    let clips = post_process_video_tensor(&tensor).expect("video tensor should restore");

    assert_eq!(clips.len(), 2);
    assert_eq!(clips[0].len(), 2);
    assert_eq!(clips[1].len(), 2);
    assert_eq!(clips[0][0].pixel_format(), PixelFormat::Rgb8);
    assert_eq!(clips[0][0].data(), &[0, 128, 255, 255, 0, 64]);
    assert_eq!(clips[1][0].data(), &[0, 255, 0, 191, 32, 223]);
}

#[test]
fn postprocess_module_restores_logc3_hdr_video_tensor() {
    let tensor = Tensor::new(
        TensorData::F32(vec![-0.814_382, 0.0, 1.0, -1.0, -0.700_684_3, -0.814_382]),
        vec![1, 2, 3, 1, 1],
        Layout::BFCHW,
    )
    .expect("HDR video tensor should be valid");

    let output =
        post_process_logc3_hdr_video_tensor(&tensor).expect("HDR video tensor should restore");

    assert_eq!(output.layout(), Layout::BFHWC);
    assert_eq!(output.shape(), [1, 2, 1, 1, 3]);
    assert_f32_slice_close(
        &output.data().to_vec::<f32>(),
        &[
            0.0,
            0.513_383_4,
            55.079_56,
            -0.017_290_42,
            0.010_591_047,
            0.0,
        ],
        1e-5,
    );
}

#[test]
fn postprocess_module_keeps_frame_axis_as_one_video_clip() {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.25]),
        vec![2, 3, 1, 1],
        Layout::NCHW,
    )
    .expect("frame tensor should be valid")
    .with_leading_axis(TensorLeadingAxis::Frames)
    .expect("NCHW tensors support explicit frame leading axis");

    let clips = post_process_video_tensor(&tensor).expect("frame tensor should restore");

    assert_eq!(clips.len(), 1);
    assert_eq!(clips[0].len(), 2);
    assert_eq!(clips[0][0].data(), &[0, 128, 255]);
    assert_eq!(clips[0][1].data(), &[255, 0, 64]);
}

fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} mismatch: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn postprocess_module_restores_object_detection_boxes() {
    let logits = [0.0, 4.0, -4.0];
    let boxes =
        [DetectionCenterBox::new(0.5, 0.5, 0.2, 0.4).expect("normalized box should be valid")];
    let target_size = ImageSize::new(10, 20).expect("target size should be valid");

    let predictions = post_process_object_detection(&logits, &boxes, 3, target_size, 0.8)
        .expect("postprocess should restore the foreground prediction");

    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].class_label, 1);
    assert_eq!(
        predictions[0].bbox,
        DetectionBoundingBox::new(8.0, 3.0, 12.0, 7.0)
            .expect("expected restored box should be valid")
    );
}

#[test]
fn postprocess_module_roundtrips_binary_rle_masks() {
    let size = ImageSize::new(2, 3).expect("mask size should be valid");
    let mask = [false, true, true, false, true, false];

    let rle = binary_mask_to_rle(&mask, size).expect("mask should encode as RLE");
    let decoded = binary_rle_to_mask(&rle).expect("RLE should decode");

    assert_eq!(decoded, mask);
}

#[test]
fn postprocess_module_restores_depth_map_size_and_unit() {
    let source_size = ImageSize::new(1, 2).expect("source size should be valid");
    let target_size = ImageSize::new(2, 2).expect("target size should be valid");

    let output = post_process_depth_map(
        &[1.0, 3.0],
        source_size,
        RecipeDepthUnit::Relative,
        Some(target_size),
    )
    .expect("depth map should resize");

    assert_eq!(output.size(), target_size);
    assert_eq!(output.unit(), RecipeDepthUnit::Relative);
    assert_eq!(output.values(), &[1.0, 3.0, 1.0, 3.0]);
}

#[test]
fn postprocess_module_restores_dense_map_channels_and_size() {
    let source_size = ImageSize::new(1, 2).expect("source size should be valid");
    let target_size = ImageSize::new(2, 2).expect("target size should be valid");

    let output = post_process_dense_map(
        &[1.0, 10.0, 100.0, 3.0, 30.0, 300.0],
        source_size,
        3,
        RecipeDenseMapTask::SurfaceNormal,
        Some(target_size),
    )
    .expect("dense map should resize");

    assert_eq!(output.task(), RecipeDenseMapTask::SurfaceNormal);
    assert_eq!(output.target_name(), None);
    assert_eq!(output.size(), target_size);
    assert_eq!(output.channels(), 3);
    assert_eq!(
        output.values(),
        &[1.0, 10.0, 100.0, 3.0, 30.0, 300.0, 1.0, 10.0, 100.0, 3.0, 30.0, 300.0,]
    );
}

#[test]
fn postprocess_module_restores_normalized_coordinates() {
    let target_size = ImageSize::new(10, 20).expect("target size should be valid");

    let output = post_process_coordinates(
        &[0.25, 0.5, 1.0, 0.0],
        Some(&[0.9, 0.4]),
        RecipeCoordinateTask::Keypoints,
        target_size,
    )
    .expect("coordinates should restore");

    assert_eq!(output.task(), RecipeCoordinateTask::Keypoints);
    assert_eq!(output.size(), target_size);
    assert_eq!(
        output.points(),
        &[
            CoordinatePoint::new(5.0, 5.0).expect("point should be valid"),
            CoordinatePoint::new(20.0, 0.0).expect("point should be valid")
        ]
    );
    assert_eq!(output.scores(), Some([0.9, 0.4].as_slice()));
}

#[test]
fn recipe_postprocess_executes_detr_descriptor_from_named_tensors() {
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("logits"),
        Tensor::new(
            TensorData::F32(vec![
                5.0, 1.0, 0.0, //
                0.0, 0.0, 6.0, //
                1.0, 4.0, 0.0, //
                0.1, 0.2, 0.0,
            ]),
            vec![2, 2, 3],
            Layout::CHW,
        )
        .expect("logits tensor should be valid"),
    );
    model_outputs.insert_tensor(
        ProcessorTensorName::other("pred_boxes"),
        Tensor::new(
            TensorData::F32(vec![
                0.5, 0.5, 0.2, 0.4, //
                0.1, 0.1, 0.1, 0.1, //
                0.25, 0.75, 0.5, 0.25, //
                0.2, 0.2, 0.1, 0.1,
            ]),
            vec![2, 2, 4],
            Layout::CHW,
        )
        .expect("box tensor should be valid"),
    );
    let target_sizes = [
        ImageSize::new(10, 20).expect("target size should be valid"),
        ImageSize::new(8, 12).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("recipe descriptor should execute");

    let [RecipePostprocessOutput::ObjectDetection { predictions, .. }] = outputs.as_slice() else {
        panic!("expected one object-detection recipe output, got {outputs:?}");
    };
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].len(), 1);
    assert_eq!(predictions[0][0].class_label, 0);
    assert_eq!(
        predictions[0][0].bbox,
        DetectionBoundingBox::new(8.0, 3.0, 12.0, 7.0)
            .expect("expected restored box should be valid")
    );
    assert_eq!(predictions[1].len(), 1);
    assert_eq!(predictions[1][0].class_label, 1);
    assert_eq!(
        predictions[1][0].bbox,
        DetectionBoundingBox::new(0.0, 5.0, 6.0, 7.0)
            .expect("expected restored box should be valid")
    );
}

#[test]
fn recipe_postprocess_executes_depth_descriptor_from_named_tensor() {
    let descriptor = RecipeDepthPostprocess::new(
        "predicted_depth",
        RecipeDepthUnit::Meters,
        Some(RecipeImageSizeSource::CallerProvided),
        true,
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::Depth(descriptor)])
        .expect("depth descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("predicted_depth"),
        Tensor::new(
            TensorData::F32(vec![1.0, 3.0, 2.0, 4.0]),
            vec![2, 1, 2],
            Layout::CHW,
        )
        .expect("depth tensor should be valid"),
    );
    let target_sizes = [
        ImageSize::new(2, 2).expect("target size should be valid"),
        ImageSize::new(2, 2).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("depth recipe descriptor should execute");

    let [RecipePostprocessOutput::Depth { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one depth recipe output, got {outputs:?}");
    };
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].unit(), RecipeDepthUnit::Meters);
    assert_eq!(outputs[0].values(), &[1.0, 3.0, 1.0, 3.0]);
    assert_eq!(outputs[1].values(), &[2.0, 4.0, 2.0, 4.0]);
}

#[test]
fn recipe_postprocess_executes_dense_map_descriptor_from_named_tensor() {
    let descriptor = RecipeDenseMapPostprocess::new(
        RecipeDenseMapTask::SurfaceNormal,
        "predicted_normals",
        Some(RecipeImageSizeSource::CallerProvided),
        true,
    )
    .with_channels(3);
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::DenseMap(descriptor)])
        .expect("dense-map descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("predicted_normals"),
        Tensor::new(
            TensorData::F32(vec![
                1.0, 3.0, 10.0, 30.0, 100.0, 300.0, //
                2.0, 4.0, 20.0, 40.0, 200.0, 400.0,
            ]),
            vec![2, 3, 1, 2],
            Layout::NCHW,
        )
        .expect("dense-map tensor should be valid"),
    );
    let target_sizes = [
        ImageSize::new(2, 2).expect("target size should be valid"),
        ImageSize::new(2, 2).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("dense-map recipe descriptor should execute");

    let [RecipePostprocessOutput::DenseMap { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one dense-map recipe output, got {outputs:?}");
    };
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].task(), RecipeDenseMapTask::SurfaceNormal);
    assert_eq!(outputs[0].target_name(), None);
    assert_eq!(outputs[0].channels(), 3);
    assert_eq!(
        outputs[0].values(),
        &[1.0, 10.0, 100.0, 3.0, 30.0, 300.0, 1.0, 10.0, 100.0, 3.0, 30.0, 300.0,]
    );
    assert_eq!(
        outputs[1].values(),
        &[2.0, 20.0, 200.0, 4.0, 40.0, 400.0, 2.0, 20.0, 200.0, 4.0, 40.0, 400.0,]
    );
}

#[test]
fn recipe_postprocess_executes_dense_map_descriptor_with_target_names() {
    let descriptor = RecipeDenseMapPostprocess::new(
        RecipeDenseMapTask::IntrinsicImage,
        "intrinsics",
        Some(RecipeImageSizeSource::CallerProvided),
        true,
    )
    .with_channels(3)
    .with_target_names(vec!["albedo".to_string(), "shading".to_string()]);
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::DenseMap(descriptor)])
        .expect("dense-map descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("intrinsics"),
        Tensor::new(
            TensorData::F32(vec![
                1.0, 3.0, 10.0, 30.0, 100.0, 300.0, //
                2.0, 4.0, 20.0, 40.0, 200.0, 400.0, //
                5.0, 7.0, 50.0, 70.0, 500.0, 700.0, //
                6.0, 8.0, 60.0, 80.0, 600.0, 800.0,
            ]),
            vec![4, 3, 1, 2],
            Layout::NCHW,
        )
        .expect("dense-map tensor should be valid"),
    );
    let target_sizes = [
        ImageSize::new(2, 2).expect("target size should be valid"),
        ImageSize::new(1, 2).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("dense-map recipe descriptor should execute");

    let [RecipePostprocessOutput::DenseMap { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one dense-map recipe output, got {outputs:?}");
    };
    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].target_name(), Some("albedo"));
    assert_eq!(outputs[1].target_name(), Some("shading"));
    assert_eq!(outputs[2].target_name(), Some("albedo"));
    assert_eq!(outputs[3].target_name(), Some("shading"));
    assert_eq!(outputs[0].size(), target_sizes[0]);
    assert_eq!(outputs[1].size(), target_sizes[0]);
    assert_eq!(outputs[2].size(), target_sizes[1]);
    assert_eq!(outputs[3].size(), target_sizes[1]);
    assert_eq!(
        outputs[0].values(),
        &[1.0, 10.0, 100.0, 3.0, 30.0, 300.0, 1.0, 10.0, 100.0, 3.0, 30.0, 300.0,]
    );
    assert_eq!(outputs[2].values(), &[5.0, 50.0, 500.0, 7.0, 70.0, 700.0]);
}

#[test]
fn recipe_postprocess_executes_logc3_video_tensor_descriptor() {
    let descriptor =
        RecipeVideoTensorPostprocess::new("decoded_video", RecipeVideoTransferFunction::LogC3);
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::VideoTensor(descriptor)])
        .expect("video-tensor descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("decoded_video"),
        Tensor::new(
            TensorData::F32(vec![-0.814_382, 0.0, 1.0, -1.0, -0.700_684_3, -0.814_382]),
            vec![1, 2, 3, 1, 1],
            Layout::BFCHW,
        )
        .expect("decoded video tensor should be valid"),
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("video-tensor recipe descriptor should execute");

    let [RecipePostprocessOutput::VideoTensor {
        output_name,
        transfer,
        tensor,
        ..
    }] = outputs.as_slice()
    else {
        panic!("expected one video-tensor recipe output, got {outputs:?}");
    };
    assert_eq!(output_name, "decoded_video");
    assert_eq!(*transfer, RecipeVideoTransferFunction::LogC3);
    assert_eq!(tensor.layout(), Layout::BFHWC);
    assert_eq!(tensor.shape(), [1, 2, 1, 1, 3]);
    assert_f32_slice_close(
        &tensor.data().to_vec::<f32>(),
        &[
            0.0,
            0.513_383_4,
            55.079_56,
            -0.017_290_42,
            0.010_591_047,
            0.0,
        ],
        1e-5,
    );
}

#[test]
fn recipe_postprocess_rejects_video_tensor_descriptor_with_image_layout() {
    let descriptor =
        RecipeVideoTensorPostprocess::new("decoded_image", RecipeVideoTransferFunction::LogC3);
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::VideoTensor(descriptor)])
        .expect("video-tensor descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("decoded_image"),
        Tensor::new(
            TensorData::F32(vec![0.0, 1.0, -1.0]),
            vec![1, 1, 3],
            Layout::HWC,
        )
        .expect("decoded image tensor should be valid"),
    );

    let err = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .unwrap_err();

    assert_eq!(
        err,
        RecipePostprocessError::UnsupportedTensorLayout {
            output: "decoded_image".to_string(),
            expected: "BFCHW or BFHWC",
            actual: Layout::HWC,
        }
    );
}

#[test]
fn recipe_postprocess_executes_coordinate_descriptor_from_named_tensors() {
    let descriptor = RecipeCoordinatePostprocess::new(
        RecipeCoordinateTask::Pose,
        "keypoints",
        Some("scores".to_string()),
        RecipeImageSizeSource::CallerProvided,
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::Coordinates(descriptor)])
        .expect("coordinate descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("keypoints"),
        Tensor::new(
            TensorData::F32(vec![
                0.25, 0.5, 1.0, 0.0, //
                0.5, 1.0, 0.0, 0.25,
            ]),
            vec![2, 2, 2],
            Layout::CHW,
        )
        .expect("coordinate tensor should be valid"),
    );
    model_outputs.insert_tensor(
        ProcessorTensorName::other("scores"),
        Tensor::new(
            TensorData::F32(vec![0.9, 0.1, 0.8, 0.2]),
            vec![2, 2],
            Layout::NC,
        )
        .expect("score tensor should be valid"),
    );
    let target_sizes = [
        ImageSize::new(10, 20).expect("target size should be valid"),
        ImageSize::new(4, 8).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("coordinate recipe descriptor should execute");

    let [RecipePostprocessOutput::Coordinates { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one coordinate recipe output, got {outputs:?}");
    };
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].task(), RecipeCoordinateTask::Pose);
    assert_eq!(
        outputs[0].points(),
        &[
            CoordinatePoint::new(5.0, 5.0).expect("point should be valid"),
            CoordinatePoint::new(20.0, 0.0).expect("point should be valid")
        ]
    );
    assert_eq!(outputs[0].scores(), Some([0.9, 0.1].as_slice()));
    assert_eq!(
        outputs[1].points(),
        &[
            CoordinatePoint::new(4.0, 4.0).expect("point should be valid"),
            CoordinatePoint::new(0.0, 1.0).expect("point should be valid")
        ]
    );
    assert_eq!(outputs[1].scores(), Some([0.8, 0.2].as_slice()));
}

#[test]
fn recipe_postprocess_executes_output_hook_descriptor_from_named_tensor() {
    let descriptor = RecipeOutputHookPostprocess::new(
        RecipeOutputHookTask::Layout,
        "layout_logits",
        Some(RecipeImageSizeSource::CallerProvided),
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::OutputHook(descriptor)])
        .expect("output-hook descriptor should be valid");
    let tensor = Tensor::new(
        TensorData::F32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        vec![2, 1, 3],
        Layout::CHW,
    )
    .expect("output-hook tensor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(ProcessorTensorName::other("layout_logits"), tensor.clone());
    let target_sizes = [
        ImageSize::new(10, 20).expect("target size should be valid"),
        ImageSize::new(30, 40).expect("target size should be valid"),
    ];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("output-hook recipe descriptor should execute");

    let [RecipePostprocessOutput::OutputHook { output, .. }] = outputs.as_slice() else {
        panic!("expected one output-hook recipe output, got {outputs:?}");
    };
    assert_eq!(output.task(), RecipeOutputHookTask::Layout);
    assert_eq!(output.output_name(), "layout_logits");
    assert_eq!(output.tensor(), &tensor);
    assert_eq!(output.target_sizes(), Some(target_sizes.as_slice()));
}

#[test]
fn recipe_postprocess_decodes_ctc_token_ids_without_claiming_text_decoding() {
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::Ocr,
        "logits",
        RecipeTokenSequenceDecoder::CtcGreedy { blank_token_id: 0 },
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::TokenSequence(descriptor)])
        .expect("token-sequence descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("logits"),
        Tensor::new(
            TensorData::F32(vec![
                0.0, 4.0, 1.0, 2.0, // token 1
                0.0, 3.0, 2.0, 1.0, // repeated token 1
                5.0, 0.0, 0.0, 0.0, // blank
                0.0, 1.0, 6.0, 2.0, // token 2
                0.0, 1.0, 7.0, 2.0, // repeated token 2
            ]),
            vec![1, 5, 4],
            Layout::CHW,
        )
        .expect("token logits should be valid"),
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("CTC token sequence should decode");

    let [RecipePostprocessOutput::TokenSequences { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one token-sequence recipe output, got {outputs:?}");
    };
    assert_eq!(outputs[0].task(), RecipeTokenSequenceTask::Ocr);
    assert_eq!(outputs[0].token_ids(), &[1, 2]);
    assert_eq!(outputs[0].scores(), &[4.0, 6.0]);
    assert_eq!(outputs[0].mean_score(), Some(5.0));
    assert_eq!(
        outputs[0].vocabulary_source(),
        RecipeTokenVocabularySource::External
    );
}

#[test]
fn recipe_postprocess_decodes_table_tokens_until_end_token() {
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::TableStructure,
        "structure_logits",
        RecipeTokenSequenceDecoder::Greedy {
            begin_token_id: Some(0),
            end_token_id: Some(4),
        },
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::TokenSequence(descriptor)])
        .expect("token-sequence descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("structure_logits"),
        Tensor::new(
            TensorData::F32(vec![
                5.0, 0.0, 0.0, 0.0, 0.0, // begin
                0.0, 0.0, 4.0, 0.0, 0.0, // token 2
                0.0, 0.0, 0.0, 3.0, 0.0, // token 3
                0.0, 0.0, 0.0, 0.0, 6.0, // end
                0.0, 9.0, 0.0, 0.0, 0.0, // ignored after end
            ]),
            vec![1, 5, 5],
            Layout::CHW,
        )
        .expect("table logits should be valid"),
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("table token sequence should decode");

    let [RecipePostprocessOutput::TokenSequences { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one token-sequence recipe output, got {outputs:?}");
    };
    assert_eq!(outputs[0].task(), RecipeTokenSequenceTask::TableStructure);
    assert_eq!(outputs[0].token_ids(), &[2, 3]);
    assert_eq!(outputs[0].scores(), &[4.0, 3.0]);
}

#[test]
fn recipe_postprocess_decodes_document_tokens_with_first_argmax_tie() {
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::DocumentText,
        "logits",
        RecipeTokenSequenceDecoder::Greedy {
            begin_token_id: Some(0),
            end_token_id: Some(2),
        },
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::TokenSequence(descriptor)])
        .expect("token-sequence descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("logits"),
        Tensor::new(
            TensorData::F32(vec![0.0, 5.0, 5.0, 0.0, 1.0, 6.0]),
            vec![1, 2, 3],
            Layout::CHW,
        )
        .expect("document logits should be valid"),
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("document token sequence should decode");

    let [RecipePostprocessOutput::TokenSequences { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one token-sequence recipe output, got {outputs:?}");
    };
    assert_eq!(outputs[0].task(), RecipeTokenSequenceTask::DocumentText);
    assert_eq!(outputs[0].token_ids(), &[1]);
    assert_eq!(outputs[0].scores(), &[5.0]);
}

#[test]
fn recipe_token_sequence_rejects_non_finite_logits() {
    let descriptor = RecipeTokenSequencePostprocess::new(
        RecipeTokenSequenceTask::DocumentText,
        "logits",
        RecipeTokenSequenceDecoder::Greedy {
            begin_token_id: None,
            end_token_id: None,
        },
    );
    let recipe = DetrImageProcessorConfig::default()
        .processor_recipe()
        .expect("DETR recipe should be valid")
        .with_postprocess(vec![ProcessorRecipePostprocess::TokenSequence(descriptor)])
        .expect("token-sequence descriptor should be valid");
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("logits"),
        Tensor::new(
            TensorData::F32(vec![0.0, f32::NAN]),
            vec![1, 1, 2],
            Layout::CHW,
        )
        .expect("token logits should be structurally valid"),
    );

    let error = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect_err("non-finite token logits must be rejected");

    assert!(matches!(
        error,
        RecipePostprocessError::NonFiniteTensorValue {
            output,
            index: 1,
            value,
        } if output == "logits" && value.is_nan()
    ));
}

#[test]
fn recipe_postprocess_executes_binary_mask_descriptor_from_preprocess_metadata() {
    let image_size = ImageSize::new(2, 2).expect("SAM image size should be valid");
    let recipe = SamImageProcessorConfig {
        image_size,
        ..Default::default()
    }
    .processor_recipe()
    .expect("SAM recipe should be valid");
    let mut preprocessing_output = ProcessorOutput::from_pixel_values(
        Tensor::new(
            TensorData::F32(vec![0.0; 2 * 3 * 2 * 2]),
            vec![2, 3, 2, 2],
            Layout::NCHW,
        )
        .expect("preprocess pixel tensor should be valid"),
    );
    let sizes = vec![image_size, image_size];
    preprocessing_output.insert_metadata(
        ProcessorMetadataName::OriginalSizes,
        ProcessorMetadataValue::ImageSizes(sizes.clone()),
    );
    preprocessing_output.insert_metadata(
        ProcessorMetadataName::ReshapedInputSizes,
        ProcessorMetadataValue::ImageSizes(sizes),
    );
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(
        ProcessorTensorName::other("pred_masks"),
        Tensor::new(
            TensorData::F32(vec![
                1.0, 3.0, 5.0, 7.0, //
                0.0, 0.0, 0.0, 0.0, //
                3.0, 1.0, 3.0, 1.0, //
                1.0, 3.0, 1.0, 3.0,
            ]),
            vec![2, 2, 2, 2],
            Layout::NCHW,
        )
        .expect("binary mask tensor should be valid"),
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &preprocessing_output,
        RecipePostprocessContext::new(),
    )
    .expect("binary mask recipe descriptor should execute");

    let [RecipePostprocessOutput::BinaryMasks { outputs, .. }] = outputs.as_slice() else {
        panic!("expected one binary mask recipe output, got {outputs:?}");
    };
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].masks(),
        &[
            vec![true, true, true, true],
            vec![false, false, false, false]
        ]
    );
    assert_eq!(
        outputs[1].masks(),
        &[vec![true, true, true, true], vec![true, true, true, true]]
    );
}

#[test]
fn recipe_postprocess_executes_semantic_segmentation_descriptor() {
    let mut descriptor = RecipeSegmentationPostprocess::new(
        RecipeSegmentationTask::Semantic,
        "class_logits",
        "mask_logits",
        Some(3),
    );
    descriptor.mask_size = Some(ImageSize::new(1, 2).expect("mask size should be valid"));
    descriptor.options.target_size_source = Some(RecipeImageSizeSource::CallerProvided);
    let recipe = segmentation_recipe(descriptor);
    let model_outputs = segmentation_model_outputs(
        vec![10.0, 0.0, -10.0, 0.0, 10.0, -10.0],
        vec![2, 3],
        vec![10.0, -10.0, -10.0, 10.0],
        vec![2, 1, 2],
    );
    let target_sizes = [ImageSize::new(2, 2).expect("target size should be valid")];

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new().with_target_sizes(&target_sizes),
    )
    .expect("semantic segmentation descriptor should execute");

    let [RecipePostprocessOutput::SemanticSegmentation { predictions, .. }] = outputs.as_slice()
    else {
        panic!("expected one semantic segmentation recipe output, got {outputs:?}");
    };
    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].size, target_sizes[0]);
    assert_eq!(predictions[0].class_ids, vec![0, 1, 0, 1]);
}

#[test]
fn recipe_postprocess_executes_instance_segmentation_descriptor() {
    let mut descriptor = RecipeSegmentationPostprocess::new(
        RecipeSegmentationTask::Instance,
        "class_logits",
        "mask_logits",
        Some(3),
    );
    descriptor.mask_size = Some(ImageSize::new(2, 2).expect("mask size should be valid"));
    let recipe = segmentation_recipe(descriptor);
    let model_outputs = segmentation_model_outputs(
        vec![8.0, 0.0, -8.0, 0.0, 8.0, -8.0],
        vec![2, 3],
        vec![8.0, 8.0, -8.0, -8.0, -8.0, -8.0, 8.0, 8.0],
        vec![2, 2, 2],
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("instance segmentation descriptor should execute");

    let [RecipePostprocessOutput::InstanceSegmentation { predictions, .. }] = outputs.as_slice()
    else {
        panic!("expected one instance segmentation recipe output, got {outputs:?}");
    };
    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].segmentation, vec![1, 1, 2, 2]);
    assert_eq!(predictions[0].segments.len(), 2);
    assert_eq!(predictions[0].segments[0].id, 1);
    assert_eq!(predictions[0].segments[1].id, 2);
}

#[test]
fn recipe_postprocess_executes_panoptic_segmentation_descriptor() {
    let mut descriptor = RecipeSegmentationPostprocess::new(
        RecipeSegmentationTask::Panoptic,
        "class_logits",
        "mask_logits",
        Some(4),
    );
    descriptor.mask_size = Some(ImageSize::new(2, 2).expect("mask size should be valid"));
    descriptor.label_ids_to_fuse = vec![2];
    let recipe = segmentation_recipe(descriptor);
    let model_outputs = segmentation_model_outputs(
        vec![0.0, 0.0, 8.0, -8.0, 0.0, 0.0, 8.0, -8.0],
        vec![2, 4],
        vec![8.0, -8.0, 8.0, -8.0, -8.0, 8.0, -8.0, 8.0],
        vec![2, 2, 2],
    );

    let outputs = post_process_recipe_outputs(
        &recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("panoptic segmentation descriptor should execute");

    let [RecipePostprocessOutput::PanopticSegmentation { predictions, .. }] = outputs.as_slice()
    else {
        panic!("expected one panoptic segmentation recipe output, got {outputs:?}");
    };
    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].segmentation, vec![1, 1, 1, 1]);
    assert!(predictions[0]
        .segments
        .iter()
        .all(|segment| segment.id == 1 && segment.label_id == 2 && segment.was_fused));
}

#[test]
fn mask2former_wrapper_executes_instance_segmentation_task() {
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::Mask2Former,
    ))
    .expect("Mask2Former preset should build");
    let model_outputs = segmentation_model_outputs_named(
        "class_queries_logits",
        vec![8.0, 0.0, -8.0, -8.0, 0.0, 8.0, -8.0, -8.0],
        vec![2, 4],
        "masks_queries_logits",
        vec![8.0, 8.0, -8.0, -8.0, -8.0, -8.0, 8.0, 8.0],
        vec![2, 2, 2],
    );
    let target_sizes = [ImageSize::new(2, 2).expect("target size should be valid")];

    let outputs = processor
        .post_process_segmentation_outputs(
            RecipeSegmentationTask::Instance,
            &model_outputs,
            &ProcessorOutput::new(),
            RecipePostprocessContext::new().with_target_sizes(&target_sizes),
        )
        .expect("Mask2Former instance output should restore");

    let [RecipePostprocessOutput::InstanceSegmentation { predictions, .. }] = outputs.as_slice()
    else {
        panic!("expected Mask2Former instance output, got {outputs:?}");
    };
    assert_eq!(predictions[0].segmentation, vec![0, 0, 1, 1]);
    assert_eq!(predictions[0].segments.len(), 2);
    assert_eq!(predictions[0].segments[0].id, 0);
    assert_eq!(predictions[0].segments[1].id, 1);
}

#[test]
fn oneformer_wrapper_executes_panoptic_segmentation_task() {
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::OneFormer,
    ))
    .expect("OneFormer preset should build");
    let model_outputs = segmentation_model_outputs_named(
        "class_queries_logits",
        vec![0.0, 0.0, 8.0, -8.0, 0.0, 0.0, 8.0, -8.0],
        vec![2, 4],
        "masks_queries_logits",
        vec![8.0, -8.0, 8.0, -8.0, -8.0, 8.0, -8.0, 8.0],
        vec![2, 2, 2],
    );
    let target_sizes = [ImageSize::new(2, 2).expect("target size should be valid")];

    let outputs = processor
        .post_process_segmentation_outputs(
            RecipeSegmentationTask::Panoptic,
            &model_outputs,
            &ProcessorOutput::new(),
            RecipePostprocessContext::new().with_target_sizes(&target_sizes),
        )
        .expect("OneFormer panoptic output should restore");

    let [RecipePostprocessOutput::PanopticSegmentation { predictions, .. }] = outputs.as_slice()
    else {
        panic!("expected OneFormer panoptic output, got {outputs:?}");
    };
    assert_eq!(predictions[0].segmentation, vec![1, 2, 1, 2]);
    assert_eq!(predictions[0].segments.len(), 2);
}

#[test]
fn query_mask_segmentation_wrapper_rejects_non_query_mask_presets() {
    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::Sam2,
    ))
    .expect("SAM2 preset should build");

    let error = processor
        .post_process_segmentation_outputs(
            RecipeSegmentationTask::Instance,
            &ProcessorOutput::new(),
            &ProcessorOutput::new(),
            RecipePostprocessContext::new(),
        )
        .expect_err("SAM2 does not use the query-mask segmentation contract");

    assert!(matches!(
        error,
        RecipePostprocessError::UnsupportedDescriptor {
            task: "query_mask_segmentation"
        }
    ));
}

#[test]
fn query_mask_presets_expose_only_tasks_supported_without_external_metadata() {
    let target_sizes = [ImageSize::new(2, 2).expect("target size should be valid")];
    for preset in [
        TaskVisionProcessorPreset::Eomt,
        TaskVisionProcessorPreset::Mask2Former,
        TaskVisionProcessorPreset::MaskFormer,
    ] {
        let processor =
            TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(preset))
                .unwrap_or_else(|error| panic!("{} should build: {error}", preset.class_name()));
        let model_outputs = segmentation_model_outputs_named(
            "class_queries_logits",
            vec![8.0, 0.0, -8.0, -8.0, 0.0, 8.0, -8.0, -8.0],
            vec![2, 4],
            "masks_queries_logits",
            vec![8.0, 8.0, -8.0, -8.0, -8.0, -8.0, 8.0, 8.0],
            vec![2, 2, 2],
        );

        for task in [
            RecipeSegmentationTask::Instance,
            RecipeSegmentationTask::Panoptic,
        ] {
            let outputs = processor
                .post_process_segmentation_outputs(
                    task,
                    &model_outputs,
                    &ProcessorOutput::new(),
                    RecipePostprocessContext::new().with_target_sizes(&target_sizes),
                )
                .unwrap_or_else(|error| {
                    panic!("{} {task:?} should restore: {error}", preset.class_name())
                });
            assert!(
                matches!(
                    (task, outputs.as_slice()),
                    (
                        RecipeSegmentationTask::Instance,
                        [RecipePostprocessOutput::InstanceSegmentation { .. }]
                    ) | (
                        RecipeSegmentationTask::Panoptic,
                        [RecipePostprocessOutput::PanopticSegmentation { .. }]
                    )
                ),
                "{} returned the wrong {task:?} output: {outputs:?}",
                preset.class_name()
            );
        }
    }

    let processor = TaskVisionImageProcessor::new(TaskVisionImageProcessorConfig::for_preset(
        TaskVisionProcessorPreset::OneFormer,
    ))
    .expect("OneFormer should build");
    let model_outputs = segmentation_model_outputs_named(
        "class_queries_logits",
        vec![8.0, 0.0, -8.0, -8.0, 0.0, 8.0, -8.0, -8.0],
        vec![2, 4],
        "masks_queries_logits",
        vec![8.0, 8.0, -8.0, -8.0, -8.0, -8.0, 8.0, 8.0],
        vec![2, 2, 2],
    );
    let error = processor
        .post_process_segmentation_outputs(
            RecipeSegmentationTask::Instance,
            &model_outputs,
            &ProcessorOutput::new(),
            RecipePostprocessContext::new().with_target_sizes(&target_sizes),
        )
        .expect_err("OneFormer instance output needs dataset class metadata");
    assert!(matches!(
        error,
        RecipePostprocessError::ExternalMetadataRequired {
            strategy: "one_former",
            task: "instance_segmentation",
            ..
        }
    ));
}
