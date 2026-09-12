use image_processors::{
    ImageFrame, ImageSize, PixelFormat, ProcessorRecipeStage, RecipeResizeTarget,
    TaskVisionImageProcessor, TaskVisionImageProcessorConfig, TaskVisionProcessorPreset,
};

#[test]
fn yolos_preserves_capped_scale_before_rounding_both_axes_down() {
    let config = TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::Yolos);
    let processor = TaskVisionImageProcessor::new(config).expect("YOLOS processor");
    for (height, width, expected_height, expected_width) in [
        (480, 640, 800, 1056),
        (640, 480, 1056, 800),
        (800, 2000, 528, 1328),
    ] {
        let frame = ImageFrame::new(
            width,
            height,
            PixelFormat::Rgb8,
            vec![127; width * height * 3],
        )
        .expect("source image");
        let output = processor
            .preprocess_image_output(&frame)
            .expect("YOLOS output");
        assert_eq!(
            output.reshaped_input_sizes().expect("resized size"),
            &[ImageSize::new(expected_height, expected_width).expect("expected size")],
        );
        assert_eq!(
            output.pixel_values().expect("pixels").shape(),
            [1, 3, expected_height, expected_width],
        );
    }
}

#[test]
fn yolos_recipe_describes_one_resample_after_axis_rounding() {
    let config = TaskVisionImageProcessorConfig::for_preset(TaskVisionProcessorPreset::Yolos);
    let recipe = config.processor_recipe().expect("YOLOS recipe");
    let targets = recipe
        .stages()
        .iter()
        .filter_map(|stage| match stage {
            ProcessorRecipeStage::Resize { resize } => Some(resize.target),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        targets,
        [RecipeResizeTarget::ShortestEdgeRoundDown {
            shortest_edge: 800,
            longest_edge: Some(1333),
            multiple: 16,
        }]
    );
    assert!(!recipe
        .stages()
        .iter()
        .any(|stage| matches!(stage, ProcessorRecipeStage::RoundToMultiple { .. })));
    let encoded = serde_json::to_vec(&recipe).expect("serialize recipe");
    let decoded: image_processors::ProcessorRecipe =
        serde_json::from_slice(&encoded).expect("deserialize recipe");
    assert_eq!(recipe, decoded);
    assert!(
        recipe.to_image_processor_config().is_err(),
        "generic fixed-size processing cannot represent this geometry"
    );
}
