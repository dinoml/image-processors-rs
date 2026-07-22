use std::collections::BTreeMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use image_processors::transforms::resize_center_crop_f32_planes;
use image_processors::{
    downsample_attention_mask, post_process_recipe_outputs, resize_center_crop_plan,
    select_aspect_ratio_bucket, BlipImageProcessor, BlipImageProcessorConfig, DepthMapU16,
    Flux2ImageProcessor, Flux2ImageProcessorConfig, HunyuanVideo15ImageProcessor,
    HunyuanVideo15ImageProcessorConfig, ImageFrame, ImageProcessorOptions, ImageSize,
    JoyImageEditImageProcessor, JoyImageEditImageProcessorConfig, Layout, Ltx2VideoHdrProcessor,
    Ltx2VideoHdrProcessorConfig, MarigoldImageProcessor, MarigoldImageProcessorConfig, Padding,
    PixelFormat, ProcessorOutput, ProcessorRecipe, ProcessorTensorName, RecipeDepthUnit,
    RecipePostprocessContext, RecipePostprocessOutput, ResizeRounding, Tensor, TensorData,
    VaeImageProcessor, VaeImageProcessorConfig, VaeImageProcessorLdm3d, VaeOutputType, VideoClip,
    VideoFrame, VisualClozeProcessor, VisualClozeProcessorConfig, WanAnimateImageProcessor,
    WanAnimateImageProcessorConfig,
};
use serde::Deserialize;
use serde_json::Value;

#[path = "common/fixture_contract.rs"]
mod fixture_contract;

use fixture_contract::{AcceptanceContract, ComparisonPolicy, FloatTolerance};

const FIXTURE: &str = include_str!("fixtures/diffusers/image_processors.json");
const FIXTURE_SCHEMA: &str = "image-processors.diffusers-parity.v2";
const DIFFUSERS_AUDIT_COMMIT: &str = "208704a27a6f362b67cd1a04fa1db0b98036d26f";
const DIFFUSERS_SOURCE_MANIFEST_SHA256: &str =
    "8d868a6ffa94e4cfa32234ac6c49a6c2d4edab72e4cdaa90ea091cecec6d49bb";
const DIFFUSERS_AUDITED_SOURCE_FILES: [&str; 10] = [
    "src/diffusers/image_processor.py",
    "src/diffusers/pipelines/deprecated/blip_diffusion/blip_image_processing.py",
    "src/diffusers/pipelines/flux2/image_processor.py",
    "src/diffusers/pipelines/hunyuan_video1_5/image_processor.py",
    "src/diffusers/pipelines/joyimage/image_processor.py",
    "src/diffusers/pipelines/ltx2/image_processor.py",
    "src/diffusers/pipelines/marigold/marigold_image_processing.py",
    "src/diffusers/pipelines/visualcloze/visualcloze_utils.py",
    "src/diffusers/pipelines/wan/image_processor.py",
    "src/diffusers/video_processor.py",
];
const GENERATOR_NUMPY_VERSION: &str = "2.3.5";
const GENERATOR_PILLOW_VERSION: &str = "10.4.0";
const GENERATOR_TORCH_VERSION: &str = "2.11.0+cpu";
const GENERATOR: &str = "scripts/parity/diffusers_image_parity.py";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    generator: String,
    upstream: Upstream,
    generator_versions: GeneratorVersions,
    acceptance: AcceptanceContract,
    cases: Vec<DiffusersCase>,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    library: String,
    version: String,
    source: String,
    commit: String,
    source_files: Vec<String>,
    source_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct GeneratorVersions {
    numpy: String,
    pillow: String,
    torch: String,
}

#[derive(Debug, Deserialize)]
struct DiffusersCase {
    family: String,
    class_name: String,
    config: BTreeMap<String, Value>,
    #[serde(default)]
    comparison: Option<ComparisonPolicy>,
    #[serde(default)]
    image: Option<FixtureImage>,
    #[serde(default)]
    depth: Option<FixtureDepth>,
    #[serde(default)]
    inputs: BTreeMap<String, Value>,
    outputs: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct FixtureImage {
    mode: String,
    width: usize,
    height: usize,
}

#[derive(Debug, Deserialize)]
struct FixtureDepth {
    mode: String,
    width: usize,
    height: usize,
    values: Vec<u16>,
}

#[derive(Debug, Deserialize)]
struct TensorSummary {
    shape: Vec<usize>,
    dtype: String,
    min: Option<f64>,
    mean: Option<f64>,
    max: Option<f64>,
    sample: Vec<f64>,
    data_base64: String,
    data_encoding: String,
}

#[test]
fn diffusers_fixture_schema_matches_expected_version() {
    let fixture = load_fixture();

    fixture.acceptance.validate();

    assert_eq!(fixture.schema, FIXTURE_SCHEMA);
    assert_eq!(fixture.generator, GENERATOR);
    assert_eq!(fixture.upstream.library, "diffusers");
    assert_eq!(fixture.upstream.commit, DIFFUSERS_AUDIT_COMMIT);
    assert_eq!(
        fixture.upstream.source_manifest_sha256,
        DIFFUSERS_SOURCE_MANIFEST_SHA256
    );
    assert_eq!(
        fixture.upstream.source_files,
        DIFFUSERS_AUDITED_SOURCE_FILES
            .iter()
            .map(|path| (*path).to_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(fixture.generator_versions.numpy, GENERATOR_NUMPY_VERSION);
    assert_eq!(fixture.generator_versions.pillow, GENERATOR_PILLOW_VERSION);
    assert_eq!(fixture.generator_versions.torch, GENERATOR_TORCH_VERSION);
    assert_eq!(
        fixture.upstream.source,
        "diffusers.image_processor+diffusers.pipelines.deprecated.blip_diffusion.blip_image_processing+diffusers.pipelines.flux2.image_processor+diffusers.pipelines.hunyuan_video1_5.image_processor+diffusers.pipelines.joyimage.image_processor+diffusers.pipelines.ltx2.image_processor+diffusers.pipelines.marigold.marigold_image_processing+diffusers.pipelines.visualcloze.visualcloze_utils+diffusers.pipelines.wan.image_processor"
    );
    assert!(
        !fixture.upstream.version.is_empty(),
        "fixture should record the upstream Diffusers version"
    );
    assert_eq!(
        fixture.acceptance.deterministic_input.locations,
        ["cases[].image", "cases[].depth", "cases[].inputs"]
    );
    assert_eq!(
        fixture.acceptance.processor_config.locations,
        ["cases[].config"]
    );
    assert_eq!(
        fixture
            .cases
            .iter()
            .filter(|case| case.comparison.is_some())
            .map(|case| case.family.as_str())
            .collect::<Vec<_>>(),
        ["ip_adapter_mask", "marigold", "wan_animate"]
    );
}

#[test]
fn vae_preprocess_and_postprocess_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("vae");
    assert_eq!(case.class_name, "VaeImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(false));
    assert_eq!(case.config_bool("do_convert_rgb"), Some(true));
    let image = case
        .image
        .as_ref()
        .expect("VAE fixture should include image");
    assert_eq!(image.mode, "RGB");
    let frame = deterministic_frame(image.width, image.height);
    let config = VaeImageProcessorConfig {
        do_resize: false,
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessor::new(config).expect("VAE config should build");

    let tensor = processor
        .preprocess_image(&frame)
        .expect("VAE preprocessing should succeed");

    assert_tensor_matches(
        &tensor,
        &case.output_summary("preprocess"),
        comparison,
        "vae preprocess",
    );

    let input = tensor_from_summary(&case.input_summary("postprocess"), Layout::NCHW);
    let output = processor
        .postprocess(&input, VaeOutputType::Array, None)
        .expect("VAE postprocessing should succeed")
        .into_tensor()
        .expect("array output should be tensor");

    assert_tensor_matches(
        &output,
        &case.output_summary("postprocess_np"),
        comparison,
        "vae postprocess",
    );
}

#[test]
fn attention_mask_downsample_matches_diffusers_ip_adapter_fixture() {
    let (case, comparison) = fixture_case("ip_adapter_mask");
    assert_eq!(case.class_name, "IPAdapterMaskProcessor");
    let mask = case.input_summary("mask");
    let values = f32_values(&mask);
    let batch_size = case.input_usize("batch_size");
    let num_queries = case.input_usize("num_queries");
    let value_embed_dim = case.input_usize("value_embed_dim");

    let output = downsample_attention_mask(
        &values,
        ImageSize::new(mask.shape[1], mask.shape[2]).expect("mask size should be valid"),
        batch_size,
        num_queries,
        value_embed_dim,
    )
    .expect("attention mask downsample should succeed");

    let reference = case.output_summary("downsample");
    assert_eq!(output.shape(), reference.shape.as_slice());
    assert_values_match(
        &output.values,
        &reference,
        comparison,
        "ip_adapter_mask downsample",
    );
}

#[test]
fn ldm3d_preprocess_and_postprocess_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("ldm3d");
    assert_eq!(case.class_name, "VaeImageProcessorLDM3D");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    let image = case
        .image
        .as_ref()
        .expect("LDM3D fixture should include image");
    let depth = case
        .depth
        .as_ref()
        .expect("LDM3D fixture should include depth");
    assert_eq!(image.mode, "RGB");
    assert_eq!(depth.mode, "I;16");
    let rgb = deterministic_frame(image.width, image.height);
    let depth_map = DepthMapU16::new(depth.width, depth.height, depth.values.clone()).unwrap();
    let config = VaeImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        ..Default::default()
    };
    let processor = VaeImageProcessorLdm3d::new(config).expect("LDM3D config should build");

    let output = processor
        .preprocess_with_options(&rgb, &depth_map, ImageProcessorOptions::default())
        .expect("LDM3D preprocessing should succeed");

    assert_tensor_matches(
        output.rgb(),
        &case.output_summary("rgb_preprocess"),
        comparison,
        "ldm3d rgb preprocess",
    );
    assert_tensor_matches(
        output.depth(),
        &case.output_summary("depth_preprocess"),
        comparison,
        "ldm3d depth preprocess",
    );

    let input = tensor_from_summary(&case.input_summary("postprocess"), Layout::NCHW);
    let output = processor
        .postprocess(&input, None)
        .expect("LDM3D postprocessing should succeed");
    let rgb_summary = case.output_summary("postprocess_rgb");
    let depth_summary = case.output_summary("postprocess_depth");

    assert_eq!(output.rgb()[0].data(), u8_values(&rgb_summary).as_slice());
    assert_eq!(
        output.depth()[0].data(),
        u16_values(&depth_summary).as_slice()
    );
}

#[test]
fn blip_preprocess_and_postprocess_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("blip");
    assert_eq!(case.class_name, "BlipImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_bool("do_center_crop"), Some(true));
    assert_eq!(case.config_bool("do_convert_rgb"), Some(true));
    let image = case
        .image
        .as_ref()
        .expect("BLIP fixture should include image");
    let frame = deterministic_frame(image.width, image.height);
    let size = case.config_size("size");
    let processor = BlipImageProcessor::new(BlipImageProcessorConfig {
        size,
        ..Default::default()
    })
    .expect("BLIP config should build");

    let tensor = processor
        .preprocess_image(&frame)
        .expect("BLIP preprocessing should succeed");

    assert_tensor_matches(
        &tensor,
        &case.output_summary("preprocess"),
        comparison,
        "blip preprocess",
    );

    let input = tensor_from_summary(&case.input_summary("postprocess"), Layout::NCHW);
    let output = processor
        .postprocess(&input, VaeOutputType::Array, None)
        .expect("BLIP postprocessing should succeed")
        .into_tensor()
        .expect("array output should be tensor");

    assert_tensor_matches(
        &output,
        &case.output_summary("postprocess_np"),
        comparison,
        "blip postprocess",
    );
}

#[test]
fn joy_image_edit_preprocess_matches_diffusers_fixture() {
    let (case, comparison) = fixture_case("joy_image_edit");
    assert_eq!(case.class_name, "JoyImageEditImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(8));
    assert_eq!(case.config_usize("basesize"), Some(1024));
    assert_eq!(case.config_str("resample"), Some("bilinear"));
    let image = case
        .image
        .as_ref()
        .expect("JoyImage fixture should include image");
    let frame = deterministic_frame(image.width, image.height);
    let processor =
        JoyImageEditImageProcessor::new(JoyImageEditImageProcessorConfig::default()).unwrap();
    let options = ImageProcessorOptions {
        height: Some(case.input_usize("height")),
        width: Some(case.input_usize("width")),
        resize_mode: None,
    };
    let target = processor
        .config()
        .target_size_for_size(
            ImageSize::new(options.height.unwrap(), options.width.unwrap()).unwrap(),
        )
        .expect("JoyImage target selection should succeed");
    let expected_target = case.output_size("target_size");

    assert_eq!(target, expected_target);

    let tensor = processor
        .preprocess_image_with_options(&frame, options)
        .expect("JoyImage preprocessing should succeed");

    assert_tensor_matches(
        &tensor,
        &case.output_summary("preprocess"),
        comparison,
        "joy_image_edit preprocess",
    );
}

#[test]
fn flux2_reference_helpers_and_preprocess_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("flux2");
    assert_eq!(case.class_name, "Flux2ImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(16));
    assert_eq!(case.config_usize("vae_latent_channels"), Some(32));
    assert_eq!(case.config_bool("do_convert_rgb"), Some(true));
    let image = case
        .image
        .as_ref()
        .expect("Flux2 fixture should include image");
    let second = case.input_object("second_image");
    assert_eq!(image.mode, "RGB");
    assert_eq!(second.get("mode").and_then(Value::as_str), Some("L"));
    let frame = deterministic_frame(image.width, image.height);
    let second_frame = deterministic_luma_frame(
        second.get("width").and_then(Value::as_u64).unwrap() as usize,
        second.get("height").and_then(Value::as_u64).unwrap() as usize,
    );
    let processor =
        Flux2ImageProcessor::new(Flux2ImageProcessorConfig::default()).expect("Flux2 config");

    assert!(processor.check_image_input(&frame).is_err());
    assert!(case
        .output_str("check_image_input_error")
        .expect("Flux2 fixture should record validation error")
        .contains("Image too small"));

    let concatenated = processor
        .concatenate_images(&[frame.clone(), second_frame])
        .expect("Flux2 concatenation should succeed");
    assert_frame_matches(
        &concatenated,
        &case.output_summary("concatenate"),
        "flux2 concatenate",
    );

    let limited = processor
        .resize_if_exceeds_area(&concatenated, case.input_usize("target_area"))
        .expect("Flux2 area-limit resize should succeed");
    assert_frame_matches(
        &limited,
        &case.output_summary("area_limited"),
        "flux2 area limit",
    );

    let tensor = processor
        .preprocess_image_with_options(
            &frame,
            ImageProcessorOptions {
                height: Some(case.input_usize("height")),
                width: Some(case.input_usize("width")),
                resize_mode: None,
            },
        )
        .expect("Flux2 preprocessing should succeed");

    assert_tensor_matches(
        &tensor,
        &case.output_summary("preprocess"),
        comparison,
        "flux2 preprocess",
    );
}

#[test]
fn visual_cloze_nested_preprocess_matches_diffusers_fixture() {
    let (case, comparison) = fixture_case("visual_cloze");
    assert_eq!(case.class_name, "VisualClozeProcessor");
    assert_eq!(case.config_usize("resolution"), Some(64));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(16));
    assert_eq!(case.config_usize("vae_latent_channels"), Some(16));
    let processor = VisualClozeProcessor::new(VisualClozeProcessorConfig {
        resolution: 64,
        ..Default::default()
    })
    .expect("VisualCloze config should build");

    let grid = case
        .inputs
        .get("grid")
        .and_then(Value::as_array)
        .expect("VisualCloze fixture should include grid");
    let cell = |row: usize, column: usize| {
        grid[row]
            .as_array()
            .and_then(|row| row[column].as_object())
            .expect("grid cell should be an object")
    };
    let frame_from_cell = |row: usize, column: usize| {
        let cell = cell(row, column);
        deterministic_frame(
            cell.get("width").and_then(Value::as_u64).unwrap() as usize,
            cell.get("height").and_then(Value::as_u64).unwrap() as usize,
        )
    };

    let output = processor
        .preprocess_image_grid(&[
            vec![Some(frame_from_cell(0, 0)), Some(frame_from_cell(0, 1))],
            vec![Some(frame_from_cell(1, 0)), None],
        ])
        .expect("VisualCloze preprocessing should succeed");

    assert_eq!(
        output.target_position(),
        case.output_usize_vec("target_position")
    );
    assert_eq!(output.metadata().rows, 2);
    assert_eq!(output.metadata().columns, 2);
    assert_eq!(
        output.metadata().target_mask,
        vec![vec![false, false], vec![false, true]]
    );
    assert_eq!(
        output.metadata().image_sizes,
        case.output_nested_sizes("image_sizes")
    );
    assert_eq!(output.metadata().target_positions[0].row, 1);
    assert_eq!(output.metadata().target_positions[0].column, 1);

    for (label, tensor) in [
        ("image_0_0", &output.images()[0][0]),
        ("image_0_1", &output.images()[0][1]),
        ("image_1_0", &output.images()[1][0]),
        ("image_1_1", &output.images()[1][1]),
    ] {
        assert_tensor_matches(tensor, &case.output_summary(label), comparison, label);
    }

    for (label, tensor) in [
        ("mask_0_0", &output.masks()[0][0]),
        ("mask_0_1", &output.masks()[0][1]),
        ("mask_1_0", &output.masks()[1][0]),
        ("mask_1_1", &output.masks()[1][1]),
    ] {
        assert_numeric_tensor_matches(tensor, &case.output_summary(label), comparison, label);
    }

    let upsampling = case.input_object("upsampling");
    let upsampling_output = processor
        .preprocess_image_upsampling(
            &deterministic_frame(2, 2),
            ImageSize::new(
                upsampling.get("height").and_then(Value::as_u64).unwrap() as usize,
                upsampling.get("width").and_then(Value::as_u64).unwrap() as usize,
            )
            .expect("upsampling size should be valid"),
        )
        .expect("VisualCloze upsampling should succeed");
    assert_tensor_matches(
        &upsampling_output.images()[0][0],
        &case.output_summary("upsampling_image"),
        comparison,
        "visual_cloze upsampling image",
    );
    assert_numeric_tensor_matches(
        &upsampling_output.masks()[0][0],
        &case.output_summary("upsampling_mask"),
        comparison,
        "visual_cloze upsampling mask",
    );
}

#[test]
fn hunyuan_video_15_bucket_selection_matches_diffusers_fixture() {
    let (case, comparison) = fixture_case("hunyuan_video_15");
    assert_eq!(case.class_name, "HunyuanVideo15ImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(16));
    assert_eq!(case.config_usize("vae_latent_channels"), Some(32));
    assert_eq!(case.config_bool("do_convert_rgb"), Some(true));
    let processor =
        HunyuanVideo15ImageProcessor::new(HunyuanVideo15ImageProcessorConfig::default())
            .expect("HunyuanVideo 1.5 config should build");
    let target_size = case.input_usize("target_size");

    for name in ["landscape", "portrait", "square"] {
        let input = case.input_size(name);
        let output = processor
            .calculate_default_height_width(input.height, input.width, target_size)
            .expect("HunyuanVideo 1.5 bucket selection should succeed");
        assert_eq!(output, case.output_size(name), "{name} bucket");
    }

    let first = deterministic_frame(16, 16);
    let second = horizontal_flip_rgb_frame(&first);
    let video = VideoClip::new(vec![VideoFrame::new(first), VideoFrame::new(second)], None)
        .expect("HunyuanVideo fixture video should be valid");
    let tensor = processor
        .preprocess_video(&video)
        .expect("HunyuanVideo video preprocessing should succeed");
    assert_tensor_matches(
        &tensor,
        &case.output_summary("video_preprocess"),
        comparison,
        "hunyuan_video_15 video preprocess",
    );
}

#[test]
fn marigold_preprocess_and_outputs_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("marigold");
    assert_eq!(case.class_name, "MarigoldImageProcessor");
    assert_eq!(case.config_usize("vae_scale_factor"), Some(4));
    assert_eq!(case.config_bool("do_normalize"), Some(true));
    assert_eq!(case.config_bool("do_range_check"), Some(true));
    assert_eq!(case.config_str("resample"), Some("bilinear"));
    let image = case
        .image
        .as_ref()
        .expect("Marigold fixture should include image");
    let frame = deterministic_frame(image.width, image.height);
    let processor = MarigoldImageProcessor::new(MarigoldImageProcessorConfig {
        vae_scale_factor: 4,
        ..Default::default()
    })
    .expect("Marigold config should build");

    let preprocessed = processor
        .preprocess_image(&frame)
        .expect("Marigold preprocessing should succeed");

    assert_tensor_matches(
        preprocessed.pixel_values(),
        &case.output_summary("preprocess"),
        comparison,
        "marigold preprocess",
    );
    let padding = case.output_usize_vec("padding");
    assert_eq!(
        preprocessed.padding(),
        Padding::new(0, padding[1], padding[0], 0)
    );
    assert_eq!(
        preprocessed.original_resolution(),
        case.output_size("original_resolution")
    );

    let max_edge = processor
        .preprocess_image_with_processing_resolution(
            &frame,
            Some(case.input_usize("processing_resolution")),
        )
        .expect("Marigold max-edge preprocessing should succeed");
    assert_tensor_matches(
        max_edge.pixel_values(),
        &case.output_summary("max_edge_preprocess"),
        comparison,
        "marigold max-edge preprocess",
    );
    let max_edge_padding = case.output_usize_vec("max_edge_padding");
    assert_eq!(
        max_edge.padding(),
        Padding::new(0, max_edge_padding[1], max_edge_padding[0], 0)
    );
    assert_eq!(
        max_edge.original_resolution(),
        case.output_size("max_edge_original_resolution")
    );

    let padded_depth = tensor_from_summary(&case.input_summary("padded_depth"), Layout::NCHW);
    let depth = processor
        .postprocess_depth(&padded_depth, &preprocessed, RecipeDepthUnit::Relative)
        .expect("Marigold depth postprocess should succeed");
    let exported = MarigoldImageProcessor::export_depth_to_16bit(&depth[0], 0.0, 1.0)
        .expect("Marigold depth export should succeed");
    assert_depth_u16_matches(
        &exported,
        &case.output_summary("depth_u16"),
        "marigold depth u16",
    );

    let normals_tensor = marigold_padded_normals_tensor();
    let normals = processor
        .postprocess_normals(&normals_tensor, &preprocessed)
        .expect("Marigold normals postprocess should succeed");
    let normals_visual =
        MarigoldImageProcessor::visualize_normals(&normals[0], false, false, false)
            .expect("Marigold normals visualization should succeed");
    assert_frame_matches(
        &normals_visual,
        &case.output_summary("normals_visual"),
        "marigold normals visual",
    );

    let uncertainty = processor
        .postprocess_uncertainty(&padded_depth, &preprocessed)
        .expect("Marigold uncertainty postprocess should succeed");
    let uncertainty_visual = MarigoldImageProcessor::visualize_uncertainty(&uncertainty[0], 100.0)
        .expect("Marigold uncertainty visualization should succeed");
    assert_frame_matches(
        &uncertainty_visual,
        &case.output_summary("uncertainty_visual"),
        "marigold uncertainty visual",
    );
}

#[test]
fn wan_animate_preprocess_matches_diffusers_fixture() {
    let (case, comparison) = fixture_case("wan_animate");
    assert_eq!(case.class_name, "WanAnimateImageProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(8));
    assert_eq!(case.config_usize("vae_latent_channels"), Some(16));
    assert_eq!(case.config_str("resample"), Some("lanczos"));
    let image = case
        .image
        .as_ref()
        .expect("Wan fixture should include image");
    let frame = deterministic_frame(image.width, image.height);
    let processor = WanAnimateImageProcessor::new(WanAnimateImageProcessorConfig::default())
        .expect("Wan config should build");
    let options = ImageProcessorOptions {
        height: Some(case.input_usize("height")),
        width: Some(case.input_usize("width")),
        resize_mode: None,
    };
    let target = processor
        .config()
        .target_size_for_size(
            ImageSize::new(options.height.unwrap(), options.width.unwrap()).unwrap(),
        )
        .expect("Wan target selection should succeed");
    let expected_target = case.output_size("target_size");

    assert_eq!(target, expected_target);
    assert_eq!(case.config_usize_vec("spatial_patch_size"), vec![2, 2]);
    assert_eq!(case.input_str("resize_mode"), Some("fill"));

    let tensor = processor
        .preprocess_image_with_options(&frame, options)
        .expect("Wan preprocessing should succeed");

    assert_tensor_matches(
        &tensor,
        &case.output_summary("preprocess"),
        comparison,
        "wan_animate preprocess",
    );
}

#[test]
fn ltx2_reference_and_hdr_postprocess_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("ltx2_video_hdr");
    assert_eq!(case.class_name, "LTX2VideoHDRProcessor");
    assert_eq!(case.config_bool("do_resize"), Some(true));
    assert_eq!(case.config_str("hdr_transform"), Some("logc3"));
    assert_eq!(case.config_usize("vae_scale_factor"), Some(32));
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig::default()).unwrap();
    let target_size = case.input_size("target_size");
    let frame_metadata = case.input_object("video_frames");
    let frame_count = frame_metadata
        .get("count")
        .and_then(Value::as_u64)
        .expect("LTX2 fixture should record a frame count") as usize;
    let frame_height = frame_metadata
        .get("height")
        .and_then(Value::as_u64)
        .expect("LTX2 fixture should record frame height") as usize;
    let frame_width = frame_metadata
        .get("width")
        .and_then(Value::as_u64)
        .expect("LTX2 fixture should record frame width") as usize;
    assert_eq!(frame_count, 2);
    assert_eq!(
        frame_metadata
            .get("second_frame_horizontal_flip")
            .and_then(Value::as_bool),
        Some(true)
    );
    let video = ltx2_reference_video(frame_width, frame_height);

    let frames = video
        .frames()
        .iter()
        .map(|frame| frame.image().clone())
        .collect::<Vec<_>>();
    let vae_preprocessed = processor
        .vae_processor()
        .preprocess_images(&frames)
        .expect("LTX2 base VAE preprocessing should succeed");
    assert_tensor_matches(
        &vae_preprocessed,
        &case.output_summary("vae_preprocess"),
        comparison,
        "ltx2 VAE preprocess",
    );

    let reference = processor
        .preprocess_reference_video_hdr(&video, target_size)
        .expect("LTX2 HDR reference preprocessing should succeed");

    assert_tensor_matches(
        &reference,
        &case.output_summary("reference_preprocess"),
        comparison,
        "ltx2 reference preprocess",
    );

    let recipe = processor
        .config()
        .processor_recipe()
        .expect("LTX2 shared video-tensor recipe should validate");
    let decoded = tensor_from_summary(&case.input_summary("postprocess"), Layout::BFCHW);
    let decoded_channel_last = decoded
        .to_layout(Layout::BFHWC)
        .expect("fixture input should convert to channel-last video layout");

    for (label, decoded) in [
        ("channel-first", decoded),
        ("channel-last", decoded_channel_last),
    ] {
        let hdr = execute_video_tensor_recipe(&recipe, decoded);
        assert_tensor_matches(
            &hdr,
            &case.output_summary("hdr_postprocess"),
            comparison,
            &format!("ltx2 {label} HDR recipe postprocess"),
        );
    }
}

#[test]
fn ltx2_video_tensor_descriptor_preserves_fixture_values_across_batches_and_pixels() {
    let (case, comparison) = fixture_case("ltx2_video_hdr");
    let processor = Ltx2VideoHdrProcessor::new(Ltx2VideoHdrProcessorConfig::default()).unwrap();
    let recipe = processor
        .config()
        .processor_recipe()
        .expect("LTX2 shared video-tensor recipe should validate");
    let decoded = tensor_from_summary(&case.input_summary("postprocess"), Layout::BFCHW)
        .to_layout(Layout::BFHWC)
        .expect("fixture input should convert to channel-last video layout");
    let expected = f32_values(&case.output_summary("hdr_postprocess"));
    let expanded_input = expand_video_fixture_values(&decoded.data().to_vec::<f32>());
    let expanded_expected = expand_video_fixture_values(&expected);
    let expanded = Tensor::new(
        TensorData::F32(expanded_input),
        [2, 2, 1, 2, 3],
        Layout::BFHWC,
    )
    .expect("expanded fixture video tensor should be valid");

    let hdr = execute_video_tensor_recipe(&recipe, expanded);

    assert_eq!(hdr.layout(), Layout::BFHWC);
    assert_eq!(hdr.shape(), [2, 2, 1, 2, 3]);
    let actual = hdr.data().to_vec::<f32>();
    assert_eq!(
        actual.len(),
        expanded_expected.len(),
        "expanded LTX2 HDR output length"
    );
    for (index, (actual, expected)) in actual.into_iter().zip(expanded_expected).enumerate() {
        assert_close(
            f64::from(actual),
            f64::from(expected),
            comparison.full_payload,
            "expanded LTX2 HDR recipe postprocess",
            &format!("values[{index}]"),
        );
    }
}

#[test]
fn pixart_bins_and_resize_crop_plan_match_diffusers_fixture() {
    let (case, comparison) = fixture_case("pixart");
    assert_eq!(case.class_name, "PixArtImageProcessor");
    let height = case.input_usize("height");
    let width = case.input_usize("width");
    let ratio_sizes = case.input_ratio_sizes("ratios");
    let candidates = ratio_sizes
        .iter()
        .map(|size| ImageSize::new(size[0], size[1]).expect("candidate size should be valid"))
        .collect::<Vec<_>>();

    let selected = select_aspect_ratio_bucket(
        ImageSize::new(height, width).expect("source size should be valid"),
        &candidates,
    )
    .expect("PixArt bin selection should succeed");

    let selected_summary = case.output_usize_vec("selected_size");
    assert_eq!(selected.height, selected_summary[0]);
    assert_eq!(selected.width, selected_summary[1]);

    let input = case.input_summary("resize_crop_input");
    assert_eq!(input.shape.len(), 4, "PixArt fixture input must be NCHW");
    let target = case.input_size("resize_crop_target");
    let plan = resize_center_crop_plan(
        ImageSize::new(input.shape[2], input.shape[3]).expect("input size should be valid"),
        target,
        ResizeRounding::Floor,
    )
    .expect("PixArt resize/crop plan should succeed");
    let expected = case.output_object("resize_crop_plan");
    let resized = expected
        .get("resized_size")
        .expect("resize_crop_plan should contain resized_size");

    assert_eq!(
        plan.resized_size,
        ImageSize::new(
            resized.get("height").and_then(Value::as_u64).unwrap() as usize,
            resized.get("width").and_then(Value::as_u64).unwrap() as usize,
        )
        .unwrap()
    );
    assert_eq!(
        plan.crop_box.y_min,
        expected.get("crop_top").and_then(Value::as_u64).unwrap() as usize
    );
    assert_eq!(
        plan.crop_box.x_min,
        expected.get("crop_left").and_then(Value::as_u64).unwrap() as usize
    );

    let plane_count = input.shape[0] * input.shape[1];
    let resized = resize_center_crop_f32_planes(
        &f32_values(&input),
        plane_count,
        ImageSize::new(input.shape[2], input.shape[3]).expect("input size should be valid"),
        target,
    )
    .expect("PixArt tensor resize and crop should succeed");
    let reference = case.output_summary("resize_crop");
    assert_eq!(
        reference.shape,
        vec![input.shape[0], input.shape[1], target.height, target.width]
    );
    assert_values_match(&resized, &reference, comparison, "pixart resize and crop");
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("fixture JSON should parse")
}

fn fixture_case(family: &str) -> (DiffusersCase, ComparisonPolicy) {
    let fixture = load_fixture();
    let default_comparison = fixture.acceptance.comparison;
    let mut matches = fixture
        .cases
        .into_iter()
        .filter(|case| case.family == family);
    let case = matches
        .next()
        .unwrap_or_else(|| panic!("missing {family} fixture case"));
    assert!(
        matches.next().is_none(),
        "fixture should contain one {family} case"
    );
    let comparison = case.comparison.unwrap_or(default_comparison);
    comparison.validate();
    (case, comparison)
}

fn deterministic_frame(width: usize, height: usize) -> ImageFrame {
    let values = (0..width * height * 3)
        .map(|index| ((index as u32 * 37 + 17) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Rgb8, values)
        .expect("deterministic fixture image should be valid")
}

fn deterministic_luma_frame(width: usize, height: usize) -> ImageFrame {
    let values = (0..width * height)
        .map(|index| ((index as u32 * 53 + 11) % 256) as u8)
        .collect();
    ImageFrame::new(width, height, PixelFormat::Luma8, values)
        .expect("deterministic fixture mask should be valid")
}

fn horizontal_flip_rgb_frame(frame: &ImageFrame) -> ImageFrame {
    let mut values = Vec::with_capacity(frame.data().len());
    for row in frame.data().chunks_exact(frame.width() * 3) {
        for pixel in row.chunks_exact(3).rev() {
            values.extend_from_slice(pixel);
        }
    }
    ImageFrame::new(frame.width(), frame.height(), PixelFormat::Rgb8, values)
        .expect("flipped deterministic frame should be valid")
}

fn ltx2_reference_video(width: usize, height: usize) -> VideoClip {
    let first = deterministic_frame(width, height);
    let second = horizontal_flip_rgb_frame(&first);
    VideoClip::new(
        vec![VideoFrame::new(first), VideoFrame::new(second)],
        Some(24.0),
    )
    .expect("LTX2 reference video should be valid")
}

fn execute_video_tensor_recipe(recipe: &ProcessorRecipe, tensor: Tensor) -> Tensor {
    let mut model_outputs = ProcessorOutput::new();
    model_outputs.insert_tensor(ProcessorTensorName::other("decoded_video"), tensor);
    let mut outputs = post_process_recipe_outputs(
        recipe,
        &model_outputs,
        &ProcessorOutput::new(),
        RecipePostprocessContext::new(),
    )
    .expect("fixture-backed video-tensor recipe should execute");
    assert_eq!(outputs.len(), 1, "LTX2 recipe should emit one output");

    match outputs.pop().expect("LTX2 recipe output should exist") {
        RecipePostprocessOutput::VideoTensor {
            output_name,
            tensor,
            ..
        } => {
            assert_eq!(output_name, "decoded_video");
            tensor
        }
        output => panic!("expected video-tensor recipe output, got {output:?}"),
    }
}

fn expand_video_fixture_values(values: &[f32]) -> Vec<f32> {
    let mut expanded = Vec::with_capacity(values.len() * 4);
    for _ in 0..2 {
        for pixel in values.chunks_exact(3) {
            expanded.extend_from_slice(pixel);
            expanded.extend_from_slice(pixel);
        }
    }
    expanded
}

fn marigold_padded_normals_tensor() -> Tensor {
    Tensor::new(
        TensorData::F32(vec![
            -1.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, -1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ]),
        vec![1, 3, 4, 4],
        Layout::NCHW,
    )
    .expect("Marigold normals tensor should be valid")
}

fn tensor_from_summary(summary: &TensorSummary, layout: Layout) -> Tensor {
    Tensor::new(
        TensorData::F32(f32_values(summary)),
        summary.shape.clone(),
        layout,
    )
    .expect("fixture tensor should be valid")
}

fn assert_tensor_matches(
    tensor: &Tensor,
    reference: &TensorSummary,
    comparison: ComparisonPolicy,
    label: &str,
) {
    assert_eq!(tensor.shape(), reference.shape.as_slice(), "{label} shape");
    assert_eq!(reference.dtype, "float32", "{label} dtype");
    let TensorData::F32(values) = tensor.data() else {
        panic!("{label} expected float32, got {:?}", tensor.data().dtype());
    };
    assert_values_match(values, reference, comparison, label);
}

fn assert_numeric_tensor_matches(
    tensor: &Tensor,
    reference: &TensorSummary,
    comparison: ComparisonPolicy,
    label: &str,
) {
    assert_eq!(tensor.shape(), reference.shape.as_slice(), "{label} shape");
    match reference.dtype.as_str() {
        "float32" => {
            let TensorData::F32(values) = tensor.data() else {
                panic!("{label} expected float32, got {:?}", tensor.data().dtype());
            };
            assert_values_match(values, reference, comparison, label);
        }
        "int64" => {
            let TensorData::I64(values) = tensor.data() else {
                panic!("{label} expected int64, got {:?}", tensor.data().dtype());
            };
            assert_eq!(values, &i64_values(reference), "{label} integer payload");
        }
        dtype => panic!("{label} unsupported numeric tensor dtype {dtype}"),
    }
}

fn assert_frame_matches(frame: &ImageFrame, reference: &TensorSummary, label: &str) {
    let expected_shape =
        if frame.channels() == 1 && reference.shape == vec![frame.height(), frame.width()] {
            vec![frame.height(), frame.width()]
        } else {
            vec![frame.height(), frame.width(), frame.channels()]
        };
    assert_eq!(reference.shape, expected_shape, "{label} shape");
    assert_eq!(reference.dtype, "uint8", "{label} dtype");
    assert_eq!(frame.data(), u8_values(reference), "{label} byte payload");
}

fn assert_depth_u16_matches(depth: &DepthMapU16, reference: &TensorSummary, label: &str) {
    assert_eq!(
        reference.shape,
        vec![depth.height(), depth.width()],
        "{label} shape"
    );
    assert_eq!(reference.dtype, "uint16", "{label} dtype");
    assert_eq!(depth.data(), u16_values(reference), "{label} depth payload");
}

fn assert_values_match(
    values: &[f32],
    reference: &TensorSummary,
    comparison: ComparisonPolicy,
    label: &str,
) {
    let stats = summarize(values);
    assert_close(
        stats.min,
        reference.min.unwrap(),
        comparison.statistics,
        label,
        "min",
    );
    assert_close(
        stats.mean,
        reference.mean.unwrap(),
        comparison.statistics,
        label,
        "mean",
    );
    assert_close(
        stats.max,
        reference.max.unwrap(),
        comparison.statistics,
        label,
        "max",
    );
    let expected_values = f32_values(reference);
    assert_eq!(values.len(), expected_values.len(), "{label} value count");
    for (index, expected) in expected_values.iter().copied().enumerate() {
        assert_close(
            f64::from(values[index]),
            f64::from(expected),
            comparison.full_payload,
            label,
            &format!("values[{index}]"),
        );
    }
    for (index, expected) in reference.sample.iter().copied().enumerate() {
        assert_close(
            f64::from(values[index]),
            expected,
            comparison.full_payload,
            label,
            &format!("sample[{index}]"),
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: f64,
    mean: f64,
    max: f64,
}

fn summarize(values: &[f32]) -> Stats {
    assert!(!values.is_empty(), "tensor should not be empty");
    let mut min = f64::from(values[0]);
    let mut max = min;
    let mut sum = 0.0;
    for &value in values {
        let value = f64::from(value);
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    Stats {
        min,
        mean: sum / values.len() as f64,
        max,
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: FloatTolerance, label: &str, field: &str) {
    let delta = (actual - expected).abs();
    assert!(
        tolerance.matches(actual, expected),
        "{label} {field} differs from Diffusers fixture: actual={actual:.8}, expected={expected:.8}, delta={delta:.8}, allowed={:.8}",
        tolerance.allowed_delta(expected)
    );
}

fn f32_values(summary: &TensorSummary) -> Vec<f32> {
    assert_eq!(summary.data_encoding, "base64_little_endian");
    let bytes = STANDARD
        .decode(&summary.data_base64)
        .expect("fixture full payload should be valid base64");
    match summary.dtype.as_str() {
        "float32" => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        "uint8" => bytes.into_iter().map(f32::from).collect(),
        "uint16" => bytes
            .chunks_exact(2)
            .map(|chunk| f32::from(u16::from_le_bytes(chunk.try_into().unwrap())))
            .collect(),
        "int64" => bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as f32)
            .collect(),
        dtype => panic!("unsupported fixture dtype {dtype}"),
    }
}

fn u8_values(summary: &TensorSummary) -> Vec<u8> {
    assert_eq!(summary.dtype, "uint8");
    STANDARD
        .decode(&summary.data_base64)
        .expect("fixture uint8 payload should be valid base64")
}

fn u16_values(summary: &TensorSummary) -> Vec<u16> {
    assert_eq!(summary.dtype, "uint16");
    STANDARD
        .decode(&summary.data_base64)
        .expect("fixture uint16 payload should be valid base64")
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn i64_values(summary: &TensorSummary) -> Vec<i64> {
    assert_eq!(summary.dtype, "int64");
    STANDARD
        .decode(&summary.data_base64)
        .expect("fixture int64 payload should be valid base64")
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("int64 chunk")))
        .collect()
}

impl DiffusersCase {
    fn input_summary(&self, name: &str) -> TensorSummary {
        serde_json::from_value(
            self.inputs
                .get(name)
                .unwrap_or_else(|| panic!("{} missing input {name}", self.family))
                .clone(),
        )
        .expect("input summary should parse")
    }

    fn output_summary(&self, name: &str) -> TensorSummary {
        serde_json::from_value(
            self.outputs
                .get(name)
                .unwrap_or_else(|| panic!("{} missing output {name}", self.family))
                .clone(),
        )
        .expect("output summary should parse")
    }

    fn input_usize(&self, name: &str) -> usize {
        self.inputs
            .get(name)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("{} missing integer input {name}", self.family))
            as usize
    }

    fn input_size(&self, name: &str) -> ImageSize {
        let value = self
            .inputs
            .get(name)
            .unwrap_or_else(|| panic!("{} missing size input {name}", self.family));
        ImageSize::new(
            value.get("height").and_then(Value::as_u64).unwrap() as usize,
            value.get("width").and_then(Value::as_u64).unwrap() as usize,
        )
        .expect("fixture size should be valid")
    }

    fn input_str(&self, name: &str) -> Option<&str> {
        self.inputs.get(name).and_then(Value::as_str)
    }

    fn input_object(&self, name: &str) -> &serde_json::Map<String, Value> {
        self.inputs
            .get(name)
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{} missing object input {name}", self.family))
    }

    fn input_ratio_sizes(&self, name: &str) -> Vec<[usize; 2]> {
        let ratios = self
            .inputs
            .get(name)
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{} missing ratio map input {name}", self.family));
        ratios
            .values()
            .map(|value| {
                let pair = value.as_array().expect("ratio value should be array");
                [
                    pair[0].as_u64().expect("height should be integer") as usize,
                    pair[1].as_u64().expect("width should be integer") as usize,
                ]
            })
            .collect()
    }

    fn output_usize_vec(&self, name: &str) -> Vec<usize> {
        self.outputs
            .get(name)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{} missing vector output {name}", self.family))
            .iter()
            .map(|value| value.as_u64().expect("vector entries should be integers") as usize)
            .collect()
    }

    fn output_object(&self, name: &str) -> &serde_json::Map<String, Value> {
        self.outputs
            .get(name)
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{} missing object output {name}", self.family))
    }

    fn output_size(&self, name: &str) -> ImageSize {
        let value = self
            .outputs
            .get(name)
            .unwrap_or_else(|| panic!("{} missing size output {name}", self.family));
        ImageSize::new(
            value.get("height").and_then(Value::as_u64).unwrap() as usize,
            value.get("width").and_then(Value::as_u64).unwrap() as usize,
        )
        .expect("fixture size should be valid")
    }

    fn output_nested_sizes(&self, name: &str) -> Vec<Vec<ImageSize>> {
        self.outputs
            .get(name)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{} missing nested size output {name}", self.family))
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("nested size row should be an array")
                    .iter()
                    .map(|size| {
                        let pair = size.as_array().expect("size should be [height, width]");
                        ImageSize::new(
                            pair[0].as_u64().expect("height should be integer") as usize,
                            pair[1].as_u64().expect("width should be integer") as usize,
                        )
                        .expect("nested fixture size should be valid")
                    })
                    .collect()
            })
            .collect()
    }

    fn output_str(&self, name: &str) -> Option<&str> {
        self.outputs.get(name).and_then(Value::as_str)
    }

    fn config_bool(&self, name: &str) -> Option<bool> {
        self.config.get(name).and_then(Value::as_bool)
    }

    fn config_str(&self, name: &str) -> Option<&str> {
        self.config.get(name).and_then(Value::as_str)
    }

    fn config_usize(&self, name: &str) -> Option<usize> {
        self.config
            .get(name)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
    }

    fn config_size(&self, name: &str) -> ImageSize {
        let value = self
            .config
            .get(name)
            .unwrap_or_else(|| panic!("{} missing size config {name}", self.family));
        ImageSize::new(
            value.get("height").and_then(Value::as_u64).unwrap() as usize,
            value.get("width").and_then(Value::as_u64).unwrap() as usize,
        )
        .expect("fixture config size should be valid")
    }

    fn config_usize_vec(&self, name: &str) -> Vec<usize> {
        self.config
            .get(name)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{} missing vector config {name}", self.family))
            .iter()
            .map(|value| value.as_u64().expect("config entries should be integers") as usize)
            .collect()
    }
}
