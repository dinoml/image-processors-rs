# image-processors

Rust-first media loading and image processor primitives for model image processors.

The current focus is the Rust crate architecture. Bindings are intentionally out of tree until the core owns the important boundaries:

- media source loading and decoded frame ownership
- image/video frame metadata
- channel and tensor layout conversion
- resize, crop, mask, normalization, and binarization kernels
- processor configs and concrete entrypoints that keep ownership in Rust

## Architecture

- `image-resize-kernels/src/lib.rs`
  - resize profiles for model parity and performance
  - fast SIMD-capable backend boundary
  - Pillow/Transformers compatibility target for byte-exact kernel work

- `image-processors/src/media.rs`
  - `MediaSource`
  - `MediaLocation`
  - `MediaHint`
  - `MediaType`
  - `DefaultMediaLoader`
  - `ImageFrame`
  - zero-copy HWC tensor views over decoded frame bytes
  - `ImageSequence`
  - `FrameTiming`
  - `FrameSampling`
  - `VideoFrame`
  - `VideoClip`
  - `LoadedMedia`
  - `PixelFormat`
  - `RemoteReadMode`
  - `RemoteLoadOptions`
  - `RemoteRedirectPolicy`
  - animated GIF/APNG/WebP image sequence decoding
  - optional URL image loading through `reqwest`
  - optional video decoding through `video-rs`
  - frame sampling for image sequences, animated image sources, video clips,
    and decoded video sources
  - buffered image URL loading, direct video URL streaming, and buffered video URL loading when `url` and `video` are both enabled

- `image-processors/src/tensor.rs`
  - `Tensor`
  - `TensorData`
  - `TensorDataView`
  - `TensorView`
  - `QuantizationParams`
  - `DType`
  - `Layout` with image, batch, batched-video, patch-batch, and nested-image tiled variants
  - `ImageLayout` and `VideoLayout` semantic axis-order helpers
  - `TensorLeadingAxis` metadata for batch tensors versus temporal frame tensors
  - affine quantized u8/i8 storage and packed u4/i4 storage
  - scalar rescale and per-channel normalization kernels

- `image-processors/src/image.rs`
  - `ImageProcessor`
  - `ImageProcessorConfig`
  - `ImageProcessorOptions`
  - `ImageProcessorWorkspace`
  - source-oriented APIs such as `ImageProcessor::open`, `ImageProcessor::open_source`, and `ImageProcessor::open_batch`
  - output-oriented APIs such as `ImageProcessor::open_output` and `ImageProcessor::preprocess_image_output`
  - per-call batch overrides through `ImageProcessor::preprocess_images_with_options`
  - batched decoded video preprocessing through `ImageProcessor::preprocess_videos`
  - reusable batch API `ImageProcessor::open_batch_into`
  - zero-copy decoded-frame views through byte-preserving image, sequence, and video entrypoints
  - semantic `ImageLayout` and `VideoLayout` helpers derived from processor output layout
  - validated construction for resize, rescale, and normalization configuration

- `image-processors/src/recipe.rs`
  - `ProcessorRecipe`
  - `ProcessorRecipeStage`
  - `RecipeResizeStage`
  - `RecipeResizeTarget`
  - `RecipeCropStage`
  - `RecipePadStage`
  - `RecipeOverlayStage`
  - `RecipePatchGridStage`
  - `RecipeImageSplitStage`
  - `RecipeAspectRatioCropStage`
  - `RecipeTiledCanvasStage`
  - `RecipeDocumentGeometryStage`
  - `RecipePatchStage`
  - `ProcessorRecipeInput`
  - `ProcessorRecipeOutput`
  - `ProcessorRecipePostprocess`
  - serde-serializable preprocessing and postprocess-only recipe data with
    validation on construction and deserialization
  - lowering for generic resize, pixel-format conversion, rescale, normalize, and binarize stages
  - smart-resize and patch-flatten stage descriptions for VLM recipes
  - common transform stage descriptions for size constraints, aspect-ratio
    buckets, shortest/longest-edge resize, area resize, cover crop, multiple
    rounding, center/absolute crop, constant and edge/reflect canvas pad,
    bottom/right pad-to-multiple with crop-back metadata, overlay, mask
    composite, pixel-format-preserving fill/canvas resize, RGB
    fit/fill, horizontal RGB concatenation, frame sampling for decoded
    sequences or clips, temporal repeat-last padding for incomplete video
    groups, frame-count plus spatial video bucket selection, patch-grid
    extraction, split-image frames, aspect-ratio crop expansion, tiled-canvas
    geometry, and document/OCR thumbnail geometry
  - patch-stage helpers for image and temporal `image_grid_thw` metadata plus
    patch flattening execution backed by the shared repeat-last temporal plan
    for Qwen and non-Qwen patch-grid profiles
  - postprocess descriptors for object detection, semantic/instance/panoptic
    segmentation, binary mask logits, depth outputs, dense image-like maps,
    optional named dense-map targets, LogC3 video tensor transfers,
    keypoints/pose coordinates, tokenizer-independent structured token
    sequences, and structured output hooks for OCR, table, layout, or document
    tensors
  - shared execution for DETR object-detection, semantic/instance/panoptic
    segmentation, binary-mask, depth, dense-map, video tensor transfer,
    coordinate, token-sequence, and output-hook recipe postprocess descriptors
  - postprocess-only LogC3 HDR video recipe construction for descriptor-backed
    decoded video outputs

- `image-processors/src/catalog.rs`
  - `ProcessorCatalog`
  - `CatalogQuery`
  - `CatalogEntry`
  - `UpstreamLibrary`
  - `ProcessorFamilyKind`
  - `CompatibilityStatus`
  - static lookup from upstream library, canonical or alias class name, model type, or known processor id to an optional recipe and family wrapper, preset collection, or utility
  - private source-controlled `catalog_data` tables with audit commits and full-value fixture-parity status
  - direct fixture-file links for 13 dedicated Transformers families, plus
    suite-level identity/commit coverage for aggregate fixtures and the
    declared auxiliary Qwen2-VL video-processor fixture

- `image-processors/src/output.rs`
  - `ProcessorOutput`
  - `ProcessorTensorName`
  - `ProcessorMetadataName`
  - typed outputs for `pixel_values`, `pixel_mask`, `original_sizes`, `reshaped_input_sizes`, `image_grid_thw`, `image_patch_counts`, nested count/mask metadata, nested image-grid layout metadata, and detection `labels`

- `image-processors/src/postprocess.rs`
  - public `postprocess` namespace for task-specific output restoration
  - descriptor-driven recipe postprocess execution for DETR object detection,
    semantic/instance/panoptic segmentation, binary masks, depth maps, dense
    image-like maps, LogC3 video tensor transfers, coordinates, structured
    token ids, and output hooks
  - object-detection box restoration, score filtering, and NMS helpers
  - semantic, instance, and panoptic segmentation output helpers
  - depth map resize-back helpers with explicit unit semantics, plus dense-map
    resize-back helpers for multi-channel normal, uncertainty, or
    intrinsic-image target outputs
  - unit-range image and video tensor restoration for channel-first and
    channel-last image batches and batched video layouts
  - LogC3 HDR video tensor restoration for LTX2-style decoded outputs, with
    recipe descriptor execution through the generic video tensor postprocess
    path, a catalogable postprocess-only recipe, and the LTX2 wrapper
  - normalized coordinate restoration helpers for keypoint, matching, and
    pose outputs
  - CTC and BOS/EOS-aware greedy token-id decoding with an explicit external
    tokenizer/vocabulary boundary
  - structured output hooks that carry named OCR, table, layout, or document
    tensors plus resolved target-size metadata for later task-specific decoding
  - binary mask-logit restoration and uncompressed RLE helpers, plus generated
    mask crop utilities and automatic-mask filtering

- `image-processors/src/processors.rs`
  - `BlipImageProcessor` and `BlipImageProcessorConfig` for Diffusers
    BLIP exact resize, OpenAI CLIP normalization, and VAE-style postprocessing
  - `VaeImageProcessor` and `VaeImageProcessorConfig`
  - `VaeImageProcessorLdm3d` for Diffusers LDM3D RGB plus depth pairs
  - `Flux2ImageProcessor` and `Flux2ImageProcessorConfig` for Diffusers
    Flux2 reference-image validation, area caps, concatenation, and VAE-style preprocessing
  - `VisualClozeProcessor` and `VisualClozeProcessorConfig` for Diffusers
    nested image grids, target masks, target positions, and upsampling inputs
  - `HunyuanVideo15ImageProcessor` and `HunyuanVideo15ImageProcessorConfig`
    for Diffusers HunyuanVideo 1.5 spatial bucket selection and VAE-style
    image/video preprocessing
  - `MarigoldImageProcessor` and `MarigoldImageProcessorConfig` for Diffusers
    Marigold max-edge preprocessing, bottom/right replicate padding, dense
    depth/normal/intrinsic/uncertainty restoration, and visualization/export helpers
  - `JoyImageEditImageProcessor` and `JoyImageEditImageProcessorConfig` for
    Diffusers JoyImage bucket resize plus center-crop reference preprocessing
  - `WanAnimateImageProcessor` and `WanAnimateImageProcessorConfig` for
    Diffusers Wan reference-image resize-fill preprocessing
  - `Ltx2VideoHdrProcessor` and `Ltx2VideoHdrProcessorConfig` for Diffusers
    LTX2 two-stage VAE-scale/f32 reference-video preprocessing and LogC3
    postprocessing
  - `ClipImageProcessor` and `ClipImageProcessorConfig`
  - `VitImageProcessor` and `VitImageProcessorConfig`
  - `VideoMaeImageProcessor` and `VideoMaeImageProcessorConfig`
  - `VivitImageProcessor` and `VivitImageProcessorConfig`
  - `QwenVlImageProcessor` and `QwenVlImageProcessorConfig`
  - `LlavaNextImageProcessor` and `LlavaNextImageProcessorConfig`
  - `PixtralImageProcessor` and `PixtralImageProcessorConfig`
  - `Idefics3ImageProcessor` and `Idefics3ImageProcessorConfig`
  - `Gemma3ImageProcessor` and `Gemma3ImageProcessorConfig`
  - `MllamaImageProcessor` and `MllamaImageProcessorConfig`
  - `DetrImageProcessor` and `DetrImageProcessorConfig`
  - `ShortestEdgeResizeConfig`
  - `SamImageProcessor` and `SamImageProcessorConfig`
  - `DocumentOcrImageProcessor` and `DocumentOcrImageProcessorConfig`
  - `EncoderImageProcessor`, `EncoderImageProcessorConfig`, and
    `EncoderImageProcessorPreset` for 19 audited encoder/classifier families
  - `Swin2SrImageProcessor` and `Swin2SrImageProcessorConfig` for restoration
  - `TaskVisionImageProcessor`, `TaskVisionImageProcessorConfig`, and
    `TaskVisionProcessorPreset` for 34 detection, segmentation, depth, and
    keypoint/matching/pose families
  - fixture-backed query-mask restoration: semantic and panoptic outputs for
    EOMT, MaskFormer, Mask2Former, and OneFormer, plus instance outputs for
    EOMT, MaskFormer, and Mask2Former
  - an explicit external dataset-metadata error for OneFormer instance output;
    the Rust API does not invent `class_info_file` or thing-class metadata
  - `MultimodalProcessor` and `MultimodalPreset` for 52 compact audited
    vision-language, document-understanding, and video processor profiles
  - `ProcessorFamilyConfig` and `ProcessorConfigError` for HF `preprocessor_config.json` ingest
  - processor-family configs with explicit device-agnostic image and video layouts
  - processor-family resize decision helpers
  - Diffusers VAE preprocessing and image/video postprocessing modes through `VaeOutputType`
  - Diffusers inpaint preprocessing and crop-aware postprocessing through `VaeImageProcessor`
  - Diffusers LDM3D preprocessing and postprocessing through `DepthMapU16`
  - CLIP/ViT explicit `size`, `crop_size`, and `do_center_crop` config
  - DETR shortest-edge/longest-edge aspect-preserving typed output with
    configurable padding, valid-region `pixel_mask`, `original_sizes`, and
    `reshaped_input_sizes`
  - DETR batched object-detection postprocessing from model logits and
    normalized boxes into per-image typed predictions
  - SAM `pixel_values`, tensor-space zero padding, `original_sizes`, and
    `reshaped_input_sizes` output API
  - SAM batched mask-logit postprocessing into per-image original-size binary
    mask planes
  - Donut/document OCR resize, thumbnail, long-axis alignment, center padding,
    rescale, normalize, and full Transformers fixture parity for mixed
    portrait/landscape batches
  - Qwen/VLM patch-flattened `pixel_values`, `image_grid_thw`,
    `original_sizes`, and `reshaped_input_sizes` output API
  - Qwen2-VL single-image, multi-image, and video full fixture parity for
    patch-flattened pixel values and temporal-height-width grid metadata
  - Qwen/VLM decoded image-batch, image-sequence, and video-clip outputs with
    image grids for image-like inputs and clip-level temporal grids for videos
  - VideoMAE decoded-frame clip preprocessing with shortest-edge resize,
    center crop, frame-leading `NCHW`/`NHWC` `pixel_values`, and full
    Transformers fixture parity for synthetic decoded clips and image
    sequences; multiple decoded clips can be stacked as batched-video
    `BFCHW`/`BFHWC` tensors with batched full-value fixture parity for the
    channels-first path
  - ViViT decoded-frame clip preprocessing with upstream offset rescale,
    shortest-edge resize, center crop, frame-leading `pixel_values`, and full
    Transformers fixture parity for synthetic decoded clips and image
    sequences; multiple decoded clips can be stacked as batched-video
    `BFCHW`/`BFHWC` tensors with batched full-value fixture parity for the
    channels-first path
  - LLaVA-NeXT AnyRes patch extraction with patch-batched `NPCHW`
    `pixel_values`, `image_patch_counts`, `original_sizes`,
    `reshaped_input_sizes`, selected-size metadata, and full Transformers
    fixture parity for mixed patch-count batches plus a custom-grid case that
    records the pinned upstream revision ignoring a channels-last request
  - Pixtral patch-aligned resize with tensor-space spatial padding,
    `image_grid_thw`, `image_patch_counts`, `original_sizes`,
    `reshaped_input_sizes`, and full Transformers fixture parity for mixed
    resized-size batches
  - Idefics3 split-image local/global frames with nested batch padding,
    `pixel_attention_mask`, `image_patch_counts`, nested `rows`/`cols`, and
    full Transformers fixture parity for reduced split geometry
  - Gemma3 pan-and-scan crop expansion with flat `NCHW`/`NHWC`
    `pixel_values`, `num_crops`, `original_sizes`, `reshaped_input_sizes`,
    and full Transformers fixture parity for mixed crop/no-crop batches
  - Mllama tiled-canvas resize with raw canvas padding, nested image/tile
    batch packing as `NIPCHW`/`NIPHWC`, `num_tiles`, `aspect_ratio_ids`,
    `aspect_ratio_mask`, and full Transformers fixture parity for reduced
    tile geometry
  - recipe descriptions for CLIP, ViT, VideoMAE, ViViT, DETR, SAM,
    Donut/document OCR geometry, Qwen/VLM, Pixtral, LLaVA-NeXT patch-grid,
    split-image geometry used by Idefics3, Gemma3 aspect-ratio crops, tiled-canvas geometry
    used by Mllama, Diffusers VAE configs, and reusable decoded-frame sampling
  - DETR object-detection and binary-mask postprocess descriptors attached to
    their family recipes, with shared descriptor execution also covering
    semantic, instance, and panoptic segmentation, depth, dense maps,
    named dense-map target groups, video tensor transfers, coordinates, and
    output hooks

- `image-processors/src/transforms.rs`
  - `resize_frame`
  - `resize_frame_with_decision`
  - `pad_frame`
  - `pad_frame_with_canvas_fill`
  - `center_crop_frame`
  - `crop_frame`
  - `overlay_frame`
  - `composite_mask_frame`
  - `crop_region`
  - `select_patch_grid_resolution`
  - `patch_grid_plan`
  - `patch_grid_image_patches`
  - `patch_grid_image_patches_with_decision`
  - `patch_grid_batch_plan`
  - `patch_aligned_resize_plan`
  - `patch_aligned_resize_size`
  - `spatial_batch_padding_plan`
  - `supported_tiled_canvas_grids`
  - `tiled_canvas_plan`
  - `tiled_canvas_aspect_ratio_id`
  - `tiled_canvas_aspect_ratio_mask`
  - `tiled_canvas_batch_metadata`
  - `nested_image_grid_metadata`
  - `split_image_resize_size`
  - `split_image_encoder_size`
  - `split_image_plan`
  - `split_image_batch_metadata`
  - `nested_frame_batch_padding_plan`
  - `aspect_ratio_crop_plan`
  - `aspect_ratio_crop_batch_metadata`
  - `should_rotate_to_match_orientation`
  - `fit_inside_size`
  - `centered_padding`
  - `prepare_coco_detection_annotation`
  - `resize_detection_annotation`
  - `normalize_detection_annotation`
  - `pad_normalized_detection_annotation`
  - `post_process_object_detection`
  - `post_process_instance_segmentation`
  - `post_process_panoptic_segmentation`
  - `post_process_semantic_segmentation`
  - `detection_box_iou`
  - `non_max_suppression`
  - `scale_image_point`
  - `scale_detection_box`
  - `normalized_point_grid`
  - `generate_layered_crop_boxes`
  - `box_near_crop_edge`
  - `pad_crop_mask`
  - `filter_generated_masks`
  - `binary_mask_to_box`
  - `binarize_mask`
  - `mask_stability_score`
  - `binary_mask_to_rle`
  - `binary_rle_to_mask`
  - `resize_padded_mask_logits`
  - `post_process_binary_mask`
  - `post_process_generated_masks`
  - `scale_factor_resize_plan`
  - `is_latent_channel_count`
  - `normalize_unit_to_signed`
  - `denormalize_signed_to_unit`
  - `logc3_to_linear`
  - `binarize_mask_to_unit_f32`
  - `inpaint_overlay`
  - `validate_image_size_constraints`
  - `select_aspect_ratio_bucket`
  - `video_size_bucket_plan`
  - `video_clip_size_bucket_plan`
  - `shortest_edge_resize_size`
  - `longest_edge_resize_size`
  - `area_resize_plan`
  - `resize_frame_to_area_limit`
  - `validate_image_crop_box`
  - `center_crop_box`
  - `resize_center_crop_plan`
  - `resize_center_crop_frame`
  - `multiple_of_resize_plan`
  - `pad_to_multiple_plan`
  - `pad_frame_to_multiple_with_canvas_fill`
  - `unpad_frame_to_size`
  - `padded_image_size`
  - `validate_padding_fill`
  - `resize_fill_plan`
  - `resize_frame_to_fill_frame`
  - `resize_rgb_to_fill_frame`
  - `concatenate_frames_horizontally_rgb`
  - `downsample_attention_mask`
  - `OverlayPosition`
  - `Padding`
  - `PadToMultiplePlan`
  - `ImageCropBox`
  - `ImageSize`
  - `CocoObjectAnnotation`
  - `DetectionBoundingBox`
  - `DetectionCenterBox`
  - `DetectionAnnotation`
  - `NormalizedDetectionAnnotation`
  - `ObjectDetectionPrediction`
  - `SemanticSegmentationPrediction`
  - `SegmentationPostProcessOptions`
  - `SegmentationPostProcessPrediction`
  - `SegmentationSegmentInfo`
  - `ImagePoint`
  - `LayeredCropBox`
  - `CropGenerationOptions`
  - `MaskFilterOptions`
  - `BinaryRleMask`
  - `FilteredMask`
  - `GeneratedMaskPrediction`
  - `ScaleFactorResizePlan`
  - `ImageSizeConstraints`
  - `AreaResizePlan`
  - `ResizeCenterCropPlan`
  - `MultipleOfResizePlan`
  - `ResizeFillPlan`
  - `ResizeRounding`
  - `CanvasFill`
  - `RgbCanvasFill`
  - `AttentionMaskDownsample`
  - `PatchGridPlan`
  - `PatchGridFrames`
  - `PatchGridBatchPlan`
  - `TiledCanvasGrid`
  - `TiledCanvasPlan`
  - `TiledCanvasBatchMetadata`
  - `SplitImagePlan`
  - `SplitImageBatchMetadata`
  - `NestedFrameBatchPaddingPlan`
  - `AspectRatioCropOptions`
  - `AspectRatioCrop`
  - `AspectRatioCropPlan`
  - `AspectRatioCropBatchMetadata`
  - `PatchAlignedResizePlan`
  - `SpatialBatchPaddingPlan`
  - `ResizeLimits`
  - `ResizeDecision`
  - `ResizeKernel`
  - `ResizeMode`
  - `ResizeFilter`
  - `ResizeParity`

## Example

```rust
use image_processors::{
    ImageFrame, ImageProcessor, ImageProcessorConfig, Layout, PixelFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NCHW,
        ..Default::default()
    })?;

    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0])?;
    let tensor = processor.preprocess_image(&frame)?;

    assert_eq!(tensor.shape(), [1, 3, 1, 1]);
    Ok(())
}
```

Video decoding is represented in the Rust API and is available behind the optional `video` feature through `video-rs`. That feature uses FFmpeg libraries through Rust bindings, so it requires FFmpeg development files discoverable by `pkg-config` or `vcpkg`; an `ffmpeg.exe` alone is not enough. See `docs/FFMPEG.md` for the Windows vcpkg setup and package-manager expectations.

## Development

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo clippy -p image-processors --all-targets --no-default-features -- -D warnings
cargo test -p image-processors --no-default-features
cargo clippy --workspace --all-targets --features url -- -D warnings
cargo test --workspace --features url
```

The `video` feature is intentionally not part of default verification unless the machine has FFmpeg development libraries installed:

```powershell
.\scripts\setup-vcpkg-ffmpeg.ps1 -CheckCargo
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Resize operations go through the workspace `image-resize-kernels` crate.
Generic `ImageProcessorConfig` defaults to the fast resampling profile, while
processor-family wrappers default to the compatibility profile so they can
track Transformers/Diffusers parity. Default builds use the Rust `image`/zune
JPEG decoder and automatic batch execution, which keeps small batches serial
and switches larger batches to parallel execution when the `parallel` feature
is enabled. Run the CLIP batch benchmark with real image files:

```powershell
cargo bench -p image-processors --bench clip_batch -- --image path\to\image.jpg --batch-sizes 1,2,4,8,16,32 --repetitions 5 --warmup 1
```

Compare optional TurboJPEG decode with:

```powershell
cargo bench -p image-processors --features turbojpeg --bench clip_batch -- --image path\to\image.jpg --decode-backend turbojpeg --batch-sizes 1,2,4,8,16,32 --repetitions 5 --warmup 1
```

Install NASM or a system libjpeg-turbo package if you want the native backend
to use libjpeg-turbo's assembly/SIMD paths; the `turbojpeg` feature itself does
not require SIMD so feature builds remain portable.

The benchmark loads files from disk, applies CLIP-style preprocessing, stacks
the batch tensor, and reports elapsed time for each batch size. Add
`--all-stages` to split decode, resize, tensor conversion, and predecoded batch
timings. Use `--decode-backend image|turbojpeg` and
`--batch-execution auto|serial|parallel` to compare decode and execution
behavior on the same harness.

For directory-scale throughput, use the dataset benchmark:

```powershell
cargo bench -p image-processors --bench clip_dataset -- `
  --image-dir path\to\images --batch-size 256 --warmup-batches 2 `
  --resize-profile compatibility --batch-execution auto --reuse-workspace --skip-errors
```

The comparable Transformers/Pillow baseline is:

```powershell
.\.venv\Scripts\python.exe scripts\bench\transformers_clip_dataset.py `
  --image-dir path\to\images --batch-size 256 --warmup-batches 2 --skip-errors
```

The Python baseline uses a `ThreadPoolExecutor` for image open/convert by
default; pass `--loader-workers` to tune the loader parallelism. The following
historical batch-size `256` results predate the exact fixed-point compatibility
kernel and should be regenerated before making current throughput claims. The
run processed `32,942` valid images and skipped `6` corrupt JPEGs:

| Path | Total | ms/image | images/sec |
| --- | ---: | ---: | ---: |
| Rust compatibility profile | `11.7 s` | `0.355919` | `2809.626` |
| Rust fast profile | `7.5 s` | `0.227595` | `4393.763` |
| Transformers/Pillow threaded loader | `203.0 s` | `6.161349` | `162.302` |

Optional Transformers parity fixture generation is documented in
`docs/PARITY.md`.

## Implemented Milestones

1. Stabilize Rust media source and decoded frame ownership.
2. Stabilize Rust tensor/layout abstractions.
3. Implement a general `ImageProcessor` as the first full Rust processor.
4. Harden optional URL and `video-rs` backend setup, behavior, and CI coverage.
5. Add processor families over shared Rust media, tensor, and transform primitives.
6. Support quantized and packed tensor storage where model processors need compact buffers.
7. Make resize kernel and parity decisions explicit for high-quality model preprocessing.
8. Keep bindings out of tree until the Rust API is coherent and tested.
