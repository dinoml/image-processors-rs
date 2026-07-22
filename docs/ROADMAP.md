# Roadmap

## Current Scope

The repository is Rust-only for now. The goal is to get the core architecture right before adding Python, Node, WASM, or other bindings.

## Implemented

- Rust workspace with the `image-resize-kernels` and `image-processors` crates.
- Rust media layer:
  - `MediaSource`
  - `MediaLocation`
  - `MediaHint`
  - `MediaType`
  - `DefaultMediaLoader`
  - decoded `ImageFrame`
  - `ImageSequence`
  - `FrameTiming`
  - `FrameSampling`
  - GIF/APNG/WebP animated image decoding to owned frames
  - `VideoFrame` and `VideoClip` API shape
  - image path loading
  - encoded image byte decoding
  - optional URL image loading through `reqwest`
  - optional video decoding through `video-rs`
  - explicit image/video URL source constructors
  - explicit buffered/streaming remote read policy
  - frame sampling for image sequences, animated image sources, video clips,
    and decoded video sources
  - disabled URL-loading behavior tests for non-`url` builds
  - disabled video-decoding behavior tests for non-`video` builds
  - animated still-image parity tests across GIF/APNG/WebP
  - decoded video fixture and source-level tests for frame sampling and
    timestamp metadata
- Native backend setup:
  - vcpkg FFmpeg manifest
  - Windows FFmpeg setup script
  - FFmpeg development-library docs
  - Windows, Linux, and macOS CI lanes for all-feature `video-rs` checks and tests
- Rust tensor layer:
  - `Tensor`
  - `TensorData`
  - `TensorDataView`
  - `TensorView`
  - `DType`
  - `Layout` with image/video layouts plus row-major `NC` patch matrices
  - `ImageLayout` and `VideoLayout`
  - float32, float16, bfloat16, uint8, integer, and bool storage variants
  - affine quantized u8/i8 storage with explicit quantization parameters
  - packed u4/i4 storage with logical element counts
  - shape/layout validation
  - zero-copy tensor views over decoded frame bytes
  - scalar rescale and per-channel normalization kernels
- Rust processor layer:
  - `ImageProcessor`
  - `ImageProcessorConfig`
  - `ImageProcessorOptions`
  - `ImageProcessorWorkspace`
  - file/source loading through `ImageProcessor::open` and `ImageProcessor::open_source`
  - typed output APIs through `ImageProcessor::open_output` and related methods
  - reusable batch preprocessing through `ImageProcessor::open_batch_into`
  - parallel preprocessing for borrowed decoded image batches
  - zero-copy decoded-frame views through byte-preserving `ImageProcessor` configs
  - per-frame zero-copy sequence and video views through byte-preserving `ImageProcessor` configs
  - processor config helpers for semantic image and video output layouts
- Rust recipe layer:
  - reusable, serde-validated preprocessing stage descriptions
  - serde-validated postprocess-only recipes for model-output restoration
    when source loading and generic preprocessing stages are not part of the
    recipe
  - generic lowering for directly composable image processor stages
  - pixel-format-preserving fill/canvas resize descriptors
  - constant, edge-extension, and reflection canvas padding descriptors
  - bottom/right pad-to-multiple plans with crop-back helpers for
    replicate-padding workflows
  - temporal repeat-last padding plans plus sequence/clip helpers for
    incomplete video groups
  - frame-count plus spatial video bucket selection plans
  - patch-stage helpers for image and temporal `image_grid_thw` metadata plus
    patch flattening execution backed by the shared repeat-last plan, used by
    Qwen and non-Qwen patch-grid profiles
  - postprocess descriptors for object detection, segmentation, binary masks,
    depth, dense image-like maps, optional named dense-map targets,
    LogC3 video tensor transfers, keypoints/pose coordinates,
    tokenizer-independent structured token sequences, and structured output
    hooks for OCR, table, layout, or document tensors
  - DETR and SAM family recipes include their object-detection and binary-mask
    postprocess descriptors, with shared descriptor execution also covering
    semantic/instance/panoptic segmentation, depth maps, dense maps, named
    dense-map target groups, video tensor transfers, normalized coordinates,
    CTC/BOS/EOS token-id decoding, and output hooks
  - a common postprocess-only LogC3 HDR video recipe and LTX2 wrapper cover
    default two-stage VAE-scale/f32 reference-video conditioning and decoded
    HDR outputs through the generic video tensor descriptor for `BFCHW` and
    `BFHWC` batches
- Rust processor output layer:
  - `ProcessorOutput`
  - `ProcessorTensorName`
  - `ProcessorMetadataName`
  - named `pixel_values` tensor output
  - named `pixel_mask` tensor output
  - named `original_sizes`, `reshaped_input_sizes`, `image_grid_thw`, and
    nested image-grid metadata
- Rust compatibility catalog:
  - immutable lookup across 119 Transformers and 12 Diffusers entries
  - canonical plus alias coverage for all 209 audited Transformers class names
  - stable recipe/family mappings pinned to exact upstream commits
  - complete-payload `FixtureParity` status for every catalog entry
  - bidirectional evidence checks connecting every fixture-parity canonical
    class and alias to a checked-in fixture at the catalog audit commit, with a
    declared Qwen2-VL family mapping for its auxiliary video-processor fixture
- Rust postprocess namespace:
  - public `postprocess` module for object-detection coordinate restoration,
    score filtering, semantic/instance/panoptic segmentation outputs, and
    binary mask/RLE restoration helpers, plus depth map resize-back, dense-map
    resize-back with optional target names, normalized coordinate restoration,
    unit-range image/video tensor-to-frame helpers, and LogC3 HDR video tensor
    restoration
  - recipe-driven postprocess execution for DETR object-detection,
    semantic/instance/panoptic segmentation, binary-mask, depth, dense-map,
    video tensor transfer, coordinate, structured token-id, and output-hook
    descriptors
  - exact-fixture query-mask semantic and panoptic wrappers for EOMT,
    MaskFormer, Mask2Former, and OneFormer, with instance wrappers for EOMT,
    MaskFormer, and Mask2Former
  - explicit external-metadata rejection for OneFormer instance output and
    explicit rejection for processors that use a different mask contract
  - descriptor-backed LogC3 HDR video postprocessing is catalogable without a
    dedicated processor-named postprocess variant
  - shared implementations remain available through `transforms` while the
    task-facing API has a dedicated namespace
- Rust processor families:
  - CLIP/ViT image processor configs and wrappers
  - CLIP/ViT explicit `size`, `crop_size`, and `do_center_crop` controls
  - Qwen/VLM smart-resize image processor config and wrapper
  - Qwen/VLM patch-flattened `pixel_values`, `image_grid_thw`,
    `original_sizes`, and `reshaped_input_sizes` output metadata
  - DETR image processor config and wrapper
  - DETR aspect-preserving typed output with configurable pad-to-max,
    explicit pad size, size-divisor rounding, valid-region `pixel_mask`,
    `original_sizes`, and `reshaped_input_sizes`
  - DETR batched object-detection postprocessing from model logits and
    normalized boxes into per-image typed predictions
  - SAM image processor config and wrapper
  - SAM longest-edge resize, square padding, `original_sizes`, and `reshaped_input_sizes` output bundle
  - SAM batched mask-logit postprocessing into per-image original-size binary
    mask planes
  - document/OCR image processor config and wrapper
  - BLIP image processor config and wrapper for exact resize, OpenAI CLIP
    normalization, and VAE-style postprocessing
  - Flux2 image processor config and wrapper for reference-image validation,
    area caps, horizontal concatenation, and VAE-style preprocessing
  - VisualCloze image processor config and wrapper for nested image grids,
    target masks, target positions, and upsampling inputs
  - HunyuanVideo 1.5 image processor config and wrapper for spatial bucket
    selection and VAE-style decoded image/video preprocessing
  - Marigold image processor config and wrapper for canonical RGB loading,
    max-edge tensor resize, bottom/right replicate padding, dense output
    restoration, and visualization/export helpers
  - JoyImage edit image processor config and wrapper for bucket selection plus
    resize-center-crop reference preprocessing
  - Wan Animate image processor config and wrapper for target rounding plus
    resize-fill reference preprocessing
  - 19 encoder/classifier presets plus a Swin2SR restoration wrapper
  - 34 task-vision presets spanning detection/grounding, segmentation/matting,
    depth/geometry, and keypoint/matching/pose
  - 52 compact multimodal presets spanning vision-language,
    document-understanding, and video inputs
  - explicit device-agnostic image and video layout helpers on processor-family configs
  - explicit resize decision helpers on processor-family configs
  - reusable batch preprocessing through fixed-family `open_batch_into` wrappers
- Parity tooling:
  - optional Transformers fixture generator for CLIP/ViT and multi-output family references
  - exact-commit full-payload generators for encoder/restoration, task-vision,
    and multimodal preset collections
  - exact-commit full-payload generator for all cataloged Diffusers processors
  - source/config/comparison contracts embedded in parity fixtures and
    validated by the Rust tests
  - parity comparison policy documentation
- Rust resize kernel crate:
  - fast resize profile boundary
  - Pillow/Transformers compatibility profile boundary
  - in-tree Pillow fixed-point coefficients with exact per-pass `u8` rounding
    for bilinear, bicubic, and Lanczos filters
  - Torchvision-compatible fixed-point `u8` resize plus float resize for
    rescale-before-resize processor paths
  - crop-aware compatibility resizing that is value-consistent with full
    resize followed by crop
  - reusable fixed-point coefficient caches and intermediate `u8` scratch
    storage through processor workspaces
  - per-worker resize workspaces for parallel path batch preprocessing
  - automatic batch scheduler that keeps small batches serial and uses parallel
    execution at larger batch sizes when available
  - exact-commit full-payload fixtures guard Pillow and Torchvision resize drift
- Rust transforms:
  - `ImageSize`
  - `Padding`
  - `ResizeLimits`
  - `ResizeDecision`
  - `ResizeKernel`
  - `ResizeParity`
  - impl-based size rounding
  - impl-based Qwen-style smart resize geometry
  - LLaVA-NeXT AnyRes resolution selection, patch-frame extraction, and
    patch-count batch padding plans
  - patch-aligned resize planning and right/bottom spatial batch padding plans
  - Mllama tile-grid selection, aspect-ratio ids, aspect-ratio masks, and
    nested batch tile metadata plans
  - nested image-grid layout metadata with validated row/column image sizes,
    per-cell target masks, and derived target positions
  - split-image resize planning, row/column metadata, and
    nested-frame `pixel_attention_mask` padding plans
  - Gemma3 pan-and-scan crop planning and `num_crops` batch metadata
  - COCO/DETR detection annotation contracts for clipped boxes, resize scaling,
    normalized center-format labels, padded-canvas box updates, and
    object-detection plus semantic, instance, and panoptic segmentation
    post-processing
  - image-point and detection-box scaling, layered crop-box generation, mask
    box extraction, threshold/stability helpers, crop-edge mask filtering,
    crop-mask padding, uncompressed RLE mask conversion, mask-logit resize back
    to original image sizes, and class-agnostic generated-mask NMS
  - common scale-factor resize plans used by Diffusers VAE processors,
    latent-channel detection,
    denormalization, LogC3 HDR inverse transfer, mask binarization helpers,
    attention-mask downsampling, and shared aspect-ratio bucket plus
    resize-center-crop planning used by PixArt-style tensor crops
  - HF `preprocessor_config.json` ingest into strong Rust processor-family
    configs for the implemented CLIP, ViT, Qwen/VLM, DETR, SAM, Flux2,
    JoyImage, Wan Animate, LTX2 HDR, and document/OCR families
  - explicit resize kernel and parity policy for high-quality preprocessing
  - constant-fill frame padding
  - pixel-format-preserving fill/canvas resize
  - edge-extension and reflection frame canvas padding
  - center crop frame kernel
  - clipped frame overlay
  - mask-based frame compositing
- Rust image processor behavior:
  - image-to-NCHW preprocessing
  - image batching
  - resize/crop/fill helpers
  - RGB/grayscale conversion
- Bench and backend decisions:
  - staged CLIP batch benchmark for decode, resize, tensor conversion, and predecoded batches
  - directory-scale CLIP dataset benchmark for compatibility, fast, and
    workspace-reuse paths
  - threaded Transformers/Pillow development baseline for Python comparison
  - `--resize-profile`, `--decode-backend`, `--batch-execution auto|serial|parallel`,
    and `--reuse-workspace` benchmark switches
  - measured TurboJPEG comparison; default `image` decoder remains the practical default for now

## Next Milestones

The next phase is less about adding processor names and more about hardening
the current parity surface, keeping long modules navigable, and making future
upstream audits cheap to classify.

1. Deepen CLIP/ViT parity:
   - Keep the fixed-point Pillow and Torchvision references green across larger
     resize/crop regimes.
   - Optimize fixed-point RGB accumulation without changing per-pass rounding.
   - Recover fused NCHW write throughput only when complete-payload fixtures
     prove the optimized path remains exact.
2. Broaden exact-commit full-payload fixtures:
   - Keep every fixture-parity catalog entry backed by complete tensor and
     metadata payload comparisons.
   - Add larger resize/crop cases for existing processor families.
   - Keep fixture provenance pinned so future upstream drift is visible.
3. Broaden reusable workspace APIs:
   - Reuse additional resize temporary buffers where practical.
   - Extend workspace recycling beyond the final tensor output and fixed-point
     intermediate buffers.
   - Keep direct resize and tensor-writer helpers small enough to test in
     isolation.
4. Continue module layout cleanup:
   - Move self-contained processor families and helper clusters out of
     `processors.rs`.
   - Split large transform and recipe helper groups behind stable re-exports.
   - Keep public paths stable while shortening implementation files.
5. Formalize decode backend decisions:
   - Keep the Rust `image` decoder as the default for now based on current
     local CLIP batch measurements.
   - Revisit optional TurboJPEG only if a target deployment proves faster with
     its native setup and RGB decode behavior.
   - Document native setup and portability tradeoffs if TurboJPEG becomes a
     recommended path.
6. Keep catalog scans current:
   - Classify future upstream processor additions conservatively before
     advertising parity.
   - Promote catalog statuses only when fixture coverage matches the documented
     output contract.

## Deferred

- Bindings remain out of tree. Revisit them only after a binding-specific
  request and without moving media, tensor, or processor ownership back out of Rust.
