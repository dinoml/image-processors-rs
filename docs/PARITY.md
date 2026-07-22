# Processor Parity

The Rust APIs remain the source of ownership and preprocessing behavior. Python
is used only to generate reference fixtures from established processors.

## Transformers Fixture Generation

Use the optional parity script with Transformers imported from the exact clean
audit checkout. The script rejects installed packages, a different Git HEAD,
or tracked source modifications:

```powershell
$transformersRoot = "C:\src\transformers"
$env:PYTHONPATH = "$transformersRoot\src"
git -C $transformersRoot rev-parse HEAD
git -C $transformersRoot status --short
```

The first command must print
`6d960ca0a0eba0d2aebc920d8080a9353da468d3`; the second must be empty.

Then record the dedicated family references:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family clip `
  --family vit `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_clip_vit.json
```

The script generates deterministic RGB input when `--image` is omitted. Each
`image-processors.transformers-parity.v1` fixture records the generator, input
formula/parameters, and exact upstream commit; each case records its class,
backend, model/config source, effective processor config, complete outputs and
metadata, and comparison policy. Canonical
Torchvision and `*Pil` backends are generated separately by default. With
`--include-output-data`, full tensors are stored losslessly as base64; with
`--include-stage-data`, full intermediate tensors are stored for whole-image
drift diagnostics. Checked dedicated fixtures cover CLIP/ViT, VideoMAE, ViViT,
DETR, SAM, Donut/document OCR, Qwen2-VL, LLaVA-NeXT, Pixtral, Idefics3, Gemma3,
and Mllama multi-output contracts.

Generate the DETR fixture with:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family detr `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_detr.json
```

Generate the SAM fixture with:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family sam `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_sam.json
```

Generate the Donut/document OCR fixture directly from the processor class. The
fixture uses a reduced document canvas to keep the checked tensor compact:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family donut `
  --width 13 `
  --height 17 `
  --image-count 2 `
  --width-step 4 `
  --height-step -6 `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_donut.json
```

Generate the Qwen2-VL fixture for both audited backends:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family qwen_vl `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_qwen_vl.json
```

Generate the Qwen2-VL multi-image fixture with deterministic generated sizes:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family qwen_vl `
  --image-count 2 `
  --width-step 6 `
  --height-step -6 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_qwen_vl_multi_image.json
```

Generate the Qwen2-VL video fixture with synthetic decoded frames:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family qwen_vl `
  --as-video `
  --image-count 3 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_qwen_vl_video.json
```

Generate the VideoMAE fixture with synthetic decoded frames. The fixture uses a
reduced `4x4` resize/crop target and nearest-neighbor sampling to keep the
checked tensor compact and exact:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family videomae `
  --width 5 `
  --height 5 `
  --image-count 3 `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_videomae.json
```

Generate the VideoMAE batched-video fixture with two same-length decoded clips:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family videomae `
  --width 5 `
  --height 5 `
  --image-count 3 `
  --video-batch-size 2 `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_videomae_batch.json
```

Generate the ViViT fixture with synthetic decoded frames. The fixture uses the
same reduced geometry as VideoMAE but keeps ViViT's default offset rescale path:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family vivit `
  --width 5 `
  --height 5 `
  --image-count 3 `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_vivit.json
```

Generate the ViViT batched-video fixture with two same-length decoded clips:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family vivit `
  --width 5 `
  --height 5 `
  --image-count 3 `
  --video-batch-size 2 `
  --include-output-data `
  --include-stage-data `
  --output image-processors\tests\fixtures\transformers_vivit_batch.json
```

Generate the LLaVA-NeXT AnyRes fixture directly from the processor class:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family llava_next `
  --width 13 `
  --height 17 `
  --image-count 2 `
  --width-step 487 `
  --height-step 483 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_llava_next.json
```

Generate the LLaVA-NeXT custom-grid fixture directly from the processor class.
At the pinned audit revision, both canonical and PIL backends return
channels-first patch tensors even when a channels-last data format is
requested; the legacy fixture filename is retained for continuity:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family llava_next_nhwc `
  --width 9 `
  --height 5 `
  --image-count 2 `
  --width-step -5 `
  --height-step 2 `
  --sample 24 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_llava_next_nhwc.json
```

Generate the Pixtral patch-aligned resize fixture directly from the processor
class:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family pixtral `
  --width 17 `
  --height 33 `
  --image-count 2 `
  --width-step 31 `
  --height-step -17 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_pixtral.json
```

Generate the Idefics3 split-image fixture directly from the processor class.
The fixture uses reduced tile geometry to keep the checked tensor compact:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family idefics3 `
  --width 9 `
  --height 7 `
  --image-count 2 `
  --width-step -5 `
  --height-step 4 `
  --sample 32 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_idefics3.json
```

Generate the Mllama tiled-image fixture directly from the processor class. The
fixture uses reduced tile geometry to keep the checked tensor compact:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family mllama `
  --width 13 `
  --height 17 `
  --image-count 2 `
  --width-step 3 `
  --height-step 2 `
  --sample 24 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_mllama.json
```

Generate the Gemma3 pan-and-scan fixture directly from the processor class. The
fixture uses reduced image size and crop geometry to keep the checked tensor
compact:

```powershell
python scripts\parity\transformers_image_parity.py `
  --family gemma3 `
  --width 13 `
  --height 17 `
  --image-count 2 `
  --width-step 3 `
  --height-step 2 `
  --include-output-data `
  --output image-processors\tests\fixtures\transformers_gemma3.json
```

Run the Rust Transformers fixture assertions with:

```powershell
cargo test -p image-processors --test clip_vit_parity
cargo test -p image-processors --test detr_parity
cargo test -p image-processors --test sam_parity
cargo test -p image-processors --test document_ocr_parity
cargo test -p image-processors --test qwen_vl_parity
cargo test -p image-processors --test videomae_parity
cargo test -p image-processors --test vivit_parity
cargo test -p image-processors --test llava_next_parity
cargo test -p image-processors --test pixtral_parity
cargo test -p image-processors --test idefics3_parity
cargo test -p image-processors --test gemma3_parity
cargo test -p image-processors --test mllama_parity
```

The remaining cataloged Transformers classes are grouped by observable Rust
behavior instead of mirroring the upstream Python hierarchy. Generate their
exact-commit fixtures with:

```powershell
python scripts\parity\transformers_encoder_restoration_parity.py `
  --transformers-source $transformersRoot
python scripts\parity\transformers_task_vision_parity.py `
  --source $transformersRoot `
  --output image-processors\tests\fixtures\transformers\catalog_task_vision.json
python scripts\parity\transformers_multimodal_parity.py `
  --transformers-root $transformersRoot
python scripts\parity\transformers_query_mask_postprocess.py `
  --transformers-root $transformersRoot `
  --output image-processors\tests\fixtures\transformers\query_mask_postprocess.json
```

These fixtures cover 20 encoder/restoration, 34 task-vision, and 52
multimodal/document/video canonical entries, plus 17, 23, and 38 audited class
aliases respectively. Encoder and task-vision inputs force real resize work;
the multimodal fixture adds a separate non-no-op resize probe for every
canonical preset and every alias. The fixtures store complete numeric and
metadata payloads. Their root acceptance contracts record full-payload and
statistics absolute/relative tolerances; integer, byte, boolean, shape, dtype,
and metadata comparisons are exact. Run them with:

```powershell
cargo test -p image-processors --test encoder_catalog_parity
cargo test -p image-processors --test task_vision_catalog_parity
cargo test -p image-processors --test multimodal_catalog_parity
cargo test -p image-processors --test query_mask_postprocess_parity
cargo test -p image-processors --test catalog_fixture_coverage
```

## Diffusers Fixture Generation

Use the Diffusers parity script to record complete deterministic references for
VAE, IP-Adapter mask, LDM3D, BlipDiffusion, Flux2, VisualCloze,
HunyuanVideo 1.5, Marigold, JoyImage Edit, Wan Animate, LTX2 HDR video, and
PixArt image-processor behavior. The generator rejects any Diffusers checkout
whose Git HEAD or audited-source manifest differs from the recorded commit;
the fixture also pins its NumPy, Pillow, and torch generator versions:

```powershell
python scripts\parity\diffusers_image_parity.py `
  --diffusers-source C:\src\diffusers `
  --output image-processors\tests\fixtures\diffusers\image_processors.json
```

Run the Rust Diffusers fixture assertions with:

```powershell
cargo test -p image-processors --test diffusers_parity
```

## Comparison Policy

Parity tests should compare:

- output names, such as `pixel_values` and `pixel_mask`
- metadata names, such as `original_sizes`, `reshaped_input_sizes`,
  `image_grid_thw`, and `video_grid_thw`
- tensor shapes and layouts
- intermediate processed-image bytes before rescale and normalization
- numeric values with explicit tolerances

The tolerance and exactness rules are fixture data, not unexplained constants
inside the tests. Dedicated cases carry a `comparison` object; aggregate suites
carry a validated root `acceptance.comparison` policy plus narrowly scoped
overrides where needed. `catalog_fixture_coverage` also proves the evidence
relationship in both directions: every `FixtureParity` canonical class and
alias is present at its catalog audit commit, and every catalog-bound fixture
identity resolves back to the catalog. `Qwen2VLVideoProcessor` is a declared
auxiliary fixture for the cataloged Qwen2-VL family, and its commit and family
link are asserted separately.

Fixture validation currently caps absolute tolerances at `0.06` and requires a
zero relative tolerance. Raising either bound therefore requires a reviewed
test-contract change; editing fixture JSON alone cannot make parity vacuous.

Avoid shape-only parity tests for processors that rescale, normalize, crop, or
pad pixels. Shape-only tests are useful as smoke tests, but they are not enough
to validate processor compatibility.

## CLIP/ViT Decisions

The checked fixture currently covers:

- CLIP: `openai/clip-vit-base-patch32`
  - shortest-edge resize to `224`
  - bicubic resize
  - center crop to `224x224`
  - CLIP mean/std normalization
- ViT: `google/vit-base-patch16-224-in21k`
  - direct resize to `224x224`
  - bilinear resize
  - no center crop
  - `[0.5, 0.5, 0.5]` mean/std normalization

The Rust test compares `pixel_values` shape, `NCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, and the pre-normalization
`processed_image` stage. Processor-family wrappers default to
`ResizeParity::Compatibility` because their purpose is to replace upstream
processors. The current absolute tolerances are:

- final full tensor values: `1e-6`
- final sampled values: `1e-6`
- min/mean/max: `0.01`
- processed-image max byte drift: `0`
- processed-image mean byte drift: `0`

`ResizeMode::Crop` uses resize-to-cover geometry plus center-crop semantics to
match Transformers. The compatibility path now asks `image-resize-kernels` to
materialize only the crop window when practical, rather than always allocating
the full resized image and cropping afterward. Resize implementation choices
live in the workspace `image-resize-kernels` crate. Its fast profile owns the
existing SIMD-oriented throughput path; its Pillow profile owns the in-tree
Pillow/Transformers compatibility kernel.

Strict byte parity is profile-specific. For the Pillow CLIP fixture the target
is Pillow bicubic over a PIL RGB input, including the same coordinate
transform, boundary handling, fixed-point coefficient math, intermediate
8-bit clipping, and u8 rounding. The in-tree Pillow profile now matches the
complete processed-image payload byte-for-byte; normalized f32 tensors are
compared element-by-element with a `1e-6` absolute tolerance. The audited
Torchvision profile likewise uses its own fixed-point coefficients and
rounding rules; canonical encoder presets and explicit `*Pil` aliases
therefore exercise and verify their respective backends instead of sharing an
approximate kernel.

## DETR Decisions

The checked DETR fixture covers `facebook/detr-resnet-50` with explicit
canonical Torchvision and `*Pil` cases. The deterministic `13x17` RGB input
resizes with DETR's default shortest-edge and longest-edge policy to
`1046x800`.

`DetrResizeConfig` preserves the upstream `shortest_edge` plus optional
`longest_edge` resize contract for typed DETR outputs. The default is
`shortest_edge = 800` and `longest_edge = 1333`.

The Rust test compares:

- `pixel_values` shape, `NCHW` layout, float32 storage, full tensor values,
  sampled values, and min/mean/max
- `pixel_mask` valid-region values exactly
- `original_sizes` and `reshaped_input_sizes` metadata

Transformers records DETR `pixel_mask` as int64 `[N,H,W]`. Rust keeps named
masks as bool `NCHW` tensors with shape `[N,1,H,W]`; the parity test compares
the same logical mask values after that representation conversion. Current
absolute tolerances are `0.05` for final tensor values and `0.01` for
min/mean/max.

## SAM Decisions

The checked SAM fixture covers `facebook/sam-vit-base` with explicit canonical
Torchvision and `*Pil` cases. The deterministic `13x17` RGB input resizes to
`1024x783` and then pads to `1024x1024`.

The Rust test compares `pixel_values` shape, `NCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, `original_sizes`, and
`reshaped_input_sizes`. SAM padding is tensor-space padding after
normalization, so the padded region stays zero. Current absolute tolerances are
`0.05` for final tensor values and `0.01` for min/mean/max.

## Qwen/VLM Decisions

The checked Qwen/VLM image fixtures cover `Qwen/Qwen2-VL-7B-Instruct` with
explicit canonical Torchvision and `*Pil` cases. The deterministic
single-image fixture uses a `13x17` RGB input that smart-resizes to `84x56`, producing
`image_grid_thw = [1, 6, 4]`. The multi-image fixture uses `13x17` and
`19x11` RGB inputs, producing `image_grid_thw = [[1, 6, 4], [1, 4, 6]]` and
concatenated `pixel_values` rows. The video fixture uses three `13x17` RGB
frames as one video input, pads the odd frame count by repeating the last
frame, and produces `video_grid_thw = [2, 6, 4]`.

The Rust test compares patch-flattened `pixel_values` shape, `NC` layout,
float32 storage, full tensor values, sampled values, min/mean/max, and
`image_grid_thw` for single-image and multi-image processor outputs. It also
compares Qwen video `pixel_values_videos` and `video_grid_thw` against the
crate's normalized video output API. Rust records `original_sizes` and
`reshaped_input_sizes` as typed metadata derived from each input and grid.
Current absolute tolerances are `0.05` for final tensor values and `0.01` for
min/mean/max.

The image fixtures contain explicit canonical Torchvision and `*Pil` cases;
backend selection never depends on an upstream default. The video case uses
the canonical video-capable path recorded by the fixture.

## VideoMAE Decisions

The checked VideoMAE fixture uses `VideoMAEImageProcessor` with synthetic
decoded frames and reduced geometry: `size={"shortest_edge": 4}`,
`crop_size={"height": 4, "width": 4}`, and nearest-neighbor resize. Transformers
emits `pixel_values` as `[batch, frames, channels, height, width]`; Rust
represents one decoded clip as a frame-leading tensor
`[frames, channels, height, width]` with `TensorLeadingAxis::Frames`, so the
parity assertion compares the same full tensor payload after dropping the
single upstream batch axis. The same fixture also checks decoded
`ImageSequence` preprocessing. When multiple decoded clips are preprocessed
together, Rust stacks them as `BFCHW`/`BFHWC` batched-video tensors. The
batched fixture keeps the full upstream `[batch, frames, channels, height,
width]` tensor payload and compares it directly against the Rust `BFCHW`
output.

## ViViT Decisions

The checked ViViT fixture uses `VivitImageProcessor` with synthetic decoded
frames and reduced geometry: `size={"shortest_edge": 4}`,
`crop_size={"height": 4, "width": 4}`, and nearest-neighbor resize. Unlike
VideoMAE, ViViT defaults to `rescale_factor=1/127.5` with `offset=True`, so
Rust folds the upstream `value * factor - 1.0` step into the tensor-stage
normalization stats. As with VideoMAE, Transformers emits
`[batch, frames, channels, height, width]` and Rust compares the full tensor
payload as `[frames, channels, height, width]` with
`TensorLeadingAxis::Frames` for both decoded `VideoClip` and `ImageSequence`
inputs. Rust also supports stacking multiple decoded clips as
`BFCHW`/`BFHWC` batched-video tensors. The batched fixture keeps the full
upstream `[batch, frames, channels, height, width]` tensor payload and
compares it directly against the Rust `BFCHW` output.

## LLaVA-NeXT Decisions

Rust now has an end-to-end `LlavaNextImageProcessor` for AnyRes patch
extraction and patch-batched `pixel_values`. The checked fixture uses
`LlavaNextImageProcessor()` directly, so it does not download from the Hub. It
records a `13x17` RGB input and a `500x500` RGB input, producing patch counts
`[7, 10]` and exercising patch-axis batch padding.

A second checked fixture uses a custom tiny grid with a
`data_format=channels_last` request. It records `9x5` and `4x7` RGB inputs,
selects canvases `[[8, 8], [8, 4]]`, and produces patch counts `[5, 3]`. The
pinned upstream revision ignores that request and returns channels-first data,
so Rust intentionally validates `NPCHW`; the legacy `_nhwc` filename is kept
for continuity.

The Rust tests compare `pixel_values` shape, `NPCHW` layout, float32
storage, full tensor values, sampled values, min/mean/max, Transformers
`image_sizes` against Rust `original_sizes`, plus `reshaped_input_sizes`,
`image_patch_counts`, selected AnyRes sizes, and derived patch grids. Current
absolute tolerances are `0.05` for final tensor values and `0.01` for
min/mean/max. LLaVA-NeXT patch extraction uses the same compatibility resize
decision as the family processor so the base image and high-resolution tiles
follow the checked upstream path.

## Pixtral Decisions

Rust now has an end-to-end `PixtralImageProcessor` for patch-aligned resizing
and tensor-space spatial batch padding. The checked fixture uses
`PixtralImageProcessor()` directly, so it does not download from the Hub. It
records `17x33` and `48x16` RGB inputs, producing resized image sizes
`[[48, 32], [16, 48]]` and exercising right/bottom tensor padding to
`48x48`.

The Rust test compares `pixel_values` shape, `NCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, Transformers `image_sizes`
against Rust `reshaped_input_sizes`, plus `original_sizes`, `image_grid_thw`,
and `image_patch_counts`. Current absolute tolerances are `0.05` for final
tensor values and `0.01` for min/mean/max.

## Idefics3 Decisions

Rust now has an end-to-end `Idefics3ImageProcessor` for longest-edge resize,
vision-encoder multiple alignment, local/global split frames, nested frame
padding, and `pixel_attention_mask`. The checked fixture uses
`Idefics3ImageProcessor` directly with `size={"longest_edge": 10}` and
`max_image_size={"longest_edge": 5}`, so it does not download from the Hub. It
records two one-image samples sized `9x7` and `4x11`, producing split frame
counts `[5, 3]` and exercising padded frame slots in the attention mask.

The Rust test compares `pixel_values` shape, `NPCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, exact logical
`pixel_attention_mask` values, `original_sizes`, `reshaped_input_sizes`,
`image_patch_counts`, and nested `rows`/`cols`. The fixture pins nearest-neighbor
resizing and the Rust parity assertion uses `ResizeParity::PixelExact`, so the
current absolute tolerance is `1e-6` for final tensor values and statistics.

## Mllama Decisions

Rust now has an end-to-end `MllamaImageProcessor` for tiled-canvas resize,
right/bottom raw padding, row-major tile splitting, nested image/tile batch
packing, and Mllama aspect-ratio metadata. The checked fixture uses
`MllamaImageProcessor` directly with `size={"height": 5, "width": 5}` and
`max_image_tiles=4`, so it does not download from the Hub. It records two
one-image samples sized `13x17` and `16x19`, both producing four 5x5 tiles.

The Rust test compares `pixel_values` shape, `NIPCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, `original_sizes`,
`reshaped_input_sizes`, `image_patch_counts`, `canvas_sizes`, nested
`num_tiles`, nested `aspect_ratio_ids`, and nested logical
`aspect_ratio_mask`. The fixture pins nearest-neighbor resizing and the Rust
parity assertion uses `ResizeParity::PixelExact`, so the current absolute
tolerance is `1e-6` for final tensor values and statistics.

## Gemma3 Decisions

Rust now has an end-to-end `Gemma3ImageProcessor` for optional pan-and-scan
crop expansion followed by fixed-size image preprocessing. The checked fixture
uses `Gemma3ImageProcessor` directly with `size={"height": 5, "width": 5}` and
pan-and-scan enabled, so it does not download from the Hub. It records two
images sized `13x17` and `16x19`, producing `num_crops=[2, 0]` and exercising
both crop expansion and the no-crop path in one flat batch.

The Rust test compares `pixel_values` shape, `NCHW` layout, float32 storage,
full tensor values, sampled values, min/mean/max, `original_sizes`,
`reshaped_input_sizes`, and exact `num_crops` metadata. The fixture pins
nearest-neighbor resizing and the Rust parity assertion uses
`ResizeParity::PixelExact`, so the current absolute tolerance is `1e-6` for
final tensor values and statistics.

## Structured Token Decisions

The multimodal aggregate fixture records complete deterministic logits and the
pinned upstream `torch.argmax` token ids/scores for PPOCRV5, PPOCRV6, and
SLANeXt postprocessing. Rust executes the same production recipe descriptors
used by those presets: CTC greedy decoding with blank id `0` for recognition,
and BOS/EOS-aware greedy decoding for table structure. Tests also pin
first-index tie behavior and control-token termination.

The compatible Rust boundary ends at token ids and scores. Existing
processor-specific helpers may combine those ids with a caller-supplied
vocabulary, but recipes do not claim tokenizer, text, HTML, or model-schema
ownership.

## Query-Mask Postprocess Decisions

The separate `image-processors.transformers-query-mask-postprocess.v1`
fixture contains 21 full-payload cases generated at the audited Transformers
commit. It checks
semantic and panoptic outputs for MaskFormer, Mask2Former, OneFormer, and EOMT,
plus instance outputs for MaskFormer, Mask2Former, and EOMT. The scenarios use
five foreground classes plus background and cover low scores, resize,
OneFormer downsampling, Mask2Former's fixed `384x384` intermediate canvas,
EOMT aspect unpadding, stuff-label fusion, overlap filtering, and empty-mask
results.

OneFormer instance restoration is deliberately not approximated: upstream
requires `class_info_file` and thing-class metadata owned by the model or
dataset artifacts, so Rust returns `ExternalMetadataRequired`. The task-facing
API returns indexed maps and segment metadata; it does not advertise the
optional COCO RLE or per-instance binary-map output modes.

## Diffusers Decisions

The checked Diffusers fixture currently covers:

- `VaeImageProcessor` image preprocessing and NumPy-style postprocessing.
- `IPAdapterMaskProcessor.downsample` complete output values.
- `VaeImageProcessorLDM3D` RGB plus 16-bit depth preprocessing and
  postprocessing.
- `BlipImageProcessor` exact resize, OpenAI CLIP normalization, and
  VAE-style postprocessing through complete tensor values.
- `Flux2ImageProcessor` reference-image validation, horizontal concatenation,
  area-limit resizing, and complete VAE-style preprocessing values.
- `VisualClozeProcessor` nested image-grid preprocessing, target masks,
  target-position metadata, and complete upsampling payloads.
- `HunyuanVideo15ImageProcessor` spatial bucket selection and complete decoded
  video preprocessing values.
- `MarigoldImageProcessor` canonical RGB preprocessing, max-edge resize,
  bottom/right replicate padding, depth export, normal visualization, and
  complete uncertainty-visualization values.
- `JoyImageEditImageProcessor` bucket selection plus complete
  resize-center-crop preprocessing values.
- `WanAnimateImageProcessor` target rounding plus complete resize-fill
  preprocessing values.
- `LTX2VideoHDRProcessor` complete default reference-video preprocessing:
  VAE-scale floor resize and normalization, f32 bilinear fit-inside resize,
  bottom/right reflect-or-replicate padding, and LogC3 HDR postprocessing.
- `PixArtImageProcessor` aspect-ratio bin selection, resize/crop geometry, and
  complete tensor resize/crop values.

The fixture intentionally uses deterministic inputs and processor-local
classes, so it does not download models from the Hub. Every array is stored as
lossless base64 little-endian data and compared element-by-element. The
fixture records the exact audited Diffusers commit, and all 12 Diffusers
catalog entries are therefore labeled `FixtureParity`.

All cataloged Transformers families resolve to fixture-backed wrappers or
compact audited presets. `*Pil` aliases resolve to the same Rust family while
selecting their recorded Pillow backend where pixel behavior differs. Every
catalog entry is `FixtureParity`, and the suite-level evidence test prevents
that label from drifting away from its checked fixture cases.

## Performance Notes

The primary local benchmark harness exercises a directory corpus with
CLIP-style resize, center crop, and normalization:

```powershell
cargo bench -p image-processors --bench clip_dataset -- `
  --image-dir path\to\images `
  --batch-size 256 --warmup-batches 2 `
  --resize-profile compatibility --batch-execution auto --reuse-workspace --skip-errors
```

The comparable Transformers/Pillow baseline uses the development script with
its default threaded image loader:

```powershell
.\.venv\Scripts\python.exe scripts\bench\transformers_clip_dataset.py `
  --image-dir path\to\images `
  --batch-size 256 --warmup-batches 2 --skip-errors
```

The following historical full-corpus results processed `32,942` valid images,
skipped `6` corrupt JPEGs, and used `171` batches. They predate the current
exact fixed-point compatibility kernel and should be regenerated before making
current throughput claims:

| Path | Total | ms/image | images/sec |
| --- | ---: | ---: | ---: |
| Rust compatibility profile | `11.7 s` | `0.355919` | `2809.626` |
| Rust fast profile | `7.5 s` | `0.227595` | `4393.763` |
| Transformers/Pillow threaded loader | `203.0 s` | `6.161349` | `162.302` |

In that historical run, the compatibility profile was about `17.31x` the
Python throughput and the fast profile about `27.07x`. The fast profile was
about `1.56x` faster than compatibility.

The staged batch harness remains useful for locating the current hot path:

```powershell
cargo bench -p image-processors --bench clip_batch -- `
  --image path\to\image.jpg --batch-sizes 1,8,32 `
  --repetitions 3 --warmup 1 `
  --resize-profile compatibility --batch-execution auto --reuse-workspace --all-stages
```

Historical mean batch timings on the first `16` valid images from that corpus:

| Compatibility stage | B1 | B8 | B32 |
| --- | ---: | ---: | ---: |
| full | `5.631 ms` | `12.782 ms` | `17.719 ms` |
| `decode_only` | `1.450 ms` | `2.328 ms` | `4.923 ms` |
| `decode_resize` | `10.446 ms` | `11.327 ms` | `18.088 ms` |
| `decode_resize_tensor` | `10.233 ms` | `13.221 ms` | `20.383 ms` |
| `predecoded_batch` | `8.307 ms` | `19.271 ms` | `23.965 ms` |

| Fast stage | B1 | B8 | B32 |
| --- | ---: | ---: | ---: |
| full | `3.067 ms` | `7.463 ms` | `15.191 ms` |
| `decode_only` | `1.446 ms` | `2.517 ms` | `4.433 ms` |
| `decode_resize` | `2.884 ms` | `3.934 ms` | `10.319 ms` |
| `decode_resize_tensor` | `3.142 ms` | `5.581 ms` | `15.574 ms` |
| `predecoded_batch` | `0.882 ms` | `5.215 ms` | `12.051 ms` |

Decode is no longer the dominant cost in this workload. The compatibility
profile's main remaining target is the Pillow-compatible RGB vertical
accumulation plus fused NCHW write path; workspace reuse and automatic batch
scheduling are already part of the measured default path.

TurboJPEG remains optional. On the current Windows setup it was slower than the
default `image` decoder for both the grayscale JPEG and an RGB JPEG copy,
partly because the TurboJPEG path decodes JPEG directly to RGB and can increase
downstream resize work for grayscale inputs.

## Dependency Boundary

Do not add Transformers, PyTorch, Pillow, or NumPy as Rust crate dependencies.
The script is a fixture generator for development machines and CI jobs that
explicitly opt into Python parity checks. Rust resize dependencies should stay
behind `image-resize-kernels` so processor APIs do not leak backend-specific
types.
