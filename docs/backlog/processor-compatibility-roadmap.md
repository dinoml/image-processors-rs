# Processor Compatibility Backlog

**Status:** Complete for the pinned audit baseline  
**Completed:** 2026-07-10

This document records the completed compatibility boundary. It is no longer a
list of unfinished work. Ongoing product work belongs in
[`../ROADMAP.md`](../ROADMAP.md), while fixture generation and acceptance
details live in [`../PARITY.md`](../PARITY.md).

## Audited Baseline

- Transformers: `6d960ca0a0eba0d2aebc920d8080a9353da468d3`.
- Diffusers: `208704a27a6f362b67cd1a04fa1db0b98036d26f`.
- Catalog: 119 Transformers entries and 12 Diffusers entries.
- Transformers identity surface: 209 canonical and `*Pil` class names.
- Compatibility status: every catalog entry is backed by full-payload fixture
  parity at its recorded audit commit.

The baseline covers the upstream image-processor identities present in the
2026-07-09 audit. It does not make a forward-compatibility claim for classes or
behavior introduced by later upstream revisions.

## Completion Boundary

The completed scope is Rust-owned media loading, image/video preprocessing,
tensor and metadata production, reusable processor recipes, and generic typed
postprocessing of model outputs. Compatibility means that the Rust result is
checked against deterministic, source-controlled upstream fixtures with an
explicit comparison policy.

The scope deliberately excludes:

- model inference and model weights,
- tokenizer vocabularies and model-specific token-to-text or token-to-schema
  conversion,
- training-time annotation encoders,
- hidden network or Hub access,
- Python, Pillow, NumPy, torch, Transformers, or Diffusers runtime dependencies
  in the Rust crates.

Those exclusions are API boundaries, not unfinished compatibility items.

## Completed Architecture

The implementation keeps upstream compatibility broad without copying the
Python class hierarchy:

1. `media` owns source loading, decoded frame ownership, image sequences,
   optional URL loading, optional FFmpeg decoding, and frame sampling.
2. `transforms` owns resize, crop, pad, fill, masks, geometry, and shared
   postprocess primitives.
3. `tensor` owns storage, dtype, shape, layout, views, and layout conversion.
4. `output` owns named tensors and typed metadata.
5. `recipe` owns serializable, validated preprocessing and postprocessing
   descriptors.
6. `postprocess` executes reusable detection, segmentation, mask, depth,
   dense-map, coordinate, video-transfer, token-sequence, and output-hook
   descriptors.
7. `processors` provides concrete family wrappers and compact audited preset
   collections where behavior needs a stronger typed front door.
8. `catalog` maps upstream identities to Rust recipes, wrappers, audit commits,
   and dedicated fixture evidence without downloads or mutable global state;
   suite-level tests cover aggregate fixture identities.

Public APIs remain Rust-shaped: explicit ownership, validated configs, typed
`Result` errors, additive feature gates, and no untyped compatibility `kwargs`.

## Completed Coverage

| Evidence group | Canonical entries | Aliases | What is checked |
| --- | ---: | ---: | --- |
| Dedicated Transformers families | 13 | 12 | CLIP, ViT, VideoMAE, ViViT, Qwen2-VL, LLaVA-NeXT, Pixtral, Idefics3, Gemma3, Mllama, DETR, SAM, and Donut |
| Encoder and restoration presets | 20 | 17 | Encoder/classifier geometry, Pillow and Torchvision backend distinctions, and Swin2SR restoration |
| Task-vision presets | 34 | 23 | Detection, grounding, segmentation, depth, keypoint, matching, pose, and task metadata |
| Multimodal presets | 52 | 38 | VLM, document/OCR/table, tiled/grid/patch inputs, and video-facing processors |
| Diffusers processors | 12 | 0 | Shared and reviewed pipeline-local image/video processors |

Together these groups cover all 131 catalog entries and all 209 audited
Transformers class names. The catalog-evidence integration test checks that
every `FixtureParity` identity appears in a source-controlled fixture at the
catalog audit commit and every catalog-bound fixture identity resolves back to
the catalog. The additional `Qwen2VLVideoProcessor` fixture is declared as
auxiliary evidence for the cataloged Qwen2-VL family and is checked at the same
commit rather than advertised as a separate catalog identity.

### Recipe and execution coverage

The reusable recipe layer covers:

- exact, shortest-edge, longest-edge, fit-within, aspect-ratio bucket, and
  smart-resize geometry,
- center crop, longest-axis alignment, explicit and divisor padding, square
  padding, thumbnailing, tiling, splitting, overlays, and patch grids,
- rescale, normalize, channel conversion, layout conversion, and typed output
  naming,
- image, image-batch, frame-sequence, and batched-video contracts,
- object detection, semantic/instance/panoptic segmentation, binary masks,
  depth, dense maps, coordinates, video transfer functions, structured token
  ids, and task-specific output hooks.

Temporal patchification uses the shared `RecipePatchStage` executor for Qwen
and non-Qwen patch-grid profiles, including a fixture-backed VideoLlama3 path.
LTX2 LogC3 restoration runs through the generic video descriptor for `BFCHW`
and `BFHWC`, including multi-batch and multi-spatial-element HDR payloads.

### Structured outputs

Query-mask processors select their output interpretation at postprocess time.
EOMT, MaskFormer, and Mask2Former have fixture-backed semantic, instance, and
panoptic restoration; OneFormer has semantic and panoptic restoration.
OneFormer instance output returns a typed external-metadata error because its
`class_info_file` and thing-class metadata belong to the model artifacts.
Processors such as SAM2 that use a different output contract are rejected
explicitly.

OCR, table, and document sequence logits can use recipe-driven CTC greedy or
BOS/EOS-aware greedy decoding. The result intentionally stops at token ids and
scores. Converting those ids into text, HTML, or another model schema requires
the caller's external tokenizer or vocabulary.

## Fixture Acceptance Contract

Full fixture parity requires complete output payload comparisons, not status
labels, shapes, samples, hashes, or summary statistics alone. Each parity
fixture records or inherits an explicit contract containing:

- schema version and generator,
- upstream library, version, source, and exact commit,
- canonical class and backend-specific alias where applicable,
- model id or deterministic local configuration source,
- deterministic input media description,
- effective processor configuration,
- output tensor and metadata names,
- exact shape, layout, dtype, integer, byte, boolean, and metadata rules,
- absolute and relative tolerances for full floating-point values and recorded
  statistics.

The checked contract caps absolute tolerance at `0.06` and requires zero
relative tolerance. A fixture-only edit cannot silently relax those bounds.

Dedicated Transformers fixtures use
`image-processors.transformers-parity.v1`. Their roots record the generator and
deterministic input contract, cases record effective config and comparison
policy, and catalog evidence names the fixture files and exact numeric source
revision. Aggregate Transformers and Diffusers fixtures carry equivalent
suite-level provenance and comparison policies. Tests deserialize and validate
these policies rather than relying on unexplained test-only constants.

Python generators are development tools only. Rust builds and tests consume
checked-in JSON and do not require a Python environment.

## Feature and CI Contract

Ordinary CI verifies formatting, Clippy with warnings denied, the default
workspace, the `url` feature, and `image-processors` with default features
disabled. The no-default build includes an observable assertion that video
sources return `VideoDecodingUnavailable`.

The `video` feature is checked on Windows, Linux, and macOS with FFmpeg
development libraries installed through vcpkg or the platform package manager.
This keeps optional native dependencies out of the default and URL-only builds
while still testing the all-feature public surface on all three CI platforms.

The local verification sequence is:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo clippy -p image-processors --all-targets --no-default-features -- -D warnings
cargo test -p image-processors --no-default-features
cargo clippy --workspace --all-targets --features url -- -D warnings
cargo test --workspace --features url
```

With FFmpeg development libraries available:

```powershell
.\scripts\setup-vcpkg-ffmpeg.ps1 -CheckCargo
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## Maintenance Triggers

The completed baseline must be re-audited when any of these change:

- the pinned Transformers or Diffusers source revision,
- the upstream image-processor class inventory or alias set,
- processor defaults, backend selection, resize semantics, output names, or
  metadata,
- a Rust recipe, transform kernel, family config, or postprocess contract,
- optional-feature behavior.

When a trigger occurs:

1. regenerate fixtures from an exact clean source checkout,
2. update catalog audit metadata and bidirectional evidence links,
3. run the full payload and feature matrix,
4. keep `FixtureParity` only when all acceptance requirements pass,
5. otherwise use `SummaryFixture`, `Implemented`, or `Planned` until evidence
   catches up.

Larger diagnostic inputs, tiled restoration APIs, training label encoders,
pose skeleton assembly, or new model-specific output schemas may be useful
future enhancements. They become compatibility backlog only when explicitly
brought into scope or required by a newly audited upstream contract.
