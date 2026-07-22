# Rust API usage guide

This guide covers the main ways to use `image-processors` from Rust. For the
full list of supported upstream processor families and their parity evidence,
see [Processor Parity](PARITY.md).

## Choose an API layer

The crate exposes three useful levels of abstraction:

1. **Processor-family wrappers** such as `ClipImageProcessor`,
   `DetrImageProcessor`, and `QwenVlImageProcessor`. Start here when matching a
   known model family.
2. **`ImageProcessor`** for a configurable resize, crop, rescale, normalize,
   pixel-format, and tensor-layout pipeline.
3. **`media`, `tensor`, `transforms`, `recipe`, and `postprocess` modules** for
   callers that need to own individual stages.

Family wrappers are the safest default because they encode the expected
normalization, resize parity, layout, and metadata contract for that family.

## Add the dependency

The workspace requires Rust 1.95 or newer. Until the crates are published,
depend on the Git repository:

```toml
[dependencies]
image-processors = { git = "https://github.com/dinoml/image-processors-rs" }
```

Optional features are additive:

```toml
[dependencies]
image-processors = {
    git = "https://github.com/dinoml/image-processors-rs",
    features = ["url", "video"]
}
```

| Feature | Default | Effect |
| --- | --- | --- |
| `parallel` | yes | Enables Rayon-backed processing for larger batches. |
| `url` | no | Enables blocking HTTPS media loading through `reqwest` and rustls. |
| `video` | no | Enables video decoding through `video-rs`; FFmpeg development libraries are required. |
| `turbojpeg` | no | Enables the optional libjpeg-turbo decoder. |

See [FFmpeg Development Libraries](FFMPEG.md) before enabling `video`.

## Use a model-family processor

The following example loads one image using the CLIP defaults. The returned
tensor owns its values and includes explicit dtype, shape, layout, and leading
axis metadata.

```no_run
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, DType, Layout,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let tensor = processor.open("image.jpg")?;

    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.shape(), [1, 3, 224, 224]);
    Ok(())
}
```

Configuration types are ordinary Rust values. Override only the fields that
must differ from the family defaults:

```rust
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, ImageLayout, ImageSize,
};

fn make_processor() -> Result<ClipImageProcessor, Box<dyn std::error::Error>> {
    Ok(ClipImageProcessor::new(ClipImageProcessorConfig {
        size: ImageSize::new(336, 336)?,
        crop_size: ImageSize::new(336, 336)?,
        output_layout: ImageLayout::ChannelsHeightWidth,
        ..ClipImageProcessorConfig::default()
    })?)
}
```

Invalid dimensions, normalization values, layouts, or incompatible options are
reported when the processor is constructed or called. Library code returns
typed errors and does not require callers to catch panics.

## Supply media

### Filesystem paths

`open` is the shortest path for one file. `open_batch` loads multiple image
paths and stacks them on a batch axis:

```no_run
use image_processors::{ClipImageProcessor, ClipImageProcessorConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let paths = ["first.jpg", "second.jpg"];
    let tensor = processor.open_batch(&paths)?;

    assert_eq!(tensor.batch(), Some(2));
    Ok(())
}
```

### Encoded bytes

Use `ImageProcessor::open_bytes` when encoded image data is already in memory.
Family wrappers expose their underlying generic processor through
`image_processor()`:

```no_run
use image_processors::{ClipImageProcessor, ClipImageProcessorConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoded = std::fs::read("image.png")?;
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let tensor = processor.image_processor().open_bytes(encoded)?;

    assert_eq!(tensor.batch(), Some(1));
    Ok(())
}
```

### Decoded frames

Use `ImageFrame` when decoding happens elsewhere or when pixels are generated
in memory. The constructor validates dimensions, pixel format, and buffer
length.

```rust
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, ImageFrame, PixelFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pixels = vec![255, 0, 0, 0, 255, 0];
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, pixels)?;
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let tensor = processor.preprocess_image(&frame)?;

    assert_eq!(tensor.shape(), [1, 3, 224, 224]);
    Ok(())
}
```

`ImageSequence` and `VideoClip` provide the equivalent owned containers for
animated images and decoded video. Sequence and video APIs preserve whether
the leading axis represents a batch or temporal frames.

### URLs

With the `url` feature enabled, construct a typed `MediaSource` and pass it to
`open_source`:

```no_run
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, MediaSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let source = MediaSource::image_url("https://example.com/image.jpg");
    let tensor = processor.open_source(source)?;

    assert_eq!(tensor.batch(), Some(1));
    Ok(())
}
```

Image URLs use buffered loading. Streaming is intended for video sources and
requires both `url` and `video`.

## Configure the generic processor

Use `ImageProcessor` when no predefined family owns the desired pipeline. Its
configuration makes pixel conversion, spatial transforms, numeric transforms,
and output layout explicit.

```rust
use image_processors::{
    ImageFrame, ImageProcessor, ImageProcessorConfig, Layout, PixelFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ImageProcessor::new(ImageProcessorConfig {
        pixel_format: Some(PixelFormat::Rgb8),
        output_layout: Layout::NHWC,
        do_rescale: true,
        rescale_factor: 1.0 / 255.0,
        ..ImageProcessorConfig::default()
    })?;

    let frame = ImageFrame::new(1, 1, PixelFormat::Rgb8, vec![255, 128, 0])?;
    let tensor = processor.preprocess_image(&frame)?;

    assert_eq!(tensor.layout(), Layout::NHWC);
    assert_eq!(tensor.shape(), [1, 1, 1, 3]);
    Ok(())
}
```

Prefer a family wrapper when matching Transformers or Diffusers behavior. The
generic configuration defaults to the fast resize profile, while family
wrappers generally select compatibility behavior backed by parity fixtures.

## Reuse batch allocations

For repeated batches, `ImageProcessorWorkspace` retains output buffers, resize
scratch space, and coefficient tables. The returned tensor owns the current
output buffer; recycle it only after every consumer has finished reading it.

```no_run
use image_processors::{
    ClipImageProcessor, ClipImageProcessorConfig, ImageProcessorWorkspace,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ClipImageProcessor::new(ClipImageProcessorConfig::default())?;
    let paths = ["first.jpg", "second.jpg"];
    let mut workspace = ImageProcessorWorkspace::new();

    let tensor = processor.open_batch_into(&paths, &mut workspace)?;
    consume(tensor.shape());
    workspace.recycle_tensor(tensor)?;

    Ok(())
}

fn consume(shape: &[usize]) {
    assert_eq!(shape[0], 2);
}
```

The default automatic scheduler keeps small batches serial and uses parallel
execution for larger batches when `parallel` is enabled. Set the family or
generic configuration's `batch_execution` field when deterministic scheduling
is more important than automatic selection.

## Read tensors

`Tensor` does not hide its storage type. Inspect `dtype`, `shape`, and `layout`,
then match the `TensorData` variant required by the model runtime:

```rust
use image_processors::{Layout, Tensor, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = Tensor::new(
        TensorData::F32(vec![0.0, 0.5, 1.0]),
        [1, 3, 1, 1],
        Layout::NCHW,
    )?;

    let TensorData::F32(values) = tensor.data() else {
        return Err("expected f32 tensor".into());
    };
    assert_eq!(values, &[0.0, 0.5, 1.0]);
    Ok(())
}
```

`TensorView` and `TensorDataView` borrow existing storage. Byte-preserving
processor configurations can expose zero-copy views over decoded frames;
value-changing resize or normalization operations require owned output.

## Read typed processor outputs

Some processors return more than `pixel_values`. For example, DETR output also
contains a validity mask and original/reshaped image sizes. Use the typed name
enums instead of string indexing:

```no_run
use image_processors::{
    DetrImageProcessor, DetrImageProcessorConfig, ProcessorMetadataName,
    ProcessorTensorName,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = DetrImageProcessor::new(DetrImageProcessorConfig::default())?;
    let output = processor.open_output("image.jpg")?;

    let pixels = output
        .tensor(&ProcessorTensorName::PixelValues)
        .ok_or("missing pixel_values")?;
    let sizes = output
        .metadata_value(&ProcessorMetadataName::OriginalSizes)
        .ok_or("missing original_sizes")?;

    assert_eq!(pixels.batch(), Some(1));
    let _ = sizes;
    Ok(())
}
```

`ProcessorTensorName::other` and `ProcessorMetadataName::other` preserve typed
access while allowing processor-specific names.

## Parse Hugging Face configuration

`ProcessorFamilyConfig::from_hf_preprocessor_json` parses supported
`preprocessor_config.json` documents into family-specific Rust configuration.
It does not download from the Hub.

```rust
use image_processors::{
    ClipImageProcessor, ProcessorFamilyConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{
        "image_processor_type": "CLIPImageProcessor",
        "size": { "height": 224, "width": 224 },
        "crop_size": { "height": 224, "width": 224 },
        "do_center_crop": true,
        "resample": 3
    }"#;

    let family = ProcessorFamilyConfig::from_hf_preprocessor_json(json)?;
    let ProcessorFamilyConfig::Clip(config) = family else {
        return Err("expected CLIP configuration".into());
    };
    let processor = ClipImageProcessor::new(config)?;

    assert_eq!(processor.config().size.width, 224);
    Ok(())
}
```

Unsupported processor identities and malformed or conflicting fields return
`ProcessorConfigError` variants.

## Query the compatibility catalog

`ProcessorCatalog` is immutable, source-controlled, and network-free. Use it to
resolve an upstream class, model type, or known processor identifier before
selecting a Rust wrapper or preset.

```rust
use image_processors::{
    CompatibilityStatus, ProcessorCatalog, ProcessorFamilyKind, UpstreamLibrary,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = ProcessorCatalog::new();
    let entry = catalog
        .find_by_class_name(UpstreamLibrary::Transformers, "CLIPImageProcessor")
        .ok_or("CLIP is not cataloged")?;

    assert_eq!(entry.family(), ProcessorFamilyKind::Clip);
    assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
    Ok(())
}
```

A catalog match describes compatibility and the relevant Rust family. It does
not construct processors, fetch configuration, or download model artifacts.

## Process video

Enable `video` and install FFmpeg development libraries first. Video-family
wrappers accept video files, `MediaSource::video_path`, decoded `VideoClip`
values, or frame collections, depending on the family.

```no_run
use image_processors::{
    MediaSource, VideoMaeImageProcessor, VideoMaeImageProcessorConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = VideoMaeImageProcessor::new(VideoMaeImageProcessorConfig::default())?;
    let source = MediaSource::video_path("clip.mp4");
    let tensor = processor.open_source(source)?;

    assert!(tensor.frames().is_some());
    Ok(())
}
```

Attach a `FrameSampling` policy to a media source to select frames while
loading:

```rust
use image_processors::{FrameSampling, MediaSource};

let source = MediaSource::video_path("clip.mp4")
    .with_frame_sampling(FrameSampling::every(2).with_max_frames(16));

assert_eq!(source.hint.frame_sampling.unwrap().stride, 2);
```

## Postprocess model outputs

Preprocessing produces model inputs. The `postprocess` module performs the
inverse geometry and task-specific restoration needed after inference,
including:

- DETR-style object detection;
- semantic, instance, and panoptic segmentation;
- binary masks and uncompressed RLE;
- depth and dense-map resize-back;
- image and video tensor restoration;
- normalized coordinate restoration;
- structured token-id decoding and output hooks.

Family wrappers expose task-specific helpers when their output contract is
known. The lower-level `post_process_recipe_outputs` function executes the
postprocessing descriptors attached to a `ProcessorRecipe`. Model inference,
tokenizer vocabularies, and model-specific token-to-text conversion remain
outside this crate.

## Error handling

The main error boundaries are:

- `MediaError` for source loading and decoding;
- `ImageProcessorError` for configuration and preprocessing;
- `TensorError` for invalid storage, shape, dtype, or layout;
- `TransformError` for resize, crop, padding, mask, and geometry operations;
- `ProcessorConfigError` for Hugging Face configuration ingestion;
- task-specific postprocessing errors in the `postprocess` module.

Applications can use `?` directly because these errors implement
`std::error::Error`. Libraries should preserve the typed variant or wrap it in
a domain error with `#[source]` rather than discard its context.

## Where to go next

- [README](../README.md): project overview, architecture, and development.
- [Processor Parity](PARITY.md): supported upstream identities, fixtures, and
  numeric comparison policy.
- [FFmpeg Development Libraries](FFMPEG.md): native setup for `video`.
- [Roadmap](ROADMAP.md): implemented scope and future work.
- Generated crate docs: item-level methods, errors, and layout contracts.
