use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use image_processors::{
    convert_frame_pixel_format, load_image_from_path_with_backend, resize_frame_with_decision,
    BatchExecution, ClipImageProcessor, ClipImageProcessorConfig, ImageDecodeBackend, ImageFrame,
    ImageProcessor, ImageProcessorConfig, ImageProcessorWorkspace, ImageSize, Layout, PixelFormat,
    ResizeDecision, ResizeFilter, ResizeMode, ResizeParity, Tensor,
    DEFAULT_PARALLEL_BATCH_THRESHOLD,
};

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32];
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_WARMUP: usize = 1;
const CLIP_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> BenchResult<()> {
    let args = BenchArgs::parse(env::args().skip(1))?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let clip_config = ClipImageProcessorConfig {
        resize_parity: args.resize_parity,
        decode_backend: args.decode_backend,
        batch_execution: args.batch_execution,
        ..Default::default()
    };
    let processor = ClipImageProcessor::new(clip_config.clone())?;
    let tensor_processor = clip_tensor_processor(args.decode_backend, args.batch_execution)?;

    println!("processor,clip");
    println!("images,{}", args.images.len());
    println!(
        "resize_profile,{}",
        format_resize_parity(processor.config().resize_parity)
    );
    println!(
        "decode_backend,{}",
        format_decode_backend(args.decode_backend)
    );
    println!(
        "batch_execution,{}",
        format_batch_execution(args.batch_execution)
    );
    println!(
        "auto_parallel_threshold,{}",
        DEFAULT_PARALLEL_BATCH_THRESHOLD
    );
    println!("reuse_workspace,{}", args.reuse_workspace);
    println!("warmup,{}", args.warmup);
    println!("repetitions,{}", args.repetitions);
    println!("stages,{}", join_stages(&args.stages));
    println!("stage,batch_size,repetitions,min_ms,mean_ms,max_ms,last_shape,last_layout");

    for stage in &args.stages {
        for batch_size in &args.batch_sizes {
            let batch_paths = batch_paths(&args.images, *batch_size);
            let predecoded = if *stage == BenchStage::PredecodedBatch {
                Some(load_batch(
                    &batch_paths,
                    args.decode_backend,
                    args.batch_execution,
                )?)
            } else {
                None
            };
            let context = BenchContext {
                processor: &processor,
                tensor_processor: &tensor_processor,
                resize_parity: args.resize_parity,
                decode_backend: args.decode_backend,
                batch_execution: args.batch_execution,
            };
            let mut workspace = args.reuse_workspace.then(ImageProcessorWorkspace::new);

            for _ in 0..args.warmup {
                let result = run_stage(
                    *stage,
                    context,
                    &batch_paths,
                    predecoded.as_deref(),
                    workspace.as_mut(),
                )?;
                std::hint::black_box(result.black_box_len);
            }

            let mut timings = Vec::with_capacity(args.repetitions);
            let mut last_shape = StageShape::NotTensor;
            let mut last_layout = StageLayout::NotTensor;
            for _ in 0..args.repetitions {
                let started = Instant::now();
                let result = run_stage(
                    *stage,
                    context,
                    &batch_paths,
                    predecoded.as_deref(),
                    workspace.as_mut(),
                )?;
                let elapsed = started.elapsed();
                std::hint::black_box(result.black_box_len);
                last_shape = result.shape;
                last_layout = result.layout;
                timings.push(elapsed);
            }

            let summary = TimingSummary::new(&timings)?;
            println!(
                "{},{},{},{:.3},{:.3},{:.3},{},{}",
                stage,
                batch_size,
                args.repetitions,
                millis(summary.min),
                millis(summary.mean),
                millis(summary.max),
                last_shape,
                last_layout
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchStage {
    Full,
    DecodeOnly,
    DecodeResize,
    DecodeResizeTensor,
    PredecodedBatch,
}

impl BenchStage {
    fn parse(value: &str) -> Result<Self, BenchError> {
        match value {
            "full" => Ok(Self::Full),
            "decode_only" => Ok(Self::DecodeOnly),
            "decode_resize" => Ok(Self::DecodeResize),
            "decode_resize_tensor" => Ok(Self::DecodeResizeTensor),
            "predecoded_batch" => Ok(Self::PredecodedBatch),
            _ => Err(BenchError::UnknownStage(value.to_owned())),
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::Full,
            Self::DecodeOnly,
            Self::DecodeResize,
            Self::DecodeResizeTensor,
            Self::PredecodedBatch,
        ]
    }
}

impl Display for BenchStage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(formatter, "full"),
            Self::DecodeOnly => write!(formatter, "decode_only"),
            Self::DecodeResize => write!(formatter, "decode_resize"),
            Self::DecodeResizeTensor => write!(formatter, "decode_resize_tensor"),
            Self::PredecodedBatch => write!(formatter, "predecoded_batch"),
        }
    }
}

#[derive(Debug)]
struct BenchArgs {
    images: Vec<PathBuf>,
    batch_sizes: Vec<usize>,
    repetitions: usize,
    warmup: usize,
    stages: Vec<BenchStage>,
    resize_parity: ResizeParity,
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
    reuse_workspace: bool,
    help: bool,
}

impl BenchArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, BenchError> {
        let mut images = Vec::new();
        let mut batch_sizes = DEFAULT_BATCH_SIZES.to_vec();
        let mut repetitions = DEFAULT_REPETITIONS;
        let mut warmup = DEFAULT_WARMUP;
        let mut stages = vec![BenchStage::Full];
        let mut resize_parity = ResizeParity::Compatibility;
        let mut decode_backend = ImageDecodeBackend::ImageCrate;
        let mut batch_execution = BatchExecution::default();
        let mut reuse_workspace = false;
        let mut help = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "-h" | "--help" => help = true,
                "--all-stages" => stages = BenchStage::all(),
                "--image" => images.push(next_path(&mut args, "--image")?),
                "--images" => {
                    let value = next_value(&mut args, "--images")?;
                    images.extend(parse_paths(&value)?);
                }
                "--batch-sizes" => {
                    batch_sizes = parse_usize_list(&next_value(&mut args, "--batch-sizes")?)?;
                }
                "--repetitions" => {
                    repetitions =
                        parse_usize("--repetitions", &next_value(&mut args, "--repetitions")?)?;
                }
                "--decode-backend" => {
                    decode_backend =
                        parse_decode_backend(&next_value(&mut args, "--decode-backend")?)?;
                }
                "--resize-profile" | "--resize-parity" => {
                    resize_parity =
                        parse_resize_parity(&next_value(&mut args, "--resize-profile")?)?;
                }
                "--batch-execution" => {
                    batch_execution =
                        parse_batch_execution(&next_value(&mut args, "--batch-execution")?)?;
                }
                "--parallel" => {
                    batch_execution = BatchExecution::Parallel;
                }
                "--reuse-workspace" => {
                    reuse_workspace = true;
                }
                "--stage" => {
                    stages = vec![BenchStage::parse(&next_value(&mut args, "--stage")?)?];
                }
                "--stages" => {
                    stages = parse_stages(&next_value(&mut args, "--stages")?)?;
                }
                "--warmup" => {
                    warmup = parse_usize("--warmup", &next_value(&mut args, "--warmup")?)?;
                }
                value if value.starts_with('-') => {
                    return Err(BenchError::UnknownArgument(value.to_owned()));
                }
                value => images.push(PathBuf::from(value)),
            }
        }

        if !help && images.is_empty() {
            return Err(BenchError::NoImages);
        }
        if !help && batch_sizes.is_empty() {
            return Err(BenchError::EmptyBatchSizes);
        }
        if !help && stages.is_empty() {
            return Err(BenchError::EmptyStages);
        }
        if !help && repetitions == 0 {
            return Err(BenchError::ZeroRepetitions);
        }
        if !help {
            for batch_size in &batch_sizes {
                if *batch_size == 0 {
                    return Err(BenchError::ZeroBatchSize);
                }
            }
        }

        Ok(Self {
            images,
            batch_sizes,
            repetitions,
            warmup,
            stages,
            resize_parity,
            decode_backend,
            batch_execution,
            reuse_workspace,
            help,
        })
    }
}

struct StageResult {
    black_box_len: usize,
    shape: StageShape,
    layout: StageLayout,
}

enum StageShape {
    Tensor(Vec<usize>),
    NotTensor,
}

impl Display for StageShape {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(shape) => write!(formatter, "{shape:?}"),
            Self::NotTensor => write!(formatter, "n/a"),
        }
    }
}

enum StageLayout {
    Tensor(String),
    NotTensor,
}

impl Display for StageLayout {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(layout) => write!(formatter, "{layout}"),
            Self::NotTensor => write!(formatter, "n/a"),
        }
    }
}

#[derive(Debug)]
struct TimingSummary {
    min: Duration,
    mean: Duration,
    max: Duration,
}

impl TimingSummary {
    fn new(values: &[Duration]) -> Result<Self, BenchError> {
        if values.is_empty() {
            return Err(BenchError::ZeroRepetitions);
        }
        let min = values
            .iter()
            .copied()
            .min()
            .ok_or(BenchError::ZeroRepetitions)?;
        let max = values
            .iter()
            .copied()
            .max()
            .ok_or(BenchError::ZeroRepetitions)?;
        let total = values.iter().copied().sum::<Duration>();
        Ok(Self {
            min,
            mean: total / values.len() as u32,
            max,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum BenchError {
    EmptyBatchSizes,
    EmptyPath,
    EmptyStages,
    MissingPredecoded,
    MissingValue(&'static str),
    NoImages,
    #[cfg(not(feature = "parallel"))]
    ParallelBatchUnavailable,
    ParseInt {
        flag: &'static str,
        value: String,
    },
    UnknownArgument(String),
    UnknownBatchExecution(String),
    UnknownDecodeBackend(String),
    UnknownResizeProfile(String),
    UnknownStage(String),
    ZeroBatchSize,
    ZeroRepetitions,
}

impl Display for BenchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBatchSizes => write!(formatter, "at least one batch size is required"),
            Self::EmptyPath => write!(formatter, "image path entries cannot be empty"),
            Self::EmptyStages => write!(formatter, "at least one benchmark stage is required"),
            Self::MissingPredecoded => write!(formatter, "predecoded stage data was not prepared"),
            Self::MissingValue(flag) => write!(formatter, "missing value for {flag}"),
            Self::NoImages => write!(formatter, "provide at least one image path"),
            #[cfg(not(feature = "parallel"))]
            Self::ParallelBatchUnavailable => {
                write!(
                    formatter,
                    "parallel batch execution requires the parallel feature"
                )
            }
            Self::ParseInt { flag, value } => {
                write!(formatter, "invalid integer for {flag}: {value}")
            }
            Self::UnknownArgument(value) => write!(formatter, "unknown argument {value}"),
            Self::UnknownBatchExecution(value) => {
                write!(formatter, "unknown batch execution mode {value}")
            }
            Self::UnknownDecodeBackend(value) => {
                write!(formatter, "unknown decode backend {value}")
            }
            Self::UnknownResizeProfile(value) => {
                write!(formatter, "unknown resize profile {value}")
            }
            Self::UnknownStage(value) => write!(formatter, "unknown benchmark stage {value}"),
            Self::ZeroBatchSize => write!(formatter, "batch sizes must be positive"),
            Self::ZeroRepetitions => write!(formatter, "repetitions must be positive"),
        }
    }
}

impl Error for BenchError {}

#[derive(Clone, Copy)]
struct BenchContext<'a> {
    processor: &'a ClipImageProcessor,
    tensor_processor: &'a ImageProcessor,
    resize_parity: ResizeParity,
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
}

fn run_stage(
    stage: BenchStage,
    context: BenchContext<'_>,
    batch_paths: &[PathBuf],
    predecoded: Option<&[ImageFrame]>,
    workspace: Option<&mut ImageProcessorWorkspace>,
) -> BenchResult<StageResult> {
    match stage {
        BenchStage::Full => match workspace {
            Some(workspace) => {
                let tensor = context.processor.open_batch_into(batch_paths, workspace)?;
                tensor_result_recycle(tensor, workspace)
            }
            None => Ok(tensor_result(context.processor.open_batch(batch_paths)?)),
        },
        BenchStage::DecodeOnly => {
            let images = load_batch(batch_paths, context.decode_backend, context.batch_execution)?;
            Ok(non_tensor_result(
                images.iter().map(|image| image.data().len()).sum(),
            ))
        }
        BenchStage::DecodeResize => {
            let images = load_and_prepare_batch(
                batch_paths,
                context.decode_backend,
                context.batch_execution,
                context.resize_parity,
            )?;
            Ok(non_tensor_result(
                images.iter().map(|image| image.data().len()).sum(),
            ))
        }
        BenchStage::DecodeResizeTensor => {
            let images = load_and_prepare_batch(
                batch_paths,
                context.decode_backend,
                context.batch_execution,
                context.resize_parity,
            )?;
            Ok(tensor_result(
                context.tensor_processor.preprocess_images(&images)?,
            ))
        }
        BenchStage::PredecodedBatch => {
            let images = predecoded.ok_or(BenchError::MissingPredecoded)?;
            Ok(tensor_result(context.processor.preprocess_images(images)?))
        }
    }
}

fn tensor_result(tensor: Tensor) -> StageResult {
    tensor_result_ref(&tensor)
}

fn tensor_result_recycle(
    tensor: Tensor,
    workspace: &mut ImageProcessorWorkspace,
) -> BenchResult<StageResult> {
    let result = tensor_result_ref(&tensor);
    workspace.recycle_tensor(tensor)?;
    Ok(result)
}

fn tensor_result_ref(tensor: &Tensor) -> StageResult {
    let black_box_len = tensor.data().len();
    let shape = StageShape::Tensor(tensor.shape().to_vec());
    let layout = StageLayout::Tensor(format!("{:?}", tensor.layout()));
    StageResult {
        black_box_len,
        shape,
        layout,
    }
}

fn non_tensor_result(black_box_len: usize) -> StageResult {
    StageResult {
        black_box_len,
        shape: StageShape::NotTensor,
        layout: StageLayout::NotTensor,
    }
}

fn clip_tensor_processor(
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
) -> BenchResult<ImageProcessor> {
    Ok(ImageProcessor::new(ImageProcessorConfig {
        do_resize: false,
        height: None,
        width: None,
        resize_mode: ResizeMode::Default,
        resample: ResizeFilter::Bicubic,
        resize_parity: ResizeParity::Resampling,
        decode_backend,
        batch_execution,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: true,
        rescale_factor: 1.0 / 255.0,
        do_normalize: true,
        image_mean: CLIP_IMAGE_MEAN.to_vec(),
        image_std: CLIP_IMAGE_STD.to_vec(),
        do_binarize: false,
        output_layout: Layout::NCHW,
    })?)
}

fn format_resize_parity(parity: ResizeParity) -> &'static str {
    match parity {
        ResizeParity::Resampling => "fast",
        ResizeParity::Compatibility => "pillow_compatibility",
        ResizeParity::Torchvision => "torchvision_compatibility",
        ResizeParity::PixelExact => "pixel_exact",
        _ => "unknown",
    }
}

fn load_batch(
    paths: &[PathBuf],
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
) -> BenchResult<Vec<ImageFrame>> {
    if should_run_parallel(batch_execution, paths.len())? {
        return load_batch_parallel(paths, decode_backend);
    }

    match batch_execution {
        BatchExecution::Auto | BatchExecution::Serial => paths
            .iter()
            .map(|path| load_image_from_path_with_backend(path, decode_backend))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        BatchExecution::Parallel => unreachable!("parallel execution returned early"),
    }
}

fn load_and_prepare_batch(
    paths: &[PathBuf],
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
    resize_parity: ResizeParity,
) -> BenchResult<Vec<ImageFrame>> {
    if should_run_parallel(batch_execution, paths.len())? {
        return load_and_prepare_batch_parallel(paths, decode_backend, resize_parity);
    }

    match batch_execution {
        BatchExecution::Auto | BatchExecution::Serial => paths
            .iter()
            .map(|path| {
                let image = load_image_from_path_with_backend(path, decode_backend)?;
                prepare_clip_frame(&image, resize_parity)
            })
            .collect::<BenchResult<Vec<_>>>(),
        BatchExecution::Parallel => unreachable!("parallel execution returned early"),
    }
}

fn should_run_parallel(batch_execution: BatchExecution, batch: usize) -> Result<bool, BenchError> {
    match batch_execution {
        BatchExecution::Auto => {
            #[cfg(feature = "parallel")]
            {
                Ok(batch >= DEFAULT_PARALLEL_BATCH_THRESHOLD)
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = batch;
                Ok(false)
            }
        }
        BatchExecution::Serial => Ok(false),
        BatchExecution::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let _ = batch;
                Ok(true)
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = batch;
                Err(BenchError::ParallelBatchUnavailable)
            }
        }
    }
}

fn prepare_clip_frame(image: &ImageFrame, resize_parity: ResizeParity) -> BenchResult<ImageFrame> {
    let resized = resize_frame_with_decision(
        image,
        ImageSize {
            height: 224,
            width: 224,
        },
        ResizeDecision::new(ResizeFilter::Bicubic, resize_parity)?,
        ResizeMode::Crop,
    )?;

    if resized.pixel_format() == PixelFormat::Rgb8 {
        Ok(resized)
    } else {
        Ok(convert_frame_pixel_format(&resized, PixelFormat::Rgb8)?)
    }
}

#[cfg(feature = "parallel")]
fn load_batch_parallel(
    paths: &[PathBuf],
    decode_backend: ImageDecodeBackend,
) -> BenchResult<Vec<ImageFrame>> {
    paths
        .par_iter()
        .map(|path| load_image_from_path_with_backend(path, decode_backend))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

#[cfg(not(feature = "parallel"))]
fn load_batch_parallel(
    _paths: &[PathBuf],
    _decode_backend: ImageDecodeBackend,
) -> BenchResult<Vec<ImageFrame>> {
    Err(Box::new(BenchError::ParallelBatchUnavailable))
}

#[cfg(feature = "parallel")]
fn load_and_prepare_batch_parallel(
    paths: &[PathBuf],
    decode_backend: ImageDecodeBackend,
    resize_parity: ResizeParity,
) -> BenchResult<Vec<ImageFrame>> {
    paths
        .par_iter()
        .map(|path| {
            let image = load_image_from_path_with_backend(path, decode_backend)?;
            prepare_clip_frame(&image, resize_parity)
        })
        .collect::<BenchResult<Vec<_>>>()
}

#[cfg(not(feature = "parallel"))]
fn load_and_prepare_batch_parallel(
    _paths: &[PathBuf],
    _decode_backend: ImageDecodeBackend,
    _resize_parity: ResizeParity,
) -> BenchResult<Vec<ImageFrame>> {
    Err(Box::new(BenchError::ParallelBatchUnavailable))
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<PathBuf, BenchError> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, BenchError> {
    args.next().ok_or(BenchError::MissingValue(flag))
}

fn parse_paths(value: &str) -> Result<Vec<PathBuf>, BenchError> {
    value
        .split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                Err(BenchError::EmptyPath)
            } else {
                Ok(PathBuf::from(trimmed))
            }
        })
        .collect()
}

fn parse_stages(value: &str) -> Result<Vec<BenchStage>, BenchError> {
    value
        .split(',')
        .map(|part| BenchStage::parse(part.trim()))
        .collect()
}

fn parse_decode_backend(value: &str) -> Result<ImageDecodeBackend, BenchError> {
    match value {
        "image" | "image_crate" | "rust_image" => Ok(ImageDecodeBackend::ImageCrate),
        "turbojpeg" | "turbo-jpeg" => Ok(ImageDecodeBackend::TurboJpeg),
        _ => Err(BenchError::UnknownDecodeBackend(value.to_owned())),
    }
}

fn parse_batch_execution(value: &str) -> Result<BatchExecution, BenchError> {
    match value {
        "auto" => Ok(BatchExecution::Auto),
        "serial" => Ok(BatchExecution::Serial),
        "parallel" => Ok(BatchExecution::Parallel),
        _ => Err(BenchError::UnknownBatchExecution(value.to_owned())),
    }
}

fn parse_resize_parity(value: &str) -> Result<ResizeParity, BenchError> {
    match value {
        "fast" | "resampling" => Ok(ResizeParity::Resampling),
        "compatibility" | "pillow" | "pillow_compatibility" => Ok(ResizeParity::Compatibility),
        "torchvision" | "torchvision_compatibility" => Ok(ResizeParity::Torchvision),
        "pixel_exact" | "pixel-exact" | "nearest" => Ok(ResizeParity::PixelExact),
        _ => Err(BenchError::UnknownResizeProfile(value.to_owned())),
    }
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, BenchError> {
    value
        .split(',')
        .map(|part| parse_positive_usize("--batch-sizes", part.trim()))
        .collect()
}

fn parse_positive_usize(flag: &'static str, value: &str) -> Result<usize, BenchError> {
    let value = parse_usize(flag, value)?;
    if value == 0 {
        Err(BenchError::ZeroBatchSize)
    } else {
        Ok(value)
    }
}

fn parse_usize(flag: &'static str, value: &str) -> Result<usize, BenchError> {
    value.parse::<usize>().map_err(|_| BenchError::ParseInt {
        flag,
        value: value.to_owned(),
    })
}

fn batch_paths(images: &[PathBuf], batch_size: usize) -> Vec<PathBuf> {
    images.iter().cycle().take(batch_size).cloned().collect()
}

fn join_stages(stages: &[BenchStage]) -> String {
    stages
        .iter()
        .map(BenchStage::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn format_decode_backend(backend: ImageDecodeBackend) -> &'static str {
    match backend {
        ImageDecodeBackend::ImageCrate => "image",
        ImageDecodeBackend::TurboJpeg => "turbojpeg",
    }
}

fn format_batch_execution(execution: BatchExecution) -> &'static str {
    match execution {
        BatchExecution::Auto => "auto",
        BatchExecution::Serial => "serial",
        BatchExecution::Parallel => "parallel",
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_usage() {
    println!(
        "Usage: cargo bench -p image-processors --bench clip_batch -- \
         --image path/to/image.jpg [--image path/to/other.png] \
         [--batch-sizes 1,2,4,8,16,32] [--repetitions 5] [--warmup 1] \
         [--resize-profile fast|compatibility] \
         [--decode-backend image|turbojpeg] [--batch-execution auto|serial|parallel] \
         [--reuse-workspace] \
         [--stage full|decode_only|decode_resize|decode_resize_tensor|predecoded_batch] \
         [--stages full,predecoded_batch] [--all-stages]"
    );
    println!("Positional image paths are also accepted.");
}
