use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use image_processors::{
    BatchExecution, ClipImageProcessor, ClipImageProcessorConfig, ImageDecodeBackend, ImageLayout,
    ImageProcessorWorkspace, ResizeParity, Tensor, DEFAULT_PARALLEL_BATCH_THRESHOLD,
};

const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_WARMUP_BATCHES: usize = 1;

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> BenchResult<()> {
    let args = BenchArgs::parse(env::args().skip(1))?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let mut paths = collect_image_paths(&args.image_dir)?;
    if let Some(limit) = args.limit {
        paths.truncate(limit);
    }
    if paths.is_empty() {
        return Err(Box::new(BenchError::NoImages));
    }

    let processor = ClipImageProcessor::new(ClipImageProcessorConfig {
        resize_parity: args.resize_parity,
        decode_backend: args.decode_backend,
        batch_execution: args.batch_execution,
        output_layout: args.output_layout,
        ..Default::default()
    })?;
    let mut workspace = args.reuse_workspace.then(ImageProcessorWorkspace::new);
    let warmup_images = warmup_images(&paths, args.batch_size, args.warmup_batches);

    for chunk in warmup_images.chunks(args.batch_size) {
        let tensor = run_batch(&processor, chunk, workspace.as_mut())?;
        std::hint::black_box(tensor.data().len());
        if args.recycle_workspace_output {
            if let Some(workspace) = workspace.as_mut() {
                workspace.recycle_tensor(tensor)?;
            }
        }
    }

    let started = Instant::now();
    let outcome = process_paths(&processor, &paths, &args)?;
    let elapsed = started.elapsed();

    println!("processor,clip");
    println!("image_dir,{}", args.image_dir.display());
    println!("images,{}", outcome.processed);
    println!("failed_images,{}", outcome.failed);
    println!("batches,{}", outcome.batches);
    println!("batch_size,{}", args.batch_size);
    println!("outer_workers,{}", args.outer_workers);
    println!("output_layout,{}", format_output_layout(args.output_layout));
    println!("warmup_batches,{}", args.warmup_batches);
    println!(
        "resize_profile,{}",
        format_resize_parity(args.resize_parity)
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
    println!("recycle_workspace_output,{}", args.recycle_workspace_output);
    println!("skip_errors,{}", args.skip_errors);
    println!("total_ms,{:.3}", millis(elapsed));
    println!(
        "images_per_second,{:.3}",
        outcome.processed as f64 / elapsed.as_secs_f64()
    );
    println!(
        "ms_per_image,{:.6}",
        millis(elapsed) / outcome.processed as f64
    );
    println!("last_shape,{:?}", outcome.last_shape.unwrap_or_default());
    println!("last_layout,{}", outcome.last_layout.unwrap_or_default());

    Ok(())
}

fn run_batch(
    processor: &ClipImageProcessor,
    paths: &[PathBuf],
    workspace: Option<&mut ImageProcessorWorkspace>,
) -> BenchResult<Tensor> {
    match workspace {
        Some(workspace) => processor
            .open_batch_into(paths, workspace)
            .map_err(Into::into),
        None => processor.open_batch(paths).map_err(Into::into),
    }
}

#[derive(Debug, Default)]
struct ChunkOutcome {
    processed: usize,
    failed: usize,
    batches: usize,
    last_shape: Option<Vec<usize>>,
    last_layout: Option<String>,
}

impl ChunkOutcome {
    fn append(&mut self, other: Self) {
        self.processed += other.processed;
        self.failed += other.failed;
        self.batches += other.batches;
        if other.last_shape.is_some() {
            self.last_shape = other.last_shape;
        }
        if other.last_layout.is_some() {
            self.last_layout = other.last_layout;
        }
    }
}

fn process_paths(
    processor: &ClipImageProcessor,
    paths: &[PathBuf],
    args: &BenchArgs,
) -> BenchResult<ChunkOutcome> {
    if args.outer_workers == 1 {
        let mut workspace = args.reuse_workspace.then(ImageProcessorWorkspace::new);
        return process_path_slice(
            processor,
            paths,
            args.batch_size,
            workspace.as_mut(),
            args.skip_errors,
            args.recycle_workspace_output,
        );
    }

    let worker_count = args.outer_workers.min(paths.len());
    let paths_per_worker = paths.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let handles = paths
            .chunks(paths_per_worker)
            .map(|worker_paths| {
                scope.spawn(move || {
                    let mut workspace = args.reuse_workspace.then(ImageProcessorWorkspace::new);
                    process_path_slice(
                        processor,
                        worker_paths,
                        args.batch_size,
                        workspace.as_mut(),
                        args.skip_errors,
                        args.recycle_workspace_output,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut outcome = ChunkOutcome::default();
        for handle in handles {
            let worker = handle
                .join()
                .map_err(|_| Box::new(BenchError::WorkerPanic) as Box<dyn Error + Send + Sync>)??;
            outcome.append(worker);
        }
        Ok(outcome)
    })
}

fn process_path_slice(
    processor: &ClipImageProcessor,
    paths: &[PathBuf],
    batch_size: usize,
    mut workspace: Option<&mut ImageProcessorWorkspace>,
    skip_errors: bool,
    recycle_workspace_output: bool,
) -> BenchResult<ChunkOutcome> {
    let mut outcome = ChunkOutcome::default();
    for chunk in paths.chunks(batch_size) {
        outcome.append(process_chunk(
            processor,
            chunk,
            workspace.as_deref_mut(),
            skip_errors,
            recycle_workspace_output,
        )?);
    }
    Ok(outcome)
}

fn process_chunk(
    processor: &ClipImageProcessor,
    paths: &[PathBuf],
    mut workspace: Option<&mut ImageProcessorWorkspace>,
    skip_errors: bool,
    recycle_workspace_output: bool,
) -> BenchResult<ChunkOutcome> {
    match run_batch(processor, paths, workspace.as_deref_mut()) {
        Ok(tensor) => {
            let shape = tensor.shape().to_vec();
            let layout = format!("{:?}", tensor.layout());
            std::hint::black_box(tensor.data().len());
            if recycle_workspace_output {
                if let Some(workspace) = workspace {
                    workspace.recycle_tensor(tensor)?;
                }
            }
            Ok(ChunkOutcome {
                processed: paths.len(),
                failed: 0,
                batches: 1,
                last_shape: Some(shape),
                last_layout: Some(layout),
            })
        }
        Err(error) if skip_errors && paths.len() == 1 => {
            eprintln!("skipping {}, error={error}", paths[0].display());
            Ok(ChunkOutcome {
                processed: 0,
                failed: 1,
                batches: 0,
                last_shape: None,
                last_layout: None,
            })
        }
        Err(error) if skip_errors => {
            let mid = paths.len() / 2;
            if mid == 0 {
                return Err(error);
            }
            let mut left = process_chunk(
                processor,
                &paths[..mid],
                workspace.as_deref_mut(),
                true,
                recycle_workspace_output,
            )?;
            let right = process_chunk(
                processor,
                &paths[mid..],
                workspace,
                true,
                recycle_workspace_output,
            )?;
            left.processed += right.processed;
            left.failed += right.failed;
            left.batches += right.batches;
            if right.last_shape.is_some() {
                left.last_shape = right.last_shape;
            }
            if right.last_layout.is_some() {
                left.last_layout = right.last_layout;
            }
            Ok(left)
        }
        Err(error) => Err(error),
    }
}

fn warmup_images(paths: &[PathBuf], batch_size: usize, warmup_batches: usize) -> Vec<PathBuf> {
    let count = batch_size.saturating_mul(warmup_batches);
    paths.iter().cycle().take(count).cloned().collect()
}

fn collect_image_paths(dir: &Path) -> BenchResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_file() && is_supported_image_path(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn is_supported_image_path(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "bmp" | "gif" | "ico" | "jpg" | "jpeg" | "png" | "pnm" | "tif" | "tiff" | "webp"
    )
}

#[derive(Debug)]
struct BenchArgs {
    image_dir: PathBuf,
    batch_size: usize,
    outer_workers: usize,
    warmup_batches: usize,
    limit: Option<usize>,
    resize_parity: ResizeParity,
    decode_backend: ImageDecodeBackend,
    batch_execution: BatchExecution,
    output_layout: ImageLayout,
    reuse_workspace: bool,
    recycle_workspace_output: bool,
    skip_errors: bool,
    help: bool,
}

impl BenchArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, BenchError> {
        let mut image_dir = None;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut outer_workers = 1usize;
        let mut warmup_batches = DEFAULT_WARMUP_BATCHES;
        let mut limit = None;
        let mut resize_parity = ResizeParity::Compatibility;
        let mut decode_backend = ImageDecodeBackend::ImageCrate;
        let mut batch_execution = BatchExecution::default();
        let mut output_layout = ImageLayout::ChannelsHeightWidth;
        let mut reuse_workspace = false;
        let mut recycle_workspace_output = true;
        let mut skip_errors = false;
        let mut help = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "-h" | "--help" => help = true,
                "--image-dir" | "--dir" => image_dir = Some(next_path(&mut args, "--image-dir")?),
                "--batch-size" => {
                    batch_size = parse_positive_usize(
                        "--batch-size",
                        &next_value(&mut args, "--batch-size")?,
                    )?;
                }
                "--outer-workers" => {
                    outer_workers = parse_positive_usize(
                        "--outer-workers",
                        &next_value(&mut args, "--outer-workers")?,
                    )?;
                }
                "--warmup-batches" => {
                    warmup_batches = parse_usize(
                        "--warmup-batches",
                        &next_value(&mut args, "--warmup-batches")?,
                    )?;
                }
                "--limit" => {
                    limit = Some(parse_positive_usize(
                        "--limit",
                        &next_value(&mut args, "--limit")?,
                    )?);
                }
                "--resize-profile" | "--resize-parity" => {
                    resize_parity =
                        parse_resize_parity(&next_value(&mut args, "--resize-profile")?)?;
                }
                "--decode-backend" => {
                    decode_backend =
                        parse_decode_backend(&next_value(&mut args, "--decode-backend")?)?;
                }
                "--batch-execution" => {
                    batch_execution =
                        parse_batch_execution(&next_value(&mut args, "--batch-execution")?)?;
                }
                "--output-layout" => {
                    output_layout =
                        parse_output_layout(&next_value(&mut args, "--output-layout")?)?;
                }
                "--reuse-workspace" => reuse_workspace = true,
                "--retain-workspace-only" => {
                    reuse_workspace = true;
                    recycle_workspace_output = false;
                }
                "--skip-errors" => skip_errors = true,
                value if value.starts_with('-') => {
                    return Err(BenchError::UnknownArgument(value.to_owned()));
                }
                value => image_dir = Some(PathBuf::from(value)),
            }
        }

        let image_dir = if help {
            PathBuf::new()
        } else {
            image_dir.ok_or(BenchError::NoImageDir)?
        };

        Ok(Self {
            image_dir,
            batch_size,
            outer_workers,
            warmup_batches,
            limit,
            resize_parity,
            decode_backend,
            batch_execution,
            output_layout,
            reuse_workspace,
            recycle_workspace_output,
            skip_errors,
            help,
        })
    }
}

#[derive(Debug)]
enum BenchError {
    NoImageDir,
    NoImages,
    ParseInt { flag: &'static str, value: String },
    UnknownArgument(String),
    UnknownBatchExecution(String),
    UnknownDecodeBackend(String),
    UnknownResizeProfile(String),
    UnknownOutputLayout(String),
    WorkerPanic,
    ZeroValue(&'static str),
}

impl Display for BenchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoImageDir => write!(formatter, "image directory is required"),
            Self::NoImages => write!(formatter, "no supported images found"),
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
            Self::UnknownOutputLayout(value) => write!(formatter, "unknown output layout {value}"),
            Self::WorkerPanic => write!(formatter, "outer preprocessing worker panicked"),
            Self::ZeroValue(flag) => write!(formatter, "{flag} must be positive"),
        }
    }
}

impl Error for BenchError {}

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
    args.next()
        .ok_or_else(|| BenchError::UnknownArgument(flag.to_owned()))
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

fn parse_output_layout(value: &str) -> Result<ImageLayout, BenchError> {
    match value {
        "nchw" | "chw" => Ok(ImageLayout::ChannelsHeightWidth),
        "nhwc" | "hwc" | "channels-last" => Ok(ImageLayout::HeightWidthChannels),
        _ => Err(BenchError::UnknownOutputLayout(value.to_owned())),
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

fn parse_positive_usize(flag: &'static str, value: &str) -> Result<usize, BenchError> {
    let value = parse_usize(flag, value)?;
    if value == 0 {
        Err(BenchError::ZeroValue(flag))
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

fn format_resize_parity(parity: ResizeParity) -> &'static str {
    match parity {
        ResizeParity::Resampling => "fast",
        ResizeParity::Compatibility => "pillow_compatibility",
        ResizeParity::Torchvision => "torchvision_compatibility",
        ResizeParity::PixelExact => "pixel_exact",
        _ => "unknown",
    }
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

fn format_output_layout(layout: ImageLayout) -> &'static str {
    match layout {
        ImageLayout::ChannelsHeightWidth => "nchw",
        ImageLayout::HeightWidthChannels => "nhwc",
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_usage() {
    println!(
        "Usage: cargo bench -p image-processors --bench clip_dataset -- \
         --image-dir path\\to\\images [--batch-size 32] [--outer-workers 1] [--limit 1000] \
         [--warmup-batches 1] [--resize-profile fast|compatibility] \
         [--decode-backend image|turbojpeg] [--batch-execution auto|serial|parallel] \
         [--output-layout nchw|nhwc] \
         [--reuse-workspace | --retain-workspace-only] [--skip-errors]"
    );
}
