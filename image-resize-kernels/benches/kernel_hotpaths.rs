use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

use image_resize_kernels::{
    resize_u8, resize_u8_crop_into_f32_with_workspace, resize_u8_crop_with_workspace,
    F32ImageLayout, ImageSize, ResizeCrop, ResizeFilter, ResizeProfile, ResizeWorkspace,
};

const DEFAULT_ITERATIONS: usize = 20;
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_WARMUP: usize = 1;

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> BenchResult<()> {
    let args = BenchArgs::parse(env::args().skip(1))?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let source = ImageSize::new(480, 640)?;
    let resized = ImageSize::new(224, 299)?;
    let target = ImageSize::new(224, 224)?;
    let crop = ResizeCrop::new(resized, 37, 0, target)?;
    let rgb = deterministic_rgb(source);
    let gray = deterministic_gray(source);
    let multiplier = [
        1.0 / 255.0 / 0.268_629_54,
        1.0 / 255.0 / 0.261_302_6,
        1.0 / 255.0 / 0.275_777_1,
    ];
    let offset = [
        -0.481_454_66 / 0.268_629_54,
        -0.457_827_5 / 0.261_302_6,
        -0.408_210_73 / 0.275_777_1,
    ];

    println!("bench,kernel_hotpaths");
    println!("source,{}x{}", source.height, source.width);
    println!("resized,{}x{}", resized.height, resized.width);
    println!("target,{}x{}", target.height, target.width);
    println!("iterations,{}", args.iterations);
    println!("warmup,{}", args.warmup);
    println!("repetitions,{}", args.repetitions);
    println!("case,repetitions,iterations,min_ms,mean_ms,max_ms,last_len");

    let mut rgb_crop_workspace = ResizeWorkspace::new();
    bench_case("rgb_crop_to_u8_bicubic", &args, || {
        let resized = resize_u8_crop_with_workspace(
            &rgb,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut rgb_crop_workspace,
        )?;
        Ok(resized.len())
    })?;

    let mut gray_crop_workspace = ResizeWorkspace::new();
    bench_case("gray_crop_to_u8_bicubic", &args, || {
        let resized = resize_u8_crop_with_workspace(
            &gray,
            source,
            1,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut gray_crop_workspace,
        )?;
        Ok(resized.len())
    })?;

    bench_case("rgb_full_resize_bicubic", &args, || {
        let resized = resize_u8(
            &rgb,
            source,
            3,
            resized,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
        )?;
        Ok(resized.len())
    })?;

    let mut f32_chw_workspace = ResizeWorkspace::new();
    let mut f32_chw = vec![0.0; target.height * target.width * 3];
    bench_case("rgb_crop_to_f32_chw_bicubic", &args, || {
        resize_u8_crop_into_f32_with_workspace(
            &rgb,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut f32_chw,
            F32ImageLayout::Chw,
            &multiplier,
            &offset,
            &mut f32_chw_workspace,
        )?;
        Ok(f32_chw.len())
    })?;

    let mut f32_hwc_workspace = ResizeWorkspace::new();
    let mut f32_hwc = vec![0.0; target.height * target.width * 3];
    bench_case("rgb_crop_to_f32_hwc_bicubic", &args, || {
        resize_u8_crop_into_f32_with_workspace(
            &rgb,
            source,
            3,
            crop,
            ResizeFilter::Bicubic,
            ResizeProfile::Pillow,
            &mut f32_hwc,
            F32ImageLayout::Hwc,
            &multiplier,
            &offset,
            &mut f32_hwc_workspace,
        )?;
        Ok(f32_hwc.len())
    })?;

    Ok(())
}

fn bench_case(
    name: &str,
    args: &BenchArgs,
    mut operation: impl FnMut() -> Result<usize, image_resize_kernels::ResizeError>,
) -> BenchResult<()> {
    for _ in 0..args.warmup {
        for _ in 0..args.iterations {
            std::hint::black_box(operation()?);
        }
    }

    let mut timings = Vec::with_capacity(args.repetitions);
    let mut last_len = 0usize;
    for _ in 0..args.repetitions {
        let started = Instant::now();
        for _ in 0..args.iterations {
            last_len = operation()?;
            std::hint::black_box(last_len);
        }
        timings.push(started.elapsed());
    }

    let summary = TimingSummary::new(&timings)?;
    println!(
        "{},{},{},{:.3},{:.3},{:.3},{}",
        name,
        args.repetitions,
        args.iterations,
        millis(summary.min),
        millis(summary.mean),
        millis(summary.max),
        last_len
    );
    Ok(())
}

#[derive(Debug)]
struct BenchArgs {
    iterations: usize,
    repetitions: usize,
    warmup: usize,
    help: bool,
}

impl BenchArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, BenchError> {
        let mut iterations = DEFAULT_ITERATIONS;
        let mut repetitions = DEFAULT_REPETITIONS;
        let mut warmup = DEFAULT_WARMUP;
        let mut help = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "-h" | "--help" => help = true,
                "--iterations" => {
                    iterations = parse_positive_usize(
                        "--iterations",
                        &next_value(&mut args, "--iterations")?,
                    )?;
                }
                "--repetitions" => {
                    repetitions = parse_positive_usize(
                        "--repetitions",
                        &next_value(&mut args, "--repetitions")?,
                    )?;
                }
                "--warmup" => {
                    warmup = parse_usize("--warmup", &next_value(&mut args, "--warmup")?)?;
                }
                value => return Err(BenchError::UnknownArgument(value.to_owned())),
            }
        }

        Ok(Self {
            iterations,
            repetitions,
            warmup,
            help,
        })
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
        let min = values
            .iter()
            .copied()
            .min()
            .ok_or(BenchError::ZeroValue("--repetitions"))?;
        let max = values
            .iter()
            .copied()
            .max()
            .ok_or(BenchError::ZeroValue("--repetitions"))?;
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
    MissingValue(&'static str),
    ParseInt { flag: &'static str, value: String },
    UnknownArgument(String),
    ZeroValue(&'static str),
}

impl Display for BenchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingValue(flag) => write!(formatter, "missing value for {flag}"),
            Self::ParseInt { flag, value } => {
                write!(formatter, "invalid integer for {flag}: {value}")
            }
            Self::UnknownArgument(value) => write!(formatter, "unknown argument {value}"),
            Self::ZeroValue(flag) => write!(formatter, "{flag} must be positive"),
        }
    }
}

impl Error for BenchError {}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, BenchError> {
    args.next().ok_or(BenchError::MissingValue(flag))
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

fn deterministic_rgb(size: ImageSize) -> Vec<u8> {
    (0..size.width * size.height * 3)
        .map(|index| ((index as u32 * 37 + 17) % 256) as u8)
        .collect()
}

fn deterministic_gray(size: ImageSize) -> Vec<u8> {
    (0..size.width * size.height)
        .map(|index| ((index as u32 * 29 + 101) % 256) as u8)
        .collect()
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_usage() {
    println!(
        "Usage: cargo bench -p image-resize-kernels --bench kernel_hotpaths -- \
         [--iterations 20] [--repetitions 5] [--warmup 1]"
    );
}
