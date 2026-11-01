use crate::transforms::TransformError;
use crate::{
    ImageFrame, ImageProcessor, ImageProcessorConfig, Layout, PixelFormat, ResizeFilter,
    ResizeParity, TensorData,
};
use serde_json::Value;

type Result<T> = std::result::Result<T, TransformError>;

fn invalid(message: impl std::fmt::Display) -> TransformError {
    TransformError::ImageFrameInvariant(message.to_string())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DownsampleMode {
    Four,
    #[default]
    Sixteen,
}

#[derive(Clone, Debug, Default)]
pub struct ProcessorOptions {
    pub downsample_mode: Option<DownsampleMode>,
    pub max_slice_nums: Option<usize>,
    pub scale_resolution: Option<usize>,
    pub slice_mode: Option<bool>,
    pub use_image_id: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct VideoOptions {
    pub source_fps: f64,
    pub duration: Option<f64>,
    pub max_num_frames: Option<usize>,
    pub stack_frames: Option<usize>,
    pub do_sample_frames: Option<bool>,
}

pub struct Raster {
    pub values: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

pub struct VisualUnit {
    pub rasters: Vec<Raster>,
    pub grid: Option<(usize, usize)>,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            source_fps: f64::NAN,
            duration: None,
            max_num_frames: None,
            stack_frames: None,
            do_sample_frames: None,
        }
    }
}

fn rounded(value: usize, factor: usize) -> usize {
    (((value as f64 / factor as f64).round_ties_even() as usize).max(1)) * factor
}

fn best_resize(
    height: usize,
    width: usize,
    resolution: usize,
    alignment: usize,
    upscale: bool,
) -> (usize, usize) {
    let (height, width) = if height as f64 * width as f64 > (resolution as f64).powi(2) || upscale {
        let ratio = width as f64 / height as f64;
        let height = (resolution as f64 / ratio.sqrt()) as usize;
        (height, (height as f64 * ratio) as usize)
    } else {
        (height, width)
    };
    (rounded(height, alignment), rounded(width, alignment))
}

fn grid(height: usize, width: usize, maximum: usize, resolution: usize) -> Option<(usize, usize)> {
    let multiple =
        ((height as f64 * width as f64 / (resolution as f64).powi(2)).ceil() as usize).min(maximum);
    if multiple <= 1 {
        return None;
    }
    let ratio = (width as f64 / height as f64).ln();
    let mut best = (1, 1);
    let mut error = f64::INFINITY;
    for slices in [multiple - 1, multiple, multiple + 1] {
        if slices <= 1 || slices > maximum {
            continue;
        }
        for rows in 1..=slices {
            if slices % rows == 0 {
                let columns = slices / rows;
                let candidate = (ratio - (columns as f64 / rows as f64).ln()).abs();
                if candidate < error || (candidate == error && rows > best.0) {
                    best = (rows, columns);
                    error = candidate;
                }
            }
        }
    }
    Some(best)
}

fn vector(settings: &Value, name: &str) -> Result<Vec<f32>> {
    if let Some(values) = settings[name].as_array() {
        values
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| invalid("invalid normalization"))
            })
            .collect()
    } else if let Some(value) = settings[name].as_f64() {
        Ok(vec![value as f32])
    } else {
        Ok(vec![0.5; 3])
    }
}

fn transform(
    frame: &ImageFrame,
    height: usize,
    width: usize,
    settings: &Value,
    normalize: bool,
) -> Result<Vec<f32>> {
    if settings["do_convert_rgb"].as_bool() == Some(false)
        && frame.pixel_format() != PixelFormat::Rgb8
    {
        return Err(invalid(
            "disabled RGB conversion requires RGB input for the native RGB vision processor",
        ));
    }
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: height != frame.height() || width != frame.width(),
        height: Some(height),
        width: Some(width),
        pixel_format: Some(PixelFormat::Rgb8),
        resample: match settings["resample"].as_u64().unwrap_or(3) {
            0 => ResizeFilter::Nearest,
            2 => ResizeFilter::Bilinear,
            3 => ResizeFilter::Bicubic,
            _ => return Err(invalid("unsupported resize filter")),
        },
        resize_parity: ResizeParity::Compatibility,
        do_rescale: normalize && settings["do_rescale"].as_bool().unwrap_or(true),
        rescale_factor: settings["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0) as f32,
        do_normalize: normalize && settings["do_normalize"].as_bool().unwrap_or(true),
        image_mean: vector(settings, "image_mean")?,
        image_std: vector(settings, "image_std")?,
        output_layout: Layout::NCHW,
        ..Default::default()
    })
    .map_err(invalid)?;
    let output = processor
        .preprocess_images(std::slice::from_ref(frame))
        .map_err(invalid)?;
    let TensorData::F32(values) = output.data() else {
        return Err(invalid("processor must produce FP32 pixels"));
    };
    Ok(values.clone())
}

pub fn image(
    frame: &ImageFrame,
    settings: &Value,
    options: &ProcessorOptions,
    patch: usize,
) -> Result<VisualUnit> {
    let maximum = options
        .max_slice_nums
        .unwrap_or(settings["max_slice_nums"].as_u64().unwrap_or(9) as usize);
    let resolution = options
        .scale_resolution
        .unwrap_or(settings["scale_resolution"].as_u64().unwrap_or(448) as usize);
    if resolution == 0 || patch == 0 {
        return Err(invalid("positive processor sizes required"));
    }
    let slice = options
        .slice_mode
        .unwrap_or(settings["slice_mode"].as_bool().unwrap_or(true));
    let resize = settings["do_resize"].as_bool().unwrap_or(true);
    let grid = if slice && resize {
        grid(frame.height(), frame.width(), maximum, resolution)
    } else {
        None
    };
    let (height, width) = if resize {
        best_resize(
            frame.height(),
            frame.width(),
            resolution,
            patch * 4,
            grid.is_none(),
        )
    } else {
        (frame.height(), frame.width())
    };
    if height % (patch * 4) != 0 || width % (patch * 4) != 0 {
        return Err(invalid(
            "source raster dimensions must align with four vision patches",
        ));
    }
    let mut rasters = vec![Raster {
        values: transform(frame, height, width, settings, true)?,
        height,
        width,
    }];
    if let Some((rows, columns)) = grid {
        let refine_height = rounded(frame.height(), rows);
        let refine_width = rounded(frame.width(), columns);
        // Source divides the rounded canvas into cells before the area resize.
        let cell_height = refine_height / rows;
        let cell_width = refine_width / columns;
        let (height, width) = best_resize(cell_height, cell_width, resolution, patch * 4, true);
        let full_height = height
            .checked_mul(rows)
            .ok_or_else(|| invalid("slice height overflow"))?;
        let full_width = width
            .checked_mul(columns)
            .ok_or_else(|| invalid("slice width overflow"))?;
        let values = transform(frame, full_height, full_width, settings, true)?;
        for row in 0..rows {
            for column in 0..columns {
                let mut slice = Vec::with_capacity(3 * height * width);
                for channel in 0..3 {
                    for y in 0..height {
                        let start = channel * full_height * full_width
                            + (row * height + y) * full_width
                            + column * width;
                        slice.extend_from_slice(&values[start..start + width]);
                    }
                }
                rasters.push(Raster {
                    values: slice,
                    height,
                    width,
                });
            }
        }
    }
    Ok(VisualUnit { rasters, grid })
}

pub fn pack(raster: &Raster, patch: usize) -> Result<Vec<f32>> {
    let expected = raster
        .height
        .checked_mul(raster.width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(TransformError::ImageSizeOverflow)?;
    if patch == 0
        || raster.height == 0
        || raster.width == 0
        || !raster.height.is_multiple_of(patch)
        || !raster.width.is_multiple_of(patch)
        || raster.values.len() != expected
    {
        return Err(invalid(
            "packed RGB raster must have patch-aligned dimensions and matching storage",
        ));
    }
    let mut values = Vec::with_capacity(raster.values.len());
    for channel in 0..3 {
        for y in 0..patch {
            for row in 0..raster.height / patch {
                for column in 0..raster.width / patch {
                    let start = channel * raster.height * raster.width
                        + (row * patch + y) * raster.width
                        + column * patch;
                    values.extend_from_slice(&raster.values[start..start + patch]);
                }
            }
        }
    }
    Ok(values)
}

fn uniform<T: Clone>(values: &[T], maximum: usize) -> Vec<T> {
    if values.len() <= maximum {
        return values.to_vec();
    }
    (0..maximum)
        .map(|i| {
            values[if maximum == 1 {
                0
            } else {
                i * (values.len() - 1) / (maximum - 1)
            }]
            .clone()
        })
        .collect()
}

fn canvas(frames: &[&ImageFrame], settings: &Value) -> Result<ImageFrame> {
    let first = frames
        .first()
        .ok_or_else(|| invalid("empty frame canvas"))?;
    let (height, width) = (first.height(), first.width());
    if frames
        .iter()
        .any(|frame| frame.height() != height || frame.width() != width)
    {
        return Err(invalid("stacked video frames must have equal dimensions"));
    }
    let ratio = |rows: usize, columns: usize| {
        (((columns * width + (columns - 1) * 6) as f64 / (rows * height + (rows - 1) * 6) as f64)
            - 1.0)
            .abs()
    };
    let (rows, columns) = match frames.len() {
        4 => (2, 2),
        2 | 3 => {
            let n = frames.len();
            let a = ratio(1, n);
            let b = ratio(n, 1);
            if a < b || (a == b && (n != 2 || width >= height)) {
                (1, n)
            } else {
                (n, 1)
            }
        }
        n => (1, n),
    };
    let full_height = rows * height + (rows - 1) * 6;
    let full_width = columns * width + (columns - 1) * 6;
    let length = full_height
        .checked_mul(full_width)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("canvas dimensions overflow"))?;
    let mut pixels = vec![0; length];
    for (i, frame) in frames.iter().enumerate() {
        let rgb = transform(frame, height, width, settings, false)?;
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    let destination = ((i / columns * (height + 6) + y) * full_width
                        + i % columns * (width + 6)
                        + x)
                        * 3
                        + channel;
                    pixels[destination] =
                        rgb[channel * height * width + y * width + x].clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    ImageFrame::new(full_width, full_height, PixelFormat::Rgb8, pixels).map_err(invalid)
}

pub fn video(
    frames: &[ImageFrame],
    options: &VideoOptions,
    settings: &Value,
    processing: &ProcessorOptions,
    patch: usize,
) -> Result<Vec<VisualUnit>> {
    let max_num_frames = options
        .max_num_frames
        .unwrap_or(settings["max_num_frames"].as_u64().unwrap_or(128) as usize);
    let stack_frames = options
        .stack_frames
        .unwrap_or(settings["stack_frames"].as_u64().unwrap_or(1) as usize);
    let do_sample_frames = options
        .do_sample_frames
        .unwrap_or(settings["do_sample_frames"].as_bool().unwrap_or(true));
    if frames.is_empty() || stack_frames == 0 {
        return Err(invalid("video requires frames and a positive stack size"));
    }
    // Preselected, unstacked frames are already the visual units. Neither
    // timestamps nor the original stream's sampling metadata are involved.
    if !do_sample_frames && stack_frames == 1 {
        return frames
            .iter()
            .map(|frame| image(frame, settings, processing, patch))
            .collect();
    }
    if frames.is_empty()
        || !options.source_fps.is_finite()
        || options.source_fps <= 0.0
        || max_num_frames == 0
        || stack_frames == 0
    {
        return Err(invalid(
            "video requires frames, positive fps and positive sampling bounds",
        ));
    }
    let duration = options
        .duration
        .unwrap_or(frames.len() as f64 / options.source_fps);
    if !duration.is_finite() || duration <= 0.0 {
        return Err(invalid("video duration must be positive"));
    }
    let index = |time: f64| ((time * options.source_fps) as usize).min(frames.len() - 1);
    let main = if duration > max_num_frames as f64 {
        let timestamps = (0..(duration / 0.1) as usize)
            .map(|i| i as f64 / 10.0)
            .collect::<Vec<_>>();
        uniform(&timestamps, max_num_frames)
    } else {
        (0..duration.ceil() as usize).map(|i| i as f64).collect()
    };
    let mut sub = Vec::new();
    for second in 0..duration.ceil() as usize {
        for j in 1..stack_frames {
            let time = second as f64 + j as f64 / stack_frames as f64;
            if time < duration {
                sub.push(time);
            }
        }
    }
    let sub = uniform(&sub, max_num_frames.saturating_mul(stack_frames - 1));
    let main_count = if do_sample_frames {
        main.len()
    } else {
        frames
            .len()
            .checked_sub(sub.len())
            .ok_or_else(|| invalid("preselected video lacks stacked subframes"))?
    };
    let mut units = Vec::new();
    for i in 0..main_count {
        let main_frame = if do_sample_frames {
            &frames[index(main[i])]
        } else {
            &frames[i]
        };
        units.push(image(main_frame, settings, processing, patch)?);
        let group = sub
            .iter()
            .enumerate()
            .filter(|(_, time)| **time >= i as f64 && **time < (i + 1) as f64)
            .map(|(j, time)| {
                if do_sample_frames {
                    &frames[index(*time)]
                } else {
                    &frames[main_count + j]
                }
            })
            .collect::<Vec<_>>();
        if !group.is_empty() {
            units.push(image(
                &canvas(&group, settings)?,
                settings,
                processing,
                patch,
            )?);
        }
    }
    Ok(units)
}
