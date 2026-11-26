use crate::transforms::TransformError;
use crate::{
    ImageCropBox, ImageFrame, ImageProcessor, ImageProcessorConfig, ImageSize, Layout, PixelFormat,
    ResizeFilter, ResizeParity, TensorData,
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

/// OCR resolution policy for small images and individual source slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SmallImageUpsize {
    /// Resize unsliced images towards the configured scale resolution.
    Always,
    /// Keep small unsliced images at their patch-aligned native resolution.
    Never,
    /// Select resolution using the pinned glyph classifier, including per-slice reductions.
    Adaptive,
}

impl SmallImageUpsize {
    /// Released processor JSON spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Never => "never",
            Self::Adaptive => "adaptive",
        }
    }
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
        resize_parity: if settings["image_processor_type"] == "MiniCPMOcrImageProcessor"
            || settings["video_processor_type"] == "MiniCPMOcrVideoProcessor"
        {
            ResizeParity::Torchvision
        } else {
            ResizeParity::Compatibility
        },
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

// Transformers 863d262, minicpm_ocr: adaptive_upsize_resolution. Keep the
// threshold's first-maximum tie policy and population probabilities in f64.
fn adaptive_upsize_resolution(frame: &ImageFrame, resolution: usize) -> Result<Option<usize>> {
    let rgb = crate::convert_frame_pixel_format(frame, PixelFormat::Rgb8)?;
    let gray = rgb
        .data()
        .chunks_exact(3)
        .map(|p| {
            ((u32::from(p[0]) * 19595 + u32::from(p[1]) * 38470 + u32::from(p[2]) * 7471 + 32768)
                >> 16) as u8
        })
        .collect::<Vec<_>>();
    let mut histogram = [0usize; 256];
    for &pixel in &gray {
        histogram[usize::from(pixel)] += 1;
    }
    let scale = 1.0 / gray.len() as f64;
    let mean = histogram
        .iter()
        .enumerate()
        .map(|(i, &n)| i as f64 * n as f64)
        .sum::<f64>()
        * scale;
    let (mut probability, mut lower_mean, mut best_variance) = (0.0f64, 0.0f64, 0.0f64);
    let mut threshold = 0u8;
    for (index, &count) in histogram.iter().enumerate() {
        let p = count as f64 * scale;
        lower_mean *= probability;
        probability += p;
        let upper = 1.0 - probability;
        if probability.min(upper) < f64::from(f32::EPSILON)
            || probability.max(upper) > 1.0 - f64::from(f32::EPSILON)
        {
            continue;
        }
        lower_mean = (lower_mean + index as f64 * p) / probability;
        let upper_mean = (mean - probability * lower_mean) / upper;
        let variance = probability * upper * (lower_mean - upper_mean).powi(2);
        if variance > best_variance {
            best_variance = variance;
            threshold = index as u8;
        }
    }
    let mut foreground = gray
        .iter()
        .map(|&value| value <= threshold)
        .collect::<Vec<_>>();
    if foreground.iter().filter(|&&value| value).count() as f64 / gray.len() as f64 > 0.5 {
        foreground.iter_mut().for_each(|value| *value = !*value);
    }
    let (_, stats) = crate::transforms::connected_mask_regions(
        &foreground,
        ImageSize {
            width: frame.width(),
            height: frame.height(),
        },
        true,
    )?;
    let mut heights = stats
        .iter()
        .skip(1)
        .filter(|stat| {
            stat[4] >= 3
                && stat[3] >= 2
                && (stat[2] as f64) < 0.9 * frame.width() as f64
                && (stat[3] as f64) < 0.9 * frame.height() as f64
        })
        .map(|stat| stat[3])
        .collect::<Vec<_>>();
    heights.sort_unstable();
    let font = if heights.is_empty() {
        0.0
    } else {
        (heights[(heights.len() - 1) / 2] + heights[heights.len() / 2]) as f64 / 2.0
    };
    let density = heights.len() as f64 * 1000.0 / gray.len() as f64;
    Ok(if font > 0.0 && font <= 12.0 && density >= 2.0 {
        Some(resolution)
    } else if gray.len() < 6272 {
        Some(84)
    } else {
        None
    })
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
    // The original MiniCPM-V API and its four-patch alignment remain unchanged.
    // The OCR processor opts into its source-specific image policy via JSON.
    let ocr_image = settings["image_processor_type"] == "MiniCPMOcrImageProcessor";
    let ocr = ocr_image || settings["video_processor_type"] == "MiniCPMOcrVideoProcessor";
    let policy = if ocr_image {
        settings["small_image_upsize"].as_str().unwrap_or("always")
    } else {
        "always"
    };
    if !matches!(policy, "always" | "never" | "adaptive") {
        return Err(invalid(
            "small_image_upsize must be always, never or adaptive",
        ));
    }
    if ocr_image && !resize {
        return Err(invalid("MiniCPM-OCR requires patch-grid resizing"));
    }
    let factor = if ocr {
        match options.downsample_mode {
            Some(DownsampleMode::Four) => 2,
            Some(DownsampleMode::Sixteen) => 4,
            None => match settings["downsample_mode"].as_str().unwrap_or("4x") {
                "4x" => 2,
                "16x" => 4,
                _ => return Err(invalid("downsample_mode must be 4x or 16x")),
            },
        }
    } else {
        4
    };
    let alignment = patch
        .checked_mul(factor)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let grid = if slice && resize {
        grid(frame.height(), frame.width(), maximum, resolution)
    } else {
        None
    };
    let adaptive = if grid.is_none() && policy == "adaptive" {
        adaptive_upsize_resolution(frame, resolution)?
    } else {
        None
    };
    let (height, width) = if resize {
        best_resize(
            frame.height(),
            frame.width(),
            adaptive.unwrap_or(resolution),
            alignment,
            grid.is_none() && (policy == "always" || adaptive.is_some()),
        )
    } else {
        (frame.height(), frame.width())
    };
    if height % alignment != 0 || width % alignment != 0 {
        return Err(invalid(
            "source raster dimensions must align with the selected vision merger",
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
        let (height, width) = best_resize(cell_height, cell_width, resolution, alignment, true);
        let full_height = height
            .checked_mul(rows)
            .ok_or_else(|| invalid("slice height overflow"))?;
        let full_width = width
            .checked_mul(columns)
            .ok_or_else(|| invalid("slice width overflow"))?;
        let values = transform(frame, full_height, full_width, settings, true)?;
        for row in 0..rows {
            for column in 0..columns {
                if policy == "adaptive" {
                    // Crop ORIGINAL cells with Python's ties-to-even boundaries,
                    // retaining the baseline tile unless fewer tokens suffice.
                    let top = (frame.height() as f64 * row as f64 / rows as f64).round_ties_even()
                        as usize;
                    let bottom = (frame.height() as f64 * (row + 1) as f64 / rows as f64)
                        .round_ties_even() as usize;
                    let left = (frame.width() as f64 * column as f64 / columns as f64)
                        .round_ties_even() as usize;
                    let right = (frame.width() as f64 * (column + 1) as f64 / columns as f64)
                        .round_ties_even() as usize;
                    if bottom > top
                        && right > left
                        && (bottom - top) as f64 * (right - left) as f64
                            <= (resolution as f64).powi(2)
                    {
                        let cell =
                            crate::crop_frame(frame, ImageCropBox::new(left, top, right, bottom))?;
                        let target = adaptive_upsize_resolution(&cell, resolution)?;
                        if target != Some(resolution) {
                            let (h, w) = best_resize(
                                cell.height(),
                                cell.width(),
                                target.unwrap_or(resolution),
                                alignment,
                                target.is_some(),
                            );
                            if h * w < height * width {
                                // _resolve_tile_size explicitly uses bicubic,
                                // independently of the overview's resample option.
                                let mut cell_settings = settings.clone();
                                cell_settings["resample"] = Value::from(3);
                                rasters.push(Raster {
                                    values: transform(&cell, h, w, &cell_settings, true)?,
                                    height: h,
                                    width: w,
                                });
                                continue;
                            }
                        }
                    }
                }
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

#[cfg(test)]
mod ocr_original_reference_tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Original {
        geometry: Vec<Geometry>,
        adaptive: Vec<Adaptive>,
        packing: Vec<Packing>,
    }
    #[derive(Deserialize)]
    struct Geometry {
        height: usize,
        width: usize,
        mode: String,
        allow_upscale: bool,
        scale_resolution: usize,
        patch_size: usize,
        best_height: usize,
        best_width: usize,
        grid: Option<[usize; 2]>,
        refine: Option<[usize; 2]>,
    }
    #[derive(Deserialize)]
    struct Adaptive {
        height: usize,
        width: usize,
        pattern: String,
        rgb: Vec<u8>,
        expected_resolution: Option<usize>,
    }
    #[derive(Deserialize)]
    struct Packing {
        height: usize,
        width: usize,
        patch_size: usize,
        input_chw: Vec<f32>,
        expected_shape: Vec<usize>,
        expected: Vec<f32>,
    }
    fn original() -> Original {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/minicpm_ocr_original.json"
        ))
        .expect("source-matched original MiniCPM-OCR CPU fixtures")
    }
    #[test]
    fn ocr_geometry_matches_original_resize_grid_and_refine() {
        for (index, case) in original().geometry.into_iter().enumerate() {
            let alignment = case.patch_size * if case.mode == "4x" { 2 } else { 4 };
            assert_eq!(
                best_resize(
                    case.height,
                    case.width,
                    case.scale_resolution,
                    alignment,
                    case.allow_upscale
                ),
                (case.best_height, case.best_width),
                "original resize case {index}"
            );
            assert_eq!(
                grid(case.height, case.width, 9, case.scale_resolution),
                case.grid.map(|[rows, columns]| (rows, columns)),
                "original grid case {index}"
            );
            if let (Some([rows, columns]), Some([height, width])) = (case.grid, case.refine) {
                let refined_height = rounded(case.height, rows);
                let refined_width = rounded(case.width, columns);
                let (tile_height, tile_width) = best_resize(
                    refined_height / rows,
                    refined_width / columns,
                    case.scale_resolution,
                    alignment,
                    true,
                );
                assert_eq!(
                    (tile_height * rows, tile_width * columns),
                    (height, width),
                    "original refinement case {index}"
                );
            }
        }
    }
    #[test]
    fn ocr_adaptive_glyph_policy_matches_original_opencv() {
        for case in original().adaptive {
            let frame = ImageFrame::new(case.width, case.height, PixelFormat::Rgb8, case.rgb)
                .expect("valid original RGB fixture");
            assert_eq!(
                adaptive_upsize_resolution(&frame, 448).expect("valid adaptive preprocessing"),
                case.expected_resolution,
                "{}x{} {}",
                case.height,
                case.width,
                case.pattern
            );
        }
    }
    #[test]
    fn ocr_patch_order_matches_original_torch_unfold() {
        for case in original().packing {
            let raster = Raster {
                values: case.input_chw,
                height: case.height,
                width: case.width,
            };
            let packed = pack(&raster, case.patch_size).expect("valid original patch fixture");
            assert_eq!(packed, case.expected);
            assert_eq!(case.expected_shape.iter().product::<usize>(), packed.len());
        }
    }
    #[test]
    fn ocr_options_reach_public_image_preparation_and_preserve_legacy_alignment() {
        let fixtures = original();
        let frame =
            ImageFrame::new(42, 28, PixelFormat::Rgb8, vec![255; 42 * 28 * 3]).expect("RGB");
        for (processor, mode, policy, expected_mode, upscale) in [
            ("MiniCPMOcrImageProcessor", "4x", "never", "4x", false),
            ("MiniCPMOcrImageProcessor", "16x", "never", "16x", false),
            ("MiniCPMOcrImageProcessor", "4x", "always", "4x", true),
            ("MiniCPMOcrImageProcessor", "16x", "always", "16x", true),
            ("MiniCPMV4_6ImageProcessor", "4x", "never", "16x", true),
        ] {
            let reference = fixtures
                .geometry
                .iter()
                .find(|case| {
                    case.height == 28
                        && case.width == 42
                        && case.mode == expected_mode
                        && case.allow_upscale == upscale
                        && case.scale_resolution == 448
                })
                .expect("original caller-policy geometry case");
            let settings = serde_json::json!({"image_processor_type":processor, "downsample_mode":mode,
                "small_image_upsize":policy,"scale_resolution":448,"slice_mode":false,
                "do_resize":true,"do_rescale":false,"do_normalize":false});
            let result = image(&frame, &settings, &ProcessorOptions::default(), 14)
                .expect("valid caller policy");
            assert_eq!(
                (result.rasters[0].height, result.rasters[0].width),
                (reference.best_height, reference.best_width),
                "{processor} {mode} {policy}"
            );
        }
    }
}
