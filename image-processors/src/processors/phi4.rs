use crate::tensor::{Layout, TensorData};
use crate::transforms::TransformError;
use crate::{ImageFrame, ImageProcessor, ImageProcessorConfig, PixelFormat};
use image::RgbImage;
use image_resize_kernels::{resize_u8, ImageSize, ResizeFilter, ResizeProfile};

type Result<T> = std::result::Result<T, TransformError>;

fn invalid(message: impl std::fmt::Display) -> TransformError {
    TransformError::ImageFrameInvariant(message.to_string())
}

#[derive(Clone, Debug)]
pub struct ImageProcessorOptions {
    pub crop_size: usize,
    pub patch_grid: usize,
    pub max_crops: usize,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Debug)]
pub struct PreparedImage {
    pub pixels: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub patch_attention_mask: Vec<u8>,
    pub mask_height: usize,
    pub mask_width: usize,
    pub num_tokens: usize,
}

#[allow(clippy::manual_is_multiple_of)]
pub fn prepare_image(image: &ImageFrame, options: &ImageProcessorOptions) -> Result<PreparedImage> {
    if image.pixel_format() != PixelFormat::Rgb8 {
        return Err(invalid("Phi4 canvas requires RGB8 pixels"));
    }
    let crop = options.crop_size;
    let grid = options.patch_grid;
    let width = image.width();
    let height = image.height();
    if width == 0
        || height == 0
        || crop == 0
        || grid == 0
        || crop % grid != 0
        || options.max_crops == 0
    {
        return Err(invalid("invalid image geometry"));
    }
    let mut columns = width.div_ceil(crop);
    let mut rows = height.div_ceil(crop);
    if columns
        .checked_mul(rows)
        .ok_or_else(|| invalid("image geometry overflow"))?
        > options.max_crops
    {
        let ratio = width as f64 / height as f64;
        let mut candidates = (1..=options.max_crops)
            .flat_map(|column| (1..=options.max_crops / column).map(move |row| (column, row)))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|&(column, row)| (column * row, column, row));
        let mut difference = f64::INFINITY;
        for (column, row) in candidates {
            let candidate = (ratio - column as f64 / row as f64).abs();
            if candidate < difference
                || (candidate == difference
                    && (width * height) as f64 > 0.5 * (crop * crop * column * row) as f64)
            {
                columns = column;
                rows = row;
                difference = candidate;
            }
        }
    }
    let target_width = columns
        .checked_mul(crop)
        .ok_or_else(|| invalid("HD width overflow"))?;
    let target_height = rows
        .checked_mul(crop)
        .ok_or_else(|| invalid("HD height overflow"))?;
    let width_ratio = target_width as f64 / width as f64;
    let height_ratio = target_height as f64 / height as f64;
    let (resized_width, resized_height) = if width_ratio < height_ratio {
        (target_width, (height as f64 * width_ratio) as usize)
    } else {
        ((width as f64 * height_ratio) as usize, target_height)
    };
    let resized = resize_u8(
        image.data(),
        ImageSize { height, width },
        3,
        ImageSize {
            height: resized_height,
            width: resized_width,
        },
        ResizeFilter::Bilinear,
        ResizeProfile::Torchvision,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let resized = RgbImage::from_raw(
        u32::try_from(resized_width).map_err(|_| invalid("resized width overflow"))?,
        u32::try_from(resized_height).map_err(|_| invalid("resized height overflow"))?,
        resized,
    )
    .ok_or_else(|| invalid("invalid resized RGB storage"))?;
    let mut canvas = RgbImage::from_pixel(
        u32::try_from(target_width).map_err(|_| invalid("HD width overflow"))?,
        u32::try_from(target_height).map_err(|_| invalid("HD height overflow"))?,
        image::Rgb([255; 3]),
    );
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    let frame = ImageFrame::new(
        target_width,
        target_height,
        PixelFormat::Rgb8,
        canvas.into_raw(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: false,
        pixel_format: Some(PixelFormat::Rgb8),
        do_rescale: true,
        rescale_factor: options.rescale_factor,
        do_normalize: true,
        image_mean: options.image_mean.to_vec(),
        image_std: options.image_std.to_vec(),
        output_layout: Layout::NCHW,
        ..ImageProcessorConfig::default()
    })
    .map_err(|error| invalid(error.to_string()))?;
    let processed = processor
        .preprocess_image(&frame)
        .map_err(|error| invalid(error.to_string()))?;
    let TensorData::F32(pixels) = processed.data() else {
        return Err(invalid("shared image processor must produce FP32 pixels"));
    };
    let patch_size = crop / grid;
    let mask_height = rows * grid;
    let mask_width = columns * grid;
    let valid_height = mask_height - (target_height - resized_height) / patch_size;
    let valid_width = mask_width - (target_width - resized_width) / patch_size;
    let mut mask = vec![0u8; mask_height * mask_width];
    for y in 0..valid_height {
        mask[y * mask_width..y * mask_width + valid_width].fill(1);
    }
    let pooled = grid.div_ceil(2);
    // Downsample each tile mask independently, as the upstream processor does.
    let useful_height = (0..rows)
        .map(|row| (valid_height.saturating_sub(row * grid).min(grid)).div_ceil(2))
        .sum::<usize>();
    let useful_width = (0..columns)
        .map(|column| (valid_width.saturating_sub(column * grid).min(grid)).div_ceil(2))
        .sum::<usize>();
    Ok(PreparedImage {
        pixels: pixels.clone(),
        height: target_height,
        width: target_width,
        patch_attention_mask: mask,
        mask_height,
        mask_width,
        num_tokens: useful_height * (useful_width + 1) + 1 + pooled * (pooled + 1),
    })
}
