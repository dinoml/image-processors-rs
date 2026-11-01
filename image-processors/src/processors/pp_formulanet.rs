use crate::transforms::{
    convert_frame_pixel_format, crop_frame, pad_frame, resize_frame_with_decision, ImageCropBox,
    ImageSize, Padding, ResizeDecision, ResizeFilter, ResizeMode,
};
use crate::{ImageFrame, ImageProcessor, ImageProcessorConfig, PixelFormat};
use serde_json::Value;

fn invalid(message: impl Into<String>) -> String {
    message.into()
}
type Result<T> = std::result::Result<T, String>;

#[allow(missing_docs)]
pub fn preprocess(images: &[ImageFrame], settings: &Value) -> Result<Vec<f32>> {
    let settings = settings.get("image_processor").unwrap_or(settings);
    let height = settings["size"]["height"]
        .as_u64()
        .ok_or_else(|| invalid("missing processor height"))? as usize;
    let width = settings["size"]["width"]
        .as_u64()
        .ok_or_else(|| invalid("missing processor width"))? as usize;
    let target = ImageSize::new(height, width).map_err(|error| invalid(error.to_string()))?;
    let enabled = |name: &str, default: bool| settings[name].as_bool().unwrap_or(default);
    let filter = match settings["resample"].as_u64().unwrap_or(2) {
        0 => ResizeFilter::Nearest,
        1 => ResizeFilter::Lanczos,
        2 => ResizeFilter::Bilinear,
        3 => ResizeFilter::Bicubic,
        value => return Err(invalid(format!("unsupported processor resample {value}"))),
    };
    let normalization = |name: &str| -> Result<Vec<f32>> {
        match &settings[name] {
            Value::Number(value) => Ok(vec![value
                .as_f64()
                .ok_or_else(|| invalid("invalid normalization value"))?
                as f32]),
            value => {
                serde_json::from_value(value.clone()).map_err(|error| invalid(error.to_string()))
            }
        }
    };
    let processor = ImageProcessor::new(ImageProcessorConfig {
        do_resize: false,
        do_rescale: enabled("do_rescale", true),
        rescale_factor: settings["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0) as f32,
        do_normalize: enabled("do_normalize", true),
        image_mean: normalization("image_mean")?,
        image_std: normalization("image_std")?,
        ..ImageProcessorConfig::default()
    })
    .map_err(|error| invalid(error.to_string()))?;
    let mut prepared = Vec::with_capacity(images.len());
    for image in images {
        let mut frame = convert_frame_pixel_format(image, PixelFormat::Rgb8)
            .map_err(|error| invalid(error.to_string()))?;
        if enabled("do_crop_margin", true) {
            frame = crop_margin(&frame)?;
        }
        if enabled("do_align_long_axis", false)
            && ((width < height && frame.width() > frame.height())
                || (width > height && frame.width() < frame.height()))
        {
            frame = crate::transforms::rotate_frame_90_clockwise(&frame)
                .map_err(|error| invalid(error.to_string()))?;
        }
        if enabled("do_resize", true) {
            let short = width.min(height);
            let (new_height, new_width) = if frame.height() <= frame.width() {
                (short, frame.width() * short / frame.height())
            } else {
                (frame.height() * short / frame.width(), short)
            };
            frame = resize_frame_with_decision(
                &frame,
                ImageSize::new(new_height, new_width)
                    .map_err(|error| invalid(error.to_string()))?,
                ResizeDecision::torchvision(filter),
                ResizeMode::Default,
            )
            .map_err(|error| invalid(error.to_string()))?;
        }
        if enabled("do_thumbnail", true) {
            let mut new_height = height.min(frame.height());
            let mut new_width = width.min(frame.width());
            if new_height != frame.height() || new_width != frame.width() {
                if frame.height() > frame.width() {
                    new_width = frame.width() * new_height / frame.height();
                } else if frame.width() > frame.height() {
                    new_height = frame.height() * new_width / frame.width();
                }
                frame = resize_frame_with_decision(
                    &frame,
                    ImageSize::new(new_height, new_width)
                        .map_err(|error| invalid(error.to_string()))?,
                    ResizeDecision::torchvision(ResizeFilter::Bicubic),
                    ResizeMode::Default,
                )
                .map_err(|error| invalid(error.to_string()))?;
            }
        }
        if enabled("do_pad", true) {
            let vertical = height
                .checked_sub(frame.height())
                .ok_or_else(|| invalid("image exceeds processor canvas"))?;
            let horizontal = width
                .checked_sub(frame.width())
                .ok_or_else(|| invalid("image exceeds processor canvas"))?;
            frame = pad_frame(
                &frame,
                Padding::new(
                    vertical / 2,
                    horizontal - horizontal / 2,
                    vertical - vertical / 2,
                    horizontal / 2,
                ),
                &[0],
            )
            .map_err(|error| invalid(error.to_string()))?;
        }
        if frame.height() != target.height || frame.width() != target.width {
            return Err(invalid(
                "processed image dimensions must match the compiled vision input",
            ));
        }
        prepared.push(frame);
    }
    let tensor = processor
        .preprocess_images(&prepared)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(tensor.data().to_vec::<f32>())
}

fn crop_margin(frame: &ImageFrame) -> Result<ImageFrame> {
    // torchvision's RGB-to-gray on byte tensors truncates back to uint8 before
    // source min/max normalization and its strict threshold comparison.
    let gray = frame
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|pixel| {
            (0.2989 * f32::from(pixel[0])
                + 0.587 * f32::from(pixel[1])
                + 0.114 * f32::from(pixel[2])) as u8
        })
        .collect::<Vec<_>>();
    let minimum = gray
        .iter()
        .copied()
        .min()
        .ok_or_else(|| invalid("empty image"))?;
    let maximum = gray
        .iter()
        .copied()
        .max()
        .ok_or_else(|| invalid("empty image"))?;
    if minimum == maximum {
        return Ok(frame.clone());
    }
    let mut left = frame.width();
    let mut top = frame.height();
    let mut right = 0;
    let mut bottom = 0;
    for (index, value) in gray.into_iter().enumerate() {
        if f32::from(value - minimum) / f32::from(maximum - minimum) * 255.0 < 200.0 {
            let column = index % frame.width();
            let row = index / frame.width();
            left = left.min(column);
            right = right.max(column + 1);
            top = top.min(row);
            bottom = bottom.max(row + 1);
        }
    }
    let bounds = ImageCropBox::new(left, top, right, bottom);
    crop_frame(frame, bounds).map_err(|error| invalid(error.to_string()))
}
