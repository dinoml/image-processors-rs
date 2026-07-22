//! Decoded image-frame resize, crop, pad, and overlay operations.

use super::*;

/// Resizes a frame to target dimensions.
///
/// # Errors
///
/// Returns an error when dimensions are too large, channel data is invalid,
/// or the resized frame cannot satisfy frame invariants.
pub fn resize_frame(
    frame: &ImageFrame,
    target: ImageSize,
    filter: ResizeFilter,
    mode: ResizeMode,
) -> Result<ImageFrame, TransformError> {
    resize_frame_with_decision(frame, target, filter.decision(), mode)
}

/// Resizes a frame with an explicit resize implementation decision.
///
/// # Errors
///
/// Returns an error when dimensions are too large, channel data is invalid,
/// or the resized frame cannot satisfy frame invariants.
pub fn resize_frame_with_decision(
    frame: &ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
    mode: ResizeMode,
) -> Result<ImageFrame, TransformError> {
    let source = frame_size(frame);
    let data = match mode {
        ResizeMode::Default => {
            resize_image_data(frame.data(), source, frame.channels(), target, decision)?
        }
        ResizeMode::Fill => resize_and_fill(frame, target, decision)?,
        ResizeMode::Crop => resize_and_crop(frame, target, decision)?,
    };

    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        data,
        frame.timing(),
    )
}

pub(crate) fn resize_frame_to_f32_torchvision(
    frame: &ImageFrame,
    target: ImageSize,
    filter: ResizeFilter,
    scale: f32,
) -> Result<Vec<f32>, TransformError> {
    image_resize_kernels::resize_u8_to_f32_torchvision(
        frame.data(),
        kernel_size(frame_size(frame)),
        frame.channels(),
        kernel_size(target),
        kernel_filter(filter),
        scale,
    )
    .map_err(transform_error_from_resize)
}

pub(crate) fn resize_frame_owned_with_decision(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
    mode: ResizeMode,
) -> Result<ImageFrame, TransformError> {
    match mode {
        ResizeMode::Default => resize_default_owned(frame, target, decision),
        ResizeMode::Fill => resize_and_fill_owned(frame, target, decision),
        ResizeMode::Crop => resize_and_crop_owned(frame, target, decision),
    }
}

pub(crate) fn resize_frame_owned_with_decision_workspace(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
    mode: ResizeMode,
    workspace: &mut KernelResizeWorkspace,
) -> Result<ImageFrame, TransformError> {
    match mode {
        ResizeMode::Crop => resize_and_crop_owned_workspace(frame, target, decision, workspace),
        ResizeMode::Default => resize_default_owned(frame, target, decision),
        ResizeMode::Fill => resize_and_fill_owned(frame, target, decision),
    }
}

/// Converts a frame to another pixel format.
///
/// # Errors
///
/// Returns an error when channel data is invalid or the new frame is invalid.
pub fn convert_frame_pixel_format(
    frame: &ImageFrame,
    pixel_format: PixelFormat,
) -> Result<ImageFrame, TransformError> {
    if frame.pixel_format() == pixel_format {
        return Ok(frame.clone());
    }

    let data = match pixel_format {
        PixelFormat::Luma8 => convert_to_grayscale(frame.data(), frame.channels())?,
        PixelFormat::Rgb8 => convert_to_rgb(frame.data(), frame.channels())?,
        PixelFormat::Rgba8 => convert_to_rgba(frame.data(), frame.channels())?,
    };

    new_frame(
        frame.width(),
        frame.height(),
        pixel_format,
        data,
        frame.timing(),
    )
}

/// Computes a non-zero crop region around foreground pixels.
///
/// # Errors
///
/// Returns an error when target dimensions or frame coordinates exceed supported limits.
pub fn crop_region(
    frame: &ImageFrame,
    target: ImageSize,
    pad: usize,
) -> Result<(usize, usize, usize, usize), TransformError> {
    let width = frame.width();
    let height = frame.height();
    let channels = frame.channels();
    let data = frame.data();
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }

    let is_zero_col = |x: usize| {
        (0..height).all(|y| {
            let idx = (y * width + x) * channels;
            pixel_luma(&data[idx..idx + channels]) == 0
        })
    };
    let is_zero_row = |y: usize| {
        (0..width).all(|x| {
            let idx = (y * width + x) * channels;
            pixel_luma(&data[idx..idx + channels]) == 0
        })
    };

    let mut crop_left = 0usize;
    for x in 0..width {
        if !is_zero_col(x) {
            break;
        }
        crop_left += 1;
    }

    let mut crop_right = 0usize;
    for x in (0..width).rev() {
        if !is_zero_col(x) {
            break;
        }
        crop_right += 1;
    }

    let mut crop_top = 0usize;
    for y in 0..height {
        if !is_zero_row(y) {
            break;
        }
        crop_top += 1;
    }

    let mut crop_bottom = 0usize;
    for y in (0..height).rev() {
        if !is_zero_row(y) {
            break;
        }
        crop_bottom += 1;
    }

    if crop_left == width || crop_top == height {
        return Ok((0, 0, width, height));
    }

    let frame_width = usize_to_isize(width, "frame width")?;
    let frame_height = usize_to_isize(height, "frame height")?;
    let mut x1 = usize_to_isize(crop_left.saturating_sub(pad), "crop left")?;
    let mut y1 = usize_to_isize(crop_top.saturating_sub(pad), "crop top")?;
    let mut x2 = width
        .saturating_sub(crop_right)
        .saturating_add(pad)
        .min(width) as isize;
    let mut y2 = height
        .saturating_sub(crop_bottom)
        .saturating_add(pad)
        .min(height) as isize;

    if x2 <= x1 || y2 <= y1 {
        return Ok((0, 0, width, height));
    }

    let ratio_crop_region = (x2 - x1) as f64 / (y2 - y1) as f64;
    let ratio_processing = target.width as f64 / target.height as f64;

    if ratio_crop_region > ratio_processing {
        let desired_height = (x2 - x1) as f64 / ratio_processing;
        let desired_height_diff = (desired_height - (y2 - y1) as f64) as isize;
        y1 -= desired_height_diff / 2;
        y2 += desired_height_diff - desired_height_diff / 2;
        if y2 >= frame_height {
            let diff = y2 - frame_height;
            y2 -= diff;
            y1 -= diff;
        }
        if y1 < 0 {
            y2 -= y1;
            y1 = 0;
        }
        if y2 >= frame_height {
            y2 = frame_height;
        }
    } else {
        let desired_width = (y2 - y1) as f64 * ratio_processing;
        let desired_width_diff = (desired_width - (x2 - x1) as f64) as isize;
        x1 -= desired_width_diff / 2;
        x2 += desired_width_diff - desired_width_diff / 2;
        if x2 >= frame_width {
            let diff = x2 - frame_width;
            x2 -= diff;
            x1 -= diff;
        }
        if x1 < 0 {
            x2 -= x1;
            x1 = 0;
        }
        if x2 >= frame_width {
            x2 = frame_width;
        }
    }

    Ok((x1 as usize, y1 as usize, x2 as usize, y2 as usize))
}

/// Pads a frame with a constant pixel value.
///
/// The `fill` slice may contain one byte, applied to every channel, or one
/// byte per frame channel.
///
/// # Errors
///
/// Returns an error when output dimensions overflow, fill length is invalid,
/// or the padded frame cannot satisfy frame invariants.
pub fn pad_frame(
    frame: &ImageFrame,
    padding: Padding,
    fill: &[u8],
) -> Result<ImageFrame, TransformError> {
    pad_frame_with_canvas_fill(frame, padding, CanvasFill::Constant(fill.to_vec()))
}

/// Pads a frame using a canvas fill behavior.
///
/// Constant fills may contain one byte, applied to every channel, or one byte
/// per frame channel. Edge and reflection fills preserve the frame's current
/// pixel format.
///
/// # Errors
///
/// Returns an error when output dimensions overflow, constant fill length is
/// invalid, or the padded frame cannot satisfy frame invariants.
pub fn pad_frame_with_canvas_fill(
    frame: &ImageFrame,
    padding: Padding,
    fill: CanvasFill,
) -> Result<ImageFrame, TransformError> {
    if padding.is_empty() {
        return Ok(frame.clone());
    }

    let source = frame_size(frame);
    let target = padded_image_size(source, padding)?;
    let channels = frame.channels();
    let mut output = vec![0; image_len(target, channels)?];
    if let CanvasFill::Constant(fill) = &fill {
        fill_pixels(&mut output, channels, fill)?;
    }
    paste(
        &mut output,
        target,
        channels,
        frame.data(),
        source,
        usize_to_isize(padding.left, "left padding")?,
        usize_to_isize(padding.top, "top padding")?,
    )?;

    let right = padding
        .left
        .checked_add(source.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let bottom = padding
        .top
        .checked_add(source.height)
        .ok_or(TransformError::ImageSizeOverflow)?;
    match fill {
        CanvasFill::Constant(_) => {}
        CanvasFill::ImageEdges => fill_pasted_border(
            &mut output,
            target,
            channels,
            PastedRegion {
                left: padding.left,
                top: padding.top,
                right,
                bottom,
            },
            BorderFillMode::Replicate,
        ),
        CanvasFill::ReflectImage => fill_pasted_border(
            &mut output,
            target,
            channels,
            PastedRegion {
                left: padding.left,
                top: padding.top,
                right,
                bottom,
            },
            BorderFillMode::Reflect,
        ),
    }

    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        output,
        frame.timing(),
    )
}

/// Clones a sequence and repeats the final frame up to a temporal multiple.
///
/// The returned sequence preserves the source loop behavior. Repeated frames
/// keep the final frame's timing metadata.
///
/// # Errors
///
/// Returns an error when the sequence is empty, `multiple` is zero, padding the
/// frame count overflows, or the output sequence violates media invariants.
pub fn repeat_last_image_sequence_to_multiple(
    sequence: &ImageSequence,
    multiple: usize,
) -> Result<ImageSequence, TransformError> {
    ImageSequence::new(
        repeat_last_frame_batch_to_multiple(sequence.frames(), multiple)?,
        sequence.loop_behavior(),
    )
    .map_err(transform_error_from_media)
}

/// Clones a video clip and repeats the final frame up to a temporal multiple.
///
/// The returned clip preserves the source FPS. Repeated frames keep the final
/// frame's timing metadata.
///
/// # Errors
///
/// Returns an error when the clip is empty, `multiple` is zero, padding the
/// frame count overflows, or the output clip violates media invariants.
pub fn repeat_last_video_clip_to_multiple(
    clip: &VideoClip,
    multiple: usize,
) -> Result<VideoClip, TransformError> {
    VideoClip::new(
        repeat_last_frame_batch_to_multiple(clip.frames(), multiple)?,
        clip.fps(),
    )
    .map_err(transform_error_from_media)
}

/// Pads every frame in a decoded image sequence with a canvas fill behavior.
///
/// # Errors
///
/// Returns an error when any frame cannot be padded or the output sequence
/// violates media invariants.
pub fn pad_image_sequence_frames_with_canvas_fill(
    sequence: &ImageSequence,
    padding: Padding,
    fill: CanvasFill,
) -> Result<ImageSequence, TransformError> {
    let frames = sequence
        .frames()
        .iter()
        .map(|frame| pad_frame_with_canvas_fill(frame, padding, fill.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    ImageSequence::new(frames, sequence.loop_behavior()).map_err(transform_error_from_media)
}

/// Pads every frame in a decoded video clip with a canvas fill behavior.
///
/// Edge and reflection fills apply independently to each decoded frame, which
/// lets video wrappers reuse the same border behavior as image recipes.
///
/// # Errors
///
/// Returns an error when any frame cannot be padded or the output clip
/// violates media invariants.
pub fn pad_video_clip_frames_with_canvas_fill(
    clip: &VideoClip,
    padding: Padding,
    fill: CanvasFill,
) -> Result<VideoClip, TransformError> {
    let frames = clip
        .frames()
        .iter()
        .map(|frame| {
            pad_frame_with_canvas_fill(frame.image(), padding, fill.clone()).map(VideoFrame::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    VideoClip::new(frames, clip.fps()).map_err(transform_error_from_media)
}

/// Resizes an image only when it exceeds `target_area`.
///
/// # Errors
///
/// Returns an error when planning fails, resizing fails, or output frame
/// invariants fail.
pub fn resize_frame_to_area_limit(
    image: &ImageFrame,
    target_area: usize,
    filter: ResizeFilter,
) -> Result<ImageFrame, TransformError> {
    let plan = area_resize_plan(frame_size(image), target_area)?;
    if !plan.should_resize() {
        return Ok(image.clone());
    }

    resize_frame(image, plan.resized_size, filter, ResizeMode::Default)
}

/// Builds an in-bounds center-crop box.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or `target_size` exceeds
/// `source_size`.
pub fn center_crop_box(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<ImageCropBox, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    if target_size.height > source_size.height || target_size.width > source_size.width {
        return Err(TransformError::CropTooLarge {
            source_size,
            target_size,
        });
    }

    let x_min = (source_size.width - target_size.width) / 2;
    let y_min = (source_size.height - target_size.height) / 2;
    Ok(ImageCropBox::new(
        x_min,
        y_min,
        x_min + target_size.width,
        y_min + target_size.height,
    ))
}

/// Builds a resize-to-cover and center-crop plan.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn resize_center_crop_plan(
    original_size: ImageSize,
    target_size: ImageSize,
    rounding: ResizeRounding,
) -> Result<ResizeCenterCropPlan, TransformError> {
    validate_size(original_size)?;
    validate_size(target_size)?;
    let resized_size = match rounding {
        ResizeRounding::Floor => cover_size(original_size, target_size)?,
        ResizeRounding::Ceil => cover_size_ceil(original_size, target_size)?,
    };
    let crop_box = center_crop_box(resized_size, target_size)?;

    Ok(ResizeCenterCropPlan {
        original_size,
        target_size,
        resized_size,
        crop_box,
    })
}

/// Resizes a frame to cover `target_size` and center-crops it.
///
/// # Errors
///
/// Returns an error when planning, resizing, cropping, or output frame
/// validation fails.
pub fn resize_center_crop_frame(
    image: &ImageFrame,
    target_size: ImageSize,
    filter: ResizeFilter,
    rounding: ResizeRounding,
) -> Result<ImageFrame, TransformError> {
    resize_center_crop_frame_with_decision(image, target_size, filter.decision(), rounding)
}

/// Resizes and center-crops a frame using an explicit resize decision.
///
/// # Errors
///
/// Returns an error when planning, resizing, cropping, or output frame
/// validation fails.
pub fn resize_center_crop_frame_with_decision(
    image: &ImageFrame,
    target_size: ImageSize,
    decision: ResizeDecision,
    rounding: ResizeRounding,
) -> Result<ImageFrame, TransformError> {
    let plan = resize_center_crop_plan(frame_size(image), target_size, rounding)?;
    let resized =
        resize_frame_with_decision(image, plan.resized_size, decision, ResizeMode::Default)?;
    crop_frame(&resized, plan.crop_box)
}

/// Builds a resize plan that rounds dimensions down to per-axis multiples.
///
/// If `requested_size` is `None`, the original image size is rounded down.
///
/// # Errors
///
/// Returns an error when dimensions or multiples are invalid or rounding
/// produces a zero dimension.
pub fn multiple_of_resize_plan(
    original_size: ImageSize,
    requested_size: Option<ImageSize>,
    multiples: ImageSize,
) -> Result<MultipleOfResizePlan, TransformError> {
    validate_size(original_size)?;
    validate_size(multiples)?;
    let requested_size = requested_size.unwrap_or(original_size);
    validate_size(requested_size)?;
    let resized_size = ImageSize::new(
        requested_size.height - requested_size.height % multiples.height,
        requested_size.width - requested_size.width % multiples.width,
    )?;

    Ok(MultipleOfResizePlan {
        original_size,
        requested_size,
        multiples,
        resized_size,
    })
}

/// Builds a bottom/right padding plan that rounds dimensions up to multiples.
///
/// The original content is anchored at the top-left corner. This makes the
/// inverse operation a deterministic crop to `source_size`, which is useful for
/// replicate-padding workflows that later restore model outputs to the
/// unpadded image extent.
///
/// # Errors
///
/// Returns an error when dimensions or multiples are invalid or rounded
/// dimensions overflow.
pub fn pad_to_multiple_plan(
    source_size: ImageSize,
    multiples: ImageSize,
) -> Result<PadToMultiplePlan, TransformError> {
    validate_size(source_size)?;
    validate_size(multiples)?;

    let padded_size = ImageSize::new(
        round_up_to_multiple_transform(source_size.height, multiples.height)?,
        round_up_to_multiple_transform(source_size.width, multiples.width)?,
    )?;
    let padding = Padding::new(
        0,
        padded_size.width - source_size.width,
        padded_size.height - source_size.height,
        0,
    );

    Ok(PadToMultiplePlan {
        source_size,
        multiples,
        padded_size,
        padding,
    })
}

/// Pads a frame to per-axis multiples using a canvas fill behavior.
///
/// `CanvasFill::ImageEdges` provides replicate padding. The original image is
/// kept at the top-left of the padded frame.
///
/// # Errors
///
/// Returns an error when planning fails, fill validation fails, or output frame
/// invariants fail.
pub fn pad_frame_to_multiple_with_canvas_fill(
    frame: &ImageFrame,
    multiples: ImageSize,
    fill: CanvasFill,
) -> Result<ImageFrame, TransformError> {
    let plan = pad_to_multiple_plan(frame_size(frame), multiples)?;
    pad_frame_with_canvas_fill(frame, plan.padding, fill)
}

/// Pads the bottom and right edges to the next per-axis multiples using
/// edge-inclusive symmetric reflection.
///
/// Each dimension always advances to the *next* multiple, including when the
/// source dimension is already divisible by its requested multiple. Reflected
/// samples repeat the edge pixel (`... b, a, a, b ...`), matching symmetric
/// padding rather than edge-exclusive reflection.
///
/// # Errors
///
/// Returns an error when a multiple is zero, output dimensions overflow, or
/// the padded frame cannot satisfy frame invariants.
pub fn pad_frame_symmetric_to_next_multiple(
    frame: &ImageFrame,
    multiples: ImageSize,
) -> Result<ImageFrame, TransformError> {
    validate_size(multiples)?;
    let source = frame_size(frame);
    let target = ImageSize::new(
        next_multiple(source.height, multiples.height)?,
        next_multiple(source.width, multiples.width)?,
    )?;
    let channels = frame.channels();
    let mut output = Vec::with_capacity(image_len(target, channels)?);
    for y in 0..target.height {
        let source_y = symmetric_source_index(y, source.height)?;
        for x in 0..target.width {
            let source_x = symmetric_source_index(x, source.width)?;
            let offset = (source_y * source.width + source_x) * channels;
            output.extend_from_slice(&frame.data()[offset..offset + channels]);
        }
    }
    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        output,
        frame.timing(),
    )
}

fn next_multiple(value: usize, multiple: usize) -> Result<usize, TransformError> {
    value
        .checked_div(multiple)
        .and_then(|units| units.checked_add(1))
        .and_then(|units| units.checked_mul(multiple))
        .ok_or(TransformError::ImageSizeOverflow)
}

fn symmetric_source_index(index: usize, length: usize) -> Result<usize, TransformError> {
    let period = length
        .checked_mul(2)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let reflected = index % period;
    if reflected < length {
        Ok(reflected)
    } else {
        Ok(period - reflected - 1)
    }
}

/// Crops a top-left anchored padded frame back to `source_size`.
///
/// This is the inverse of [`pad_to_multiple_plan`] for bottom/right padding.
///
/// # Errors
///
/// Returns an error when `source_size` is invalid, exceeds the frame extent, or
/// output frame invariants fail.
pub fn unpad_frame_to_size(
    frame: &ImageFrame,
    source_size: ImageSize,
) -> Result<ImageFrame, TransformError> {
    validate_size(source_size)?;
    crop_frame(
        frame,
        ImageCropBox::new(0, 0, source_size.width, source_size.height),
    )
}

/// Builds a temporal repeat-last padding plan.
///
/// `frame_count` is rounded up to `multiple` by appending references to the
/// last frame. This mirrors temporal patch grouping used by VLM/video
/// processors without requiring processor-specific indexing code.
///
/// # Errors
///
/// Returns an error when `frame_count` is zero, `multiple` is zero, or padding
/// the frame count overflows `usize`.
pub fn temporal_repeat_last_plan(
    frame_count: usize,
    multiple: usize,
) -> Result<TemporalRepeatLastPlan, TransformError> {
    if frame_count == 0 {
        return Err(TransformError::EmptyFrameBatch);
    }
    if multiple == 0 {
        return Err(TransformError::InvalidTemporalMultiple(multiple));
    }

    let remainder = frame_count % multiple;
    let padding_frames = if remainder == 0 {
        0
    } else {
        multiple - remainder
    };
    let target_frame_count = frame_count
        .checked_add(padding_frames)
        .ok_or(TransformError::ImageSizeOverflow)?;

    Ok(TemporalRepeatLastPlan {
        source_frame_count: frame_count,
        multiple,
        target_frame_count,
        padding_frames,
    })
}

/// Clones a frame batch and repeats the final item up to a temporal multiple.
///
/// # Errors
///
/// Returns an error when `frames` is empty, `multiple` is zero, or padding the
/// frame count overflows `usize`.
pub fn repeat_last_frame_batch_to_multiple<T: Clone>(
    frames: &[T],
    multiple: usize,
) -> Result<Vec<T>, TransformError> {
    let plan = temporal_repeat_last_plan(frames.len(), multiple)?;
    let mut output = Vec::with_capacity(plan.target_frame_count);
    for source_index in plan.source_indices() {
        output.push(frames[source_index].clone());
    }
    Ok(output)
}

/// Builds a resize-and-fill plan.
///
/// The source is resized to fit inside `target_size` while preserving aspect
/// ratio, then centered with padding around the resized image.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn resize_fill_plan(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<ResizeFillPlan, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    let resized_size = contain_size(source_size, target_size)?;
    let padding = centered_padding(resized_size, target_size)?;

    Ok(ResizeFillPlan {
        source_size,
        target_size,
        resized_size,
        padding,
    })
}

/// Builds a fit-inside resize plan with only bottom/right padding.
///
/// The source is resized only when it does not fit inside `target_size`. When
/// both source axes are already within the target, the source dimensions are
/// preserved and only bottom/right padding is requested.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn bottom_right_resize_pad_plan(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<BottomRightResizePadPlan, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    let resized_size = if source_size.height <= target_size.height
        && source_size.width <= target_size.width
    {
        source_size
    } else {
        let height_limited = (target_size.height as u128)
            .checked_mul(source_size.width as u128)
            .ok_or(TransformError::ImageSizeOverflow)?
            <= (target_size.width as u128)
                .checked_mul(source_size.height as u128)
                .ok_or(TransformError::ImageSizeOverflow)?;

        let (height, width) = if height_limited {
            (
                target_size.height,
                round_ratio_dimension(source_size.width, target_size.height, source_size.height)?,
            )
        } else {
            (
                round_ratio_dimension(source_size.height, target_size.width, source_size.width)?,
                target_size.width,
            )
        };
        ImageSize::new(height.max(1), width.max(1))?
    };

    if resized_size.height > target_size.height || resized_size.width > target_size.width {
        return Err(TransformError::CropTooLarge {
            source_size: resized_size,
            target_size,
        });
    }

    let padding = Padding::new(
        0,
        target_size.width - resized_size.width,
        target_size.height - resized_size.height,
        0,
    );

    Ok(BottomRightResizePadPlan {
        source_size,
        target_size,
        resized_size,
        padding,
    })
}

/// Resizes an image to fit on an RGB canvas.
///
/// # Errors
///
/// Returns an error when planning, RGB conversion, resizing, filling, pasting,
/// or output frame validation fails.
pub fn resize_rgb_to_fill_frame(
    image: &ImageFrame,
    target_size: ImageSize,
    fill: RgbCanvasFill,
    filter: ResizeFilter,
) -> Result<ImageFrame, TransformError> {
    let rgb = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
    resize_frame_to_fill_frame(&rgb, target_size, CanvasFill::from(fill), filter)
}

/// Resizes an image to fit on a canvas using its current pixel format.
///
/// Constant fills may contain one byte, applied to every channel, or one byte
/// per frame channel. Edge and reflection fills preserve the resized image
/// pixels and do not convert pixel formats.
///
/// # Errors
///
/// Returns an error when planning, resizing, filling, pasting, or output frame
/// validation fails.
pub fn resize_frame_to_fill_frame(
    image: &ImageFrame,
    target_size: ImageSize,
    fill: CanvasFill,
    filter: ResizeFilter,
) -> Result<ImageFrame, TransformError> {
    if let CanvasFill::Constant(bytes) = &fill {
        validate_padding_fill(bytes, image.pixel_format())?;
    }
    match fill {
        CanvasFill::ImageEdges => resize_frame(image, target_size, filter, ResizeMode::Fill),
        fill => {
            let plan = resize_fill_plan(frame_size(image), target_size)?;
            let resized = resize_frame(image, plan.resized_size, filter, ResizeMode::Default)?;
            pad_frame_with_canvas_fill(&resized, plan.padding, fill)
        }
    }
}

/// Resizes an image to fit inside a target, then pads the bottom/right edges.
///
/// Constant fills may contain one byte, applied to every channel, or one byte
/// per frame channel. Edge and reflection fills preserve the resized image
/// pixels and do not convert pixel formats.
///
/// # Errors
///
/// Returns an error when planning, resizing, padding, or output frame
/// validation fails.
pub fn resize_frame_to_bottom_right_padded_frame(
    image: &ImageFrame,
    target_size: ImageSize,
    fill: CanvasFill,
    decision: ResizeDecision,
) -> Result<ImageFrame, TransformError> {
    let plan = bottom_right_resize_pad_plan(frame_size(image), target_size)?;
    let resized = if plan.resized_size == plan.source_size {
        image.clone()
    } else {
        resize_frame_with_decision(image, plan.resized_size, decision, ResizeMode::Default)?
    };
    pad_frame_with_canvas_fill(&resized, plan.padding, fill)
}

/// Concatenates frames horizontally on an RGB canvas.
///
/// # Errors
///
/// Returns an error when no images are provided, dimensions overflow, RGB
/// conversion fails, coordinates exceed supported limits, or output frame
/// invariants fail.
pub fn concatenate_frames_horizontally_rgb(
    images: &[ImageFrame],
    fill: [u8; 3],
) -> Result<ImageFrame, TransformError> {
    if images.is_empty() {
        return Err(TransformError::EmptyImageBatch);
    }

    let mut total_width = 0usize;
    let mut max_height = 0usize;
    for image in images {
        total_width = total_width
            .checked_add(image.width())
            .ok_or(TransformError::ImageSizeOverflow)?;
        max_height = max_height.max(image.height());
    }
    let target_size = ImageSize::new(max_height, total_width)?;
    let channels = PixelFormat::Rgb8.channels();
    let mut output = vec![0; image_len(target_size, channels)?];
    fill_pixels(&mut output, channels, &fill)?;

    let mut x_offset = 0usize;
    for image in images {
        let rgb = convert_frame_pixel_format(image, PixelFormat::Rgb8)?;
        let y_offset = (max_height - rgb.height()) / 2;
        paste(
            &mut output,
            target_size,
            channels,
            rgb.data(),
            frame_size(&rgb),
            usize_to_isize(x_offset, "x offset")?,
            usize_to_isize(y_offset, "y offset")?,
        )?;
        x_offset = x_offset
            .checked_add(rgb.width())
            .ok_or(TransformError::ImageSizeOverflow)?;
    }

    new_frame(
        total_width,
        max_height,
        PixelFormat::Rgb8,
        output,
        images[0].timing(),
    )
}

/// Crops a frame around its center.
///
/// When the source dimension exceeds the target by an odd number of pixels,
/// the crop keeps the extra pixel on the bottom or right side.
///
/// # Errors
///
/// Returns an error when `target` is larger than the source frame or output
/// frame invariants fail.
pub fn center_crop_frame(
    frame: &ImageFrame,
    target: ImageSize,
) -> Result<ImageFrame, TransformError> {
    let source = frame_size(frame);
    if target.height > source.height || target.width > source.width {
        return Err(TransformError::CropTooLarge {
            source_size: source,
            target_size: target,
        });
    }

    if target == source {
        return Ok(frame.clone());
    }

    let origin_x = (source.width - target.width) / 2;
    let origin_y = (source.height - target.height) / 2;
    let data = crop_image_data(
        frame.data(),
        source,
        frame.channels(),
        origin_x,
        origin_y,
        target,
    )?;
    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        data,
        frame.timing(),
    )
}

/// Crops a frame to an absolute pixel crop box.
///
/// # Errors
///
/// Returns an error when the crop box is empty or outside the source frame, or
/// output frame invariants fail.
pub fn crop_frame(
    frame: &ImageFrame,
    crop_box: ImageCropBox,
) -> Result<ImageFrame, TransformError> {
    let source = frame_size(frame);
    validate_image_crop_box(crop_box, source)?;

    if crop_box.x_min == 0
        && crop_box.y_min == 0
        && crop_box.x_max == source.width
        && crop_box.y_max == source.height
    {
        return Ok(frame.clone());
    }

    let target = image_crop_box_size(crop_box)?;
    let data = crop_image_data(
        frame.data(),
        source,
        frame.channels(),
        crop_box.x_min,
        crop_box.y_min,
        target,
    )?;
    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        data,
        frame.timing(),
    )
}

/// Overlays a foreground frame onto a background frame.
///
/// The output keeps the background size, pixel format, and timing metadata.
/// Foreground pixels outside the background bounds are clipped. `Rgba8`
/// foreground pixels are alpha-blended over the background; other formats copy
/// foreground pixels into the covered region.
///
/// # Errors
///
/// Returns an error when foreground pixel format differs from the background,
/// frame dimensions exceed supported coordinate limits, or output frame
/// invariants fail.
pub fn overlay_frame(
    background: &ImageFrame,
    foreground: &ImageFrame,
    position: OverlayPosition,
) -> Result<ImageFrame, TransformError> {
    validate_pixel_format(
        "foreground",
        background.pixel_format(),
        foreground.pixel_format(),
    )?;

    let background_size = frame_size(background);
    let foreground_size = frame_size(foreground);
    ensure_isize_size(background_size)?;
    ensure_isize_size(foreground_size)?;

    let mut output = background.data().to_vec();
    overlay_pixels(
        &mut output,
        background_size,
        background.pixel_format(),
        foreground.data(),
        foreground_size,
        position,
    )?;
    new_frame(
        background.width(),
        background.height(),
        background.pixel_format(),
        output,
        background.timing(),
    )
}

/// Composites foreground over background using mask luminance as opacity.
///
/// The background and foreground must have identical dimensions and pixel
/// format. The mask must have the same dimensions; its luminance selects the
/// foreground contribution, where `0` keeps the background and `255` selects
/// the foreground.
///
/// # Errors
///
/// Returns an error when foreground size or format differs from the background,
/// mask size differs from the background, buffer sizes overflow, or output
/// frame invariants fail.
pub fn composite_mask_frame(
    background: &ImageFrame,
    foreground: &ImageFrame,
    mask: &ImageFrame,
) -> Result<ImageFrame, TransformError> {
    let size = frame_size(background);
    validate_frame_size("foreground", size, frame_size(foreground))?;
    validate_frame_size("mask", size, frame_size(mask))?;
    validate_pixel_format(
        "foreground",
        background.pixel_format(),
        foreground.pixel_format(),
    )?;

    let channels = background.channels();
    let mask_channels = mask.channels();
    let pixels = checked_pixels(size)?;
    let mut output = Vec::with_capacity(image_len(size, channels)?);

    for pixel in 0..pixels {
        let background_idx = pixel * channels;
        let mask_idx = pixel * mask_channels;
        let opacity = pixel_luma(&mask.data()[mask_idx..mask_idx + mask_channels]);
        for channel in 0..channels {
            output.push(blend_u8(
                background.data()[background_idx + channel],
                foreground.data()[background_idx + channel],
                opacity,
            ));
        }
    }

    new_frame(
        background.width(),
        background.height(),
        background.pixel_format(),
        output,
        background.timing(),
    )
}

fn resize_image_data(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    image_resize_kernels::resize_u8(
        values,
        kernel_size(source),
        channels,
        kernel_size(target),
        kernel_filter(decision.filter()),
        resize_profile(decision.parity()),
    )
    .map_err(transform_error_from_resize)
}

fn resize_image_data_owned(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    image_resize_kernels::resize_u8_owned(
        values,
        kernel_size(source),
        channels,
        kernel_size(target),
        kernel_filter(decision.filter()),
        resize_profile(decision.parity()),
    )
    .map_err(transform_error_from_resize)
}

fn resize_image_data_crop(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    crop: KernelResizeCrop,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    image_resize_kernels::resize_u8_crop(
        values,
        kernel_size(source),
        channels,
        crop,
        kernel_filter(decision.filter()),
        resize_profile(decision.parity()),
    )
    .map_err(transform_error_from_resize)
}

fn resize_image_data_crop_owned(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    crop: KernelResizeCrop,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    image_resize_kernels::resize_u8_crop_owned(
        values,
        kernel_size(source),
        channels,
        crop,
        kernel_filter(decision.filter()),
        resize_profile(decision.parity()),
    )
    .map_err(transform_error_from_resize)
}

fn resize_image_data_crop_owned_workspace(
    values: Vec<u8>,
    source: ImageSize,
    channels: usize,
    crop: KernelResizeCrop,
    decision: ResizeDecision,
    workspace: &mut KernelResizeWorkspace,
) -> Result<Vec<u8>, TransformError> {
    image_resize_kernels::resize_u8_crop_owned_with_workspace(
        values,
        kernel_size(source),
        channels,
        crop,
        kernel_filter(decision.filter()),
        resize_profile(decision.parity()),
        workspace,
    )
    .map_err(transform_error_from_resize)
}

fn kernel_crop(
    resized: ImageSize,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
) -> Result<KernelResizeCrop, TransformError> {
    KernelResizeCrop::new(
        kernel_size(resized),
        origin_x,
        origin_y,
        kernel_size(target),
    )
    .map_err(transform_error_from_resize)
}

fn kernel_size(size: ImageSize) -> KernelImageSize {
    KernelImageSize {
        height: size.height,
        width: size.width,
    }
}

fn kernel_filter(filter: ResizeFilter) -> KernelResizeFilter {
    match filter {
        ResizeFilter::Nearest => KernelResizeFilter::Nearest,
        ResizeFilter::Bilinear => KernelResizeFilter::Bilinear,
        ResizeFilter::Bicubic => KernelResizeFilter::Bicubic,
        ResizeFilter::Lanczos => KernelResizeFilter::Lanczos,
    }
}

fn resize_profile(parity: ResizeParity) -> ResizeProfile {
    match parity {
        ResizeParity::Compatibility => ResizeProfile::Pillow,
        ResizeParity::Torchvision => ResizeProfile::Torchvision,
        ResizeParity::Resampling | ResizeParity::PixelExact => ResizeProfile::Fast,
    }
}

fn transform_error_from_resize(error: KernelResizeError) -> TransformError {
    match error {
        KernelResizeError::InvalidSize { height, width } => {
            TransformError::InvalidSize { height, width }
        }
        KernelResizeError::InvalidBufferLength { expected, actual } => {
            TransformError::InvalidBufferLength { expected, actual }
        }
        KernelResizeError::ImageSizeOverflow => TransformError::ImageSizeOverflow,
        KernelResizeError::DimensionTooLarge {
            dimension,
            value,
            max,
        } => TransformError::DimensionTooLarge {
            dimension,
            value,
            max,
        },
        KernelResizeError::UnsupportedChannels(channels) => {
            TransformError::UnsupportedChannels(channels)
        }
        KernelResizeError::CropWindowOutOfBounds {
            resized_height,
            resized_width,
            target_height,
            target_width,
            ..
        } => TransformError::CropTooLarge {
            source_size: ImageSize {
                height: resized_height,
                width: resized_width,
            },
            target_size: ImageSize {
                height: target_height,
                width: target_width,
            },
        },
        other => TransformError::ResizeKernel {
            message: other.to_string(),
        },
    }
}

fn resize_default_owned(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<ImageFrame, TransformError> {
    let (width, height, pixel_format, data, timing) = frame.into_parts();
    let source = ImageSize { height, width };
    let resized = resize_image_data_owned(data, source, pixel_format.channels(), target, decision)?;
    new_frame(target.width, target.height, pixel_format, resized, timing)
}

fn resize_and_fill(
    frame: &ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    let source = frame_size(frame);
    let plan = resize_fill_plan(source, target)?;
    let resized = resize_image_data(
        frame.data(),
        source,
        frame.channels(),
        plan.resized_size,
        decision,
    )?;
    let mut output = vec![0; image_len(target, frame.channels())?];
    let offset_x = usize_to_isize(plan.padding.left, "padding left")?;
    let offset_y = usize_to_isize(plan.padding.top, "padding top")?;
    paste(
        &mut output,
        target,
        frame.channels(),
        &resized,
        plan.resized_size,
        offset_x,
        offset_y,
    )?;
    let right = plan
        .padding
        .left
        .checked_add(plan.resized_size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let bottom = plan
        .padding
        .top
        .checked_add(plan.resized_size.height)
        .ok_or(TransformError::ImageSizeOverflow)?;
    fill_pasted_edges(
        &mut output,
        target,
        frame.channels(),
        plan.padding.left,
        plan.padding.top,
        right,
        bottom,
    );
    Ok(output)
}

fn resize_and_fill_owned(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<ImageFrame, TransformError> {
    let (width, height, pixel_format, data, timing) = frame.into_parts();
    let source = ImageSize { height, width };
    let channels = pixel_format.channels();
    let plan = resize_fill_plan(source, target)?;
    let resized = resize_image_data_owned(data, source, channels, plan.resized_size, decision)?;
    let mut output = vec![0; image_len(target, channels)?];
    let offset_x = usize_to_isize(plan.padding.left, "padding left")?;
    let offset_y = usize_to_isize(plan.padding.top, "padding top")?;
    paste(
        &mut output,
        target,
        channels,
        &resized,
        plan.resized_size,
        offset_x,
        offset_y,
    )?;
    let right = plan
        .padding
        .left
        .checked_add(plan.resized_size.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let bottom = plan
        .padding
        .top
        .checked_add(plan.resized_size.height)
        .ok_or(TransformError::ImageSizeOverflow)?;
    fill_pasted_edges(
        &mut output,
        target,
        channels,
        plan.padding.left,
        plan.padding.top,
        right,
        bottom,
    );
    new_frame(target.width, target.height, pixel_format, output, timing)
}

/// Computes the output size produced by applying padding to an image size.
///
/// # Errors
///
/// Returns an error when the padded dimensions overflow `usize`.
pub fn padded_image_size(source: ImageSize, padding: Padding) -> Result<ImageSize, TransformError> {
    let height = source
        .height
        .checked_add(padding.top)
        .and_then(|height| height.checked_add(padding.bottom))
        .ok_or(TransformError::ImageSizeOverflow)?;
    let width = source
        .width
        .checked_add(padding.left)
        .and_then(|width| width.checked_add(padding.right))
        .ok_or(TransformError::ImageSizeOverflow)?;
    ImageSize::new(height, width)
}

/// Validates a constant padding fill for a pixel format.
///
/// # Errors
///
/// Returns an error when the fill is not scalar and does not contain exactly
/// one byte per channel for `pixel_format`.
pub fn validate_padding_fill(fill: &[u8], pixel_format: PixelFormat) -> Result<(), TransformError> {
    validate_pixel_fill(fill, pixel_format.channels())
}

fn fill_pixels(values: &mut [u8], channels: usize, fill: &[u8]) -> Result<(), TransformError> {
    validate_pixel_fill(fill, channels)?;
    if fill.len() == 1 {
        values.fill(fill[0]);
        return Ok(());
    }

    for pixel in values.chunks_exact_mut(channels) {
        pixel.copy_from_slice(fill);
    }
    Ok(())
}

pub(super) fn crop_image_data(
    values: &[u8],
    source: ImageSize,
    channels: usize,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
) -> Result<Vec<u8>, TransformError> {
    validate_len(values.len(), image_len(source, channels)?)?;
    let row_len = target
        .width
        .checked_mul(channels)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut output = Vec::with_capacity(image_len(target, channels)?);
    for y in 0..target.height {
        let source_start = ((origin_y + y) * source.width + origin_x)
            .checked_mul(channels)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let source_end = source_start
            .checked_add(row_len)
            .ok_or(TransformError::ImageSizeOverflow)?;
        output.extend_from_slice(&values[source_start..source_end]);
    }
    Ok(output)
}

fn resize_and_crop_owned(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<ImageFrame, TransformError> {
    let (width, height, pixel_format, data, timing) = frame.into_parts();
    let source = ImageSize { height, width };
    let channels = pixel_format.channels();
    let resized_size = cover_size(source, target)?;
    let origin_x = (resized_size.width - target.width) / 2;
    let origin_y = (resized_size.height - target.height) / 2;
    let crop = kernel_crop(resized_size, origin_x, origin_y, target)?;
    let cropped = resize_image_data_crop_owned(data, source, channels, crop, decision)?;
    new_frame(target.width, target.height, pixel_format, cropped, timing)
}

fn resize_and_crop_owned_workspace(
    frame: ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
    workspace: &mut KernelResizeWorkspace,
) -> Result<ImageFrame, TransformError> {
    let (width, height, pixel_format, data, timing) = frame.into_parts();
    let source = ImageSize { height, width };
    let channels = pixel_format.channels();
    let resized_size = cover_size(source, target)?;
    let origin_x = (resized_size.width - target.width) / 2;
    let origin_y = (resized_size.height - target.height) / 2;
    let crop = kernel_crop(resized_size, origin_x, origin_y, target)?;
    let cropped =
        resize_image_data_crop_owned_workspace(data, source, channels, crop, decision, workspace)?;
    new_frame(target.width, target.height, pixel_format, cropped, timing)
}

fn resize_and_crop(
    frame: &ImageFrame,
    target: ImageSize,
    decision: ResizeDecision,
) -> Result<Vec<u8>, TransformError> {
    let source = frame_size(frame);
    let resized_size = cover_size(source, target)?;
    let origin_x = (resized_size.width - target.width) / 2;
    let origin_y = (resized_size.height - target.height) / 2;
    let crop = kernel_crop(resized_size, origin_x, origin_y, target)?;
    resize_image_data_crop(frame.data(), source, frame.channels(), crop, decision)
}

fn cover_size(source: ImageSize, target: ImageSize) -> Result<ImageSize, TransformError> {
    let ratio = target.width as f64 / target.height as f64;
    let source_ratio = source.width as f64 / source.height as f64;
    let width = if ratio > source_ratio {
        target.width
    } else {
        checked_ratio_dimension(source.width, target.height, source.height)?
    }
    .max(target.width);
    let height = if ratio <= source_ratio {
        target.height
    } else {
        checked_ratio_dimension(source.height, target.width, source.width)?
    }
    .max(target.height);
    Ok(ImageSize { height, width })
}

fn cover_size_ceil(source: ImageSize, target: ImageSize) -> Result<ImageSize, TransformError> {
    let scale_by_height = (target.height as f64 / source.height as f64)
        >= (target.width as f64 / source.width as f64);
    let size = if scale_by_height {
        ImageSize {
            height: target.height,
            width: ceil_ratio_dimension(source.width, target.height, source.height)?
                .max(target.width),
        }
    } else {
        ImageSize {
            height: ceil_ratio_dimension(source.height, target.width, source.width)?
                .max(target.height),
            width: target.width,
        }
    };
    validate_size(size)?;
    Ok(size)
}

fn contain_size(source: ImageSize, target: ImageSize) -> Result<ImageSize, TransformError> {
    let ratio = target.width as f64 / target.height as f64;
    let source_ratio = source.width as f64 / source.height as f64;
    let width = if ratio < source_ratio {
        target.width
    } else {
        checked_ratio_dimension(source.width, target.height, source.height)?
    }
    .max(1);
    let height = if ratio >= source_ratio {
        target.height
    } else {
        checked_ratio_dimension(source.height, target.width, source.width)?
    }
    .max(1);
    Ok(ImageSize { height, width })
}

fn paste(
    dst: &mut [u8],
    dst_size: ImageSize,
    channels: usize,
    src: &[u8],
    src_size: ImageSize,
    offset_x: isize,
    offset_y: isize,
) -> Result<(), TransformError> {
    ensure_isize_size(dst_size)?;
    ensure_isize_size(src_size)?;
    for sy in 0..src_size.height {
        let dy = sy as isize + offset_y;
        if dy < 0 || dy >= dst_size.height as isize {
            continue;
        }
        for sx in 0..src_size.width {
            let dx = sx as isize + offset_x;
            if dx < 0 || dx >= dst_size.width as isize {
                continue;
            }
            for c in 0..channels {
                let src_idx = (sy * src_size.width + sx) * channels + c;
                let dst_idx = (dy as usize * dst_size.width + dx as usize) * channels + c;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    Ok(())
}

fn overlay_pixels(
    dst: &mut [u8],
    dst_size: ImageSize,
    pixel_format: PixelFormat,
    src: &[u8],
    src_size: ImageSize,
    position: OverlayPosition,
) -> Result<(), TransformError> {
    let channels = pixel_format.channels();
    let dst_height = usize_to_isize(dst_size.height, "background height")?;
    let dst_width = usize_to_isize(dst_size.width, "background width")?;

    for sy in 0..src_size.height {
        let Some(dy) = offset_coordinate(sy, position.y, dst_height)? else {
            continue;
        };
        for sx in 0..src_size.width {
            let Some(dx) = offset_coordinate(sx, position.x, dst_width)? else {
                continue;
            };
            let src_idx = (sy * src_size.width + sx) * channels;
            let dst_idx = (dy * dst_size.width + dx) * channels;
            blend_overlay_pixel(
                &mut dst[dst_idx..dst_idx + channels],
                &src[src_idx..src_idx + channels],
                pixel_format,
            );
        }
    }
    Ok(())
}

fn blend_overlay_pixel(dst: &mut [u8], src: &[u8], pixel_format: PixelFormat) {
    match pixel_format {
        PixelFormat::Rgba8 => {
            let alpha = src[3];
            for channel in 0..3 {
                dst[channel] = blend_u8(dst[channel], src[channel], alpha);
            }
            dst[3] = source_over_alpha(dst[3], alpha);
        }
        PixelFormat::Luma8 | PixelFormat::Rgb8 => dst.copy_from_slice(src),
    }
}

fn blend_u8(background: u8, foreground: u8, opacity: u8) -> u8 {
    let inverse = u16::from(u8::MAX - opacity);
    let opacity = u16::from(opacity);
    ((u16::from(foreground) * opacity + u16::from(background) * inverse + 127) / 255) as u8
}

fn source_over_alpha(background: u8, foreground: u8) -> u8 {
    foreground.saturating_add(
        ((u16::from(background) * u16::from(u8::MAX - foreground) + 127) / 255) as u8,
    )
}

fn offset_coordinate(
    coordinate: usize,
    offset: isize,
    limit: isize,
) -> Result<Option<usize>, TransformError> {
    let coordinate = usize_to_isize(coordinate, "overlay coordinate")?;
    let Some(value) = coordinate.checked_add(offset) else {
        return Ok(None);
    };
    if value < 0 || value >= limit {
        Ok(None)
    } else {
        Ok(Some(value as usize))
    }
}

fn fill_pasted_edges(
    values: &mut [u8],
    size: ImageSize,
    channels: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
) {
    if left >= right || top >= bottom {
        return;
    }

    fill_pasted_border(
        values,
        size,
        channels,
        PastedRegion {
            left,
            top,
            right,
            bottom,
        },
        BorderFillMode::Replicate,
    );
}

fn fill_pasted_border(
    values: &mut [u8],
    size: ImageSize,
    channels: usize,
    region: PastedRegion,
    mode: BorderFillMode,
) {
    if region.left >= region.right || region.top >= region.bottom {
        return;
    }

    for y in 0..size.height {
        for x in 0..size.width {
            if y >= region.top && y < region.bottom && x >= region.left && x < region.right {
                continue;
            }
            let src_y = border_source_coordinate(y, region.top, region.bottom, mode);
            let src_x = border_source_coordinate(x, region.left, region.right, mode);
            copy_pixel(values, size.width, channels, src_y, src_x, y, x);
        }
    }
}

fn border_source_coordinate(
    coordinate: usize,
    start: usize,
    end: usize,
    mode: BorderFillMode,
) -> usize {
    match mode {
        BorderFillMode::Replicate => coordinate.clamp(start, end - 1),
        BorderFillMode::Reflect => reflect_coordinate(coordinate, start, end),
    }
}

fn reflect_coordinate(coordinate: usize, start: usize, end: usize) -> usize {
    let span = end - start;
    if span <= 1 {
        return start;
    }

    let period = span * 2 - 2;
    let offset = start.abs_diff(coordinate);
    let reflected = offset % period;
    start
        + if reflected < span {
            reflected
        } else {
            period - reflected
        }
}

fn copy_pixel(
    values: &mut [u8],
    width: usize,
    channels: usize,
    src_y: usize,
    src_x: usize,
    dst_y: usize,
    dst_x: usize,
) {
    for c in 0..channels {
        let src = (src_y * width + src_x) * channels + c;
        let dst = (dst_y * width + dst_x) * channels + c;
        values[dst] = values[src];
    }
}

pub(super) fn convert_to_rgb(values: &[u8], channels: usize) -> Result<Vec<u8>, TransformError> {
    validate_pixel_chunks(values, channels)?;
    match channels {
        1 => Ok(values
            .iter()
            .flat_map(|value| [*value, *value, *value])
            .collect()),
        3 => Ok(values.to_vec()),
        4 => Ok(values
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect()),
        _ => Err(TransformError::UnsupportedChannels(channels)),
    }
}

fn convert_to_grayscale(values: &[u8], channels: usize) -> Result<Vec<u8>, TransformError> {
    validate_pixel_chunks(values, channels)?;
    match channels {
        1 => Ok(values.to_vec()),
        3 | 4 => Ok(values
            .chunks_exact(channels)
            .map(|pixel| rgb_to_luma(pixel[0], pixel[1], pixel[2]))
            .collect()),
        _ => Err(TransformError::UnsupportedChannels(channels)),
    }
}

fn convert_to_rgba(values: &[u8], channels: usize) -> Result<Vec<u8>, TransformError> {
    validate_pixel_chunks(values, channels)?;
    match channels {
        1 => Ok(values
            .iter()
            .flat_map(|value| [*value, *value, *value, 255])
            .collect()),
        3 => Ok(values
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
            .collect()),
        4 => Ok(values.to_vec()),
        _ => Err(TransformError::UnsupportedChannels(channels)),
    }
}

fn rgb_to_luma(red: u8, green: u8, blue: u8) -> u8 {
    ((red as u32 * 299 + green as u32 * 587 + blue as u32 * 114 + 500) / 1000) as u8
}

fn pixel_luma(pixel: &[u8]) -> u8 {
    match pixel.len() {
        0 => 0,
        1 => pixel[0],
        _ => rgb_to_luma(pixel[0], pixel[1], pixel[2]),
    }
}

pub(super) fn frame_size(frame: &ImageFrame) -> ImageSize {
    ImageSize {
        height: frame.height(),
        width: frame.width(),
    }
}

pub(super) fn new_frame(
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    data: Vec<u8>,
    timing: crate::media::FrameTiming,
) -> Result<ImageFrame, TransformError> {
    ImageFrame::new(width, height, pixel_format, data)
        .map(|frame| frame.with_timing(timing))
        .map_err(transform_error_from_media)
}

fn transform_error_from_media(error: MediaError) -> TransformError {
    match error {
        MediaError::InvalidFrameDimensions { width, height } => {
            TransformError::InvalidSize { height, width }
        }
        MediaError::InvalidBufferLength { expected, actual } => {
            TransformError::InvalidBufferLength { expected, actual }
        }
        MediaError::FrameSizeOverflow => TransformError::ImageSizeOverflow,
        error => TransformError::ImageFrameInvariant(error.to_string()),
    }
}
