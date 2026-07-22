use super::*;

pub(super) fn direct_resize_crop(
    frame: &ImageFrame,
    target: ImageSize,
) -> Result<KernelResizeCrop, ImageProcessorError> {
    let source = ImageSize {
        height: frame.height(),
        width: frame.width(),
    };
    let resized = direct_cover_size(source, target)?;
    let origin_x = (resized.width - target.width) / 2;
    let origin_y = (resized.height - target.height) / 2;
    KernelResizeCrop::new(
        kernel_size(resized.height, resized.width),
        origin_x,
        origin_y,
        kernel_size(target.height, target.width),
    )
    .map_err(ImageProcessorError::ResizeKernel)
}

pub(super) fn direct_cover_size(
    source: ImageSize,
    target: ImageSize,
) -> Result<ImageSize, ImageProcessorError> {
    let ratio = target.width as f64 / target.height as f64;
    let source_ratio = source.width as f64 / source.height as f64;
    let width = if ratio > source_ratio {
        target.width
    } else {
        direct_checked_ratio_dimension(source.width, target.height, source.height)?
    }
    .max(target.width);
    let height = if ratio <= source_ratio {
        target.height
    } else {
        direct_checked_ratio_dimension(source.height, target.width, source.width)?
    }
    .max(target.height);
    Ok(ImageSize { height, width })
}

pub(super) fn direct_checked_ratio_dimension(
    numerator: usize,
    multiplier: usize,
    divisor: usize,
) -> Result<usize, ImageProcessorError> {
    debug_assert!(divisor > 0);
    let value = (numerator as u128) * (multiplier as u128) / (divisor as u128);
    if value > usize::MAX as u128 {
        return Err(ImageProcessorError::ResizeKernel(
            KernelResizeError::ImageSizeOverflow,
        ));
    }
    Ok(value as usize)
}

pub(super) fn kernel_size(height: usize, width: usize) -> KernelImageSize {
    KernelImageSize { height, width }
}

pub(super) fn kernel_filter(filter: ResizeFilter) -> KernelResizeFilter {
    match filter {
        ResizeFilter::Nearest => KernelResizeFilter::Nearest,
        ResizeFilter::Bilinear => KernelResizeFilter::Bilinear,
        ResizeFilter::Bicubic => KernelResizeFilter::Bicubic,
        ResizeFilter::Lanczos => KernelResizeFilter::Lanczos,
    }
}

pub(super) fn kernel_profile(parity: ResizeParity) -> KernelResizeProfile {
    match parity {
        ResizeParity::Compatibility => KernelResizeProfile::Pillow,
        ResizeParity::Torchvision => KernelResizeProfile::Torchvision,
        ResizeParity::Resampling | ResizeParity::PixelExact => KernelResizeProfile::Fast,
    }
}

pub(super) fn kernel_f32_layout(
    layout: Layout,
) -> Result<KernelF32ImageLayout, ImageProcessorError> {
    match layout {
        Layout::NCHW => Ok(KernelF32ImageLayout::Chw),
        Layout::NHWC => Ok(KernelF32ImageLayout::Hwc),
        Layout::NC
        | Layout::CHW
        | Layout::HWC
        | Layout::NPCHW
        | Layout::NPHWC
        | Layout::BFCHW
        | Layout::BFHWC
        | Layout::NIPCHW
        | Layout::NIPHWC => Err(ImageProcessorError::UnsupportedLayout(layout)),
    }
}
