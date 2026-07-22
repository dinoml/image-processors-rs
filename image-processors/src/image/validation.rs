use super::*;

pub(super) fn validate_config(config: &ImageProcessorConfig) -> Result<(), ImageProcessorError> {
    validate_output_layout(config.output_layout)?;
    validate_configured_dimension("height", config.height)?;
    validate_configured_dimension("width", config.width)?;
    ResizeDecision::new(config.resample, config.resize_parity)?;
    if !config.rescale_factor.is_finite() {
        return Err(ImageProcessorError::NonFiniteRescaleFactor(
            config.rescale_factor,
        ));
    }

    validate_finite_stats("image_mean", &config.image_mean)?;
    validate_finite_stats("image_std", &config.image_std)?;
    validate_positive_std(&config.image_std)?;

    if config.do_normalize {
        validate_normalization_config("image_mean", &config.image_mean, config.pixel_format)?;
        validate_normalization_config("image_std", &config.image_std, config.pixel_format)?;
    }

    Ok(())
}

pub(super) fn resize_decision_for_parts(
    filter: ResizeFilter,
    parity: ResizeParity,
) -> ResizeDecision {
    match parity {
        ResizeParity::Resampling => ResizeDecision::resampling(filter),
        ResizeParity::Compatibility => ResizeDecision::compatibility(filter),
        ResizeParity::Torchvision => ResizeDecision::torchvision(filter),
        ResizeParity::PixelExact => ResizeDecision::pixel_exact(),
    }
}

#[cfg(feature = "parallel")]
pub(super) fn parallel_worker_count(items: usize) -> usize {
    debug_assert!(items > 0);
    rayon::current_num_threads().min(items).max(1)
}

fn validate_output_layout(layout: Layout) -> Result<(), ImageProcessorError> {
    match layout {
        Layout::NCHW | Layout::NHWC => Ok(()),
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

pub(super) fn validate_configured_dimension(
    field: &'static str,
    value: Option<usize>,
) -> Result<(), ImageProcessorError> {
    if value == Some(0) {
        Err(ImageProcessorError::InvalidConfiguredDimension { field, value: 0 })
    } else {
        Ok(())
    }
}

fn validate_finite_stats(field: &'static str, values: &[f32]) -> Result<(), ImageProcessorError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ImageProcessorError::NonFiniteNormalizationStat {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_positive_std(values: &[f32]) -> Result<(), ImageProcessorError> {
    for (index, value) in values.iter().copied().enumerate() {
        if value <= 0.0 {
            return Err(ImageProcessorError::NonPositiveNormalizationStd { index, value });
        }
    }
    Ok(())
}

fn validate_normalization_config(
    field: &'static str,
    values: &[f32],
    pixel_format: Option<PixelFormat>,
) -> Result<(), ImageProcessorError> {
    if values.is_empty() {
        return Err(ImageProcessorError::EmptyNormalizationStats { field });
    }
    if let Some(pixel_format) = pixel_format {
        validate_channel_stats(values, pixel_format.channels())?;
    }
    Ok(())
}

pub(super) fn validate_channel_stats(
    values: &[f32],
    channels: usize,
) -> Result<(), ImageProcessorError> {
    if values.len() == 1 || values.len() == channels {
        Ok(())
    } else {
        Err(ImageProcessorError::InvalidNormalizationStats {
            channels,
            actual: values.len(),
        })
    }
}

pub(super) fn channel_stat(values: &[f32], channel: usize) -> f32 {
    if values.len() == 1 {
        values[0]
    } else {
        values[channel]
    }
}
