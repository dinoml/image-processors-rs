//! Floating-point OpenCV preprocessing with ordinary processor tensor options.

use super::*;

/// Resize an interleaved floating-point image without byte quantization.
///
/// # Errors
/// Reports invalid dimensions, storage lengths, channels, nonfinite inputs or
/// native OpenCV resize failures.
pub fn resize_f32_opencv_linear(
    values: &[f32],
    source: ImageSize,
    channels: usize,
    target: ImageSize,
) -> Result<Vec<f32>, ImageProcessorError> {
    Ok(image_resize_kernels::resize_f32_opencv_linear(
        values,
        resize_kernel::kernel_size(source.height, source.width),
        channels,
        resize_kernel::kernel_size(target.height, target.width),
    )?)
}

impl ImageProcessor {
    /// Divide decoded bytes in FP64 before FP32 OpenCV linear interpolation.
    ///
    /// Pixel conversion precedes division. The divisor is an additional first
    /// stage; configured `do_rescale`, normalization and binarization still run
    /// after resizing. Disable `do_rescale` when the division is the only desired
    /// rescaling. Output layout uses the same tensor conversion as other paths.
    /// Existing byte-based entrypoints and defaults are unchanged.
    ///
    /// # Errors
    /// Rejects invalid divisors, unsupported resize modes/filters/parity, invalid
    /// image data or tensor layouts, and propagates resize/normalization failures.
    pub fn preprocess_image_divided_opencv_f32(
        &self,
        image: &ImageFrame,
        divisor: f64,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        let mut frame = Cow::Borrowed(image);
        if let Some(format) = self.config.pixel_format {
            if frame.pixel_format() != format {
                frame = Cow::Owned(convert_frame_pixel_format(frame.as_ref(), format)?);
            }
        }
        let source = ImageSize::new(frame.height(), frame.width())?;
        let target = if self.config.do_resize {
            if self.config.resample != ResizeFilter::Bilinear
                || self.config.resize_parity != ResizeParity::OpenCvLinear
            {
                return Err(ImageProcessorError::UnsupportedProcessorOption {
                    field: "FP32 OpenCV resize requires bilinear/OpenCvLinear",
                });
            }
            if options.resize_mode.unwrap_or(self.config.resize_mode) != ResizeMode::Default {
                return Err(ImageProcessorError::UnsupportedProcessorOption {
                    field: "FP32 OpenCV resize mode",
                });
            }
            self.target_size_for_frame(frame.as_ref(), options)?
        } else {
            source
        };
        let channels = frame.pixel_format().channels();
        let values = image_resize_kernels::resize_u8_to_f32_opencv_linear(
            frame.data(),
            resize_kernel::kernel_size(source.height, source.width),
            channels,
            resize_kernel::kernel_size(target.height, target.width),
            divisor,
        )?;
        let mut tensor = Tensor::new(
            TensorData::F32(values),
            vec![1, target.height, target.width, channels],
            Layout::NHWC,
        )?;
        if self.config.do_rescale {
            tensor = tensor.rescale(self.config.rescale_factor)?;
        }
        if self.config.do_normalize {
            tensor = tensor.normalize_channels(&self.config.image_mean, &self.config.image_std)?;
        }
        if self.config.do_binarize {
            tensor = tensor.binarize();
        }
        Ok(tensor.to_layout(self.config.output_layout)?)
    }
}
