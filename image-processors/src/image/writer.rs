use super::resize_kernel::{
    direct_resize_crop, kernel_f32_layout, kernel_filter, kernel_profile, kernel_size,
};
use super::validation::{channel_stat, validate_channel_stats};
use super::*;

pub(super) struct BatchTensorWriter {
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    pixels: usize,
    output_len: usize,
    layout: Layout,
    transform: ChannelValueTransform,
    do_binarize: bool,
}

impl BatchTensorWriter {
    pub(super) fn new(
        config: &ImageProcessorConfig,
        frame: &ImageFrame,
        batch: usize,
    ) -> Result<Self, ImageProcessorError> {
        let pixels = frame
            .height()
            .checked_mul(frame.width())
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let elements = pixels
            .checked_mul(frame.channels())
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let output_len = elements
            .checked_mul(batch)
            .ok_or(TensorError::ShapeElementCountOverflow)?;

        Ok(Self {
            batch,
            height: frame.height(),
            width: frame.width(),
            channels: frame.channels(),
            pixels,
            output_len,
            layout: config.output_layout,
            transform: ChannelValueTransform::new(config, frame.channels())?,
            do_binarize: config.do_binarize,
        })
    }

    pub(super) fn new_parts(
        config: &ImageProcessorConfig,
        size: ImageSize,
        channels: usize,
        batch: usize,
    ) -> Result<Self, ImageProcessorError> {
        let pixels = size
            .height
            .checked_mul(size.width)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let elements = pixels
            .checked_mul(channels)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let output_len = elements
            .checked_mul(batch)
            .ok_or(TensorError::ShapeElementCountOverflow)?;

        Ok(Self {
            batch,
            height: size.height,
            width: size.width,
            channels,
            pixels,
            output_len,
            layout: config.output_layout,
            transform: ChannelValueTransform::new(config, channels)?,
            do_binarize: config.do_binarize,
        })
    }

    pub(super) fn output_len(&self) -> usize {
        self.output_len
    }

    pub(super) fn item_len(&self) -> usize {
        self.output_len / self.batch
    }

    pub(super) fn shape(&self) -> Result<Vec<usize>, ImageProcessorError> {
        match self.layout {
            Layout::NCHW => Ok(vec![self.batch, self.channels, self.height, self.width]),
            Layout::NHWC => Ok(vec![self.batch, self.height, self.width, self.channels]),
            Layout::NC
            | Layout::CHW
            | Layout::HWC
            | Layout::NPCHW
            | Layout::NPHWC
            | Layout::BFCHW
            | Layout::BFHWC
            | Layout::NIPCHW
            | Layout::NIPHWC => Err(ImageProcessorError::UnsupportedLayout(self.layout)),
        }
    }

    pub(super) fn finish(
        &self,
        values: Vec<f32>,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Tensor, ImageProcessorError> {
        Tensor::new(TensorData::F32(values), self.shape()?, self.layout)
            .and_then(|tensor| tensor.with_leading_axis(leading_axis))
            .map_err(ImageProcessorError::Tensor)
    }

    pub(super) fn finish_f16<'a>(
        &self,
        values: &'a [f16],
        leading_axis: TensorLeadingAxis,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        TensorView::new(TensorDataView::F16(values), self.shape()?, self.layout)
            .and_then(|tensor| tensor.with_leading_axis(leading_axis))
            .map_err(ImageProcessorError::Tensor)
    }

    pub(super) fn validate_output_len(&self, actual: usize) -> Result<(), ImageProcessorError> {
        if actual == self.output_len {
            Ok(())
        } else {
            Err(TensorError::InvalidElementCount {
                expected: self.output_len,
                actual,
            }
            .into())
        }
    }

    pub(super) fn write_frame<T: TensorOutputElement>(
        &self,
        batch_index: usize,
        frame: &ImageFrame,
        output: &mut [T],
    ) -> Result<(), ImageProcessorError> {
        let item_len = self.item_len();
        let start = batch_index
            .checked_mul(item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let end = start
            .checked_add(item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let output = output
            .get_mut(start..end)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        self.write_frame_slice(frame, output)
    }

    pub(super) fn batch_slice<'a, T>(
        &self,
        batch_index: usize,
        output: &'a mut [T],
    ) -> Result<&'a mut [T], ImageProcessorError> {
        let item_len = self.item_len();
        let start = batch_index
            .checked_mul(item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        let end = start
            .checked_add(item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        output
            .get_mut(start..end)
            .ok_or_else(|| TensorError::ShapeElementCountOverflow.into())
    }

    pub(super) fn write_frame_slice<T: TensorOutputElement>(
        &self,
        frame: &ImageFrame,
        output: &mut [T],
    ) -> Result<(), ImageProcessorError> {
        if frame.height() != self.height
            || frame.width() != self.width
            || frame.channels() != self.channels
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if output.len() != self.item_len() {
            return Err(TensorError::InvalidElementCount {
                expected: self.item_len(),
                actual: output.len(),
            }
            .into());
        }

        match self.layout {
            Layout::NCHW => self.write_frame_nchw(frame, output),
            Layout::NHWC => self.write_frame_nhwc(frame, output),
            Layout::NC
            | Layout::CHW
            | Layout::HWC
            | Layout::NPCHW
            | Layout::NPHWC
            | Layout::BFCHW
            | Layout::BFHWC
            | Layout::NIPCHW
            | Layout::NIPHWC => return Err(ImageProcessorError::UnsupportedLayout(self.layout)),
        }
        Ok(())
    }

    fn write_frame_nchw<T: TensorOutputElement>(&self, frame: &ImageFrame, output: &mut [T]) {
        for (pixel_index, pixel) in frame.data().chunks_exact(self.channels).enumerate() {
            for (channel, value) in pixel.iter().copied().enumerate() {
                let destination = channel * self.pixels + pixel_index;
                output[destination] = T::from_f32(self.transform_value(channel, value));
            }
        }
    }

    fn write_frame_nhwc<T: TensorOutputElement>(&self, frame: &ImageFrame, output: &mut [T]) {
        for (pixel_index, pixel) in frame.data().chunks_exact(self.channels).enumerate() {
            let pixel_offset = pixel_index * self.channels;
            for (channel, value) in pixel.iter().copied().enumerate() {
                output[pixel_offset + channel] = T::from_f32(self.transform_value(channel, value));
            }
        }
    }

    fn transform_value(&self, channel: usize, value: u8) -> f32 {
        let value = self.transform.apply(channel, value);
        if self.do_binarize {
            if value < 0.5 {
                0.0
            } else {
                1.0
            }
        } else {
            value
        }
    }

    pub(super) fn write_resize_crop_direct<T: TensorOutputElement>(
        &self,
        frame: &ImageFrame,
        decision: ResizeDecision,
        output: &mut [T],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<(), ImageProcessorError> {
        let output = self.batch_slice(0, output)?;
        self.write_resize_crop_slice_direct(frame, decision, output, workspace)
    }

    pub(super) fn write_resize_crop_slice_direct<T: TensorOutputElement>(
        &self,
        frame: &ImageFrame,
        decision: ResizeDecision,
        output: &mut [T],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<(), ImageProcessorError> {
        self.write_resize_crop_slice_direct_with_resize(
            frame,
            decision,
            output,
            workspace.resize_workspace(),
        )
    }

    pub(super) fn write_resize_crop_slice_direct_with_resize<T: TensorOutputElement>(
        &self,
        frame: &ImageFrame,
        decision: ResizeDecision,
        output: &mut [T],
        workspace: &mut ResizeWorkspace,
    ) -> Result<(), ImageProcessorError> {
        if frame.channels() != self.channels {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        if output.len() != self.item_len() {
            return Err(TensorError::InvalidElementCount {
                expected: self.item_len(),
                actual: output.len(),
            }
            .into());
        }

        let target = ImageSize {
            height: self.height,
            width: self.width,
        };
        let crop = direct_resize_crop(frame, target)?;
        T::resize_u8_crop(
            frame.data(),
            kernel_size(frame.height(), frame.width()),
            frame.channels(),
            crop,
            kernel_filter(decision.filter()),
            kernel_profile(decision.parity()),
            output,
            kernel_f32_layout(self.layout)?,
            &self.transform.multiplier,
            &self.transform.offset,
            workspace,
        )
        .map_err(ImageProcessorError::ResizeKernel)
    }
}

pub(super) trait TensorOutputElement: Sized {
    fn from_f32(value: f32) -> Self;

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the low-level resize destination API"
    )]
    fn resize_u8_crop(
        values: &[u8],
        source: KernelImageSize,
        channels: usize,
        crop: KernelResizeCrop,
        filter: KernelResizeFilter,
        profile: KernelResizeProfile,
        output: &mut [Self],
        layout: KernelF32ImageLayout,
        multiplier: &[f32],
        offset: &[f32],
        workspace: &mut ResizeWorkspace,
    ) -> Result<(), KernelResizeError>;
}

impl TensorOutputElement for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn resize_u8_crop(
        values: &[u8],
        source: KernelImageSize,
        channels: usize,
        crop: KernelResizeCrop,
        filter: KernelResizeFilter,
        profile: KernelResizeProfile,
        output: &mut [Self],
        layout: KernelF32ImageLayout,
        multiplier: &[f32],
        offset: &[f32],
        workspace: &mut ResizeWorkspace,
    ) -> Result<(), KernelResizeError> {
        image_resize_kernels::resize_u8_crop_into_f32_with_workspace(
            values, source, channels, crop, filter, profile, output, layout, multiplier, offset,
            workspace,
        )
    }
}

impl TensorOutputElement for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    fn resize_u8_crop(
        values: &[u8],
        source: KernelImageSize,
        channels: usize,
        crop: KernelResizeCrop,
        filter: KernelResizeFilter,
        profile: KernelResizeProfile,
        output: &mut [Self],
        layout: KernelF32ImageLayout,
        multiplier: &[f32],
        offset: &[f32],
        workspace: &mut ResizeWorkspace,
    ) -> Result<(), KernelResizeError> {
        image_resize_kernels::resize_u8_crop_into_f16_with_workspace(
            values, source, channels, crop, filter, profile, output, layout, multiplier, offset,
            workspace,
        )
    }
}

struct ChannelValueTransform {
    multiplier: Vec<f32>,
    offset: Vec<f32>,
}

impl ChannelValueTransform {
    pub(super) fn new(
        config: &ImageProcessorConfig,
        channels: usize,
    ) -> Result<Self, ImageProcessorError> {
        let scale = if config.do_rescale {
            config.rescale_factor
        } else {
            1.0
        };

        if config.do_normalize {
            validate_channel_stats(&config.image_mean, channels)?;
            validate_channel_stats(&config.image_std, channels)?;
            let multiplier = (0..channels)
                .map(|channel| scale / channel_stat(&config.image_std, channel))
                .collect();
            let offset = (0..channels)
                .map(|channel| {
                    -channel_stat(&config.image_mean, channel)
                        / channel_stat(&config.image_std, channel)
                })
                .collect();
            Ok(Self { multiplier, offset })
        } else {
            Ok(Self {
                multiplier: vec![scale; channels],
                offset: vec![0.0; channels],
            })
        }
    }

    pub(super) fn apply(&self, channel: usize, value: u8) -> f32 {
        value as f32 * self.multiplier[channel] + self.offset[channel]
    }
}
