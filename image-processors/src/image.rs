//! Image processor configuration and preprocessing.
//!
//! This module provides a configurable processor that loads media sources,
//! prepares decoded frames, and returns tensors in model-friendly layouts.

use std::borrow::Cow;
use std::path::Path;

use half::f16;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::media::{
    load_image_from_path_with_backend, DefaultMediaLoader, ImageDecodeBackend, ImageFrame,
    ImageSequence, LoadedMedia, MediaError, MediaSource, PixelFormat, VideoClip,
};
use crate::output::ProcessorOutput;
use crate::tensor::{
    ImageLayout, Layout, Tensor, TensorData, TensorDataView, TensorError, TensorLeadingAxis,
    TensorView, VideoLayout,
};
use crate::transforms::{
    convert_frame_pixel_format, resize_frame_owned_with_decision,
    resize_frame_owned_with_decision_workspace, resize_frame_with_decision, ImageSize,
    ResizeDecision, ResizeFilter, ResizeMode, ResizeParity, TransformError,
};
use image_resize_kernels::{
    F32ImageLayout as KernelF32ImageLayout, ImageSize as KernelImageSize,
    ResizeCrop as KernelResizeCrop, ResizeError as KernelResizeError,
    ResizeFilter as KernelResizeFilter, ResizeProfile as KernelResizeProfile, ResizeWorkspace,
};

mod config;
mod error;
mod resize_kernel;
mod stack;
mod validation;
mod workspace;
mod writer;

pub use config::{
    default_batch_execution, BatchExecution, ImageProcessorConfig, ImageProcessorOptions,
    DEFAULT_PARALLEL_BATCH_THRESHOLD,
};
pub use error::ImageProcessorError;
pub use workspace::ImageProcessorWorkspace;

pub(crate) use stack::stack_frame_tensors_as_video_batch;
use stack::stack_tensors;
#[cfg(feature = "parallel")]
use validation::parallel_worker_count;
use validation::validate_config;
use writer::{BatchTensorWriter, TensorOutputElement};

/// Configurable image and video preprocessor.
#[derive(Clone, Debug, Default)]
pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl ImageProcessor {
    /// Creates a processor from validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when resize dimensions, output layout, rescale factor,
    /// or normalization statistics are invalid.
    pub fn new(config: ImageProcessorConfig) -> Result<Self, ImageProcessorError> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    /// Returns this processor's configuration.
    pub fn config(&self) -> &ImageProcessorConfig {
        &self.config
    }

    /// Loads and preprocesses media from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Tensor, ImageProcessorError> {
        self.open_with_options(path, ImageProcessorOptions::default())
    }

    /// Loads and preprocesses media from a filesystem path as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_output(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.open(path).map(ProcessorOutput::from_pixel_values)
    }

    /// Loads and preprocesses media from a path with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_with_options(
        &self,
        path: impl AsRef<Path>,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.open_source_with_options(MediaSource::path(path.as_ref()), options)
    }

    /// Loads and preprocesses a media source.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_source(&self, source: MediaSource) -> Result<Tensor, ImageProcessorError> {
        self.open_source_with_options(source, ImageProcessorOptions::default())
    }

    /// Loads and preprocesses a media source as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_source_output(
        &self,
        source: MediaSource,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.open_source(source)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Loads and preprocesses a media source with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when loading, decoding, resizing, or tensor conversion fails.
    pub fn open_source_with_options(
        &self,
        source: MediaSource,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        match DefaultMediaLoader.load_with_remote_options_and_image_decode_backend(
            source,
            &Default::default(),
            self.config.decode_backend,
        )? {
            LoadedMedia::Image(image) => self.preprocess_image_with_options(&image, options),
            LoadedMedia::ImageSequence(sequence) => {
                self.preprocess_image_sequence_with_options(&sequence, options)
            }
            LoadedMedia::Video(video) => self.preprocess_video_with_options(&video, options),
        }
    }

    /// Decodes and preprocesses image bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, resizing, or tensor conversion fails.
    pub fn open_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<Tensor, ImageProcessorError> {
        self.open_source(MediaSource::image_bytes(bytes))
    }

    /// Decodes and preprocesses image bytes as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, resizing, or tensor conversion fails.
    pub fn open_bytes_output(
        &self,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.open_bytes(bytes)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Loads image paths and preprocesses them as a batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
    pub fn open_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Tensor, ImageProcessorError> {
        if self.should_run_parallel(paths.len())? {
            return self.open_batch_parallel(paths, ImageProcessorOptions::default());
        }

        self.preprocess_owned_image_batch_with_options(
            paths
                .iter()
                .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend)),
            paths.len(),
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Batch,
        )
    }

    /// Loads image paths and preprocesses them as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
    pub fn open_batch_output<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.open_batch(paths)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Loads image paths using reusable buffers and returns a batch tensor.
    ///
    /// The returned tensor owns its output buffer. Call
    /// [`ImageProcessorWorkspace::recycle_tensor`] after consuming the tensor
    /// to make that buffer available to the next workspace call.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, loading fails, or shapes are incompatible.
    pub fn open_batch_into<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError> {
        if self.should_run_parallel(paths.len())? {
            return self.open_batch_parallel_into(
                paths,
                ImageProcessorOptions::default(),
                workspace,
            );
        }

        self.preprocess_owned_image_batch_with_options_into(
            paths
                .iter()
                .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend)),
            paths.len(),
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Batch,
            workspace,
        )
    }

    pub(crate) fn open_batch_f16_into_slice<'a, P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        output: &'a mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        if self.should_run_parallel(paths.len())? {
            return self.open_batch_parallel_f16_into_slice(
                paths,
                ImageProcessorOptions::default(),
                output,
                workspace,
            );
        }

        self.preprocess_owned_image_batch_f16_into_slice(
            paths
                .iter()
                .map(|path| load_image_from_path_with_backend(path, self.config.decode_backend)),
            paths.len(),
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Batch,
            output,
            workspace,
        )
    }

    /// Preprocesses one decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, normalization, or tensor conversion fails.
    pub fn preprocess_image(&self, image: &ImageFrame) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_with_options(image, ImageProcessorOptions::default())
    }

    /// Preprocesses one decoded image frame as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, normalization, or tensor conversion fails.
    pub fn preprocess_image_output(
        &self,
        image: &ImageFrame,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image(image)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses one decoded image frame with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, normalization, or tensor conversion fails.
    pub fn preprocess_image_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            std::slice::from_ref(image),
            options,
            TensorLeadingAxis::Batch,
        )
    }

    /// Preprocesses one decoded image frame with per-call options as typed output.
    ///
    /// # Errors
    ///
    /// Returns an error when resizing, normalization, or tensor conversion fails.
    pub fn preprocess_image_output_with_options(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_with_options(image, options)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Returns a zero-copy tensor view for byte-preserving image preprocessing.
    ///
    /// This entrypoint borrows decoded frame bytes directly. It is available
    /// only when preprocessing does not resize, convert pixel format, rescale,
    /// normalize, binarize, or transpose away from NHWC layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured preprocessing requires owned output
    /// or when tensor view construction fails.
    pub fn preprocess_image_view<'a>(
        &self,
        image: &'a ImageFrame,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        self.preprocess_image_view_with_options(image, ImageProcessorOptions::default())
    }

    /// Returns a zero-copy tensor view with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured preprocessing requires owned output
    /// or when tensor view construction fails.
    pub fn preprocess_image_view_with_options<'a>(
        &self,
        image: &'a ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        self.zero_copy_image_view(image, options, TensorLeadingAxis::Batch)
    }

    /// Preprocesses decoded image frames as a batch tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or processed shapes are incompatible.
    pub fn preprocess_images(&self, images: &[ImageFrame]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            images,
            ImageProcessorOptions::default(),
            TensorLeadingAxis::Batch,
        )
    }

    /// Preprocesses decoded image frames as a batch tensor with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or processed shapes are incompatible.
    pub fn preprocess_images_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(images, options, TensorLeadingAxis::Batch)
    }

    /// Preprocesses decoded image frames as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty or processed shapes are incompatible.
    pub fn preprocess_images_output(
        &self,
        images: &[ImageFrame],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_images(images)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses an image sequence as a frame tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty or frame preprocessing fails.
    pub fn preprocess_image_sequence(
        &self,
        sequence: &ImageSequence,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_sequence_with_options(sequence, ImageProcessorOptions::default())
    }

    /// Preprocesses an image sequence as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty or frame preprocessing fails.
    pub fn preprocess_image_sequence_output(
        &self,
        sequence: &ImageSequence,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_image_sequence(sequence)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Returns zero-copy tensor views for byte-preserving image sequence preprocessing.
    ///
    /// Each returned view borrows one decoded frame and is marked with a frame
    /// leading axis. A single stacked zero-copy tensor is not returned because
    /// decoded sequence frames are owned as separate buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured preprocessing requires owned output
    /// or when tensor view construction fails.
    pub fn preprocess_image_sequence_views<'a>(
        &self,
        sequence: &'a ImageSequence,
    ) -> Result<Vec<TensorView<'a>>, ImageProcessorError> {
        self.preprocess_image_sequence_views_with_options(
            sequence,
            ImageProcessorOptions::default(),
        )
    }

    /// Returns zero-copy image sequence tensor views with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is empty, configured preprocessing
    /// requires owned output, or tensor view construction fails.
    pub fn preprocess_image_sequence_views_with_options<'a>(
        &self,
        sequence: &'a ImageSequence,
        options: ImageProcessorOptions,
    ) -> Result<Vec<TensorView<'a>>, ImageProcessorError> {
        if sequence.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        sequence
            .frames()
            .iter()
            .map(|image| self.zero_copy_image_view(image, options, TensorLeadingAxis::Frames))
            .collect()
    }

    /// Preprocesses a video clip as a frame tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty or frame preprocessing fails.
    pub fn preprocess_video(&self, video: &VideoClip) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_video_with_options(video, ImageProcessorOptions::default())
    }

    /// Preprocesses a video clip as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty or frame preprocessing fails.
    pub fn preprocess_video_output(
        &self,
        video: &VideoClip,
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_video(video)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Preprocesses decoded video clips as a batched video tensor.
    ///
    /// The output shape is `[batch, frames, channels, height, width]` for
    /// channel-first output and `[batch, frames, height, width, channels]` for
    /// channel-last output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, frame
    /// preprocessing fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos(&self, videos: &[VideoClip]) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_videos_with_options(videos, ImageProcessorOptions::default())
    }

    /// Preprocesses decoded video clips as typed processor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, any clip is empty, frame
    /// preprocessing fails, or processed clip shapes are incompatible.
    pub fn preprocess_videos_output(
        &self,
        videos: &[VideoClip],
    ) -> Result<ProcessorOutput, ImageProcessorError> {
        self.preprocess_videos(videos)
            .map(ProcessorOutput::from_pixel_values)
    }

    /// Returns zero-copy tensor views for byte-preserving video preprocessing.
    ///
    /// Each returned view borrows one decoded video frame and is marked with a
    /// frame leading axis. A single stacked zero-copy tensor is not returned
    /// because decoded video frames are owned as separate buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured preprocessing requires owned output
    /// or when tensor view construction fails.
    pub fn preprocess_video_views<'a>(
        &self,
        video: &'a VideoClip,
    ) -> Result<Vec<TensorView<'a>>, ImageProcessorError> {
        self.preprocess_video_views_with_options(video, ImageProcessorOptions::default())
    }

    /// Returns zero-copy video tensor views with per-call options.
    ///
    /// # Errors
    ///
    /// Returns an error when the clip is empty, configured preprocessing
    /// requires owned output, or tensor view construction fails.
    pub fn preprocess_video_views_with_options<'a>(
        &self,
        video: &'a VideoClip,
        options: ImageProcessorOptions,
    ) -> Result<Vec<TensorView<'a>>, ImageProcessorError> {
        if video.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        video
            .frames()
            .iter()
            .map(|frame| {
                self.zero_copy_image_view(frame.image(), options, TensorLeadingAxis::Frames)
            })
            .collect()
    }

    fn prepare_frame<'a>(
        &self,
        image: &'a ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Cow<'a, ImageFrame>, ImageProcessorError> {
        let mut frame = Cow::Borrowed(image);

        if self.config.do_resize {
            let target = self.target_size_for_frame(frame.as_ref(), options)?;
            frame = Cow::Owned(resize_frame_with_decision(
                frame.as_ref(),
                target,
                self.config.resize_decision(),
                options.resize_mode.unwrap_or(self.config.resize_mode),
            )?);
        }

        if let Some(pixel_format) = self.config.pixel_format {
            if frame.pixel_format() != pixel_format {
                frame = Cow::Owned(convert_frame_pixel_format(frame.as_ref(), pixel_format)?);
            }
        }

        Ok(frame)
    }

    fn prepare_frame_owned(
        &self,
        image: ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageFrame, ImageProcessorError> {
        let mut frame = image;

        if self.config.do_resize {
            let target = self.target_size_for_frame(&frame, options)?;
            frame = resize_frame_owned_with_decision(
                frame,
                target,
                self.config.resize_decision(),
                options.resize_mode.unwrap_or(self.config.resize_mode),
            )?;
        }

        if let Some(pixel_format) = self.config.pixel_format {
            if frame.pixel_format() != pixel_format {
                frame = convert_frame_pixel_format(&frame, pixel_format)?;
            }
        }

        Ok(frame)
    }

    fn prepare_frame_owned_with_workspace(
        &self,
        image: ImageFrame,
        options: ImageProcessorOptions,
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<ImageFrame, ImageProcessorError> {
        self.prepare_frame_owned_with_resize_workspace(image, options, workspace.resize_workspace())
    }

    fn prepare_frame_owned_with_resize_workspace(
        &self,
        image: ImageFrame,
        options: ImageProcessorOptions,
        workspace: &mut ResizeWorkspace,
    ) -> Result<ImageFrame, ImageProcessorError> {
        let mut frame = image;

        if self.config.do_resize {
            let target = self.target_size_for_frame(&frame, options)?;
            frame = resize_frame_owned_with_decision_workspace(
                frame,
                target,
                self.config.resize_decision(),
                options.resize_mode.unwrap_or(self.config.resize_mode),
                workspace,
            )?;
        }

        if let Some(pixel_format) = self.config.pixel_format {
            if frame.pixel_format() != pixel_format {
                frame = convert_frame_pixel_format(&frame, pixel_format)?;
            }
        }

        Ok(frame)
    }

    fn validate_zero_copy_image_view(
        &self,
        image: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<(), ImageProcessorError> {
        if self.config.output_layout != Layout::NHWC {
            return Err(ImageProcessorError::ZeroCopyUnavailable {
                reason: "output layout must be NHWC",
            });
        }

        if self.config.do_rescale {
            return Err(ImageProcessorError::ZeroCopyUnavailable {
                reason: "rescale converts pixel bytes",
            });
        }

        if self.config.do_normalize {
            return Err(ImageProcessorError::ZeroCopyUnavailable {
                reason: "normalize converts pixel bytes",
            });
        }

        if self.config.do_binarize {
            return Err(ImageProcessorError::ZeroCopyUnavailable {
                reason: "binarize converts pixel bytes",
            });
        }

        if self.config.do_resize {
            let target = self.target_size_for_frame(image, options)?;
            if target.height != image.height() || target.width != image.width() {
                return Err(ImageProcessorError::ZeroCopyUnavailable {
                    reason: "resize requires owned output",
                });
            }
        }

        if let Some(pixel_format) = self.config.pixel_format {
            if pixel_format != image.pixel_format() {
                return Err(ImageProcessorError::ZeroCopyUnavailable {
                    reason: "pixel format conversion requires owned output",
                });
            }
        }

        Ok(())
    }

    fn zero_copy_image_view<'a>(
        &self,
        image: &'a ImageFrame,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        self.validate_zero_copy_image_view(image, options)?;
        TensorView::new(
            TensorDataView::U8(image.data()),
            vec![1, image.height(), image.width(), image.channels()],
            Layout::NHWC,
        )
        .and_then(|view| view.with_leading_axis(leading_axis))
        .map_err(ImageProcessorError::Tensor)
    }

    fn preprocess_image_sequence_with_options(
        &self,
        sequence: &ImageSequence,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        self.preprocess_image_batch_with_options(
            sequence.frames(),
            options,
            TensorLeadingAxis::Frames,
        )
    }

    fn preprocess_video_with_options(
        &self,
        video: &VideoClip,
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        if video.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let tensors = video
            .frames()
            .iter()
            .map(|frame| self.preprocess_image_with_options(frame.image(), options))
            .collect::<Result<Vec<_>, _>>()?;
        stack_tensors(tensors, TensorLeadingAxis::Frames)
    }

    fn preprocess_videos_with_options(
        &self,
        videos: &[VideoClip],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        if videos.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let tensors = videos
            .iter()
            .map(|video| self.preprocess_video_with_options(video, options))
            .collect::<Result<Vec<_>, _>>()?;
        stack_frame_tensors_as_video_batch(tensors)
    }

    #[cfg_attr(
        feature = "parallel",
        allow(
            clippy::unnecessary_wraps,
            reason = "the no-parallel build returns an unavailable-feature error"
        )
    )]
    fn should_run_parallel(&self, batch: usize) -> Result<bool, ImageProcessorError> {
        match self.config.batch_execution {
            BatchExecution::Auto => {
                #[cfg(feature = "parallel")]
                {
                    Ok(batch >= DEFAULT_PARALLEL_BATCH_THRESHOLD)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = batch;
                    Ok(false)
                }
            }
            BatchExecution::Serial => Ok(false),
            BatchExecution::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    Ok(true)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = batch;
                    Err(ImageProcessorError::ParallelBatchUnavailable)
                }
            }
        }
    }

    fn preprocess_image_batch_with_options(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Tensor, ImageProcessorError> {
        if images.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        #[cfg(feature = "parallel")]
        if self.should_run_parallel(images.len())? {
            return self.preprocess_image_batch_with_options_parallel(
                images,
                options,
                leading_axis,
            );
        }

        let first = self.prepare_frame(&images[0], options)?;
        let writer = BatchTensorWriter::new(&self.config, first.as_ref(), images.len())?;
        let mut values = vec![0.0; writer.output_len()];
        writer.write_frame(0, first.as_ref(), &mut values)?;

        for (batch_index, image) in images.iter().enumerate().skip(1) {
            let frame = self.prepare_frame(image, options)?;
            writer.write_frame(batch_index, frame.as_ref(), &mut values)?;
        }

        writer.finish(values, leading_axis)
    }

    #[cfg(feature = "parallel")]
    fn preprocess_image_batch_with_options_parallel(
        &self,
        images: &[ImageFrame],
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Tensor, ImageProcessorError> {
        let first = self.prepare_frame(&images[0], options)?;
        let writer = BatchTensorWriter::new(&self.config, first.as_ref(), images.len())?;
        let item_len = writer.item_len();
        let mut values = vec![0.0; writer.output_len()];
        let first_output = values
            .get_mut(..item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        writer.write_frame_slice(first.as_ref(), first_output)?;

        let results = values[item_len..]
            .par_chunks_mut(item_len)
            .zip(images[1..].par_iter())
            .enumerate()
            .map(|(offset, (output, image))| {
                let index = offset + 1;
                let result = (|| {
                    let frame = self.prepare_frame(image, options)?;
                    writer.write_frame_slice(frame.as_ref(), output)
                })();
                (index, result)
            })
            .collect::<Vec<_>>();

        for (_, result) in results {
            result?;
        }

        writer.finish(values, leading_axis)
    }

    fn preprocess_owned_image_batch_with_options<I>(
        &self,
        mut images: I,
        batch: usize,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
    ) -> Result<Tensor, ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        if batch == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = images.next().ok_or(ImageProcessorError::EmptyBatch)??;
        let first = self.prepare_frame_owned(first, options)?;
        let writer = BatchTensorWriter::new(&self.config, &first, batch)?;
        let mut values = vec![0.0; writer.output_len()];
        writer.write_frame(0, &first, &mut values)?;

        let mut written = 1usize;
        for image in images {
            let frame = self.prepare_frame_owned(image?, options)?;
            writer.write_frame(written, &frame, &mut values)?;
            written += 1;
        }

        if written != batch {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }

        writer.finish(values, leading_axis)
    }

    fn preprocess_owned_image_batch_with_options_into<I>(
        &self,
        mut images: I,
        batch: usize,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        if batch == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = images.next().ok_or(ImageProcessorError::EmptyBatch)??;
        if let Some(tensor) = self.try_preprocess_owned_resize_crop_batch_into(
            &first,
            &mut images,
            batch,
            options,
            leading_axis,
            workspace,
        )? {
            return Ok(tensor);
        }

        let first = self.prepare_frame_owned_with_workspace(first, options, workspace)?;
        let writer = BatchTensorWriter::new(&self.config, &first, batch)?;
        let mut values = workspace.take_f32_values(writer.output_len());
        self.write_prepared_owned_batch(
            &first,
            &mut images,
            batch,
            options,
            &writer,
            &mut values,
            workspace,
        )?;

        writer.finish(values, leading_axis)
    }

    fn preprocess_owned_image_batch_f16_into_slice<'a, I>(
        &self,
        mut images: I,
        batch: usize,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
        output: &'a mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<TensorView<'a>, ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        if batch == 0 {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = images.next().ok_or(ImageProcessorError::EmptyBatch)??;
        if let Some(writer) = self.try_preprocess_owned_resize_crop_batch_f16_into_slice(
            &first,
            &mut images,
            batch,
            options,
            output,
            workspace,
        )? {
            return writer.finish_f16(output, leading_axis);
        }

        let first = self.prepare_frame_owned_with_workspace(first, options, workspace)?;
        let writer = BatchTensorWriter::new(&self.config, &first, batch)?;
        writer.validate_output_len(output.len())?;
        self.write_prepared_owned_batch(
            &first,
            &mut images,
            batch,
            options,
            &writer,
            output,
            workspace,
        )?;

        writer.finish_f16(output, leading_axis)
    }

    fn try_preprocess_owned_resize_crop_batch_f16_into_slice<I>(
        &self,
        first: &ImageFrame,
        images: &mut I,
        batch: usize,
        options: ImageProcessorOptions,
        output: &mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Option<BatchTensorWriter>, ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        let Some(first) = self.prepare_resize_crop_direct_frame(first, options)? else {
            return Ok(None);
        };

        let target = self.target_size_for_frame(first.as_ref(), options)?;
        let writer =
            BatchTensorWriter::new_parts(&self.config, target, first.as_ref().channels(), batch)?;
        writer.validate_output_len(output.len())?;
        self.write_owned_resize_crop_batch(
            first.as_ref(),
            images,
            batch,
            options,
            &writer,
            output,
            workspace,
        )?;

        Ok(Some(writer))
    }

    fn try_preprocess_owned_resize_crop_batch_into<I>(
        &self,
        first: &ImageFrame,
        images: &mut I,
        batch: usize,
        options: ImageProcessorOptions,
        leading_axis: TensorLeadingAxis,
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Option<Tensor>, ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        let Some(first) = self.prepare_resize_crop_direct_frame(first, options)? else {
            return Ok(None);
        };

        let target = self.target_size_for_frame(first.as_ref(), options)?;
        let writer =
            BatchTensorWriter::new_parts(&self.config, target, first.as_ref().channels(), batch)?;
        let mut values = workspace.take_f32_values(writer.output_len());
        self.write_owned_resize_crop_batch(
            first.as_ref(),
            images,
            batch,
            options,
            &writer,
            &mut values,
            workspace,
        )?;

        writer.finish(values, leading_axis).map(Some)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "shares serial batch orchestration across owned f32 and borrowed f16 outputs"
    )]
    fn write_prepared_owned_batch<I, T: TensorOutputElement>(
        &self,
        first: &ImageFrame,
        images: &mut I,
        batch: usize,
        options: ImageProcessorOptions,
        writer: &BatchTensorWriter,
        output: &mut [T],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<(), ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        writer.write_frame(0, first, output)?;
        let mut written = 1usize;
        for image in images {
            let frame = self.prepare_frame_owned_with_workspace(image?, options, workspace)?;
            writer.write_frame(written, &frame, output)?;
            written += 1;
        }
        if written != batch {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "shares direct crop orchestration across owned f32 and borrowed f16 outputs"
    )]
    fn write_owned_resize_crop_batch<I, T: TensorOutputElement>(
        &self,
        first: &ImageFrame,
        images: &mut I,
        batch: usize,
        options: ImageProcessorOptions,
        writer: &BatchTensorWriter,
        output: &mut [T],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<(), ImageProcessorError>
    where
        I: Iterator<Item = Result<ImageFrame, MediaError>>,
    {
        writer.write_resize_crop_direct(first, self.config.resize_decision(), output, workspace)?;
        let mut written = 1usize;
        for image in images {
            let frame = image?;
            let Some(frame) = self.prepare_resize_crop_direct_frame(&frame, options)? else {
                return Err(ImageProcessorError::IncompatibleBatchShapes);
            };
            let item_output = writer.batch_slice(written, output)?;
            writer.write_resize_crop_slice_direct(
                frame.as_ref(),
                self.config.resize_decision(),
                item_output,
                workspace,
            )?;
            written += 1;
        }
        if written != batch {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        Ok(())
    }

    fn prepare_resize_crop_direct_frame<'a>(
        &self,
        frame: &'a ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<Option<Cow<'a, ImageFrame>>, ImageProcessorError> {
        if !self.config.do_resize
            || options.resize_mode.unwrap_or(self.config.resize_mode) != ResizeMode::Crop
            || self.config.pixel_format != Some(PixelFormat::Rgb8)
            || self.config.do_binarize
            || !matches!(self.config.output_layout, Layout::NCHW | Layout::NHWC)
        {
            return Ok(None);
        }

        let frame = if frame.pixel_format() == PixelFormat::Rgb8 {
            Cow::Borrowed(frame)
        } else {
            Cow::Owned(convert_frame_pixel_format(frame, PixelFormat::Rgb8)?)
        };

        if frame.channels() == 3 {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    fn target_size_for_frame(
        &self,
        frame: &ImageFrame,
        options: ImageProcessorOptions,
    ) -> Result<ImageSize, ImageProcessorError> {
        Ok(ImageSize::new(
            options
                .height
                .or(self.config.height)
                .unwrap_or_else(|| frame.height()),
            options
                .width
                .or(self.config.width)
                .unwrap_or_else(|| frame.width()),
        )?)
    }

    #[cfg(feature = "parallel")]
    fn try_open_batch_parallel_resize_crop<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        first: &ImageFrame,
        options: ImageProcessorOptions,
        workspace: Option<&mut ImageProcessorWorkspace>,
    ) -> Result<Option<Tensor>, ImageProcessorError> {
        let Some(first) = self.prepare_resize_crop_direct_frame(first, options)? else {
            return Ok(None);
        };

        let target = self.target_size_for_frame(first.as_ref(), options)?;
        let writer = BatchTensorWriter::new_parts(
            &self.config,
            target,
            first.as_ref().channels(),
            paths.len(),
        )?;
        let mut local_workspace = ImageProcessorWorkspace::new();
        let workspace = workspace.unwrap_or(&mut local_workspace);
        let mut values = workspace.take_f32_values(writer.output_len());
        self.write_open_batch_parallel_resize_crop(
            paths,
            first.as_ref(),
            options,
            &writer,
            &mut values,
            workspace,
        )?;

        writer.finish(values, TensorLeadingAxis::Batch).map(Some)
    }

    #[cfg(feature = "parallel")]
    fn try_open_batch_parallel_resize_crop_f16_into_slice<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        first: &ImageFrame,
        options: ImageProcessorOptions,
        output: &mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Option<BatchTensorWriter>, ImageProcessorError> {
        let Some(first) = self.prepare_resize_crop_direct_frame(first, options)? else {
            return Ok(None);
        };

        let target = self.target_size_for_frame(first.as_ref(), options)?;
        let writer = BatchTensorWriter::new_parts(
            &self.config,
            target,
            first.as_ref().channels(),
            paths.len(),
        )?;
        self.write_open_batch_parallel_resize_crop(
            paths,
            first.as_ref(),
            options,
            &writer,
            output,
            workspace,
        )?;

        Ok(Some(writer))
    }

    #[cfg(feature = "parallel")]
    fn write_open_batch_parallel_resize_crop<P, T>(
        &self,
        paths: &[P],
        first: &ImageFrame,
        options: ImageProcessorOptions,
        writer: &BatchTensorWriter,
        output: &mut [T],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<(), ImageProcessorError>
    where
        P: AsRef<Path> + Sync,
        T: TensorOutputElement + Send,
    {
        writer.validate_output_len(output.len())?;
        let item_len = writer.item_len();
        let first_output = output
            .get_mut(..item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        writer.write_resize_crop_slice_direct(
            first,
            self.config.resize_decision(),
            first_output,
            workspace,
        )?;

        let results = if paths.len() > 1 {
            let remaining = paths.len() - 1;
            let worker_count = parallel_worker_count(remaining);
            let items_per_worker = remaining.div_ceil(worker_count);
            let values_per_worker = item_len * items_per_worker;
            let workers = workspace.parallel_resize_workspaces(worker_count);
            output[item_len..]
                .par_chunks_mut(values_per_worker)
                .zip(paths[1..].par_chunks(items_per_worker))
                .zip(workers.par_iter_mut())
                .enumerate()
                .flat_map(|(worker_index, ((outputs, paths), worker))| {
                    paths
                        .iter()
                        .zip(outputs.chunks_mut(item_len))
                        .enumerate()
                        .map(|(local_index, (path, output))| {
                            let index = 1 + worker_index * items_per_worker + local_index;
                            let result = (|| {
                                let image = load_image_from_path_with_backend(
                                    path,
                                    self.config.decode_backend,
                                )?;
                                let Some(image) =
                                    self.prepare_resize_crop_direct_frame(&image, options)?
                                else {
                                    return Err(ImageProcessorError::IncompatibleBatchShapes);
                                };
                                writer.write_resize_crop_slice_direct_with_resize(
                                    image.as_ref(),
                                    self.config.resize_decision(),
                                    output,
                                    worker.resize_workspace(),
                                )
                            })();
                            (index, result)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for (_, result) in results {
            result?;
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn open_batch_parallel<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        if paths.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = load_image_from_path_with_backend(&paths[0], self.config.decode_backend)?;
        if let Some(tensor) =
            self.try_open_batch_parallel_resize_crop(paths, &first, options, None)?
        {
            return Ok(tensor);
        }

        let first = self.prepare_frame_owned(first, options)?;
        let writer = BatchTensorWriter::new(&self.config, &first, paths.len())?;
        let item_len = writer.item_len();
        let mut values = vec![0.0; writer.output_len()];
        let first_output = values
            .get_mut(..item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        writer.write_frame_slice(&first, first_output)?;

        let results = values[item_len..]
            .par_chunks_mut(item_len)
            .zip(paths[1..].par_iter())
            .enumerate()
            .map_init(
                ImageProcessorWorkspace::new,
                |workspace, (offset, (output, path))| {
                    let index = offset + 1;
                    let result = (|| {
                        let image =
                            load_image_from_path_with_backend(path, self.config.decode_backend)?;
                        let frame =
                            self.prepare_frame_owned_with_workspace(image, options, workspace)?;
                        writer.write_frame_slice(&frame, output)
                    })();
                    (index, result)
                },
            )
            .collect::<Vec<_>>();

        for (_, result) in results {
            result?;
        }

        writer.finish(values, TensorLeadingAxis::Batch)
    }

    #[cfg(feature = "parallel")]
    fn open_batch_parallel_into<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        options: ImageProcessorOptions,
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError> {
        if paths.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = load_image_from_path_with_backend(&paths[0], self.config.decode_backend)?;
        if let Some(tensor) =
            self.try_open_batch_parallel_resize_crop(paths, &first, options, Some(workspace))?
        {
            return Ok(tensor);
        }

        let first = self.prepare_frame_owned_with_workspace(first, options, workspace)?;
        let writer = BatchTensorWriter::new(&self.config, &first, paths.len())?;
        let item_len = writer.item_len();
        let mut values = workspace.take_f32_values(writer.output_len());
        let first_output = values
            .get_mut(..item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        writer.write_frame_slice(&first, first_output)?;

        let results = if paths.len() > 1 {
            let remaining = paths.len() - 1;
            let worker_count = parallel_worker_count(remaining);
            let items_per_worker = remaining.div_ceil(worker_count);
            let values_per_worker = item_len * items_per_worker;
            let workers = workspace.parallel_resize_workspaces(worker_count);
            values[item_len..]
                .par_chunks_mut(values_per_worker)
                .zip(paths[1..].par_chunks(items_per_worker))
                .zip(workers.par_iter_mut())
                .enumerate()
                .flat_map(|(worker_index, ((outputs, paths), worker))| {
                    paths
                        .iter()
                        .zip(outputs.chunks_mut(item_len))
                        .enumerate()
                        .map(|(local_index, (path, output))| {
                            let index = 1 + worker_index * items_per_worker + local_index;
                            let result = (|| {
                                let image = load_image_from_path_with_backend(
                                    path,
                                    self.config.decode_backend,
                                )?;
                                let frame = self.prepare_frame_owned_with_resize_workspace(
                                    image,
                                    options,
                                    worker.resize_workspace(),
                                )?;
                                writer.write_frame_slice(&frame, output)
                            })();
                            (index, result)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        for (_, result) in results {
            result?;
        }

        writer.finish(values, TensorLeadingAxis::Batch)
    }

    #[cfg(feature = "parallel")]
    fn open_batch_parallel_f16_into_slice<'a, P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        options: ImageProcessorOptions,
        output: &'a mut [f16],
        workspace: &mut ImageProcessorWorkspace,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        if paths.is_empty() {
            return Err(ImageProcessorError::EmptyBatch);
        }

        let first = load_image_from_path_with_backend(&paths[0], self.config.decode_backend)?;
        if let Some(writer) = self.try_open_batch_parallel_resize_crop_f16_into_slice(
            paths, &first, options, output, workspace,
        )? {
            return writer.finish_f16(output, TensorLeadingAxis::Batch);
        }

        let first = self.prepare_frame_owned_with_workspace(first, options, workspace)?;
        let writer = BatchTensorWriter::new(&self.config, &first, paths.len())?;
        writer.validate_output_len(output.len())?;
        let item_len = writer.item_len();
        let first_output = output
            .get_mut(..item_len)
            .ok_or(TensorError::ShapeElementCountOverflow)?;
        writer.write_frame_slice(&first, first_output)?;

        let results = if paths.len() > 1 {
            let remaining = paths.len() - 1;
            let worker_count = parallel_worker_count(remaining);
            let items_per_worker = remaining.div_ceil(worker_count);
            let values_per_worker = item_len * items_per_worker;
            let workers = workspace.parallel_resize_workspaces(worker_count);
            output[item_len..]
                .par_chunks_mut(values_per_worker)
                .zip(paths[1..].par_chunks(items_per_worker))
                .zip(workers.par_iter_mut())
                .enumerate()
                .flat_map(|(worker_index, ((outputs, paths), worker))| {
                    paths
                        .iter()
                        .zip(outputs.chunks_mut(item_len))
                        .enumerate()
                        .map(|(local_index, (path, output))| {
                            let index = 1 + worker_index * items_per_worker + local_index;
                            let result = (|| {
                                let image = load_image_from_path_with_backend(
                                    path,
                                    self.config.decode_backend,
                                )?;
                                let frame = self.prepare_frame_owned_with_resize_workspace(
                                    image,
                                    options,
                                    worker.resize_workspace(),
                                )?;
                                writer.write_frame_slice(&frame, output)
                            })();
                            (index, result)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        for (_, result) in results {
            result?;
        }

        writer.finish_f16(output, TensorLeadingAxis::Batch)
    }

    #[cfg(not(feature = "parallel"))]
    fn open_batch_parallel<P: AsRef<Path> + Sync>(
        &self,
        _paths: &[P],
        _options: ImageProcessorOptions,
    ) -> Result<Tensor, ImageProcessorError> {
        Err(ImageProcessorError::ParallelBatchUnavailable)
    }

    #[cfg(not(feature = "parallel"))]
    fn open_batch_parallel_into<P: AsRef<Path> + Sync>(
        &self,
        _paths: &[P],
        _options: ImageProcessorOptions,
        _workspace: &mut ImageProcessorWorkspace,
    ) -> Result<Tensor, ImageProcessorError> {
        Err(ImageProcessorError::ParallelBatchUnavailable)
    }

    #[cfg(not(feature = "parallel"))]
    fn open_batch_parallel_f16_into_slice<'a, P: AsRef<Path> + Sync>(
        &self,
        _paths: &[P],
        _options: ImageProcessorOptions,
        _output: &'a mut [f16],
        _workspace: &mut ImageProcessorWorkspace,
    ) -> Result<TensorView<'a>, ImageProcessorError> {
        Err(ImageProcessorError::ParallelBatchUnavailable)
    }
}

#[cfg(test)]
mod tests;
