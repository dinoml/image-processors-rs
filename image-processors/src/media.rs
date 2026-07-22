//! Media loading and decoded frame ownership.
//!
//! This module provides source descriptions, decoded image/video frame types,
//! optional URL loading, optional video decoding, and frame sampling utilities.

use std::fs::File;
#[cfg(feature = "url")]
use std::io::Read;
#[cfg(all(feature = "video", feature = "url"))]
use std::io::Write;
use std::io::{BufRead, BufReader, Cursor, Seek};
use std::path::{Path, PathBuf};
use std::time::Duration;

use ::image::codecs::gif::GifDecoder;
use ::image::codecs::png::PngDecoder;
use ::image::codecs::webp::WebPDecoder;
use ::image::metadata::LoopCount;
use ::image::{AnimationDecoder, DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::tensor::{ImageLayout, TensorDataView, TensorError, TensorView};

/// Pixel storage format for decoded image frames.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PixelFormat {
    /// One 8-bit luminance channel per pixel.
    Luma8,
    /// Three 8-bit RGB channels per pixel.
    Rgb8,
    /// Four 8-bit RGBA channels per pixel.
    Rgba8,
}

impl PixelFormat {
    /// Returns the number of channels per pixel.
    pub fn channels(self) -> usize {
        match self {
            Self::Luma8 => 1,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }

    /// Returns a short human-readable format name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Luma8 => "L",
            Self::Rgb8 => "RGB",
            Self::Rgba8 => "RGBA",
        }
    }
}

/// JPEG decode implementation backend.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum ImageDecodeBackend {
    /// Use the Rust `image` crate for all supported image formats.
    #[default]
    ImageCrate,
    /// Use `turbojpeg` for JPEG images and the `image` crate for other formats.
    TurboJpeg,
}

/// Broad media kind used for source routing.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MediaType {
    /// Still or animated image media.
    Image,
    /// Video media.
    Video,
}

/// Optional source hints used when media type inference is ambiguous.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MediaHint {
    /// Explicit media type override.
    pub media_type: Option<MediaType>,
    /// Explicit format or file extension hint.
    pub format: Option<String>,
    /// Optional frame sampling to apply after decoding.
    pub frame_sampling: Option<FrameSampling>,
    /// Optional remote read mode for URL sources.
    pub remote_read_mode: Option<RemoteReadMode>,
}

impl MediaHint {
    /// Returns a hint that treats the source as image media.
    pub fn image() -> Self {
        Self {
            media_type: Some(MediaType::Image),
            format: None,
            frame_sampling: None,
            remote_read_mode: None,
        }
    }

    /// Returns a hint that treats the source as video media.
    pub fn video() -> Self {
        Self {
            media_type: Some(MediaType::Video),
            format: None,
            frame_sampling: None,
            remote_read_mode: None,
        }
    }

    /// Returns this hint with an explicit format name.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Returns this hint with frame sampling options.
    pub fn with_frame_sampling(mut self, sampling: FrameSampling) -> Self {
        self.frame_sampling = Some(sampling);
        self
    }

    /// Returns this hint with a remote read mode.
    pub fn with_remote_read_mode(mut self, mode: RemoteReadMode) -> Self {
        self.remote_read_mode = Some(mode);
        self
    }
}

/// Concrete media source location.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MediaLocation {
    /// Local filesystem path.
    Path(PathBuf),
    /// Encoded media bytes already held in memory.
    Bytes(Vec<u8>),
    /// Remote URL.
    Url(String),
}

/// A media input plus optional decoding and loading hints.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MediaSource {
    /// Source location.
    pub location: MediaLocation,
    /// Optional media loading hints.
    pub hint: MediaHint,
}

impl MediaSource {
    /// Creates a source from a filesystem path.
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self {
            location: MediaLocation::Path(path.into()),
            hint: MediaHint::default(),
        }
    }

    /// Creates an image source from a filesystem path.
    pub fn image_path(path: impl Into<PathBuf>) -> Self {
        Self::path(path).with_media_type(MediaType::Image)
    }

    /// Creates a video source from a filesystem path.
    pub fn video_path(path: impl Into<PathBuf>) -> Self {
        Self::path(path).with_media_type(MediaType::Video)
    }

    /// Creates a source from encoded bytes.
    pub fn bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            location: MediaLocation::Bytes(bytes.into()),
            hint: MediaHint::default(),
        }
    }

    /// Creates an image source from encoded bytes.
    pub fn image_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::bytes(bytes).with_media_type(MediaType::Image)
    }

    /// Creates a source from a URL string.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            location: MediaLocation::Url(url.into()),
            hint: MediaHint::default(),
        }
    }

    /// Creates an image source from a URL string.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::url(url).with_media_type(MediaType::Image)
    }

    /// Creates a video source from a URL string.
    pub fn video_url(url: impl Into<String>) -> Self {
        Self::url(url).with_media_type(MediaType::Video)
    }

    /// Returns this source with an explicit media type.
    pub fn with_media_type(mut self, media_type: MediaType) -> Self {
        self.hint.media_type = Some(media_type);
        self
    }

    /// Returns this source with an explicit format name.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.hint.format = Some(format.into());
        self
    }

    /// Returns this source with frame sampling options.
    pub fn with_frame_sampling(mut self, sampling: FrameSampling) -> Self {
        self.hint.frame_sampling = Some(sampling);
        self
    }

    /// Returns this source with a remote read mode.
    pub fn with_remote_read_mode(mut self, mode: RemoteReadMode) -> Self {
        self.hint.remote_read_mode = Some(mode);
        self
    }

    /// Returns this source configured for buffered remote loading.
    pub fn buffered(mut self) -> Self {
        self.hint.remote_read_mode = Some(RemoteReadMode::Buffered);
        self
    }

    /// Returns this source configured for streaming remote loading.
    pub fn streaming(mut self) -> Self {
        self.hint.remote_read_mode = Some(RemoteReadMode::Streaming);
        self
    }

    /// Returns the explicit or inferred media type.
    pub fn media_type(&self) -> Option<MediaType> {
        self.hint
            .media_type
            .or_else(|| self.inferred_media_type_from_format())
            .or_else(|| self.inferred_media_type_from_location())
    }

    fn inferred_media_type_from_format(&self) -> Option<MediaType> {
        self.hint
            .format
            .as_deref()
            .and_then(infer_media_type_from_format)
    }

    fn inferred_media_type_from_location(&self) -> Option<MediaType> {
        match &self.location {
            MediaLocation::Path(path) => infer_media_type_from_path(path),
            MediaLocation::Bytes(bytes) => {
                image::guess_format(bytes).ok().map(|_| MediaType::Image)
            }
            MediaLocation::Url(url) => infer_media_type_from_url(url),
        }
    }
}

/// Timing metadata for a decoded frame.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FrameTiming {
    /// Zero-based frame index in the decoded stream.
    pub index: usize,
    /// Frame presentation timestamp in milliseconds, when available.
    pub timestamp_ms: Option<f64>,
    /// Frame display duration in milliseconds, when available.
    pub duration_ms: Option<f64>,
}

impl FrameTiming {
    /// Creates timing metadata for a frame index.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            timestamp_ms: None,
            duration_ms: None,
        }
    }
}

/// Owned decoded image frame data.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageFrame {
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    data: Vec<u8>,
    timing: FrameTiming,
}

impl ImageFrame {
    /// Creates an owned decoded image frame.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are zero, the frame size overflows
    /// `usize`, or the data length does not match the dimensions and format.
    pub fn new(
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self, MediaError> {
        if width == 0 || height == 0 {
            return Err(MediaError::InvalidFrameDimensions { width, height });
        }

        let expected = frame_len(width, height, pixel_format.channels())?;
        if data.len() != expected {
            return Err(MediaError::InvalidBufferLength {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            pixel_format,
            data,
            timing: FrameTiming::default(),
        })
    }

    /// Returns this frame with updated timing metadata.
    pub fn with_timing(mut self, timing: FrameTiming) -> Self {
        self.timing = timing;
        self
    }

    /// Returns the frame width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the frame height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the pixel storage format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Returns the raw interleaved pixel bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub(crate) fn into_parts(self) -> (usize, usize, PixelFormat, Vec<u8>, FrameTiming) {
        (
            self.width,
            self.height,
            self.pixel_format,
            self.data,
            self.timing,
        )
    }

    /// Returns frame timing metadata.
    pub fn timing(&self) -> FrameTiming {
        self.timing
    }

    /// Returns the number of channels per pixel.
    pub fn channels(&self) -> usize {
        self.pixel_format.channels()
    }

    /// Returns a zero-copy HWC tensor view over this frame's pixel bytes.
    ///
    /// The view uses [`ImageLayout::HeightWidthChannels`] because decoded
    /// frame storage is interleaved by pixel.
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal frame invariants are violated.
    pub fn as_tensor_view(&self) -> Result<TensorView<'_>, TensorError> {
        TensorView::image(
            TensorDataView::U8(self.data()),
            self.height(),
            self.width(),
            self.channels(),
            ImageLayout::HeightWidthChannels,
        )
    }

    /// Converts an `image` crate dynamic image into an owned frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the converted frame dimensions or buffer length are invalid.
    pub fn from_dynamic_image(image: DynamicImage) -> Result<Self, MediaError> {
        match image {
            DynamicImage::ImageLuma8(buffer) => Self::new(
                buffer.width() as usize,
                buffer.height() as usize,
                PixelFormat::Luma8,
                buffer.into_raw(),
            ),
            DynamicImage::ImageRgb8(buffer) => Self::new(
                buffer.width() as usize,
                buffer.height() as usize,
                PixelFormat::Rgb8,
                buffer.into_raw(),
            ),
            DynamicImage::ImageRgba8(buffer) => Self::new(
                buffer.width() as usize,
                buffer.height() as usize,
                PixelFormat::Rgba8,
                buffer.into_raw(),
            ),
            other => {
                let buffer = other.to_rgba8();
                Self::new(
                    buffer.width() as usize,
                    buffer.height() as usize,
                    PixelFormat::Rgba8,
                    buffer.into_raw(),
                )
            }
        }
    }
}

/// Loop behavior for an animated image sequence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum LoopBehavior {
    /// Play the sequence once.
    Once,
    /// Repeat the sequence indefinitely.
    Infinite,
    /// Repeat the sequence a finite number of times.
    Count(u32),
}

/// Owned decoded image frame sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSequence {
    frames: Vec<ImageFrame>,
    loop_behavior: LoopBehavior,
    duration_ms: Option<f64>,
}

impl ImageSequence {
    /// Creates an image sequence from decoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error when `frames` is empty.
    pub fn new(frames: Vec<ImageFrame>, loop_behavior: LoopBehavior) -> Result<Self, MediaError> {
        if frames.is_empty() {
            return Err(MediaError::EmptyFrameSequence);
        }

        let duration_ms = sequence_duration_ms(&frames);
        Ok(Self {
            frames,
            loop_behavior,
            duration_ms,
        })
    }

    /// Creates a single-frame image sequence.
    pub fn from_frame(frame: ImageFrame) -> Self {
        Self {
            frames: vec![frame],
            loop_behavior: LoopBehavior::Once,
            duration_ms: None,
        }
    }

    /// Returns true when the sequence has more than one frame.
    pub fn is_animated(&self) -> bool {
        self.frames.len() > 1
    }

    /// Returns the number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns true when the sequence contains no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns all decoded frames.
    pub fn frames(&self) -> &[ImageFrame] {
        &self.frames
    }

    /// Returns the sequence loop behavior.
    pub fn loop_behavior(&self) -> LoopBehavior {
        self.loop_behavior
    }

    /// Returns the total duration in milliseconds when all frame durations are known.
    pub fn duration_ms(&self) -> Option<f64> {
        self.duration_ms
    }

    /// Returns the first frame.
    ///
    /// # Panics
    ///
    /// Panics only if the internal non-empty sequence invariant is violated.
    pub fn first_frame(&self) -> &ImageFrame {
        &self.frames[0]
    }

    fn into_first_frame(self) -> ImageFrame {
        self.frames
            .into_iter()
            .next()
            .expect("ImageSequence must contain at least one frame")
    }

    /// Returns owned frames selected by `sampling`.
    ///
    /// # Errors
    ///
    /// Returns an error when sampling options are invalid or start out of range.
    pub fn sample_frames(&self, sampling: FrameSampling) -> Result<Vec<ImageFrame>, MediaError> {
        Ok(sampling
            .sample_indices(self.frames.len())?
            .into_iter()
            .map(|index| self.frames[index].clone())
            .collect())
    }

    /// Returns a new sequence containing frames selected by `sampling`.
    ///
    /// # Errors
    ///
    /// Returns an error when sampling options are invalid or produce no frames.
    pub fn sampled(&self, sampling: FrameSampling) -> Result<Self, MediaError> {
        Self::new(self.sample_frames(sampling)?, self.loop_behavior)
    }
}

/// Owned decoded video frame.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame {
    image: ImageFrame,
}

impl VideoFrame {
    /// Creates a video frame from an image frame.
    pub fn new(image: ImageFrame) -> Self {
        Self { image }
    }

    /// Returns the decoded image data for this video frame.
    pub fn image(&self) -> &ImageFrame {
        &self.image
    }

    /// Returns frame timing metadata.
    pub fn timing(&self) -> FrameTiming {
        self.image.timing
    }
}

/// Owned decoded video clip.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoClip {
    frames: Vec<VideoFrame>,
    fps: Option<f64>,
    duration_ms: Option<f64>,
}

impl VideoClip {
    /// Creates a video clip from decoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error when `frames` is empty.
    pub fn new(frames: Vec<VideoFrame>, fps: Option<f64>) -> Result<Self, MediaError> {
        if frames.is_empty() {
            return Err(MediaError::EmptyFrameSequence);
        }

        let duration_ms = frames.iter().try_fold(0.0, |total, frame| {
            frame.timing().duration_ms.map(|duration| total + duration)
        });
        Ok(Self {
            frames,
            fps,
            duration_ms,
        })
    }

    /// Returns the number of decoded frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns true when the clip contains no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns all decoded video frames.
    pub fn frames(&self) -> &[VideoFrame] {
        &self.frames
    }

    /// Returns frames per second when known.
    pub fn fps(&self) -> Option<f64> {
        self.fps
    }

    /// Returns duration in milliseconds when known.
    pub fn duration_ms(&self) -> Option<f64> {
        self.duration_ms
    }

    /// Returns owned frames selected by `sampling`.
    ///
    /// # Errors
    ///
    /// Returns an error when sampling options are invalid or start out of range.
    pub fn sample_frames(&self, sampling: FrameSampling) -> Result<Vec<VideoFrame>, MediaError> {
        Ok(sampling
            .sample_indices(self.frames.len())?
            .into_iter()
            .map(|index| self.frames[index].clone())
            .collect())
    }

    /// Returns a new clip containing frames selected by `sampling`.
    ///
    /// # Errors
    ///
    /// Returns an error when sampling options are invalid or produce no frames.
    pub fn sampled(&self, sampling: FrameSampling) -> Result<Self, MediaError> {
        Self::new(self.sample_frames(sampling)?, self.fps)
    }
}

/// Frame sampling parameters for decoded sequences and clips.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FrameSampling {
    /// First decoded frame index to consider.
    pub start_index: usize,
    /// Step between selected frame indices.
    pub stride: usize,
    /// Maximum number of frames to select.
    pub max_frames: Option<usize>,
}

impl Default for FrameSampling {
    fn default() -> Self {
        Self {
            start_index: 0,
            stride: 1,
            max_frames: None,
        }
    }
}

impl FrameSampling {
    /// Selects every frame.
    pub fn all() -> Self {
        Self::default()
    }

    /// Selects only the first frame.
    pub fn first() -> Self {
        Self {
            max_frames: Some(1),
            ..Self::default()
        }
    }

    /// Selects every `stride` frames.
    pub fn every(stride: usize) -> Self {
        Self {
            stride,
            ..Self::default()
        }
    }

    /// Returns this sampling configuration with a frame limit.
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = Some(max_frames);
        self
    }

    /// Returns whether `index` is accepted by this sampling configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when sampling options are invalid.
    pub fn accepts_index(self, index: usize) -> Result<bool, MediaError> {
        self.validate()?;
        Ok(index >= self.start_index && (index - self.start_index).is_multiple_of(self.stride))
    }

    /// Returns whether `selected` has reached the configured frame limit.
    pub fn reached_limit(self, selected: usize) -> bool {
        self.max_frames
            .is_some_and(|max_frames| selected >= max_frames)
    }

    /// Validates this sampling configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `stride` or `max_frames` is zero.
    pub fn validate(self) -> Result<(), MediaError> {
        if self.stride == 0 {
            return Err(MediaError::InvalidFrameStride);
        }
        if self.max_frames == Some(0) {
            return Err(MediaError::InvalidFrameLimit);
        }
        Ok(())
    }

    /// Returns selected frame indices for a sequence length.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid sampling, empty sequences, or an out-of-range start.
    pub fn sample_indices(self, len: usize) -> Result<Vec<usize>, MediaError> {
        self.validate()?;
        if len == 0 {
            return Err(MediaError::EmptyFrameSequence);
        }
        if self.start_index >= len {
            return Err(MediaError::FrameStartOutOfRange {
                start: self.start_index,
                len,
            });
        }

        let indices = (self.start_index..len).step_by(self.stride);
        Ok(match self.max_frames {
            Some(max_frames) => indices.take(max_frames).collect(),
            None => indices.collect(),
        })
    }
}

/// Options controlling video decoding.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VideoDecodeOptions {
    /// Frame sampling applied during decode.
    pub sampling: FrameSampling,
    /// Remote read mode used for URL video sources.
    pub remote_read_mode: RemoteReadMode,
}

impl Default for VideoDecodeOptions {
    fn default() -> Self {
        Self {
            sampling: FrameSampling::first(),
            remote_read_mode: RemoteReadMode::Streaming,
        }
    }
}

impl VideoDecodeOptions {
    /// Returns these options with updated frame sampling.
    pub fn with_frame_sampling(mut self, sampling: FrameSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Returns these options with an updated remote read mode.
    pub fn with_remote_read_mode(mut self, mode: RemoteReadMode) -> Self {
        self.remote_read_mode = mode;
        self
    }
}

/// Strategy for loading remote media.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum RemoteReadMode {
    /// Download the remote body before decoding.
    Buffered,
    /// Stream directly into the decoder when supported.
    Streaming,
}

/// Redirect policy used by remote URL loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteRedirectPolicy {
    /// Do not follow redirects.
    None,
    /// Follow redirects up to the configured limit.
    FollowLimited(usize),
}

impl Default for RemoteRedirectPolicy {
    fn default() -> Self {
        Self::FollowLimited(10)
    }
}

/// Options for bounded remote media loading.
#[derive(Clone, Debug)]
pub struct RemoteLoadOptions {
    timeout: Duration,
    max_bytes: u64,
    user_agent: Option<String>,
    redirect_policy: RemoteRedirectPolicy,
    #[cfg(feature = "url")]
    client: Option<reqwest::blocking::Client>,
}

impl Default for RemoteLoadOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_bytes: 128 * 1024 * 1024,
            user_agent: Some(format!("image-processors/{}", env!("CARGO_PKG_VERSION"))),
            redirect_policy: RemoteRedirectPolicy::default(),
            #[cfg(feature = "url")]
            client: None,
        }
    }
}

impl RemoteLoadOptions {
    /// Returns the request timeout.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the maximum number of response bytes to read.
    #[must_use]
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Returns the configured user agent, if any.
    #[must_use]
    pub fn user_agent(&self) -> Option<&str> {
        self.user_agent.as_deref()
    }

    /// Returns the redirect policy.
    #[must_use]
    pub fn redirect_policy(&self) -> RemoteRedirectPolicy {
        self.redirect_policy
    }

    /// Returns these options with an updated timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Returns these options with an updated byte limit.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Returns these options with an updated user agent.
    #[must_use]
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Returns these options without sending a user agent.
    #[must_use]
    pub fn without_user_agent(mut self) -> Self {
        self.user_agent = None;
        self
    }

    /// Returns these options with an updated redirect policy.
    #[must_use]
    pub fn with_redirect_policy(mut self, redirect_policy: RemoteRedirectPolicy) -> Self {
        self.redirect_policy = redirect_policy;
        self
    }

    /// Returns the configured reqwest client, if any.
    #[cfg(feature = "url")]
    #[must_use]
    pub fn client(&self) -> Option<&reqwest::blocking::Client> {
        self.client.as_ref()
    }

    /// Returns these options with an explicit reqwest client.
    #[cfg(feature = "url")]
    #[must_use]
    pub fn with_client(mut self, client: reqwest::blocking::Client) -> Self {
        self.client = Some(client);
        self
    }
}

/// Media loaded by a [`MediaLoader`].
#[derive(Clone, Debug, PartialEq)]
pub enum LoadedMedia {
    /// A single decoded image frame.
    Image(ImageFrame),
    /// A decoded animated image sequence.
    ImageSequence(ImageSequence),
    /// A decoded video clip.
    Video(VideoClip),
}

impl LoadedMedia {
    fn from_image_sequence(sequence: ImageSequence) -> Self {
        if sequence.is_animated() {
            Self::ImageSequence(sequence)
        } else {
            Self::Image(sequence.into_first_frame())
        }
    }
}

/// Loads media from a source descriptor.
pub trait MediaLoader {
    /// Loads the requested media source.
    ///
    /// # Errors
    ///
    /// Returns an error when source loading, decoding, or sampling fails.
    fn load(&self, source: MediaSource) -> Result<LoadedMedia, MediaError>;
}

/// Default filesystem, byte, URL, and video media loader.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultMediaLoader;

impl DefaultMediaLoader {
    /// Loads media with explicit remote loading options.
    ///
    /// # Errors
    ///
    /// Returns an error when source loading, decoding, or sampling fails.
    pub fn load_with_remote_options(
        &self,
        source: MediaSource,
        remote_options: &RemoteLoadOptions,
    ) -> Result<LoadedMedia, MediaError> {
        self.load_with_remote_options_and_image_decode_backend(
            source,
            remote_options,
            ImageDecodeBackend::ImageCrate,
        )
    }

    /// Loads media with explicit remote loading options and image decode backend.
    ///
    /// # Errors
    ///
    /// Returns an error when source loading, decoding, or sampling fails.
    pub fn load_with_remote_options_and_image_decode_backend(
        &self,
        source: MediaSource,
        remote_options: &RemoteLoadOptions,
        image_decode_backend: ImageDecodeBackend,
    ) -> Result<LoadedMedia, MediaError> {
        let media_type = source.media_type();
        let frame_sampling = source.hint.frame_sampling;

        match media_type {
            Some(MediaType::Video) => match source.location {
                MediaLocation::Path(path) => load_video_from_path_with_options(
                    path,
                    video_options_from_hint(frame_sampling, source.hint.remote_read_mode),
                )
                .map(LoadedMedia::Video),
                MediaLocation::Bytes(_) => Err(MediaError::VideoDecodingUnavailable),
                MediaLocation::Url(url) => load_video_from_url_with_remote_options(
                    &url,
                    video_options_from_hint(frame_sampling, source.hint.remote_read_mode),
                    remote_options,
                )
                .map(LoadedMedia::Video),
            },
            Some(MediaType::Image) | None => match source.location {
                MediaLocation::Path(path) => {
                    load_image_sequence_from_path_with_backend(path, image_decode_backend)
                        .and_then(|sequence| apply_sequence_sampling(sequence, frame_sampling))
                        .map(LoadedMedia::from_image_sequence)
                }
                MediaLocation::Bytes(bytes) => {
                    { decode_image_sequence_bytes_with_backend(&bytes, image_decode_backend) }
                        .and_then(|sequence| apply_sequence_sampling(sequence, frame_sampling))
                        .map(LoadedMedia::from_image_sequence)
                }
                MediaLocation::Url(url) => {
                    load_image_sequence_from_url_with_remote_options_and_image_decode_backend(
                        &url,
                        frame_sampling,
                        image_remote_read_mode(source.hint.remote_read_mode),
                        remote_options,
                        image_decode_backend,
                    )
                }
                .map(LoadedMedia::from_image_sequence),
            },
        }
    }
}

impl MediaLoader for DefaultMediaLoader {
    fn load(&self, source: MediaSource) -> Result<LoadedMedia, MediaError> {
        self.load_with_remote_options(source, &RemoteLoadOptions::default())
    }
}

/// Errors returned by media loading and decoding.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MediaError {
    /// Frame dimensions were zero.
    #[error("frame dimensions must be positive, got width={width}, height={height}")]
    InvalidFrameDimensions {
        /// Frame width in pixels.
        width: usize,
        /// Frame height in pixels.
        height: usize,
    },
    /// Frame data length did not match dimensions and pixel format.
    #[error("invalid buffer length: expected {expected} bytes, got {actual}")]
    InvalidBufferLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Frame dimensions and channel count overflowed `usize`.
    #[error("frame size overflows usize")]
    FrameSizeOverflow,
    /// A decoded frame sequence was empty.
    #[error("frame sequence cannot be empty")]
    EmptyFrameSequence,
    /// Frame sampling stride was zero.
    #[error("frame sampling stride must be positive")]
    InvalidFrameStride,
    /// Frame sampling limit was zero.
    #[error("frame sampling max_frames must be positive when configured")]
    InvalidFrameLimit,
    /// Frame sampling started past the available frames.
    #[error("frame sampling start index {start} is out of range for {len} frames")]
    FrameStartOutOfRange {
        /// Requested start index.
        start: usize,
        /// Available frame count.
        len: usize,
    },
    /// Remote loading was requested in a build without URL support.
    #[error("remote media loading is not available for URL: {0}")]
    RemoteLoadingUnavailable(String),
    /// Remote streaming is not supported for the media type.
    #[error("remote streaming is not available for {0:?} media")]
    RemoteStreamingUnavailable(MediaType),
    /// Remote buffering is not supported for the media type.
    #[error("remote buffering is not available for {0:?} media")]
    RemoteBufferingUnavailable(MediaType),
    /// Remote response exceeded the configured byte limit.
    #[error("remote response body exceeds configured limit of {limit} bytes")]
    RemoteBodyTooLarge {
        /// Configured byte limit.
        limit: u64,
    },
    /// Video URL parsing failed.
    #[error("invalid video URL: {0}")]
    InvalidVideoUrl(String),
    /// Decoded video frame data was not contiguous.
    #[error("decoded video frame data is not contiguous")]
    VideoFrameNotContiguous,
    /// Image decoding failed.
    #[error("image decoding failed: {0}")]
    Image(#[from] image::ImageError),
    /// Image file I/O failed.
    #[error("image file I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// TurboJPEG decoding was requested in a build without TurboJPEG support.
    #[error("turbojpeg image decoding is not available in this build")]
    TurboJpegUnavailable,
    /// TurboJPEG decoding failed.
    #[error("turbojpeg image decoding failed: {message}")]
    TurboJpeg {
        /// Backend error message.
        message: String,
    },
    /// URL fetching failed.
    #[cfg(feature = "url")]
    #[error("URL fetch failed: {0}")]
    Url(#[from] reqwest::Error),
    /// Video decoding failed.
    #[cfg(feature = "video")]
    #[error("video decoding failed: {0}")]
    Video(#[from] video_rs::Error),
    /// Video backend initialization failed.
    #[cfg(feature = "video")]
    #[error("video backend initialization failed: {0}")]
    VideoInit(String),
    /// Video decoding was requested in a build without video support.
    #[error("video decoding is not available in this build")]
    VideoDecodingUnavailable,
}

/// Loads the first image frame from a filesystem path.
///
/// # Errors
///
/// Returns an error when file I/O, format detection, or image decoding fails.
pub fn load_image_from_path(path: impl AsRef<Path>) -> Result<ImageFrame, MediaError> {
    load_image_sequence_from_path(path).map(ImageSequence::into_first_frame)
}

/// Loads the first image frame from a filesystem path with an explicit image decode backend.
///
/// # Errors
///
/// Returns an error when file I/O, format detection, or image decoding fails.
pub fn load_image_from_path_with_backend(
    path: impl AsRef<Path>,
    backend: ImageDecodeBackend,
) -> Result<ImageFrame, MediaError> {
    load_image_sequence_from_path_with_backend(path, backend).map(ImageSequence::into_first_frame)
}

/// Loads an image frame sequence from a filesystem path.
///
/// # Errors
///
/// Returns an error when file I/O, format detection, or image decoding fails.
pub fn load_image_sequence_from_path(path: impl AsRef<Path>) -> Result<ImageSequence, MediaError> {
    load_image_sequence_from_path_with_backend(path, ImageDecodeBackend::ImageCrate)
}

/// Loads an image frame sequence from a filesystem path with an explicit image decode backend.
///
/// # Errors
///
/// Returns an error when file I/O, format detection, or image decoding fails.
pub fn load_image_sequence_from_path_with_backend(
    path: impl AsRef<Path>,
    backend: ImageDecodeBackend,
) -> Result<ImageSequence, MediaError> {
    let path = path.as_ref();
    match image_format_from_path(path) {
        Some(ImageFormat::Gif) => decode_gif_sequence(BufReader::new(File::open(path)?)),
        Some(ImageFormat::Png) => decode_png_sequence_or_static_path(path),
        Some(ImageFormat::WebP) => decode_webp_sequence(BufReader::new(File::open(path)?)),
        Some(ImageFormat::Jpeg) if backend == ImageDecodeBackend::TurboJpeg => {
            decode_jpeg_path_turbo(path).map(ImageSequence::from_frame)
        }
        _ => {
            let image = image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?;
            Ok(ImageSequence::from_frame(ImageFrame::from_dynamic_image(
                image,
            )?))
        }
    }
}

/// Decodes the first image frame from encoded bytes.
///
/// # Errors
///
/// Returns an error when format detection or image decoding fails.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<ImageFrame, MediaError> {
    decode_image_sequence_bytes(bytes).map(ImageSequence::into_first_frame)
}

/// Decodes the first image frame from encoded bytes with an explicit image decode backend.
///
/// # Errors
///
/// Returns an error when format detection or image decoding fails.
pub fn decode_image_bytes_with_backend(
    bytes: &[u8],
    backend: ImageDecodeBackend,
) -> Result<ImageFrame, MediaError> {
    decode_image_sequence_bytes_with_backend(bytes, backend).map(ImageSequence::into_first_frame)
}

/// Decodes an image frame sequence from encoded bytes.
///
/// # Errors
///
/// Returns an error when format detection or image decoding fails.
pub fn decode_image_sequence_bytes(bytes: &[u8]) -> Result<ImageSequence, MediaError> {
    decode_image_sequence_bytes_with_backend(bytes, ImageDecodeBackend::ImageCrate)
}

/// Decodes an image frame sequence from encoded bytes with an explicit image decode backend.
///
/// # Errors
///
/// Returns an error when format detection or image decoding fails.
pub fn decode_image_sequence_bytes_with_backend(
    bytes: &[u8],
    backend: ImageDecodeBackend,
) -> Result<ImageSequence, MediaError> {
    match image::guess_format(bytes).ok() {
        Some(ImageFormat::Gif) => decode_gif_sequence(Cursor::new(bytes)),
        Some(ImageFormat::Png) => decode_png_sequence_or_static_bytes(bytes),
        Some(ImageFormat::WebP) => decode_webp_sequence(Cursor::new(bytes)),
        Some(ImageFormat::Jpeg) if backend == ImageDecodeBackend::TurboJpeg => {
            decode_jpeg_bytes_turbo(bytes).map(ImageSequence::from_frame)
        }
        _ => Ok(ImageSequence::from_frame(ImageFrame::from_dynamic_image(
            image::load_from_memory(bytes)?,
        )?)),
    }
}

/// Loads a video clip from a filesystem path with default options.
///
/// # Errors
///
/// Returns an error when video decoding is unavailable or decoding fails.
pub fn load_video_from_path(_path: impl AsRef<Path>) -> Result<VideoClip, MediaError> {
    load_video_from_path_with_options(_path, VideoDecodeOptions::default())
}

#[cfg(feature = "url")]
/// Loads an image sequence from a URL in buffered mode.
///
/// # Errors
///
/// Returns an error when fetching, byte limits, image decoding, or sampling fails.
pub fn load_image_sequence_from_url(
    url: &str,
    sampling: Option<FrameSampling>,
) -> Result<ImageSequence, MediaError> {
    load_image_sequence_from_url_with_mode(url, sampling, RemoteReadMode::Buffered)
}

#[cfg(feature = "url")]
/// Loads an image sequence from a URL with an explicit remote read mode.
///
/// # Errors
///
/// Returns an error when fetching, byte limits, image decoding, or sampling fails.
pub fn load_image_sequence_from_url_with_mode(
    url: &str,
    sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
) -> Result<ImageSequence, MediaError> {
    load_image_sequence_from_url_with_remote_options(
        url,
        sampling,
        mode,
        &RemoteLoadOptions::default(),
    )
}

#[cfg(feature = "url")]
/// Loads an image sequence from a URL with explicit remote options.
///
/// # Errors
///
/// Returns an error when fetching, byte limits, image decoding, or sampling fails.
pub fn load_image_sequence_from_url_with_remote_options(
    url: &str,
    sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
    remote_options: &RemoteLoadOptions,
) -> Result<ImageSequence, MediaError> {
    load_image_sequence_from_url_with_remote_options_and_image_decode_backend(
        url,
        sampling,
        mode,
        remote_options,
        ImageDecodeBackend::ImageCrate,
    )
}

#[cfg(feature = "url")]
/// Loads an image sequence from a URL with explicit remote options and image decode backend.
///
/// # Errors
///
/// Returns an error when fetching, byte limits, image decoding, or sampling fails.
pub fn load_image_sequence_from_url_with_remote_options_and_image_decode_backend(
    url: &str,
    sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
    remote_options: &RemoteLoadOptions,
    image_decode_backend: ImageDecodeBackend,
) -> Result<ImageSequence, MediaError> {
    if mode == RemoteReadMode::Streaming {
        return Err(MediaError::RemoteStreamingUnavailable(MediaType::Image));
    }

    let bytes = download_url(url, remote_options)?;
    decode_image_sequence_bytes_with_backend(&bytes, image_decode_backend)
        .and_then(|sequence| apply_sequence_sampling(sequence, sampling))
}

#[cfg(not(feature = "url"))]
/// Reports that image URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns [`MediaError::RemoteLoadingUnavailable`] unless streaming was requested.
pub fn load_image_sequence_from_url(
    url: &str,
    _sampling: Option<FrameSampling>,
) -> Result<ImageSequence, MediaError> {
    Err(MediaError::RemoteLoadingUnavailable(url.to_owned()))
}

#[cfg(not(feature = "url"))]
/// Reports that image URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns a remote-loading or remote-streaming error.
pub fn load_image_sequence_from_url_with_mode(
    url: &str,
    _sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
) -> Result<ImageSequence, MediaError> {
    if mode == RemoteReadMode::Streaming {
        Err(MediaError::RemoteStreamingUnavailable(MediaType::Image))
    } else {
        Err(MediaError::RemoteLoadingUnavailable(url.to_owned()))
    }
}

#[cfg(not(feature = "url"))]
/// Reports that image URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns a remote-loading or remote-streaming error.
pub fn load_image_sequence_from_url_with_remote_options(
    url: &str,
    _sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
    _remote_options: &RemoteLoadOptions,
) -> Result<ImageSequence, MediaError> {
    load_image_sequence_from_url_with_remote_options_and_image_decode_backend(
        url,
        _sampling,
        mode,
        _remote_options,
        ImageDecodeBackend::ImageCrate,
    )
}

#[cfg(not(feature = "url"))]
/// Reports that image URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns a remote-loading or remote-streaming error.
pub fn load_image_sequence_from_url_with_remote_options_and_image_decode_backend(
    url: &str,
    _sampling: Option<FrameSampling>,
    mode: RemoteReadMode,
    _remote_options: &RemoteLoadOptions,
    _image_decode_backend: ImageDecodeBackend,
) -> Result<ImageSequence, MediaError> {
    if mode == RemoteReadMode::Streaming {
        Err(MediaError::RemoteStreamingUnavailable(MediaType::Image))
    } else {
        Err(MediaError::RemoteLoadingUnavailable(url.to_owned()))
    }
}

#[cfg(feature = "video")]
/// Loads a video clip from a filesystem path with explicit options.
///
/// # Errors
///
/// Returns an error when backend initialization, decoding, or sampling fails.
pub fn load_video_from_path_with_options(
    path: impl AsRef<Path>,
    options: VideoDecodeOptions,
) -> Result<VideoClip, MediaError> {
    init_video_backend()?;
    decode_video_with_options(video_rs::decode::Decoder::new(path.as_ref())?, options)
}

#[cfg(not(feature = "video"))]
/// Reports that video path loading is unavailable in this build.
///
/// # Errors
///
/// Always returns [`MediaError::VideoDecodingUnavailable`].
pub fn load_video_from_path_with_options(
    _path: impl AsRef<Path>,
    _options: VideoDecodeOptions,
) -> Result<VideoClip, MediaError> {
    Err(MediaError::VideoDecodingUnavailable)
}

#[cfg(feature = "video")]
/// Loads a video clip from a URL with default remote options.
///
/// # Errors
///
/// Returns an error when backend initialization, URL parsing, fetching, decoding, or sampling fails.
pub fn load_video_from_url_with_options(
    url: &str,
    options: VideoDecodeOptions,
) -> Result<VideoClip, MediaError> {
    load_video_from_url_with_remote_options(url, options, &RemoteLoadOptions::default())
}

#[cfg(feature = "video")]
/// Loads a video clip from a URL with explicit remote options.
///
/// # Errors
///
/// Returns an error when backend initialization, URL parsing, fetching, decoding, or sampling fails.
pub fn load_video_from_url_with_remote_options(
    url: &str,
    options: VideoDecodeOptions,
    remote_options: &RemoteLoadOptions,
) -> Result<VideoClip, MediaError> {
    init_video_backend()?;
    match options.remote_read_mode {
        RemoteReadMode::Streaming => {
            let url = url
                .parse::<video_rs::Url>()
                .map_err(|_| MediaError::InvalidVideoUrl(url.to_owned()))?;
            decode_video_with_options(video_rs::decode::Decoder::new(url)?, options)
        }
        RemoteReadMode::Buffered => load_buffered_video_url(url, options, remote_options),
    }
}

#[cfg(not(feature = "video"))]
/// Reports that video URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns [`MediaError::VideoDecodingUnavailable`].
pub fn load_video_from_url_with_options(
    _url: &str,
    _options: VideoDecodeOptions,
) -> Result<VideoClip, MediaError> {
    Err(MediaError::VideoDecodingUnavailable)
}

#[cfg(not(feature = "video"))]
/// Reports that video URL loading is unavailable in this build.
///
/// # Errors
///
/// Always returns [`MediaError::VideoDecodingUnavailable`].
pub fn load_video_from_url_with_remote_options(
    _url: &str,
    _options: VideoDecodeOptions,
    _remote_options: &RemoteLoadOptions,
) -> Result<VideoClip, MediaError> {
    Err(MediaError::VideoDecodingUnavailable)
}

fn decode_jpeg_path_turbo(path: &Path) -> Result<ImageFrame, MediaError> {
    let bytes = std::fs::read(path)?;
    decode_jpeg_bytes_turbo(&bytes)
}

#[cfg(feature = "turbojpeg")]
fn decode_jpeg_bytes_turbo(bytes: &[u8]) -> Result<ImageFrame, MediaError> {
    let image = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)
        .map_err(turbojpeg_decode_error)?;
    ImageFrame::new(image.width, image.height, PixelFormat::Rgb8, image.pixels)
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg_bytes_turbo(_bytes: &[u8]) -> Result<ImageFrame, MediaError> {
    Err(MediaError::TurboJpegUnavailable)
}

#[cfg(feature = "turbojpeg")]
fn turbojpeg_decode_error(error: turbojpeg::Error) -> MediaError {
    MediaError::TurboJpeg {
        message: error.to_string(),
    }
}

fn apply_sequence_sampling(
    sequence: ImageSequence,
    sampling: Option<FrameSampling>,
) -> Result<ImageSequence, MediaError> {
    match sampling {
        Some(sampling) => sequence.sampled(sampling),
        None => Ok(sequence),
    }
}

fn video_options_from_hint(
    sampling: Option<FrameSampling>,
    remote_read_mode: Option<RemoteReadMode>,
) -> VideoDecodeOptions {
    let mut options = VideoDecodeOptions::default();
    if let Some(sampling) = sampling {
        options.sampling = sampling;
    }
    if let Some(remote_read_mode) = remote_read_mode {
        options.remote_read_mode = remote_read_mode;
    }
    options
}

fn image_remote_read_mode(remote_read_mode: Option<RemoteReadMode>) -> RemoteReadMode {
    remote_read_mode.unwrap_or(RemoteReadMode::Buffered)
}

#[cfg(feature = "url")]
fn download_url(url: &str, options: &RemoteLoadOptions) -> Result<Vec<u8>, MediaError> {
    let response = if let Some(client) = options.client.as_ref() {
        client.get(url).send()?
    } else {
        build_remote_client(options)?.get(url).send()?
    }
    .error_for_status()?;

    if response
        .content_length()
        .is_some_and(|length| length > options.max_bytes)
    {
        return Err(MediaError::RemoteBodyTooLarge {
            limit: options.max_bytes,
        });
    }

    read_limited_response(response, options.max_bytes)
}

#[cfg(feature = "url")]
fn build_remote_client(
    options: &RemoteLoadOptions,
) -> Result<reqwest::blocking::Client, MediaError> {
    let mut builder = reqwest::blocking::Client::builder()
        .timeout(options.timeout)
        .redirect(match options.redirect_policy {
            RemoteRedirectPolicy::None => reqwest::redirect::Policy::none(),
            RemoteRedirectPolicy::FollowLimited(limit) => reqwest::redirect::Policy::limited(limit),
        });

    if let Some(user_agent) = &options.user_agent {
        builder = builder.user_agent(user_agent.clone());
    }

    Ok(builder.build()?)
}

#[cfg(feature = "url")]
fn read_limited_response(
    response: reqwest::blocking::Response,
    max_bytes: u64,
) -> Result<Vec<u8>, MediaError> {
    let read_limit = max_bytes.saturating_add(1);
    let mut reader = response.take(read_limit);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        Err(MediaError::RemoteBodyTooLarge { limit: max_bytes })
    } else {
        Ok(bytes)
    }
}

#[cfg(feature = "video")]
fn init_video_backend() -> Result<(), MediaError> {
    video_rs::init().map_err(|error| MediaError::VideoInit(error.to_string()))
}

#[cfg(all(feature = "video", feature = "url"))]
fn load_buffered_video_url(
    url: &str,
    options: VideoDecodeOptions,
    remote_options: &RemoteLoadOptions,
) -> Result<VideoClip, MediaError> {
    let bytes = download_url(url, remote_options)?;
    let suffix = temporary_video_suffix(url);
    let mut file = tempfile::Builder::new()
        .prefix("image-processors-")
        .suffix(&suffix)
        .tempfile()?;
    file.write_all(&bytes)?;
    file.flush()?;
    load_video_from_path_with_options(file.path(), options)
}

#[cfg(all(feature = "video", not(feature = "url")))]
fn load_buffered_video_url(
    _url: &str,
    _options: VideoDecodeOptions,
    _remote_options: &RemoteLoadOptions,
) -> Result<VideoClip, MediaError> {
    Err(MediaError::RemoteBufferingUnavailable(MediaType::Video))
}

#[cfg(all(feature = "video", feature = "url"))]
fn temporary_video_suffix(url: &str) -> String {
    let extension = url
        .split(['?', '#'])
        .next()
        .and_then(|path| path.rsplit_once('.'))
        .map(|(_, extension)| extension)
        .filter(|extension| {
            !extension.is_empty()
                && extension.len() <= 8
                && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
        .unwrap_or("video");
    format!(".{extension}")
}

fn decode_png_sequence_or_static_path(path: &Path) -> Result<ImageSequence, MediaError> {
    match decode_png_sequence(BufReader::new(File::open(path)?))? {
        Some(sequence) => Ok(sequence),
        None => {
            let image = image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?;
            Ok(ImageSequence::from_frame(ImageFrame::from_dynamic_image(
                image,
            )?))
        }
    }
}

fn decode_png_sequence_or_static_bytes(bytes: &[u8]) -> Result<ImageSequence, MediaError> {
    match decode_png_sequence(Cursor::new(bytes))? {
        Some(sequence) => Ok(sequence),
        None => Ok(ImageSequence::from_frame(ImageFrame::from_dynamic_image(
            image::load_from_memory(bytes)?,
        )?)),
    }
}

fn decode_gif_sequence<R>(reader: R) -> Result<ImageSequence, MediaError>
where
    R: BufRead + Seek,
{
    let decoder = GifDecoder::new(reader)?;
    let loop_behavior = loop_behavior_from_image(decoder.loop_count());
    let frames = decoder.into_frames().collect_frames()?;
    animation_frames_to_sequence(frames, loop_behavior)
}

fn decode_png_sequence<R>(reader: R) -> Result<Option<ImageSequence>, MediaError>
where
    R: BufRead + Seek,
{
    let decoder = PngDecoder::new(reader)?.apng()?;
    let loop_behavior = loop_behavior_from_image(decoder.loop_count());
    let frames = decoder.into_frames().collect_frames()?;
    if frames.is_empty() {
        Ok(None)
    } else {
        animation_frames_to_sequence(frames, loop_behavior).map(Some)
    }
}

fn decode_webp_sequence<R>(reader: R) -> Result<ImageSequence, MediaError>
where
    R: BufRead + Seek,
{
    let decoder = WebPDecoder::new(reader)?;
    let loop_behavior = loop_behavior_from_image(decoder.loop_count());
    let frames = decoder.into_frames().collect_frames()?;
    animation_frames_to_sequence(frames, loop_behavior)
}

fn animation_frames_to_sequence(
    frames: Vec<::image::Frame>,
    loop_behavior: LoopBehavior,
) -> Result<ImageSequence, MediaError> {
    let mut timestamp_ms = 0.0;
    let frames = frames
        .into_iter()
        .enumerate()
        .map(|(index, frame)| {
            let duration_ms = frame_duration_ms(frame.delay());
            let buffer = frame.into_buffer();
            let image = ImageFrame::new(
                buffer.width() as usize,
                buffer.height() as usize,
                PixelFormat::Rgba8,
                buffer.into_raw(),
            )?
            .with_timing(FrameTiming {
                index,
                timestamp_ms: Some(timestamp_ms),
                duration_ms,
            });
            if let Some(duration_ms) = duration_ms {
                timestamp_ms += duration_ms;
            }
            Ok(image)
        })
        .collect::<Result<Vec<_>, MediaError>>()?;

    ImageSequence::new(frames, loop_behavior)
}

#[cfg(feature = "video")]
fn decode_video_with_options(
    mut decoder: video_rs::decode::Decoder,
    options: VideoDecodeOptions,
) -> Result<VideoClip, MediaError> {
    options.sampling.validate()?;

    let fps = positive_f64(decoder.frame_rate() as f64);
    let duration_ms = decoder
        .duration()
        .ok()
        .map(|duration| duration.as_secs_f64() * 1000.0);

    let mut frames = Vec::new();
    for (decoded_index, result) in decoder.decode_iter().enumerate() {
        let (timestamp, frame) = match result {
            Ok(frame) => frame,
            Err(video_rs::Error::DecodeExhausted | video_rs::Error::ReadExhausted) => break,
            Err(error) => return Err(MediaError::Video(error)),
        };

        if !options.sampling.accepts_index(decoded_index)? {
            continue;
        }

        let (height, width, channels) = frame.dim();
        if channels != PixelFormat::Rgb8.channels() {
            return Err(MediaError::InvalidBufferLength {
                expected: width * height * PixelFormat::Rgb8.channels(),
                actual: width * height * channels,
            });
        }

        let data = frame
            .as_slice()
            .ok_or(MediaError::VideoFrameNotContiguous)?
            .to_vec();
        let image =
            ImageFrame::new(width, height, PixelFormat::Rgb8, data)?.with_timing(FrameTiming {
                index: decoded_index,
                timestamp_ms: Some(timestamp.as_secs_f64() * 1000.0),
                duration_ms: fps.map(|fps| 1000.0 / fps),
            });
        frames.push(VideoFrame::new(image));

        if options.sampling.reached_limit(frames.len()) {
            break;
        }
    }

    if frames.is_empty() {
        return Err(MediaError::EmptyFrameSequence);
    }

    let mut clip = VideoClip::new(frames, fps)?;
    clip.duration_ms = duration_ms.or(clip.duration_ms);
    Ok(clip)
}

fn frame_duration_ms(delay: ::image::Delay) -> Option<f64> {
    let (numerator, denominator) = delay.numer_denom_ms();
    if numerator == 0 || denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn loop_behavior_from_image(loop_count: LoopCount) -> LoopBehavior {
    match loop_count {
        LoopCount::Infinite => LoopBehavior::Infinite,
        LoopCount::Finite(count) if count.get() == 1 => LoopBehavior::Once,
        LoopCount::Finite(count) => LoopBehavior::Count(count.get()),
    }
}

fn sequence_duration_ms(frames: &[ImageFrame]) -> Option<f64> {
    frames.iter().try_fold(0.0, |total, frame| {
        frame.timing.duration_ms.map(|duration| total + duration)
    })
}

#[cfg(feature = "video")]
fn positive_f64(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn frame_len(width: usize, height: usize, channels: usize) -> Result<usize, MediaError> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(MediaError::FrameSizeOverflow)
}

fn infer_media_type_from_path(path: &Path) -> Option<MediaType> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .and_then(infer_media_type_from_format)
}

fn infer_media_type_from_url(url: &str) -> Option<MediaType> {
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .trim_end_matches('/');
    path.rsplit_once('.')
        .map(|(_, extension)| extension)
        .and_then(infer_media_type_from_format)
}

fn infer_media_type_from_format(format: &str) -> Option<MediaType> {
    let format = format.trim_start_matches('.').to_ascii_lowercase();
    if image_format_from_name(&format).is_some() {
        return Some(MediaType::Image);
    }
    if matches!(
        format.as_str(),
        "avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "webm"
    ) {
        return Some(MediaType::Video);
    }
    None
}

fn image_format_from_path(path: &Path) -> Option<ImageFormat> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .and_then(image_format_from_name)
}

fn image_format_from_name(format: &str) -> Option<ImageFormat> {
    ImageFormat::from_extension(format.trim_start_matches('.'))
}

#[cfg(test)]
mod tests;
