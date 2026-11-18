//! Synchronous bounded raw-frame processing with external FFmpeg tools.

use std::{
    ffi::{OsStr, OsString},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
};

use super::{
    FrameSampling, FrameTiming, ImageFrame, MediaError, MediaLocation, MediaSource, PixelFormat,
    VideoClip, VideoFrame,
};

/// External video tools and an optional output frame-rate override.
#[derive(Debug, Clone)]
pub struct VideoStreamOptions {
    /// FFmpeg executable for raw RGB decoding and H.264/audio-copy encoding.
    pub ffmpeg: PathBuf,
    /// FFprobe executable for source dimensions and average frame rate.
    pub ffprobe: PathBuf,
    /// Output/decoded-clip metadata FPS, or source average FPS when absent.
    pub fps: Option<f64>,
}
impl Default for VideoStreamOptions {
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            fps: None,
        }
    }
}

/// Filters applied by FFmpeg before RGB frames are delivered to the caller.
/// The default retains source dimensions and cadence, independently of the
/// metadata-only FPS override in [`VideoStreamOptions`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RgbVideoFilter {
    /// Output width and height. Uses FFmpeg's default `scale` filter settings.
    pub size: Option<(usize, usize)>,
    /// Actual frame-rate conversion with FFmpeg's `fps` filter, before scaling.
    pub fps: Option<f64>,
}

impl RgbVideoFilter {
    fn expression(self) -> Result<Option<String>, MediaStreamError> {
        let mut filters = Vec::new();
        if let Some(fps) = self.fps {
            if !fps.is_finite() || fps <= 0.0 {
                return Err(invalid("RGB filter frame rate must be finite and positive"));
            }
            filters.push(format!("fps=fps={fps}"));
        }
        if let Some((width, height)) = self.size {
            if width == 0 || height == 0 {
                return Err(invalid("RGB filter dimensions must be positive"));
            }
            filters.push(format!("scale={width}:{height}"));
        }
        Ok((!filters.is_empty()).then(|| filters.join(",")))
    }
}

/// Frame-stream setup, media contract or child-process failure.
#[derive(Debug, thiserror::Error)]
pub enum MediaStreamError {
    /// Invalid media or unsuccessful external tool.
    #[error("{0}")]
    Invalid(String),
    /// Filesystem or pipe failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Invalid FFprobe JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Invalid decoded-frame storage.
    #[error(transparent)]
    Media(#[from] MediaError),
}

struct Process {
    child: Child,
    waited: bool,
}
impl Process {
    fn new(child: Child) -> Self {
        Self {
            child,
            waited: false,
        }
    }
    fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.waited = true;
        Ok(status)
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        // Reap only children created by this call, including callback failures.
        if !self.waited {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
fn invalid(message: impl Into<String>) -> MediaStreamError {
    MediaStreamError::Invalid(message.into())
}

fn tool_command(program: &Path) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut command = Command::new(program);
        command.creation_flags(0x0800_0000);
        command
    }
    #[cfg(not(windows))]
    {
        Command::new(program)
    }
}

/// Probe a video and stream each RGB frame through a caller-owned transform.
///
/// `output_size` receives source width/height before encoding starts. The frame
/// callback must return RGB8 at those output dimensions. Output is H.264/yuv420p;
/// audio streams are copied from the source. Source average FPS is preserved
/// unless explicitly overridden. Buffering is bounded to one frame per stage.
/// Arguments are passed directly to child processes, never through a shell.
/// This API needs the external executables, not native FFmpeg bindings.
///
/// # Errors
/// Returns probe, decode, callback, output-contract, encoding or remux failures.
/// Every exit closes pipes and reaps the processes owned by this call.
pub fn transcode_rgb_video<E>(
    input: &Path,
    output: &Path,
    options: &VideoStreamOptions,
    output_size: impl FnOnce(usize, usize) -> Result<(usize, usize), E>,
    mut transform: impl FnMut(&ImageFrame) -> Result<ImageFrame, E>,
) -> Result<(), E>
where
    E: From<MediaStreamError> + From<std::io::Error> + From<serde_json::Error>,
{
    if input == output
        || (output.exists() && std::fs::canonicalize(input)? == std::fs::canonicalize(output)?)
    {
        return Err(invalid("video output must differ from the source").into());
    }
    let VideoProbe {
        width,
        height,
        length,
        fps,
    } = probe_rgb_video(input.as_os_str(), options)?;
    let (output_width, output_height) = output_size(width, height)?;
    if output_width == 0 || output_height == 0 {
        return Err(invalid("empty output video frame").into());
    }
    let mut decoder = Process::new(
        tool_command(&options.ffmpeg)
            .args(["-v", "error", "-i"])
            .arg(input)
            .args([
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let mut encoder = Process::new(
        tool_command(&options.ffmpeg)
            .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s"])
            .arg(format!("{output_width}x{output_height}"))
            .arg("-r")
            .arg(fps.to_string())
            .args(["-i", "pipe:0", "-i"])
            .arg(input)
            .args([
                "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a",
                "copy", "-y",
            ])
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let mut reader = decoder
        .child
        .stdout
        .take()
        .ok_or_else(|| invalid("missing FFmpeg decode pipe"))?;
    let mut writer = encoder
        .child
        .stdin
        .take()
        .ok_or_else(|| invalid("missing FFmpeg encode pipe"))?;
    let mut bytes = vec![0; length];
    while read_rgb_frame(&mut reader, &mut bytes)? {
        let frame = ImageFrame::new(width, height, PixelFormat::Rgb8, bytes.clone())
            .map_err(MediaStreamError::from)?;
        let enhanced = transform(&frame)?;
        if enhanced.width() != output_width
            || enhanced.height() != output_height
            || enhanced.pixel_format() != PixelFormat::Rgb8
        {
            return Err(invalid("transformed frame differs from the RGB encoder contract").into());
        }
        writer.write_all(enhanced.data())?;
    }
    drop(reader);
    drop(writer);
    let decoder_status = decoder.wait()?;
    let encoder_status = encoder.wait()?;
    if !decoder_status.success() || !encoder_status.success() {
        return Err(invalid(format!(
            "video pipeline failed: decoder {decoder_status}, encoder {encoder_status}"
        ))
        .into());
    }
    Ok(())
}

struct VideoProbe {
    width: usize,
    height: usize,
    length: usize,
    fps: f64,
}

fn probe_rgb_video(
    input: &OsStr,
    options: &VideoStreamOptions,
) -> Result<VideoProbe, MediaStreamError> {
    let probe = tool_command(&options.ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
        ])
        .arg("-i")
        .arg(input)
        .stdin(Stdio::null())
        .output()?;
    if !probe.status.success() {
        return Err(invalid(format!(
            "FFprobe failed: {}",
            String::from_utf8_lossy(&probe.stderr)
        )));
    }
    let metadata: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let stream = metadata["streams"]
        .as_array()
        .and_then(|streams| streams.first())
        .ok_or_else(|| invalid("source has no video stream"))?;
    let dimension = |name: &str| -> Result<usize, MediaStreamError> {
        let value = stream[name]
            .as_u64()
            .ok_or_else(|| invalid(format!("missing video {name}")))?;
        usize::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("invalid video dimension"))
    };
    let width = dimension("width")?;
    let height = dimension("height")?;
    let length = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid("video frame size overflow"))?;
    let fps = match options.fps {
        Some(fps) => fps,
        None => {
            let rate = stream["avg_frame_rate"]
                .as_str()
                .ok_or_else(|| invalid("missing video frame rate"))?;
            let (numerator, denominator) = rate.split_once('/').unwrap_or((rate, "1"));
            let numerator: f64 = numerator
                .parse()
                .map_err(|_| invalid("invalid FPS numerator"))?;
            let denominator: f64 = denominator
                .parse()
                .map_err(|_| invalid("invalid FPS denominator"))?;
            numerator / denominator
        }
    };
    if !fps.is_finite() || fps <= 0.0 {
        return Err(invalid("video FPS must be finite and positive"));
    }
    Ok(VideoProbe {
        width,
        height,
        length,
        fps,
    })
}

fn read_rgb_frame(reader: &mut impl Read, bytes: &mut [u8]) -> std::io::Result<bool> {
    match reader.read_exact(&mut bytes[..1]) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(error),
    }
    // Incomplete frames must remain failures, rather than shortened success.
    reader.read_exact(&mut bytes[1..])?;
    Ok(true)
}

/// Decode a local video or HTTP(S) video URL using external FFmpeg tools.
/// This API is available without the native `video` or `url` Cargo features.
/// Frame selection uses the existing stride/limit policy; callers can request
/// all frames and apply count-based selection after decoding. No encoded or
/// converted files are written. Only this call's child processes are reaped.
///
/// # Errors
/// Reports invalid sampling, unsupported sources, failed tools, truncated raw
/// frames or invalid decoded storage. Requires FFmpeg and FFprobe executables.
pub fn decode_rgb_video(
    source: &MediaSource,
    sampling: FrameSampling,
    options: &VideoStreamOptions,
) -> Result<VideoClip, MediaStreamError> {
    decode_rgb_video_filtered(source, sampling, options, RgbVideoFilter::default())
}

/// Decode RGB video with FFmpeg frame-rate conversion and/or resizing.
/// Filters run before `sampling`, so frame indices address the filtered stream.
/// The resulting FPS is the filter rate when specified, otherwise the existing
/// source/metadata rate. Without an explicit output size, autorotation remains
/// disabled so storage matches the probed geometry. Uses external tools without
/// native FFmpeg bindings.
///
/// # Errors
/// Reports invalid filters, sampling, sources, tools or truncated RGB storage.
pub fn decode_rgb_video_filtered(
    source: &MediaSource,
    sampling: FrameSampling,
    options: &VideoStreamOptions,
    filter: RgbVideoFilter,
) -> Result<VideoClip, MediaStreamError> {
    sampling.validate()?;
    let expression = filter.expression()?;
    let input: OsString = match &source.location {
        MediaLocation::Path(path) => path.as_os_str().to_owned(),
        MediaLocation::Url(url) if url.starts_with("https://") || url.starts_with("http://") => {
            url.as_str().into()
        }
        MediaLocation::Url(_) => {
            return Err(invalid("external video decoding requires an HTTP(S) URL"));
        }
        MediaLocation::Bytes(_) => {
            return Err(invalid("external video decoding requires a path or URL"));
        }
    };
    let VideoProbe {
        width,
        height,
        length,
        fps,
    } = probe_rgb_video(&input, options)?;
    let (width, height, length) = if let Some((width, height)) = filter.size {
        let length = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| invalid("filtered RGB frame size overflow"))?;
        (width, height, length)
    } else {
        (width, height, length)
    };
    let fps = filter.fps.unwrap_or(fps);
    let mut command = tool_command(&options.ffmpeg);
    command.args(["-v", "error"]);
    // Preserve the existing decoder's no-autorotate behavior. A requested filter
    // follows the ordinary FFmpeg input/filter contract used by source pipelines.
    if filter.size.is_none() {
        command.arg("-noautorotate");
    }
    command.arg("-i").arg(&input).args(["-map", "0:v:0"]);
    if let Some(expression) = expression {
        command.arg("-vf").arg(expression);
    }
    let mut decoder = Process::new(
        command
            .args([
                "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let mut reader = decoder
        .child
        .stdout
        .take()
        .ok_or_else(|| invalid("missing FFmpeg decode pipe"))?;
    let mut bytes = vec![0; length];
    let mut frames = Vec::new();
    let mut index = 0usize;
    let mut completed = false;
    loop {
        if !read_rgb_frame(&mut reader, &mut bytes)? {
            completed = true;
            break;
        }
        if sampling.accepts_index(index)? {
            let image = ImageFrame::new(width, height, PixelFormat::Rgb8, bytes.clone())?
                .with_timing(FrameTiming {
                    index,
                    timestamp_ms: None,
                    duration_ms: None,
                });
            frames.push(VideoFrame::new(image));
            if sampling.reached_limit(frames.len()) {
                break;
            }
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| invalid("video frame index overflow"))?;
    }
    drop(reader);
    if completed {
        let status = decoder.wait()?;
        if !status.success() {
            return Err(invalid(format!("FFmpeg decoding failed: {status}")));
        }
    }
    // On a fulfilled frame limit, Process::drop stops only our decoder.
    Ok(VideoClip::new(frames, Some(fps))?)
}

/// Decode one image using FFmpeg's `scale` filter and RGB24 output.
/// Encoded bytes, paths and HTTP(S) URLs use FFmpeg's `image2pipe` demuxer.
/// No intermediate files or native bindings are used.
///
/// # Errors
/// Reports unsupported sources, invalid dimensions, failed tools or short output.
pub fn decode_rgb_image(
    source: &MediaSource,
    size: (usize, usize),
    options: &VideoStreamOptions,
) -> Result<ImageFrame, MediaStreamError> {
    let mut command = tool_command(&options.ffmpeg);
    command.args(["-v", "error"]);
    let bytes = match &source.location {
        MediaLocation::Path(path) => {
            command.args(["-f", "image2pipe", "-i"]).arg(path);
            None
        }
        MediaLocation::Url(url) if url.starts_with("http://") || url.starts_with("https://") => {
            command.args(["-f", "image2pipe", "-i"]).arg(url);
            None
        }
        MediaLocation::Bytes(bytes) => {
            command.args(["-f", "image2pipe", "-i", "pipe:0"]);
            Some(bytes.as_slice())
        }
        MediaLocation::Url(_) => {
            return Err(invalid("external image decoding requires an HTTP(S) URL"))
        }
    };
    single_rgb_frame(command, bytes, size)
}

/// Resize a decoded frame using the same FFmpeg `scale` filter as encoded media.
/// Pixel conversion and resizing both run in the existing external-tool owner.
///
/// # Errors
/// Reports invalid dimensions, failed tools, broken pipes or short RGB output.
pub fn scale_rgb_frame(
    frame: &ImageFrame,
    size: (usize, usize),
    options: &VideoStreamOptions,
) -> Result<ImageFrame, MediaStreamError> {
    let format = match frame.pixel_format() {
        PixelFormat::Rgb8 => "rgb24",
        PixelFormat::Rgba8 => "rgba",
        PixelFormat::Luma8 => "gray",
    };
    let mut command = tool_command(&options.ffmpeg);
    command
        .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", format, "-s"])
        .arg(format!("{}x{}", frame.width(), frame.height()))
        .args(["-i", "pipe:0"]);
    Ok(single_rgb_frame(command, Some(frame.data()), size)?.with_timing(frame.timing()))
}

fn single_rgb_frame(
    mut command: Command,
    input: Option<&[u8]>,
    size: (usize, usize),
) -> Result<ImageFrame, MediaStreamError> {
    let (width, height) = size;
    let expression = RgbVideoFilter {
        size: Some(size),
        fps: None,
    }
    .expression()?
    .ok_or_else(|| invalid("missing RGB scale filter"))?;
    let length = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid("RGB frame size overflow"))?;
    let mut decoder = Process::new(
        command
            .arg("-vf")
            .arg(expression)
            .args([
                "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ])
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let mut reader = decoder
        .child
        .stdout
        .take()
        .ok_or_else(|| invalid("missing RGB output pipe"))?;
    let stdin = decoder.child.stdin.take();
    let mut bytes = vec![0; length];
    std::thread::scope(|scope| -> Result<(), MediaStreamError> {
        // Writing and reading concurrently avoids pipe-capacity deadlocks for
        // large encoded images or decoded raw frames. Only this child is owned.
        let writer = stdin
            .zip(input)
            .map(|(mut stdin, input)| scope.spawn(move || stdin.write_all(input)));
        let read = read_rgb_frame(&mut reader, &mut bytes);
        drop(reader);
        let written = writer
            .map(|writer| {
                writer
                    .join()
                    .map_err(|_| invalid("RGB pipe writer panicked"))
            })
            .transpose()?;
        let status = decoder.wait()?;
        if !status.success() {
            return Err(invalid(format!("FFmpeg image decoding failed: {status}")));
        }
        if !read? {
            return Err(invalid("FFmpeg produced no RGB frame"));
        }
        if let Some(Err(error)) = written {
            // A successful first-frame decoder may close its encoded input
            // before trailing animation frames or metadata have been written.
            if error.kind() != std::io::ErrorKind::BrokenPipe {
                return Err(error.into());
            }
        }
        Ok(())
    })?;
    Ok(ImageFrame::new(width, height, PixelFormat::Rgb8, bytes)?)
}
