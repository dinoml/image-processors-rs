use crate::tensor::{DType, Layout, TensorDataView};

use super::*;

#[test]
fn image_frame_validates_buffer_length() {
    let err = ImageFrame::new(2, 2, PixelFormat::Rgb8, vec![0; 11]).unwrap_err();
    assert!(matches!(
        err,
        MediaError::InvalidBufferLength {
            expected: 12,
            actual: 11
        }
    ));
}

#[test]
fn image_frame_rejects_zero_dimensions() {
    let err = ImageFrame::new(0, 1, PixelFormat::Luma8, Vec::new()).unwrap_err();

    assert!(matches!(
        err,
        MediaError::InvalidFrameDimensions {
            width: 0,
            height: 1
        }
    ));
}

#[test]
fn image_frame_tensor_view_borrows_interleaved_pixels() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();

    let view = frame.as_tensor_view().unwrap();
    let TensorDataView::U8(values) = view.data() else {
        panic!("expected borrowed u8 tensor data");
    };

    assert!(std::ptr::eq(values.as_ptr(), frame.data().as_ptr()));
    assert_eq!(values, frame.data());
    assert_eq!(view.shape(), [1, 2, 3]);
    assert_eq!(view.layout(), Layout::HWC);
    assert_eq!(view.dtype(), DType::U8);
}

#[test]
fn remote_load_options_default_to_bounded_fetching() {
    let options = RemoteLoadOptions::default();

    assert_eq!(options.timeout(), Duration::from_secs(30));
    assert_eq!(options.max_bytes(), 128 * 1024 * 1024);
    assert_eq!(
        options.redirect_policy(),
        RemoteRedirectPolicy::FollowLimited(10)
    );
    assert!(options.user_agent().is_some());
}

#[test]
fn media_source_infers_type_from_path_and_format() {
    assert_eq!(
        MediaSource::path("frame.png").media_type(),
        Some(MediaType::Image)
    );
    assert_eq!(
        MediaSource::path("clip.mp4").media_type(),
        Some(MediaType::Video)
    );
    assert_eq!(
        MediaSource::bytes(Vec::<u8>::new())
            .with_format("jpeg")
            .media_type(),
        Some(MediaType::Image)
    );
    assert_eq!(
        MediaSource::url("https://example.com/media/frame.webp?download=1").media_type(),
        Some(MediaType::Image)
    );
    assert_eq!(
        MediaSource::url("https://example.com/media/clip.mp4#frame").media_type(),
        Some(MediaType::Video)
    );
    assert_eq!(
        MediaSource::image_url("https://example.com/media").media_type(),
        Some(MediaType::Image)
    );
    assert_eq!(
        MediaSource::video_url("https://example.com/media").media_type(),
        Some(MediaType::Video)
    );
}

#[test]
fn media_source_carries_frame_sampling_hint() {
    let source = MediaSource::video_path("clip.mp4")
        .with_frame_sampling(FrameSampling::every(3).with_max_frames(2))
        .with_remote_read_mode(RemoteReadMode::Buffered);

    assert_eq!(
        source.hint.frame_sampling,
        Some(FrameSampling {
            start_index: 0,
            stride: 3,
            max_frames: Some(2),
        })
    );
    assert_eq!(source.hint.remote_read_mode, Some(RemoteReadMode::Buffered));
}

#[test]
fn image_url_streaming_is_explicitly_unsupported() {
    let err = DefaultMediaLoader
        .load(MediaSource::image_url("https://example.com/image.png").streaming())
        .unwrap_err();

    assert!(matches!(
        err,
        MediaError::RemoteStreamingUnavailable(MediaType::Image)
    ));
}

#[cfg(all(feature = "video", not(feature = "url")))]
#[test]
fn buffered_video_urls_require_url_fetching() {
    let err = DefaultMediaLoader
        .load(MediaSource::video_url("https://example.com/video.mp4").buffered())
        .unwrap_err();

    assert!(matches!(
        err,
        MediaError::RemoteBufferingUnavailable(MediaType::Video)
    ));
}

#[test]
fn frame_sampling_selects_strided_indices() {
    let sampling = FrameSampling {
        start_index: 1,
        stride: 2,
        max_frames: Some(2),
    };

    assert_eq!(sampling.sample_indices(6).unwrap(), vec![1, 3]);
    assert!(sampling.accepts_index(1).unwrap());
    assert!(!sampling.accepts_index(2).unwrap());
    assert!(sampling.reached_limit(2));
}

#[test]
fn frame_sampling_rejects_zero_frame_limit() {
    let sampling = FrameSampling {
        max_frames: Some(0),
        ..FrameSampling::default()
    };

    assert!(matches!(
        sampling.sample_indices(1).unwrap_err(),
        MediaError::InvalidFrameLimit
    ));
}

#[test]
fn image_sequence_samples_frames() {
    let frames = (0..4)
        .map(|index| {
            ImageFrame::new(1, 1, PixelFormat::Luma8, vec![index as u8])
                .unwrap()
                .with_timing(FrameTiming::new(index))
        })
        .collect::<Vec<_>>();
    let sequence = ImageSequence::new(frames, LoopBehavior::Once).unwrap();

    let sampled = sequence.sample_frames(FrameSampling::every(2)).unwrap();

    assert_eq!(sampled.len(), 2);
    assert_eq!(sampled[0].data, vec![0]);
    assert_eq!(sampled[1].data, vec![2]);
}

#[test]
fn image_sequence_into_first_frame_reuses_pixel_buffer() {
    let frame = ImageFrame::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let data_ptr = frame.data().as_ptr();
    let sequence = ImageSequence::from_frame(frame);

    let frame = sequence.into_first_frame();

    assert_eq!(frame.data().as_ptr(), data_ptr);
}

#[test]
fn video_clip_samples_frames() {
    let frames = (0..4)
        .map(|index| {
            VideoFrame::new(
                ImageFrame::new(1, 1, PixelFormat::Luma8, vec![index as u8])
                    .unwrap()
                    .with_timing(FrameTiming::new(index)),
            )
        })
        .collect::<Vec<_>>();
    let clip = VideoClip::new(frames, Some(30.0)).unwrap();

    let sampled = clip.sampled(FrameSampling::every(2)).unwrap();

    assert_eq!(sampled.frames.len(), 2);
    assert_eq!(sampled.frames[0].image.data, vec![0]);
    assert_eq!(sampled.frames[1].image.data, vec![2]);
    assert_eq!(sampled.fps, Some(30.0));
}

#[test]
fn default_loader_applies_frame_sampling_to_animated_image_sources() {
    let gif = animated_gif_bytes_from_colors(&[
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
    ]);
    let source = MediaSource::image_bytes(gif).with_frame_sampling(FrameSampling {
        start_index: 1,
        stride: 2,
        max_frames: Some(2),
    });

    let loaded = DefaultMediaLoader.load(source).unwrap();

    let LoadedMedia::ImageSequence(sequence) = loaded else {
        panic!("expected sampled animated image sequence");
    };
    assert_eq!(sequence.len(), 2);
    assert_eq!(sequence.frames()[0].timing().index, 1);
    assert_eq!(sequence.frames()[1].timing().index, 3);
    assert_eq!(sequence.frames()[0].data(), &[0, 255, 0, 255]);
    assert_eq!(sequence.frames()[1].data(), &[255, 255, 255, 255]);
}

#[cfg(feature = "video")]
#[test]
fn decodes_video_fixture_with_sampling_and_timestamps() {
    let path = tiny_video_fixture_path();
    write_tiny_video_fixture(&path).unwrap();

    let options = VideoDecodeOptions::default().with_frame_sampling(FrameSampling {
        start_index: 1,
        stride: 2,
        max_frames: Some(2),
    });
    let result = load_video_from_path_with_options(&path, options);
    let _ = std::fs::remove_file(&path);
    let clip = result.unwrap();

    assert_eq!(clip.frames.len(), 2);
    assert_eq!(clip.frames[0].timing().index, 1);
    assert_eq!(clip.frames[1].timing().index, 3);
    assert!(clip.fps.is_some_and(|fps| (fps - 30.0).abs() < 0.1));

    let first = clip.frames[0].timing();
    let second = clip.frames[1].timing();
    assert!(first.timestamp_ms.is_some_and(|timestamp| timestamp >= 0.0));
    assert!(
        second
            .timestamp_ms
            .zip(first.timestamp_ms)
            .is_some_and(|(second, first)| second > first),
        "{first:?} {second:?}"
    );
    assert!(first.duration_ms.is_some_and(|duration| duration > 0.0));
}

#[cfg(feature = "video")]
#[test]
fn default_loader_applies_frame_sampling_to_video_sources() {
    let path = tiny_video_fixture_path();
    write_tiny_video_fixture(&path).unwrap();

    let source = MediaSource::video_path(&path).with_frame_sampling(FrameSampling {
        start_index: 1,
        stride: 2,
        max_frames: Some(2),
    });
    let result = DefaultMediaLoader.load(source);
    let _ = std::fs::remove_file(&path);
    let loaded = result.unwrap();

    let LoadedMedia::Video(clip) = loaded else {
        panic!("expected sampled video clip");
    };
    assert_eq!(clip.frames().len(), 2);
    assert_eq!(clip.frames()[0].timing().index, 1);
    assert_eq!(clip.frames()[1].timing().index, 3);
}

#[cfg(not(feature = "url"))]
#[test]
fn default_loader_reports_disabled_url_loading() {
    let err = DefaultMediaLoader
        .load(MediaSource::image_url("https://example.com/image.png"))
        .unwrap_err();

    assert!(matches!(err, MediaError::RemoteLoadingUnavailable(_)));
}

#[cfg(not(feature = "video"))]
#[test]
fn default_loader_reports_disabled_video_decoding() {
    let err = DefaultMediaLoader
        .load(MediaSource::video_path("clip.mp4"))
        .unwrap_err();

    assert!(matches!(err, MediaError::VideoDecodingUnavailable));
}

#[test]
fn decodes_animated_still_formats_to_sequences() {
    for (format, bytes) in [
        ("gif", animated_gif_bytes()),
        ("apng", animated_png_bytes()),
        ("webp", animated_webp_bytes()),
    ] {
        let sequence = decode_image_sequence_bytes(&bytes).unwrap();

        assert_eq!(sequence.frames.len(), 2, "{format}");
        assert!(sequence.is_animated(), "{format}");
        assert_eq!(sequence.frames[0].pixel_format, PixelFormat::Rgba8);
        assert_eq!(sequence.frames[0].timing.index, 0);
        assert_eq!(sequence.frames[1].timing.index, 1);
    }
}

#[test]
fn decodes_image_bytes() {
    use ::image::codecs::png::PngEncoder;
    use ::image::{ColorType, ImageEncoder};

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&[255, 0, 0], 1, 1, ColorType::Rgb8.into())
        .unwrap();

    let frame = decode_image_bytes(&png).unwrap();

    assert_eq!(frame.width, 1);
    assert_eq!(frame.height, 1);
    assert_eq!(frame.channels(), 3);
    assert_eq!(frame.data, vec![255, 0, 0]);
}

#[cfg(not(feature = "turbojpeg"))]
#[test]
fn turbojpeg_backend_reports_missing_feature_when_disabled() {
    let jpeg = jpeg_bytes();

    let err = decode_image_bytes_with_backend(&jpeg, ImageDecodeBackend::TurboJpeg).unwrap_err();

    assert!(matches!(err, MediaError::TurboJpegUnavailable));
}

#[cfg(feature = "turbojpeg")]
#[test]
fn turbojpeg_backend_decodes_jpeg_bytes() {
    let jpeg = jpeg_bytes();

    let frame = decode_image_bytes_with_backend(&jpeg, ImageDecodeBackend::TurboJpeg).unwrap();

    assert_eq!(frame.width, 2);
    assert_eq!(frame.height, 1);
    assert_eq!(frame.pixel_format, PixelFormat::Rgb8);
    assert_eq!(frame.data.len(), 2 * 3);
}

#[test]
fn default_loader_decodes_image_sources() {
    use ::image::codecs::png::PngEncoder;
    use ::image::{ColorType, ImageEncoder};

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&[0, 255, 0], 1, 1, ColorType::Rgb8.into())
        .unwrap();

    let loaded = DefaultMediaLoader
        .load(MediaSource::image_bytes(png))
        .unwrap();

    match loaded {
        LoadedMedia::Image(frame) => assert_eq!(frame.data, vec![0, 255, 0]),
        LoadedMedia::ImageSequence(_) | LoadedMedia::Video(_) => panic!("expected image media"),
    }
}

fn jpeg_bytes() -> Vec<u8> {
    use ::image::codecs::jpeg::JpegEncoder;
    use ::image::{ColorType, ImageEncoder};

    let mut jpeg = Vec::new();
    JpegEncoder::new(&mut jpeg)
        .write_image(&[255, 0, 0, 0, 255, 0], 2, 1, ColorType::Rgb8.into())
        .unwrap();
    jpeg
}

fn animated_gif_bytes() -> Vec<u8> {
    animated_gif_bytes_from_colors(&[[255, 0, 0, 255], [0, 255, 0, 255]])
}

fn animated_gif_bytes_from_colors(colors: &[[u8; 4]]) -> Vec<u8> {
    use ::image::codecs::gif::GifEncoder;
    use ::image::{Delay, Frame, ImageBuffer, Rgba};

    let frames = colors
        .iter()
        .enumerate()
        .map(|(index, color)| {
            Frame::from_parts(
                ImageBuffer::from_pixel(1, 1, Rgba(*color)),
                0,
                0,
                Delay::from_numer_denom_ms(10 + index as u32 * 10, 1),
            )
        })
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    GifEncoder::new(&mut bytes).encode_frames(frames).unwrap();
    bytes
}

fn animated_png_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(2, 0).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.set_frame_delay(10, 1).unwrap();
        writer.write_image_data(&[255, 0, 0, 255]).unwrap();
        writer.set_frame_delay(20, 1).unwrap();
        writer.write_image_data(&[0, 255, 0, 255]).unwrap();
        writer.finish().unwrap();
    }
    bytes
}

fn animated_webp_bytes() -> Vec<u8> {
    let frame_a = webp_frame_payload([255, 0, 0, 255]);
    let frame_b = webp_frame_payload([0, 255, 0, 255]);
    let mut body = Vec::new();

    let mut vp8x = Vec::new();
    vp8x.push(0b0000_0010);
    vp8x.extend_from_slice(&[0, 0, 0]);
    push_u24_le(&mut vp8x, 0);
    push_u24_le(&mut vp8x, 0);
    push_riff_chunk(&mut body, b"VP8X", &vp8x);

    let mut anim = Vec::new();
    anim.extend_from_slice(&[0, 0, 0, 0]);
    anim.extend_from_slice(&0u16.to_le_bytes());
    push_riff_chunk(&mut body, b"ANIM", &anim);

    push_riff_chunk(&mut body, b"ANMF", &webp_animation_frame(10, &frame_a));
    push_riff_chunk(&mut body, b"ANMF", &webp_animation_frame(20, &frame_b));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
    bytes.extend_from_slice(b"WEBP");
    bytes.extend_from_slice(&body);
    bytes
}

fn webp_frame_payload(rgba: [u8; 4]) -> Vec<u8> {
    use ::image::codecs::webp::WebPEncoder;
    use ::image::{ExtendedColorType, ImageEncoder};

    let mut bytes = Vec::new();
    WebPEncoder::new_lossless(&mut bytes)
        .write_image(&rgba, 1, 1, ExtendedColorType::Rgba8)
        .unwrap();

    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    bytes[12..].to_vec()
}

fn webp_animation_frame(duration_ms: u32, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    push_u24_le(&mut frame, 0);
    push_u24_le(&mut frame, 0);
    push_u24_le(&mut frame, 0);
    push_u24_le(&mut frame, 0);
    push_u24_le(&mut frame, duration_ms);
    frame.push(0);
    frame.extend_from_slice(payload);
    frame
}

fn push_riff_chunk(output: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    output.extend_from_slice(tag);
    output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    output.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        output.push(0);
    }
}

fn push_u24_le(output: &mut Vec<u8>, value: u32) {
    output.push((value & 0xff) as u8);
    output.push(((value >> 8) & 0xff) as u8);
    output.push(((value >> 16) & 0xff) as u8);
}

#[cfg(feature = "video")]
fn write_tiny_video_fixture(path: &Path) -> Result<(), String> {
    use base64::Engine;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(TINY_MPEG4_MP4_BASE64)
        .map_err(|error| format!("fixture decode: {error}"))?;
    std::fs::write(path, bytes).map_err(|error| format!("fixture write: {error}"))
}

#[cfg(feature = "video")]
fn tiny_video_fixture_path() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "image-processors-video-test-{}-{nanos}.mp4",
        std::process::id()
    ));
    path
}

#[cfg(feature = "video")]
const TINY_MPEG4_MP4_BASE64: &str = concat!(
    "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",
);
