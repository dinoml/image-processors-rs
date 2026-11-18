use image_processors::media::{
    decode_rgb_image, decode_rgb_video, decode_rgb_video_filtered, scale_rgb_frame, RgbVideoFilter,
};
use image_processors::{
    FrameSampling, FrameTiming, ImageFrame, MediaSource, PixelFormat, VideoStreamOptions,
};
use serde::Deserialize;
use std::process::Command;

#[derive(Deserialize)]
struct Reference {
    image_cases: Vec<ImageCase>,
    fps_cases: Vec<FpsCase>,
}

#[derive(Deserialize)]
struct ImageCase {
    name: String,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
    policy: String,
    input: Vec<u8>,
    expected: Vec<u8>,
}

#[derive(Deserialize)]
struct FpsCase {
    source_rate: String,
    target_rate: String,
    input_frames: usize,
    expected_source_indices: Vec<usize>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("fixtures/magi_ffmpeg.json"))
        .expect("valid independently generated FFmpeg reference")
}

fn hidden_command(executable: impl AsRef<std::ffi::OsStr>) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut command = Command::new(executable);
        command.creation_flags(0x0800_0000);
        command
    }
    #[cfg(not(windows))]
    {
        Command::new(executable)
    }
}

fn tools() -> Option<VideoStreamOptions> {
    let tools = VideoStreamOptions::default();
    let available = [&tools.ffmpeg, &tools.ffprobe].into_iter().all(|path| {
        hidden_command(path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
    });
    if !available {
        assert!(
            std::env::var_os("DINOML_REQUIRE_FFMPEG").is_none(),
            "explicit FFmpeg qualification requires working ffmpeg and ffprobe"
        );
        eprintln!("SKIP FFmpeg reference tests: external tools unavailable");
        return None;
    }
    Some(tools)
}

fn png_bytes(case: &ImageCase) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, case.width as u32, case.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("valid tiny PNG dimensions");
        writer
            .write_image_data(&case.input)
            .expect("complete RGB fixture");
    }
    bytes
}

#[test]
fn fit_image_ingress_and_decoded_frame_scaling_match_original_ffmpeg_pixels() {
    let Some(tools) = tools() else { return };
    for case in reference()
        .image_cases
        .into_iter()
        .filter(|case| case.policy == "fit")
    {
        let size = (case.target_width, case.target_height);
        let decoded = decode_rgb_image(&MediaSource::image_bytes(png_bytes(&case)), size, &tools)
            .expect("decode the original image2pipe fit fixture");
        assert_eq!(decoded.data(), case.expected, "image ingress {}", case.name);
        let timing = FrameTiming {
            index: 7,
            timestamp_ms: Some(3.0),
            duration_ms: Some(2.0),
        };
        let frame = ImageFrame::new(case.width, case.height, PixelFormat::Rgb8, case.input)
            .expect("complete source frame")
            .with_timing(timing);
        let scaled = scale_rgb_frame(&frame, size, &tools).expect("scale decoded RGB fixture");
        assert_eq!(scaled.data(), case.expected, "decoded frame {}", case.name);
        assert_eq!(scaled.timing(), timing);
    }
}

#[test]
fn fps_filter_matches_reference_duplication_and_dropping_and_preserves_legacy_decoder() {
    let Some(mut tools) = tools() else { return };
    tools.fps = Some(11.0); // Existing option changes metadata without resampling.
    for case in reference().fps_cases {
        let temp = tempfile::tempdir().expect("isolated fixture directory");
        let raw_path = temp.path().join("input.rgb");
        let video_path = temp.path().join("input.nut");
        let input: Vec<u8> = (0..case.input_frames)
            .flat_map(|index| [index as u8 * 7, index as u8 * 5, index as u8 * 3].repeat(4))
            .collect();
        std::fs::write(&raw_path, &input).expect("write tiny raw RGB frames");
        let output = hidden_command(&tools.ffmpeg)
            .args([
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "2x2",
                "-framerate",
                &case.source_rate,
                "-i",
            ])
            .arg(&raw_path)
            .args(["-c:v", "rawvideo", "-f", "nut"])
            .arg(&video_path)
            .output()
            .expect("encode lossless timestamped fixture");
        assert!(
            output.status.success(),
            "fixture encoder: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let source = MediaSource::path(&video_path);
        let legacy = decode_rgb_video(&source, FrameSampling::default(), &tools)
            .expect("legacy decode without frame conversion");
        assert_eq!(legacy.frames().len(), case.input_frames);
        assert_eq!(legacy.fps(), Some(11.0));
        for (index, frame) in legacy.frames().iter().enumerate() {
            assert_eq!(frame.image().data(), &input[index * 12..(index + 1) * 12]);
        }
        let fps: f64 = case.target_rate.parse().expect("numeric target rate");
        let filter = RgbVideoFilter {
            size: Some((2, 2)),
            fps: Some(fps),
        };
        let converted =
            decode_rgb_video_filtered(&source, FrameSampling::default(), &tools, filter)
                .expect("apply actual FFmpeg fps and scale filters");
        let indices: Vec<usize> = converted
            .frames()
            .iter()
            .map(|frame| usize::from(frame.image().data()[0] / 7))
            .collect();
        assert_eq!(
            indices, case.expected_source_indices,
            "{} -> {}",
            case.source_rate, case.target_rate
        );
        assert_eq!(converted.fps(), Some(fps));
        for (frame, &index) in converted.frames().iter().zip(&indices) {
            assert_eq!(frame.image().data(), &input[index * 12..(index + 1) * 12]);
        }
        let selected = decode_rgb_video_filtered(
            &source,
            FrameSampling {
                start_index: 1,
                stride: 2,
                max_frames: Some(2),
            },
            &tools,
            filter,
        )
        .expect("sampling applies after fps conversion");
        assert_eq!(
            selected
                .frames()
                .iter()
                .map(|frame| usize::from(frame.image().data()[0] / 7))
                .collect::<Vec<_>>(),
            vec![indices[1], indices[3]]
        );
    }
}

#[test]
fn invalid_filters_fail_without_starting_external_tools() {
    let tools = VideoStreamOptions {
        ffmpeg: "missing-ffmpeg".into(),
        ffprobe: "missing-ffprobe".into(),
        fps: None,
    };
    let source = MediaSource::image_bytes(vec![]);
    assert!(decode_rgb_image(&source, (0, 2), &tools).is_err());
    assert!(decode_rgb_video_filtered(
        &MediaSource::path("missing-video"),
        FrameSampling::default(),
        &tools,
        RgbVideoFilter {
            size: None,
            fps: Some(f64::NAN)
        }
    )
    .is_err());
}
