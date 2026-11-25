// Pinned original CPU frame-index references; provenance and explicit policy boundaries are in the fixture.
use image_processors::FrameSampling;
use serde_json::{json, Value};

#[test]
fn timestamp_sampling_matches_original_cpu_references() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/timestamp_sampling_reference.json"))?;
    let mut failures = Vec::new();
    let mut skipped = Vec::new();
    let mut comparisons = 0;
    for case in fixture["cases"].as_array().expect("reference cases") {
        let input = &case["inputs"];
        if input["align_to_multiple"]
            .as_i64()
            .is_some_and(|value| value != 1)
        {
            skipped.push(json!({"case":case["name"],"reason":"alignment is caller policy"}));
            continue;
        }
        if input["start_frame"].as_i64().is_some_and(|value| value < 0) {
            skipped.push(json!({"case":case["name"],"reason":"negative index is unrepresentable in typed usize API"}));
            continue;
        }
        let len = input["total_frames"].as_u64().unwrap() as usize;
        let range = if input.get("start_frame").is_some() || input.get("end_frame").is_some() {
            Some(
                input["start_frame"].as_u64().unwrap_or(0) as usize
                    ..input["end_frame"]
                        .as_u64()
                        .map_or(len, |value| value as usize),
            )
        } else {
            None
        };
        let result = FrameSampling::timestamp_grid_indices(
            len,
            input["source_fps"].as_f64().unwrap(),
            input["target_fps"].as_f64().unwrap(),
            input["max_sampled_frames"].as_u64().unwrap() as usize,
            range,
            input["uniform_over_clip"].as_bool().unwrap_or(false),
        );
        comparisons += 1;
        if case["expected"].get("error").is_some() {
            if result.is_ok() {
                failures.push(json!({"case":case["name"],"reason":"accepted invalid controls"}));
            }
            continue;
        }
        match result {
            Ok((indices, fps)) => {
                let expected: Vec<usize> =
                    serde_json::from_value(case["expected"]["indices"].clone())?;
                let expected_fps = case["expected"]["conditioning_fps"].as_f64().unwrap();
                if indices != expected
                    || (fps - expected_fps).abs() > 1e-12 * expected_fps.abs().max(1.0)
                {
                    failures.push(
                        json!({"case":case["name"],"actual":indices,"expected":expected,
                        "fps":fps,"expectedFps":expected_fps}),
                    );
                }
            }
            Err(error) => failures.push(json!({"case":case["name"],"error":error.to_string()})),
        }
    }
    println!(
        "{}",
        json!({"comparisons":comparisons,"failures":failures,"skipped":skipped})
    );
    assert_eq!(comparisons, 31);
    assert_eq!(skipped.len(), 5);
    assert!(
        failures.is_empty(),
        "Original CPU reference mismatches: {failures:?}"
    );
    Ok(())
}

#[test]
fn video_clip_uses_explicit_nominal_fps() -> Result<(), Box<dyn std::error::Error>> {
    use image_processors::{ImageFrame, PixelFormat, VideoClip, VideoFrame};
    let frames = (0..13)
        .map(|index| ImageFrame::new(1, 1, PixelFormat::Luma8, vec![index]).map(VideoFrame::new))
        .collect::<Result<Vec<_>, _>>()?;
    let clip = VideoClip::new(frames.clone(), Some(2.0))?;
    let (indices, fps) = clip.timestamp_grid_indices(24.0, 64, None, false)?;
    // Original deduplicate_high_requested_fps fixture: preserve every unique frame.
    assert_eq!(indices, (0..13).collect::<Vec<_>>());
    assert_eq!(fps, 2.0);
    assert!(VideoClip::new(frames.clone(), None)?
        .timestamp_grid_indices(1.0, 4, None, false)
        .is_err());
    assert!(VideoClip::new(frames, Some(f64::NAN))?
        .timestamp_grid_indices(1.0, 4, None, false)
        .is_err());
    Ok(())
}
