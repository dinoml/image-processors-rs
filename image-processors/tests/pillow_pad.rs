// Independent Pillow 10.4.0 ImageOps.pad reference bytes, generated before this API.
// DeepSeek consumes RGB; the additional L/RGBA fixtures exercise shared geometry.
use image_processors::transforms::{
    pillow_pad_frame, pillow_pad_plan, pillow_patch_grid_image_patches, resize_fill_plan,
    ImageSize, ResizeDecision, ResizeFilter,
};
use image_processors::{ImageFrame, PixelFormat};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Reference {
    pillow_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Case {
    mode: String,
    source_size: [usize; 2],
    target_size: [usize; 2],
    resized_size: [usize; 2],
    filter: String,
    fill: Vec<u8>,
    input: Vec<u8>,
    expected: Vec<u8>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("fixtures/pillow_pad.json")).expect("valid Pillow fixture")
}

fn size(wh: [usize; 2]) -> ImageSize {
    ImageSize::new(wh[1], wh[0]).expect("nonzero fixture dimensions")
}

#[test]
fn contained_dimensions_match_pillow_halfway_rounding() {
    let reference = reference();
    assert_eq!(reference.pillow_version, "10.4.0");
    assert_eq!(reference.cases.len(), 36);
    for case in reference.cases {
        let plan = pillow_pad_plan(size(case.source_size), size(case.target_size))
            .expect("valid contained geometry");
        assert_eq!(
            [plan.resized_size.width, plan.resized_size.height],
            case.resized_size,
            "{} {:?} -> {:?}",
            case.mode,
            case.source_size,
            case.target_size
        );
    }
}

#[test]
fn rgb_pixels_match_pillow_nearest_and_bicubic_including_odd_padding() {
    let mut count = 0;
    for case in reference()
        .cases
        .into_iter()
        .filter(|case| case.mode == "RGB")
    {
        let frame = ImageFrame::new(
            case.source_size[0],
            case.source_size[1],
            PixelFormat::Rgb8,
            case.input,
        )
        .expect("complete RGB fixture");
        let filter = match case.filter.as_str() {
            "nearest" => ResizeFilter::Nearest,
            "bicubic" => ResizeFilter::Bicubic,
            other => panic!("unexpected fixture filter {other}"),
        };
        let output = pillow_pad_frame(
            &frame,
            size(case.target_size),
            &case.fill,
            ResizeDecision::compatibility(filter),
        )
        .expect("valid RGB padding fixture");
        assert_eq!(output.width(), case.target_size[0]);
        assert_eq!(output.height(), case.target_size[1]);
        assert_eq!(
            output.data(),
            case.expected,
            "{:?} -> {:?}, {}",
            case.source_size,
            case.target_size,
            case.filter
        );
        count += 1;
    }
    assert_eq!(count, 12);
}

#[test]
fn existing_fill_plan_keeps_its_floor_centering_policy() {
    let original = resize_fill_plan(size([4, 2]), size([5, 5])).expect("legacy geometry");
    let pillow = pillow_pad_plan(size([4, 2]), size([5, 5])).expect("Pillow geometry");
    assert_eq!(original.padding.top, 1);
    assert_eq!(pillow.padding.top, 2);
    assert_eq!(original.resized_size, pillow.resized_size);
}

#[test]
fn patch_grid_preserves_pillow_global_view_fill_and_row_major_tiles() {
    for case in reference()
        .cases
        .into_iter()
        .filter(|case| case.mode == "RGB")
    {
        let frame = ImageFrame::new(
            case.source_size[0],
            case.source_size[1],
            PixelFormat::Rgb8,
            case.input,
        )
        .expect("complete RGB fixture");
        let filter = match case.filter.as_str() {
            "nearest" => ResizeFilter::Nearest,
            "bicubic" => ResizeFilter::Bicubic,
            other => panic!("unexpected fixture filter {other}"),
        };
        let target = size(case.target_size);
        let grid = pillow_patch_grid_image_patches(
            &frame,
            &[target],
            target,
            1,
            ResizeDecision::compatibility(filter),
            &case.fill,
        )
        .expect("valid one-pixel tile grid");
        assert_eq!(grid.frames[0].data(), case.expected);
        assert_eq!(
            [grid.plan.resized_size.width, grid.plan.resized_size.height],
            case.resized_size,
        );
        let tiled: Vec<u8> = grid.frames[1..]
            .iter()
            .flat_map(|tile| tile.data().iter().copied())
            .collect();
        assert_eq!(
            tiled, case.expected,
            "row-major tiles {:?}",
            case.source_size
        );
        assert_eq!(grid.frames.len(), 1 + target.height * target.width);
    }
}
