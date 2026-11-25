//! Caller policy from pinned CellposeModel.eval: normalized multi-slice 3D
//! inference forces norm3D, while standalone and stitched stacks retain it.

use image_processors::processors::cellpose::{
    normalize_planes, predict_stack, predict_volume, CellposeError, CellposeImage,
    CellposeNormalization, CellposeOptions, CellposeVolume, CellposeVolumeOptions,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    shape: Vec<usize>,
    options: CellposeNormalization,
    input: Vec<f32>,
    expected_error: Option<String>,
}

fn sparse_cases() -> Vec<Case> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "fixtures/cellpose_local_normalization_original.json"
    ))
    .expect("valid unmodified-original normalization fixture");
    fixture
        .cases
        .into_iter()
        .filter(|case| case.expected_error.is_some())
        .collect()
}

fn first_volume_batch(
    volume: &CellposeVolume,
    options: &CellposeVolumeOptions,
) -> Result<Vec<f32>, CellposeError> {
    let mut captured = None;
    let result = predict_volume::<CellposeError>(volume, 32, 1, options, |pixels, _, _| {
        captured = Some(pixels.to_vec());
        Err(CellposeError("stopped after capturing model input".into()))
    });
    captured.ok_or_else(|| result.expect_err("callback stops before model execution"))
}

fn options(normalization: CellposeNormalization) -> CellposeVolumeOptions {
    CellposeVolumeOptions {
        inference: CellposeOptions {
            normalization,
            augment: false,
            compute_masks: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn assert_effective_volume_policy(volume: &CellposeVolume, options: &CellposeVolumeOptions) {
    let before = serde_json::to_value(options).expect("serializable caller options");
    let mut explicit = options.clone();
    explicit.inference.normalization.norm_3d = true;
    let expected = first_volume_batch(volume, &explicit).expect("explicit volume normalization");
    let actual = first_volume_batch(volume, options).expect("effective volume normalization");
    assert_eq!(
        actual, expected,
        "same exact model input as explicit norm3D"
    );
    assert_eq!(
        serde_json::to_value(options).expect("serializable caller options"),
        before,
        "effective policy must not mutate caller options"
    );
}

#[test]
fn multi_slice_volume_uses_global_percentiles_even_when_norm3d_is_false() {
    let values: Vec<f32> = (0..2 * 11 * 17)
        .map(|index| (index % (11 * 17)) as f32 + (index / (11 * 17)) as f32 * 1000.0)
        .collect();
    let volume =
        CellposeVolume::from_array(&values, &[2, 11, 17], 0, None).expect("valid two-slice volume");
    for invert in [false, true] {
        assert_effective_volume_policy(
            &volume,
            &options(CellposeNormalization {
                norm_3d: false,
                invert,
                ..Default::default()
            }),
        );
    }
}

#[test]
fn volume_policy_accepts_original_sparse_per_slice_local_normalization_cases() {
    let cases = sparse_cases();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let volume = CellposeVolume {
            shape: [case.shape[0], case.shape[1], case.shape[2]],
            pixels: case.input,
        };
        assert_effective_volume_policy(&volume, &options(case.options));
    }
}

#[test]
fn disabled_volume_normalization_does_not_force_normalization() {
    let values: Vec<f32> = (0..2 * 11 * 17).map(|index| index as f32).collect();
    let volume =
        CellposeVolume::from_array(&values, &[2, 11, 17], 0, None).expect("valid two-slice volume");
    for blocksize in [0, 4] {
        let options = options(CellposeNormalization {
            normalize: false,
            norm_3d: false,
            tile_norm_blocksize: blocksize,
            ..Default::default()
        });
        let actual = first_volume_batch(&volume, &options).expect("normalization disabled");
        let mut global = options.clone();
        global.inference.normalization.norm_3d = true;
        assert_eq!(
            actual,
            first_volume_batch(&volume, &global).expect("normalization still disabled")
        );
        assert!(!options.inference.normalization.norm_3d);
    }
}

#[test]
fn disabled_normalization_preserves_image_and_stitched_stack_inputs() {
    let images: Vec<CellposeImage> = (0..2)
        .map(|z| CellposeImage {
            height: 11,
            width: 17,
            pixels: (0..3 * 11 * 17)
                .map(|index| (index % (11 * 17) + z * 1000) as f32)
                .collect(),
        })
        .collect();
    let baseline = options(CellposeNormalization {
        normalize: false,
        ..Default::default()
    });
    let disabled = options(CellposeNormalization {
        normalize: false,
        norm_3d: false,
        lowhigh: Some(vec![[0.0, 10.0]]),
        invert: true,
        tile_norm_blocksize: 4,
        smooth_radius: 1.0,
        ..Default::default()
    });
    for count in [1, 2] {
        let capture = |options: &CellposeOptions| {
            let mut captured = None;
            let result = predict_stack::<CellposeError>(
                &images[..count],
                32,
                1,
                options,
                0.5,
                |pixels, _, _| {
                    captured = Some(pixels.to_vec());
                    Err(CellposeError("captured before model execution".into()))
                },
            );
            assert!(result.is_err());
            captured.expect("disabled normalization reaches input callback")
        };
        assert_eq!(capture(&disabled.inference), capture(&baseline.inference));
    }
}

#[test]
fn standalone_normalization_retains_original_sparse_per_slice_errors() {
    for case in sparse_cases() {
        let error = normalize_planes(&mut case.input.clone(), &case.shape, &case.options)
            .expect_err("standalone norm3D=false retains original sparse rejection");
        assert!(error.to_string().contains("norm3D=False"));
    }
}

#[test]
fn stitched_stacks_retain_original_sparse_per_slice_errors() {
    for case in sparse_cases() {
        let [depth, height, width] = [case.shape[0], case.shape[1], case.shape[2]];
        let area = height * width;
        let images: Vec<CellposeImage> = (0..depth)
            .map(|z| CellposeImage {
                height,
                width,
                pixels: (0..3)
                    .flat_map(|channel| {
                        case.input[(channel * depth + z) * area..(channel * depth + z + 1) * area]
                            .iter()
                            .copied()
                    })
                    .collect(),
            })
            .collect();
        let options = options(case.options);
        let error =
            predict_stack::<CellposeError>(&images, 32, 1, &options.inference, 0.5, |_, _, _| {
                panic!("sparse stitched stack must reject before model execution")
            })
            .expect_err("stitched norm3D=false retains original sparse rejection");
        assert!(error.to_string().contains("norm3D=False"));
        assert!(!options.inference.normalization.norm_3d);
    }
}
