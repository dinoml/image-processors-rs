// Independent NumPy FP32 oracle: clip unit pixels, scale by255, then cast/truncate.
// Inputs are exact IEEE754 bits. Nearest uses floor(x+0.5); even uses NumPy rint.
use image_processors::postprocess::{
    image_frame_from_chw_unit_values_with_rounding,
    image_frame_from_chw_unit_values_with_rounding_mode, FloatByteRounding, ImageSize,
};

struct Case {
    label: &'static str,
    width: usize,
    height: usize,
    channels: usize,
    bits: &'static [u32],
    truncated: &'static [u8],
    nearest: &'static [u8],
    even: &'static [u8],
}

const CASES: &[Case] = &[
    Case {
        label: "gray",
        width: 4,
        height: 4,
        channels: 1,
        bits: &[
            0xbc008081, 0xba808081, 0x00000000, 0x3afbdd25, 0x3b008081, 0x3bbf77ca, 0x3bc0c0c1,
            0x3c20a0a1, 0x3e800000, 0x3e99999a, 0x3f000000, 0x3f004040, 0x3f7f7cee, 0x3f7f7f7f,
            0x3f800000, 0x3f810101,
        ],
        truncated: &[0, 0, 0, 0, 0, 1, 1, 2, 63, 76, 127, 127, 254, 254, 255, 255],
        nearest: &[0, 0, 0, 0, 1, 1, 2, 3, 64, 77, 128, 128, 254, 255, 255, 255],
        even: &[0, 0, 0, 0, 0, 1, 2, 2, 64, 76, 128, 128, 254, 254, 255, 255],
    },
    Case {
        label: "rgb",
        width: 2,
        height: 2,
        channels: 3,
        bits: &[
            0xbc008081, 0x3b008081, 0x3f000000, 0x3f810101, 0xba808081, 0x3bbf77ca, 0x3f004040,
            0x3f800000, 0x00000000, 0x3bc0c0c1, 0x3f7f7cee, 0x3f7f7f7f,
        ],
        truncated: &[0, 0, 0, 0, 1, 1, 127, 127, 254, 255, 255, 254],
        nearest: &[0, 0, 0, 1, 1, 2, 128, 128, 254, 255, 255, 255],
        even: &[0, 0, 0, 0, 1, 2, 128, 128, 254, 255, 255, 254],
    },
    Case {
        label: "rgba",
        width: 2,
        height: 2,
        channels: 4,
        bits: &[
            0xbc008081, 0x3b008081, 0x3f000000, 0x3f810101, 0xba808081, 0x3bbf77ca, 0x3f004040,
            0x3f800000, 0x00000000, 0x3bc0c0c1, 0x3f7f7cee, 0x3f7f7f7f, 0x3afbdd25, 0x3c20a0a1,
            0x3e800000, 0x3e99999a,
        ],
        truncated: &[0, 0, 0, 0, 0, 1, 1, 2, 127, 127, 254, 63, 255, 255, 254, 76],
        nearest: &[0, 0, 0, 0, 1, 1, 2, 3, 128, 128, 254, 64, 255, 255, 255, 77],
        even: &[0, 0, 0, 0, 0, 1, 2, 2, 128, 128, 254, 64, 255, 255, 254, 76],
    },
];

#[test]
fn truncation_matches_numpy_for_gray_rgb_and_rgba() {
    for case in CASES {
        let values: Vec<f32> = case.bits.iter().copied().map(f32::from_bits).collect();
        let size = ImageSize::new(case.width, case.height).expect("nonzero fixture size");
        let image = image_frame_from_chw_unit_values_with_rounding_mode(
            &values,
            size,
            case.channels,
            FloatByteRounding::Truncate,
        )
        .expect("valid CHW fixture");
        assert_eq!(image.data(), case.truncated, "{} truncation", case.label);
    }
}

#[test]
fn existing_rounding_policies_retain_their_byte_results() {
    for case in CASES {
        let values: Vec<f32> = case.bits.iter().copied().map(f32::from_bits).collect();
        let size = ImageSize::new(case.width, case.height).expect("nonzero fixture size");
        for (mode, ties_even, expected) in [
            (FloatByteRounding::Nearest, false, case.nearest),
            (FloatByteRounding::NearestEven, true, case.even),
        ] {
            let explicit = image_frame_from_chw_unit_values_with_rounding_mode(
                &values,
                size,
                case.channels,
                mode,
            )
            .expect("valid explicit rounding fixture");
            let existing = image_frame_from_chw_unit_values_with_rounding(
                &values,
                size,
                case.channels,
                ties_even,
            )
            .expect("valid existing rounding fixture");
            assert_eq!(explicit.data(), expected, "{} {mode:?}", case.label);
            assert_eq!(
                existing.data(),
                expected,
                "{} existing {mode:?}",
                case.label
            );
        }
    }
}
