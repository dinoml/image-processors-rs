use image_processors::processors::siglip2::Siglip2ImageProcessor;
use image_processors::{ImageFrame, ImageSize, PixelFormat};

#[test]
fn original_binary_search_preserves_portrait_landscape_and_square_patch_grids() {
    let processor = Siglip2ImageProcessor::new(16, 256).expect("valid profile");
    for (height, width, expected) in [
        (480, 640, (224, 288)),
        (640, 480, (288, 224)),
        (480, 480, (256, 256)),
        (100, 800, (96, 672)),
    ] {
        let actual = processor
            .resized_size(ImageSize { height, width })
            .expect("valid geometry");
        assert_eq!((actual.height, actual.width), expected);
    }
}

#[test]
fn patch_padding_is_zero_after_normalization_and_mask_matches_real_patches() {
    let processor = Siglip2ImageProcessor::new(16, 256).expect("valid profile");
    let image =
        ImageFrame::new(640, 480, PixelFormat::Rgb8, vec![255; 640 * 480 * 3]).expect("image");
    let output = processor.preprocess(&image).expect("preprocess");
    assert_eq!(output.spatial_shapes, [14, 18]);
    assert_eq!(output.shape, [1, 256, 768]);
    assert!(output.pixel_values[..252 * 768]
        .iter()
        .all(|v| (*v - 1.0).abs() < 1e-6));
    assert!(output.pixel_values[252 * 768..].iter().all(|v| *v == 0.0));
    assert_eq!(
        output.pixel_attention_mask,
        [vec![1; 252], vec![0; 4]].concat()
    );
}

#[test]
fn invalid_dimensions_and_overflowing_patch_buffers_are_rejected() {
    assert!(Siglip2ImageProcessor::new(0, 256).is_err());
    assert!(Siglip2ImageProcessor::new(16, 0).is_err());
    assert!(Siglip2ImageProcessor::new(usize::MAX, 256).is_err());
    let processor = Siglip2ImageProcessor::new(16, 256).expect("valid profile");
    assert!(processor
        .resized_size(ImageSize {
            height: 0,
            width: 5
        })
        .is_err());
}

#[test]
fn patch_order_is_grid_then_patch_row_then_patch_column_then_rgb() {
    let processor = Siglip2ImageProcessor::new(2, 4).expect("valid profile");
    let bytes = (0..48u8).collect::<Vec<_>>();
    let image = ImageFrame::new(4, 4, PixelFormat::Rgb8, bytes.clone()).expect("image");
    let output = processor.preprocess(&image).expect("preprocess");
    let source_pixels = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    let expected = source_pixels
        .into_iter()
        .flat_map(|pixel| {
            bytes[pixel * 3..pixel * 3 + 3]
                .iter()
                .map(|v| (f32::from(*v) / 255.0 - 0.5) / 0.5)
        })
        .collect::<Vec<_>>();
    assert!(output
        .pixel_values
        .iter()
        .zip(expected)
        .all(|(a, b)| (*a - b).abs() < 2e-7));
    assert_eq!(output.pixel_attention_mask, vec![1; 4]);
}
