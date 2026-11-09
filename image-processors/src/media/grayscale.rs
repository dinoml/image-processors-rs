//! Eight-bit grayscale decoding with codec-specific integer conversion.
//!
//! JPEG luminance follows the decoder's grayscale output rather than a second
//! RGB conversion. PNG follows libpng's png_set_rgb_to_gray(0.299, 0.587),
//! png_do_rgb_to_gray and png_set_strip_16 arithmetic, without linking libpng.

use std::{io::Cursor, path::Path};

use ::image::{ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader};

use super::{ImageFrame, MediaError};

/// Decode an eight-bit grayscale image from a local encoded image.
///
/// JPEG is decoded directly to luminance using the existing TurboJPEG feature.
/// PNG retains native sample depth until grayscale conversion and strips the
/// low byte of sixteen-bit results. Alpha is discarded and EXIF orientation
/// is applied. Existing general RGB/sequence loading APIs remain independent.
///
/// # Errors
/// Reports IO, invalid image data, unsupported image formats or JPEG decoding
/// without the `turbojpeg` feature.
pub fn load_grayscale_image_from_path(path: impl AsRef<Path>) -> Result<ImageFrame, MediaError> {
    decode_grayscale_image_bytes(&std::fs::read(path)?)
}

/// Decode encoded image bytes to eight-bit grayscale, respecting codec precision.
///
/// PNG's integer coefficients are truncated to fifteen bits. Eight-bit sums
/// truncate; sixteen-bit sums round before taking their high byte. JPEG uses
/// native luminance output, including CMYK/YCCK handling, rather than RGB first.
/// BMP, PNM and TIFF use their decoder's rounded fourteen-bit BT.601 policy;
/// PNM strips sixteen-bit samples before that conversion. Other codecs reuse
/// the existing decoded-pixel grayscale conversion.
///
/// # Errors
/// Reports malformed images, unavailable TurboJPEG support or invalid geometry.
pub fn decode_grayscale_image_bytes(bytes: &[u8]) -> Result<ImageFrame, MediaError> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let format = reader.format();
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = if format == Some(ImageFormat::Jpeg) {
        decode_jpeg_grayscale(bytes)?
    } else {
        let image = DynamicImage::from_decoder(decoder)?;
        if format == Some(ImageFormat::Png) {
            png_grayscale(image)
        } else if matches!(
            format,
            Some(ImageFormat::Bmp | ImageFormat::Pnm | ImageFormat::Tiff)
        ) {
            bt601_decoder_grayscale(image, format == Some(ImageFormat::Pnm))
        } else {
            // Keep the existing decoded-frame policy for other codecs.
            let frame = ImageFrame::from_dynamic_image(image)?;
            let gray = crate::transforms::grayscale_opencv(&frame)
                .map_err(|error| image::ImageError::IoError(std::io::Error::other(error)))?;
            let width = u32::try_from(gray.width()).map_err(|_| MediaError::FrameSizeOverflow)?;
            let height = u32::try_from(gray.height()).map_err(|_| MediaError::FrameSizeOverflow)?;
            let actual = gray.data().len();
            let buffer = image::GrayImage::from_raw(width, height, gray.data).ok_or(
                MediaError::InvalidBufferLength {
                    expected: frame.width() * frame.height(),
                    actual,
                },
            )?;
            DynamicImage::ImageLuma8(buffer)
        }
    };
    image.apply_orientation(orientation);
    ImageFrame::from_dynamic_image(image)
}

fn bt601_decoder_grayscale(image: DynamicImage, strip_16: bool) -> DynamicImage {
    let width = image.width();
    let height = image.height();
    // The PNM reader drops low bytes before icvCvt_BGR2Gray_8u_C3C1R.
    // TIFF's eight-bit RGBA reader instead scales RGB16 by (sample+128)/257;
    // the existing DynamicImage conversion supplies that depth conversion.
    let gray = if strip_16
        && matches!(
            image.color(),
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
        ) {
        let rgb = image.into_rgb16();
        image::GrayImage::from_fn(width, height, |x, y| {
            let pixel = rgb.get_pixel(x, y);
            let red = u32::from(pixel[0] >> 8);
            let green = u32::from(pixel[1] >> 8);
            let blue = u32::from(pixel[2] >> 8);
            image::Luma([((red * 4899 + green * 9617 + blue * 1868 + 8192) >> 14) as u8])
        })
    } else {
        let rgb = image.into_rgb8();
        image::GrayImage::from_fn(width, height, |x, y| {
            let pixel = rgb.get_pixel(x, y);
            image::Luma([((u32::from(pixel[0]) * 4899
                + u32::from(pixel[1]) * 9617
                + u32::from(pixel[2]) * 1868
                + 8192)
                >> 14) as u8])
        })
    };
    DynamicImage::ImageLuma8(gray)
}

fn png_grayscale(image: DynamicImage) -> DynamicImage {
    let width = image.width();
    let height = image.height();
    match image {
        DynamicImage::ImageLuma8(_) => image,
        DynamicImage::ImageLumaA8(buffer) => {
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(width, height, |x, y| {
                image::Luma([buffer.get_pixel(x, y)[0]])
            }))
        }
        DynamicImage::ImageLuma16(buffer) => {
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(width, height, |x, y| {
                image::Luma([(buffer.get_pixel(x, y)[0] >> 8) as u8])
            }))
        }
        DynamicImage::ImageLumaA16(buffer) => {
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(width, height, |x, y| {
                image::Luma([(buffer.get_pixel(x, y)[0] >> 8) as u8])
            }))
        }
        other => {
            // libpng png_set_rgb_to_gray_fixed truncates 29900*32768/100000
            // and 58700*32768/100000, then derives blue to sum to 32768.
            let gray = if matches!(other.color(), ColorType::Rgb16 | ColorType::Rgba16) {
                let rgb = other.into_rgb16();
                image::GrayImage::from_fn(width, height, |x, y| {
                    let pixel = rgb.get_pixel(x, y);
                    let sum = u32::from(pixel[0]) * 9797
                        + u32::from(pixel[1]) * 19234
                        + u32::from(pixel[2]) * 3737;
                    image::Luma([((sum + 16384) >> 23) as u8])
                })
            } else {
                let rgb = other.into_rgb8();
                image::GrayImage::from_fn(width, height, |x, y| {
                    let pixel = rgb.get_pixel(x, y);
                    let sum = u32::from(pixel[0]) * 9797
                        + u32::from(pixel[1]) * 19234
                        + u32::from(pixel[2]) * 3737;
                    image::Luma([(sum >> 15) as u8])
                })
            };
            DynamicImage::ImageLuma8(gray)
        }
    }
}

#[cfg(feature = "turbojpeg")]
fn decode_jpeg_grayscale(bytes: &[u8]) -> Result<DynamicImage, MediaError> {
    let header = turbojpeg::read_header(bytes).map_err(super::turbojpeg_decode_error)?;
    let cmyk = matches!(
        header.colorspace,
        turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
    );
    let format = if cmyk {
        turbojpeg::PixelFormat::CMYK
    } else {
        turbojpeg::PixelFormat::GRAY
    };
    let image = turbojpeg::decompress(bytes, format).map_err(super::turbojpeg_decode_error)?;
    let width = u32::try_from(image.width).map_err(|_| MediaError::FrameSizeOverflow)?;
    let height = u32::try_from(image.height).map_err(|_| MediaError::FrameSizeOverflow)?;
    let pixels = if cmyk {
        // grfmt_jpeg.cpp's four-component path uses utils.cpp's inverted-CMYK
        // conversion, with fourteen-bit BT.601 coefficients and descaling.
        image
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| {
                let black = u32::from(pixel[3]);
                let red = black - (((255 - u32::from(pixel[0])) * black) >> 8);
                let green = black - (((255 - u32::from(pixel[1])) * black) >> 8);
                let blue = black - (((255 - u32::from(pixel[2])) * black) >> 8);
                ((red * 4899 + green * 9617 + blue * 1868 + 8192) >> 14) as u8
            })
            .collect()
    } else {
        image.pixels
    };
    let actual = pixels.len();
    let expected = image
        .width
        .checked_mul(image.height)
        .ok_or(MediaError::FrameSizeOverflow)?;
    let buffer = image::GrayImage::from_raw(width, height, pixels)
        .ok_or(MediaError::InvalidBufferLength { expected, actual })?;
    Ok(DynamicImage::ImageLuma8(buffer))
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg_grayscale(_bytes: &[u8]) -> Result<DynamicImage, MediaError> {
    Err(MediaError::TurboJpegUnavailable)
}
