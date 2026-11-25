//! In-memory JPEG reference preprocessing using the existing TurboJPEG feature.

use super::{ImageFrame, MediaError};

/// Encode an RGB reference with baseline JPEG 4:2:0 and decode it back to RGB.
///
/// Uses libjpeg's standard quality tables and default unoptimized Huffman tables,
/// matching Pillow's ordinary RGB JPEG save policy. Geometry, alpha compositing
/// and orientation belong to the caller's existing preprocessing stages. The
/// source timing is retained, and no files are written.
///
/// # Errors
/// Rejects non-RGB frames and quality outside 1..=100. Reports codec failures,
/// or `TurboJpegUnavailable` when built without the existing `turbojpeg` feature.
pub fn jpeg_reference_roundtrip(frame: &ImageFrame, quality: u8) -> Result<ImageFrame, MediaError> {
    if frame.pixel_format() != super::PixelFormat::Rgb8 || !(1..=100).contains(&quality) {
        return Err(MediaError::InvalidJpegReference(
            "supply RGB8 pixels and JPEG quality in 1..=100".into(),
        ));
    }
    roundtrip(frame, quality)
}

#[cfg(feature = "turbojpeg")]
fn roundtrip(frame: &ImageFrame, quality: u8) -> Result<ImageFrame, MediaError> {
    let mut compressor = turbojpeg::Compressor::new().map_err(super::turbojpeg_decode_error)?;
    compressor
        .set_quality(i32::from(quality))
        .map_err(super::turbojpeg_decode_error)?;
    compressor
        .set_subsamp(turbojpeg::Subsamp::Sub2x2)
        .map_err(super::turbojpeg_decode_error)?;
    let bytes = compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: frame.data(),
            width: frame.width(),
            height: frame.height(),
            pitch: frame.width() * 3,
            format: turbojpeg::PixelFormat::RGB,
        })
        .map_err(super::turbojpeg_decode_error)?;
    Ok(super::decode_jpeg_bytes_turbo(&bytes)?.with_timing(frame.timing()))
}

#[cfg(not(feature = "turbojpeg"))]
fn roundtrip(_frame: &ImageFrame, _quality: u8) -> Result<ImageFrame, MediaError> {
    Err(MediaError::TurboJpegUnavailable)
}
