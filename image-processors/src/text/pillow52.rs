//! Pillow 5.2 `_imagingft.c` BASIC layout, getsize and grayscale render policy.
//! This explicitly selects the fallback engine, not Pillow's optional RAQM path.

use std::{borrow::Borrow, io};

use freetype::{
    face::{KerningMode, LoadFlag},
    Face, Library,
};

use super::{Result, TrueTypeFont};

/// Separate legacy `getsize`, `getmask().size` and `getoffset` results.
#[derive(Clone, Copy, Debug)]
pub struct Pillow52TextMetrics {
    /// Pillow 5.2 `FreeTypeFont.getsize`, including the returned offset.
    pub size: [u32; 2],
    /// Pillow 5.2 `FreeTypeFont.getmask().size`, excluding the offset.
    pub mask_size: [u32; 2],
    /// Top-left placement of the mask relative to the requested drawing origin.
    pub offset: [i32; 2],
}

struct GlyphInfo {
    index: u32,
    advance: i64,
}

fn font_error(error: freetype::Error) -> io::Error {
    io::Error::other(error.to_string())
}

// This is CEIL, including for negative values, not modern Pillow's ROUND.
fn pixel(value: i64) -> i64 {
    (value + 63) >> 6
}

// FreeType positions use C long: 32 bits on Windows and 64 bits on Linux.
#[allow(clippy::unnecessary_cast)]
fn position(value: freetype::ffi::FT_Pos) -> i64 {
    value as i64
}

fn dimension(value: i64) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "font extent overflow"))
}

fn layout<T: Borrow<[u8]>>(face: &Face<T>, text: &str) -> Result<Vec<GlyphInfo>> {
    let mut glyphs: Vec<GlyphInfo> = Vec::with_capacity(text.chars().count());
    let mut previous = 0;
    for character in text.chars() {
        let index = face.get_char_index(character as usize).unwrap_or(0);
        face.load_glyph(index, LoadFlag::RENDER | LoadFlag::NO_BITMAP)
            .map_err(font_error)?;
        if face.has_kerning() && previous != 0 && index != 0 {
            // Legacy BASIC adds PIXEL(delta.x) to the previous 26.6 advance.
            // Preserve that historical unit boundary rather than fixing it.
            if let Ok(delta) = face.get_kerning(previous, index, KerningMode::KerningDefault) {
                if let Some(last) = glyphs.last_mut() {
                    last.advance += pixel(position(delta.x));
                }
            }
        }
        glyphs.push(GlyphInfo {
            index,
            advance: position(face.glyph().metrics().horiAdvance),
        });
        previous = index;
    }
    Ok(glyphs)
}

fn metrics<T: Borrow<[u8]>>(face: &Face<T>, glyphs: &[GlyphInfo]) -> Result<Pillow52TextMetrics> {
    let mut x = 0_i64;
    let mut xoffset = 0_i64;
    let mut bearing_y = 0_i64;
    let mut ymin = 0_i64;
    let mut ymax = 0_i64;
    for (i, info) in glyphs.iter().enumerate() {
        face.load_glyph(info.index, LoadFlag::DEFAULT | LoadFlag::NO_BITMAP)
            .map_err(font_error)?;
        let slot = face.glyph();
        let metric = slot.metrics();
        if i == 0 && metric.horiBearingX < 0 {
            xoffset = position(metric.horiBearingX);
            x -= xoffset;
        }
        x += info.advance;
        if i + 1 == glyphs.len() {
            let offset = info.advance - position(metric.width) - position(metric.horiBearingX);
            if offset < 0 {
                x -= offset;
            }
        }
        // FT_Glyph_Get_CBox(SUBPIXELS) is the outline control box, not a
        // curve-extrema box or a rendered bitmap box. NO_BITMAP selects outlines.
        let bbox = slot
            .get_glyph()
            .map_err(font_error)?
            .get_cbox(freetype::ffi::FT_GLYPH_BBOX_SUBPIXELS);
        ymin = ymin.min(position(bbox.yMin));
        ymax = ymax.max(position(bbox.yMax));
        bearing_y = bearing_y.max(position(metric.horiBearingY));
    }
    let yoffset = if glyphs.is_empty() {
        0
    } else {
        x -= xoffset;
        let size = face
            .size_metrics()
            .ok_or_else(|| io::Error::other("missing font size metrics"))?;
        pixel(position(size.ascender) - bearing_y)
    };
    let mask_size = [dimension(pixel(x))?, dimension(pixel(ymax - ymin))?];
    let offset = [
        i32::try_from(pixel(xoffset)).map_err(|_| io::Error::other("font offset overflow"))?,
        i32::try_from(yoffset).map_err(|_| io::Error::other("font offset overflow"))?,
    ];
    Ok(Pillow52TextMetrics {
        size: [
            dimension(i64::from(mask_size[0]) + i64::from(offset[0]))?,
            dimension(i64::from(mask_size[1]) + i64::from(offset[1]))?,
        ],
        mask_size,
        offset,
    })
}

impl TrueTypeFont {
    fn pillow52_face(&self, pixel_size: u32) -> Result<Face<&[u8]>> {
        if pixel_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero font size",
            ));
        }
        let library = Library::init().map_err(font_error)?;
        // The face retains the library and borrows this font's existing buffer.
        let face = library
            .new_memory_face2(self.data.as_slice(), 0)
            .map_err(font_error)?;
        face.set_pixel_sizes(0, pixel_size).map_err(font_error)?;
        Ok(face)
    }

    /// Measure with the hinted FreeType BASIC policy from Pillow 5.2.0.
    /// RAQM shaping is deliberately not selected or inferred from the host.
    /// # Errors
    /// Returns errors from font loading, sizing or glyph layout.
    pub fn measure_pillow52_basic(
        &self,
        text: &str,
        pixel_size: u32,
    ) -> Result<Pillow52TextMetrics> {
        let face = self.pillow52_face(pixel_size)?;
        metrics(&face, &layout(&face, text)?)
    }

    /// Draw black RGB text using Pillow 5.2 BASIC grayscale masks and legacy offsets.
    /// Glyph overlap uses maximum coverage before one RGB blend, as in Pillow.
    /// Existing outline overlay methods and their stroke policy are unchanged.
    /// # Errors
    /// Returns errors for invalid image geometry, font data or overflowing extents.
    pub fn draw_rgb_pillow52_basic(
        &self,
        pixels: &mut [u8],
        width: usize,
        height: usize,
        text: &str,
        origin: [i32; 2],
        pixel_size: u32,
    ) -> Result<()> {
        if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(pixels.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid RGB text target",
            ));
        }
        let face = self.pillow52_face(pixel_size)?;
        let glyphs = layout(&face, text)?;
        let geometry = metrics(&face, &glyphs)?;
        let [mw, mh] = geometry.mask_size.map(|value| value as usize);
        let length = mw
            .checked_mul(mh)
            .ok_or_else(|| io::Error::other("text mask overflow"))?;
        let mut mask = vec![0_u8; length];
        let flags = LoadFlag::RENDER | LoadFlag::NO_BITMAP;
        let mut ascender = 0_i64;
        let mut last_bearing_x = 0_i64;
        for info in &glyphs {
            face.load_glyph(info.index, flags).map_err(font_error)?;
            let slot = face.glyph();
            ascender = ascender.max(i64::from(slot.bitmap().rows() - slot.bitmap_top()));
            last_bearing_x = position(slot.metrics().horiBearingX);
        }
        // Preserve the legacy render loop's initial last-glyph bearing check.
        let mut x = if last_bearing_x < 0 {
            -last_bearing_x
        } else {
            0
        };
        for (i, info) in glyphs.iter().enumerate() {
            face.load_glyph(info.index, flags).map_err(font_error)?;
            let slot = face.glyph();
            if i == 0 && slot.metrics().horiBearingX < 0 {
                x = -position(slot.metrics().horiBearingX);
            }
            let bitmap = slot.bitmap();
            if bitmap.width() == 0 || bitmap.rows() == 0 {
                x += info.advance;
                continue;
            }
            let pitch = bitmap.pitch();
            if pitch < 0
                || bitmap.pixel_mode().map_err(font_error)? != freetype::bitmap::PixelMode::Gray
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported font bitmap layout",
                ));
            }
            let xx = pixel(x) + i64::from(slot.bitmap_left());
            let yy = mh as i64 - pixel(position(slot.metrics().horiBearingY)) - ascender;
            for row in 0..bitmap.rows() as usize {
                let dy = yy + row as i64;
                if !(0..mh as i64).contains(&dy) {
                    continue;
                }
                for column in 0..bitmap.width() as usize {
                    let dx = xx + column as i64;
                    if (0..mw as i64).contains(&dx) {
                        let source = bitmap.buffer()[row * pitch as usize + column];
                        let target = &mut mask[dy as usize * mw + dx as usize];
                        *target = (*target).max(source);
                    }
                }
            }
            x += info.advance;
        }
        for row in 0..mh {
            let y = i64::from(origin[1]) + i64::from(geometry.offset[1]) + row as i64;
            if !(0..height as i64).contains(&y) {
                continue;
            }
            for column in 0..mw {
                let x = i64::from(origin[0]) + i64::from(geometry.offset[0]) + column as i64;
                if !(0..width as i64).contains(&x) {
                    continue;
                }
                let alpha = u32::from(mask[row * mw + column]);
                let at = (y as usize * width + x as usize) * 3;
                for channel in 0..3 {
                    let value = u32::from(pixels[at + channel]) * (255 - alpha) + 128;
                    pixels[at + channel] = ((value + (value >> 8)) >> 8) as u8;
                }
            }
        }
        Ok(())
    }
}
