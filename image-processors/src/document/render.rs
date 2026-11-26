//! Dependency-free PDF page rasterization using the shared native font raster.

use super::{
    pdf::{Parser, PdfDocument, Value},
    DocumentError, PositionedCharacter, PositionedTextRun, TextLayerStatistics,
};
use crate::{
    media::{ImageFrame, PixelFormat},
    text::{
        gray::{self, Position, Segment},
        TrueTypeFont,
    },
};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, DocumentError>;
type Matrix = [f64; 6];
const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
fn invalid(message: &'static str) -> DocumentError {
    DocumentError::InvalidPdf(message)
}
fn multiply(a: Matrix, b: Matrix) -> Matrix {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}
fn point(matrix: Matrix, x: f64, y: f64) -> [f64; 2] {
    [
        matrix[0] * x + matrix[2] * y + matrix[4],
        matrix[1] * x + matrix[3] * y + matrix[5],
    ]
}
fn position(point: [f64; 2]) -> Position {
    Position {
        x: (point[0] * 64.0).round() as i64,
        y: (-point[1] * 64.0).round() as i64,
    }
}
fn number(value: &Value) -> Result<f64> {
    if let Value::Number(value) = value {
        Ok(*value)
    } else {
        Err(invalid("expected numeric operand"))
    }
}
fn numbers<const N: usize>(values: &[Value]) -> Result<[f64; N]> {
    if values.len() != N {
        return Err(invalid("wrong content operand count"));
    }
    let mut out = [0.0; N];
    for (out, value) in out.iter_mut().zip(values) {
        *out = number(value)?;
    }
    Ok(out)
}
fn page_geometry(
    document: &PdfDocument,
    page: &BTreeMap<String, Value>,
) -> Result<([f64; 2], Matrix)> {
    let media = document.resolve(
        page.get("MediaBox")
            .ok_or_else(|| invalid("page lacks MediaBox"))?,
    )?;
    let Value::Array(media) = media else {
        return Err(invalid("invalid MediaBox"));
    };
    let media = numbers::<4>(&media)?;
    let mut crop = media;
    if let Some(value) = page.get("CropBox") {
        let Value::Array(values) = document.resolve(value)? else {
            return Err(invalid("invalid CropBox"));
        };
        crop = numbers::<4>(&values)?;
        crop[0] = crop[0].max(media[0]);
        crop[1] = crop[1].max(media[1]);
        crop[2] = crop[2].min(media[2]);
        crop[3] = crop[3].min(media[3]);
    }
    let (w, h) = (crop[2] - crop[0], crop[3] - crop[1]);
    if w <= 0.0 || h <= 0.0 {
        return Err(invalid("empty page crop"));
    }
    let rotation = page
        .get("Rotate")
        .map(|value| document.resolve(value).and_then(|value| number(&value)))
        .transpose()?
        .unwrap_or(0.0);
    if rotation.fract() != 0.0 || rotation % 90.0 != 0.0 {
        return Err(invalid("invalid page rotation"));
    }
    Ok(match (rotation as i64).rem_euclid(360) {
        0 => ([w, h], [1.0, 0.0, 0.0, -1.0, -crop[0], crop[3]]),
        90 => ([h, w], [0.0, 1.0, 1.0, 0.0, -crop[1], -crop[0]]),
        180 => ([w, h], [-1.0, 0.0, 0.0, 1.0, crop[2], -crop[1]]),
        _ => ([h, w], [0.0, -1.0, -1.0, 0.0, crop[3], crop[2]]),
    })
}

fn unicode_text(bytes: &[u8]) -> Result<String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(invalid("odd UTF16 ToUnicode destination"));
    }
    let units = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    String::from_utf16(&units).map_err(|_| invalid("invalid UTF16 ToUnicode destination"))
}
fn unicode_map(bytes: &[u8]) -> Result<BTreeMap<Vec<u8>, String>> {
    let mut parser = Parser { bytes, at: 0 };
    let mut output = BTreeMap::new();
    while let Some(value) = parser.next_value()? {
        match value {
            Value::Keyword(op) if op == "beginbfchar" => loop {
                let value = parser
                    .next_value()?
                    .ok_or_else(|| invalid("unterminated bfchar"))?;
                if matches!(&value,Value::Keyword(op) if op=="endbfchar") {
                    break;
                }
                let Value::Bytes(source) = value else {
                    return Err(invalid("invalid bfchar source"));
                };
                let Some(Value::Bytes(destination)) = parser.next_value()? else {
                    return Err(invalid("invalid bfchar destination"));
                };
                output.insert(source, unicode_text(&destination)?);
                if output.len() > 65536 {
                    return Err(invalid("ToUnicode map exceeds limit"));
                }
            },
            Value::Keyword(op) if op == "beginbfrange" => loop {
                let value = parser
                    .next_value()?
                    .ok_or_else(|| invalid("unterminated bfrange"))?;
                if matches!(&value,Value::Keyword(op) if op=="endbfrange") {
                    break;
                }
                let Value::Bytes(first) = value else {
                    return Err(invalid("invalid bfrange source"));
                };
                let Some(Value::Bytes(last)) = parser.next_value()? else {
                    return Err(invalid("invalid bfrange end"));
                };
                if first.is_empty() || first.len() > 4 || last.len() != first.len() {
                    return Err(invalid("invalid bfrange code width"));
                }
                let first_value = first.iter().fold(0u32, |a, b| (a << 8) | u32::from(*b));
                let last_value = last.iter().fold(0u32, |a, b| (a << 8) | u32::from(*b));
                let count = last_value
                    .checked_sub(first_value)
                    .and_then(|n| n.checked_add(1))
                    .filter(|&n| n <= 65536)
                    .ok_or_else(|| invalid("invalid bfrange length"))?;
                let destination = parser
                    .next_value()?
                    .ok_or_else(|| invalid("missing bfrange destination"))?;
                for offset in 0..count {
                    let key = (first_value + offset).to_be_bytes();
                    let key = key[4 - first.len()..].to_vec();
                    let bytes = match &destination {
                        Value::Bytes(start) => {
                            let mut bytes = start.clone();
                            let mut carry = offset;
                            for byte in bytes.iter_mut().rev() {
                                let sum = u32::from(*byte) + (carry & 255);
                                *byte = sum as u8;
                                carry = (carry >> 8) + (sum >> 8);
                            }
                            if carry != 0 {
                                return Err(invalid("bfrange destination overflow"));
                            }
                            bytes
                        }
                        Value::Array(values) => match values.get(offset as usize) {
                            Some(Value::Bytes(bytes)) => bytes.clone(),
                            _ => return Err(invalid("invalid bfrange destination array")),
                        },
                        _ => return Err(invalid("invalid bfrange destination")),
                    };
                    output.insert(key, unicode_text(&bytes)?);
                    if output.len() > 65536 {
                        return Err(invalid("ToUnicode map exceeds limit"));
                    }
                }
            },
            _ => {}
        }
    }
    Ok(output)
}

/// Explicit font assets and resource limits for native PDF rendering.
///
/// Nonembedded fonts are resolved only through caller-selected assets. No
/// platform font substitution, PDFium or FreeType is loaded. Native outline
/// rasterization is a distinct profile from any PDFium Helvetica fixture.
#[derive(Default)]
pub struct PdfRenderOptions {
    /// Maps PDF BaseFont names, without subset prefixes, to sfnt font bytes.
    pub fonts: BTreeMap<String, Vec<u8>>,
    /// Maximum page and intermediate bitmap pixels; zero selects 100 megapixels.
    pub pixel_limit: usize,
    /// Maximum decoded content/font stream bytes; zero selects 64 MiB.
    pub stream_limit: usize,
}

/// A raster and text layer in one cropped, rotated page coordinate frame.
pub struct RenderedPdfPage {
    /// Eight-bit grayscale page raster.
    pub image: ImageFrame,
    /// Cropped and rotated page width/height in PDF points.
    pub size: [f64; 2],
    /// Upright, on-page text runs in source order.
    pub runs: Vec<PositionedTextRun>,
    /// Source counts before rotated and off-page characters are removed.
    pub statistics: TextLayerStatistics,
}

#[derive(Clone)]
struct State {
    matrix: Matrix,
    clip: Vec<u8>,
    fill: u8,
    stroke: u8,
    width: f64,
    text: Matrix,
    line: Matrix,
    font: String,
    size: f64,
    leading: f64,
    spacing: f64,
    word_spacing: f64,
    horizontal: f64,
    rise: f64,
    mode: u8,
}

struct Font {
    face: TrueTypeFont,
    first: usize,
    widths: Vec<f64>,
    unicode: BTreeMap<Vec<u8>, String>,
}

struct Renderer<'a> {
    document: &'a PdfDocument,
    options: &'a PdfRenderOptions,
    resources: BTreeMap<String, Value>,
    fonts: BTreeMap<String, Font>,
    width: usize,
    height: usize,
    size: [f64; 2],
    scale: f64,
    view: Matrix,
    pixels: Vec<u8>,
    state: State,
    stack: Vec<State>,
    path: Vec<Segment>,
    start: Option<Position>,
    current: Option<Position>,
    clipping: bool,
    runs: Vec<PositionedTextRun>,
    pending: Vec<PositionedCharacter>,
    statistics: TextLayerStatistics,
    content_index: usize,
    in_text: bool,
    extract_text: bool,
    form_depth: usize,
}

impl PdfDocument {
    /// Cropped, rotated page dimensions in PDF points without rendering text.
    /// # Errors
    /// Reports invalid page indices, boxes or page rotation.
    pub fn page_size(&self, index: usize) -> Result<[f64; 2]> {
        let pages = self.pages()?;
        let page = pages
            .get(index)
            .ok_or_else(|| invalid("page index outside document"))?;
        Ok(page_geometry(self, page)?.0)
    }
    /// Render a page and extract positioned upright text using native owners.
    ///
    /// # Errors
    /// Reports malformed pages, missing explicit font assets, resource limits
    /// and unsupported PDF features. Unsupported operators never disappear
    /// silently. Font and antialiasing parity is tied to the supplied assets.
    pub fn render_page(
        &self,
        index: usize,
        scale: f64,
        options: &PdfRenderOptions,
    ) -> Result<RenderedPdfPage> {
        self.render_page_impl(index, scale, options, true)
    }

    /// Render a page for OCR without extracting or assessing its text layer.
    /// # Errors
    /// Uses the same format and resource boundaries as [`Self::render_page`].
    pub fn render_ocr_page(
        &self,
        index: usize,
        scale: f64,
        options: &PdfRenderOptions,
    ) -> Result<ImageFrame> {
        Ok(self.render_page_impl(index, scale, options, false)?.image)
    }

    fn render_page_impl(
        &self,
        index: usize,
        scale: f64,
        options: &PdfRenderOptions,
        extract_text: bool,
    ) -> Result<RenderedPdfPage> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(DocumentError::InvalidPage("invalid PDF render scale"));
        }
        let pages = self.pages()?;
        let page = pages
            .get(index)
            .ok_or_else(|| invalid("page index outside document"))?;
        let (size, view) = page_geometry(self, page)?;
        let (width, height) = (
            (size[0] * scale).ceil() as usize,
            (size[1] * scale).ceil() as usize,
        );
        let limit = if options.pixel_limit == 0 {
            100_000_000
        } else {
            options.pixel_limit
        };
        let count = width
            .checked_mul(height)
            .filter(|&n| n > 0 && n <= limit)
            .ok_or_else(|| invalid("page raster exceeds pixel limit"))?;
        let resources = page
            .get("Resources")
            .map(|value| self.dictionary(value))
            .transpose()?
            .unwrap_or_default();
        let state = State {
            matrix: IDENTITY,
            clip: vec![255; count],
            fill: 0,
            stroke: 0,
            width: 1.0,
            text: IDENTITY,
            line: IDENTITY,
            font: String::new(),
            size: 0.0,
            leading: 0.0,
            spacing: 0.0,
            word_spacing: 0.0,
            horizontal: 1.0,
            rise: 0.0,
            mode: 0,
        };
        let mut renderer = Renderer {
            document: self,
            options,
            resources,
            fonts: BTreeMap::new(),
            width,
            height,
            size,
            scale,
            view,
            pixels: vec![255; count],
            state,
            stack: Vec::new(),
            path: Vec::new(),
            start: None,
            current: None,
            clipping: false,
            runs: Vec::new(),
            pending: Vec::new(),
            statistics: TextLayerStatistics::default(),
            content_index: 0,
            in_text: false,
            extract_text,
            form_depth: 0,
        };
        if let Some(contents) = page.get("Contents") {
            let values = match self.resolve(contents)? {
                Value::Array(values) => values,
                _ => vec![contents.clone()],
            };
            let mut program = Vec::new();
            for value in values {
                let (_, bytes) = self.stream(&value, renderer.stream_limit())?;
                if program
                    .len()
                    .checked_add(bytes.len())
                    .is_none_or(|n| n > renderer.stream_limit())
                {
                    return Err(invalid("combined content stream exceeds limit"));
                }
                program.extend(bytes);
                program.push(b'\n');
            }
            renderer.program(&program)?;
        }
        renderer.flush_run();
        Ok(RenderedPdfPage {
            image: ImageFrame::new(width, height, PixelFormat::Luma8, renderer.pixels)?,
            size,
            runs: renderer.runs,
            statistics: renderer.statistics,
        })
    }
}

impl Renderer<'_> {
    fn stream_limit(&self) -> usize {
        if self.options.stream_limit == 0 {
            64 * 1024 * 1024
        } else {
            self.options.stream_limit
        }
    }
    fn transform(&self) -> Matrix {
        multiply(
            [self.scale, 0.0, 0.0, self.scale, 0.0, 0.0],
            multiply(self.view, self.state.matrix),
        )
    }
    fn mapped(&self, x: f64, y: f64) -> Result<Position> {
        let p = point(self.transform(), x, y);
        if p.iter().any(|v| !v.is_finite() || v.abs() > 1_000_000.0) {
            return Err(invalid("path coordinate exceeds limit"));
        }
        Ok(position(p))
    }
    fn close(&mut self) {
        if let Some(start) = self.start {
            self.path.push(Segment::Line(start));
            self.current = Some(start);
        }
    }
    fn closed_path(&self) -> Vec<Segment> {
        let mut result = Vec::new();
        let mut start = None;
        for &segment in &self.path {
            if let Segment::Move(p) = segment {
                if let Some(p) = start {
                    result.push(Segment::Line(p));
                }
                start = Some(p);
            }
            result.push(segment);
        }
        if let Some(p) = start {
            result.push(Segment::Line(p));
        }
        result
    }
    fn coverage(&self, segments: &[Segment]) -> Result<Vec<u8>> {
        let mut bounds = [i64::MAX, i64::MAX, i64::MIN, i64::MIN];
        for segment in segments {
            let points: &[Position] = match segment {
                Segment::Move(p) | Segment::Line(p) => std::slice::from_ref(p),
                Segment::Conic(a, b) => {
                    for p in [a, b] {
                        bounds[0] = bounds[0].min(p.x);
                        bounds[1] = bounds[1].min(p.y);
                        bounds[2] = bounds[2].max(p.x);
                        bounds[3] = bounds[3].max(p.y);
                    }
                    continue;
                }
                Segment::Cubic(a, b, c) => {
                    for p in [a, b, c] {
                        bounds[0] = bounds[0].min(p.x);
                        bounds[1] = bounds[1].min(p.y);
                        bounds[2] = bounds[2].max(p.x);
                        bounds[3] = bounds[3].max(p.y);
                    }
                    continue;
                }
            };
            for p in points {
                bounds[0] = bounds[0].min(p.x);
                bounds[1] = bounds[1].min(p.y);
                bounds[2] = bounds[2].max(p.x);
                bounds[3] = bounds[3].max(p.y);
            }
        }
        if segments.is_empty() {
            return Ok(vec![0; self.pixels.len()]);
        }
        let extent = ((bounds[2] - bounds[0]) / 64 + 2)
            .checked_mul((bounds[3] - bounds[1]) / 64 + 2)
            .ok_or_else(|| invalid("path bounds overflow"))?;
        let limit = if self.options.pixel_limit == 0 {
            100_000_000
        } else {
            self.options.pixel_limit
        };
        if extent < 0 || extent as u64 > limit as u64 {
            return Err(invalid("path raster exceeds pixel limit"));
        }
        let bitmap = gray::rasterize(segments)?;
        let mut coverage = vec![0; self.pixels.len()];
        for y in 0..bitmap.height {
            for x in 0..bitmap.width {
                let xx = i64::from(bitmap.left) + x as i64;
                let yy = -(i64::from(bitmap.top)) + y as i64;
                if xx >= 0 && yy >= 0 && xx < self.width as i64 && yy < self.height as i64 {
                    coverage[yy as usize * self.width + xx as usize] =
                        bitmap.pixels[y * bitmap.width + x];
                }
            }
        }
        Ok(coverage)
    }
    fn paint(&mut self, segments: &[Segment], color: u8) -> Result<()> {
        let coverage = self.coverage(segments)?;
        for ((pixel, &alpha), &clip) in self.pixels.iter_mut().zip(&coverage).zip(&self.state.clip)
        {
            let alpha = (u32::from(alpha) * u32::from(clip) + 127) / 255;
            *pixel =
                ((u32::from(*pixel) * (255 - alpha) + u32::from(color) * alpha + 127) / 255) as u8;
        }
        Ok(())
    }
    fn finish_path(&mut self, fill: bool, stroke: bool) -> Result<()> {
        if fill {
            self.paint(&self.closed_path(), self.state.fill)?;
        }
        if stroke {
            let mut outline = Vec::new();
            let mut previous = None;
            let radius = self.state.width
                * self.scale
                * (self.state.matrix[0].hypot(self.state.matrix[1]))
                * 32.0;
            for &segment in &self.path {
                match segment {
                    Segment::Move(p) => previous = Some(p),
                    Segment::Line(p) => {
                        if let Some(a) = previous {
                            if a.x != p.x && a.y != p.y {
                                return Err(DocumentError::UnsupportedPdf(
                                    "non-axis-aligned stroked paths",
                                ));
                            }
                            let r = radius.round() as i64;
                            let (l, b, rgt, t) = (
                                a.x.min(p.x) - r,
                                a.y.min(p.y) - r,
                                a.x.max(p.x) + r,
                                a.y.max(p.y) + r,
                            );
                            outline.extend([
                                Segment::Move(Position { x: l, y: b }),
                                Segment::Line(Position { x: rgt, y: b }),
                                Segment::Line(Position { x: rgt, y: t }),
                                Segment::Line(Position { x: l, y: t }),
                                Segment::Line(Position { x: l, y: b }),
                            ]);
                        }
                        previous = Some(p);
                    }
                    _ => return Err(DocumentError::UnsupportedPdf("curved stroked paths")),
                }
            }
            self.paint(&outline, self.state.stroke)?;
        }
        if self.clipping {
            let coverage = self.coverage(&self.closed_path())?;
            for (clip, alpha) in self.state.clip.iter_mut().zip(coverage) {
                *clip = ((u32::from(*clip) * u32::from(alpha) + 127) / 255) as u8;
            }
        }
        self.path.clear();
        self.start = None;
        self.current = None;
        self.clipping = false;
        Ok(())
    }
    fn flush_run(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let characters = std::mem::take(&mut self.pending);
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for c in &characters {
            if !crate::text::is_text_whitespace(c.character) {
                bounds[0] = bounds[0].min(c.bounds[0]);
                bounds[1] = bounds[1].min(c.bounds[1]);
                bounds[2] = bounds[2].max(c.bounds[2]);
                bounds[3] = bounds[3].max(c.bounds[3]);
            }
        }
        if !bounds[0].is_finite() {
            return;
        }
        let raw = characters.iter().map(|c| c.character).collect::<String>();
        self.runs.push(PositionedTextRun {
            text: raw
                .split(crate::text::is_text_whitespace)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            bounds,
            characters,
        });
    }
    fn font(&mut self) -> Result<()> {
        if self.fonts.contains_key(&self.state.font) {
            return Ok(());
        }
        let fonts = self.document.dictionary(
            self.resources
                .get("Font")
                .ok_or_else(|| invalid("text lacks font resources"))?,
        )?;
        let font = self.document.dictionary(
            fonts
                .get(&self.state.font)
                .ok_or_else(|| invalid("unknown font resource"))?,
        )?;
        if !matches!(font.get("Subtype"),Some(Value::Name(name)) if name=="TrueType" || name=="Type1")
        {
            return Err(DocumentError::UnsupportedPdf("font subtype"));
        }
        let name = match font.get("BaseFont") {
            Some(Value::Name(name)) => name.rsplit('+').next().unwrap_or(name),
            _ => return Err(invalid("font lacks BaseFont")),
        };
        let mut bytes = None;
        if let Some(descriptor) = font.get("FontDescriptor") {
            let descriptor = self.document.dictionary(descriptor)?;
            if let Some(file) = descriptor.get("FontFile2") {
                bytes = Some(self.document.stream(file, self.stream_limit())?.1);
            }
        }
        let bytes = bytes
            .or_else(|| self.options.fonts.get(name).cloned())
            .ok_or(DocumentError::UnsupportedPdf(
                "nonembedded font requires an explicit caller font asset",
            ))?;
        let first = font
            .get("FirstChar")
            .map(number)
            .transpose()?
            .unwrap_or(0.0) as usize;
        let widths = if let Some(widths) = font.get("Widths") {
            let Value::Array(values) = self.document.resolve(widths)? else {
                return Err(invalid("invalid font Widths"));
            };
            values.iter().map(number).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let unicode = font
            .get("ToUnicode")
            .map(|source| {
                self.document
                    .stream(source, self.stream_limit())
                    .and_then(|(_, bytes)| unicode_map(&bytes))
            })
            .transpose()?
            .unwrap_or_default();
        if let Some(encoding) = font.get("Encoding") {
            if !matches!(self.document.resolve(encoding)?,Value::Name(ref name) if name=="WinAnsiEncoding" || name=="StandardEncoding")
            {
                return Err(DocumentError::UnsupportedPdf("font encoding differences"));
            }
        }
        if unicode.keys().any(|key| key.len() != 1) {
            return Err(DocumentError::UnsupportedPdf("multibyte simple font maps"));
        }
        self.fonts.insert(
            self.state.font.clone(),
            Font {
                face: TrueTypeFont::from_bytes(bytes)?,
                first,
                widths,
                unicode,
            },
        );
        Ok(())
    }
    fn show(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.in_text || self.state.size <= 0.0 {
            return Err(invalid("text shown outside initialized text state"));
        }
        self.font()?;
        if !matches!(self.state.mode, 0 | 3) {
            return Err(DocumentError::UnsupportedPdf("stroked or clipping text"));
        }
        for &code in bytes {
            let font = &self.fonts[&self.state.font];
            let mapped = font.unicode.get(&vec![code]);
            let character = if let Some(mapped) = mapped {
                let mut chars = mapped.chars();
                let c = chars
                    .next()
                    .ok_or_else(|| invalid("empty Unicode glyph map"))?;
                if chars.next().is_some() {
                    return Err(DocumentError::UnsupportedPdf("multi-scalar glyph mappings"));
                }
                c
            } else if code < 128 {
                char::from(code)
            } else {
                return Err(DocumentError::UnsupportedPdf(
                    "non-ASCII simple font encoding without Unicode map",
                ));
            };
            let local = multiply(
                self.state.text,
                [
                    self.state.size * self.state.horizontal,
                    0.0,
                    0.0,
                    self.state.size,
                    0.0,
                    self.state.rise,
                ],
            );
            let transform = multiply(self.transform(), local);
            if transform
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 1_000_000.0)
            {
                return Err(invalid("glyph transform exceeds limit"));
            }
            let font = &self.fonts[&self.state.font];
            let pdf_advance = font
                .widths
                .get(usize::from(code).wrapping_sub(font.first))
                .map(|width| width.trunc() / 1000.0);
            let (segments, advance, bounds) =
                font.face
                    .document_glyph(character, transform, self.state.size, pdf_advance)?;
            let bounds = bounds.map(|v| v / self.scale);
            let vector = point(
                multiply(self.view, multiply(self.state.matrix, self.state.text)),
                1.0,
                0.0,
            );
            let origin = point(
                multiply(self.view, multiply(self.state.matrix, self.state.text)),
                0.0,
                0.0,
            );
            let rotated = vector[0] - origin[0] <= 0.0
                || (vector[1] - origin[1]).abs() > 1e-3 * (vector[0] - origin[0]).abs().max(1e-9);
            let broken = super::is_broken_document_character(character);
            if self.extract_text && !crate::text::is_text_whitespace(character) {
                self.statistics.characters += 1;
                self.statistics.rotated_characters += usize::from(rotated);
                self.statistics.bad_characters += usize::from(broken);
            }
            let center = [(bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5];
            if self.extract_text {
                if rotated && !crate::text::is_text_whitespace(character) {
                    self.flush_run();
                } else if center[0] >= 0.0
                    && center[0] <= self.size[0]
                    && center[1] >= 0.0
                    && center[1] <= self.size[1]
                {
                    self.pending.push(PositionedCharacter {
                        character,
                        bounds,
                        content_index: self.content_index,
                        render_mode: self.state.mode,
                        rotated,
                        broken_font_map: broken,
                    });
                }
            }
            if self.state.mode == 0 {
                self.paint(&segments, self.state.fill)?;
            }
            self.content_index += 1;
            let advance = (advance * self.state.size
                + self.state.spacing
                + if code == b' ' {
                    self.state.word_spacing
                } else {
                    0.0
                })
                * self.state.horizontal;
            self.state.text = multiply(self.state.text, [1.0, 0.0, 0.0, 1.0, advance, 0.0]);
        }
        Ok(())
    }
    fn next_line(&mut self) {
        self.flush_run();
        self.state.line = multiply(
            self.state.line,
            [1.0, 0.0, 0.0, 1.0, 0.0, -self.state.leading],
        );
        self.state.text = self.state.line;
    }
    fn xobject(&mut self, name: &str) -> Result<()> {
        let objects = self.document.dictionary(
            self.resources
                .get("XObject")
                .ok_or_else(|| invalid("missing XObject resources"))?,
        )?;
        let source = objects
            .get(name)
            .ok_or_else(|| invalid("unknown XObject resource"))?;
        let (dictionary, bytes) = self.document.stream(source, self.stream_limit())?;
        match dictionary.get("Subtype") {
            Some(Value::Name(kind)) if kind == "Form" => {
                if self.form_depth >= 32 {
                    return Err(invalid("form nesting exceeds limit"));
                }
                self.flush_run();
                let state = self.state.clone();
                let resources = self.resources.clone();
                let fonts = std::mem::take(&mut self.fonts);
                let path = std::mem::take(&mut self.path);
                let start = self.start.take();
                let current = self.current.take();
                let clipping = self.clipping;
                self.clipping = false;
                if let Some(value) = dictionary.get("Matrix") {
                    let Value::Array(values) = self.document.resolve(value)? else {
                        return Err(invalid("invalid form Matrix"));
                    };
                    self.state.matrix = multiply(self.state.matrix, numbers::<6>(&values)?);
                }
                if let Some(value) = dictionary.get("Resources") {
                    self.resources = self.document.dictionary(value)?;
                }
                if let Some(value) = dictionary.get("BBox") {
                    let Value::Array(values) = self.document.resolve(value)? else {
                        return Err(invalid("invalid form BBox"));
                    };
                    let [l, b, r, t] = numbers::<4>(&values)?;
                    let corners = [
                        self.mapped(l, b)?,
                        self.mapped(r, b)?,
                        self.mapped(r, t)?,
                        self.mapped(l, t)?,
                    ];
                    let segments = [
                        Segment::Move(corners[0]),
                        Segment::Line(corners[1]),
                        Segment::Line(corners[2]),
                        Segment::Line(corners[3]),
                        Segment::Line(corners[0]),
                    ];
                    let coverage = self.coverage(&segments)?;
                    for (clip, alpha) in self.state.clip.iter_mut().zip(coverage) {
                        *clip = ((u32::from(*clip) * u32::from(alpha) + 127) / 255) as u8;
                    }
                }
                self.form_depth += 1;
                self.program(&bytes)?;
                self.form_depth -= 1;
                self.flush_run();
                self.state = state;
                self.resources = resources;
                self.fonts = fonts;
                self.path = path;
                self.start = start;
                self.current = current;
                self.clipping = clipping;
            }
            Some(Value::Name(kind)) if kind == "Image" => {
                if dictionary.contains_key("SMask")
                    || dictionary.contains_key("Mask")
                    || dictionary.contains_key("ImageMask")
                {
                    return Err(DocumentError::UnsupportedPdf("image masks"));
                }
                let integer = |key: &str| -> Result<usize> {
                    let value = self.document.resolve(
                        dictionary
                            .get(key)
                            .ok_or_else(|| invalid("image lacks dimensions"))?,
                    )?;
                    let value = number(&value)?;
                    if value <= 0.0 || value.fract() != 0.0 || value > 1_000_000.0 {
                        return Err(invalid("invalid image dimensions"));
                    }
                    Ok(value as usize)
                };
                let (width, height) = (integer("Width")?, integer("Height")?);
                let limit = if self.options.pixel_limit == 0 {
                    100_000_000
                } else {
                    self.options.pixel_limit
                };
                if width.checked_mul(height).is_none_or(|n| n > limit) {
                    return Err(invalid("image exceeds pixel limit"));
                }
                let jpeg=match dictionary.get("Filter").map(|v|self.document.resolve(v)).transpose()?{Some(Value::Name(name))=>name=="DCTDecode" || name=="DCT",Some(Value::Array(filters))=>filters.iter().any(|filter|matches!(filter,Value::Name(name) if name=="DCTDecode" || name=="DCT")),_=>false};
                let image = if jpeg {
                    crate::media::decode_grayscale_image_bytes(&bytes)?
                } else {
                    if integer("BitsPerComponent")? != 8 {
                        return Err(DocumentError::UnsupportedPdf("image sample bit depth"));
                    }
                    let colors = self.document.resolve(
                        dictionary
                            .get("ColorSpace")
                            .ok_or_else(|| invalid("image lacks ColorSpace"))?,
                    )?;
                    let format = match colors {
                        Value::Name(name) if name == "DeviceGray" => PixelFormat::Luma8,
                        Value::Name(name) if name == "DeviceRGB" => PixelFormat::Rgb8,
                        _ => return Err(DocumentError::UnsupportedPdf("image color space")),
                    };
                    let image = ImageFrame::new(width, height, format, bytes)?;
                    crate::transforms::grayscale_opencv(&image)?
                };
                if image.width() != width || image.height() != height {
                    return Err(invalid("decoded image dimensions differ from dictionary"));
                }
                let source_size = crate::transforms::ImageSize::new(height, width)?;
                let target_size = crate::transforms::ImageSize::new(self.height, self.width)?;
                let matrix = multiply(
                    [1.0, 0.0, 0.0, 1.0, -0.5, -0.5],
                    multiply(
                        self.transform(),
                        [
                            1.0 / width as f64,
                            0.0,
                            0.0,
                            -1.0 / height as f64,
                            0.5 / width as f64,
                            1.0 - 0.5 / height as f64,
                        ],
                    ),
                );
                let forward = [
                    [matrix[0], matrix[2], matrix[4]],
                    [matrix[1], matrix[3], matrix[5]],
                ];
                let interpolate = matches!(dictionary.get("Interpolate"),Some(Value::Keyword(name)) if name=="true");
                let mut sampled = if interpolate {
                    crate::transforms::warp_affine_u8(
                        image.data(),
                        source_size,
                        target_size,
                        forward,
                        &[255],
                    )?
                } else {
                    let inverse = crate::transforms::invert_affine(forward)?;
                    let mut coordinates = Vec::with_capacity(self.pixels.len());
                    for y in 0..self.height {
                        for x in 0..self.width {
                            coordinates.push([
                                (inverse[0][0] * x as f64
                                    + inverse[0][1] * y as f64
                                    + inverse[0][2]) as f32,
                                (inverse[1][0] * x as f64
                                    + inverse[1][1] * y as f64
                                    + inverse[1][2]) as f32,
                            ]);
                        }
                    }
                    crate::transforms::remap_nearest_u8(
                        image.data(),
                        source_size,
                        target_size,
                        &coordinates,
                        1,
                    )?
                };
                if let Some(value) = dictionary.get("Decode") {
                    let Value::Array(values) = self.document.resolve(value)? else {
                        return Err(invalid("invalid image Decode"));
                    };
                    let decode = values.iter().map(number).collect::<Result<Vec<_>>>()?;
                    if decode == [1.0, 0.0] {
                        for value in &mut sampled {
                            *value = 255 - *value;
                        }
                    } else if decode != [0.0, 1.0] && decode != [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] {
                        return Err(DocumentError::UnsupportedPdf("image decode range"));
                    }
                }
                let corners = [
                    self.mapped(0.0, 0.0)?,
                    self.mapped(1.0, 0.0)?,
                    self.mapped(1.0, 1.0)?,
                    self.mapped(0.0, 1.0)?,
                ];
                let coverage = self.coverage(&[
                    Segment::Move(corners[0]),
                    Segment::Line(corners[1]),
                    Segment::Line(corners[2]),
                    Segment::Line(corners[3]),
                    Segment::Line(corners[0]),
                ])?;
                for (((pixel, &value), &cover), &clip) in self
                    .pixels
                    .iter_mut()
                    .zip(&sampled)
                    .zip(&coverage)
                    .zip(&self.state.clip)
                {
                    let alpha = (u32::from(cover) * u32::from(clip) + 127) / 255;
                    *pixel = ((u32::from(*pixel) * (255 - alpha) + u32::from(value) * alpha + 127)
                        / 255) as u8;
                }
            }
            _ => return Err(DocumentError::UnsupportedPdf("XObject subtype")),
        }
        Ok(())
    }
    fn program(&mut self, bytes: &[u8]) -> Result<()> {
        let mut parser = Parser { bytes, at: 0 };
        let mut operands = Vec::new();
        let stack_depth = self.stack.len();
        while let Some(value) = parser.next_value()? {
            let Value::Keyword(op) = value else {
                if operands.len() >= 100_000 {
                    return Err(invalid("content operands exceed limit"));
                }
                operands.push(value);
                continue;
            };
            if self.extract_text
                && self.form_depth <= 4
                && matches!(op.as_str(), "Tj" | "TJ" | "'" | "\"")
            {
                self.statistics.text_objects += 1;
                self.statistics.invisible_text_objects += usize::from(self.state.mode == 3);
            }
            match op.as_str() {
                "q" => {
                    if self.stack.len() >= 64 {
                        return Err(invalid("graphics state depth exceeds limit"));
                    }
                    self.stack.push(self.state.clone());
                }
                "Q" => self.state = self.stack.pop().ok_or_else(|| invalid("unbalanced Q"))?,
                "cm" => self.state.matrix = multiply(self.state.matrix, numbers::<6>(&operands)?),
                "Do" => {
                    let [Value::Name(name)] = operands.as_slice() else {
                        return Err(invalid("invalid Do"));
                    };
                    self.xobject(name)?;
                }
                "g" | "G" => {
                    let [gray] = numbers::<1>(&operands)?;
                    let gray = (gray.clamp(0.0, 1.0) * 255.0).round() as u8;
                    if op == "g" {
                        self.state.fill = gray;
                    } else {
                        self.state.stroke = gray;
                    }
                }
                "rg" | "RG" => {
                    let [r, g, b] = numbers::<3>(&operands)?;
                    let gray =
                        ((r * 0.299 + g * 0.587 + b * 0.114).clamp(0.0, 1.0) * 255.0).round() as u8;
                    if op == "rg" {
                        self.state.fill = gray;
                    } else {
                        self.state.stroke = gray;
                    }
                }
                "w" => {
                    self.state.width = numbers::<1>(&operands)?[0];
                    if self.state.width < 0.0 {
                        return Err(invalid("negative stroke width"));
                    }
                }
                "m" => {
                    let [x, y] = numbers::<2>(&operands)?;
                    let p = self.mapped(x, y)?;
                    self.path.push(Segment::Move(p));
                    self.start = Some(p);
                    self.current = Some(p);
                }
                "l" => {
                    let [x, y] = numbers::<2>(&operands)?;
                    let p = self.mapped(x, y)?;
                    if self.current.is_none() {
                        return Err(invalid("line without current point"));
                    }
                    self.path.push(Segment::Line(p));
                    self.current = Some(p);
                }
                "c" => {
                    let [a, b, c, d, x, y] = numbers::<6>(&operands)?;
                    let p = self.mapped(x, y)?;
                    self.path
                        .push(Segment::Cubic(self.mapped(a, b)?, self.mapped(c, d)?, p));
                    self.current = Some(p);
                }
                "h" => self.close(),
                "re" => {
                    let [x, y, w, h] = numbers::<4>(&operands)?;
                    let points = [
                        self.mapped(x, y)?,
                        self.mapped(x + w, y)?,
                        self.mapped(x + w, y + h)?,
                        self.mapped(x, y + h)?,
                    ];
                    self.path.push(Segment::Move(points[0]));
                    self.path
                        .extend(points[1..].iter().copied().map(Segment::Line));
                    self.start = Some(points[0]);
                    self.close();
                }
                "W" => self.clipping = true,
                "n" => self.finish_path(false, false)?,
                "f" | "F" => self.finish_path(true, false)?,
                "S" => self.finish_path(false, true)?,
                "s" => {
                    self.close();
                    self.finish_path(false, true)?;
                }
                "B" => self.finish_path(true, true)?,
                "b" => {
                    self.close();
                    self.finish_path(true, true)?;
                }
                "BT" => {
                    self.flush_run();
                    self.in_text = true;
                    self.state.text = IDENTITY;
                    self.state.line = IDENTITY;
                }
                "ET" => {
                    self.flush_run();
                    self.in_text = false;
                }
                "Tf" => {
                    if operands.len() != 2 {
                        return Err(invalid("invalid Tf"));
                    }
                    let Value::Name(name) = &operands[0] else {
                        return Err(invalid("invalid font resource"));
                    };
                    self.state.font = name.clone();
                    self.state.size = number(&operands[1])?;
                }
                "Tm" => {
                    self.flush_run();
                    self.state.text = numbers::<6>(&operands)?;
                    self.state.line = self.state.text;
                }
                "Td" | "TD" => {
                    let [x, y] = numbers::<2>(&operands)?;
                    self.flush_run();
                    if op == "TD" {
                        self.state.leading = -y;
                    }
                    self.state.line = multiply(self.state.line, [1.0, 0.0, 0.0, 1.0, x, y]);
                    self.state.text = self.state.line;
                }
                "T*" => self.next_line(),
                "TL" => self.state.leading = numbers::<1>(&operands)?[0],
                "Tc" => self.state.spacing = numbers::<1>(&operands)?[0],
                "Tw" => self.state.word_spacing = numbers::<1>(&operands)?[0],
                "Tz" => self.state.horizontal = numbers::<1>(&operands)?[0] / 100.0,
                "Ts" => self.state.rise = numbers::<1>(&operands)?[0],
                "Tr" => {
                    let mode = numbers::<1>(&operands)?[0];
                    if mode.fract() != 0.0 || !(0.0..=7.0).contains(&mode) {
                        return Err(invalid("invalid text render mode"));
                    }
                    self.state.mode = mode as u8;
                }
                "Tj" => {
                    if let [Value::Bytes(bytes)] = operands.as_slice() {
                        self.show(bytes)?;
                    } else {
                        return Err(invalid("invalid Tj"));
                    }
                }
                "TJ" => {
                    let [Value::Array(values)] = operands.as_slice() else {
                        return Err(invalid("invalid TJ"));
                    };
                    for value in values {
                        match value {
                            Value::Bytes(bytes) => self.show(bytes)?,
                            Value::Number(offset) => {
                                self.state.text = multiply(
                                    self.state.text,
                                    [
                                        1.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        -offset * self.state.size * self.state.horizontal / 1000.0,
                                        0.0,
                                    ],
                                )
                            }
                            _ => return Err(invalid("invalid TJ member")),
                        }
                    }
                }
                "'" => {
                    self.next_line();
                    if let [Value::Bytes(bytes)] = operands.as_slice() {
                        self.show(bytes)?;
                    } else {
                        return Err(invalid("invalid quote text operator"));
                    }
                }
                "\"" => {
                    let [word, spacing, Value::Bytes(bytes)] = operands.as_slice() else {
                        return Err(invalid("invalid double-quote text operator"));
                    };
                    self.state.word_spacing = number(word)?;
                    self.state.spacing = number(spacing)?;
                    self.next_line();
                    self.show(bytes)?;
                }
                _ => return Err(DocumentError::UnsupportedPdf("content-stream operator")),
            }
            operands.clear();
        }
        if !operands.is_empty() || self.stack.len() != stack_depth || self.in_text {
            return Err(invalid("unterminated page graphics/text state"));
        }
        Ok(())
    }
}
