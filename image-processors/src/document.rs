//! Positioned document text and OCR/text-layer selection independent of model graphs.

use crate::media::{ImageFrame, PixelFormat};
use serde::{Deserialize, Serialize};

mod geometry;
mod inflate;
mod pdf;
mod render;
mod unicode;
pub use geometry::{estimate_document_skew, rotate_document_page, unrotate_document_box};
pub use pdf::PdfDocument;
pub use render::{PdfRenderOptions, RenderedPdfPage};
pub use unicode::{document_right_to_left, is_broken_document_character, is_document_letter};

/// Recognize a PDF header within the first 1024 input bytes.
/// The original bytes and absolute object offsets remain unchanged.
pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(1024)]
        .windows(5)
        .any(|header| header == b"%PDF-")
}

/// Failure to interpret document geometry or positioned text.
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    /// The page or render does not meet the document coordinate contract.
    #[error("invalid document page: {0}")]
    InvalidPage(&'static str),
    /// Input uses a document representation not implemented by this decoder.
    #[error("unsupported PDF feature: {0}")]
    UnsupportedPdf(&'static str),
    /// Input bytes violate the PDF syntax or object graph.
    #[error("invalid PDF: {0}")]
    InvalidPdf(&'static str),
    /// A shared raster decoder failed.
    #[error(transparent)]
    Media(#[from] crate::media::MediaError),
    /// Input could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Shared image geometry failed.
    #[error(transparent)]
    Transform(#[from] crate::transforms::TransformError),
    /// Shared affine resampling failed.
    #[error(transparent)]
    Affine(#[from] crate::transforms::FaceGeometryError),
}

/// Character geometry in the rendered PDF frame, with a top-left origin.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PositionedCharacter {
    /// Unicode scalar supplied by the PDF font mapping.
    pub character: char,
    /// Left, top, right, bottom, in PDF points after crop and page rotation.
    pub bounds: [f64; 4],
    /// Content-stream index, retained when text is assigned to detected lines.
    pub content_index: usize,
    /// Text rendering mode from the source graphics state.
    pub render_mode: u8,
    /// True when the glyph is not upright in the rendered page frame.
    pub rotated: bool,
    /// True when the source font map produced a missing/unassigned code point.
    pub broken_font_map: bool,
}

/// One source text run, retaining character order and boxes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PositionedTextRun {
    /// Text in source content order.
    pub text: String,
    /// Union box in rendered-frame PDF points.
    pub bounds: [f64; 4],
    /// Per-character text and geometry.
    pub characters: Vec<PositionedCharacter>,
}

/// Reason a document's own text cannot be trusted automatically.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum TextLayerRejection {
    /// No positioned text exists.
    Missing,
    /// More than five percent of characters use invisible rendering mode.
    Invisible,
    /// More than half a percent of mapped characters are invalid.
    BrokenFontMap,
    /// More than three percent of nonempty text runs lack visible ink.
    NoInk,
}

/// Original text-layer trust measurements, without changing PDF content order.
#[derive(Clone, Debug, Serialize)]
pub struct TextLayerAssessment {
    /// Number of source characters.
    pub characters: usize,
    /// Number of characters with broken font mappings.
    pub bad_characters: usize,
    /// Number of non-upright source characters.
    pub rotated_characters: usize,
    /// Fraction of nonempty text runs without dark pixels.
    pub no_ink_fraction: f64,
    /// Absent when the text layer passes all original checks.
    pub rejection: Option<TextLayerRejection>,
}

/// Counts collected while decoding source text objects, before glyph filtering.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct TextLayerStatistics {
    /// Non-whitespace source characters, including rotated/off-page characters.
    pub characters: usize,
    /// Non-whitespace source characters with invalid font mappings.
    pub bad_characters: usize,
    /// Non-whitespace source characters whose baseline is not upright.
    pub rotated_characters: usize,
    /// Source text objects, including nested form objects to depth four.
    pub text_objects: usize,
    /// Source text objects whose rendering mode is three.
    pub invisible_text_objects: usize,
}

/// Assess positioned text against a grayscale render at pixels_per_point.
///
/// The caller supplies mapped glyphs in the same cropped/rotated coordinate
/// frame as the render. Broken mapping information belongs to the decoder,
/// rather than a second incomplete Unicode-category table in an OCR model.
///
/// # Errors
/// Reports invalid render format, scale or nonfinite character geometry.
pub fn assess_text_layer(
    runs: &[PositionedTextRun],
    statistics: TextLayerStatistics,
    render: &ImageFrame,
    pixels_per_point: f64,
) -> Result<TextLayerAssessment, DocumentError> {
    if render.pixel_format() != PixelFormat::Luma8
        || !pixels_per_point.is_finite()
        || pixels_per_point <= 0.0
    {
        return Err(DocumentError::InvalidPage(
            "text-layer assessment requires Luma8 and a positive finite scale",
        ));
    }
    let mut result = TextLayerAssessment {
        characters: statistics.characters,
        bad_characters: statistics.bad_characters,
        rotated_characters: statistics.rotated_characters,
        no_ink_fraction: 0.0,
        rejection: None,
    };
    for character in runs.iter().flat_map(|run| &run.characters) {
        if character.bounds.iter().any(|value| !value.is_finite()) {
            return Err(DocumentError::InvalidPage("nonfinite character bounds"));
        }
    }
    if statistics.invisible_text_objects > statistics.text_objects {
        return Err(DocumentError::InvalidPage(
            "invisible text object count exceeds total",
        ));
    }
    if statistics.invisible_text_objects as f64 > 0.05 * statistics.text_objects.max(1) as f64 {
        result.rejection = Some(TextLayerRejection::Invisible);
    } else if runs.is_empty() {
        result.rejection = Some(TextLayerRejection::Missing);
    } else if result.bad_characters as f64 > 0.005 * result.characters.max(1) as f64 {
        result.rejection = Some(TextLayerRejection::BrokenFontMap);
    } else {
        let mut checked = 0;
        let mut empty = 0;
        for run in runs {
            if run.bounds.iter().any(|value| !value.is_finite()) {
                return Err(DocumentError::InvalidPage("nonfinite text run bounds"));
            }
            checked += 1;
            let [left, top, right, bottom] = run.bounds.map(|value| value * pixels_per_point);
            let left = left.trunc().clamp(0.0, render.width() as f64) as usize;
            let right = (right.trunc() + 1.0).clamp(0.0, render.width() as f64) as usize;
            let top = top.trunc().clamp(0.0, render.height() as f64) as usize;
            let bottom = (bottom.trunc() + 1.0).clamp(0.0, render.height() as f64) as usize;
            let ink = (top..bottom)
                .any(|y| (left..right).any(|x| render.data()[y * render.width() + x] < 160));
            empty += usize::from(!ink);
        }
        result.no_ink_fraction = empty as f64 / checked.max(1) as f64;
        if result.no_ink_fraction > 0.03 {
            result.rejection = Some(TextLayerRejection::NoInk);
        }
    }
    Ok(result)
}

/// Check content order against geometry and optional pairwise model logits.
///
/// # Errors
/// Reports invalid geometry or a pairwise matrix of the wrong size.
pub fn pdf_content_order_coherent(
    boxes: &[[f64; 4]],
    pair_logits: Option<&[f32]>,
) -> Result<bool, DocumentError> {
    if boxes.iter().flatten().any(|value| !value.is_finite()) {
        return Err(DocumentError::InvalidPage("nonfinite line bounds"));
    }
    if let Some(logits) = pair_logits {
        if boxes.len().checked_mul(boxes.len()) != Some(logits.len()) {
            return Err(DocumentError::InvalidPage(
                "reading-order matrix size differs from line count",
            ));
        }
    }
    if boxes.len() < 3 {
        return Ok(true);
    }
    let mut heights = boxes
        .iter()
        .map(|bounds| bounds[3] - bounds[1])
        .collect::<Vec<_>>();
    heights.sort_by(f64::total_cmp);
    let middle = heights.len() / 2;
    let median = if heights.len() % 2 == 0 {
        (heights[middle - 1] + heights[middle]) * 0.5
    } else {
        heights[middle]
    };
    let backtracks = boxes
        .windows(2)
        .filter(|pair| pair[1][1] < pair[0][1] - 0.5 * median && pair[1][0] < pair[0][2] - median)
        .count();
    if backtracks > 1.max((0.02 * boxes.len() as f64) as usize) {
        return Ok(false);
    }
    Ok(pair_logits.is_none_or(|logits| {
        let agreed = (0..boxes.len() - 1)
            .filter(|i| logits[i * boxes.len() + i + 1] > 0.0)
            .count();
        agreed as f64 / (boxes.len() - 1) as f64 >= 0.85
    }))
}

/// Group line indices by stable width order and explicit batch/area limits.
///
/// # Errors
/// Reports zero capacities or a nonfinite/less-than-one width ratio.
pub fn group_document_lines(
    widths: &[usize],
    max_batch: usize,
    max_ratio: f64,
    min_width: usize,
    max_area: usize,
) -> Result<Vec<Vec<usize>>, DocumentError> {
    if max_batch == 0
        || min_width == 0
        || max_area == 0
        || !max_ratio.is_finite()
        || max_ratio < 1.0
    {
        return Err(DocumentError::InvalidPage("invalid line batch limits"));
    }
    let mut order = (0..widths.len()).collect::<Vec<_>>();
    order.sort_by_key(|&i| widths[i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for index in order {
        if let Some(batch) = batches.last_mut() {
            let width = widths[index].max(min_width);
            if batch.len() < max_batch
                && width as f64 <= max_ratio * widths[batch[0]].max(min_width) as f64
                && (batch.len() + 1)
                    .checked_mul(width)
                    .is_some_and(|area| area <= max_area)
            {
                batch.push(index);
                continue;
            }
        }
        batches.push(vec![index]);
    }
    Ok(batches)
}
