use std::fmt::Write as _;
use std::{fs, io::Write};

use base64::{engine::general_purpose::STANDARD, Engine};
use image_processors::document::{
    assess_text_layer, is_pdf, PdfDocument, PdfRenderOptions, RenderedPdfPage, TextLayerRejection,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("fixtures/native_document/reference.json"))
        .expect("original document reference JSON")
}

fn bytes(value: &Value) -> Vec<u8> {
    STANDARD
        .decode(value.as_str().expect("fixture base64 string"))
        .expect("valid original fixture bytes")
}

fn numbers(value: &Value) -> String {
    value
        .as_array()
        .expect("numeric PDF array")
        .iter()
        .map(|v| v.as_f64().expect("finite fixture number").to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn check_box(actual: &[f64], expected: &Value, name: &str) {
    let expected = expected.as_array().expect("reference box");
    assert_eq!(actual.len(), expected.len(), "{name}: box rank");
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            (a - b.as_f64().expect("reference coordinate")).abs() <= 0.001,
            "{name}: coordinate {a} differs from {b}"
        );
    }
}

fn check_layer(page: &RenderedPdfPage, expected: &Value, scale: f64, name: &str) {
    let assessment = assess_text_layer(&page.runs, page.statistics, &page.image, scale)
        .expect("valid actual rendered text layer");
    assert_eq!(
        assessment.rejection.is_none(),
        expected["trusted"]
            .as_bool()
            .expect("reference trust decision"),
        "{name}: trust decision"
    );
    let invisible = assessment.rejection == Some(TextLayerRejection::Invisible);
    assert_eq!(
        invisible,
        expected["reason"]
            .as_str()
            .expect("reference rejection reason")
            .starts_with("invisible text"),
        "{name}: invisible overlay"
    );
    // This is the same layer boundary used by the Indra caller: invisible
    // overlays provide no readable text, while other assessments retain runs.
    let runs = if invisible { &[][..] } else { &page.runs };
    let lines = expected["lines"].as_array().expect("reference text runs");
    assert_eq!(runs.len(), lines.len(), "{name}: text run count");
    for (actual, original) in runs.iter().zip(lines) {
        assert_eq!(actual.text, original["text"], "{name}: text and order");
        check_box(&actual.bounds, &original["bbox"], name);
        let chars = original["chars"].as_array().expect("reference characters");
        assert_eq!(actual.characters.len(), chars.len(), "{name}: characters");
        for (actual, original) in actual.characters.iter().zip(chars) {
            assert_eq!(actual.character.to_string(), original[0], "{name}: scalar");
            check_box(&actual.bounds, &original[1], name);
        }
    }
    for (key, actual) in [
        ("chars", page.statistics.characters),
        ("bad", page.statistics.bad_characters),
        ("rotated", page.statistics.rotated_characters),
    ] {
        // The original returns before counting characters for invisible layers.
        if let Some(original) = expected["stats"][key].as_u64() {
            assert_eq!(actual as u64, original, "{name}: {key}");
        }
    }
}

#[test]
fn vector_pdf_rasters_match_original_indra() {
    let data = reference();
    for case in data["vectors"].as_array().expect("vector cases") {
        let name = case["name"].as_str().expect("case name");
        let document = PdfDocument::from_bytes(bytes(&case["pdfBase64"]))
            .expect("valid independent original PDF fixture");
        for original in case["reference"].as_array().expect("render references") {
            let scale = original["scale"].as_f64().expect("render scale");
            let page = document
                .render_page(0, scale, &PdfRenderOptions::default())
                .expect("native vector PDF render");
            let raster = image::load_from_memory(&bytes(&original["grayPngBase64"]))
                .expect("original PDFium grayscale PNG")
                .to_luma8();
            assert_eq!(page.image.width() as u64, original["width"], "{name}");
            assert_eq!(page.image.height() as u64, original["height"], "{name}");
            assert_eq!(page.image.data(), raster.as_raw(), "{name}: exact raster");
            check_layer(&page, &original["textLayer"], scale, name);
        }
    }
}

fn stream(data: &[u8], extra: &str) -> Vec<u8> {
    let mut output = format!("<< /Length {} {extra} >>\nstream\n", data.len()).into_bytes();
    output.extend_from_slice(data);
    output.extend_from_slice(b"\nendstream");
    output
}

fn embedded_pdf(case: &Value, metrics: &Value, font: &[u8]) -> Vec<u8> {
    let mut page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 160 120] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R /Rotate {}",
        case["rotation"]
    );
    if case["crop"].is_array() {
        write!(&mut page, " /CropBox [{}]", numbers(&case["crop"]))
            .expect("write crop to PDF page");
    }
    page.push_str(" >>");
    let to_unicode = case
        .get("toUnicodeBase64")
        .filter(|v| v.is_string())
        .map(bytes);
    let font_dictionary = format!(
        "<< /Type /Font /Subtype /TrueType /BaseFont /Arial /Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 126 /Widths [{}] /FontDescriptor 5 0 R{} >>",
        numbers(&metrics["widths"]),
        if to_unicode.is_some() { " /ToUnicode 8 0 R" } else { "" }
    );
    let descriptor = format!(
        "<< /Type /FontDescriptor /FontName /Arial /Flags 32 /FontBBox [{}] /ItalicAngle 0 /Ascent {} /Descent {} /CapHeight {} /StemV 80 /FontFile2 6 0 R >>",
        numbers(&metrics["bbox"]), metrics["ascent"], metrics["descent"], metrics["capHeight"]
    );
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        page.into_bytes(),
        font_dictionary.into_bytes(),
        descriptor.into_bytes(),
        stream(font, &format!("/Length1 {}", font.len())),
        stream(
            &bytes(&case["contentBase64"]),
            if case["flate"] == true {
                "/Filter /FlateDecode"
            } else {
                ""
            },
        ),
    ];
    if let Some(mapping) = to_unicode {
        objects.push(stream(&mapping, ""));
    }
    let mut output = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(output.len());
        writeln!(&mut output, "{} 0 obj", i + 1).expect("write test object header");
        output.extend_from_slice(object);
        output.extend_from_slice(b"\nendobj\n");
    }
    let xref = output.len();
    writeln!(&mut output, "xref\n0 {}", objects.len() + 1).expect("write test xref header");
    output.extend_from_slice(b"0000000000 65535 f \n");
    for offset in offsets {
        writeln!(&mut output, "{offset:010} 00000 n ").expect("write test xref entry");
    }
    writeln!(
        &mut output,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF",
        objects.len() + 1
    )
    .expect("write test trailer");
    output
}

#[test]
fn pdf_header_recognition_matches_original_window() {
    let data = reference();
    for case in data["headerCases"]
        .as_array()
        .expect("header recognition cases")
    {
        assert_eq!(
            is_pdf(&bytes(&case["bytesBase64"])),
            case["expected"]
                .as_bool()
                .expect("original header predicate"),
            "{}: exact original 1024-byte window",
            case["name"]
        );
    }
}

#[test]
fn extracted_unicode_does_not_select_the_painted_glyph() {
    let Some(path) = std::env::var_os("DINOML_ARIAL_REFERENCE") else {
        assert_ne!(
            std::env::var("DINOML_REQUIRE_FONT_REFERENCE").as_deref(),
            Ok("1"),
            "the required pinned Arial fixture is unavailable"
        );
        return;
    };
    let font = fs::read(path).expect("read existing pinned reference font asset");
    let data = reference();
    for name in [
        "unicode-glyph",
        "broken-unicode-glyph",
        "replacement-unicode-glyph",
    ] {
        let case = data["embeddedCases"]
            .as_array()
            .expect("embedded cases")
            .iter()
            .find(|c| c["name"] == name)
            .expect("mapping regression fixture");
        let mut unmapped = case.clone();
        unmapped["toUnicodeBase64"] = Value::Null;
        let control = PdfDocument::from_bytes(embedded_pdf(&unmapped, &data["fontMetrics"], &font))
            .expect("control PDF using identical glyph codes");
        let mapped = PdfDocument::from_bytes(embedded_pdf(case, &data["fontMetrics"], &font))
            .expect("PDF with extraction-only Unicode mapping");
        let expected = control
            .render_ocr_page(0, 1.0, &PdfRenderOptions::default())
            .expect("control glyph raster");
        let page = mapped
            .render_page(0, 1.0, &PdfRenderOptions::default())
            .expect("mapped text extraction with original glyph");
        let ocr = mapped
            .render_ocr_page(0, 1.0, &PdfRenderOptions::default())
            .expect("OCR rendering ignores extraction mapping");
        assert_eq!(
            page.image.data(),
            expected.data(),
            "{name}: text-mode glyph identity"
        );
        assert_eq!(
            ocr.data(),
            expected.data(),
            "{name}: OCR-only glyph identity"
        );
    }
}

#[test]
fn embedded_font_text_geometry_matches_original_indra() {
    let Some(path) = std::env::var_os("DINOML_ARIAL_REFERENCE") else {
        assert_ne!(
            std::env::var("DINOML_REQUIRE_FONT_REFERENCE").as_deref(),
            Ok("1"),
            "the required pinned Arial fixture is unavailable"
        );
        return;
    };
    let font = fs::read(path).expect("read existing pinned reference font asset");
    let data = reference();
    for case in data["embeddedCases"]
        .as_array()
        .expect("embedded font cases")
    {
        let name = case["name"].as_str().expect("case name");
        let document = PdfDocument::from_bytes(embedded_pdf(case, &data["fontMetrics"], &font))
            .expect("valid original-content PDF with exact reference font");
        for original in case["reference"].as_array().expect("text references") {
            let scale = original["scale"].as_f64().expect("render scale");
            let page = document
                .render_page(0, scale, &PdfRenderOptions::default())
                .expect("native embedded-font PDF render");
            assert_eq!(page.image.width() as u64, original["width"], "{name}");
            assert_eq!(page.image.height() as u64, original["height"], "{name}");
            // Font antialiasing has an explicit native profile. These checks
            // cover original text/geometry/trust, without claiming pixel parity.
            check_layer(&page, &original["textLayer"], scale, name);
        }
    }
}
