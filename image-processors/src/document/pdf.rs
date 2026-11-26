//! Bounded PDF object parsing without native document libraries.

use super::{inflate, DocumentError};
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, DocumentError>;
fn invalid(message: &'static str) -> DocumentError {
    DocumentError::InvalidPdf(message)
}

#[derive(Clone, Debug)]
pub(super) enum Value {
    Number(f64),
    Name(String),
    Bytes(Vec<u8>),
    Array(Vec<Value>),
    Dictionary(BTreeMap<String, Value>),
    Reference(u32, u16),
    Keyword(String),
}

pub(super) struct Parser<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) at: usize,
}
impl Parser<'_> {
    pub(super) fn next_value(&mut self) -> Result<Option<Value>> {
        self.whitespace();
        if self.at == self.bytes.len() {
            Ok(None)
        } else {
            self.value(0).map(Some)
        }
    }
    fn whitespace(&mut self) {
        loop {
            while self
                .bytes
                .get(self.at)
                .is_some_and(|b| b.is_ascii_whitespace() || *b == 0)
            {
                self.at += 1;
            }
            if self.bytes.get(self.at) != Some(&b'%') {
                break;
            }
            while self
                .bytes
                .get(self.at)
                .is_some_and(|b| *b != b'\r' && *b != b'\n')
            {
                self.at += 1;
            }
        }
    }
    pub(super) fn word(&mut self) -> Result<String> {
        self.whitespace();
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| !b.is_ascii_whitespace() && !b"()<>[]{}/%\0".contains(b))
        {
            self.at += 1;
        }
        if start == self.at {
            return Err(invalid("missing token"));
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .map(str::to_owned)
            .map_err(|_| invalid("non-ASCII PDF keyword"))
    }
    fn integer(&mut self) -> Result<usize> {
        self.word()?
            .parse()
            .map_err(|_| invalid("expected unsigned integer"))
    }
    pub(super) fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 64 {
            return Err(invalid("object nesting exceeds limit"));
        }
        self.whitespace();
        match self
            .bytes
            .get(self.at)
            .copied()
            .ok_or_else(|| invalid("truncated object"))?
        {
            b'/' => {
                self.at += 1;
                let mut bytes = Vec::new();
                while let Some(&byte) = self.bytes.get(self.at) {
                    if byte.is_ascii_whitespace() || b"()<>[]{}/%\0".contains(&byte) {
                        break;
                    }
                    self.at += 1;
                    if byte == b'#' {
                        let hex = self
                            .bytes
                            .get(self.at..self.at + 2)
                            .ok_or_else(|| invalid("truncated name escape"))?;
                        let hex =
                            std::str::from_utf8(hex).map_err(|_| invalid("invalid name escape"))?;
                        bytes.push(
                            u8::from_str_radix(hex, 16)
                                .map_err(|_| invalid("invalid name escape"))?,
                        );
                        self.at += 2;
                    } else {
                        bytes.push(byte);
                    }
                }
                Ok(Value::Name(
                    String::from_utf8(bytes).map_err(|_| invalid("non-UTF8 resource name"))?,
                ))
            }
            b'[' => {
                self.at += 1;
                let mut values = Vec::new();
                loop {
                    self.whitespace();
                    if self.bytes.get(self.at) == Some(&b']') {
                        self.at += 1;
                        break;
                    }
                    if values.len() >= 1_000_000 {
                        return Err(invalid("array exceeds limit"));
                    }
                    values.push(self.value(depth + 1)?);
                }
                Ok(Value::Array(values))
            }
            b'<' if self.bytes.get(self.at + 1) == Some(&b'<') => {
                self.at += 2;
                let mut values = BTreeMap::new();
                loop {
                    self.whitespace();
                    if self.bytes.get(self.at..self.at + 2) == Some(b">>") {
                        self.at += 2;
                        break;
                    }
                    let Value::Name(name) = self.value(depth + 1)? else {
                        return Err(invalid("dictionary key is not a name"));
                    };
                    if values.len() >= 100_000 {
                        return Err(invalid("dictionary exceeds limit"));
                    }
                    values.insert(name, self.value(depth + 1)?);
                }
                Ok(Value::Dictionary(values))
            }
            b'<' => {
                self.at += 1;
                let mut hex = Vec::new();
                loop {
                    let byte = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| invalid("unterminated hex string"))?;
                    self.at += 1;
                    if byte == b'>' {
                        break;
                    }
                    if !byte.is_ascii_whitespace() {
                        hex.push(byte);
                    }
                }
                if hex.len() % 2 != 0 {
                    hex.push(b'0');
                }
                let mut bytes = Vec::new();
                for chunk in hex.as_chunks::<2>().0 {
                    let text =
                        std::str::from_utf8(chunk).map_err(|_| invalid("invalid hex string"))?;
                    bytes.push(
                        u8::from_str_radix(text, 16).map_err(|_| invalid("invalid hex string"))?,
                    );
                }
                Ok(Value::Bytes(bytes))
            }
            b'(' => {
                self.at += 1;
                let mut nesting = 1usize;
                let mut bytes = Vec::new();
                while nesting != 0 {
                    let mut byte = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| invalid("unterminated literal string"))?;
                    self.at += 1;
                    match byte {
                        b'(' => {
                            nesting += 1;
                            bytes.push(byte);
                        }
                        b')' => {
                            nesting -= 1;
                            if nesting != 0 {
                                bytes.push(byte);
                            }
                        }
                        b'\\' => {
                            byte = *self
                                .bytes
                                .get(self.at)
                                .ok_or_else(|| invalid("truncated string escape"))?;
                            self.at += 1;
                            match byte {
                                b'\n' => {}
                                b'\r' => {
                                    if self.bytes.get(self.at) == Some(&b'\n') {
                                        self.at += 1;
                                    }
                                }
                                b'n' => bytes.push(b'\n'),
                                b'r' => bytes.push(b'\r'),
                                b't' => bytes.push(b'\t'),
                                b'b' => bytes.push(8),
                                b'f' => bytes.push(12),
                                b'0'..=b'7' => {
                                    let mut octal = u16::from(byte - b'0');
                                    for _ in 0..2 {
                                        if let Some(&next @ b'0'..=b'7') = self.bytes.get(self.at) {
                                            octal = octal * 8 + u16::from(next - b'0');
                                            self.at += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    bytes.push(octal as u8);
                                }
                                _ => bytes.push(byte),
                            }
                        }
                        b'\r' => {
                            if self.bytes.get(self.at) == Some(&b'\n') {
                                self.at += 1;
                            }
                            bytes.push(b'\n');
                        }
                        _ => bytes.push(byte),
                    }
                }
                Ok(Value::Bytes(bytes))
            }
            _ => {
                let word = self.word()?;
                let Ok(number) = word.parse::<f64>() else {
                    return Ok(Value::Keyword(word));
                };
                if !number.is_finite() {
                    return Err(invalid("nonfinite PDF number"));
                }
                let saved = self.at;
                if let (Ok(object), Ok(generation)) = (
                    word.parse::<u32>(),
                    self.word()
                        .and_then(|s| s.parse::<u16>().map_err(|_| invalid("invalid reference"))),
                ) {
                    if matches!(self.word().as_deref(), Ok("R")) {
                        return Ok(Value::Reference(object, generation));
                    }
                }
                self.at = saved;
                Ok(Value::Number(number))
            }
        }
    }
}

/// A bounded, immutable PDF object source using classic cross-reference tables.
///
/// Encryption, compressed cross-reference/object streams and incremental hybrid
/// cross references are reported explicitly rather than partially decoded.
pub struct PdfDocument {
    bytes: Vec<u8>,
    offsets: BTreeMap<(u32, u16), usize>,
    root: (u32, u16),
}

impl PdfDocument {
    /// Open PDF bytes without executing native libraries or external tools.
    ///
    /// # Errors
    /// Reports malformed object references, encrypted PDFs, unsupported cross
    /// references and resource limits. Stream decoding remains bounded separately.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if !super::is_pdf(&bytes) {
            return Err(invalid("missing PDF header"));
        }
        let start = bytes
            .windows(9)
            .rposition(|value| value == b"startxref")
            .ok_or_else(|| invalid("missing startxref"))?;
        let mut cursor = Parser {
            bytes: &bytes,
            at: start + 9,
        };
        let mut next = cursor.integer()?;
        let mut visited = BTreeSet::new();
        let mut offsets = BTreeMap::new();
        let mut latest_objects = BTreeSet::new();
        let mut root = None;
        loop {
            if !visited.insert(next) || visited.len() > 1024 {
                return Err(invalid("cyclic or excessive cross references"));
            }
            let mut cursor = Parser {
                bytes: &bytes,
                at: next,
            };
            if cursor.word()?.as_str() != "xref" {
                return Err(DocumentError::UnsupportedPdf("cross-reference streams"));
            }
            loop {
                let saved = cursor.at;
                if cursor.word()?.as_str() == "trailer" {
                    break;
                }
                cursor.at = saved;
                let first = cursor.integer()?;
                let count = cursor.integer()?;
                if count > 1_000_000 || first.checked_add(count).is_none() {
                    return Err(invalid("cross-reference subsection exceeds limit"));
                }
                for number in first..first + count {
                    let offset = cursor.integer()?;
                    let generation = u16::try_from(cursor.integer()?)
                        .map_err(|_| invalid("invalid generation"))?;
                    let kind = cursor.word()?;
                    let latest = latest_objects.insert(number);
                    if kind == "n" {
                        if offset >= bytes.len() {
                            return Err(invalid("object offset outside PDF"));
                        }
                        if latest {
                            offsets.insert(
                                (
                                    u32::try_from(number)
                                        .map_err(|_| invalid("object number overflow"))?,
                                    generation,
                                ),
                                offset,
                            );
                        }
                    } else if kind != "f" {
                        return Err(invalid("invalid cross-reference entry"));
                    }
                }
            }
            let Value::Dictionary(trailer) = cursor.value(0)? else {
                return Err(invalid("missing trailer dictionary"));
            };
            if trailer.contains_key("Encrypt") {
                return Err(DocumentError::UnsupportedPdf("encrypted documents"));
            }
            if trailer.contains_key("XRefStm") {
                return Err(DocumentError::UnsupportedPdf("hybrid cross references"));
            }
            if root.is_none() {
                if let Some(Value::Reference(object, generation)) = trailer.get("Root") {
                    root = Some((*object, *generation));
                }
            }
            match trailer.get("Prev") {
                Some(Value::Number(offset)) if *offset >= 0.0 && offset.fract() == 0.0 => {
                    next = *offset as usize
                }
                None => break,
                _ => return Err(invalid("invalid previous cross-reference offset")),
            }
        }
        Ok(Self {
            bytes,
            offsets,
            root: root.ok_or_else(|| invalid("missing catalog reference"))?,
        })
    }

    pub(super) fn object(&self, object: u32, generation: u16) -> Result<(Value, usize)> {
        let at = *self
            .offsets
            .get(&(object, generation))
            .ok_or_else(|| invalid("unknown indirect object"))?;
        let mut cursor = Parser {
            bytes: &self.bytes,
            at,
        };
        if cursor.integer()? != object as usize
            || cursor.integer()? != usize::from(generation)
            || cursor.word()?.as_str() != "obj"
        {
            return Err(invalid("cross-reference object identity mismatch"));
        }
        let value = cursor.value(0)?;
        Ok((value, cursor.at))
    }

    pub(super) fn resolve(&self, value: &Value) -> Result<Value> {
        let mut value = value.clone();
        for _ in 0..64 {
            let Value::Reference(object, generation) = value else {
                return Ok(value);
            };
            value = self.object(object, generation)?.0;
        }
        Err(invalid("cyclic indirect reference"))
    }

    pub(super) fn dictionary(&self, value: &Value) -> Result<BTreeMap<String, Value>> {
        let Value::Dictionary(value) = self.resolve(value)? else {
            return Err(invalid("expected dictionary"));
        };
        Ok(value)
    }

    /// Count leaf pages in the catalog's page tree.
    ///
    /// # Errors
    /// Reports malformed, cyclic or excessively deep page trees.
    pub fn page_count(&self) -> Result<usize> {
        Ok(self.pages()?.len())
    }

    pub(super) fn pages(&self) -> Result<Vec<BTreeMap<String, Value>>> {
        let catalog = self.dictionary(&Value::Reference(self.root.0, self.root.1))?;
        let root = catalog
            .get("Pages")
            .ok_or_else(|| invalid("catalog lacks Pages"))?;
        let mut pages = Vec::new();
        self.walk_pages(root, &BTreeMap::new(), 0, &mut pages)?;
        Ok(pages)
    }
    fn walk_pages(
        &self,
        value: &Value,
        inherited: &BTreeMap<String, Value>,
        depth: usize,
        pages: &mut Vec<BTreeMap<String, Value>>,
    ) -> Result<()> {
        if depth > 64 || pages.len() > 100_000 {
            return Err(invalid("page tree exceeds limit"));
        }
        let mut dictionary = self.dictionary(value)?;
        for key in ["Resources", "MediaBox", "CropBox", "Rotate"] {
            if !dictionary.contains_key(key) {
                if let Some(value) = inherited.get(key) {
                    dictionary.insert(key.to_owned(), value.clone());
                }
            }
        }
        match dictionary.get("Type") {
            Some(Value::Name(name)) if name == "Page" => pages.push(dictionary),
            Some(Value::Name(name)) if name == "Pages" => {
                let kids = dictionary
                    .get("Kids")
                    .ok_or_else(|| invalid("page tree lacks Kids"))?;
                let Value::Array(kids) = self.resolve(kids)? else {
                    return Err(invalid("invalid page Kids"));
                };
                for child in kids {
                    self.walk_pages(&child, &dictionary, depth + 1, pages)?;
                }
            }
            _ => return Err(invalid("invalid page tree type")),
        }
        Ok(())
    }

    pub(super) fn stream(
        &self,
        value: &Value,
        limit: usize,
    ) -> Result<(BTreeMap<String, Value>, Vec<u8>)> {
        let Value::Reference(object, generation) = value else {
            return Err(invalid("stream must be an indirect object"));
        };
        let (dictionary, at) = self.object(*object, *generation)?;
        let dictionary = self.dictionary(&dictionary)?;
        let length = dictionary
            .get("Length")
            .ok_or_else(|| invalid("stream lacks Length"))?;
        let Value::Number(length) = self.resolve(length)? else {
            return Err(invalid("invalid stream Length"));
        };
        if length < 0.0 || length.fract() != 0.0 || length > self.bytes.len() as f64 {
            return Err(invalid("invalid stream length"));
        }
        let mut cursor = Parser {
            bytes: &self.bytes,
            at,
        };
        if cursor.word()?.as_str() != "stream" {
            return Err(invalid("object lacks stream"));
        }
        if self.bytes.get(cursor.at) == Some(&b'\r') {
            cursor.at += 1;
        }
        if self.bytes.get(cursor.at) == Some(&b'\n') {
            cursor.at += 1;
        } else if self.bytes.get(cursor.at.wrapping_sub(1)) != Some(&b'\r') {
            return Err(invalid("stream lacks line ending"));
        }
        let end = cursor
            .at
            .checked_add(length as usize)
            .ok_or_else(|| invalid("stream length overflow"))?;
        let mut bytes = self
            .bytes
            .get(cursor.at..end)
            .ok_or_else(|| invalid("truncated stream"))?
            .to_vec();
        let filters = match dictionary
            .get("Filter")
            .map(|value| self.resolve(value))
            .transpose()?
        {
            None => Vec::new(),
            Some(Value::Name(name)) => vec![Value::Name(name)],
            Some(Value::Array(filters)) => filters,
            _ => return Err(invalid("invalid stream filters")),
        };
        for filter in filters {
            match self.resolve(&filter)? {
                Value::Name(name) if name == "FlateDecode" || name == "Fl" => {
                    bytes = inflate::decode(&bytes, limit)?
                }
                Value::Name(name) if name == "DCTDecode" || name == "DCT" => {}
                _ => return Err(DocumentError::UnsupportedPdf("stream filter")),
            }
        }
        if dictionary.contains_key("DecodeParms") {
            return Err(DocumentError::UnsupportedPdf("stream decode parameters"));
        }
        if bytes.len() > limit {
            return Err(invalid("decoded stream exceeds limit"));
        }
        Ok((dictionary, bytes))
    }
}
