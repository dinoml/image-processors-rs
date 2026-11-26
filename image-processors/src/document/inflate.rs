//! Bounded RFC 1950/1951 stream decoding for PDF FlateDecode streams.

use super::DocumentError;

fn invalid() -> DocumentError {
    DocumentError::InvalidPdf("malformed FlateDecode stream")
}

struct Bits<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl Bits<'_> {
    fn read(&mut self, count: usize) -> Result<usize, DocumentError> {
        let end = self.offset.checked_add(count).ok_or_else(invalid)?;
        if end > self.bytes.len().saturating_mul(8) || count > 16 {
            return Err(invalid());
        }
        let mut value = 0;
        for bit in 0..count {
            value |=
                usize::from((self.bytes[(self.offset + bit) / 8] >> ((self.offset + bit) % 8)) & 1)
                    << bit;
        }
        self.offset = end;
        Ok(value)
    }
}

struct Huffman {
    codes: Vec<(usize, usize, usize)>,
    maximum: usize,
}
#[derive(Clone, Copy)]
enum Alphabet {
    CodeLengths,
    LiteralLength,
    Distance,
}
impl Huffman {
    fn new(lengths: &[usize], alphabet: Alphabet) -> Result<Self, DocumentError> {
        let maximum = lengths.iter().copied().max().unwrap_or(0);
        if maximum > 15 {
            return Err(invalid());
        }
        let mut counts = [0usize; 16];
        for &length in lengths {
            counts[length] += 1;
        }
        counts[0] = 0;
        // An unused distance alphabet can be empty. Literal/length alphabets
        // must contain EOB; the caller checks that symbol separately.
        if maximum == 0 && !matches!(alphabet, Alphabet::Distance) {
            return Err(invalid());
        }
        let mut remaining = 1i32;
        for &count in &counts[1..] {
            remaining = (remaining << 1) - count as i32;
            if remaining < 0 {
                return Err(invalid());
            }
        }
        // Code-length alphabets must be complete. RFC 1951's one-bit,
        // single-symbol literal/distance exception is not a general license
        // for an incomplete tree. Empty distances remain invalid if used.
        if remaining != 0
            && maximum != 0
            && (matches!(alphabet, Alphabet::CodeLengths) || maximum != 1)
        {
            return Err(invalid());
        }
        let mut next = [0usize; 16];
        let mut code = 0;
        for length in 1..16 {
            code = (code + counts[length - 1]) << 1;
            next[length] = code;
        }
        if next[maximum] + counts[maximum] > 1usize << maximum {
            return Err(invalid());
        }
        let mut codes = Vec::new();
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let mut value = next[length];
            next[length] += 1;
            let mut reversed = 0;
            for _ in 0..length {
                reversed = (reversed << 1) | (value & 1);
                value >>= 1;
            }
            codes.push((length, reversed, symbol));
        }
        Ok(Self { codes, maximum })
    }
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<usize, DocumentError> {
        let mut code = 0;
        for length in 1..=self.maximum {
            code |= bits.read(1)? << (length - 1);
            if let Some(&(_, _, symbol)) = self
                .codes
                .iter()
                .find(|&&(n, c, _)| n == length && c == code)
            {
                return Ok(symbol);
            }
        }
        Err(invalid())
    }
}

pub(super) fn decode(bytes: &[u8], limit: usize) -> Result<Vec<u8>, DocumentError> {
    if bytes.len() < 6
        || bytes[0] & 15 != 8
        || bytes[0] >> 4 > 7
        || ((usize::from(bytes[0]) << 8) | usize::from(bytes[1])) % 31 != 0
        || bytes[1] & 32 != 0
    {
        return Err(invalid());
    }
    let mut bits = Bits {
        bytes: &bytes[2..bytes.len() - 4],
        offset: 0,
    };
    let mut output = Vec::new();
    loop {
        let last = bits.read(1)?;
        let kind = bits.read(2)?;
        if kind == 0 {
            bits.offset = (bits.offset + 7) & !7;
            let length = bits.read(16)?;
            if bits.read(16)? != (length ^ 65535)
                || output.len().checked_add(length).is_none_or(|n| n > limit)
            {
                return Err(invalid());
            }
            for _ in 0..length {
                output.push(bits.read(8)? as u8);
            }
        } else if kind == 1 || kind == 2 {
            let (literal, distance) = if kind == 1 {
                let mut lengths = vec![8; 288];
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                (
                    Huffman::new(&lengths, Alphabet::LiteralLength)?,
                    Huffman::new(&[5; 32], Alphabet::Distance)?,
                )
            } else {
                let literals = bits.read(5)? + 257;
                let distances = bits.read(5)? + 1;
                if literals > 286 || distances > 30 {
                    return Err(invalid());
                }
                let count = bits.read(4)? + 4;
                let order = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut lengths = [0; 19];
                for &index in &order[..count] {
                    lengths[index] = bits.read(3)?;
                }
                let codes = Huffman::new(&lengths, Alphabet::CodeLengths)?;
                let mut lengths = Vec::new();
                while lengths.len() < literals + distances {
                    match codes.symbol(&mut bits)? {
                        value @ 0..=15 => lengths.push(value),
                        16 => {
                            let previous = *lengths.last().ok_or_else(invalid)?;
                            let repeat = bits.read(2)? + 3;
                            lengths.extend(std::iter::repeat_n(previous, repeat));
                        }
                        17 => {
                            let repeat = bits.read(3)? + 3;
                            lengths.extend(std::iter::repeat_n(0, repeat));
                        }
                        18 => {
                            let repeat = bits.read(7)? + 11;
                            lengths.extend(std::iter::repeat_n(0, repeat));
                        }
                        _ => return Err(invalid()),
                    }
                    if lengths.len() > literals + distances {
                        return Err(invalid());
                    }
                }
                if lengths[256] == 0 {
                    return Err(invalid());
                }
                (
                    Huffman::new(&lengths[..literals], Alphabet::LiteralLength)?,
                    Huffman::new(&lengths[literals..], Alphabet::Distance)?,
                )
            };
            const LENGTH: [usize; 29] = [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83,
                99, 115, 131, 163, 195, 227, 258,
            ];
            const LENGTH_BITS: [usize; 29] = [
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                0,
            ];
            const DISTANCE: [usize; 30] = [
                1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
            ];
            const DISTANCE_BITS: [usize; 30] = [
                0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
                12, 12, 13, 13,
            ];
            loop {
                match literal.symbol(&mut bits)? {
                    value @ 0..=255 => {
                        if output.len() >= limit {
                            return Err(invalid());
                        }
                        output.push(value as u8);
                    }
                    256 => break,
                    value @ 257..=285 => {
                        let index = value - 257;
                        let length = LENGTH[index] + bits.read(LENGTH_BITS[index])?;
                        let index = distance.symbol(&mut bits)?;
                        if index >= DISTANCE.len() {
                            return Err(invalid());
                        }
                        let offset = DISTANCE[index] + bits.read(DISTANCE_BITS[index])?;
                        if offset > output.len()
                            || output.len().checked_add(length).is_none_or(|n| n > limit)
                        {
                            return Err(invalid());
                        }
                        for _ in 0..length {
                            output.push(output[output.len() - offset]);
                        }
                    }
                    _ => return Err(invalid()),
                }
            }
        } else {
            return Err(invalid());
        }
        if last != 0 {
            break;
        }
    }
    let (mut low, mut high) = (1u32, 0u32);
    for &byte in &output {
        low = (low + u32::from(byte)) % 65521;
        high = (high + low) % 65521;
    }
    if bytes[bytes.len() - 4..] != ((high << 16) | low).to_be_bytes() {
        return Err(invalid());
    }
    Ok(output)
}
