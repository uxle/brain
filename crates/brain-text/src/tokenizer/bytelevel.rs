//! # Byte-Level Encoding & Unicode Byte Mapping (GPT-2 Style)
//!
//! Bijective, reversible 1-to-1 byte <-> Unicode mapping guaranteeing full arbitrary binary and text round-tripping.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use std::collections::HashMap;

/// Configuration for Byte-Level tokenization.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ByteLevelConfig {
    /// Prepend prefix space to words.
    pub add_prefix_space: bool,
    /// Trim whitespace offsets.
    pub trim_offsets: bool,
    /// Use regex pre-tokenization.
    pub use_regex: bool,
}

/// Bijective Byte-Level Encoder / Decoder (GPT-2 character mapping).
#[derive(Debug, Clone)]
pub struct ByteLevelEncoder {
    /// Configuration options.
    pub config: ByteLevelConfig,
    byte_to_char_table: [char; 256],
    char_to_byte_table: HashMap<char, u8>,
}

impl Default for ByteLevelEncoder {
    fn default() -> Self {
        Self::new(ByteLevelConfig::default())
    }
}

impl ByteLevelEncoder {
    /// Creates a new `ByteLevelEncoder`.
    pub fn new(config: ByteLevelConfig) -> Self {
        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = HashMap::with_capacity(256);
        let mut n = 0u32;

        for b in 0u8..=255u8 {
            let is_printable = (b >= b'!' && b <= b'~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            let c = if is_printable {
                b as char
            } else {
                let shifted = 256 + n;
                n += 1;
                char::from_u32(shifted).unwrap_or('?')
            };

            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b);
        }

        Self {
            config,
            byte_to_char_table: byte_to_char,
            char_to_byte_table: char_to_byte,
        }
    }

    /// Converts a single byte `0..255` into its unique mapped Unicode character.
    pub fn byte_to_char(&self, b: u8) -> char {
        self.byte_to_char_table[b as usize]
    }

    /// Converts a mapped Unicode character back into its original byte `0..255`.
    pub fn char_to_byte(&self, c: char) -> Option<u8> {
        self.char_to_byte_table.get(&c).copied()
    }

    /// Encodes arbitrary raw byte slice into a mapped Unicode string.
    pub fn encode_bytes(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.byte_to_char(b)).collect()
    }

    /// Decodes a mapped Unicode string back into original raw bytes.
    pub fn decode_bytes(&self, text: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(text.len());
        for c in text.chars() {
            if let Some(b) = self.char_to_byte(c) {
                bytes.push(b);
            }
        }
        bytes
    }

    /// Decodes a mapped Unicode string back into a standard UTF-8 string (lossy fallback if invalid).
    pub fn decode_to_string(&self, text: &str) -> String {
        let bytes = self.decode_bytes(text);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::vocab::*;
    use crate::text_ops::*;
    use crate::features::*;
    use crate::similarity::*;
    use crate::lm::*;
    use crate::process::*;
    use crate::optimize::*;
    use crate::analyze::*;
    use crate::compute::*;
    use crate::helper::*;
    use crate::transform::*;
    use crate::builder::*;
    use crate::tokenizer::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::post::*;
    use crate::embedding::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::fasttext::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_bytelevel_bijective_mapping_1() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 1 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }
}
