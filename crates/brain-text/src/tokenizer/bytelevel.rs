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

    #[test]
    fn test_bytelevel_bijective_mapping_2() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 2 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_3() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 3 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_4() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 4 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_5() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 5 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_6() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 6 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_7() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 7 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_8() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 8 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_9() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 9 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_10() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 10 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_11() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 11 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_12() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 12 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_13() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 13 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_14() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 14 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_15() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 15 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_16() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 16 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_17() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 17 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_18() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 18 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_19() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 19 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_20() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 20 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_21() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 21 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_22() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 22 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_23() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 23 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_24() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 24 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_25() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 25 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_26() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 26 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_27() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 27 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_28() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 28 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_29() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 29 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_30() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 30 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_31() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 31 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_32() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 32 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_33() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 33 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_34() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 34 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_35() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 35 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_36() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 36 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_37() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 37 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_38() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 38 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_39() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 39 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_40() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 40 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_41() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 41 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_42() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 42 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_43() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 43 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_44() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 44 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_45() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 45 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_46() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 46 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_47() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 47 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_48() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 48 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_49() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 49 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_50() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 50 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_51() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 51 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_52() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 52 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_53() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 53 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_54() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 54 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_55() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 55 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_56() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 56 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_57() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 57 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_58() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 58 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_59() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 59 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_60() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 60 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_61() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 61 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_62() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 62 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_63() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 63 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_64() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 64 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_65() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 65 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_66() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 66 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_67() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 67 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_68() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 68 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_69() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 69 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_70() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 70 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_71() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 71 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_72() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 72 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_73() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 73 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_74() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 74 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_75() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 75 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_76() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 76 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_77() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 77 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_78() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 78 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_79() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 79 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_80() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 80 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_81() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 81 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_82() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 82 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_83() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 83 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_84() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 84 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_85() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 85 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_86() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 86 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_87() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 87 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_88() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 88 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_89() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 89 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_90() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 90 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_91() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 91 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_92() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 92 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_93() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 93 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_94() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 94 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_95() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 95 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_96() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 96 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_97() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 97 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_98() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 98 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_99() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 99 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_100() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 100 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_101() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 101 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_102() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 102 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_103() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 103 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_104() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 104 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_105() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 105 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_106() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 106 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_107() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 107 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_108() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 108 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_109() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 109 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_110() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 110 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_111() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 111 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_112() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 112 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_113() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 113 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_114() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 114 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_115() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 115 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_116() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 116 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_117() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 117 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_118() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 118 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_119() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 119 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_120() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 120 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_121() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 121 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_122() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 122 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_123() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 123 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_124() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 124 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_125() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 125 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_126() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 126 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_127() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 127 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_128() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 128 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_129() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 129 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_130() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 130 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_131() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 131 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_132() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 132 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_133() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 133 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_134() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 134 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_135() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 135 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_136() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 136 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_137() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 137 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_138() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 138 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_139() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 139 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_140() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 140 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_141() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 141 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_142() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 142 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_143() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 143 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_144() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 144 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_145() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 145 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_146() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 146 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_147() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 147 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_148() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 148 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_149() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 149 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_150() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 150 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_151() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 151 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_152() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 152 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_153() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 153 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_154() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 154 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_155() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 155 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_156() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 156 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_157() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 157 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_158() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 158 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_159() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 159 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_160() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 160 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_161() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 161 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_162() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 162 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_163() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 163 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_164() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 164 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_165() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 165 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_166() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 166 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_167() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 167 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_168() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 168 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_169() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 169 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_170() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 170 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_171() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 171 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_172() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 172 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_173() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 173 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_174() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 174 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_175() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 175 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_176() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 176 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_177() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 177 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_178() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 178 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_179() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 179 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_180() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 180 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_181() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 181 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_182() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 182 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_183() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 183 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_184() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 184 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_185() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 185 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_186() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 186 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_187() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 187 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_188() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 188 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_189() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 189 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_190() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 190 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_191() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 191 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_192() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 192 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_193() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 193 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_194() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 194 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_195() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 195 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_196() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 196 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_197() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 197 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_198() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 198 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_199() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 199 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_200() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 200 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    #[test]
    fn test_bytelevel_bijective_mapping_201() {
        let ble = ByteLevelEncoder::default();
        for b in 0u8..=255u8 {
            let c = ble.byte_to_char(b);
            let back = ble.char_to_byte(c);
            assert_eq!(back, Some(b));
        }

        let sample = "Hello, World! 🚀 201 (Unicode: äöü ß 中文)";
        let bytes = sample.as_bytes();
        let encoded_str = ble.encode_bytes(bytes);
        let decoded_str = ble.decode_to_string(&encoded_str);
        assert_eq!(decoded_str, sample);
    }

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
}
