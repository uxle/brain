//! # Unicode, Byte, and Mathematical Text Utilities
//!
//! Character transformations, diacritic stripping, edit distance algorithms, and string metrics.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult};
use std::collections::HashSet;

/// Deterministic Pseudo-Random Number Generator (XorShift64) for text augmentations.
#[derive(Debug, Clone)]
pub struct TextRng {
    state: u64,
}

impl TextRng {
    /// Creates a new `TextRng` with a non-zero seed.
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state: s }
    }

    /// Generates next pseudo-random 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates next uniform floating point value in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generates a random integer in `[0, upper)`.
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() % (upper as u64)) as usize
        }
    }
}

impl Default for TextRng {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Unicode character and string transformation helpers.
pub mod unicode_helpers {
    /// Converts a string slice to lowercase adhering to Unicode casing rules.
    pub fn to_lowercase_unicode(s: &str) -> String {
        s.to_lowercase()
    }

    /// Converts a string slice to uppercase adhering to Unicode casing rules.
    pub fn to_uppercase_unicode(s: &str) -> String {
        s.to_uppercase()
    }

    /// Strips accents and combining diacritical marks from characters.
    pub fn strip_accents(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' => result.push('a'),
                'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' => result.push('e'),
                'í' | 'ì' | 'î' | 'ï' | 'ī' | 'į' => result.push('i'),
                'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' | 'ō' => result.push('o'),
                'ú' | 'ù' | 'û' | 'ü' | 'ū' | 'ų' => result.push('u'),
                'ý' | 'ÿ' => result.push('y'),
                'ç' | 'ć' | 'č' => result.push('c'),
                'ñ' | 'ń' => result.push('n'),
                'š' | 'ś' => result.push('s'),
                'ž' | 'ź' | 'ż' => result.push('z'),
                'Á' | 'À' | 'Â' | 'Ä' | 'Ã' | 'Å' | 'Ā' => result.push('A'),
                'É' | 'È' | 'Ê' | 'Ë' | 'Ē' | 'Ė' | 'Ę' => result.push('E'),
                'Í' | 'Ì' | 'Î' | 'Ï' | 'Ī' | 'Į' => result.push('I'),
                'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' | 'Ø' | 'Ō' => result.push('O'),
                'Ú' | 'Ù' | 'Û' | 'Ü' | 'Ū' | 'Ų' => result.push('U'),
                'Ý' => result.push('Y'),
                'Ç' | 'Ć' | 'Č' => result.push('C'),
                'Ñ' | 'Ń' => result.push('N'),
                'Š' | 'Ś' => result.push('S'),
                'Ž' | 'Ź' | 'Ż' => result.push('Z'),
                _ => result.push(c),
            }
        }
        result
    }

    /// Returns true if character is considered whitespace.
    pub fn is_whitespace(c: char) -> bool {
        c.is_whitespace() || c == '\u{00A0}' || c == '\u{200B}'
    }

    /// Returns true if character is a punctuation symbol.
    pub fn is_punctuation(c: char) -> bool {
        c.is_ascii_punctuation()
            || ('\u{2000}'..='\u{206F}').contains(&c)
            || ('\u{2E00}'..='\u{2E7F}').contains(&c)
            || ('\u{3000}'..='\u{303F}').contains(&c)
    }

    /// Returns true if character is a control character.
    pub fn is_control(c: char) -> bool {
        c.is_control() && !c.is_whitespace()
    }

    /// Returns true if character is within CJK Unified Ideographs range.
    pub fn is_chinese_char(c: char) -> bool {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u)
            || (0x3400..=0x4DBF).contains(&u)
            || (0x20000..=0x2A6DF).contains(&u)
            || (0xF900..=0xFAFF).contains(&u)
    }
}

/// Splits text into tokens on contiguous whitespace boundaries.
pub fn split_ws(s: &str) -> Vec<String> {
    s.split_whitespace().map(|w| w.to_string()).collect()
}

/// Encodes a string slice into UTF-8 bytes.
pub fn byte_encode(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// Decodes UTF-8 bytes into a standard Rust String.
pub fn byte_decode(bytes: &[u8]) -> TextResult<String> {
    String::from_utf8(bytes.to_vec()).map_err(|e| TextError::DecodingFailed(e.to_string()))
}

/// Computes Levenshtein edit distance between two string slices.
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let n = chars1.len();
    let m = chars2.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n][m]
}

/// Computes Damerau-Levenshtein edit distance (allowing adjacent transpositions).
pub fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let n = chars1.len();
    let m = chars2.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);

            if i > 1 && j > 1 && chars1[i - 1] == chars2[j - 2] && chars1[i - 2] == chars2[j - 1] {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + cost);
            }
        }
    }
    dp[n][m]
}

/// Computes Jaccard similarity between two token collections.
pub fn jaccard_similarity(set1: &[String], set2: &[String]) -> f64 {
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    let s1: HashSet<&String> = set1.iter().collect();
    let s2: HashSet<&String> = set2.iter().collect();
    let intersection = s1.intersection(&s2).count();
    let union = s1.union(&s2).count();
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Computes cosine similarity between two numeric slices.
pub fn cosine_similarity_slice(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm1 = 0.0f32;
    let mut norm2 = 0.0f32;
    for i in 0..v1.len() {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if norm1 <= 0.0 || norm2 <= 0.0 {
        0.0
    } else {
        dot / (norm1.sqrt() * norm2.sqrt())
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
    fn test_utils_algorithms_1() {
        let mut rng = TextRng::new(1 as u64);
        let f = rng.next_f32();
        assert!((0.0..1.0).contains(&f));
        let r = rng.gen_range(100);
        assert!(r < 100);

        let s = "Héllo Wörld 1";
        let stripped = unicode_helpers::strip_accents(s);
        assert_eq!(stripped, "Hello World 1");

        let ws = split_ws("hello   world   1");
        assert_eq!(ws, vec!["hello", "world", "1"]);

        let enc = byte_encode("abc_1");
        let dec = byte_decode(&enc).unwrap();
        assert_eq!(dec, "abc_1");

        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);

        let t1 = vec!["a".to_string(), "b".to_string()];
        let t2 = vec!["b".to_string(), "c".to_string()];
        let jacc = jaccard_similarity(&t1, &t2);
        assert!((jacc - 1.0/3.0).abs() < 1e-5);

        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!((cosine_similarity_slice(&v1, &v2) - 1.0).abs() < 1e-5);
    }
}
