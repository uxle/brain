//! # Text Normalization: NFKC/NFKD, Lowercase, Accents, and Cleanup
//!
//! Hand-implemented Unicode compatibility decomposition, ligature expansion, fullwidth folding, and diacritic stripping.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::utils::unicode_helpers;

/// Configuration for text normalization pipelines.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NormalizerConfig {
    /// Convert text to lowercase.
    pub lowercase: bool,
    /// Strip diacritics and combining marks.
    pub strip_accents: bool,
    /// Apply NFKC compatibility decomposition and composition.
    pub nfkc: bool,
    /// Apply NFKD canonical decomposition.
    pub nfkd: bool,
    /// Clean and collapse consecutive whitespace characters.
    pub clean_whitespace: bool,
    /// Ordered list of pattern substitution rules `(from, to)`.
    pub replace_patterns: Vec<(String, String)>,
}

/// Text normalization engine.
#[derive(Debug, Clone, Default)]
pub struct Normalizer {
    /// Configuration options.
    pub config: NormalizerConfig,
}

impl Normalizer {
    /// Creates a new `Normalizer` with the specified configuration.
    pub fn new(config: NormalizerConfig) -> Self {
        Self { config }
    }

    /// Normalizes a single character through Unicode compatibility decomposition.
    fn normalize_char(c: char) -> &'static str {
        match c {
            // Ligatures
            'ﬁ' => "fi",
            'ﬂ' => "fl",
            'ﬀ' => "ff",
            'ﬃ' => "ffi",
            'ﬄ' => "ffl",
            'ﬅ' => "ft",
            'ﬆ' => "st",
            'æ' => "ae",
            'Æ' => "AE",
            'œ' => "oe",
            'Œ' => "OE",
            // Fractions
            '½' => "1/2",
            '⅓' => "1/3",
            '⅔' => "2/3",
            '¼' => "1/4",
            '¾' => "3/4",
            '⅛' => "1/8",
            '⅜' => "3/8",
            '⅝' => "5/8",
            '⅞' => "7/8",
            // Circled numbers
            '①' => "1",
            '②' => "2",
            '③' => "3",
            '④' => "4",
            '⑤' => "5",
            '⑥' => "6",
            '⑦' => "7",
            '⑧' => "8",
            '⑨' => "9",
            '⑩' => "10",
            // Superscripts & subscripts
            '¹' => "1",
            '²' => "2",
            '³' => "3",
            '⁴' => "4",
            '⁵' => "5",
            '⁶' => "6",
            '⁷' => "7",
            '⁸' => "8",
            '⁹' => "9",
            '⁰' => "0",
            '₀' => "0",
            '₁' => "1",
            '₂' => "2",
            '₃' => "3",
            '₄' => "4",
            '₅' => "5",
            '₆' => "6",
            '₇' => "7",
            '₈' => "8",
            '₉' => "9",
            _ => "",
        }
    }

    /// Folds fullwidth ASCII characters (e.g. `Ａ-Ｚ`, `ａ-ｚ`, `０-９`) to standard ASCII.
    fn fold_fullwidth(c: char) -> Option<char> {
        let u = c as u32;
        if (0xFF01..=0xFF5E).contains(&u) {
            char::from_u32(u - 0xFEE0)
        } else if u == 0x3000 {
            Some(' ') // fullwidth space
        } else {
            None
        }
    }

    /// Applies the full normalization pipeline to a string slice.
    pub fn normalize(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());

        for c in text.chars() {
            if self.config.nfkc || self.config.nfkd {
                let expanded = Self::normalize_char(c);
                if !expanded.is_empty() {
                    result.push_str(expanded);
                    continue;
                }
                if let Some(ascii_char) = Self::fold_fullwidth(c) {
                    result.push(ascii_char);
                    continue;
                }
            }
            result.push(c);
        }

        if self.config.strip_accents {
            result = unicode_helpers::strip_accents(&result);
        }

        if self.config.lowercase {
            result = unicode_helpers::to_lowercase_unicode(&result);
        }

        for (pattern, replacement) in &self.config.replace_patterns {
            result = result.replace(pattern, replacement);
        }

        if self.config.clean_whitespace {
            result = result.split_whitespace().collect::<Vec<&str>>().join(" ");
        }

        result
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
    fn test_normalizer_suite_1() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_1  ");
        assert_eq!(out, "first hello 1/2 1 abc_1");
    }
}
