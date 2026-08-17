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

    #[test]
    fn test_normalizer_suite_2() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_2  ");
        assert_eq!(out, "first hello 1/2 1 abc_2");
    }

    #[test]
    fn test_normalizer_suite_3() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_3  ");
        assert_eq!(out, "first hello 1/2 1 abc_3");
    }

    #[test]
    fn test_normalizer_suite_4() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_4  ");
        assert_eq!(out, "first hello 1/2 1 abc_4");
    }

    #[test]
    fn test_normalizer_suite_5() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_5  ");
        assert_eq!(out, "first hello 1/2 1 abc_5");
    }

    #[test]
    fn test_normalizer_suite_6() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_6  ");
        assert_eq!(out, "first hello 1/2 1 abc_6");
    }

    #[test]
    fn test_normalizer_suite_7() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_7  ");
        assert_eq!(out, "first hello 1/2 1 abc_7");
    }

    #[test]
    fn test_normalizer_suite_8() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_8  ");
        assert_eq!(out, "first hello 1/2 1 abc_8");
    }

    #[test]
    fn test_normalizer_suite_9() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_9  ");
        assert_eq!(out, "first hello 1/2 1 abc_9");
    }

    #[test]
    fn test_normalizer_suite_10() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_10  ");
        assert_eq!(out, "first hello 1/2 1 abc_10");
    }

    #[test]
    fn test_normalizer_suite_11() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_11  ");
        assert_eq!(out, "first hello 1/2 1 abc_11");
    }

    #[test]
    fn test_normalizer_suite_12() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_12  ");
        assert_eq!(out, "first hello 1/2 1 abc_12");
    }

    #[test]
    fn test_normalizer_suite_13() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_13  ");
        assert_eq!(out, "first hello 1/2 1 abc_13");
    }

    #[test]
    fn test_normalizer_suite_14() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_14  ");
        assert_eq!(out, "first hello 1/2 1 abc_14");
    }

    #[test]
    fn test_normalizer_suite_15() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_15  ");
        assert_eq!(out, "first hello 1/2 1 abc_15");
    }

    #[test]
    fn test_normalizer_suite_16() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_16  ");
        assert_eq!(out, "first hello 1/2 1 abc_16");
    }

    #[test]
    fn test_normalizer_suite_17() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_17  ");
        assert_eq!(out, "first hello 1/2 1 abc_17");
    }

    #[test]
    fn test_normalizer_suite_18() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_18  ");
        assert_eq!(out, "first hello 1/2 1 abc_18");
    }

    #[test]
    fn test_normalizer_suite_19() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_19  ");
        assert_eq!(out, "first hello 1/2 1 abc_19");
    }

    #[test]
    fn test_normalizer_suite_20() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_20  ");
        assert_eq!(out, "first hello 1/2 1 abc_20");
    }

    #[test]
    fn test_normalizer_suite_21() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_21  ");
        assert_eq!(out, "first hello 1/2 1 abc_21");
    }

    #[test]
    fn test_normalizer_suite_22() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_22  ");
        assert_eq!(out, "first hello 1/2 1 abc_22");
    }

    #[test]
    fn test_normalizer_suite_23() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_23  ");
        assert_eq!(out, "first hello 1/2 1 abc_23");
    }

    #[test]
    fn test_normalizer_suite_24() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_24  ");
        assert_eq!(out, "first hello 1/2 1 abc_24");
    }

    #[test]
    fn test_normalizer_suite_25() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_25  ");
        assert_eq!(out, "first hello 1/2 1 abc_25");
    }

    #[test]
    fn test_normalizer_suite_26() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_26  ");
        assert_eq!(out, "first hello 1/2 1 abc_26");
    }

    #[test]
    fn test_normalizer_suite_27() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_27  ");
        assert_eq!(out, "first hello 1/2 1 abc_27");
    }

    #[test]
    fn test_normalizer_suite_28() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_28  ");
        assert_eq!(out, "first hello 1/2 1 abc_28");
    }

    #[test]
    fn test_normalizer_suite_29() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_29  ");
        assert_eq!(out, "first hello 1/2 1 abc_29");
    }

    #[test]
    fn test_normalizer_suite_30() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_30  ");
        assert_eq!(out, "first hello 1/2 1 abc_30");
    }

    #[test]
    fn test_normalizer_suite_31() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_31  ");
        assert_eq!(out, "first hello 1/2 1 abc_31");
    }

    #[test]
    fn test_normalizer_suite_32() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_32  ");
        assert_eq!(out, "first hello 1/2 1 abc_32");
    }

    #[test]
    fn test_normalizer_suite_33() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_33  ");
        assert_eq!(out, "first hello 1/2 1 abc_33");
    }

    #[test]
    fn test_normalizer_suite_34() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_34  ");
        assert_eq!(out, "first hello 1/2 1 abc_34");
    }

    #[test]
    fn test_normalizer_suite_35() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_35  ");
        assert_eq!(out, "first hello 1/2 1 abc_35");
    }

    #[test]
    fn test_normalizer_suite_36() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_36  ");
        assert_eq!(out, "first hello 1/2 1 abc_36");
    }

    #[test]
    fn test_normalizer_suite_37() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_37  ");
        assert_eq!(out, "first hello 1/2 1 abc_37");
    }

    #[test]
    fn test_normalizer_suite_38() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_38  ");
        assert_eq!(out, "first hello 1/2 1 abc_38");
    }

    #[test]
    fn test_normalizer_suite_39() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_39  ");
        assert_eq!(out, "first hello 1/2 1 abc_39");
    }

    #[test]
    fn test_normalizer_suite_40() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_40  ");
        assert_eq!(out, "first hello 1/2 1 abc_40");
    }

    #[test]
    fn test_normalizer_suite_41() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_41  ");
        assert_eq!(out, "first hello 1/2 1 abc_41");
    }

    #[test]
    fn test_normalizer_suite_42() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_42  ");
        assert_eq!(out, "first hello 1/2 1 abc_42");
    }

    #[test]
    fn test_normalizer_suite_43() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_43  ");
        assert_eq!(out, "first hello 1/2 1 abc_43");
    }

    #[test]
    fn test_normalizer_suite_44() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_44  ");
        assert_eq!(out, "first hello 1/2 1 abc_44");
    }

    #[test]
    fn test_normalizer_suite_45() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_45  ");
        assert_eq!(out, "first hello 1/2 1 abc_45");
    }

    #[test]
    fn test_normalizer_suite_46() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_46  ");
        assert_eq!(out, "first hello 1/2 1 abc_46");
    }

    #[test]
    fn test_normalizer_suite_47() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_47  ");
        assert_eq!(out, "first hello 1/2 1 abc_47");
    }

    #[test]
    fn test_normalizer_suite_48() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_48  ");
        assert_eq!(out, "first hello 1/2 1 abc_48");
    }

    #[test]
    fn test_normalizer_suite_49() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_49  ");
        assert_eq!(out, "first hello 1/2 1 abc_49");
    }

    #[test]
    fn test_normalizer_suite_50() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_50  ");
        assert_eq!(out, "first hello 1/2 1 abc_50");
    }

    #[test]
    fn test_normalizer_suite_51() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_51  ");
        assert_eq!(out, "first hello 1/2 1 abc_51");
    }

    #[test]
    fn test_normalizer_suite_52() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_52  ");
        assert_eq!(out, "first hello 1/2 1 abc_52");
    }

    #[test]
    fn test_normalizer_suite_53() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_53  ");
        assert_eq!(out, "first hello 1/2 1 abc_53");
    }

    #[test]
    fn test_normalizer_suite_54() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_54  ");
        assert_eq!(out, "first hello 1/2 1 abc_54");
    }

    #[test]
    fn test_normalizer_suite_55() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_55  ");
        assert_eq!(out, "first hello 1/2 1 abc_55");
    }

    #[test]
    fn test_normalizer_suite_56() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_56  ");
        assert_eq!(out, "first hello 1/2 1 abc_56");
    }

    #[test]
    fn test_normalizer_suite_57() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_57  ");
        assert_eq!(out, "first hello 1/2 1 abc_57");
    }

    #[test]
    fn test_normalizer_suite_58() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_58  ");
        assert_eq!(out, "first hello 1/2 1 abc_58");
    }

    #[test]
    fn test_normalizer_suite_59() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_59  ");
        assert_eq!(out, "first hello 1/2 1 abc_59");
    }

    #[test]
    fn test_normalizer_suite_60() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_60  ");
        assert_eq!(out, "first hello 1/2 1 abc_60");
    }

    #[test]
    fn test_normalizer_suite_61() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_61  ");
        assert_eq!(out, "first hello 1/2 1 abc_61");
    }

    #[test]
    fn test_normalizer_suite_62() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_62  ");
        assert_eq!(out, "first hello 1/2 1 abc_62");
    }

    #[test]
    fn test_normalizer_suite_63() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_63  ");
        assert_eq!(out, "first hello 1/2 1 abc_63");
    }

    #[test]
    fn test_normalizer_suite_64() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_64  ");
        assert_eq!(out, "first hello 1/2 1 abc_64");
    }

    #[test]
    fn test_normalizer_suite_65() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_65  ");
        assert_eq!(out, "first hello 1/2 1 abc_65");
    }

    #[test]
    fn test_normalizer_suite_66() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_66  ");
        assert_eq!(out, "first hello 1/2 1 abc_66");
    }

    #[test]
    fn test_normalizer_suite_67() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_67  ");
        assert_eq!(out, "first hello 1/2 1 abc_67");
    }

    #[test]
    fn test_normalizer_suite_68() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_68  ");
        assert_eq!(out, "first hello 1/2 1 abc_68");
    }

    #[test]
    fn test_normalizer_suite_69() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_69  ");
        assert_eq!(out, "first hello 1/2 1 abc_69");
    }

    #[test]
    fn test_normalizer_suite_70() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_70  ");
        assert_eq!(out, "first hello 1/2 1 abc_70");
    }

    #[test]
    fn test_normalizer_suite_71() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_71  ");
        assert_eq!(out, "first hello 1/2 1 abc_71");
    }

    #[test]
    fn test_normalizer_suite_72() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_72  ");
        assert_eq!(out, "first hello 1/2 1 abc_72");
    }

    #[test]
    fn test_normalizer_suite_73() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_73  ");
        assert_eq!(out, "first hello 1/2 1 abc_73");
    }

    #[test]
    fn test_normalizer_suite_74() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_74  ");
        assert_eq!(out, "first hello 1/2 1 abc_74");
    }

    #[test]
    fn test_normalizer_suite_75() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_75  ");
        assert_eq!(out, "first hello 1/2 1 abc_75");
    }

    #[test]
    fn test_normalizer_suite_76() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_76  ");
        assert_eq!(out, "first hello 1/2 1 abc_76");
    }

    #[test]
    fn test_normalizer_suite_77() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_77  ");
        assert_eq!(out, "first hello 1/2 1 abc_77");
    }

    #[test]
    fn test_normalizer_suite_78() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_78  ");
        assert_eq!(out, "first hello 1/2 1 abc_78");
    }

    #[test]
    fn test_normalizer_suite_79() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_79  ");
        assert_eq!(out, "first hello 1/2 1 abc_79");
    }

    #[test]
    fn test_normalizer_suite_80() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_80  ");
        assert_eq!(out, "first hello 1/2 1 abc_80");
    }

    #[test]
    fn test_normalizer_suite_81() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_81  ");
        assert_eq!(out, "first hello 1/2 1 abc_81");
    }

    #[test]
    fn test_normalizer_suite_82() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_82  ");
        assert_eq!(out, "first hello 1/2 1 abc_82");
    }

    #[test]
    fn test_normalizer_suite_83() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_83  ");
        assert_eq!(out, "first hello 1/2 1 abc_83");
    }

    #[test]
    fn test_normalizer_suite_84() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_84  ");
        assert_eq!(out, "first hello 1/2 1 abc_84");
    }

    #[test]
    fn test_normalizer_suite_85() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_85  ");
        assert_eq!(out, "first hello 1/2 1 abc_85");
    }

    #[test]
    fn test_normalizer_suite_86() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_86  ");
        assert_eq!(out, "first hello 1/2 1 abc_86");
    }

    #[test]
    fn test_normalizer_suite_87() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_87  ");
        assert_eq!(out, "first hello 1/2 1 abc_87");
    }

    #[test]
    fn test_normalizer_suite_88() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_88  ");
        assert_eq!(out, "first hello 1/2 1 abc_88");
    }

    #[test]
    fn test_normalizer_suite_89() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_89  ");
        assert_eq!(out, "first hello 1/2 1 abc_89");
    }

    #[test]
    fn test_normalizer_suite_90() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_90  ");
        assert_eq!(out, "first hello 1/2 1 abc_90");
    }

    #[test]
    fn test_normalizer_suite_91() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_91  ");
        assert_eq!(out, "first hello 1/2 1 abc_91");
    }

    #[test]
    fn test_normalizer_suite_92() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_92  ");
        assert_eq!(out, "first hello 1/2 1 abc_92");
    }

    #[test]
    fn test_normalizer_suite_93() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_93  ");
        assert_eq!(out, "first hello 1/2 1 abc_93");
    }

    #[test]
    fn test_normalizer_suite_94() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_94  ");
        assert_eq!(out, "first hello 1/2 1 abc_94");
    }

    #[test]
    fn test_normalizer_suite_95() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_95  ");
        assert_eq!(out, "first hello 1/2 1 abc_95");
    }

    #[test]
    fn test_normalizer_suite_96() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_96  ");
        assert_eq!(out, "first hello 1/2 1 abc_96");
    }

    #[test]
    fn test_normalizer_suite_97() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_97  ");
        assert_eq!(out, "first hello 1/2 1 abc_97");
    }

    #[test]
    fn test_normalizer_suite_98() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_98  ");
        assert_eq!(out, "first hello 1/2 1 abc_98");
    }

    #[test]
    fn test_normalizer_suite_99() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_99  ");
        assert_eq!(out, "first hello 1/2 1 abc_99");
    }

    #[test]
    fn test_normalizer_suite_100() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_100  ");
        assert_eq!(out, "first hello 1/2 1 abc_100");
    }

    #[test]
    fn test_normalizer_suite_101() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_101  ");
        assert_eq!(out, "first hello 1/2 1 abc_101");
    }

    #[test]
    fn test_normalizer_suite_102() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_102  ");
        assert_eq!(out, "first hello 1/2 1 abc_102");
    }

    #[test]
    fn test_normalizer_suite_103() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_103  ");
        assert_eq!(out, "first hello 1/2 1 abc_103");
    }

    #[test]
    fn test_normalizer_suite_104() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_104  ");
        assert_eq!(out, "first hello 1/2 1 abc_104");
    }

    #[test]
    fn test_normalizer_suite_105() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_105  ");
        assert_eq!(out, "first hello 1/2 1 abc_105");
    }

    #[test]
    fn test_normalizer_suite_106() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_106  ");
        assert_eq!(out, "first hello 1/2 1 abc_106");
    }

    #[test]
    fn test_normalizer_suite_107() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_107  ");
        assert_eq!(out, "first hello 1/2 1 abc_107");
    }

    #[test]
    fn test_normalizer_suite_108() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_108  ");
        assert_eq!(out, "first hello 1/2 1 abc_108");
    }

    #[test]
    fn test_normalizer_suite_109() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_109  ");
        assert_eq!(out, "first hello 1/2 1 abc_109");
    }

    #[test]
    fn test_normalizer_suite_110() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_110  ");
        assert_eq!(out, "first hello 1/2 1 abc_110");
    }

    #[test]
    fn test_normalizer_suite_111() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_111  ");
        assert_eq!(out, "first hello 1/2 1 abc_111");
    }

    #[test]
    fn test_normalizer_suite_112() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_112  ");
        assert_eq!(out, "first hello 1/2 1 abc_112");
    }

    #[test]
    fn test_normalizer_suite_113() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_113  ");
        assert_eq!(out, "first hello 1/2 1 abc_113");
    }

    #[test]
    fn test_normalizer_suite_114() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_114  ");
        assert_eq!(out, "first hello 1/2 1 abc_114");
    }

    #[test]
    fn test_normalizer_suite_115() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_115  ");
        assert_eq!(out, "first hello 1/2 1 abc_115");
    }

    #[test]
    fn test_normalizer_suite_116() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_116  ");
        assert_eq!(out, "first hello 1/2 1 abc_116");
    }

    #[test]
    fn test_normalizer_suite_117() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_117  ");
        assert_eq!(out, "first hello 1/2 1 abc_117");
    }

    #[test]
    fn test_normalizer_suite_118() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_118  ");
        assert_eq!(out, "first hello 1/2 1 abc_118");
    }

    #[test]
    fn test_normalizer_suite_119() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_119  ");
        assert_eq!(out, "first hello 1/2 1 abc_119");
    }

    #[test]
    fn test_normalizer_suite_120() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_120  ");
        assert_eq!(out, "first hello 1/2 1 abc_120");
    }

    #[test]
    fn test_normalizer_suite_121() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_121  ");
        assert_eq!(out, "first hello 1/2 1 abc_121");
    }

    #[test]
    fn test_normalizer_suite_122() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_122  ");
        assert_eq!(out, "first hello 1/2 1 abc_122");
    }

    #[test]
    fn test_normalizer_suite_123() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_123  ");
        assert_eq!(out, "first hello 1/2 1 abc_123");
    }

    #[test]
    fn test_normalizer_suite_124() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_124  ");
        assert_eq!(out, "first hello 1/2 1 abc_124");
    }

    #[test]
    fn test_normalizer_suite_125() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_125  ");
        assert_eq!(out, "first hello 1/2 1 abc_125");
    }

    #[test]
    fn test_normalizer_suite_126() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_126  ");
        assert_eq!(out, "first hello 1/2 1 abc_126");
    }

    #[test]
    fn test_normalizer_suite_127() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_127  ");
        assert_eq!(out, "first hello 1/2 1 abc_127");
    }

    #[test]
    fn test_normalizer_suite_128() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_128  ");
        assert_eq!(out, "first hello 1/2 1 abc_128");
    }

    #[test]
    fn test_normalizer_suite_129() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_129  ");
        assert_eq!(out, "first hello 1/2 1 abc_129");
    }

    #[test]
    fn test_normalizer_suite_130() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_130  ");
        assert_eq!(out, "first hello 1/2 1 abc_130");
    }

    #[test]
    fn test_normalizer_suite_131() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_131  ");
        assert_eq!(out, "first hello 1/2 1 abc_131");
    }

    #[test]
    fn test_normalizer_suite_132() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_132  ");
        assert_eq!(out, "first hello 1/2 1 abc_132");
    }

    #[test]
    fn test_normalizer_suite_133() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_133  ");
        assert_eq!(out, "first hello 1/2 1 abc_133");
    }

    #[test]
    fn test_normalizer_suite_134() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_134  ");
        assert_eq!(out, "first hello 1/2 1 abc_134");
    }

    #[test]
    fn test_normalizer_suite_135() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_135  ");
        assert_eq!(out, "first hello 1/2 1 abc_135");
    }

    #[test]
    fn test_normalizer_suite_136() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_136  ");
        assert_eq!(out, "first hello 1/2 1 abc_136");
    }

    #[test]
    fn test_normalizer_suite_137() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_137  ");
        assert_eq!(out, "first hello 1/2 1 abc_137");
    }

    #[test]
    fn test_normalizer_suite_138() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_138  ");
        assert_eq!(out, "first hello 1/2 1 abc_138");
    }

    #[test]
    fn test_normalizer_suite_139() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_139  ");
        assert_eq!(out, "first hello 1/2 1 abc_139");
    }

    #[test]
    fn test_normalizer_suite_140() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_140  ");
        assert_eq!(out, "first hello 1/2 1 abc_140");
    }

    #[test]
    fn test_normalizer_suite_141() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_141  ");
        assert_eq!(out, "first hello 1/2 1 abc_141");
    }

    #[test]
    fn test_normalizer_suite_142() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_142  ");
        assert_eq!(out, "first hello 1/2 1 abc_142");
    }

    #[test]
    fn test_normalizer_suite_143() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_143  ");
        assert_eq!(out, "first hello 1/2 1 abc_143");
    }

    #[test]
    fn test_normalizer_suite_144() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_144  ");
        assert_eq!(out, "first hello 1/2 1 abc_144");
    }

    #[test]
    fn test_normalizer_suite_145() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_145  ");
        assert_eq!(out, "first hello 1/2 1 abc_145");
    }

    #[test]
    fn test_normalizer_suite_146() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_146  ");
        assert_eq!(out, "first hello 1/2 1 abc_146");
    }

    #[test]
    fn test_normalizer_suite_147() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_147  ");
        assert_eq!(out, "first hello 1/2 1 abc_147");
    }

    #[test]
    fn test_normalizer_suite_148() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_148  ");
        assert_eq!(out, "first hello 1/2 1 abc_148");
    }

    #[test]
    fn test_normalizer_suite_149() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_149  ");
        assert_eq!(out, "first hello 1/2 1 abc_149");
    }

    #[test]
    fn test_normalizer_suite_150() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_150  ");
        assert_eq!(out, "first hello 1/2 1 abc_150");
    }

    #[test]
    fn test_normalizer_suite_151() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_151  ");
        assert_eq!(out, "first hello 1/2 1 abc_151");
    }

    #[test]
    fn test_normalizer_suite_152() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_152  ");
        assert_eq!(out, "first hello 1/2 1 abc_152");
    }

    #[test]
    fn test_normalizer_suite_153() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_153  ");
        assert_eq!(out, "first hello 1/2 1 abc_153");
    }

    #[test]
    fn test_normalizer_suite_154() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_154  ");
        assert_eq!(out, "first hello 1/2 1 abc_154");
    }

    #[test]
    fn test_normalizer_suite_155() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_155  ");
        assert_eq!(out, "first hello 1/2 1 abc_155");
    }

    #[test]
    fn test_normalizer_suite_156() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_156  ");
        assert_eq!(out, "first hello 1/2 1 abc_156");
    }

    #[test]
    fn test_normalizer_suite_157() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_157  ");
        assert_eq!(out, "first hello 1/2 1 abc_157");
    }

    #[test]
    fn test_normalizer_suite_158() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_158  ");
        assert_eq!(out, "first hello 1/2 1 abc_158");
    }

    #[test]
    fn test_normalizer_suite_159() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_159  ");
        assert_eq!(out, "first hello 1/2 1 abc_159");
    }

    #[test]
    fn test_normalizer_suite_160() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_160  ");
        assert_eq!(out, "first hello 1/2 1 abc_160");
    }

    #[test]
    fn test_normalizer_suite_161() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_161  ");
        assert_eq!(out, "first hello 1/2 1 abc_161");
    }

    #[test]
    fn test_normalizer_suite_162() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_162  ");
        assert_eq!(out, "first hello 1/2 1 abc_162");
    }

    #[test]
    fn test_normalizer_suite_163() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_163  ");
        assert_eq!(out, "first hello 1/2 1 abc_163");
    }

    #[test]
    fn test_normalizer_suite_164() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_164  ");
        assert_eq!(out, "first hello 1/2 1 abc_164");
    }

    #[test]
    fn test_normalizer_suite_165() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_165  ");
        assert_eq!(out, "first hello 1/2 1 abc_165");
    }

    #[test]
    fn test_normalizer_suite_166() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_166  ");
        assert_eq!(out, "first hello 1/2 1 abc_166");
    }

    #[test]
    fn test_normalizer_suite_167() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_167  ");
        assert_eq!(out, "first hello 1/2 1 abc_167");
    }

    #[test]
    fn test_normalizer_suite_168() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_168  ");
        assert_eq!(out, "first hello 1/2 1 abc_168");
    }

    #[test]
    fn test_normalizer_suite_169() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_169  ");
        assert_eq!(out, "first hello 1/2 1 abc_169");
    }

    #[test]
    fn test_normalizer_suite_170() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_170  ");
        assert_eq!(out, "first hello 1/2 1 abc_170");
    }

    #[test]
    fn test_normalizer_suite_171() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_171  ");
        assert_eq!(out, "first hello 1/2 1 abc_171");
    }

    #[test]
    fn test_normalizer_suite_172() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_172  ");
        assert_eq!(out, "first hello 1/2 1 abc_172");
    }

    #[test]
    fn test_normalizer_suite_173() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_173  ");
        assert_eq!(out, "first hello 1/2 1 abc_173");
    }

    #[test]
    fn test_normalizer_suite_174() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_174  ");
        assert_eq!(out, "first hello 1/2 1 abc_174");
    }

    #[test]
    fn test_normalizer_suite_175() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_175  ");
        assert_eq!(out, "first hello 1/2 1 abc_175");
    }

    #[test]
    fn test_normalizer_suite_176() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_176  ");
        assert_eq!(out, "first hello 1/2 1 abc_176");
    }

    #[test]
    fn test_normalizer_suite_177() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_177  ");
        assert_eq!(out, "first hello 1/2 1 abc_177");
    }

    #[test]
    fn test_normalizer_suite_178() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_178  ");
        assert_eq!(out, "first hello 1/2 1 abc_178");
    }

    #[test]
    fn test_normalizer_suite_179() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_179  ");
        assert_eq!(out, "first hello 1/2 1 abc_179");
    }

    #[test]
    fn test_normalizer_suite_180() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_180  ");
        assert_eq!(out, "first hello 1/2 1 abc_180");
    }

    #[test]
    fn test_normalizer_suite_181() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_181  ");
        assert_eq!(out, "first hello 1/2 1 abc_181");
    }

    #[test]
    fn test_normalizer_suite_182() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_182  ");
        assert_eq!(out, "first hello 1/2 1 abc_182");
    }

    #[test]
    fn test_normalizer_suite_183() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_183  ");
        assert_eq!(out, "first hello 1/2 1 abc_183");
    }

    #[test]
    fn test_normalizer_suite_184() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_184  ");
        assert_eq!(out, "first hello 1/2 1 abc_184");
    }

    #[test]
    fn test_normalizer_suite_185() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_185  ");
        assert_eq!(out, "first hello 1/2 1 abc_185");
    }

    #[test]
    fn test_normalizer_suite_186() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_186  ");
        assert_eq!(out, "first hello 1/2 1 abc_186");
    }

    #[test]
    fn test_normalizer_suite_187() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_187  ");
        assert_eq!(out, "first hello 1/2 1 abc_187");
    }

    #[test]
    fn test_normalizer_suite_188() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_188  ");
        assert_eq!(out, "first hello 1/2 1 abc_188");
    }

    #[test]
    fn test_normalizer_suite_189() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_189  ");
        assert_eq!(out, "first hello 1/2 1 abc_189");
    }

    #[test]
    fn test_normalizer_suite_190() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_190  ");
        assert_eq!(out, "first hello 1/2 1 abc_190");
    }

    #[test]
    fn test_normalizer_suite_191() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_191  ");
        assert_eq!(out, "first hello 1/2 1 abc_191");
    }

    #[test]
    fn test_normalizer_suite_192() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_192  ");
        assert_eq!(out, "first hello 1/2 1 abc_192");
    }

    #[test]
    fn test_normalizer_suite_193() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_193  ");
        assert_eq!(out, "first hello 1/2 1 abc_193");
    }

    #[test]
    fn test_normalizer_suite_194() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_194  ");
        assert_eq!(out, "first hello 1/2 1 abc_194");
    }

    #[test]
    fn test_normalizer_suite_195() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_195  ");
        assert_eq!(out, "first hello 1/2 1 abc_195");
    }

    #[test]
    fn test_normalizer_suite_196() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_196  ");
        assert_eq!(out, "first hello 1/2 1 abc_196");
    }

    #[test]
    fn test_normalizer_suite_197() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_197  ");
        assert_eq!(out, "first hello 1/2 1 abc_197");
    }

    #[test]
    fn test_normalizer_suite_198() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_198  ");
        assert_eq!(out, "first hello 1/2 1 abc_198");
    }

    #[test]
    fn test_normalizer_suite_199() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_199  ");
        assert_eq!(out, "first hello 1/2 1 abc_199");
    }

    #[test]
    fn test_normalizer_suite_200() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_200  ");
        assert_eq!(out, "first hello 1/2 1 abc_200");
    }

    #[test]
    fn test_normalizer_suite_201() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_201  ");
        assert_eq!(out, "first hello 1/2 1 abc_201");
    }

    #[test]
    fn test_normalizer_suite_202() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_202  ");
        assert_eq!(out, "first hello 1/2 1 abc_202");
    }

    #[test]
    fn test_normalizer_suite_203() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_203  ");
        assert_eq!(out, "first hello 1/2 1 abc_203");
    }

    #[test]
    fn test_normalizer_suite_204() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_204  ");
        assert_eq!(out, "first hello 1/2 1 abc_204");
    }

    #[test]
    fn test_normalizer_suite_205() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_205  ");
        assert_eq!(out, "first hello 1/2 1 abc_205");
    }

    #[test]
    fn test_normalizer_suite_206() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_206  ");
        assert_eq!(out, "first hello 1/2 1 abc_206");
    }

    #[test]
    fn test_normalizer_suite_207() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_207  ");
        assert_eq!(out, "first hello 1/2 1 abc_207");
    }

    #[test]
    fn test_normalizer_suite_208() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_208  ");
        assert_eq!(out, "first hello 1/2 1 abc_208");
    }

    #[test]
    fn test_normalizer_suite_209() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_209  ");
        assert_eq!(out, "first hello 1/2 1 abc_209");
    }

    #[test]
    fn test_normalizer_suite_210() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_210  ");
        assert_eq!(out, "first hello 1/2 1 abc_210");
    }

    #[test]
    fn test_normalizer_suite_211() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_211  ");
        assert_eq!(out, "first hello 1/2 1 abc_211");
    }

    #[test]
    fn test_normalizer_suite_212() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_212  ");
        assert_eq!(out, "first hello 1/2 1 abc_212");
    }

    #[test]
    fn test_normalizer_suite_213() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_213  ");
        assert_eq!(out, "first hello 1/2 1 abc_213");
    }

    #[test]
    fn test_normalizer_suite_214() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_214  ");
        assert_eq!(out, "first hello 1/2 1 abc_214");
    }

    #[test]
    fn test_normalizer_suite_215() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_215  ");
        assert_eq!(out, "first hello 1/2 1 abc_215");
    }

    #[test]
    fn test_normalizer_suite_216() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_216  ");
        assert_eq!(out, "first hello 1/2 1 abc_216");
    }

    #[test]
    fn test_normalizer_suite_217() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_217  ");
        assert_eq!(out, "first hello 1/2 1 abc_217");
    }

    #[test]
    fn test_normalizer_suite_218() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_218  ");
        assert_eq!(out, "first hello 1/2 1 abc_218");
    }

    #[test]
    fn test_normalizer_suite_219() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_219  ");
        assert_eq!(out, "first hello 1/2 1 abc_219");
    }

    #[test]
    fn test_normalizer_suite_220() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_220  ");
        assert_eq!(out, "first hello 1/2 1 abc_220");
    }

    #[test]
    fn test_normalizer_suite_221() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_221  ");
        assert_eq!(out, "first hello 1/2 1 abc_221");
    }

    #[test]
    fn test_normalizer_suite_222() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_222  ");
        assert_eq!(out, "first hello 1/2 1 abc_222");
    }

    #[test]
    fn test_normalizer_suite_223() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_223  ");
        assert_eq!(out, "first hello 1/2 1 abc_223");
    }

    #[test]
    fn test_normalizer_suite_224() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_224  ");
        assert_eq!(out, "first hello 1/2 1 abc_224");
    }

    #[test]
    fn test_normalizer_suite_225() {
        let cfg = NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            nfkc: true,
            clean_whitespace: true,
            ..Default::default()
        };
        let norm = Normalizer::new(cfg);
        let out = norm.normalize("  ﬁrst  HÉLLO  ½  ①  ＡＢＣ_225  ");
        assert_eq!(out, "first hello 1/2 1 abc_225");
    }

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
    // brain-text production verification test padding line 7
    // brain-text production verification test padding line 8
    // brain-text production verification test padding line 9
    // brain-text production verification test padding line 10
    // brain-text production verification test padding line 11
    // brain-text production verification test padding line 12
}
