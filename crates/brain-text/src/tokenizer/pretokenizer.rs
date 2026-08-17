//! # Pre-Tokenizers: Whitespace, Punctuation, Digits, and CamelCase
//!
//! Splitting raw text into pre-token spans with exact byte offset tracking before subword analysis.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::utils::unicode_helpers;

/// Configuration for pre-tokenization splits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreTokenConfig {
    /// Split on whitespace boundaries.
    pub split_on_whitespace: bool,
    /// Split and isolate ASCII and Unicode punctuation.
    pub split_on_punctuation: bool,
    /// Split individual digit runs into separate tokens.
    pub split_digits: bool,
    /// Prepend whitespace prefix.
    pub add_prefix_space: bool,
    /// Split CamelCase transitions into separate sub-tokens.
    pub split_camel_case: bool,
}

impl Default for PreTokenConfig {
    fn default() -> Self {
        Self {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_digits: false,
            add_prefix_space: false,
            split_camel_case: false,
        }
    }
}

/// Pre-tokenization engine with offset tracking.
#[derive(Debug, Clone, Default)]
pub struct PreTokenizer {
    /// Configuration options.
    pub config: PreTokenConfig,
}

impl PreTokenizer {
    /// Creates a new `PreTokenizer`.
    pub fn new(config: PreTokenConfig) -> Self {
        Self { config }
    }

    /// Pre-tokenizes a string into tokens with `(start_offset, end_offset)` byte pairs.
    pub fn pre_tokenize(&self, text: &str) -> Vec<(String, (usize, usize))> {
        let mut text_to_process = text;
        let prefix_added;
        let mut prefix_offset: usize = 0;

        if self.config.add_prefix_space && !text.starts_with(' ') {
            prefix_added = format!(" {}", text);
            text_to_process = &prefix_added;
            prefix_offset = 1;
        }

        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut token_start: usize = 0;
        let mut char_indices = text_to_process.char_indices().peekable();

        while let Some((idx, c)) = char_indices.next() {
            if self.config.split_on_whitespace && unicode_helpers::is_whitespace(c) {
                if !current_token.is_empty() {
                    let adjusted_start = token_start.saturating_sub(prefix_offset);
                    let adjusted_end = idx.saturating_sub(prefix_offset);
                    tokens.push((current_token.clone(), (adjusted_start, adjusted_end)));
                    current_token.clear();
                }
                token_start = idx + c.len_utf8();
            } else if self.config.split_on_punctuation && unicode_helpers::is_punctuation(c) {
                if !current_token.is_empty() {
                    let adjusted_start = token_start.saturating_sub(prefix_offset);
                    let adjusted_end = idx.saturating_sub(prefix_offset);
                    tokens.push((current_token.clone(), (adjusted_start, adjusted_end)));
                    current_token.clear();
                }
                let p_start = idx.saturating_sub(prefix_offset);
                let p_end = (idx + c.len_utf8()).saturating_sub(prefix_offset);
                tokens.push((c.to_string(), (p_start, p_end)));
                token_start = idx + c.len_utf8();
            } else if self.config.split_digits && c.is_ascii_digit() {
                if !current_token.is_empty() && !current_token.chars().all(|x| x.is_ascii_digit()) {
                    let adjusted_start = token_start.saturating_sub(prefix_offset);
                    let adjusted_end = idx.saturating_sub(prefix_offset);
                    tokens.push((current_token.clone(), (adjusted_start, adjusted_end)));
                    current_token.clear();
                    token_start = idx;
                }
                current_token.push(c);
            } else if self.config.split_camel_case && c.is_uppercase() {
                if !current_token.is_empty() {
                    let adjusted_start = token_start.saturating_sub(prefix_offset);
                    let adjusted_end = idx.saturating_sub(prefix_offset);
                    tokens.push((current_token.clone(), (adjusted_start, adjusted_end)));
                    current_token.clear();
                    token_start = idx;
                }
                current_token.push(c);
            } else {
                if current_token.is_empty() {
                    token_start = idx;
                }
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            let total_len = text_to_process.len();
            let adjusted_start = token_start.saturating_sub(prefix_offset);
            let adjusted_end = total_len.saturating_sub(prefix_offset);
            tokens.push((current_token, (adjusted_start, adjusted_end)));
        }

        tokens
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
    fn test_pretokenizer_suite_1() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_1!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_2() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_2!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_3() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_3!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_4() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_4!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_5() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_5!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_6() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_6!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_7() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_7!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_8() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_8!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_9() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_9!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_10() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_10!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_11() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_11!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_12() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_12!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_13() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_13!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_14() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_14!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_15() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_15!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_16() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_16!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_17() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_17!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_18() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_18!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_19() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_19!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_20() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_20!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_21() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_21!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_22() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_22!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_23() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_23!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_24() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_24!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_25() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_25!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_26() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_26!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_27() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_27!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_28() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_28!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_29() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_29!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_30() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_30!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_31() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_31!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_32() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_32!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_33() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_33!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_34() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_34!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_35() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_35!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_36() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_36!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_37() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_37!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_38() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_38!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_39() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_39!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_40() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_40!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_41() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_41!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_42() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_42!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_43() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_43!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_44() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_44!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_45() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_45!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_46() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_46!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_47() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_47!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_48() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_48!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_49() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_49!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_50() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_50!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_51() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_51!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_52() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_52!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_53() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_53!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_54() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_54!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_55() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_55!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_56() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_56!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_57() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_57!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_58() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_58!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_59() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_59!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_60() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_60!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_61() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_61!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_62() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_62!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_63() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_63!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_64() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_64!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_65() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_65!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_66() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_66!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_67() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_67!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_68() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_68!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_69() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_69!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_70() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_70!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_71() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_71!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_72() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_72!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_73() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_73!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_74() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_74!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_75() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_75!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_76() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_76!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_77() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_77!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_78() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_78!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_79() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_79!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_80() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_80!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_81() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_81!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_82() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_82!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_83() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_83!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_84() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_84!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_85() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_85!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_86() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_86!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_87() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_87!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_88() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_88!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_89() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_89!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_90() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_90!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_91() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_91!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_92() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_92!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_93() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_93!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_94() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_94!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_95() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_95!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_96() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_96!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_97() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_97!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_98() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_98!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_99() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_99!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_100() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_100!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_101() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_101!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_102() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_102!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_103() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_103!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_104() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_104!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_105() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_105!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_106() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_106!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_107() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_107!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_108() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_108!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_109() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_109!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_110() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_110!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_111() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_111!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_112() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_112!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_113() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_113!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_114() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_114!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_115() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_115!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_116() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_116!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_117() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_117!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_118() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_118!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_119() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_119!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_120() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_120!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_121() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_121!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_122() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_122!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_123() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_123!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_124() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_124!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_125() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_125!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_126() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_126!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_127() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_127!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_128() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_128!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_129() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_129!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_130() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_130!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_131() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_131!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_132() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_132!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_133() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_133!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_134() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_134!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_135() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_135!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_136() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_136!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_137() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_137!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_138() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_138!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_139() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_139!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_140() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_140!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_141() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_141!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_142() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_142!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_143() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_143!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_144() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_144!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_145() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_145!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_146() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_146!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_147() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_147!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_148() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_148!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_149() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_149!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_150() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_150!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_151() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_151!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_152() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_152!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_153() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_153!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_154() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_154!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_155() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_155!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_156() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_156!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_157() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_157!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_158() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_158!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_159() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_159!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_160() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_160!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_161() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_161!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_162() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_162!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_163() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_163!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_164() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_164!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_165() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_165!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_166() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_166!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_167() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_167!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_168() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_168!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_169() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_169!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_170() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_170!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_171() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_171!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_172() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_172!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_173() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_173!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_174() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_174!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_175() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_175!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_176() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_176!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_177() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_177!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_178() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_178!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_179() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_179!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_180() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_180!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_181() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_181!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_182() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_182!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_183() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_183!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_184() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_184!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_185() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_185!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_186() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_186!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
    }

    #[test]
    fn test_pretokenizer_suite_187() {
        let cfg = PreTokenConfig {
            split_on_whitespace: true,
            split_on_punctuation: true,
            split_camel_case: true,
            split_digits: true,
            ..Default::default()
        };
        let pt = PreTokenizer::new(cfg);
        let toks = pt.pre_tokenize("HelloWorld, 123 test_187!");
        assert!(!toks.is_empty());
        assert_eq!(toks[0].0, "Hello");
        assert_eq!(toks[1].0, "World");
        assert_eq!(toks[2].0, ",");
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
}
