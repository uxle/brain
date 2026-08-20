//! # Pre-Tokenizers: Whitespace, Punctuation, Digits, and CamelCase
//!
//! Splitting raw text into pre-token spans with exact byte offset tracking before subword analysis.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision,
        clippy::float_cmp,
        clippy::len_zero
    )]
    use super::*;
    use crate::analyze::*;
    use crate::builder::*;
    use crate::compute::*;
    use crate::config::*;
    use crate::core::*;
    use crate::embedding::fasttext::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::*;
    use crate::features::*;
    use crate::helper::*;
    use crate::lm::*;
    use crate::ops::*;
    use crate::optimize::*;
    use crate::process::*;
    use crate::similarity::*;
    use crate::text_ops::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::post::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::*;
    use crate::transform::*;
    use crate::utils::*;
    use crate::vocab::*;
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
}
