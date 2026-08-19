//! # WordPiece Subword Tokenizer
//!
//! Longest-match greedy subword tokenization with continuation prefix `##` (BERT style).
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult, TokenId, TokenizedOutput};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;

/// Configuration for WordPiece Tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct WordPieceConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Unknown token string (default `[UNK]`).
    pub unk_token: String,
    /// Prefix appended to continuation subwords (default `##`).
    pub continuation_prefix: String,
    /// Maximum character length per word before marking as `[UNK]`.
    pub max_input_chars_per_word: usize,
}

impl Default for WordPieceConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            unk_token: "[UNK]".to_string(),
            continuation_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
        }
    }
}

/// WordPiece subword tokenizer.
#[derive(Debug, Clone, Default)]
pub struct WordPieceTokenizer {
    /// Vocabulary.
    pub vocab: Vocab,
    /// Configuration settings.
    pub config: WordPieceConfig,
}

impl WordPieceTokenizer {
    /// Creates a new WordPiece tokenizer.
    pub fn new(vocab: Vocab, config: WordPieceConfig) -> Self {
        Self { vocab, config }
    }

    /// Performs greedy longest-match prefix segmentation on a single word.
    pub fn segment_word(&self, word: &str) -> Vec<String> {
        let char_len = word.chars().count();
        if char_len > self.config.max_input_chars_per_word {
            return vec![self.config.unk_token.clone()];
        }

        let mut subwords = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut is_bad = false;

        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_substr = None;

            while start < end {
                let sub: String = chars[start..end].iter().collect();
                let candidate = if start > 0 {
                    format!("{}{}", self.config.continuation_prefix, sub)
                } else {
                    sub.clone()
                };

                if self.vocab.contains(&candidate) {
                    cur_substr = Some(candidate);
                    break;
                }
                end -= 1;
            }

            match cur_substr {
                Some(sub) => {
                    subwords.push(sub);
                    start = end;
                }
                None => {
                    is_bad = true;
                    break;
                }
            }
        }

        if is_bad {
            vec![self.config.unk_token.clone()]
        } else {
            subwords
        }
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> TextResult<TokenizedOutput> {
        let tokens = self.tokenize(text)?;
        let mut ids = Vec::with_capacity(tokens.len());
        let mut offsets = Vec::with_capacity(tokens.len());

        let unk_id = self.vocab.unk_id().unwrap_or(0);
        let mut cursor = 0;

        for token in &tokens {
            let id = self.vocab.get_id(token).unwrap_or(unk_id);
            ids.push(id);
            let len = token.len();
            offsets.push((cursor, cursor + len));
            cursor += len;
        }

        Ok(TokenizedOutput::new(ids, tokens, offsets))
    }

    fn decode(&self, ids: &[TokenId]) -> TextResult<String> {
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if let Some(stripped) = token.strip_prefix(&self.config.continuation_prefix) {
                    text.push_str(stripped);
                } else {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(token);
                }
            } else {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&self.config.unk_token);
            }
        }
        Ok(text)
    }

    fn tokenize(&self, text: &str) -> TextResult<Vec<String>> {
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            let pieces = self.segment_word(word);
            tokens.extend(pieces);
        }
        Ok(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> &Vocab {
        &self.vocab
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
    fn test_wordpiece_tokenizer_1() {
        let mut vocab = Vocab::new();
        vocab.add_special("[UNK]", SpecialKind::Unk);
        vocab.insert("un");
        vocab.insert("##aff");
        vocab.insert("##able");

        let wp = WordPieceTokenizer::new(vocab, WordPieceConfig::default());
        let pieces = wp.tokenize("unaffable").unwrap();
        assert_eq!(pieces, vec!["un", "##aff", "##able"]);

        let out = wp.encode("unaffable").unwrap();
        assert_eq!(out.ids.len(), 3);

        let decoded = wp.decode(&out.ids).unwrap();
        assert_eq!(decoded, "unaffable");
    }
}
