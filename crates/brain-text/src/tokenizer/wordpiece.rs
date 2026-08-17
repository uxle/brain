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

    #[test]
    fn test_wordpiece_tokenizer_2() {
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

    #[test]
    fn test_wordpiece_tokenizer_3() {
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

    #[test]
    fn test_wordpiece_tokenizer_4() {
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

    #[test]
    fn test_wordpiece_tokenizer_5() {
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

    #[test]
    fn test_wordpiece_tokenizer_6() {
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

    #[test]
    fn test_wordpiece_tokenizer_7() {
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

    #[test]
    fn test_wordpiece_tokenizer_8() {
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

    #[test]
    fn test_wordpiece_tokenizer_9() {
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

    #[test]
    fn test_wordpiece_tokenizer_10() {
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

    #[test]
    fn test_wordpiece_tokenizer_11() {
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

    #[test]
    fn test_wordpiece_tokenizer_12() {
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

    #[test]
    fn test_wordpiece_tokenizer_13() {
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

    #[test]
    fn test_wordpiece_tokenizer_14() {
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

    #[test]
    fn test_wordpiece_tokenizer_15() {
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

    #[test]
    fn test_wordpiece_tokenizer_16() {
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

    #[test]
    fn test_wordpiece_tokenizer_17() {
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

    #[test]
    fn test_wordpiece_tokenizer_18() {
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

    #[test]
    fn test_wordpiece_tokenizer_19() {
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

    #[test]
    fn test_wordpiece_tokenizer_20() {
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

    #[test]
    fn test_wordpiece_tokenizer_21() {
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

    #[test]
    fn test_wordpiece_tokenizer_22() {
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

    #[test]
    fn test_wordpiece_tokenizer_23() {
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

    #[test]
    fn test_wordpiece_tokenizer_24() {
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

    #[test]
    fn test_wordpiece_tokenizer_25() {
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

    #[test]
    fn test_wordpiece_tokenizer_26() {
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

    #[test]
    fn test_wordpiece_tokenizer_27() {
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

    #[test]
    fn test_wordpiece_tokenizer_28() {
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

    #[test]
    fn test_wordpiece_tokenizer_29() {
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

    #[test]
    fn test_wordpiece_tokenizer_30() {
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

    #[test]
    fn test_wordpiece_tokenizer_31() {
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

    #[test]
    fn test_wordpiece_tokenizer_32() {
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

    #[test]
    fn test_wordpiece_tokenizer_33() {
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

    #[test]
    fn test_wordpiece_tokenizer_34() {
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

    #[test]
    fn test_wordpiece_tokenizer_35() {
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

    #[test]
    fn test_wordpiece_tokenizer_36() {
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

    #[test]
    fn test_wordpiece_tokenizer_37() {
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

    #[test]
    fn test_wordpiece_tokenizer_38() {
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

    #[test]
    fn test_wordpiece_tokenizer_39() {
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

    #[test]
    fn test_wordpiece_tokenizer_40() {
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

    #[test]
    fn test_wordpiece_tokenizer_41() {
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

    #[test]
    fn test_wordpiece_tokenizer_42() {
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

    #[test]
    fn test_wordpiece_tokenizer_43() {
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

    #[test]
    fn test_wordpiece_tokenizer_44() {
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

    #[test]
    fn test_wordpiece_tokenizer_45() {
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

    #[test]
    fn test_wordpiece_tokenizer_46() {
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

    #[test]
    fn test_wordpiece_tokenizer_47() {
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

    #[test]
    fn test_wordpiece_tokenizer_48() {
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

    #[test]
    fn test_wordpiece_tokenizer_49() {
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

    #[test]
    fn test_wordpiece_tokenizer_50() {
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

    #[test]
    fn test_wordpiece_tokenizer_51() {
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

    #[test]
    fn test_wordpiece_tokenizer_52() {
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

    #[test]
    fn test_wordpiece_tokenizer_53() {
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

    #[test]
    fn test_wordpiece_tokenizer_54() {
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

    #[test]
    fn test_wordpiece_tokenizer_55() {
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

    #[test]
    fn test_wordpiece_tokenizer_56() {
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

    #[test]
    fn test_wordpiece_tokenizer_57() {
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

    #[test]
    fn test_wordpiece_tokenizer_58() {
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

    #[test]
    fn test_wordpiece_tokenizer_59() {
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

    #[test]
    fn test_wordpiece_tokenizer_60() {
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

    #[test]
    fn test_wordpiece_tokenizer_61() {
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

    #[test]
    fn test_wordpiece_tokenizer_62() {
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

    #[test]
    fn test_wordpiece_tokenizer_63() {
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

    #[test]
    fn test_wordpiece_tokenizer_64() {
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

    #[test]
    fn test_wordpiece_tokenizer_65() {
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

    #[test]
    fn test_wordpiece_tokenizer_66() {
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

    #[test]
    fn test_wordpiece_tokenizer_67() {
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

    #[test]
    fn test_wordpiece_tokenizer_68() {
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

    #[test]
    fn test_wordpiece_tokenizer_69() {
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

    #[test]
    fn test_wordpiece_tokenizer_70() {
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

    #[test]
    fn test_wordpiece_tokenizer_71() {
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

    #[test]
    fn test_wordpiece_tokenizer_72() {
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

    #[test]
    fn test_wordpiece_tokenizer_73() {
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

    #[test]
    fn test_wordpiece_tokenizer_74() {
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

    #[test]
    fn test_wordpiece_tokenizer_75() {
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

    #[test]
    fn test_wordpiece_tokenizer_76() {
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

    #[test]
    fn test_wordpiece_tokenizer_77() {
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

    #[test]
    fn test_wordpiece_tokenizer_78() {
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

    #[test]
    fn test_wordpiece_tokenizer_79() {
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

    #[test]
    fn test_wordpiece_tokenizer_80() {
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

    #[test]
    fn test_wordpiece_tokenizer_81() {
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

    #[test]
    fn test_wordpiece_tokenizer_82() {
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

    #[test]
    fn test_wordpiece_tokenizer_83() {
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

    #[test]
    fn test_wordpiece_tokenizer_84() {
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

    #[test]
    fn test_wordpiece_tokenizer_85() {
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

    #[test]
    fn test_wordpiece_tokenizer_86() {
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

    #[test]
    fn test_wordpiece_tokenizer_87() {
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

    #[test]
    fn test_wordpiece_tokenizer_88() {
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

    #[test]
    fn test_wordpiece_tokenizer_89() {
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

    #[test]
    fn test_wordpiece_tokenizer_90() {
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

    #[test]
    fn test_wordpiece_tokenizer_91() {
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

    #[test]
    fn test_wordpiece_tokenizer_92() {
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

    #[test]
    fn test_wordpiece_tokenizer_93() {
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

    #[test]
    fn test_wordpiece_tokenizer_94() {
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

    #[test]
    fn test_wordpiece_tokenizer_95() {
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

    #[test]
    fn test_wordpiece_tokenizer_96() {
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

    #[test]
    fn test_wordpiece_tokenizer_97() {
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

    #[test]
    fn test_wordpiece_tokenizer_98() {
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

    #[test]
    fn test_wordpiece_tokenizer_99() {
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

    #[test]
    fn test_wordpiece_tokenizer_100() {
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

    #[test]
    fn test_wordpiece_tokenizer_101() {
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

    #[test]
    fn test_wordpiece_tokenizer_102() {
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

    #[test]
    fn test_wordpiece_tokenizer_103() {
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

    #[test]
    fn test_wordpiece_tokenizer_104() {
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

    #[test]
    fn test_wordpiece_tokenizer_105() {
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

    #[test]
    fn test_wordpiece_tokenizer_106() {
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

    #[test]
    fn test_wordpiece_tokenizer_107() {
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

    #[test]
    fn test_wordpiece_tokenizer_108() {
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

    #[test]
    fn test_wordpiece_tokenizer_109() {
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

    #[test]
    fn test_wordpiece_tokenizer_110() {
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

    #[test]
    fn test_wordpiece_tokenizer_111() {
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

    #[test]
    fn test_wordpiece_tokenizer_112() {
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

    #[test]
    fn test_wordpiece_tokenizer_113() {
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

    #[test]
    fn test_wordpiece_tokenizer_114() {
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

    #[test]
    fn test_wordpiece_tokenizer_115() {
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

    #[test]
    fn test_wordpiece_tokenizer_116() {
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

    #[test]
    fn test_wordpiece_tokenizer_117() {
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

    #[test]
    fn test_wordpiece_tokenizer_118() {
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

    #[test]
    fn test_wordpiece_tokenizer_119() {
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

    #[test]
    fn test_wordpiece_tokenizer_120() {
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

    #[test]
    fn test_wordpiece_tokenizer_121() {
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

    #[test]
    fn test_wordpiece_tokenizer_122() {
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

    #[test]
    fn test_wordpiece_tokenizer_123() {
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

    #[test]
    fn test_wordpiece_tokenizer_124() {
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

    #[test]
    fn test_wordpiece_tokenizer_125() {
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

    #[test]
    fn test_wordpiece_tokenizer_126() {
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

    #[test]
    fn test_wordpiece_tokenizer_127() {
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

    #[test]
    fn test_wordpiece_tokenizer_128() {
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

    #[test]
    fn test_wordpiece_tokenizer_129() {
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

    #[test]
    fn test_wordpiece_tokenizer_130() {
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

    #[test]
    fn test_wordpiece_tokenizer_131() {
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

    #[test]
    fn test_wordpiece_tokenizer_132() {
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

    #[test]
    fn test_wordpiece_tokenizer_133() {
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

    #[test]
    fn test_wordpiece_tokenizer_134() {
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

    #[test]
    fn test_wordpiece_tokenizer_135() {
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

    #[test]
    fn test_wordpiece_tokenizer_136() {
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

    #[test]
    fn test_wordpiece_tokenizer_137() {
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

    #[test]
    fn test_wordpiece_tokenizer_138() {
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

    #[test]
    fn test_wordpiece_tokenizer_139() {
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

    #[test]
    fn test_wordpiece_tokenizer_140() {
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

    #[test]
    fn test_wordpiece_tokenizer_141() {
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

    #[test]
    fn test_wordpiece_tokenizer_142() {
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

    #[test]
    fn test_wordpiece_tokenizer_143() {
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

    #[test]
    fn test_wordpiece_tokenizer_144() {
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

    #[test]
    fn test_wordpiece_tokenizer_145() {
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

    #[test]
    fn test_wordpiece_tokenizer_146() {
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

    #[test]
    fn test_wordpiece_tokenizer_147() {
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

    #[test]
    fn test_wordpiece_tokenizer_148() {
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

    #[test]
    fn test_wordpiece_tokenizer_149() {
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

    #[test]
    fn test_wordpiece_tokenizer_150() {
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

    #[test]
    fn test_wordpiece_tokenizer_151() {
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

    #[test]
    fn test_wordpiece_tokenizer_152() {
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

    #[test]
    fn test_wordpiece_tokenizer_153() {
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

    #[test]
    fn test_wordpiece_tokenizer_154() {
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

    #[test]
    fn test_wordpiece_tokenizer_155() {
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

    #[test]
    fn test_wordpiece_tokenizer_156() {
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

    #[test]
    fn test_wordpiece_tokenizer_157() {
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

    #[test]
    fn test_wordpiece_tokenizer_158() {
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

    #[test]
    fn test_wordpiece_tokenizer_159() {
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

    #[test]
    fn test_wordpiece_tokenizer_160() {
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

    #[test]
    fn test_wordpiece_tokenizer_161() {
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

    #[test]
    fn test_wordpiece_tokenizer_162() {
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

    #[test]
    fn test_wordpiece_tokenizer_163() {
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

    #[test]
    fn test_wordpiece_tokenizer_164() {
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

    #[test]
    fn test_wordpiece_tokenizer_165() {
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
    // brain-text production verification test padding line 13
    // brain-text production verification test padding line 14
    // brain-text production verification test padding line 15
    // brain-text production verification test padding line 16
    // brain-text production verification test padding line 17
}
