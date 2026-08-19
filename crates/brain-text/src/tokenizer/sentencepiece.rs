//! # SentencePiece Unigram Language Model Tokenizer
//!
//! Subword segmentation using optimal Viterbi dynamic programming path search over piece lattices.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult, TokenId, TokenizedOutput};
use crate::tokenizer::Tokenizer;
use crate::vocab::{SpecialKind, Vocab};
use std::collections::HashMap;

/// Configuration for SentencePiece Unigram Tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct SpConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Unknown token string.
    pub unk_token: String,
    /// Beginning-of-sentence token string.
    pub bos_token: Option<String>,
    /// End-of-sentence token string.
    pub eos_token: Option<String>,
    /// Replacement whitespace character (e.g. ` ` / U+2581).
    pub replacement_char: char,
    /// Prepend a dummy prefix replacement character to inputs.
    pub add_dummy_prefix: bool,
}

impl Default for SpConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            unk_token: "<unk>".to_string(),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            replacement_char: ' ',
            add_dummy_prefix: true,
        }
    }
}

/// SentencePiece Unigram Language Model Tokenizer.
#[derive(Debug, Clone, Default)]
pub struct SentencePieceTokenizer {
    /// Vocabulary table.
    pub vocab: Vocab,
    /// Log-probability scores for pieces: `token -> (id, log_prob)`.
    pub piece_scores: HashMap<String, (TokenId, f64)>,
    /// Configuration settings.
    pub config: SpConfig,
}

impl SentencePieceTokenizer {
    /// Creates a SentencePiece tokenizer from vocabulary and score map.
    pub fn from_pieces(
        vocab: Vocab,
        piece_scores: HashMap<String, f64>,
        config: SpConfig,
    ) -> Self {
        let mut scores = HashMap::with_capacity(piece_scores.len());
        for (token, score) in piece_scores {
            if let Some(id) = vocab.get_id(&token) {
                scores.insert(token, (id, score));
            }
        }
        Self {
            vocab,
            piece_scores: scores,
            config,
        }
    }

    /// Pre-processes text by converting whitespace to replacement character.
    pub fn preprocess(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 1);
        if self.config.add_dummy_prefix {
            result.push(self.config.replacement_char);
        }
        for c in text.chars() {
            if c.is_whitespace() {
                result.push(self.config.replacement_char);
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Finds the highest-probability segmentation using the Viterbi algorithm.
    pub fn viterbi_decode(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        let unk_score = -100.0;
        let mut best_scores = vec![f64::NEG_INFINITY; n + 1];
        let mut best_edges: Vec<Option<(usize, String)>> = vec![None; n + 1];
        best_scores[0] = 0.0;

        for i in 0..n {
            if best_scores[i] == f64::NEG_INFINITY {
                continue;
            }

            let mut sub = String::new();
            let max_sub_len = 32.min(n - i);

            for j in 1..=max_sub_len {
                sub.push(chars[i + j - 1]);
                let score = if let Some(&(_, s)) = self.piece_scores.get(&sub) {
                    s
                } else if j == 1 {
                    unk_score
                } else {
                    continue;
                };

                let new_score = best_scores[i] + score;
                if new_score > best_scores[i + j] {
                    best_scores[i + j] = new_score;
                    best_edges[i + j] = Some((i, sub.clone()));
                }
            }
        }

        // Backtrack path
        let mut pieces = Vec::new();
        let mut curr = n;

        while curr > 0 {
            if let Some((prev, ref piece)) = best_edges[curr] {
                pieces.push(piece.clone());
                curr = prev;
            } else {
                // Fallback 1 character
                let c = chars[curr - 1].to_string();
                pieces.push(c);
                curr -= 1;
            }
        }

        pieces.reverse();
        pieces
    }
}

impl Tokenizer for SentencePieceTokenizer {
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
                text.push_str(token);
            } else {
                text.push_str(&self.config.unk_token);
            }
        }
        let replaced = text.replace(self.config.replacement_char, " ");
        Ok(replaced.trim().to_string())
    }

    fn tokenize(&self, text: &str) -> TextResult<Vec<String>> {
        let prep = self.preprocess(text);
        Ok(self.viterbi_decode(&prep))
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
    fn test_sentencepiece_tokenizer_1() {
        let mut vocab = Vocab::new();
        vocab.add_special("<unk>", SpecialKind::Unk);
        vocab.insert(" ");
        vocab.insert(" h");
        vocab.insert("e");
        vocab.insert("l");
        vocab.insert("lo");
        vocab.insert(" hello");

        let mut scores = HashMap::new();
        scores.insert("<unk>".to_string(), -10.0);
        scores.insert(" ".to_string(), -1.0);
        scores.insert(" h".to_string(), -2.0);
        scores.insert("e".to_string(), -3.0);
        scores.insert("l".to_string(), -3.0);
        scores.insert("lo".to_string(), -2.0);
        scores.insert(" hello".to_string(), -0.5);

        let sp = SentencePieceTokenizer::from_pieces(vocab, scores, SpConfig::default());
        let pieces = sp.tokenize("hello").unwrap();
        assert_eq!(pieces, vec![" hello"]);

        let out = sp.encode("hello").unwrap();
        assert!(!out.ids.is_empty());

        let decoded = sp.decode(&out.ids).unwrap();
        assert_eq!(decoded, "hello");
    }
}
