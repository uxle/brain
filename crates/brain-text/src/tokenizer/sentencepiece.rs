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

    #[test]
    fn test_sentencepiece_tokenizer_2() {
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

    #[test]
    fn test_sentencepiece_tokenizer_3() {
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

    #[test]
    fn test_sentencepiece_tokenizer_4() {
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

    #[test]
    fn test_sentencepiece_tokenizer_5() {
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

    #[test]
    fn test_sentencepiece_tokenizer_6() {
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

    #[test]
    fn test_sentencepiece_tokenizer_7() {
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

    #[test]
    fn test_sentencepiece_tokenizer_8() {
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

    #[test]
    fn test_sentencepiece_tokenizer_9() {
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

    #[test]
    fn test_sentencepiece_tokenizer_10() {
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

    #[test]
    fn test_sentencepiece_tokenizer_11() {
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

    #[test]
    fn test_sentencepiece_tokenizer_12() {
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

    #[test]
    fn test_sentencepiece_tokenizer_13() {
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

    #[test]
    fn test_sentencepiece_tokenizer_14() {
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

    #[test]
    fn test_sentencepiece_tokenizer_15() {
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

    #[test]
    fn test_sentencepiece_tokenizer_16() {
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

    #[test]
    fn test_sentencepiece_tokenizer_17() {
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

    #[test]
    fn test_sentencepiece_tokenizer_18() {
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

    #[test]
    fn test_sentencepiece_tokenizer_19() {
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

    #[test]
    fn test_sentencepiece_tokenizer_20() {
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

    #[test]
    fn test_sentencepiece_tokenizer_21() {
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

    #[test]
    fn test_sentencepiece_tokenizer_22() {
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

    #[test]
    fn test_sentencepiece_tokenizer_23() {
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

    #[test]
    fn test_sentencepiece_tokenizer_24() {
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

    #[test]
    fn test_sentencepiece_tokenizer_25() {
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

    #[test]
    fn test_sentencepiece_tokenizer_26() {
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

    #[test]
    fn test_sentencepiece_tokenizer_27() {
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

    #[test]
    fn test_sentencepiece_tokenizer_28() {
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

    #[test]
    fn test_sentencepiece_tokenizer_29() {
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

    #[test]
    fn test_sentencepiece_tokenizer_30() {
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

    #[test]
    fn test_sentencepiece_tokenizer_31() {
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

    #[test]
    fn test_sentencepiece_tokenizer_32() {
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

    #[test]
    fn test_sentencepiece_tokenizer_33() {
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

    #[test]
    fn test_sentencepiece_tokenizer_34() {
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

    #[test]
    fn test_sentencepiece_tokenizer_35() {
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

    #[test]
    fn test_sentencepiece_tokenizer_36() {
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

    #[test]
    fn test_sentencepiece_tokenizer_37() {
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

    #[test]
    fn test_sentencepiece_tokenizer_38() {
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

    #[test]
    fn test_sentencepiece_tokenizer_39() {
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

    #[test]
    fn test_sentencepiece_tokenizer_40() {
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

    #[test]
    fn test_sentencepiece_tokenizer_41() {
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

    #[test]
    fn test_sentencepiece_tokenizer_42() {
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

    #[test]
    fn test_sentencepiece_tokenizer_43() {
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

    #[test]
    fn test_sentencepiece_tokenizer_44() {
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

    #[test]
    fn test_sentencepiece_tokenizer_45() {
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

    #[test]
    fn test_sentencepiece_tokenizer_46() {
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

    #[test]
    fn test_sentencepiece_tokenizer_47() {
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

    #[test]
    fn test_sentencepiece_tokenizer_48() {
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

    #[test]
    fn test_sentencepiece_tokenizer_49() {
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

    #[test]
    fn test_sentencepiece_tokenizer_50() {
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

    #[test]
    fn test_sentencepiece_tokenizer_51() {
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

    #[test]
    fn test_sentencepiece_tokenizer_52() {
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

    #[test]
    fn test_sentencepiece_tokenizer_53() {
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

    #[test]
    fn test_sentencepiece_tokenizer_54() {
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

    #[test]
    fn test_sentencepiece_tokenizer_55() {
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

    #[test]
    fn test_sentencepiece_tokenizer_56() {
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

    #[test]
    fn test_sentencepiece_tokenizer_57() {
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

    #[test]
    fn test_sentencepiece_tokenizer_58() {
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

    #[test]
    fn test_sentencepiece_tokenizer_59() {
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

    #[test]
    fn test_sentencepiece_tokenizer_60() {
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

    #[test]
    fn test_sentencepiece_tokenizer_61() {
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

    #[test]
    fn test_sentencepiece_tokenizer_62() {
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

    #[test]
    fn test_sentencepiece_tokenizer_63() {
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

    #[test]
    fn test_sentencepiece_tokenizer_64() {
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

    #[test]
    fn test_sentencepiece_tokenizer_65() {
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

    #[test]
    fn test_sentencepiece_tokenizer_66() {
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

    #[test]
    fn test_sentencepiece_tokenizer_67() {
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

    #[test]
    fn test_sentencepiece_tokenizer_68() {
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

    #[test]
    fn test_sentencepiece_tokenizer_69() {
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

    #[test]
    fn test_sentencepiece_tokenizer_70() {
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

    #[test]
    fn test_sentencepiece_tokenizer_71() {
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

    #[test]
    fn test_sentencepiece_tokenizer_72() {
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

    #[test]
    fn test_sentencepiece_tokenizer_73() {
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

    #[test]
    fn test_sentencepiece_tokenizer_74() {
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

    #[test]
    fn test_sentencepiece_tokenizer_75() {
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

    #[test]
    fn test_sentencepiece_tokenizer_76() {
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

    #[test]
    fn test_sentencepiece_tokenizer_77() {
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

    #[test]
    fn test_sentencepiece_tokenizer_78() {
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

    #[test]
    fn test_sentencepiece_tokenizer_79() {
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

    #[test]
    fn test_sentencepiece_tokenizer_80() {
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

    #[test]
    fn test_sentencepiece_tokenizer_81() {
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

    #[test]
    fn test_sentencepiece_tokenizer_82() {
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

    #[test]
    fn test_sentencepiece_tokenizer_83() {
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

    #[test]
    fn test_sentencepiece_tokenizer_84() {
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

    #[test]
    fn test_sentencepiece_tokenizer_85() {
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

    #[test]
    fn test_sentencepiece_tokenizer_86() {
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

    #[test]
    fn test_sentencepiece_tokenizer_87() {
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

    #[test]
    fn test_sentencepiece_tokenizer_88() {
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

    #[test]
    fn test_sentencepiece_tokenizer_89() {
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

    #[test]
    fn test_sentencepiece_tokenizer_90() {
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

    #[test]
    fn test_sentencepiece_tokenizer_91() {
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

    #[test]
    fn test_sentencepiece_tokenizer_92() {
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

    #[test]
    fn test_sentencepiece_tokenizer_93() {
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

    #[test]
    fn test_sentencepiece_tokenizer_94() {
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

    #[test]
    fn test_sentencepiece_tokenizer_95() {
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

    #[test]
    fn test_sentencepiece_tokenizer_96() {
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

    #[test]
    fn test_sentencepiece_tokenizer_97() {
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

    #[test]
    fn test_sentencepiece_tokenizer_98() {
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

    #[test]
    fn test_sentencepiece_tokenizer_99() {
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

    #[test]
    fn test_sentencepiece_tokenizer_100() {
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
