//! # Byte-Pair Encoding (BPE) Tokenizer
//!
//! Subword tokenization using iterative pair merge hierarchies (GPT-2 & RoBERTa styles).
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult, TokenId, TokenizedOutput};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;
use std::collections::HashMap;

/// Configuration for BPE Tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct BpeConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Minimum frequency for candidate merges.
    pub min_frequency: usize,
    /// Whether to operate in byte-level space.
    pub byte_level: bool,
    /// Subword dropout probability for stochastic tokenization (BPE-dropout).
    pub dropout: Option<f32>,
    /// Fallback unknown token.
    pub unk_token: String,
    /// Suffix appended to end-of-word pieces (e.g. `</w>`).
    pub end_of_word_suffix: Option<String>,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            min_frequency: 2,
            byte_level: false,
            dropout: None,
            unk_token: "<unk>".to_string(),
            end_of_word_suffix: Some("</w>".to_string()),
        }
    }
}

/// Byte-Pair Encoding Tokenizer.
#[derive(Debug, Clone, Default)]
pub struct BpeTokenizer {
    /// Vocabulary of tokens.
    pub vocab: Vocab,
    /// Merge rules mapping `(piece_a, piece_b)` to its merge rank (lower is earlier).
    pub merges: HashMap<(String, String), usize>,
    /// Tokenizer configuration.
    pub config: BpeConfig,
}

impl BpeTokenizer {
    /// Creates a new BPE Tokenizer from a vocabulary and ordered merge list.
    pub fn from_vocab_and_merges(
        vocab: Vocab,
        merges: Vec<((String, String), usize)>,
        config: BpeConfig,
    ) -> Self {
        let merges_map: HashMap<(String, String), usize> = merges.into_iter().collect();
        Self {
            vocab,
            merges: merges_map,
            config,
        }
    }

    /// Performs BPE subword segmentation on a single pre-tokenized word string.
    pub fn bpe_segment_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(ref suffix) = self.config.end_of_word_suffix {
            if let Some(last) = symbols.last_mut() {
                last.push_str(suffix);
            }
        }

        if symbols.len() <= 1 {
            return symbols;
        }

        loop {
            let mut min_rank = usize::MAX;
            let mut best_pair_idx = None;

            for i in 0..(symbols.len() - 1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair_idx = Some(i);
                    }
                }
            }

            match best_pair_idx {
                Some(idx) => {
                    let first = symbols[idx].clone();
                    let second = symbols[idx + 1].clone();
                    let merged = format!("{}{}", first, second);

                    let mut new_symbols = Vec::with_capacity(symbols.len() - 1);
                    let mut i = 0;
                    while i < symbols.len() {
                        if i == idx {
                            new_symbols.push(merged.clone());
                            i += 2;
                        } else {
                            new_symbols.push(symbols[i].clone());
                            i += 1;
                        }
                    }
                    symbols = new_symbols;
                    if symbols.len() <= 1 {
                        break;
                    }
                }
                None => break,
            }
        }

        symbols
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> TextResult<TokenizedOutput> {
        let tokens = self.tokenize(text)?;
        let mut ids = Vec::with_capacity(tokens.len());
        let mut offsets = Vec::with_capacity(tokens.len());

        let unk_id = self.vocab.unk_id().unwrap_or(0);
        let mut cursor = 0;

        for token in &tokens {
            let clean_token = if let Some(ref suffix) = self.config.end_of_word_suffix {
                token.strip_suffix(suffix).unwrap_or(token)
            } else {
                token.as_str()
            };

            let id = self.vocab.get_id(token).unwrap_or(unk_id);
            ids.push(id);

            let token_len = clean_token.len();
            offsets.push((cursor, cursor + token_len));
            cursor += token_len;
        }

        Ok(TokenizedOutput::new(ids, tokens, offsets))
    }

    fn decode(&self, ids: &[TokenId]) -> TextResult<String> {
        let mut result = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if let Some(ref suffix) = self.config.end_of_word_suffix {
                    if let Some(stripped) = token.strip_suffix(suffix) {
                        result.push_str(stripped);
                        result.push(' ');
                    } else {
                        result.push_str(token);
                    }
                } else {
                    result.push_str(token);
                }
            } else {
                result.push_str(&self.config.unk_token);
            }
        }
        Ok(result.trim_end().to_string())
    }

    fn tokenize(&self, text: &str) -> TextResult<Vec<String>> {
        let words = text.split_whitespace();
        let mut tokens = Vec::new();
        for word in words {
            let pieces = self.bpe_segment_word(word);
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
    fn test_bpe_tokenizer_1() {
        let mut vocab = Vocab::new();
        let unk = vocab.add_special("<unk>", SpecialKind::Unk);
        vocab.insert("h");
        vocab.insert("e");
        vocab.insert("l");
        vocab.insert("o</w>");
        vocab.insert("he");
        vocab.insert("ll");
        vocab.insert("hello</w>");

        let mut merges = Vec::new();
        merges.push((("h".to_string(), "e".to_string()), 0));
        merges.push((("l".to_string(), "l".to_string()), 1));
        merges.push((("he".to_string(), "ll".to_string()), 2));
        merges.push((("hell".to_string(), "o</w>".to_string()), 3));

        let bpe = BpeTokenizer::from_vocab_and_merges(vocab, merges, BpeConfig::default());
        let pieces = bpe.tokenize("hello").unwrap();
        assert!(!pieces.is_empty());

        let out = bpe.encode("hello").unwrap();
        assert!(!out.ids.is_empty());

        let decoded = bpe.decode(&out.ids).unwrap();
        assert_eq!(decoded, "hello");
    }
}
