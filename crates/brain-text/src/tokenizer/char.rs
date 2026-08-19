//! # Character & Word Level Tokenizers
//!
//! Pure character-level code point tokenizers and whitespace/punctuation word tokenizers.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult, TokenId, TokenizedOutput};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;

/// Configuration for character tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharConfig {
    /// Convert to lowercase.
    pub lowercase: bool,
    /// Unknown token string.
    pub unk_token: String,
}

impl Default for CharConfig {
    fn default() -> Self {
        Self {
            lowercase: false,
            unk_token: "<unk>".to_string(),
        }
    }
}

/// Pure character-level tokenizer.
#[derive(Debug, Clone, Default)]
pub struct CharTokenizer {
    /// Vocabulary of characters.
    pub vocab: Vocab,
    /// Configuration options.
    pub config: CharConfig,
}

impl CharTokenizer {
    /// Creates a new character tokenizer.
    pub fn new(vocab: Vocab, config: CharConfig) -> Self {
        Self { vocab, config }
    }
}

impl Tokenizer for CharTokenizer {
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
        Ok(text)
    }

    fn tokenize(&self, text: &str) -> TextResult<Vec<String>> {
        let processed = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        Ok(processed.chars().map(|c| c.to_string()).collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
}

/// Configuration for word tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordConfig {
    /// Convert to lowercase.
    pub lowercase: bool,
    /// Fallback unknown token.
    pub unk_token: String,
}

impl Default for WordConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            unk_token: "[UNK]".to_string(),
        }
    }
}

/// Word-level tokenizer splitting on whitespace and punctuation.
#[derive(Debug, Clone, Default)]
pub struct WordTokenizer {
    /// Vocabulary of words.
    pub vocab: Vocab,
    /// Configuration options.
    pub config: WordConfig,
}

impl WordTokenizer {
    /// Creates a new word tokenizer.
    pub fn new(vocab: Vocab, config: WordConfig) -> Self {
        Self { vocab, config }
    }
}

impl Tokenizer for WordTokenizer {
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
        let mut words = Vec::new();
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                words.push(token.to_string());
            } else {
                words.push(self.config.unk_token.clone());
            }
        }
        Ok(words.join(" "))
    }

    fn tokenize(&self, text: &str) -> TextResult<Vec<String>> {
        let processed = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let mut tokens = Vec::new();
        let mut cur = String::new();

        for c in processed.chars() {
            if c.is_whitespace() {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
            } else if c.is_ascii_punctuation() {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
                tokens.push(c.to_string());
            } else {
                cur.push(c);
            }
        }

        if !cur.is_empty() {
            tokens.push(cur);
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
    fn test_char_and_word_tokenizers_1() {
        let mut char_v = Vocab::new();
        char_v.insert("a");
        char_v.insert("b");
        char_v.insert("c");
        let ct = CharTokenizer::new(char_v, CharConfig::default());
        let c_toks = ct.tokenize("abc").unwrap();
        assert_eq!(c_toks, vec!["a", "b", "c"]);

        let mut word_v = Vocab::new();
        word_v.insert("hello");
        word_v.insert("world");
        word_v.insert("!");
        let wt = WordTokenizer::new(word_v, WordConfig::default());
        let w_toks = wt.tokenize("Hello, World!").unwrap();
        assert_eq!(w_toks, vec!["hello", ",", "world", "!"]);
    }
}
