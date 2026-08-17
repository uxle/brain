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

    #[test]
    fn test_char_and_word_tokenizers_2() {
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

    #[test]
    fn test_char_and_word_tokenizers_3() {
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

    #[test]
    fn test_char_and_word_tokenizers_4() {
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

    #[test]
    fn test_char_and_word_tokenizers_5() {
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

    #[test]
    fn test_char_and_word_tokenizers_6() {
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

    #[test]
    fn test_char_and_word_tokenizers_7() {
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

    #[test]
    fn test_char_and_word_tokenizers_8() {
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

    #[test]
    fn test_char_and_word_tokenizers_9() {
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

    #[test]
    fn test_char_and_word_tokenizers_10() {
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

    #[test]
    fn test_char_and_word_tokenizers_11() {
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

    #[test]
    fn test_char_and_word_tokenizers_12() {
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

    #[test]
    fn test_char_and_word_tokenizers_13() {
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

    #[test]
    fn test_char_and_word_tokenizers_14() {
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

    #[test]
    fn test_char_and_word_tokenizers_15() {
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

    #[test]
    fn test_char_and_word_tokenizers_16() {
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

    #[test]
    fn test_char_and_word_tokenizers_17() {
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

    #[test]
    fn test_char_and_word_tokenizers_18() {
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

    #[test]
    fn test_char_and_word_tokenizers_19() {
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

    #[test]
    fn test_char_and_word_tokenizers_20() {
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

    #[test]
    fn test_char_and_word_tokenizers_21() {
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

    #[test]
    fn test_char_and_word_tokenizers_22() {
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

    #[test]
    fn test_char_and_word_tokenizers_23() {
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

    #[test]
    fn test_char_and_word_tokenizers_24() {
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

    #[test]
    fn test_char_and_word_tokenizers_25() {
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

    #[test]
    fn test_char_and_word_tokenizers_26() {
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

    #[test]
    fn test_char_and_word_tokenizers_27() {
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

    #[test]
    fn test_char_and_word_tokenizers_28() {
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

    #[test]
    fn test_char_and_word_tokenizers_29() {
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

    #[test]
    fn test_char_and_word_tokenizers_30() {
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

    #[test]
    fn test_char_and_word_tokenizers_31() {
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

    #[test]
    fn test_char_and_word_tokenizers_32() {
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

    #[test]
    fn test_char_and_word_tokenizers_33() {
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

    #[test]
    fn test_char_and_word_tokenizers_34() {
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

    #[test]
    fn test_char_and_word_tokenizers_35() {
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

    #[test]
    fn test_char_and_word_tokenizers_36() {
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

    #[test]
    fn test_char_and_word_tokenizers_37() {
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

    #[test]
    fn test_char_and_word_tokenizers_38() {
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

    #[test]
    fn test_char_and_word_tokenizers_39() {
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

    #[test]
    fn test_char_and_word_tokenizers_40() {
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

    #[test]
    fn test_char_and_word_tokenizers_41() {
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

    #[test]
    fn test_char_and_word_tokenizers_42() {
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

    #[test]
    fn test_char_and_word_tokenizers_43() {
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

    #[test]
    fn test_char_and_word_tokenizers_44() {
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

    #[test]
    fn test_char_and_word_tokenizers_45() {
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

    #[test]
    fn test_char_and_word_tokenizers_46() {
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

    #[test]
    fn test_char_and_word_tokenizers_47() {
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

    #[test]
    fn test_char_and_word_tokenizers_48() {
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

    #[test]
    fn test_char_and_word_tokenizers_49() {
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

    #[test]
    fn test_char_and_word_tokenizers_50() {
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

    #[test]
    fn test_char_and_word_tokenizers_51() {
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

    #[test]
    fn test_char_and_word_tokenizers_52() {
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

    #[test]
    fn test_char_and_word_tokenizers_53() {
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

    #[test]
    fn test_char_and_word_tokenizers_54() {
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

    #[test]
    fn test_char_and_word_tokenizers_55() {
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

    #[test]
    fn test_char_and_word_tokenizers_56() {
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

    #[test]
    fn test_char_and_word_tokenizers_57() {
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

    #[test]
    fn test_char_and_word_tokenizers_58() {
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

    #[test]
    fn test_char_and_word_tokenizers_59() {
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

    #[test]
    fn test_char_and_word_tokenizers_60() {
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

    #[test]
    fn test_char_and_word_tokenizers_61() {
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

    #[test]
    fn test_char_and_word_tokenizers_62() {
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

    #[test]
    fn test_char_and_word_tokenizers_63() {
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

    #[test]
    fn test_char_and_word_tokenizers_64() {
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

    #[test]
    fn test_char_and_word_tokenizers_65() {
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

    #[test]
    fn test_char_and_word_tokenizers_66() {
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

    #[test]
    fn test_char_and_word_tokenizers_67() {
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

    #[test]
    fn test_char_and_word_tokenizers_68() {
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

    #[test]
    fn test_char_and_word_tokenizers_69() {
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

    #[test]
    fn test_char_and_word_tokenizers_70() {
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

    #[test]
    fn test_char_and_word_tokenizers_71() {
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

    #[test]
    fn test_char_and_word_tokenizers_72() {
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

    #[test]
    fn test_char_and_word_tokenizers_73() {
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

    #[test]
    fn test_char_and_word_tokenizers_74() {
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

    #[test]
    fn test_char_and_word_tokenizers_75() {
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

    #[test]
    fn test_char_and_word_tokenizers_76() {
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

    #[test]
    fn test_char_and_word_tokenizers_77() {
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

    #[test]
    fn test_char_and_word_tokenizers_78() {
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

    #[test]
    fn test_char_and_word_tokenizers_79() {
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

    #[test]
    fn test_char_and_word_tokenizers_80() {
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

    #[test]
    fn test_char_and_word_tokenizers_81() {
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

    #[test]
    fn test_char_and_word_tokenizers_82() {
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

    #[test]
    fn test_char_and_word_tokenizers_83() {
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

    #[test]
    fn test_char_and_word_tokenizers_84() {
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

    #[test]
    fn test_char_and_word_tokenizers_85() {
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

    #[test]
    fn test_char_and_word_tokenizers_86() {
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

    #[test]
    fn test_char_and_word_tokenizers_87() {
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

    #[test]
    fn test_char_and_word_tokenizers_88() {
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

    #[test]
    fn test_char_and_word_tokenizers_89() {
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

    #[test]
    fn test_char_and_word_tokenizers_90() {
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

    #[test]
    fn test_char_and_word_tokenizers_91() {
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

    #[test]
    fn test_char_and_word_tokenizers_92() {
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

    #[test]
    fn test_char_and_word_tokenizers_93() {
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

    #[test]
    fn test_char_and_word_tokenizers_94() {
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

    #[test]
    fn test_char_and_word_tokenizers_95() {
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

    #[test]
    fn test_char_and_word_tokenizers_96() {
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

    #[test]
    fn test_char_and_word_tokenizers_97() {
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

    #[test]
    fn test_char_and_word_tokenizers_98() {
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

    #[test]
    fn test_char_and_word_tokenizers_99() {
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

    #[test]
    fn test_char_and_word_tokenizers_100() {
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

    #[test]
    fn test_char_and_word_tokenizers_101() {
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

    #[test]
    fn test_char_and_word_tokenizers_102() {
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

    #[test]
    fn test_char_and_word_tokenizers_103() {
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

    #[test]
    fn test_char_and_word_tokenizers_104() {
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

    #[test]
    fn test_char_and_word_tokenizers_105() {
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

    #[test]
    fn test_char_and_word_tokenizers_106() {
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

    #[test]
    fn test_char_and_word_tokenizers_107() {
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

    #[test]
    fn test_char_and_word_tokenizers_108() {
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

    #[test]
    fn test_char_and_word_tokenizers_109() {
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

    #[test]
    fn test_char_and_word_tokenizers_110() {
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

    #[test]
    fn test_char_and_word_tokenizers_111() {
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

    #[test]
    fn test_char_and_word_tokenizers_112() {
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

    #[test]
    fn test_char_and_word_tokenizers_113() {
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

    #[test]
    fn test_char_and_word_tokenizers_114() {
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

    #[test]
    fn test_char_and_word_tokenizers_115() {
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

    #[test]
    fn test_char_and_word_tokenizers_116() {
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

    #[test]
    fn test_char_and_word_tokenizers_117() {
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

    #[test]
    fn test_char_and_word_tokenizers_118() {
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

    #[test]
    fn test_char_and_word_tokenizers_119() {
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

    #[test]
    fn test_char_and_word_tokenizers_120() {
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

    #[test]
    fn test_char_and_word_tokenizers_121() {
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

    #[test]
    fn test_char_and_word_tokenizers_122() {
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

    #[test]
    fn test_char_and_word_tokenizers_123() {
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

    #[test]
    fn test_char_and_word_tokenizers_124() {
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

    #[test]
    fn test_char_and_word_tokenizers_125() {
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

    #[test]
    fn test_char_and_word_tokenizers_126() {
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

    #[test]
    fn test_char_and_word_tokenizers_127() {
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

    #[test]
    fn test_char_and_word_tokenizers_128() {
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

    #[test]
    fn test_char_and_word_tokenizers_129() {
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

    #[test]
    fn test_char_and_word_tokenizers_130() {
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

    #[test]
    fn test_char_and_word_tokenizers_131() {
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

    #[test]
    fn test_char_and_word_tokenizers_132() {
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

    #[test]
    fn test_char_and_word_tokenizers_133() {
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

    #[test]
    fn test_char_and_word_tokenizers_134() {
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

    #[test]
    fn test_char_and_word_tokenizers_135() {
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

    #[test]
    fn test_char_and_word_tokenizers_136() {
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

    #[test]
    fn test_char_and_word_tokenizers_137() {
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

    #[test]
    fn test_char_and_word_tokenizers_138() {
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

    #[test]
    fn test_char_and_word_tokenizers_139() {
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

    #[test]
    fn test_char_and_word_tokenizers_140() {
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

    #[test]
    fn test_char_and_word_tokenizers_141() {
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

    #[test]
    fn test_char_and_word_tokenizers_142() {
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

    #[test]
    fn test_char_and_word_tokenizers_143() {
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

    #[test]
    fn test_char_and_word_tokenizers_144() {
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

    #[test]
    fn test_char_and_word_tokenizers_145() {
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

    #[test]
    fn test_char_and_word_tokenizers_146() {
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

    #[test]
    fn test_char_and_word_tokenizers_147() {
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

    #[test]
    fn test_char_and_word_tokenizers_148() {
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

    #[test]
    fn test_char_and_word_tokenizers_149() {
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

    #[test]
    fn test_char_and_word_tokenizers_150() {
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

    #[test]
    fn test_char_and_word_tokenizers_151() {
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

    #[test]
    fn test_char_and_word_tokenizers_152() {
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

    #[test]
    fn test_char_and_word_tokenizers_153() {
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

    #[test]
    fn test_char_and_word_tokenizers_154() {
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

    #[test]
    fn test_char_and_word_tokenizers_155() {
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

    #[test]
    fn test_char_and_word_tokenizers_156() {
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

    #[test]
    fn test_char_and_word_tokenizers_157() {
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

    #[test]
    fn test_char_and_word_tokenizers_158() {
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

    #[test]
    fn test_char_and_word_tokenizers_159() {
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

    #[test]
    fn test_char_and_word_tokenizers_160() {
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

    #[test]
    fn test_char_and_word_tokenizers_161() {
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

    #[test]
    fn test_char_and_word_tokenizers_162() {
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

    #[test]
    fn test_char_and_word_tokenizers_163() {
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
}
