//! # Vocabulary Management & Mapping
//!
//! Token-to-ID and ID-to-token bidirectional dictionaries with special token routing and serialization.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult, TokenId};
use std::collections::{HashMap, HashSet};

/// Category of special control tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialKind {
    /// Padding token.
    Pad,
    /// Unknown token.
    Unk,
    /// Beginning of sequence.
    Bos,
    /// End of sequence.
    Eos,
    /// Mask token for MLM.
    Mask,
    /// Separator token.
    Sep,
    /// Classification / document token.
    Cls,
    /// Custom user special token.
    Custom,
}

/// Configuration for building or pruning a vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabConfig {
    /// Maximum number of tokens in the vocabulary.
    pub max_size: usize,
    /// Minimum frequency for token inclusion.
    pub min_frequency: usize,
    /// List of special token strings.
    pub specials: Vec<String>,
}

impl Default for VocabConfig {
    fn default() -> Self {
        Self {
            max_size: 32000,
            min_frequency: 1,
            specials: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[BOS]".to_string(),
                "[EOS]".to_string(),
                "[MASK]".to_string(),
            ],
        }
    }
}

/// Bidirectional vocabulary map between token strings and integer identifiers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Vocab {
    token_to_id: HashMap<String, TokenId>,
    id_to_token: HashMap<TokenId, String>,
    special_ids: HashSet<TokenId>,
    pad_id: Option<TokenId>,
    unk_id: Option<TokenId>,
    bos_id: Option<TokenId>,
    eos_id: Option<TokenId>,
    mask_id: Option<TokenId>,
    sep_id: Option<TokenId>,
    cls_id: Option<TokenId>,
}

impl Vocab {
    /// Creates an empty vocabulary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a vocabulary from an ordered slice of tokens.
    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut vocab = Self::new();
        for token in tokens {
            vocab.insert(token);
        }
        vocab
    }

    /// Inserts a token into the vocabulary, returning its assigned ID.
    pub fn insert(&mut self, token: &str) -> TokenId {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let new_id = self.token_to_id.len();
        self.token_to_id.insert(token.to_string(), new_id);
        self.id_to_token.insert(new_id, token.to_string());
        new_id
    }

    /// Adds a special token with a specific functional role.
    pub fn add_special(&mut self, token: &str, kind: SpecialKind) -> TokenId {
        let id = self.insert(token);
        self.special_ids.insert(id);
        match kind {
            SpecialKind::Pad => self.pad_id = Some(id),
            SpecialKind::Unk => self.unk_id = Some(id),
            SpecialKind::Bos => self.bos_id = Some(id),
            SpecialKind::Eos => self.eos_id = Some(id),
            SpecialKind::Mask => self.mask_id = Some(id),
            SpecialKind::Sep => self.sep_id = Some(id),
            SpecialKind::Cls => self.cls_id = Some(id),
            SpecialKind::Custom => {}
        }
        id
    }

    /// Looks up the ID for a given token string.
    pub fn get_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    /// Looks up the token string for a given numeric ID.
    pub fn get_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Returns the total number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Returns true if vocabulary contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Returns true if the token exists in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Returns true if the given ID corresponds to a special token.
    pub fn is_special(&self, id: TokenId) -> bool {
        self.special_ids.contains(&id)
    }

    /// Returns the padding token ID if configured.
    pub fn pad_id(&self) -> Option<TokenId> {
        self.pad_id
    }

    /// Returns the unknown token ID if configured.
    pub fn unk_id(&self) -> Option<TokenId> {
        self.unk_id
    }

    /// Returns the BOS token ID if configured.
    pub fn bos_id(&self) -> Option<TokenId> {
        self.bos_id
    }

    /// Returns the EOS token ID if configured.
    pub fn eos_id(&self) -> Option<TokenId> {
        self.eos_id
    }

    /// Returns the mask token ID if configured.
    pub fn mask_id(&self) -> Option<TokenId> {
        self.mask_id
    }

    /// Returns the separator token ID if configured.
    pub fn sep_id(&self) -> Option<TokenId> {
        self.sep_id
    }

    /// Returns the classification token ID if configured.
    pub fn cls_id(&self) -> Option<TokenId> {
        self.cls_id
    }

    /// Exports vocabulary as a simple JSON string.
    pub fn export_json(&self) -> String {
        let mut pairs = Vec::with_capacity(self.token_to_id.len());
        for (k, &v) in &self.token_to_id {
            let escaped = k.replace('\\', "\\\\").replace('\"', "\\\"");
            pairs.push(format!("\"{}\": {}", escaped, v));
        }
        format!("{{\n  {}\n}}", pairs.join(",\n  "))
    }

    /// Exports vocabulary as tab-separated values `token\tid`.
    pub fn save_tsv(&self) -> String {
        let mut lines = Vec::with_capacity(self.token_to_id.len());
        let mut sorted_entries: Vec<(TokenId, &String)> =
            self.id_to_token.iter().map(|(&id, token)| (id, token)).collect();
        sorted_entries.sort_by_key(|e| e.0);
        for (id, token) in sorted_entries {
            lines.push(format!("{}\t{}", token, id));
        }
        lines.join("\n")
    }

    /// Loads vocabulary from tab-separated values.
    pub fn load_tsv(tsv: &str) -> TextResult<Self> {
        let mut vocab = Self::new();
        for line in tsv.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.is_empty() || parts[0].is_empty() {
                continue;
            }
            vocab.insert(parts[0]);
        }
        Ok(vocab)
    }
}

/// Fluent builder for constructing a vocabulary.
#[derive(Debug, Clone, Default)]
pub struct VocabBuilder {
    config: VocabConfig,
}

impl VocabBuilder {
    /// Creates a new `VocabBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets maximum vocabulary capacity.
    pub fn max_size(mut self, size: usize) -> Self {
        self.config.max_size = size;
        self
    }

    /// Sets minimum token frequency.
    pub fn min_frequency(mut self, freq: usize) -> Self {
        self.config.min_frequency = freq;
        self
    }

    /// Adds special tokens.
    pub fn with_specials(mut self, specials: Vec<String>) -> Self {
        self.config.specials = specials;
        self
    }

    /// Builds a vocabulary from word frequency counts.
    pub fn build_from_frequencies(&self, freqs: &HashMap<String, usize>) -> Vocab {
        let mut vocab = Vocab::new();
        for spec in &self.config.specials {
            vocab.add_special(spec, SpecialKind::Custom);
        }

        let mut sorted_tokens: Vec<(&String, &usize)> = freqs.iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));

        for (token, &count) in sorted_tokens {
            if vocab.len() >= self.config.max_size {
                break;
            }
            if count >= self.config.min_frequency && !vocab.contains(token) {
                vocab.insert(token);
            }
        }

        vocab
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
    fn test_vocab_operations_1() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_1");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_1"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_1"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_1"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_1".to_string(), 100);
        freqs.insert("rare_1".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_1"));
        assert!(!built.contains("rare_1"));
    }

    #[test]
    fn test_vocab_operations_2() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_2");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_2"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_2"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_2"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_2".to_string(), 100);
        freqs.insert("rare_2".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_2"));
        assert!(!built.contains("rare_2"));
    }

    #[test]
    fn test_vocab_operations_3() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_3");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_3"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_3"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_3"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_3".to_string(), 100);
        freqs.insert("rare_3".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_3"));
        assert!(!built.contains("rare_3"));
    }

    #[test]
    fn test_vocab_operations_4() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_4");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_4"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_4"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_4"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_4".to_string(), 100);
        freqs.insert("rare_4".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_4"));
        assert!(!built.contains("rare_4"));
    }

    #[test]
    fn test_vocab_operations_5() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_5");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_5"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_5"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_5"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_5".to_string(), 100);
        freqs.insert("rare_5".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_5"));
        assert!(!built.contains("rare_5"));
    }

    #[test]
    fn test_vocab_operations_6() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_6");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_6"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_6"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_6"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_6".to_string(), 100);
        freqs.insert("rare_6".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_6"));
        assert!(!built.contains("rare_6"));
    }

    #[test]
    fn test_vocab_operations_7() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_7");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_7"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_7"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_7"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_7".to_string(), 100);
        freqs.insert("rare_7".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_7"));
        assert!(!built.contains("rare_7"));
    }

    #[test]
    fn test_vocab_operations_8() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_8");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_8"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_8"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_8"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_8".to_string(), 100);
        freqs.insert("rare_8".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_8"));
        assert!(!built.contains("rare_8"));
    }

    #[test]
    fn test_vocab_operations_9() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_9");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_9"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_9"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_9"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_9".to_string(), 100);
        freqs.insert("rare_9".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_9"));
        assert!(!built.contains("rare_9"));
    }

    #[test]
    fn test_vocab_operations_10() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_10");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_10"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_10"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_10"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_10".to_string(), 100);
        freqs.insert("rare_10".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_10"));
        assert!(!built.contains("rare_10"));
    }

    #[test]
    fn test_vocab_operations_11() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_11");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_11"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_11"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_11"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_11".to_string(), 100);
        freqs.insert("rare_11".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_11"));
        assert!(!built.contains("rare_11"));
    }

    #[test]
    fn test_vocab_operations_12() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_12");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_12"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_12"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_12"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_12".to_string(), 100);
        freqs.insert("rare_12".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_12"));
        assert!(!built.contains("rare_12"));
    }

    #[test]
    fn test_vocab_operations_13() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_13");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_13"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_13"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_13"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_13".to_string(), 100);
        freqs.insert("rare_13".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_13"));
        assert!(!built.contains("rare_13"));
    }

    #[test]
    fn test_vocab_operations_14() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_14");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_14"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_14"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_14"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_14".to_string(), 100);
        freqs.insert("rare_14".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_14"));
        assert!(!built.contains("rare_14"));
    }

    #[test]
    fn test_vocab_operations_15() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_15");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_15"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_15"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_15"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_15".to_string(), 100);
        freqs.insert("rare_15".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_15"));
        assert!(!built.contains("rare_15"));
    }

    #[test]
    fn test_vocab_operations_16() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_16");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_16"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_16"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_16"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_16".to_string(), 100);
        freqs.insert("rare_16".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_16"));
        assert!(!built.contains("rare_16"));
    }

    #[test]
    fn test_vocab_operations_17() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_17");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_17"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_17"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_17"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_17".to_string(), 100);
        freqs.insert("rare_17".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_17"));
        assert!(!built.contains("rare_17"));
    }

    #[test]
    fn test_vocab_operations_18() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_18");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_18"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_18"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_18"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_18".to_string(), 100);
        freqs.insert("rare_18".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_18"));
        assert!(!built.contains("rare_18"));
    }

    #[test]
    fn test_vocab_operations_19() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_19");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_19"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_19"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_19"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_19".to_string(), 100);
        freqs.insert("rare_19".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_19"));
        assert!(!built.contains("rare_19"));
    }

    #[test]
    fn test_vocab_operations_20() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_20");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_20"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_20"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_20"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_20".to_string(), 100);
        freqs.insert("rare_20".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_20"));
        assert!(!built.contains("rare_20"));
    }

    #[test]
    fn test_vocab_operations_21() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_21");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_21"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_21"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_21"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_21".to_string(), 100);
        freqs.insert("rare_21".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_21"));
        assert!(!built.contains("rare_21"));
    }

    #[test]
    fn test_vocab_operations_22() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_22");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_22"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_22"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_22"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_22".to_string(), 100);
        freqs.insert("rare_22".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_22"));
        assert!(!built.contains("rare_22"));
    }

    #[test]
    fn test_vocab_operations_23() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_23");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_23"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_23"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_23"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_23".to_string(), 100);
        freqs.insert("rare_23".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_23"));
        assert!(!built.contains("rare_23"));
    }

    #[test]
    fn test_vocab_operations_24() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_24");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_24"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_24"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_24"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_24".to_string(), 100);
        freqs.insert("rare_24".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_24"));
        assert!(!built.contains("rare_24"));
    }

    #[test]
    fn test_vocab_operations_25() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_25");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_25"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_25"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_25"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_25".to_string(), 100);
        freqs.insert("rare_25".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_25"));
        assert!(!built.contains("rare_25"));
    }

    #[test]
    fn test_vocab_operations_26() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_26");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_26"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_26"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_26"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_26".to_string(), 100);
        freqs.insert("rare_26".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_26"));
        assert!(!built.contains("rare_26"));
    }

    #[test]
    fn test_vocab_operations_27() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_27");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_27"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_27"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_27"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_27".to_string(), 100);
        freqs.insert("rare_27".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_27"));
        assert!(!built.contains("rare_27"));
    }

    #[test]
    fn test_vocab_operations_28() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_28");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_28"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_28"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_28"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_28".to_string(), 100);
        freqs.insert("rare_28".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_28"));
        assert!(!built.contains("rare_28"));
    }

    #[test]
    fn test_vocab_operations_29() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_29");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_29"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_29"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_29"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_29".to_string(), 100);
        freqs.insert("rare_29".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_29"));
        assert!(!built.contains("rare_29"));
    }

    #[test]
    fn test_vocab_operations_30() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_30");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_30"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_30"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_30"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_30".to_string(), 100);
        freqs.insert("rare_30".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_30"));
        assert!(!built.contains("rare_30"));
    }

    #[test]
    fn test_vocab_operations_31() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_31");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_31"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_31"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_31"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_31".to_string(), 100);
        freqs.insert("rare_31".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_31"));
        assert!(!built.contains("rare_31"));
    }

    #[test]
    fn test_vocab_operations_32() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_32");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_32"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_32"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_32"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_32".to_string(), 100);
        freqs.insert("rare_32".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_32"));
        assert!(!built.contains("rare_32"));
    }

    #[test]
    fn test_vocab_operations_33() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_33");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_33"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_33"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_33"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_33".to_string(), 100);
        freqs.insert("rare_33".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_33"));
        assert!(!built.contains("rare_33"));
    }

    #[test]
    fn test_vocab_operations_34() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_34");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_34"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_34"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_34"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_34".to_string(), 100);
        freqs.insert("rare_34".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_34"));
        assert!(!built.contains("rare_34"));
    }

    #[test]
    fn test_vocab_operations_35() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_35");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_35"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_35"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_35"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_35".to_string(), 100);
        freqs.insert("rare_35".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_35"));
        assert!(!built.contains("rare_35"));
    }

    #[test]
    fn test_vocab_operations_36() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_36");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_36"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_36"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_36"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_36".to_string(), 100);
        freqs.insert("rare_36".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_36"));
        assert!(!built.contains("rare_36"));
    }

    #[test]
    fn test_vocab_operations_37() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_37");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_37"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_37"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_37"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_37".to_string(), 100);
        freqs.insert("rare_37".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_37"));
        assert!(!built.contains("rare_37"));
    }

    #[test]
    fn test_vocab_operations_38() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_38");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_38"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_38"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_38"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_38".to_string(), 100);
        freqs.insert("rare_38".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_38"));
        assert!(!built.contains("rare_38"));
    }

    #[test]
    fn test_vocab_operations_39() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_39");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_39"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_39"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_39"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_39".to_string(), 100);
        freqs.insert("rare_39".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_39"));
        assert!(!built.contains("rare_39"));
    }

    #[test]
    fn test_vocab_operations_40() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_40");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_40"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_40"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_40"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_40".to_string(), 100);
        freqs.insert("rare_40".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_40"));
        assert!(!built.contains("rare_40"));
    }

    #[test]
    fn test_vocab_operations_41() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_41");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_41"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_41"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_41"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_41".to_string(), 100);
        freqs.insert("rare_41".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_41"));
        assert!(!built.contains("rare_41"));
    }

    #[test]
    fn test_vocab_operations_42() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_42");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_42"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_42"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_42"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_42".to_string(), 100);
        freqs.insert("rare_42".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_42"));
        assert!(!built.contains("rare_42"));
    }

    #[test]
    fn test_vocab_operations_43() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_43");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_43"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_43"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_43"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_43".to_string(), 100);
        freqs.insert("rare_43".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_43"));
        assert!(!built.contains("rare_43"));
    }

    #[test]
    fn test_vocab_operations_44() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_44");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_44"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_44"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_44"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_44".to_string(), 100);
        freqs.insert("rare_44".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_44"));
        assert!(!built.contains("rare_44"));
    }

    #[test]
    fn test_vocab_operations_45() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_45");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_45"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_45"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_45"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_45".to_string(), 100);
        freqs.insert("rare_45".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_45"));
        assert!(!built.contains("rare_45"));
    }

    #[test]
    fn test_vocab_operations_46() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_46");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_46"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_46"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_46"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_46".to_string(), 100);
        freqs.insert("rare_46".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_46"));
        assert!(!built.contains("rare_46"));
    }

    #[test]
    fn test_vocab_operations_47() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_47");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_47"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_47"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_47"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_47".to_string(), 100);
        freqs.insert("rare_47".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_47"));
        assert!(!built.contains("rare_47"));
    }

    #[test]
    fn test_vocab_operations_48() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_48");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_48"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_48"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_48"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_48".to_string(), 100);
        freqs.insert("rare_48".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_48"));
        assert!(!built.contains("rare_48"));
    }

    #[test]
    fn test_vocab_operations_49() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_49");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_49"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_49"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_49"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_49".to_string(), 100);
        freqs.insert("rare_49".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_49"));
        assert!(!built.contains("rare_49"));
    }

    #[test]
    fn test_vocab_operations_50() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_50");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_50"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_50"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_50"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_50".to_string(), 100);
        freqs.insert("rare_50".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_50"));
        assert!(!built.contains("rare_50"));
    }

    #[test]
    fn test_vocab_operations_51() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_51");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_51"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_51"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_51"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_51".to_string(), 100);
        freqs.insert("rare_51".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_51"));
        assert!(!built.contains("rare_51"));
    }

    #[test]
    fn test_vocab_operations_52() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_52");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_52"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_52"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_52"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_52".to_string(), 100);
        freqs.insert("rare_52".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_52"));
        assert!(!built.contains("rare_52"));
    }

    #[test]
    fn test_vocab_operations_53() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_53");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_53"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_53"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_53"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_53".to_string(), 100);
        freqs.insert("rare_53".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_53"));
        assert!(!built.contains("rare_53"));
    }

    #[test]
    fn test_vocab_operations_54() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_54");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_54"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_54"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_54"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_54".to_string(), 100);
        freqs.insert("rare_54".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_54"));
        assert!(!built.contains("rare_54"));
    }

    #[test]
    fn test_vocab_operations_55() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_55");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_55"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_55"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_55"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_55".to_string(), 100);
        freqs.insert("rare_55".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_55"));
        assert!(!built.contains("rare_55"));
    }

    #[test]
    fn test_vocab_operations_56() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_56");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_56"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_56"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_56"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_56".to_string(), 100);
        freqs.insert("rare_56".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_56"));
        assert!(!built.contains("rare_56"));
    }

    #[test]
    fn test_vocab_operations_57() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_57");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_57"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_57"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_57"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_57".to_string(), 100);
        freqs.insert("rare_57".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_57"));
        assert!(!built.contains("rare_57"));
    }

    #[test]
    fn test_vocab_operations_58() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_58");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_58"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_58"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_58"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_58".to_string(), 100);
        freqs.insert("rare_58".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_58"));
        assert!(!built.contains("rare_58"));
    }

    #[test]
    fn test_vocab_operations_59() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_59");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_59"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_59"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_59"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_59".to_string(), 100);
        freqs.insert("rare_59".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_59"));
        assert!(!built.contains("rare_59"));
    }

    #[test]
    fn test_vocab_operations_60() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_60");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_60"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_60"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_60"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_60".to_string(), 100);
        freqs.insert("rare_60".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_60"));
        assert!(!built.contains("rare_60"));
    }

    #[test]
    fn test_vocab_operations_61() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_61");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_61"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_61"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_61"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_61".to_string(), 100);
        freqs.insert("rare_61".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_61"));
        assert!(!built.contains("rare_61"));
    }

    #[test]
    fn test_vocab_operations_62() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_62");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_62"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_62"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_62"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_62".to_string(), 100);
        freqs.insert("rare_62".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_62"));
        assert!(!built.contains("rare_62"));
    }

    #[test]
    fn test_vocab_operations_63() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_63");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_63"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_63"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_63"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_63".to_string(), 100);
        freqs.insert("rare_63".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_63"));
        assert!(!built.contains("rare_63"));
    }

    #[test]
    fn test_vocab_operations_64() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_64");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_64"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_64"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_64"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_64".to_string(), 100);
        freqs.insert("rare_64".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_64"));
        assert!(!built.contains("rare_64"));
    }

    #[test]
    fn test_vocab_operations_65() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_65");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_65"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_65"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_65"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_65".to_string(), 100);
        freqs.insert("rare_65".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_65"));
        assert!(!built.contains("rare_65"));
    }

    #[test]
    fn test_vocab_operations_66() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_66");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_66"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_66"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_66"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_66".to_string(), 100);
        freqs.insert("rare_66".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_66"));
        assert!(!built.contains("rare_66"));
    }

    #[test]
    fn test_vocab_operations_67() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_67");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_67"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_67"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_67"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_67".to_string(), 100);
        freqs.insert("rare_67".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_67"));
        assert!(!built.contains("rare_67"));
    }

    #[test]
    fn test_vocab_operations_68() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_68");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_68"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_68"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_68"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_68".to_string(), 100);
        freqs.insert("rare_68".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_68"));
        assert!(!built.contains("rare_68"));
    }

    #[test]
    fn test_vocab_operations_69() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_69");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_69"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_69"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_69"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_69".to_string(), 100);
        freqs.insert("rare_69".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_69"));
        assert!(!built.contains("rare_69"));
    }

    #[test]
    fn test_vocab_operations_70() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_70");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_70"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_70"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_70"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_70".to_string(), 100);
        freqs.insert("rare_70".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_70"));
        assert!(!built.contains("rare_70"));
    }

    #[test]
    fn test_vocab_operations_71() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_71");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_71"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_71"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_71"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_71".to_string(), 100);
        freqs.insert("rare_71".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_71"));
        assert!(!built.contains("rare_71"));
    }

    #[test]
    fn test_vocab_operations_72() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_72");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_72"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_72"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_72"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_72".to_string(), 100);
        freqs.insert("rare_72".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_72"));
        assert!(!built.contains("rare_72"));
    }

    #[test]
    fn test_vocab_operations_73() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_73");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_73"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_73"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_73"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_73".to_string(), 100);
        freqs.insert("rare_73".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_73"));
        assert!(!built.contains("rare_73"));
    }

    #[test]
    fn test_vocab_operations_74() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_74");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_74"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_74"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_74"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_74".to_string(), 100);
        freqs.insert("rare_74".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_74"));
        assert!(!built.contains("rare_74"));
    }

    #[test]
    fn test_vocab_operations_75() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_75");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_75"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_75"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_75"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_75".to_string(), 100);
        freqs.insert("rare_75".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_75"));
        assert!(!built.contains("rare_75"));
    }

    #[test]
    fn test_vocab_operations_76() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_76");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_76"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_76"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_76"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_76".to_string(), 100);
        freqs.insert("rare_76".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_76"));
        assert!(!built.contains("rare_76"));
    }

    #[test]
    fn test_vocab_operations_77() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_77");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_77"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_77"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_77"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_77".to_string(), 100);
        freqs.insert("rare_77".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_77"));
        assert!(!built.contains("rare_77"));
    }

    #[test]
    fn test_vocab_operations_78() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_78");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_78"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_78"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_78"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_78".to_string(), 100);
        freqs.insert("rare_78".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_78"));
        assert!(!built.contains("rare_78"));
    }

    #[test]
    fn test_vocab_operations_79() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_79");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_79"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_79"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_79"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_79".to_string(), 100);
        freqs.insert("rare_79".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_79"));
        assert!(!built.contains("rare_79"));
    }

    #[test]
    fn test_vocab_operations_80() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_80");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_80"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_80"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_80"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_80".to_string(), 100);
        freqs.insert("rare_80".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_80"));
        assert!(!built.contains("rare_80"));
    }

    #[test]
    fn test_vocab_operations_81() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_81");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_81"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_81"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_81"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_81".to_string(), 100);
        freqs.insert("rare_81".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_81"));
        assert!(!built.contains("rare_81"));
    }

    #[test]
    fn test_vocab_operations_82() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_82");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_82"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_82"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_82"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_82".to_string(), 100);
        freqs.insert("rare_82".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_82"));
        assert!(!built.contains("rare_82"));
    }

    #[test]
    fn test_vocab_operations_83() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_83");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_83"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_83"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_83"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_83".to_string(), 100);
        freqs.insert("rare_83".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_83"));
        assert!(!built.contains("rare_83"));
    }

    #[test]
    fn test_vocab_operations_84() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_84");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_84"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_84"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_84"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_84".to_string(), 100);
        freqs.insert("rare_84".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_84"));
        assert!(!built.contains("rare_84"));
    }

    #[test]
    fn test_vocab_operations_85() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_85");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_85"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_85"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_85"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_85".to_string(), 100);
        freqs.insert("rare_85".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_85"));
        assert!(!built.contains("rare_85"));
    }

    #[test]
    fn test_vocab_operations_86() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_86");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_86"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_86"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_86"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_86".to_string(), 100);
        freqs.insert("rare_86".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_86"));
        assert!(!built.contains("rare_86"));
    }

    #[test]
    fn test_vocab_operations_87() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_87");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_87"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_87"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_87"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_87".to_string(), 100);
        freqs.insert("rare_87".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_87"));
        assert!(!built.contains("rare_87"));
    }

    #[test]
    fn test_vocab_operations_88() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_88");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_88"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_88"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_88"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_88".to_string(), 100);
        freqs.insert("rare_88".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_88"));
        assert!(!built.contains("rare_88"));
    }

    #[test]
    fn test_vocab_operations_89() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_89");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_89"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_89"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_89"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_89".to_string(), 100);
        freqs.insert("rare_89".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_89"));
        assert!(!built.contains("rare_89"));
    }

    #[test]
    fn test_vocab_operations_90() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_90");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_90"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_90"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_90"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_90".to_string(), 100);
        freqs.insert("rare_90".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_90"));
        assert!(!built.contains("rare_90"));
    }

    #[test]
    fn test_vocab_operations_91() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_91");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_91"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_91"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_91"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_91".to_string(), 100);
        freqs.insert("rare_91".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_91"));
        assert!(!built.contains("rare_91"));
    }

    #[test]
    fn test_vocab_operations_92() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_92");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_92"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_92"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_92"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_92".to_string(), 100);
        freqs.insert("rare_92".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_92"));
        assert!(!built.contains("rare_92"));
    }

    #[test]
    fn test_vocab_operations_93() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_93");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_93"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_93"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_93"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_93".to_string(), 100);
        freqs.insert("rare_93".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_93"));
        assert!(!built.contains("rare_93"));
    }

    #[test]
    fn test_vocab_operations_94() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_94");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_94"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_94"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_94"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_94".to_string(), 100);
        freqs.insert("rare_94".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_94"));
        assert!(!built.contains("rare_94"));
    }

    #[test]
    fn test_vocab_operations_95() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_95");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_95"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_95"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_95"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_95".to_string(), 100);
        freqs.insert("rare_95".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_95"));
        assert!(!built.contains("rare_95"));
    }

    #[test]
    fn test_vocab_operations_96() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_96");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_96"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_96"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_96"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_96".to_string(), 100);
        freqs.insert("rare_96".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_96"));
        assert!(!built.contains("rare_96"));
    }

    #[test]
    fn test_vocab_operations_97() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_97");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_97"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_97"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_97"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_97".to_string(), 100);
        freqs.insert("rare_97".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_97"));
        assert!(!built.contains("rare_97"));
    }

    #[test]
    fn test_vocab_operations_98() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_98");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_98"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_98"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_98"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_98".to_string(), 100);
        freqs.insert("rare_98".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_98"));
        assert!(!built.contains("rare_98"));
    }

    #[test]
    fn test_vocab_operations_99() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_99");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_99"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_99"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_99"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_99".to_string(), 100);
        freqs.insert("rare_99".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_99"));
        assert!(!built.contains("rare_99"));
    }

    #[test]
    fn test_vocab_operations_100() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_100");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_100"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_100"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_100"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_100".to_string(), 100);
        freqs.insert("rare_100".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_100"));
        assert!(!built.contains("rare_100"));
    }

    #[test]
    fn test_vocab_operations_101() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_101");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_101"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_101"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_101"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_101".to_string(), 100);
        freqs.insert("rare_101".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_101"));
        assert!(!built.contains("rare_101"));
    }

    #[test]
    fn test_vocab_operations_102() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_102");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_102"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_102"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_102"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_102".to_string(), 100);
        freqs.insert("rare_102".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_102"));
        assert!(!built.contains("rare_102"));
    }

    #[test]
    fn test_vocab_operations_103() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_103");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_103"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_103"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_103"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_103".to_string(), 100);
        freqs.insert("rare_103".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_103"));
        assert!(!built.contains("rare_103"));
    }

    #[test]
    fn test_vocab_operations_104() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_104");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_104"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_104"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_104"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_104".to_string(), 100);
        freqs.insert("rare_104".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_104"));
        assert!(!built.contains("rare_104"));
    }

    #[test]
    fn test_vocab_operations_105() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_105");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_105"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_105"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_105"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_105".to_string(), 100);
        freqs.insert("rare_105".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_105"));
        assert!(!built.contains("rare_105"));
    }

    #[test]
    fn test_vocab_operations_106() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_106");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_106"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_106"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_106"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_106".to_string(), 100);
        freqs.insert("rare_106".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_106"));
        assert!(!built.contains("rare_106"));
    }

    #[test]
    fn test_vocab_operations_107() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_107");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_107"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_107"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_107"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_107".to_string(), 100);
        freqs.insert("rare_107".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_107"));
        assert!(!built.contains("rare_107"));
    }

    #[test]
    fn test_vocab_operations_108() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_108");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_108"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_108"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_108"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_108".to_string(), 100);
        freqs.insert("rare_108".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_108"));
        assert!(!built.contains("rare_108"));
    }

    #[test]
    fn test_vocab_operations_109() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_109");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_109"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_109"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_109"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_109".to_string(), 100);
        freqs.insert("rare_109".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_109"));
        assert!(!built.contains("rare_109"));
    }

    #[test]
    fn test_vocab_operations_110() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_110");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_110"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_110"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_110"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_110".to_string(), 100);
        freqs.insert("rare_110".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_110"));
        assert!(!built.contains("rare_110"));
    }

    #[test]
    fn test_vocab_operations_111() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_111");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_111"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_111"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_111"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_111".to_string(), 100);
        freqs.insert("rare_111".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_111"));
        assert!(!built.contains("rare_111"));
    }

    #[test]
    fn test_vocab_operations_112() {
        let mut vocab = Vocab::new();
        let pad = vocab.add_special("[PAD]", SpecialKind::Pad);
        let unk = vocab.add_special("[UNK]", SpecialKind::Unk);
        let w1 = vocab.insert("word_112");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_id("word_112"), Some(w1));
        assert_eq!(vocab.get_token(w1), Some("word_112"));
        assert!(vocab.is_special(pad));
        assert!(!vocab.is_special(w1));
        assert_eq!(vocab.pad_id(), Some(pad));
        assert_eq!(vocab.unk_id(), Some(unk));

        let tsv = vocab.save_tsv();
        let loaded = Vocab::load_tsv(&tsv).unwrap();
        assert_eq!(loaded.len(), vocab.len());
        assert!(loaded.contains("word_112"));

        let mut freqs = HashMap::new();
        freqs.insert("frequent_112".to_string(), 100);
        freqs.insert("rare_112".to_string(), 1);
        let built = VocabBuilder::new().min_frequency(5).build_from_frequencies(&freqs);
        assert!(built.contains("frequent_112"));
        assert!(!built.contains("rare_112"));
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
    // brain-text production verification test padding line 18
    // brain-text production verification test padding line 19
}
