//! # Vocabulary Management & Mapping
//!
//! Token-to-ID and ID-to-token bidirectional dictionaries with special token routing and serialization.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

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
        let mut sorted_entries: Vec<(TokenId, &String)> = self
            .id_to_token
            .iter()
            .map(|(&id, token)| (id, token))
            .collect();
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
        let built = VocabBuilder::new()
            .min_frequency(5)
            .build_from_frequencies(&freqs);
        assert!(built.contains("frequent_1"));
        assert!(!built.contains("rare_1"));
    }
}
