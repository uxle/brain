//! # Tokenizer Architecture & Common Interface
//!
//! Universal `Tokenizer` trait, batch execution, and subword algorithm abstractions.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod bpe;
pub mod bytelevel;
pub mod char;
pub mod normalizer;
pub mod post;
pub mod pretokenizer;
pub mod sentencepiece;
pub mod trainer;
pub mod wordpiece;

use crate::core::{TextError, TextResult, TokenId, TokenizedOutput};
use crate::vocab::Vocab;

/// Universal interface for all NLP tokenizers (BPE, SentencePiece, WordPiece, Char, Word).
pub trait Tokenizer: Send + Sync {
    /// Encodes a raw text string into token IDs and metadata.
    fn encode(&self, text: &str) -> TextResult<TokenizedOutput>;

    /// Decodes a sequence of token IDs back into a reconstructed text string.
    fn decode(&self, ids: &[TokenId]) -> TextResult<String>;

    /// Tokenizes a text string into individual token substring pieces.
    fn tokenize(&self, text: &str) -> TextResult<Vec<String>>;

    /// Returns the total size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Returns a reference to the underlying vocabulary.
    fn get_vocab(&self) -> &Vocab;

    /// Converts a numeric token ID to its string representation.
    fn id_to_token(&self, id: TokenId) -> Option<String> {
        self.get_vocab().get_token(id).map(|s| s.to_string())
    }

    /// Converts a string token to its numerical ID.
    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.get_vocab().get_id(token)
    }

    /// Returns the padding token ID if defined.
    fn pad_token_id(&self) -> Option<TokenId> {
        self.get_vocab().pad_id()
    }

    /// Returns the unknown token ID if defined.
    fn unk_token_id(&self) -> Option<TokenId> {
        self.get_vocab().unk_id()
    }

    /// Returns the beginning-of-sequence token ID if defined.
    fn bos_token_id(&self) -> Option<TokenId> {
        self.get_vocab().bos_id()
    }

    /// Returns the end-of-sequence token ID if defined.
    fn eos_token_id(&self) -> Option<TokenId> {
        self.get_vocab().eos_id()
    }

    /// Returns the mask token ID if defined.
    fn mask_token_id(&self) -> Option<TokenId> {
        self.get_vocab().mask_id()
    }

    /// Encodes a batch of strings concurrently or sequentially.
    fn encode_batch(&self, texts: &[&str]) -> TextResult<Vec<TokenizedOutput>> {
        texts.iter().map(|&t| self.encode(t)).collect()
    }

    /// Decodes a batch of token ID sequences into reconstructed text strings.
    fn decode_batch(&self, batch_ids: &[Vec<TokenId>]) -> TextResult<Vec<String>> {
        batch_ids.iter().map(|ids| self.decode(ids)).collect()
    }
}

/// Tokenizer error alias for standard `TextError`.
pub type TokenizerError = TextError;

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
    fn test_tokenizer_trait_helpers_1() {
        let mut vocab = Vocab::new();
        vocab.add_special("[PAD]", SpecialKind::Pad);
        vocab.add_special("[UNK]", SpecialKind::Unk);
        vocab.insert("hello_1");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.pad_id(), Some(0));
        assert_eq!(vocab.unk_id(), Some(1));
    }
}
