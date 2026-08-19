//! # Language Modeling Preprocessing & Attention Mask Utilities
//!
//! Causal LM target shifting, Masked LM token replacement, packing, and sliding window chunking.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::TokenId;
use crate::ops::mask_tokens;
use crate::utils::TextRng;

/// Configuration for LM sequence preparation.
#[derive(Debug, Clone, PartialEq)]
pub struct LmConfig {
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Masking probability for Masked Language Modeling (typically 0.15).
    pub mlm_prob: f32,
    /// Mask token identifier.
    pub mask_token_id: TokenId,
    /// Padding token identifier.
    pub pad_token_id: TokenId,
    /// Optional classification / BOS token identifier.
    pub cls_token_id: Option<TokenId>,
    /// Optional separator / EOS token identifier.
    pub sep_token_id: Option<TokenId>,
    /// Whether to pack multiple short documents into fixed length chunks.
    pub pack_documents: bool,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            mlm_prob: 0.15,
            mask_token_id: 103,
            pad_token_id: 0,
            cls_token_id: Some(101),
            sep_token_id: Some(102),
            pack_documents: false,
        }
    }
}

/// Preprocessing utilities for Causal (autoregressive) and Masked Language Models.
pub struct LmPreprocessor;

impl LmPreprocessor {
    /// Prepares sequences for Causal Language Modeling: input `tokens[0..N-1]`, target `tokens[1..N]`.
    pub fn prepare_causal_lm(
        tokens: &[TokenId],
        max_len: usize,
    ) -> (Vec<TokenId>, Vec<TokenId>, Vec<u8>) {
        if tokens.len() < 2 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let total = tokens.len().min(max_len + 1);
        let input_slice = &tokens[0..total - 1];
        let target_slice = &tokens[1..total];

        let mut input_ids = input_slice.to_vec();
        let mut labels = target_slice.to_vec();
        let mut attention_mask = vec![1u8; input_ids.len()];

        while input_ids.len() < max_len {
            input_ids.push(0);
            labels.push(0);
            attention_mask.push(0);
        }

        (input_ids, labels, attention_mask)
    }

    /// Prepares sequences for Masked Language Modeling with standard `-100` ignore index for unmasked tokens.
    pub fn prepare_masked_lm(
        tokens: &[TokenId],
        config: &LmConfig,
        vocab_size: usize,
        rng: &mut TextRng,
    ) -> (Vec<TokenId>, Vec<i64>, Vec<u8>) {
        let len = tokens.len().min(config.max_seq_len);
        let current_tokens = &tokens[..len];

        let (masked_ids, mask_flags) = mask_tokens(
            current_tokens,
            config.mlm_prob,
            config.mask_token_id,
            vocab_size,
            rng,
        );

        let mut labels: Vec<i64> = current_tokens
            .iter()
            .zip(mask_flags.iter())
            .map(|(&orig_id, &is_masked)| if is_masked { orig_id as i64 } else { -100i64 })
            .collect();

        let mut input_ids = masked_ids;
        let mut attention_mask = vec![1u8; input_ids.len()];

        while input_ids.len() < config.max_seq_len {
            input_ids.push(config.pad_token_id);
            labels.push(-100);
            attention_mask.push(0);
        }

        (input_ids, labels, attention_mask)
    }

    /// Generates lower-triangular causal attention mask (`1` for allowed attention, `0` for masked future tokens).
    pub fn create_causal_mask(seq_len: usize) -> Vec<Vec<u8>> {
        let mut mask = vec![vec![0u8; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i][j] = 1u8;
            }
        }
        mask
    }

    /// Generates prefix-LM mask (bidirectional attention over prefix, causal over target generation).
    pub fn create_prefix_lm_mask(prefix_len: usize, total_len: usize) -> Vec<Vec<u8>> {
        let mut mask = vec![vec![0u8; total_len]; total_len];
        for i in 0..total_len {
            for j in 0..total_len {
                if j <= i || (i < prefix_len && j < prefix_len) {
                    mask[i][j] = 1u8;
                }
            }
        }
        mask
    }

    /// Packs multiple short document sequences separated by `eos_id` into uniform fixed-length batches.
    pub fn pack_documents(
        docs: &[Vec<TokenId>],
        max_len: usize,
        eos_id: TokenId,
    ) -> Vec<Vec<TokenId>> {
        crate::ops::pack_sequences(docs, max_len, eos_id)
    }

    /// Slices long token sequences into overlapping sliding window chunks.
    pub fn sliding_window_chunks(
        tokens: &[TokenId],
        window_size: usize,
        stride: usize,
    ) -> Vec<Vec<TokenId>> {
        if tokens.is_empty() || window_size == 0 || stride == 0 {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + window_size).min(tokens.len());
            let mut chunk = tokens[start..end].to_vec();
            while chunk.len() < window_size {
                chunk.push(0);
            }
            chunks.push(chunk);
            if start + window_size >= tokens.len() {
                break;
            }
            start += stride;
        }

        chunks
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
    fn test_lm_utilities_1() {
        let tokens = vec![10, 20, 30, 40, 50, 1];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(1 as u64);
        let cfg = LmConfig { max_seq_len: 6, mlm_prob: 0.5, mask_token_id: 999, ..Default::default() };
        let (mlm_in, mlm_lbl, mlm_mask) = LmPreprocessor::prepare_masked_lm(&tokens, &cfg, 1000, &mut rng);
        assert_eq!(mlm_in.len(), 6);
        assert_eq!(mlm_lbl.len(), 6);
        assert_eq!(mlm_mask.len(), 6);

        let causal_m = LmPreprocessor::create_causal_mask(3);
        assert_eq!(causal_m[0], vec![1, 0, 0]);
        assert_eq!(causal_m[1], vec![1, 1, 0]);
        assert_eq!(causal_m[2], vec![1, 1, 1]);

        let prefix_m = LmPreprocessor::create_prefix_lm_mask(2, 3);
        assert_eq!(prefix_m[0], vec![1, 1, 0]);
        assert_eq!(prefix_m[1], vec![1, 1, 0]);
        assert_eq!(prefix_m[2], vec![1, 1, 1]);

        let chunks = LmPreprocessor::sliding_window_chunks(&tokens, 3, 2);
        assert!(!chunks.is_empty());
    }
}
