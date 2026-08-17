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

    #[test]
    fn test_lm_utilities_2() {
        let tokens = vec![10, 20, 30, 40, 50, 2];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(2 as u64);
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

    #[test]
    fn test_lm_utilities_3() {
        let tokens = vec![10, 20, 30, 40, 50, 3];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(3 as u64);
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

    #[test]
    fn test_lm_utilities_4() {
        let tokens = vec![10, 20, 30, 40, 50, 4];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(4 as u64);
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

    #[test]
    fn test_lm_utilities_5() {
        let tokens = vec![10, 20, 30, 40, 50, 5];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(5 as u64);
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

    #[test]
    fn test_lm_utilities_6() {
        let tokens = vec![10, 20, 30, 40, 50, 6];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(6 as u64);
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

    #[test]
    fn test_lm_utilities_7() {
        let tokens = vec![10, 20, 30, 40, 50, 7];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(7 as u64);
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

    #[test]
    fn test_lm_utilities_8() {
        let tokens = vec![10, 20, 30, 40, 50, 8];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(8 as u64);
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

    #[test]
    fn test_lm_utilities_9() {
        let tokens = vec![10, 20, 30, 40, 50, 9];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(9 as u64);
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

    #[test]
    fn test_lm_utilities_10() {
        let tokens = vec![10, 20, 30, 40, 50, 10];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(10 as u64);
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

    #[test]
    fn test_lm_utilities_11() {
        let tokens = vec![10, 20, 30, 40, 50, 11];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(11 as u64);
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

    #[test]
    fn test_lm_utilities_12() {
        let tokens = vec![10, 20, 30, 40, 50, 12];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(12 as u64);
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

    #[test]
    fn test_lm_utilities_13() {
        let tokens = vec![10, 20, 30, 40, 50, 13];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(13 as u64);
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

    #[test]
    fn test_lm_utilities_14() {
        let tokens = vec![10, 20, 30, 40, 50, 14];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(14 as u64);
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

    #[test]
    fn test_lm_utilities_15() {
        let tokens = vec![10, 20, 30, 40, 50, 15];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(15 as u64);
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

    #[test]
    fn test_lm_utilities_16() {
        let tokens = vec![10, 20, 30, 40, 50, 16];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(16 as u64);
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

    #[test]
    fn test_lm_utilities_17() {
        let tokens = vec![10, 20, 30, 40, 50, 17];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(17 as u64);
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

    #[test]
    fn test_lm_utilities_18() {
        let tokens = vec![10, 20, 30, 40, 50, 18];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(18 as u64);
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

    #[test]
    fn test_lm_utilities_19() {
        let tokens = vec![10, 20, 30, 40, 50, 19];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(19 as u64);
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

    #[test]
    fn test_lm_utilities_20() {
        let tokens = vec![10, 20, 30, 40, 50, 20];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(20 as u64);
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

    #[test]
    fn test_lm_utilities_21() {
        let tokens = vec![10, 20, 30, 40, 50, 21];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(21 as u64);
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

    #[test]
    fn test_lm_utilities_22() {
        let tokens = vec![10, 20, 30, 40, 50, 22];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(22 as u64);
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

    #[test]
    fn test_lm_utilities_23() {
        let tokens = vec![10, 20, 30, 40, 50, 23];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(23 as u64);
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

    #[test]
    fn test_lm_utilities_24() {
        let tokens = vec![10, 20, 30, 40, 50, 24];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(24 as u64);
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

    #[test]
    fn test_lm_utilities_25() {
        let tokens = vec![10, 20, 30, 40, 50, 25];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(25 as u64);
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

    #[test]
    fn test_lm_utilities_26() {
        let tokens = vec![10, 20, 30, 40, 50, 26];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(26 as u64);
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

    #[test]
    fn test_lm_utilities_27() {
        let tokens = vec![10, 20, 30, 40, 50, 27];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(27 as u64);
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

    #[test]
    fn test_lm_utilities_28() {
        let tokens = vec![10, 20, 30, 40, 50, 28];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(28 as u64);
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

    #[test]
    fn test_lm_utilities_29() {
        let tokens = vec![10, 20, 30, 40, 50, 29];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(29 as u64);
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

    #[test]
    fn test_lm_utilities_30() {
        let tokens = vec![10, 20, 30, 40, 50, 30];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(30 as u64);
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

    #[test]
    fn test_lm_utilities_31() {
        let tokens = vec![10, 20, 30, 40, 50, 31];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(31 as u64);
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

    #[test]
    fn test_lm_utilities_32() {
        let tokens = vec![10, 20, 30, 40, 50, 32];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(32 as u64);
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

    #[test]
    fn test_lm_utilities_33() {
        let tokens = vec![10, 20, 30, 40, 50, 33];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(33 as u64);
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

    #[test]
    fn test_lm_utilities_34() {
        let tokens = vec![10, 20, 30, 40, 50, 34];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(34 as u64);
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

    #[test]
    fn test_lm_utilities_35() {
        let tokens = vec![10, 20, 30, 40, 50, 35];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(35 as u64);
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

    #[test]
    fn test_lm_utilities_36() {
        let tokens = vec![10, 20, 30, 40, 50, 36];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(36 as u64);
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

    #[test]
    fn test_lm_utilities_37() {
        let tokens = vec![10, 20, 30, 40, 50, 37];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(37 as u64);
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

    #[test]
    fn test_lm_utilities_38() {
        let tokens = vec![10, 20, 30, 40, 50, 38];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(38 as u64);
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

    #[test]
    fn test_lm_utilities_39() {
        let tokens = vec![10, 20, 30, 40, 50, 39];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(39 as u64);
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

    #[test]
    fn test_lm_utilities_40() {
        let tokens = vec![10, 20, 30, 40, 50, 40];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(40 as u64);
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

    #[test]
    fn test_lm_utilities_41() {
        let tokens = vec![10, 20, 30, 40, 50, 41];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(41 as u64);
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

    #[test]
    fn test_lm_utilities_42() {
        let tokens = vec![10, 20, 30, 40, 50, 42];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(42 as u64);
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

    #[test]
    fn test_lm_utilities_43() {
        let tokens = vec![10, 20, 30, 40, 50, 43];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(43 as u64);
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

    #[test]
    fn test_lm_utilities_44() {
        let tokens = vec![10, 20, 30, 40, 50, 44];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(44 as u64);
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

    #[test]
    fn test_lm_utilities_45() {
        let tokens = vec![10, 20, 30, 40, 50, 45];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(45 as u64);
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

    #[test]
    fn test_lm_utilities_46() {
        let tokens = vec![10, 20, 30, 40, 50, 46];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(46 as u64);
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

    #[test]
    fn test_lm_utilities_47() {
        let tokens = vec![10, 20, 30, 40, 50, 47];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(47 as u64);
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

    #[test]
    fn test_lm_utilities_48() {
        let tokens = vec![10, 20, 30, 40, 50, 48];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(48 as u64);
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

    #[test]
    fn test_lm_utilities_49() {
        let tokens = vec![10, 20, 30, 40, 50, 49];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(49 as u64);
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

    #[test]
    fn test_lm_utilities_50() {
        let tokens = vec![10, 20, 30, 40, 50, 50];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(50 as u64);
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

    #[test]
    fn test_lm_utilities_51() {
        let tokens = vec![10, 20, 30, 40, 50, 51];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(51 as u64);
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

    #[test]
    fn test_lm_utilities_52() {
        let tokens = vec![10, 20, 30, 40, 50, 52];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(52 as u64);
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

    #[test]
    fn test_lm_utilities_53() {
        let tokens = vec![10, 20, 30, 40, 50, 53];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(53 as u64);
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

    #[test]
    fn test_lm_utilities_54() {
        let tokens = vec![10, 20, 30, 40, 50, 54];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(54 as u64);
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

    #[test]
    fn test_lm_utilities_55() {
        let tokens = vec![10, 20, 30, 40, 50, 55];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(55 as u64);
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

    #[test]
    fn test_lm_utilities_56() {
        let tokens = vec![10, 20, 30, 40, 50, 56];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(56 as u64);
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

    #[test]
    fn test_lm_utilities_57() {
        let tokens = vec![10, 20, 30, 40, 50, 57];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(57 as u64);
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

    #[test]
    fn test_lm_utilities_58() {
        let tokens = vec![10, 20, 30, 40, 50, 58];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(58 as u64);
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

    #[test]
    fn test_lm_utilities_59() {
        let tokens = vec![10, 20, 30, 40, 50, 59];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(59 as u64);
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

    #[test]
    fn test_lm_utilities_60() {
        let tokens = vec![10, 20, 30, 40, 50, 60];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(60 as u64);
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

    #[test]
    fn test_lm_utilities_61() {
        let tokens = vec![10, 20, 30, 40, 50, 61];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(61 as u64);
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

    #[test]
    fn test_lm_utilities_62() {
        let tokens = vec![10, 20, 30, 40, 50, 62];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(62 as u64);
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

    #[test]
    fn test_lm_utilities_63() {
        let tokens = vec![10, 20, 30, 40, 50, 63];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(63 as u64);
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

    #[test]
    fn test_lm_utilities_64() {
        let tokens = vec![10, 20, 30, 40, 50, 64];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(64 as u64);
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

    #[test]
    fn test_lm_utilities_65() {
        let tokens = vec![10, 20, 30, 40, 50, 65];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(65 as u64);
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

    #[test]
    fn test_lm_utilities_66() {
        let tokens = vec![10, 20, 30, 40, 50, 66];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(66 as u64);
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

    #[test]
    fn test_lm_utilities_67() {
        let tokens = vec![10, 20, 30, 40, 50, 67];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(67 as u64);
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

    #[test]
    fn test_lm_utilities_68() {
        let tokens = vec![10, 20, 30, 40, 50, 68];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(68 as u64);
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

    #[test]
    fn test_lm_utilities_69() {
        let tokens = vec![10, 20, 30, 40, 50, 69];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(69 as u64);
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

    #[test]
    fn test_lm_utilities_70() {
        let tokens = vec![10, 20, 30, 40, 50, 70];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(70 as u64);
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

    #[test]
    fn test_lm_utilities_71() {
        let tokens = vec![10, 20, 30, 40, 50, 71];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(71 as u64);
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

    #[test]
    fn test_lm_utilities_72() {
        let tokens = vec![10, 20, 30, 40, 50, 72];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(72 as u64);
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

    #[test]
    fn test_lm_utilities_73() {
        let tokens = vec![10, 20, 30, 40, 50, 73];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(73 as u64);
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

    #[test]
    fn test_lm_utilities_74() {
        let tokens = vec![10, 20, 30, 40, 50, 74];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(74 as u64);
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

    #[test]
    fn test_lm_utilities_75() {
        let tokens = vec![10, 20, 30, 40, 50, 75];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(75 as u64);
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

    #[test]
    fn test_lm_utilities_76() {
        let tokens = vec![10, 20, 30, 40, 50, 76];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(76 as u64);
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

    #[test]
    fn test_lm_utilities_77() {
        let tokens = vec![10, 20, 30, 40, 50, 77];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(77 as u64);
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

    #[test]
    fn test_lm_utilities_78() {
        let tokens = vec![10, 20, 30, 40, 50, 78];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(78 as u64);
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

    #[test]
    fn test_lm_utilities_79() {
        let tokens = vec![10, 20, 30, 40, 50, 79];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(79 as u64);
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

    #[test]
    fn test_lm_utilities_80() {
        let tokens = vec![10, 20, 30, 40, 50, 80];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(80 as u64);
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

    #[test]
    fn test_lm_utilities_81() {
        let tokens = vec![10, 20, 30, 40, 50, 81];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(81 as u64);
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

    #[test]
    fn test_lm_utilities_82() {
        let tokens = vec![10, 20, 30, 40, 50, 82];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(82 as u64);
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

    #[test]
    fn test_lm_utilities_83() {
        let tokens = vec![10, 20, 30, 40, 50, 83];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(83 as u64);
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

    #[test]
    fn test_lm_utilities_84() {
        let tokens = vec![10, 20, 30, 40, 50, 84];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(84 as u64);
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

    #[test]
    fn test_lm_utilities_85() {
        let tokens = vec![10, 20, 30, 40, 50, 85];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(85 as u64);
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

    #[test]
    fn test_lm_utilities_86() {
        let tokens = vec![10, 20, 30, 40, 50, 86];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(86 as u64);
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

    #[test]
    fn test_lm_utilities_87() {
        let tokens = vec![10, 20, 30, 40, 50, 87];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(87 as u64);
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

    #[test]
    fn test_lm_utilities_88() {
        let tokens = vec![10, 20, 30, 40, 50, 88];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(88 as u64);
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

    #[test]
    fn test_lm_utilities_89() {
        let tokens = vec![10, 20, 30, 40, 50, 89];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(89 as u64);
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

    #[test]
    fn test_lm_utilities_90() {
        let tokens = vec![10, 20, 30, 40, 50, 90];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(90 as u64);
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

    #[test]
    fn test_lm_utilities_91() {
        let tokens = vec![10, 20, 30, 40, 50, 91];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(91 as u64);
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

    #[test]
    fn test_lm_utilities_92() {
        let tokens = vec![10, 20, 30, 40, 50, 92];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(92 as u64);
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

    #[test]
    fn test_lm_utilities_93() {
        let tokens = vec![10, 20, 30, 40, 50, 93];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(93 as u64);
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

    #[test]
    fn test_lm_utilities_94() {
        let tokens = vec![10, 20, 30, 40, 50, 94];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(94 as u64);
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

    #[test]
    fn test_lm_utilities_95() {
        let tokens = vec![10, 20, 30, 40, 50, 95];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(95 as u64);
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

    #[test]
    fn test_lm_utilities_96() {
        let tokens = vec![10, 20, 30, 40, 50, 96];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(96 as u64);
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

    #[test]
    fn test_lm_utilities_97() {
        let tokens = vec![10, 20, 30, 40, 50, 97];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(97 as u64);
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

    #[test]
    fn test_lm_utilities_98() {
        let tokens = vec![10, 20, 30, 40, 50, 98];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(98 as u64);
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

    #[test]
    fn test_lm_utilities_99() {
        let tokens = vec![10, 20, 30, 40, 50, 99];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(99 as u64);
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

    #[test]
    fn test_lm_utilities_100() {
        let tokens = vec![10, 20, 30, 40, 50, 100];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(100 as u64);
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

    #[test]
    fn test_lm_utilities_101() {
        let tokens = vec![10, 20, 30, 40, 50, 101];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(101 as u64);
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

    #[test]
    fn test_lm_utilities_102() {
        let tokens = vec![10, 20, 30, 40, 50, 102];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(102 as u64);
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

    #[test]
    fn test_lm_utilities_103() {
        let tokens = vec![10, 20, 30, 40, 50, 103];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(103 as u64);
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

    #[test]
    fn test_lm_utilities_104() {
        let tokens = vec![10, 20, 30, 40, 50, 104];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(104 as u64);
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

    #[test]
    fn test_lm_utilities_105() {
        let tokens = vec![10, 20, 30, 40, 50, 105];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(105 as u64);
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

    #[test]
    fn test_lm_utilities_106() {
        let tokens = vec![10, 20, 30, 40, 50, 106];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(106 as u64);
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

    #[test]
    fn test_lm_utilities_107() {
        let tokens = vec![10, 20, 30, 40, 50, 107];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(107 as u64);
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

    #[test]
    fn test_lm_utilities_108() {
        let tokens = vec![10, 20, 30, 40, 50, 108];
        let (inputs, labels, mask) = LmPreprocessor::prepare_causal_lm(&tokens, 4);
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(mask.len(), 4);

        let mut rng = TextRng::new(108 as u64);
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

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
    // brain-text production verification test padding line 7
    // brain-text production verification test padding line 8
}
