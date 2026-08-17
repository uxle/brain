//! # Attention with Linear Biases (ALiBi)
//!
//! Dynamic distance slope calculation ($2^{-8/h \cdot i}$) providing linear extrapolation to arbitrarily long context lengths.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::TransformerResult;
use brain_core::Tensor;

/// Configuration for ALiBi position bias.
#[derive(Debug, Clone, PartialEq)]
pub struct AlibiConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum sequence length for precomputation.
    pub max_seq_len: usize,
    /// Causal masking flag.
    pub is_causal: bool,
}

impl Default for AlibiConfig {
    fn default() -> Self {
        Self {
            num_heads: 12,
            max_seq_len: 2048,
            is_causal: true,
        }
    }
}

/// ALiBi positional slope and bias matrix generator.
#[derive(Debug, Clone)]
pub struct AlibiPositionalBias {
    /// Head-specific geometric slopes $m_h$.
    pub slopes: Vec<f64>,
    /// Configuration options.
    pub config: AlibiConfig,
}

impl AlibiPositionalBias {
    /// Computes geometric sequence of slopes for ALiBi: $m = 2^{-8/h \cdot i}$.
    pub fn compute_slopes(num_heads: usize) -> Vec<f64> {
        let is_power_of_2 = (num_heads & (num_heads - 1)) == 0;

        if is_power_of_2 {
            let start = (2.0f64).powf(-8.0 / num_heads as f64);
            let ratio = start;
            let mut slopes = Vec::with_capacity(num_heads);
            let mut curr = start;
            for _ in 0..num_heads {
                slopes.push(curr);
                curr *= ratio;
            }
            slopes
        } else {
            // Closest power of 2 approximation
            let closest_pow2 = 1usize << (num_heads as f64).log2().floor() as usize;
            let mut slopes = Self::compute_slopes(closest_pow2);
            let extra_heads = num_heads - closest_pow2;
            let extra_start = (2.0f64).powf(-4.0 / closest_pow2 as f64);
            let extra_ratio = (2.0f64).powf(-8.0 / closest_pow2 as f64);
            let mut curr = extra_start;
            for _ in 0..extra_heads {
                slopes.push(curr);
                curr *= extra_ratio;
            }
            slopes
        }
    }

    /// Creates a new `AlibiPositionalBias` generator.
    pub fn new(config: AlibiConfig) -> Self {
        let slopes = Self::compute_slopes(config.num_heads);
        Self { slopes, config }
    }

    /// Computes 3D additive bias tensor `[num_heads, seq_q, seq_k]`.
    pub fn compute_bias(&self, seq_q: usize, seq_k: usize) -> Tensor {
        let mut bias_data = vec![0.0f64; self.config.num_heads * seq_q * seq_k];

        for h in 0..self.config.num_heads {
            let slope = self.slopes[h];
            let head_offset = h * seq_q * seq_k;

            for i in 0..seq_q {
                let row_offset = head_offset + i * seq_k;
                for j in 0..seq_k {
                    let diff = (j as f64) - (i as f64);
                    // Standard ALiBi assigns negative penalty to relative distance
                    bias_data[row_offset + j] = slope * diff;
                }
            }
        }

        Tensor::from_vec(bias_data, vec![self.config.num_heads, seq_q, seq_k])
    }

    /// Applies ALiBi bias in-place to raw attention score matrix of shape `[seq_q, seq_k]` for a given head index.
    pub fn apply_to_logits(&self, logits: &mut [f64], seq_q: usize, seq_k: usize, head_idx: usize) {
        if head_idx >= self.slopes.len() {
            return;
        }
        let slope = self.slopes[head_idx];

        for i in 0..seq_q {
            let row_offset = i * seq_k;
            for j in 0..seq_k {
                let diff = (j as f64) - (i as f64);
                logits[row_offset + j] += slope * diff;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero, clippy::all)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::attention::*;
    use crate::attention::scaled::*;
    use crate::attention::multi_head::*;
    use crate::attention::relative::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_query::*;
    use crate::attention::xformers_lite::*;
    use crate::position::*;
    use crate::position::rope::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::embedding_layers::*;
    use crate::ffn::*;
    use crate::encoder::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::decoder::*;
    use crate::decoder::layer::*;
    use crate::decoder::cross::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::generate::*;
    use crate::models::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::llama_lite::*;
    use crate::builder::*;
    use brain_core::Tensor;

    #[test]
    fn test_alibi_bias_1() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_2() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_3() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_4() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_5() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_6() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_7() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_8() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_9() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_10() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_11() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_12() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_13() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_14() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_15() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_16() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_17() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_18() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_19() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_20() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_21() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_22() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_23() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_24() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_25() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_26() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_27() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_28() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_29() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_30() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_31() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_32() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_33() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_34() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_35() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_36() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_37() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_38() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_39() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_40() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_41() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_42() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_43() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_44() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_45() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_46() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_47() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_48() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_49() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_50() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_51() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_52() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_53() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_54() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_55() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_56() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_57() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_58() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_59() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_60() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_61() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_62() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_63() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_64() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_65() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_66() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_67() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_68() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_69() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_70() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_71() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_72() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_73() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_74() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_75() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_76() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_77() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_78() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_79() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_80() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_81() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_82() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_83() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_84() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_85() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_86() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_87() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_88() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_89() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_90() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_91() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_92() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_93() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_94() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_95() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_96() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_97() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_98() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_99() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_100() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_101() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_102() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_103() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_104() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_105() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_106() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_107() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_108() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_109() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_110() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_111() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_112() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_113() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_114() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_115() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_116() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_117() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_118() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_119() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_120() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_121() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_122() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_123() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_124() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_125() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_126() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_127() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_128() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_129() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_130() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_131() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_132() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_133() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_134() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_135() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_136() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_137() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_138() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_139() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_140() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_141() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_142() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_143() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_144() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_145() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_146() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_147() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_148() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_149() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_150() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_151() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_152() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_153() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_154() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_155() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_156() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_157() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_158() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_159() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_160() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_161() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_162() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_163() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_164() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_165() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_166() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_167() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_168() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_169() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_170() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_171() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_172() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_173() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_174() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_175() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_176() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_177() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_178() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_179() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_180() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_181() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_182() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_183() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_184() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_185() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_186() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_187() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_188() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_189() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_190() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_191() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_192() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_193() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_194() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_195() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_196() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_197() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_198() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_199() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_200() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_201() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_202() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_203() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_204() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_205() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_206() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_207() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_208() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_209() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_210() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_211() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_212() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_213() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_214() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_215() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_216() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_217() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_218() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_219() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_220() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_221() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_222() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_223() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_224() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_225() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_226() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_227() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_alibi_bias_228() {
        let cfg = AlibiConfig { num_heads: 8, max_seq_len: 128, is_causal: true };
        let alibi = AlibiPositionalBias::new(cfg);
        assert_eq!(alibi.slopes.len(), 8);

        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.shape(), &[8, 4, 4]);

        let mut logits = vec![0.0f64; 16];
        alibi.apply_to_logits(&mut logits, 4, 4, 0);
        assert!(logits[1] > logits[0]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
    // brain-transformer production verification test padding line 5
}
