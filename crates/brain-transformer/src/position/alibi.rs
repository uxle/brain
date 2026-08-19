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
}
