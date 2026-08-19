//! # FlashAttention-Lite: Block-Tiled Memory-Efficient Attention
//!
//! Tiling query and key blocks with online softmax rescaling, achieving $O(1)$ memory without allocating the full $N \times N$ attention matrix.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for FlashAttention-lite kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashLiteConfig {
    /// Query block tile size $B_r$ (typically 16 to 64).
    pub block_m: usize,
    /// Key/Value block tile size $B_c$ (typically 16 to 64).
    pub block_n: usize,
    /// Custom scale factor.
    pub scale: Option<f64>,
    /// Enforce causal masking.
    pub is_causal: bool,
}

impl Default for FlashLiteConfig {
    fn default() -> Self {
        Self {
            block_m: 16,
            block_n: 16,
            scale: None,
            is_causal: false,
        }
    }
}

/// FlashAttention-lite executor.
pub struct FlashAttentionLite;

impl FlashAttentionLite {
    /// Executes block-tiled memory-efficient attention for a single head `Q [seq_q, d]`, `K [seq_k, d]`, `V [seq_k, d_v]`.
    pub fn forward_head_tiled(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        v_dim: usize,
        config: &FlashLiteConfig,
    ) -> Vec<f64> {
        let scale = config.scale.unwrap_or(1.0 / (dim as f64).sqrt());
        let b_m = config.block_m.max(1);
        let b_n = config.block_n.max(1);

        let mut out = vec![0.0f64; seq_q * v_dim];
        let mut row_max = vec![f64::NEG_INFINITY; seq_q];
        let mut row_sum = vec![0.0f64; seq_q];

        let num_q_blocks = (seq_q + b_m - 1) / b_m;
        let num_k_blocks = (seq_k + b_n - 1) / b_n;

        for q_blk in 0..num_q_blocks {
            let q_start = q_blk * b_m;
            let q_end = (q_start + b_m).min(seq_q);
            let curr_m = q_end - q_start;

            for k_blk in 0..num_k_blocks {
                let k_start = k_blk * b_n;
                let k_end = (k_start + b_n).min(seq_k);
                let curr_n = k_end - k_start;

                // If causal and this key block is entirely in future of this query block, skip
                if config.is_causal && k_start > q_end - 1 {
                    continue;
                }

                // Compute block tile logits: S_ij = Q_i * K_j^T * scale
                for i in 0..curr_m {
                    let global_i = q_start + i;
                    let q_offset = global_i * dim;

                    let mut tile_scores = Vec::with_capacity(curr_n);
                    for j in 0..curr_n {
                        let global_j = k_start + j;
                        if config.is_causal && global_j > global_i {
                            tile_scores.push(f64::NEG_INFINITY);
                        } else {
                            let k_offset = global_j * dim;
                            let mut dot = 0.0f64;
                            for d in 0..dim {
                                dot += q[q_offset + d] * k[k_offset + d];
                            }
                            tile_scores.push(dot * scale);
                        }
                    }

                    // Online Softmax step
                    let tile_max = tile_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if tile_max > f64::NEG_INFINITY {
                        let prev_max = row_max[global_i];
                        let new_max = prev_max.max(tile_max);

                        let prev_sum = row_sum[global_i];
                        let alpha = if prev_max > f64::NEG_INFINITY {
                            (prev_max - new_max).exp()
                        } else {
                            0.0
                        };

                        let mut tile_sum = 0.0f64;
                        for s in tile_scores.iter_mut() {
                            if *s > f64::NEG_INFINITY {
                                let exp_val = (*s - new_max).exp();
                                *s = exp_val;
                                tile_sum += exp_val;
                            } else {
                                *s = 0.0;
                            }
                        }

                        let new_sum = prev_sum * alpha + tile_sum;

                        // Rescale existing output accumulator and add new tile contribution
                        let out_offset = global_i * v_dim;
                        for d in 0..v_dim {
                            let old_val = out[out_offset + d];
                            let mut tile_val = 0.0f64;
                            for j in 0..curr_n {
                                let global_j = k_start + j;
                                let v_offset = global_j * v_dim;
                                tile_val += tile_scores[j] * v[v_offset + d];
                            }
                            out[out_offset + d] = old_val * alpha + tile_val;
                        }

                        row_max[global_i] = new_max;
                        row_sum[global_i] = new_sum;
                    }
                }
            }

            // Normalize row sums
            for i in 0..curr_m {
                let global_i = q_start + i;
                let sum = row_sum[global_i];
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    let out_offset = global_i * v_dim;
                    for d in 0..v_dim {
                        out[out_offset + d] *= inv_sum;
                    }
                }
            }
        }

        out
    }

    /// Computes full 4D FlashAttention-lite forward pass `[batch_size, num_heads, seq_len, head_dim]`.
    pub fn forward(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: &FlashLiteConfig,
    ) -> TransformerResult<Tensor> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(TransformerError::DimensionMismatch {
                expected: 4,
                found: q_shape.len(),
            });
        }

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_q = q_shape[2];
        let head_dim = q_shape[3];

        let seq_k = k_shape[2];
        let v_head_dim = v_shape[3];

        let q_data = query.data();
        let k_data = key.data();
        let v_data = value.data();

        let mut out_data = vec![0.0f64; batch_size * num_heads * seq_q * v_head_dim];

        for b in 0..batch_size {
            for h in 0..num_heads {
                let q_offset = (b * num_heads + h) * seq_q * head_dim;
                let k_offset = (b * num_heads + h) * seq_k * head_dim;
                let v_offset = (b * num_heads + h) * seq_k * v_head_dim;
                let out_offset = (b * num_heads + h) * seq_q * v_head_dim;

                let q_head = &q_data[q_offset..q_offset + seq_q * head_dim];
                let k_head = &k_data[k_offset..k_offset + seq_k * head_dim];
                let v_head = &v_data[v_offset..v_offset + seq_k * v_head_dim];

                let head_out = Self::forward_head_tiled(
                    q_head,
                    k_head,
                    v_head,
                    seq_q,
                    seq_k,
                    head_dim,
                    v_head_dim,
                    config,
                );

                out_data[out_offset..out_offset + seq_q * v_head_dim].copy_from_slice(&head_out);
            }
        }

        Ok(Tensor::from_vec(out_data, vec![batch_size, num_heads, seq_q, v_head_dim]))
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
    fn test_flash_attention_lite_1() {
        let cfg = FlashLiteConfig { block_m: 2, block_n: 2, is_causal: false, scale: None };
        let q = Tensor::from_vec(vec![1.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);
        let k_t = Tensor::from_vec(vec![1.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);
        let v = Tensor::from_vec(vec![2.0; 1 * 1 * 4 * 8], vec![1, 1, 4, 8]);

        let flash_out = FlashAttentionLite::forward(&q, &k_t, &v, &cfg).unwrap();
        assert_eq!(flash_out.shape(), &[1, 1, 4, 8]);

        // Check equivalence with standard reference SDPA
        let (ref_out, _) = crate::attention::scaled::scaled_dot_product_attention(
            &q, &k_t, &v, &crate::core::AttentionMask::None, None
        ).unwrap();

        assert!(crate::utils::all_close(flash_out.data(), ref_out.data(), 1e-4, 1e-4));
    }
}
