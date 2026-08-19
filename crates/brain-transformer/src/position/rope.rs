//! # Rotary Position Embedding (RoPE)
//!
//! Pairwise complex 2D rotation for 1D language sequences and 2D Vision Transformer spatial grids.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for Rotary Position Embedding (RoPE).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Head dimension $d_k$ (must be an even number).
    pub dim: usize,
    /// Maximum context sequence length.
    pub max_seq_len: usize,
    /// Base frequency hyperparameter (theta, e.g. 10000.0 or 500000.0).
    pub theta: f32,
    /// Frequency scaling factor for context window expansion.
    pub scaling_factor: f32,
    /// Enable 2D RoPE for spatial grids (Vision Transformer).
    pub is_2d: bool,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            max_seq_len: 2048,
            theta: 10000.0,
            scaling_factor: 1.0,
            is_2d: false,
        }
    }
}

/// Rotary Position Embedding engine.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cosine frequency matrix `[max_seq_len, dim / 2]`.
    pub cos_table: Vec<Vec<f64>>,
    /// Precomputed sine frequency matrix `[max_seq_len, dim / 2]`.
    pub sin_table: Vec<Vec<f64>>,
    /// Configuration options.
    pub config: RopeConfig,
}

impl RotaryEmbedding {
    /// Computes precomputed cosine and sine frequency matrices for RoPE.
    pub fn compute_frequencies(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
        scaling_factor: f32,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let half_dim = dim / 2;
        let mut cos_table = vec![vec![0.0f64; half_dim]; max_seq_len];
        let mut sin_table = vec![vec![0.0f64; half_dim]; max_seq_len];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / (theta as f64).powf((2 * i) as f64 / dim as f64);
                let scaled_pos = (pos as f64) / (scaling_factor as f64);
                let angle = scaled_pos * freq;
                cos_table[pos][i] = angle.cos();
                sin_table[pos][i] = angle.sin();
            }
        }

        (cos_table, sin_table)
    }

    /// Creates a new `RotaryEmbedding` layer.
    pub fn new(config: RopeConfig) -> Self {
        let (cos_table, sin_table) = Self::compute_frequencies(
            config.dim,
            config.max_seq_len,
            config.theta,
            config.scaling_factor,
        );
        Self {
            cos_table,
            sin_table,
            config,
        }
    }

    /// Applies 1D interleaved RoPE in-place to a 4D tensor `[batch_size, num_heads, seq_len, head_dim]`.
    pub fn apply_rope_4d(&self, tensor: &mut Tensor, pos_offset: usize) -> TransformerResult<()> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(TransformerError::DimensionMismatch {
                expected: 4,
                found: shape.len(),
            });
        }

        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        let data = tensor.data_mut();

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let pos = pos_offset + s;
                    if pos >= self.config.max_seq_len {
                        continue;
                    }

                    let offset = (b * num_heads + h) * seq_len * head_dim + s * head_dim;

                    for i in 0..half_dim {
                        let x1 = data[offset + i];
                        let x2 = data[offset + half_dim + i];

                        let c = self.cos_table[pos][i];
                        let s_val = self.sin_table[pos][i];

                        // Complex rotation: (x1 + i x2) * (cos + i sin)
                        let rot_x1 = x1 * c - x2 * s_val;
                        let rot_x2 = x1 * s_val + x2 * c;

                        data[offset + i] = rot_x1;
                        data[offset + half_dim + i] = rot_x2;
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies 2D RoPE for Vision Transformer grids of shape `[batch_size, num_heads, height * width, head_dim]`.
    pub fn apply_rope_2d(
        &self,
        tensor: &mut Tensor,
        height: usize,
        width: usize,
    ) -> TransformerResult<()> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(TransformerError::DimensionMismatch {
                expected: 4,
                found: shape.len(),
            });
        }

        let batch_size = shape[0];
        let num_heads = shape[1];
        let total_patches = height * width;
        let head_dim = shape[3];
        let quarter_dim = head_dim / 4;

        let data = tensor.data_mut();

        for b in 0..batch_size {
            for h in 0..num_heads {
                for r in 0..height {
                    for c in 0..width {
                        let patch_idx = r * width + c;
                        let offset = (b * num_heads + h) * total_patches * head_dim + patch_idx * head_dim;

                        // Height rotation on first half
                        for i in 0..quarter_dim {
                            let x1 = data[offset + i];
                            let x2 = data[offset + quarter_dim + i];
                            let cos_val = self.cos_table[r][i];
                            let sin_val = self.sin_table[r][i];
                            data[offset + i] = x1 * cos_val - x2 * sin_val;
                            data[offset + quarter_dim + i] = x1 * sin_val + x2 * cos_val;
                        }

                        // Width rotation on second half
                        let w_base = 2 * quarter_dim;
                        for i in 0..quarter_dim {
                            let x1 = data[offset + w_base + i];
                            let x2 = data[offset + w_base + quarter_dim + i];
                            let cos_val = self.cos_table[c][i];
                            let sin_val = self.sin_table[c][i];
                            data[offset + w_base + i] = x1 * cos_val - x2 * sin_val;
                            data[offset + w_base + quarter_dim + i] = x1 * sin_val + x2 * cos_val;
                        }
                    }
                }
            }
        }

        Ok(())
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
    fn test_rope_embedding_1() {
        let cfg = RopeConfig { dim: 16, max_seq_len: 64, theta: 10000.0, scaling_factor: 1.0, is_2d: false };
        let rope = RotaryEmbedding::new(cfg);
        assert_eq!(rope.cos_table.len(), 64);
        assert_eq!(rope.cos_table[0].len(), 8);

        let mut t = Tensor::from_vec(vec![1.0; 2 * 2 * 4 * 16], vec![2, 2, 4, 16]);
        rope.apply_rope_4d(&mut t, 0).unwrap();
        assert_eq!(t.shape(), &[2, 2, 4, 16]);

        let mut t2d = Tensor::from_vec(vec![1.0; 1 * 1 * 16 * 16], vec![1, 1, 16, 16]);
        rope.apply_rope_2d(&mut t2d, 4, 4).unwrap();
        assert_eq!(t2d.shape(), &[1, 1, 16, 16]);
    }
}
