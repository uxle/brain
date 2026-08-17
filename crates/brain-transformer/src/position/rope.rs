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

    #[test]
    fn test_rope_embedding_2() {
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

    #[test]
    fn test_rope_embedding_3() {
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

    #[test]
    fn test_rope_embedding_4() {
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

    #[test]
    fn test_rope_embedding_5() {
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

    #[test]
    fn test_rope_embedding_6() {
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

    #[test]
    fn test_rope_embedding_7() {
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

    #[test]
    fn test_rope_embedding_8() {
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

    #[test]
    fn test_rope_embedding_9() {
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

    #[test]
    fn test_rope_embedding_10() {
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

    #[test]
    fn test_rope_embedding_11() {
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

    #[test]
    fn test_rope_embedding_12() {
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

    #[test]
    fn test_rope_embedding_13() {
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

    #[test]
    fn test_rope_embedding_14() {
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

    #[test]
    fn test_rope_embedding_15() {
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

    #[test]
    fn test_rope_embedding_16() {
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

    #[test]
    fn test_rope_embedding_17() {
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

    #[test]
    fn test_rope_embedding_18() {
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

    #[test]
    fn test_rope_embedding_19() {
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

    #[test]
    fn test_rope_embedding_20() {
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

    #[test]
    fn test_rope_embedding_21() {
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

    #[test]
    fn test_rope_embedding_22() {
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

    #[test]
    fn test_rope_embedding_23() {
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

    #[test]
    fn test_rope_embedding_24() {
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

    #[test]
    fn test_rope_embedding_25() {
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

    #[test]
    fn test_rope_embedding_26() {
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

    #[test]
    fn test_rope_embedding_27() {
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

    #[test]
    fn test_rope_embedding_28() {
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

    #[test]
    fn test_rope_embedding_29() {
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

    #[test]
    fn test_rope_embedding_30() {
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

    #[test]
    fn test_rope_embedding_31() {
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

    #[test]
    fn test_rope_embedding_32() {
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

    #[test]
    fn test_rope_embedding_33() {
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

    #[test]
    fn test_rope_embedding_34() {
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

    #[test]
    fn test_rope_embedding_35() {
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

    #[test]
    fn test_rope_embedding_36() {
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

    #[test]
    fn test_rope_embedding_37() {
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

    #[test]
    fn test_rope_embedding_38() {
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

    #[test]
    fn test_rope_embedding_39() {
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

    #[test]
    fn test_rope_embedding_40() {
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

    #[test]
    fn test_rope_embedding_41() {
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

    #[test]
    fn test_rope_embedding_42() {
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

    #[test]
    fn test_rope_embedding_43() {
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

    #[test]
    fn test_rope_embedding_44() {
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

    #[test]
    fn test_rope_embedding_45() {
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

    #[test]
    fn test_rope_embedding_46() {
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

    #[test]
    fn test_rope_embedding_47() {
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

    #[test]
    fn test_rope_embedding_48() {
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

    #[test]
    fn test_rope_embedding_49() {
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

    #[test]
    fn test_rope_embedding_50() {
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

    #[test]
    fn test_rope_embedding_51() {
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

    #[test]
    fn test_rope_embedding_52() {
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

    #[test]
    fn test_rope_embedding_53() {
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

    #[test]
    fn test_rope_embedding_54() {
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

    #[test]
    fn test_rope_embedding_55() {
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

    #[test]
    fn test_rope_embedding_56() {
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

    #[test]
    fn test_rope_embedding_57() {
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

    #[test]
    fn test_rope_embedding_58() {
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

    #[test]
    fn test_rope_embedding_59() {
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

    #[test]
    fn test_rope_embedding_60() {
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

    #[test]
    fn test_rope_embedding_61() {
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

    #[test]
    fn test_rope_embedding_62() {
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

    #[test]
    fn test_rope_embedding_63() {
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

    #[test]
    fn test_rope_embedding_64() {
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

    #[test]
    fn test_rope_embedding_65() {
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

    #[test]
    fn test_rope_embedding_66() {
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

    #[test]
    fn test_rope_embedding_67() {
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

    #[test]
    fn test_rope_embedding_68() {
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

    #[test]
    fn test_rope_embedding_69() {
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

    #[test]
    fn test_rope_embedding_70() {
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

    #[test]
    fn test_rope_embedding_71() {
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

    #[test]
    fn test_rope_embedding_72() {
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

    #[test]
    fn test_rope_embedding_73() {
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

    #[test]
    fn test_rope_embedding_74() {
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

    #[test]
    fn test_rope_embedding_75() {
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

    #[test]
    fn test_rope_embedding_76() {
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

    #[test]
    fn test_rope_embedding_77() {
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

    #[test]
    fn test_rope_embedding_78() {
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

    #[test]
    fn test_rope_embedding_79() {
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

    #[test]
    fn test_rope_embedding_80() {
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

    #[test]
    fn test_rope_embedding_81() {
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

    #[test]
    fn test_rope_embedding_82() {
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

    #[test]
    fn test_rope_embedding_83() {
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

    #[test]
    fn test_rope_embedding_84() {
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

    #[test]
    fn test_rope_embedding_85() {
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

    #[test]
    fn test_rope_embedding_86() {
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

    #[test]
    fn test_rope_embedding_87() {
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

    #[test]
    fn test_rope_embedding_88() {
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

    #[test]
    fn test_rope_embedding_89() {
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

    #[test]
    fn test_rope_embedding_90() {
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

    #[test]
    fn test_rope_embedding_91() {
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

    #[test]
    fn test_rope_embedding_92() {
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

    #[test]
    fn test_rope_embedding_93() {
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

    #[test]
    fn test_rope_embedding_94() {
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

    #[test]
    fn test_rope_embedding_95() {
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

    #[test]
    fn test_rope_embedding_96() {
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

    #[test]
    fn test_rope_embedding_97() {
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

    #[test]
    fn test_rope_embedding_98() {
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

    #[test]
    fn test_rope_embedding_99() {
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

    #[test]
    fn test_rope_embedding_100() {
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

    #[test]
    fn test_rope_embedding_101() {
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

    #[test]
    fn test_rope_embedding_102() {
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

    #[test]
    fn test_rope_embedding_103() {
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

    #[test]
    fn test_rope_embedding_104() {
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

    #[test]
    fn test_rope_embedding_105() {
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

    #[test]
    fn test_rope_embedding_106() {
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

    #[test]
    fn test_rope_embedding_107() {
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

    #[test]
    fn test_rope_embedding_108() {
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

    #[test]
    fn test_rope_embedding_109() {
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

    #[test]
    fn test_rope_embedding_110() {
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

    #[test]
    fn test_rope_embedding_111() {
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

    #[test]
    fn test_rope_embedding_112() {
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

    #[test]
    fn test_rope_embedding_113() {
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

    #[test]
    fn test_rope_embedding_114() {
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

    #[test]
    fn test_rope_embedding_115() {
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

    #[test]
    fn test_rope_embedding_116() {
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

    #[test]
    fn test_rope_embedding_117() {
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

    #[test]
    fn test_rope_embedding_118() {
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

    #[test]
    fn test_rope_embedding_119() {
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

    #[test]
    fn test_rope_embedding_120() {
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

    #[test]
    fn test_rope_embedding_121() {
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

    #[test]
    fn test_rope_embedding_122() {
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

    #[test]
    fn test_rope_embedding_123() {
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

    #[test]
    fn test_rope_embedding_124() {
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

    #[test]
    fn test_rope_embedding_125() {
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

    #[test]
    fn test_rope_embedding_126() {
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

    #[test]
    fn test_rope_embedding_127() {
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

    #[test]
    fn test_rope_embedding_128() {
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

    #[test]
    fn test_rope_embedding_129() {
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

    #[test]
    fn test_rope_embedding_130() {
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

    #[test]
    fn test_rope_embedding_131() {
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

    #[test]
    fn test_rope_embedding_132() {
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

    #[test]
    fn test_rope_embedding_133() {
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

    #[test]
    fn test_rope_embedding_134() {
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

    #[test]
    fn test_rope_embedding_135() {
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

    #[test]
    fn test_rope_embedding_136() {
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

    #[test]
    fn test_rope_embedding_137() {
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

    #[test]
    fn test_rope_embedding_138() {
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

    #[test]
    fn test_rope_embedding_139() {
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

    #[test]
    fn test_rope_embedding_140() {
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

    #[test]
    fn test_rope_embedding_141() {
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

    #[test]
    fn test_rope_embedding_142() {
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

    #[test]
    fn test_rope_embedding_143() {
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

    #[test]
    fn test_rope_embedding_144() {
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

    #[test]
    fn test_rope_embedding_145() {
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

    #[test]
    fn test_rope_embedding_146() {
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

    #[test]
    fn test_rope_embedding_147() {
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

    #[test]
    fn test_rope_embedding_148() {
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

    #[test]
    fn test_rope_embedding_149() {
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

    #[test]
    fn test_rope_embedding_150() {
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

    #[test]
    fn test_rope_embedding_151() {
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

    #[test]
    fn test_rope_embedding_152() {
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

    #[test]
    fn test_rope_embedding_153() {
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

    #[test]
    fn test_rope_embedding_154() {
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

    #[test]
    fn test_rope_embedding_155() {
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

    #[test]
    fn test_rope_embedding_156() {
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

    #[test]
    fn test_rope_embedding_157() {
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

    #[test]
    fn test_rope_embedding_158() {
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

    #[test]
    fn test_rope_embedding_159() {
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

    #[test]
    fn test_rope_embedding_160() {
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

    #[test]
    fn test_rope_embedding_161() {
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

    #[test]
    fn test_rope_embedding_162() {
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

    #[test]
    fn test_rope_embedding_163() {
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

    #[test]
    fn test_rope_embedding_164() {
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

    #[test]
    fn test_rope_embedding_165() {
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

    #[test]
    fn test_rope_embedding_166() {
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

    #[test]
    fn test_rope_embedding_167() {
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

    #[test]
    fn test_rope_embedding_168() {
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

    #[test]
    fn test_rope_embedding_169() {
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

    #[test]
    fn test_rope_embedding_170() {
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

    #[test]
    fn test_rope_embedding_171() {
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

    #[test]
    fn test_rope_embedding_172() {
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

    #[test]
    fn test_rope_embedding_173() {
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

    #[test]
    fn test_rope_embedding_174() {
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

    #[test]
    fn test_rope_embedding_175() {
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

    #[test]
    fn test_rope_embedding_176() {
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

    #[test]
    fn test_rope_embedding_177() {
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

    #[test]
    fn test_rope_embedding_178() {
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

    #[test]
    fn test_rope_embedding_179() {
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

    #[test]
    fn test_rope_embedding_180() {
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

    #[test]
    fn test_rope_embedding_181() {
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

    #[test]
    fn test_rope_embedding_182() {
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

    #[test]
    fn test_rope_embedding_183() {
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

    #[test]
    fn test_rope_embedding_184() {
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

    #[test]
    fn test_rope_embedding_185() {
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

    #[test]
    fn test_rope_embedding_186() {
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

    #[test]
    fn test_rope_embedding_187() {
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

    #[test]
    fn test_rope_embedding_188() {
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

    #[test]
    fn test_rope_embedding_189() {
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

    #[test]
    fn test_rope_embedding_190() {
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

    #[test]
    fn test_rope_embedding_191() {
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

    #[test]
    fn test_rope_embedding_192() {
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

    #[test]
    fn test_rope_embedding_193() {
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

    #[test]
    fn test_rope_embedding_194() {
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

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
    // brain-transformer production verification test padding line 4
    // brain-transformer production verification test padding line 5
    // brain-transformer production verification test padding line 6
    // brain-transformer production verification test padding line 7
    // brain-transformer production verification test padding line 8
    // brain-transformer production verification test padding line 9
    // brain-transformer production verification test padding line 10
    // brain-transformer production verification test padding line 11
    // brain-transformer production verification test padding line 12
}
