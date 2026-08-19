//! # ViT Transformer Blocks for brain-vit
//!
//! Each block consists of:
//! 1. Pre-LayerNorm
//! 2. Multi-head self-attention (simplified single-head per batch for std-only)
//! 3. Residual + Pre-LayerNorm
//! 4. MLP (two linear layers with GELU)
//! 5. Residual + optional stochastic depth

use crate::core::{VitError, VitResult, Tensor2D, Tensor3D, SimpleRng};
use crate::config::VitBlockConfig;
use crate::ops::{layer_norm_2d, mlp_forward, linear, scaled_dot_product_attention};

/// A single ViT transformer block.
///
/// # Example
/// ```rust
/// use brain_vit::vit::blocks::VitBlock;
/// use brain_vit::config::VitBlockConfig;
/// use brain_vit::core::Tensor3D;
/// let cfg = VitBlockConfig { embed_dim: 16, num_heads: 2, mlp_ratio: 2.0, ..Default::default() };
/// let block = VitBlock::new(&cfg, 0).unwrap();
/// let tokens = Tensor3D::zeros(1, 5, 16);
/// let out = block.forward(&tokens).unwrap();
/// assert_eq!(out.batch, 1);
/// ```
pub struct VitBlock {
    /// Block configuration.
    pub config: VitBlockConfig,
    /// QKV projection weight `[3 * embed_dim, embed_dim]`.
    pub qkv_w: Vec<f64>,
    /// QKV projection bias `[3 * embed_dim]`.
    pub qkv_b: Vec<f64>,
    /// Output projection weight `[embed_dim, embed_dim]`.
    pub out_w: Vec<f64>,
    /// MLP fc1 weight `[mlp_dim, embed_dim]`.
    pub mlp1_w: Vec<f64>,
    /// MLP fc1 bias `[mlp_dim]`.
    pub mlp1_b: Vec<f64>,
    /// MLP fc2 weight `[embed_dim, mlp_dim]`.
    pub mlp2_w: Vec<f64>,
    /// MLP fc2 bias `[embed_dim]`.
    pub mlp2_b: Vec<f64>,
    /// Random number generator for stochastic depth.
    pub rng: SimpleRng,
    /// Whether block is in training mode.
    pub training: bool,
}

impl VitBlock {
    /// Create a new ViT block with Xavier initialization.
    pub fn new(config: &VitBlockConfig, seed: u64) -> VitResult<Self> {
        config.validate()?;
        let mut rng = SimpleRng::new(seed);
        let d = config.embed_dim;
        let mlp_dim = config.mlp_dim();

        let qkv_w = rng.xavier_uniform(3 * d, d);
        let qkv_b = if config.qkv_bias { rng.gen_vec(3 * d, 0.0, 0.0) } else { vec![0.0; 3 * d] };
        let out_w = rng.xavier_uniform(d, d);
        let mlp1_w = rng.xavier_uniform(mlp_dim, d);
        let mlp1_b = vec![0.0f64; mlp_dim];
        let mlp2_w = rng.xavier_uniform(d, mlp_dim);
        let mlp2_b = vec![0.0f64; d];

        Ok(Self {
            config: config.clone(),
            qkv_w, qkv_b, out_w,
            mlp1_w, mlp1_b, mlp2_w, mlp2_b,
            rng,
            training: true,
        })
    }

    /// Forward pass on `[B, N, D]` token tensor.
    pub fn forward(&self, tokens: &Tensor3D) -> VitResult<Tensor3D> {
        let (batch, seq_len, embed_dim) = (tokens.batch, tokens.seq, tokens.dim);
        if embed_dim != self.config.embed_dim {
            return Err(VitError::DimMismatch { expected: self.config.embed_dim, got: embed_dim });
        }

        let head_dim = self.config.head_dim();
        let mlp_dim = self.config.mlp_dim();
        let eps = self.config.layer_norm_eps;
        let activation = self.config.activation;

        let qkv_w = Tensor2D::from_data(3 * embed_dim, embed_dim, self.qkv_w.clone())?;
        let out_w_mat = Tensor2D::from_data(embed_dim, embed_dim, self.out_w.clone())?;
        let mlp1_w_mat = Tensor2D::from_data(mlp_dim, embed_dim, self.mlp1_w.clone())?;
        let mlp2_w_mat = Tensor2D::from_data(embed_dim, mlp_dim, self.mlp2_w.clone())?;

        let mut out_data = vec![0.0f64; batch * seq_len * embed_dim];

        for b in 0..batch {
            let tok_data: Vec<f64> = (0..seq_len).flat_map(|s|
                tokens.data[b * seq_len * embed_dim + s * embed_dim
                    ..b * seq_len * embed_dim + (s + 1) * embed_dim].iter().copied()
            ).collect();
            let tok_mat = Tensor2D::from_data(seq_len, embed_dim, tok_data)?;

            // Pre-LN
            let normed = layer_norm_2d(&tok_mat, eps);

            // QKV projection
            let qkv = linear(&normed, &qkv_w, Some(&self.qkv_b))?;

            // Use first head's Q, K, V (simplified)
            let q_data: Vec<f64> = (0..seq_len)
                .flat_map(|s| qkv.data[s * 3 * embed_dim..s * 3 * embed_dim + head_dim].iter().copied())
                .collect();
            let k_data: Vec<f64> = (0..seq_len)
                .flat_map(|s| qkv.data[s * 3 * embed_dim + embed_dim
                    ..s * 3 * embed_dim + embed_dim + head_dim].iter().copied())
                .collect();
            let v_data: Vec<f64> = (0..seq_len)
                .flat_map(|s| qkv.data[s * 3 * embed_dim + 2 * embed_dim
                    ..s * 3 * embed_dim + 2 * embed_dim + head_dim].iter().copied())
                .collect();

            let q = Tensor2D::from_data(seq_len, head_dim, q_data)?;
            let k = Tensor2D::from_data(seq_len, head_dim, k_data)?;
            let v = Tensor2D::from_data(seq_len, head_dim, v_data)?;
            let (attn_out, _) = scaled_dot_product_attention(&q, &k, &v)?;

            // Pad attn output to full embed_dim (repeat head tile)
            let mut attn_full = vec![0.0f64; seq_len * embed_dim];
            for s in 0..seq_len {
                for d in 0..embed_dim {
                    attn_full[s * embed_dim + d] = attn_out.get(s, d % head_dim);
                }
            }
            let attn_mat = Tensor2D::from_data(seq_len, embed_dim, attn_full)?;
            let projected = linear(&attn_mat, &out_w_mat, None)?;

            // Residual 1
            let after_attn = tok_mat.add(&projected)?;

            // Pre-LN 2
            let normed2 = layer_norm_2d(&after_attn, eps);

            // MLP
            let mlp_out = mlp_forward(
                &normed2,
                &mlp1_w_mat, &self.mlp1_b,
                &mlp2_w_mat, &self.mlp2_b,
                &activation,
            )?;

            // Residual 2
            let final_tok = after_attn.add(&mlp_out)?;
            let dst = b * seq_len * embed_dim;
            out_data[dst..dst + seq_len * embed_dim].copy_from_slice(&final_tok.data);
        }

        Tensor3D::from_data(batch, seq_len, embed_dim, out_data)
    }

    /// Parameter count of this block.
    pub fn num_params(&self) -> usize {
        self.qkv_w.len() + self.qkv_b.len() + self.out_w.len()
            + self.mlp1_w.len() + self.mlp1_b.len()
            + self.mlp2_w.len() + self.mlp2_b.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{VitBlockConfig, Activation};
    use crate::core::Tensor3D;

    fn small_cfg() -> VitBlockConfig {
        VitBlockConfig {
            embed_dim: 16,
            num_heads: 2,
            mlp_ratio: 2.0,
            attn_dropout: 0.0,
            mlp_dropout: 0.0,
            drop_path_rate: 0.0,
            activation: Activation::Relu,
            layer_norm_eps: 1e-5,
            qkv_bias: true,
        }
    }

    #[test]
    fn test_block_new() {
        let b = VitBlock::new(&small_cfg(), 0).unwrap();
        assert_eq!(b.config.embed_dim, 16);
        assert_eq!(b.config.mlp_dim(), 32);
    }

    #[test]
    fn test_block_forward_shape() {
        let block = VitBlock::new(&small_cfg(), 0).unwrap();
        let t = Tensor3D::zeros(2, 5, 16);
        let out = block.forward(&t).unwrap();
        assert_eq!(out.batch, 2);
        assert_eq!(out.seq, 5);
        assert_eq!(out.dim, 16);
    }

    #[test]
    fn test_block_forward_finite() {
        let block = VitBlock::new(&small_cfg(), 1).unwrap();
        let t = Tensor3D::from_data(1, 5, 16, (0..80).map(|x| x as f64 * 0.01).collect()).unwrap();
        let out = block.forward(&t).unwrap();
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_block_residual_preserves_shape() {
        let block = VitBlock::new(&small_cfg(), 2).unwrap();
        let t = Tensor3D::zeros(3, 4, 16);
        let out = block.forward(&t).unwrap();
        assert_eq!(out.data.len(), 3 * 4 * 16);
    }

    #[test]
    fn test_block_deterministic() {
        let b1 = VitBlock::new(&small_cfg(), 42).unwrap();
        let b2 = VitBlock::new(&small_cfg(), 42).unwrap();
        let t = Tensor3D::from_data(1, 3, 16, vec![0.1f64; 48]).unwrap();
        let o1 = b1.forward(&t).unwrap();
        let o2 = b2.forward(&t).unwrap();
        for (a, b) in o1.data.iter().zip(o2.data.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_block_dim_mismatch_err() {
        let block = VitBlock::new(&small_cfg(), 0).unwrap();
        let t = Tensor3D::zeros(1, 5, 32); // wrong dim
        assert!(block.forward(&t).is_err());
    }

    #[test]
    fn test_block_num_params_positive() {
        let b = VitBlock::new(&small_cfg(), 0).unwrap();
        assert!(b.num_params() > 0);
    }

    #[test]
    fn test_block_invalid_config() {
        let mut cfg = small_cfg();
        cfg.num_heads = 3; // 16 % 3 != 0
        assert!(VitBlock::new(&cfg, 0).is_err());
    }

    #[test]
    fn test_block_single_token() {
        let block = VitBlock::new(&small_cfg(), 0).unwrap();
        let t = Tensor3D::zeros(1, 1, 16);
        let out = block.forward(&t).unwrap();
        assert_eq!(out.seq, 1);
    }

    #[test]
    fn test_block_many_tokens() {
        let block = VitBlock::new(&small_cfg(), 0).unwrap();
        let t = Tensor3D::zeros(1, 100, 16);
        let out = block.forward(&t).unwrap();
        assert_eq!(out.seq, 100);
    }

    #[test]
    fn test_block_batch_independence() {
        let block = VitBlock::new(&small_cfg(), 0).unwrap();
        let single = vec![0.5f64; 1 * 5 * 16];
        let double = vec![0.5f64; 2 * 5 * 16];
        let t1 = Tensor3D::from_data(1, 5, 16, single).unwrap();
        let t2 = Tensor3D::from_data(2, 5, 16, double).unwrap();
        let o1 = block.forward(&t1).unwrap();
        let o2 = block.forward(&t2).unwrap();
        for (a, b) in o1.data.iter().zip(o2.data[..80].iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_block_gelu_activation() {
        let mut cfg = small_cfg();
        cfg.activation = Activation::Gelu;
        let block = VitBlock::new(&cfg, 0).unwrap();
        let t = Tensor3D::zeros(1, 4, 16);
        let out = block.forward(&t).unwrap();
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_block_silu_activation() {
        let mut cfg = small_cfg();
        cfg.activation = Activation::Silu;
        let block = VitBlock::new(&cfg, 0).unwrap();
        let t = Tensor3D::zeros(1, 3, 16);
        let out = block.forward(&t).unwrap();
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_block_different_seeds_differ() {
        let b1 = VitBlock::new(&small_cfg(), 10).unwrap();
        let b2 = VitBlock::new(&small_cfg(), 20).unwrap();
        assert_ne!(b1.qkv_w, b2.qkv_w);
    }

    #[test]
    fn test_block_training_flag() {
        let mut b = VitBlock::new(&small_cfg(), 0).unwrap();
        assert!(b.training);
        b.training = false;
        assert!(!b.training);
    }

    #[test]
    fn test_block_no_qkv_bias() {
        let mut cfg = small_cfg();
        cfg.qkv_bias = false;
        let block = VitBlock::new(&cfg, 0).unwrap();
        assert!(block.qkv_b.iter().all(|&v| v == 0.0));
    }
}
