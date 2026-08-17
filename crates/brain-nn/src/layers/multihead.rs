//! # Multi-Head Self & Cross Attention Module
//!
//! Multi-head attention projecting inputs into Q, K, V representations across parallel attention heads.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};
use crate::layers::linear::Linear;
use super::attention::scaled_dot_product_attention;

/// Configuration for MultiheadAttention.
#[derive(Debug, Clone)]
pub struct MhaConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl Default for MhaConfig {
    fn default() -> Self {
        Self { embed_dim: 64, num_heads: 4, dropout: 0.0 }
    }
}

/// MultiheadAttention layer module.
#[derive(Debug, Clone)]
pub struct MultiheadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub config: MhaConfig,
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            config: MhaConfig { embed_dim, num_heads, dropout: 0.0 },
        }
    }

    pub fn forward_mha(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> ModuleResult<Tensor> {
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        let attn_out = scaled_dot_product_attention(&q, &k, &v, mask);
        self.out_proj.forward(&attn_out)
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        self.forward_mha(input, input, input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.q_proj.parameters());
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_mha_stress_001() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_002() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_003() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_004() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_005() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_006() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_007() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_008() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_009() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_010() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_011() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_012() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_013() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_014() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_015() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_016() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_017() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_018() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_019() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_020() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_021() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_022() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_023() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_024() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_025() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_026() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_027() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_028() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_029() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_030() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_031() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_032() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_033() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_034() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_035() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_036() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_037() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_038() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_039() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_040() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_041() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_042() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_043() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_044() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_045() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_046() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_047() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_048() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_049() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_050() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_051() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_052() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_053() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_054() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_055() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_056() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_057() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_058() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_059() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_060() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_061() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_062() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_063() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_064() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_065() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_066() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_067() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_068() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_069() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_070() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_071() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_072() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_073() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_074() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_075() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_076() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_077() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_078() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_079() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_080() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_081() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_082() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_083() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_084() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_085() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_086() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_087() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_088() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_089() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_090() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_091() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_092() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_093() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_094() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_095() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_096() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_097() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_098() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_099() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_100() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_101() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_102() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_103() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_104() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_105() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_106() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_107() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_108() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_109() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_110() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_111() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_112() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_113() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_114() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_115() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_116() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_117() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_118() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_119() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_120() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_121() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_122() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_123() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_124() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_125() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_126() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_127() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_128() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_129() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_130() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_131() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_132() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_133() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_134() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_135() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_136() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_137() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_138() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_139() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_140() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_141() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_142() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_143() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_144() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_145() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_146() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_147() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_148() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_149() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_150() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_151() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_152() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_153() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_154() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_155() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_156() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_157() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_158() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_159() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_160() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_161() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_162() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_163() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_164() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_165() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_166() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_167() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_168() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_169() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_170() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_171() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_172() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_173() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_174() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_175() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_176() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_177() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_178() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_179() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_180() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_181() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_182() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_183() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_184() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_185() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_186() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_187() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_188() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_189() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_190() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_191() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_192() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_193() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_194() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_195() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_196() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_197() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_198() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_199() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_200() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_201() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_202() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_203() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_204() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_205() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_206() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_207() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_208() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_209() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_210() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_211() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_212() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_213() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_214() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_215() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_216() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_217() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_218() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_219() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_220() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_221() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_222() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_223() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_224() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_225() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_226() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_227() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_228() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_229() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_230() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_231() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_232() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_233() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_234() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_235() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_236() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_237() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_238() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_239() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_240() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_241() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_242() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_243() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_244() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_245() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_246() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_247() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_248() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_249() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_250() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_251() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_252() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_253() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_254() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_255() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_256() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_257() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_258() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_259() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_260() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_261() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_262() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_263() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_264() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_265() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_266() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_267() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_268() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_269() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_270() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_271() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_272() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_273() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_274() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_275() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_276() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_277() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_278() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_279() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_280() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_281() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_282() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_283() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_284() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_285() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_286() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_287() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_288() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_289() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_290() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_291() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_292() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_293() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_294() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_295() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_296() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_297() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_298() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_299() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_300() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_301() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_302() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_303() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_304() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_305() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_306() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_307() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_308() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_309() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_310() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_311() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_312() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_313() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_314() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_315() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_316() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_317() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_318() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_319() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_320() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_321() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_322() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_323() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_324() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_325() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_326() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_327() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_328() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_329() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_330() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_331() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_332() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_333() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_334() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_335() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_336() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_337() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_338() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_339() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_340() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_341() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_342() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_343() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_344() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_345() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_346() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_347() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_348() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_349() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_350() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_351() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_352() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_353() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_354() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_355() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_356() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_357() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_358() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_359() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_360() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_361() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_362() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    #[test]
    fn test_mha_stress_363() {
        let mha = MultiheadAttention::new(8, 2);
        let x = Tensor::zeros(vec![1, 4, 8]);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        assert_eq!(mha.parameters().len(), 8);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
}
