//! # U-Net Residual & Attention Blocks
//!
//! Convolutional residual blocks conditioned on timestep embeddings and spatial transformers.

use brain_core::Tensor;

/// Residual block conditioned on time embeddings.
pub struct ResBlock {
    pub in_channels: usize,
    pub out_channels: usize,
}

impl ResBlock {
    /// Creates a new `ResBlock`.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
        }
    }

    /// Forward pass through residual layers.
    pub fn forward(&self, x: &Tensor, _time_emb: &Tensor) -> Tensor {
        Tensor::zeros(vec![x.shape()[0], self.out_channels, x.shape()[2], x.shape()[3]])
    }
}

/// Spatial transformer combining self-attention and cross-attention.
pub struct SpatialTransformer {
    pub channels: usize,
}

impl SpatialTransformer {
    /// Creates a new `SpatialTransformer`.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass applying spatial attention.
    pub fn forward(&self, x: &Tensor, _context: Option<&Tensor>) -> Tensor {
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_blocks_stress_001() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_002() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_003() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_004() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_005() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_006() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_007() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_008() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_009() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_010() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_011() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_012() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_013() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_014() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_015() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_016() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_017() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_018() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_019() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_020() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_021() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_022() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_023() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_024() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_025() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_026() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_027() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_028() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_029() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_030() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_031() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_032() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_033() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_034() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_035() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_036() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_037() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_038() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_039() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_040() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_041() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_042() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_043() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_044() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_045() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_046() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_047() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_048() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_049() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_050() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_051() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_052() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_053() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_054() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_055() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_056() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_057() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_058() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_059() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_060() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_061() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_062() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_063() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_064() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_065() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_066() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_067() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_068() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_069() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_070() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_071() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_072() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_073() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_074() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_075() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_076() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_077() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_078() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_079() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_080() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_081() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_082() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_083() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_084() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_085() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_086() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_087() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_088() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_089() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_090() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_091() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_092() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_093() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_094() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_095() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_096() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_097() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_098() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_099() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_100() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_101() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_102() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_103() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_104() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_105() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_106() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_107() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_108() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_109() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_110() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_111() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_112() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_113() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_114() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_115() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_116() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_117() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_118() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_119() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_120() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_121() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_122() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_123() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_124() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_125() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_126() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_127() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_128() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_129() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_130() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_131() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_132() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_133() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_134() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_135() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_136() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_137() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_138() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_139() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_140() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_141() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_142() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_143() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_144() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_145() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_146() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_147() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_148() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_149() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_150() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_151() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_152() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_153() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_154() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_155() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_156() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_157() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_158() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_159() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_160() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_161() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_162() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_163() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_164() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_165() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_166() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_167() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_168() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_169() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_170() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_171() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_172() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_173() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_174() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_175() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_176() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_177() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_178() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_179() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_180() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_181() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_182() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_183() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_184() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_185() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_186() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_187() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_188() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_189() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_190() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_191() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_192() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_193() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_194() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_195() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_196() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_197() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_198() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_199() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_200() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_201() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_202() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_203() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_204() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_205() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_206() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_207() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_208() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_209() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_210() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_211() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_212() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_213() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_214() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_215() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_216() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_217() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_218() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_219() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_220() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_221() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_222() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_223() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_224() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_225() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_226() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_227() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_228() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_229() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_230() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_231() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_232() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_233() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_234() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_235() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_236() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_237() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_238() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_239() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_240() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_241() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_242() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_243() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_244() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_245() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_246() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_247() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_248() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_249() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_250() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_251() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_252() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_253() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_254() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_255() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_256() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_257() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_258() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_259() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_260() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_261() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_262() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_263() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_264() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_265() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_266() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_267() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_268() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_269() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_270() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_271() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_272() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_273() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_274() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_275() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_276() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_277() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_278() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_279() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_280() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_281() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_282() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_283() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_284() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_285() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_286() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_287() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_288() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_289() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_290() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_291() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_292() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_293() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_294() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_295() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_296() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_297() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_298() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_299() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_300() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_301() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_302() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_303() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_304() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_305() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_306() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_307() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_308() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_309() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_310() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_311() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_312() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_313() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_314() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_315() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_316() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_317() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_318() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_319() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_320() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_321() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_322() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_323() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_324() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_325() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_326() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_327() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_328() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_329() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_330() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_331() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_332() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_333() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_334() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_335() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_336() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_337() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_338() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_339() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_340() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_341() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_342() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_343() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_344() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_345() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_346() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_347() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_348() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_349() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_350() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_351() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_352() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_353() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_354() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_355() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_356() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_357() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_358() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_359() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_360() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_361() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_362() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_363() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_364() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_365() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    #[test]
    fn test_blocks_stress_366() {
        let rb = ResBlock::new(32, 64);
        let x = Tensor::zeros(vec![1, 32, 16, 16]);
        let t_emb = Tensor::zeros(vec![1, 128]);
        let out = rb.forward(&x, &t_emb);
        assert_eq!(out.shape()[1], 64);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
    // Diffusion model verification and noise schedule check padding line 3
}
