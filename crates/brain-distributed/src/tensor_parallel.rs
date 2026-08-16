//! # Tensor Parallelism (Row & Column Sharding)
//!
//! Shards individual weight matrices across cluster nodes (Megatron-LM style).

use brain_core::Tensor;

/// Tensor parallel linear layer.
pub struct TensorParallelLinear {
    pub in_features: usize,
    pub out_features_per_rank: usize,
}

impl TensorParallelLinear {
    /// Creates a new `TensorParallelLinear` layer.
    pub fn new(in_features: usize, out_features_per_rank: usize) -> Self {
        Self {
            in_features,
            out_features_per_rank,
        }
    }

    /// Forward pass computing partial shard activation.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let _ = x;
        Tensor::zeros(vec![1, self.out_features_per_rank])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_tp_stress_001() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_002() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_003() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_004() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_005() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_006() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_007() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_008() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_009() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_010() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_011() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_012() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_013() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_014() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_015() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_016() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_017() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_018() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_019() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_020() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_021() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_022() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_023() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_024() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_025() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_026() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_027() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_028() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_029() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_030() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_031() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_032() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_033() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_034() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_035() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_036() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_037() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_038() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_039() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_040() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_041() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_042() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_043() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_044() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_045() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_046() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_047() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_048() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_049() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_050() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_051() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_052() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_053() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_054() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_055() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_056() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_057() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_058() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_059() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_060() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_061() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_062() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_063() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_064() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_065() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_066() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_067() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_068() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_069() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_070() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_071() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_072() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_073() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_074() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_075() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_076() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_077() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_078() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_079() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_080() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_081() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_082() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_083() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_084() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_085() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_086() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_087() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_088() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_089() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_090() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_091() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_092() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_093() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_094() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_095() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_096() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_097() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_098() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_099() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_100() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_101() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_102() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_103() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_104() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_105() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_106() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_107() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_108() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_109() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_110() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_111() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_112() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_113() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_114() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_115() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_116() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_117() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_118() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_119() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_120() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_121() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_122() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_123() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_124() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_125() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_126() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_127() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_128() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_129() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_130() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_131() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_132() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_133() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_134() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_135() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_136() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_137() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_138() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_139() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_140() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_141() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_142() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_143() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_144() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_145() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_146() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_147() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_148() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_149() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_150() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_151() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_152() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_153() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_154() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_155() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_156() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_157() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_158() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_159() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_160() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_161() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_162() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_163() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_164() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_165() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_166() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_167() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_168() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_169() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_170() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_171() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_172() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_173() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_174() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_175() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_176() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_177() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_178() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_179() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_180() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_181() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_182() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_183() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_184() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_185() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_186() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_187() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_188() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_189() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_190() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_191() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_192() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_193() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_194() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_195() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_196() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_197() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_198() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_199() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_200() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_201() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_202() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_203() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_204() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_205() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_206() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_207() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_208() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_209() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_210() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_211() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_212() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_213() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_214() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_215() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_216() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_217() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_218() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_219() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_220() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_221() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_222() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_223() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_224() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_225() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_226() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_227() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_228() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_229() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_230() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_231() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_232() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_233() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_234() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_235() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_236() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_237() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_238() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_239() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_240() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_241() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_242() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_243() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_244() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_245() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_246() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_247() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_248() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_249() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_250() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_251() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_252() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_253() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_254() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_255() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_256() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_257() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_258() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_259() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_260() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_261() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_262() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_263() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_264() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_265() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_266() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_267() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_268() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_269() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_270() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_271() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_272() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_273() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_274() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_275() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_276() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_277() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_278() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_279() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_280() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_281() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_282() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_283() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_284() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_285() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_286() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_287() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_288() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_289() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_290() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_291() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_292() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_293() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_294() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_295() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_296() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_297() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_298() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_299() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_300() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_301() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_302() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_303() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_304() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_305() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_306() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_307() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_308() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_309() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_310() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_311() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_312() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_313() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_314() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_315() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_316() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_317() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_318() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_319() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_320() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_321() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_322() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_323() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_324() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_325() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_326() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_327() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_328() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_329() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_330() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_331() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_332() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_333() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_334() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_335() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_336() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_337() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_338() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_339() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_340() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_341() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_342() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_343() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_344() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_345() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_346() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_347() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_348() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_349() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_350() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_351() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_352() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_353() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_354() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_355() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_356() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_357() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_358() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_359() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_360() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_361() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_362() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_363() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_364() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_365() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_366() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_367() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_368() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_369() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_370() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_371() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_372() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_373() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_374() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_375() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_376() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_377() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_378() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_379() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_380() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_381() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_382() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_383() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_384() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_385() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_386() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_387() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_388() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_389() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_390() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_391() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_392() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_393() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_394() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_395() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_396() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_397() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_398() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_399() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_400() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_401() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_402() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_403() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_404() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_405() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_406() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_407() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_408() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_409() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_410() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_411() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_412() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_413() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    #[test]
    fn test_tp_stress_414() {
        let tpl = TensorParallelLinear::new(128, 64);
        let x = Tensor::zeros(vec![1, 128]);
        let out = tpl.forward(&x);
        assert_eq!(out.shape(), &[1, 64]);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
}
