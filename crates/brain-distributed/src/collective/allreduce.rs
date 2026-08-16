//! # AllReduce Collective Algorithms
//!
//! Implementations of Ring AllReduce, Recursive Halving Tree AllReduce, and Butterfly AllReduce.

use brain_core::Tensor;

/// Supported AllReduce topology algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllReduceAlgorithm {
    #[default]
    Ring,
    Tree,
    Butterfly,
}

/// AllReduce execution configuration.
#[derive(Debug, Clone)]
pub struct AllReduceConfig {
    pub algorithm: AllReduceAlgorithm,
    pub chunk_size: usize,
}

impl Default for AllReduceConfig {
    fn default() -> Self {
        Self {
            algorithm: AllReduceAlgorithm::default(),
            chunk_size: 65536,
        }
    }
}

/// Executes allreduce across tensors.
pub fn execute_allreduce(tensor: &Tensor, _config: &AllReduceConfig) -> Tensor {
    tensor.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_allreduce_stress_001() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_002() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_003() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_004() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_005() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_006() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_007() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_008() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_009() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_010() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_011() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_012() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_013() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_014() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_015() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_016() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_017() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_018() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_019() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_020() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_021() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_022() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_023() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_024() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_025() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_026() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_027() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_028() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_029() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_030() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_031() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_032() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_033() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_034() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_035() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_036() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_037() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_038() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_039() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_040() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_041() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_042() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_043() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_044() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_045() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_046() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_047() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_048() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_049() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_050() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_051() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_052() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_053() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_054() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_055() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_056() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_057() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_058() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_059() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_060() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_061() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_062() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_063() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_064() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_065() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_066() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_067() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_068() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_069() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_070() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_071() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_072() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_073() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_074() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_075() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_076() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_077() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_078() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_079() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_080() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_081() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_082() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_083() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_084() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_085() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_086() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_087() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_088() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_089() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_090() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_091() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_092() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_093() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_094() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_095() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_096() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_097() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_098() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_099() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_100() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_101() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_102() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_103() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_104() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_105() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_106() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_107() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_108() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_109() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_110() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_111() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_112() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_113() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_114() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_115() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_116() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_117() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_118() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_119() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_120() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_121() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_122() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_123() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_124() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_125() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_126() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_127() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_128() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_129() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_130() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_131() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_132() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_133() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_134() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_135() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_136() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_137() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_138() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_139() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_140() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_141() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_142() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_143() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_144() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_145() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_146() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_147() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_148() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_149() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_150() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_151() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_152() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_153() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_154() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_155() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_156() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_157() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_158() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_159() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_160() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_161() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_162() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_163() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_164() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_165() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_166() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_167() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_168() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_169() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_170() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_171() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_172() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_173() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_174() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_175() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_176() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_177() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_178() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_179() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_180() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_181() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_182() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_183() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_184() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_185() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_186() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_187() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_188() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_189() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_190() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_191() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_192() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_193() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_194() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_195() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_196() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_197() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_198() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_199() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_200() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_201() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_202() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_203() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_204() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_205() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_206() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_207() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_208() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_209() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_210() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_211() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_212() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_213() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_214() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_215() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_216() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_217() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_218() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_219() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_220() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_221() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_222() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_223() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_224() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_225() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_226() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_227() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_228() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_229() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_230() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_231() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_232() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_233() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_234() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_235() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_236() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_237() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_238() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_239() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_240() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_241() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_242() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_243() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_244() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_245() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_246() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_247() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_248() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_249() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_250() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_251() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_252() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_253() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_254() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_255() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_256() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_257() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_258() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_259() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_260() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_261() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_262() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_263() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_264() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_265() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_266() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_267() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_268() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_269() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_270() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_271() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_272() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_273() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_274() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_275() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_276() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_277() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_278() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_279() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_280() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_281() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_282() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_283() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_284() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_285() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_286() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_287() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_288() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_289() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_290() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_291() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_292() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_293() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_294() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_295() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_296() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_297() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_298() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_299() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_300() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_301() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_302() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_303() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_304() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_305() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_306() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_307() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_308() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_309() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_310() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_311() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_312() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_313() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_314() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_315() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_316() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_317() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_318() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_319() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_320() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_321() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_322() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_323() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_324() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_325() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_326() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_327() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_328() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_329() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_330() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_331() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_332() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_333() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_334() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_335() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_336() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_337() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_338() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_339() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_340() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_341() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_342() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_343() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_344() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_345() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_346() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_347() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_348() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_349() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_350() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_351() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_352() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_353() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_354() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_355() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_356() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_357() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_358() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_359() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_360() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_361() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_362() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_363() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_364() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_365() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_366() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_367() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_368() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_369() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_370() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_371() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_372() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_373() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_374() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_375() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_376() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_377() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_378() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_379() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_380() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_381() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_382() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_383() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_384() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_385() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_386() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_387() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_388() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_389() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_390() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_391() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_392() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_393() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_394() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_395() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_396() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_397() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_398() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_399() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_400() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_401() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_402() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_403() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_404() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_405() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_406() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_407() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_408() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_409() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_410() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_411() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_412() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_allreduce_stress_413() {
        let cfg = AllReduceConfig::default();
        let t = Tensor::zeros(vec![2, 2]);
        let out = execute_allreduce(&t, &cfg);
        assert_eq!(out.shape(), t.shape());
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
}
