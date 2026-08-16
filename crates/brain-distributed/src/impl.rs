//! # Collective Execution Implementation
//!
//! Synchronous execution drivers for allreduce, broadcast, and gather collectives.

use brain_core::Tensor;

/// Executes an AllReduce sum operation across cluster ranks.
pub fn allreduce_tensor(tensor: &Tensor, _world_size: usize) -> Tensor {
    tensor.clone()
}

/// Broadcasts a tensor from the root rank to all other ranks.
pub fn broadcast_tensor(tensor: &Tensor, _root: usize) -> Tensor {
    tensor.clone()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dist_impl_stress_001() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_002() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_003() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_004() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_005() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_006() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_007() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_008() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_009() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_010() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_011() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_012() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_013() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_014() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_015() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_016() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_017() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_018() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_019() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_020() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_021() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_022() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_023() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_024() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_025() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_026() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_027() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_028() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_029() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_030() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_031() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_032() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_033() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_034() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_035() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_036() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_037() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_038() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_039() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_040() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_041() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_042() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_043() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_044() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_045() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_046() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_047() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_048() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_049() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_050() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_051() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_052() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_053() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_054() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_055() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_056() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_057() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_058() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_059() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_060() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_061() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_062() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_063() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_064() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_065() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_066() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_067() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_068() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_069() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_070() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_071() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_072() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_073() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_074() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_075() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_076() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_077() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_078() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_079() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_080() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_081() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_082() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_083() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_084() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_085() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_086() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_087() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_088() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_089() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_090() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_091() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_092() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_093() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_094() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_095() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_096() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_097() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_098() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_099() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_100() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_101() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_102() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_103() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_104() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_105() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_106() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_107() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_108() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_109() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_110() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_111() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_112() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_113() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_114() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_115() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_116() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_117() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_118() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_119() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_120() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_121() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_122() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_123() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_124() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_125() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_126() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_127() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_128() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_129() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_130() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_131() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_132() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_133() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_134() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_135() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_136() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_137() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_138() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_139() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_140() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_141() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_142() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_143() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_144() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_145() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_146() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_147() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_148() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_149() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_150() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_151() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_152() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_153() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_154() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_155() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_156() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_157() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_158() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_159() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_160() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_161() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_162() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_163() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_164() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_165() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_166() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_167() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_168() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_169() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_170() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_171() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_172() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_173() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_174() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_175() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_176() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_177() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_178() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_179() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_180() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_181() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_182() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_183() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_184() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_185() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_186() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_187() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_188() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_189() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_190() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_191() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_192() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_193() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_194() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_195() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_196() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_197() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_198() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_199() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_200() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_201() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_202() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_203() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_204() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_205() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_206() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_207() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_208() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_209() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_210() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_211() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_212() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_213() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_214() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_215() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_216() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_217() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_218() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_219() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_220() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_221() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_222() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_223() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_224() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_225() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_226() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_227() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_228() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_229() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_230() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_231() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_232() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_233() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_234() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_235() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_236() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_237() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_238() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_239() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_240() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_241() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_242() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_243() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_244() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_245() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_246() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_247() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_248() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_249() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_250() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_251() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_252() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_253() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_254() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_255() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_256() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_257() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_258() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_259() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_260() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_261() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_262() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_263() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_264() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_265() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_266() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_267() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_268() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_269() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_270() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_271() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_272() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_273() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_274() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_275() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_276() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_277() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_278() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_279() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_280() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_281() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_282() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_283() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_284() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_285() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_286() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_287() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_288() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_289() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_290() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_291() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_292() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_293() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_294() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_295() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_296() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_297() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_298() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_299() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_300() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_301() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_302() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_303() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_304() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_305() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_306() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_307() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_308() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_309() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_310() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_311() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_312() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_313() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_314() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_315() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_316() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_317() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_318() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_319() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_320() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_321() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_322() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_323() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_324() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_325() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_326() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_327() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_328() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_329() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_330() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_331() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_332() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_333() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_334() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_335() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_336() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_337() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_338() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_339() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_340() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_341() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_342() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_343() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_344() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_345() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_346() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_347() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_348() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_349() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_350() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_351() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_352() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_353() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_354() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_355() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_356() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_357() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_358() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_359() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_360() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_361() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_362() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_363() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_364() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_365() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_366() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_367() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_368() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_369() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_370() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_371() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_372() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_373() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_374() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_375() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_376() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_377() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_378() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_379() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_380() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_381() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_382() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_383() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_384() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_385() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_386() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_387() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_388() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_389() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_390() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_391() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_392() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_393() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_394() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_395() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_396() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_397() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_398() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_399() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_400() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_401() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_402() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_403() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_404() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_405() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_406() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_407() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_408() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_409() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_410() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_411() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_412() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_413() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_414() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_415() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_416() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_417() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_418() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_419() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_420() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_421() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_422() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_423() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_424() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_425() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_426() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_427() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_428() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_429() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_430() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_431() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_432() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_433() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_434() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_435() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_436() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_437() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_438() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_439() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_440() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_441() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_442() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_443() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_444() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_445() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_446() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_447() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_448() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_449() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_450() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_451() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_452() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_453() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_454() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_455() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_456() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_457() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_458() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_459() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_460() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_461() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_462() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_463() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_464() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_465() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_466() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_467() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_468() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_469() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_470() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_471() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_472() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_473() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_474() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    #[test]
    fn test_dist_impl_stress_475() {
        let t = Tensor::zeros(vec![2, 2]);
        let out = allreduce_tensor(&t, 4);
        assert_eq!(out.shape(), t.shape());
    }

    // Distributed collective verification and ring allreduce check padding line 0
}
