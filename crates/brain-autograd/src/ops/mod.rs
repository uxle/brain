//! # Autograd Forward Operations Subsystem
//!
//! Exposes differentiable forward operator primitives and gradient rules.

pub mod activation_grad;
pub mod binary;
pub mod broadcast_grad;
pub mod conv_grad;
pub mod fft_grad;
pub mod index_grad;
pub mod linalg_grad;
pub mod pool_grad;
pub mod quant_grad;
pub mod reduction_grad;
pub mod sparse_grad;
pub mod tensor_grad;
pub mod unary;

pub use binary::{add, div, matmul, mul, pow, sub};
pub use conv_grad::{conv2d, conv_transpose2d};
pub use pool_grad::{avg_pool2d, max_pool2d};
pub use unary::{exp, log, log_softmax, mean, neg, relu, sigmoid, softmax, sqrt, sum, tanh};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use brain_core::Tensor;

    #[test]
    fn test_ops_mod_stress_001() {
        let a = Value::scalar(1.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_002() {
        let a = Value::scalar(1.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_003() {
        let a = Value::scalar(1.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_004() {
        let a = Value::scalar(1.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_005() {
        let a = Value::scalar(1.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_006() {
        let a = Value::scalar(1.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_007() {
        let a = Value::scalar(1.7000000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_008() {
        let a = Value::scalar(1.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_009() {
        let a = Value::scalar(1.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_010() {
        let a = Value::scalar(2.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_011() {
        let a = Value::scalar(2.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_012() {
        let a = Value::scalar(2.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_013() {
        let a = Value::scalar(2.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_014() {
        let a = Value::scalar(2.4000000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_015() {
        let a = Value::scalar(2.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_016() {
        let a = Value::scalar(2.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_017() {
        let a = Value::scalar(2.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_018() {
        let a = Value::scalar(2.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_019() {
        let a = Value::scalar(2.9000000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_020() {
        let a = Value::scalar(3.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_021() {
        let a = Value::scalar(3.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_022() {
        let a = Value::scalar(3.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_023() {
        let a = Value::scalar(3.3000000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_024() {
        let a = Value::scalar(3.4000000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_025() {
        let a = Value::scalar(3.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_026() {
        let a = Value::scalar(3.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_027() {
        let a = Value::scalar(3.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_028() {
        let a = Value::scalar(3.8000000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_029() {
        let a = Value::scalar(3.9000000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_030() {
        let a = Value::scalar(4.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_031() {
        let a = Value::scalar(4.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_032() {
        let a = Value::scalar(4.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_033() {
        let a = Value::scalar(4.300000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_034() {
        let a = Value::scalar(4.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_035() {
        let a = Value::scalar(4.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_036() {
        let a = Value::scalar(4.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_037() {
        let a = Value::scalar(4.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_038() {
        let a = Value::scalar(4.800000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_039() {
        let a = Value::scalar(4.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_040() {
        let a = Value::scalar(5.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_041() {
        let a = Value::scalar(5.1000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_042() {
        let a = Value::scalar(5.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_043() {
        let a = Value::scalar(5.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_044() {
        let a = Value::scalar(5.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_045() {
        let a = Value::scalar(5.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_046() {
        let a = Value::scalar(5.6000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_047() {
        let a = Value::scalar(5.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_048() {
        let a = Value::scalar(5.800000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_049() {
        let a = Value::scalar(5.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_050() {
        let a = Value::scalar(6.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_051() {
        let a = Value::scalar(6.1000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_052() {
        let a = Value::scalar(6.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_053() {
        let a = Value::scalar(6.300000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_054() {
        let a = Value::scalar(6.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_055() {
        let a = Value::scalar(6.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_056() {
        let a = Value::scalar(6.6000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_057() {
        let a = Value::scalar(6.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_058() {
        let a = Value::scalar(6.800000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_059() {
        let a = Value::scalar(6.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_060() {
        let a = Value::scalar(7.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_061() {
        let a = Value::scalar(7.1000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_062() {
        let a = Value::scalar(7.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_063() {
        let a = Value::scalar(7.300000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_064() {
        let a = Value::scalar(7.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_065() {
        let a = Value::scalar(7.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_066() {
        let a = Value::scalar(7.6000000000000005);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_067() {
        let a = Value::scalar(7.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_068() {
        let a = Value::scalar(7.800000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_069() {
        let a = Value::scalar(7.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_070() {
        let a = Value::scalar(8.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_071() {
        let a = Value::scalar(8.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_072() {
        let a = Value::scalar(8.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_073() {
        let a = Value::scalar(8.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_074() {
        let a = Value::scalar(8.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_075() {
        let a = Value::scalar(8.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_076() {
        let a = Value::scalar(8.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_077() {
        let a = Value::scalar(8.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_078() {
        let a = Value::scalar(8.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_079() {
        let a = Value::scalar(8.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_080() {
        let a = Value::scalar(9.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_081() {
        let a = Value::scalar(9.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_082() {
        let a = Value::scalar(9.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_083() {
        let a = Value::scalar(9.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_084() {
        let a = Value::scalar(9.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_085() {
        let a = Value::scalar(9.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_086() {
        let a = Value::scalar(9.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_087() {
        let a = Value::scalar(9.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_088() {
        let a = Value::scalar(9.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_089() {
        let a = Value::scalar(9.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_090() {
        let a = Value::scalar(10.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_091() {
        let a = Value::scalar(10.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_092() {
        let a = Value::scalar(10.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_093() {
        let a = Value::scalar(10.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_094() {
        let a = Value::scalar(10.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_095() {
        let a = Value::scalar(10.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_096() {
        let a = Value::scalar(10.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_097() {
        let a = Value::scalar(10.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_098() {
        let a = Value::scalar(10.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_099() {
        let a = Value::scalar(10.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_100() {
        let a = Value::scalar(11.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_101() {
        let a = Value::scalar(11.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_102() {
        let a = Value::scalar(11.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_103() {
        let a = Value::scalar(11.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_104() {
        let a = Value::scalar(11.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_105() {
        let a = Value::scalar(11.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_106() {
        let a = Value::scalar(11.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_107() {
        let a = Value::scalar(11.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_108() {
        let a = Value::scalar(11.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_109() {
        let a = Value::scalar(11.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_110() {
        let a = Value::scalar(12.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_111() {
        let a = Value::scalar(12.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_112() {
        let a = Value::scalar(12.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_113() {
        let a = Value::scalar(12.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_114() {
        let a = Value::scalar(12.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_115() {
        let a = Value::scalar(12.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_116() {
        let a = Value::scalar(12.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_117() {
        let a = Value::scalar(12.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_118() {
        let a = Value::scalar(12.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_119() {
        let a = Value::scalar(12.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_120() {
        let a = Value::scalar(13.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_121() {
        let a = Value::scalar(13.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_122() {
        let a = Value::scalar(13.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_123() {
        let a = Value::scalar(13.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_124() {
        let a = Value::scalar(13.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_125() {
        let a = Value::scalar(13.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_126() {
        let a = Value::scalar(13.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_127() {
        let a = Value::scalar(13.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_128() {
        let a = Value::scalar(13.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_129() {
        let a = Value::scalar(13.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_130() {
        let a = Value::scalar(14.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_131() {
        let a = Value::scalar(14.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_132() {
        let a = Value::scalar(14.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_133() {
        let a = Value::scalar(14.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_134() {
        let a = Value::scalar(14.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_135() {
        let a = Value::scalar(14.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_136() {
        let a = Value::scalar(14.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_137() {
        let a = Value::scalar(14.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_138() {
        let a = Value::scalar(14.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_139() {
        let a = Value::scalar(14.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_140() {
        let a = Value::scalar(15.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_141() {
        let a = Value::scalar(15.100000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_142() {
        let a = Value::scalar(15.200000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_143() {
        let a = Value::scalar(15.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_144() {
        let a = Value::scalar(15.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_145() {
        let a = Value::scalar(15.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_146() {
        let a = Value::scalar(15.600000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_147() {
        let a = Value::scalar(15.700000000000001);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_148() {
        let a = Value::scalar(15.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_149() {
        let a = Value::scalar(15.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_150() {
        let a = Value::scalar(16.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_151() {
        let a = Value::scalar(16.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_152() {
        let a = Value::scalar(16.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_153() {
        let a = Value::scalar(16.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_154() {
        let a = Value::scalar(16.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_155() {
        let a = Value::scalar(16.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_156() {
        let a = Value::scalar(16.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_157() {
        let a = Value::scalar(16.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_158() {
        let a = Value::scalar(16.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_159() {
        let a = Value::scalar(16.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_160() {
        let a = Value::scalar(17.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_161() {
        let a = Value::scalar(17.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_162() {
        let a = Value::scalar(17.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_163() {
        let a = Value::scalar(17.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_164() {
        let a = Value::scalar(17.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_165() {
        let a = Value::scalar(17.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_166() {
        let a = Value::scalar(17.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_167() {
        let a = Value::scalar(17.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_168() {
        let a = Value::scalar(17.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_169() {
        let a = Value::scalar(17.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_170() {
        let a = Value::scalar(18.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_171() {
        let a = Value::scalar(18.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_172() {
        let a = Value::scalar(18.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_173() {
        let a = Value::scalar(18.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_174() {
        let a = Value::scalar(18.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_175() {
        let a = Value::scalar(18.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_176() {
        let a = Value::scalar(18.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_177() {
        let a = Value::scalar(18.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_178() {
        let a = Value::scalar(18.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_179() {
        let a = Value::scalar(18.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_180() {
        let a = Value::scalar(19.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_181() {
        let a = Value::scalar(19.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_182() {
        let a = Value::scalar(19.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_183() {
        let a = Value::scalar(19.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_184() {
        let a = Value::scalar(19.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_185() {
        let a = Value::scalar(19.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_186() {
        let a = Value::scalar(19.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_187() {
        let a = Value::scalar(19.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_188() {
        let a = Value::scalar(19.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_189() {
        let a = Value::scalar(19.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_190() {
        let a = Value::scalar(20.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_191() {
        let a = Value::scalar(20.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_192() {
        let a = Value::scalar(20.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_193() {
        let a = Value::scalar(20.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_194() {
        let a = Value::scalar(20.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_195() {
        let a = Value::scalar(20.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_196() {
        let a = Value::scalar(20.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_197() {
        let a = Value::scalar(20.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_198() {
        let a = Value::scalar(20.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_199() {
        let a = Value::scalar(20.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_200() {
        let a = Value::scalar(21.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_201() {
        let a = Value::scalar(21.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_202() {
        let a = Value::scalar(21.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_203() {
        let a = Value::scalar(21.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_204() {
        let a = Value::scalar(21.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_205() {
        let a = Value::scalar(21.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_206() {
        let a = Value::scalar(21.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_207() {
        let a = Value::scalar(21.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_208() {
        let a = Value::scalar(21.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_209() {
        let a = Value::scalar(21.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_210() {
        let a = Value::scalar(22.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_211() {
        let a = Value::scalar(22.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_212() {
        let a = Value::scalar(22.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_213() {
        let a = Value::scalar(22.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_214() {
        let a = Value::scalar(22.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_215() {
        let a = Value::scalar(22.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_216() {
        let a = Value::scalar(22.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_217() {
        let a = Value::scalar(22.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_218() {
        let a = Value::scalar(22.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_219() {
        let a = Value::scalar(22.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_220() {
        let a = Value::scalar(23.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_221() {
        let a = Value::scalar(23.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_222() {
        let a = Value::scalar(23.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_223() {
        let a = Value::scalar(23.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_224() {
        let a = Value::scalar(23.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_225() {
        let a = Value::scalar(23.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_226() {
        let a = Value::scalar(23.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_227() {
        let a = Value::scalar(23.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_228() {
        let a = Value::scalar(23.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_229() {
        let a = Value::scalar(23.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_230() {
        let a = Value::scalar(24.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_231() {
        let a = Value::scalar(24.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_232() {
        let a = Value::scalar(24.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_233() {
        let a = Value::scalar(24.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_234() {
        let a = Value::scalar(24.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_235() {
        let a = Value::scalar(24.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_236() {
        let a = Value::scalar(24.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_237() {
        let a = Value::scalar(24.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_238() {
        let a = Value::scalar(24.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_239() {
        let a = Value::scalar(24.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_240() {
        let a = Value::scalar(25.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_241() {
        let a = Value::scalar(25.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_242() {
        let a = Value::scalar(25.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_243() {
        let a = Value::scalar(25.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_244() {
        let a = Value::scalar(25.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_245() {
        let a = Value::scalar(25.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_246() {
        let a = Value::scalar(25.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_247() {
        let a = Value::scalar(25.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_248() {
        let a = Value::scalar(25.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_249() {
        let a = Value::scalar(25.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_250() {
        let a = Value::scalar(26.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_251() {
        let a = Value::scalar(26.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_252() {
        let a = Value::scalar(26.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_253() {
        let a = Value::scalar(26.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_254() {
        let a = Value::scalar(26.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_255() {
        let a = Value::scalar(26.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_256() {
        let a = Value::scalar(26.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_257() {
        let a = Value::scalar(26.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_258() {
        let a = Value::scalar(26.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_259() {
        let a = Value::scalar(26.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_260() {
        let a = Value::scalar(27.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_261() {
        let a = Value::scalar(27.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_262() {
        let a = Value::scalar(27.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_263() {
        let a = Value::scalar(27.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_264() {
        let a = Value::scalar(27.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_265() {
        let a = Value::scalar(27.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_266() {
        let a = Value::scalar(27.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_267() {
        let a = Value::scalar(27.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_268() {
        let a = Value::scalar(27.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_269() {
        let a = Value::scalar(27.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_270() {
        let a = Value::scalar(28.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_271() {
        let a = Value::scalar(28.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_272() {
        let a = Value::scalar(28.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_273() {
        let a = Value::scalar(28.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_274() {
        let a = Value::scalar(28.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_275() {
        let a = Value::scalar(28.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_276() {
        let a = Value::scalar(28.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_277() {
        let a = Value::scalar(28.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_278() {
        let a = Value::scalar(28.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_279() {
        let a = Value::scalar(28.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_280() {
        let a = Value::scalar(29.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_281() {
        let a = Value::scalar(29.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_282() {
        let a = Value::scalar(29.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_283() {
        let a = Value::scalar(29.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_284() {
        let a = Value::scalar(29.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_285() {
        let a = Value::scalar(29.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_286() {
        let a = Value::scalar(29.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_287() {
        let a = Value::scalar(29.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_288() {
        let a = Value::scalar(29.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_289() {
        let a = Value::scalar(29.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_290() {
        let a = Value::scalar(30.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_291() {
        let a = Value::scalar(30.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_292() {
        let a = Value::scalar(30.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_293() {
        let a = Value::scalar(30.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_294() {
        let a = Value::scalar(30.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_295() {
        let a = Value::scalar(30.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_296() {
        let a = Value::scalar(30.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_297() {
        let a = Value::scalar(30.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_298() {
        let a = Value::scalar(30.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_299() {
        let a = Value::scalar(30.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_300() {
        let a = Value::scalar(31.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_301() {
        let a = Value::scalar(31.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_302() {
        let a = Value::scalar(31.200000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_303() {
        let a = Value::scalar(31.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_304() {
        let a = Value::scalar(31.400000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_305() {
        let a = Value::scalar(31.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_306() {
        let a = Value::scalar(31.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_307() {
        let a = Value::scalar(31.700000000000003);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_308() {
        let a = Value::scalar(31.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_309() {
        let a = Value::scalar(31.900000000000002);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_310() {
        let a = Value::scalar(32.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_311() {
        let a = Value::scalar(32.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_312() {
        let a = Value::scalar(32.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_313() {
        let a = Value::scalar(32.3);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_314() {
        let a = Value::scalar(32.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_315() {
        let a = Value::scalar(32.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_316() {
        let a = Value::scalar(32.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_317() {
        let a = Value::scalar(32.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_318() {
        let a = Value::scalar(32.8);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_319() {
        let a = Value::scalar(32.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_320() {
        let a = Value::scalar(33.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_321() {
        let a = Value::scalar(33.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_322() {
        let a = Value::scalar(33.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_323() {
        let a = Value::scalar(33.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_324() {
        let a = Value::scalar(33.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_325() {
        let a = Value::scalar(33.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_326() {
        let a = Value::scalar(33.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_327() {
        let a = Value::scalar(33.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_328() {
        let a = Value::scalar(33.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_329() {
        let a = Value::scalar(33.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_330() {
        let a = Value::scalar(34.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_331() {
        let a = Value::scalar(34.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_332() {
        let a = Value::scalar(34.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_333() {
        let a = Value::scalar(34.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_334() {
        let a = Value::scalar(34.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_335() {
        let a = Value::scalar(34.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_336() {
        let a = Value::scalar(34.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_337() {
        let a = Value::scalar(34.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_338() {
        let a = Value::scalar(34.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_339() {
        let a = Value::scalar(34.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_340() {
        let a = Value::scalar(35.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_341() {
        let a = Value::scalar(35.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_342() {
        let a = Value::scalar(35.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_343() {
        let a = Value::scalar(35.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_344() {
        let a = Value::scalar(35.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_345() {
        let a = Value::scalar(35.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_346() {
        let a = Value::scalar(35.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_347() {
        let a = Value::scalar(35.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_348() {
        let a = Value::scalar(35.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_349() {
        let a = Value::scalar(35.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_350() {
        let a = Value::scalar(36.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_351() {
        let a = Value::scalar(36.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_352() {
        let a = Value::scalar(36.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_353() {
        let a = Value::scalar(36.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_354() {
        let a = Value::scalar(36.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_355() {
        let a = Value::scalar(36.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_356() {
        let a = Value::scalar(36.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_357() {
        let a = Value::scalar(36.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_358() {
        let a = Value::scalar(36.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_359() {
        let a = Value::scalar(36.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_360() {
        let a = Value::scalar(37.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_361() {
        let a = Value::scalar(37.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_362() {
        let a = Value::scalar(37.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_363() {
        let a = Value::scalar(37.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_364() {
        let a = Value::scalar(37.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_365() {
        let a = Value::scalar(37.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_366() {
        let a = Value::scalar(37.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_367() {
        let a = Value::scalar(37.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_368() {
        let a = Value::scalar(37.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_369() {
        let a = Value::scalar(37.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_370() {
        let a = Value::scalar(38.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_371() {
        let a = Value::scalar(38.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_372() {
        let a = Value::scalar(38.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_373() {
        let a = Value::scalar(38.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_374() {
        let a = Value::scalar(38.4);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_375() {
        let a = Value::scalar(38.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_376() {
        let a = Value::scalar(38.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_377() {
        let a = Value::scalar(38.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_378() {
        let a = Value::scalar(38.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_379() {
        let a = Value::scalar(38.9);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_380() {
        let a = Value::scalar(39.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_381() {
        let a = Value::scalar(39.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_382() {
        let a = Value::scalar(39.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_383() {
        let a = Value::scalar(39.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_384() {
        let a = Value::scalar(39.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_385() {
        let a = Value::scalar(39.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_386() {
        let a = Value::scalar(39.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_387() {
        let a = Value::scalar(39.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_388() {
        let a = Value::scalar(39.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_389() {
        let a = Value::scalar(39.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_390() {
        let a = Value::scalar(40.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_391() {
        let a = Value::scalar(40.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_392() {
        let a = Value::scalar(40.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_393() {
        let a = Value::scalar(40.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_394() {
        let a = Value::scalar(40.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_395() {
        let a = Value::scalar(40.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_396() {
        let a = Value::scalar(40.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_397() {
        let a = Value::scalar(40.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_398() {
        let a = Value::scalar(40.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_399() {
        let a = Value::scalar(40.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_400() {
        let a = Value::scalar(41.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_401() {
        let a = Value::scalar(41.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_402() {
        let a = Value::scalar(41.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_403() {
        let a = Value::scalar(41.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_404() {
        let a = Value::scalar(41.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_405() {
        let a = Value::scalar(41.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_406() {
        let a = Value::scalar(41.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_407() {
        let a = Value::scalar(41.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_408() {
        let a = Value::scalar(41.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_409() {
        let a = Value::scalar(41.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_410() {
        let a = Value::scalar(42.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_411() {
        let a = Value::scalar(42.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_412() {
        let a = Value::scalar(42.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_413() {
        let a = Value::scalar(42.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_414() {
        let a = Value::scalar(42.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_415() {
        let a = Value::scalar(42.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_416() {
        let a = Value::scalar(42.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_417() {
        let a = Value::scalar(42.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_418() {
        let a = Value::scalar(42.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_419() {
        let a = Value::scalar(42.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_420() {
        let a = Value::scalar(43.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_421() {
        let a = Value::scalar(43.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_422() {
        let a = Value::scalar(43.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_423() {
        let a = Value::scalar(43.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_424() {
        let a = Value::scalar(43.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_425() {
        let a = Value::scalar(43.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_426() {
        let a = Value::scalar(43.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_427() {
        let a = Value::scalar(43.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_428() {
        let a = Value::scalar(43.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_429() {
        let a = Value::scalar(43.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_430() {
        let a = Value::scalar(44.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_431() {
        let a = Value::scalar(44.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_432() {
        let a = Value::scalar(44.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_433() {
        let a = Value::scalar(44.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_434() {
        let a = Value::scalar(44.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_435() {
        let a = Value::scalar(44.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_436() {
        let a = Value::scalar(44.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_437() {
        let a = Value::scalar(44.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_438() {
        let a = Value::scalar(44.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_439() {
        let a = Value::scalar(44.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_440() {
        let a = Value::scalar(45.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_441() {
        let a = Value::scalar(45.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_442() {
        let a = Value::scalar(45.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_443() {
        let a = Value::scalar(45.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_444() {
        let a = Value::scalar(45.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_445() {
        let a = Value::scalar(45.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_446() {
        let a = Value::scalar(45.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_447() {
        let a = Value::scalar(45.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_448() {
        let a = Value::scalar(45.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_449() {
        let a = Value::scalar(45.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_450() {
        let a = Value::scalar(46.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_451() {
        let a = Value::scalar(46.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_452() {
        let a = Value::scalar(46.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_453() {
        let a = Value::scalar(46.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_454() {
        let a = Value::scalar(46.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_455() {
        let a = Value::scalar(46.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_456() {
        let a = Value::scalar(46.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_457() {
        let a = Value::scalar(46.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_458() {
        let a = Value::scalar(46.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_459() {
        let a = Value::scalar(46.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_460() {
        let a = Value::scalar(47.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_461() {
        let a = Value::scalar(47.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_462() {
        let a = Value::scalar(47.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_463() {
        let a = Value::scalar(47.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_464() {
        let a = Value::scalar(47.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_465() {
        let a = Value::scalar(47.5);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_466() {
        let a = Value::scalar(47.6);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_467() {
        let a = Value::scalar(47.7);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_468() {
        let a = Value::scalar(47.800000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_469() {
        let a = Value::scalar(47.900000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_470() {
        let a = Value::scalar(48.0);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_471() {
        let a = Value::scalar(48.1);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_472() {
        let a = Value::scalar(48.2);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_473() {
        let a = Value::scalar(48.300000000000004);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }

    #[test]
    fn test_ops_mod_stress_474() {
        let a = Value::scalar(48.400000000000006);
        let b = exp(&a);
        assert!(b.data().get(0) > 0.0);
    }
}
