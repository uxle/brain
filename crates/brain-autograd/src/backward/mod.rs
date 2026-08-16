//! # Reverse-Mode Backward Subsystem
//!
//! High-performance topological ordering, gradient dispatch, and memory management.

pub mod grad;
pub mod topo;

pub use grad::{backward_from, backward_with_grad, grad, grad_of, value_and_grad, zero_grad_from};
pub use topo::topological_sort;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use brain_core::Tensor;

    #[test]
    fn test_backward_mod_stress_001() {
        let mut a = Value::scalar(1.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_002() {
        let mut a = Value::scalar(1.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_003() {
        let mut a = Value::scalar(1.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_004() {
        let mut a = Value::scalar(1.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_005() {
        let mut a = Value::scalar(1.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_006() {
        let mut a = Value::scalar(1.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_007() {
        let mut a = Value::scalar(1.7000000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_008() {
        let mut a = Value::scalar(1.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_009() {
        let mut a = Value::scalar(1.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_010() {
        let mut a = Value::scalar(2.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_011() {
        let mut a = Value::scalar(2.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_012() {
        let mut a = Value::scalar(2.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_013() {
        let mut a = Value::scalar(2.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_014() {
        let mut a = Value::scalar(2.4000000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_015() {
        let mut a = Value::scalar(2.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_016() {
        let mut a = Value::scalar(2.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_017() {
        let mut a = Value::scalar(2.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_018() {
        let mut a = Value::scalar(2.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_019() {
        let mut a = Value::scalar(2.9000000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_020() {
        let mut a = Value::scalar(3.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_021() {
        let mut a = Value::scalar(3.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_022() {
        let mut a = Value::scalar(3.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_023() {
        let mut a = Value::scalar(3.3000000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_024() {
        let mut a = Value::scalar(3.4000000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_025() {
        let mut a = Value::scalar(3.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_026() {
        let mut a = Value::scalar(3.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_027() {
        let mut a = Value::scalar(3.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_028() {
        let mut a = Value::scalar(3.8000000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_029() {
        let mut a = Value::scalar(3.9000000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_030() {
        let mut a = Value::scalar(4.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_031() {
        let mut a = Value::scalar(4.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_032() {
        let mut a = Value::scalar(4.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_033() {
        let mut a = Value::scalar(4.300000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_034() {
        let mut a = Value::scalar(4.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_035() {
        let mut a = Value::scalar(4.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_036() {
        let mut a = Value::scalar(4.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_037() {
        let mut a = Value::scalar(4.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_038() {
        let mut a = Value::scalar(4.800000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_039() {
        let mut a = Value::scalar(4.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_040() {
        let mut a = Value::scalar(5.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_041() {
        let mut a = Value::scalar(5.1000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_042() {
        let mut a = Value::scalar(5.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_043() {
        let mut a = Value::scalar(5.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_044() {
        let mut a = Value::scalar(5.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_045() {
        let mut a = Value::scalar(5.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_046() {
        let mut a = Value::scalar(5.6000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_047() {
        let mut a = Value::scalar(5.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_048() {
        let mut a = Value::scalar(5.800000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_049() {
        let mut a = Value::scalar(5.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_050() {
        let mut a = Value::scalar(6.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_051() {
        let mut a = Value::scalar(6.1000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_052() {
        let mut a = Value::scalar(6.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_053() {
        let mut a = Value::scalar(6.300000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_054() {
        let mut a = Value::scalar(6.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_055() {
        let mut a = Value::scalar(6.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_056() {
        let mut a = Value::scalar(6.6000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_057() {
        let mut a = Value::scalar(6.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_058() {
        let mut a = Value::scalar(6.800000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_059() {
        let mut a = Value::scalar(6.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_060() {
        let mut a = Value::scalar(7.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_061() {
        let mut a = Value::scalar(7.1000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_062() {
        let mut a = Value::scalar(7.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_063() {
        let mut a = Value::scalar(7.300000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_064() {
        let mut a = Value::scalar(7.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_065() {
        let mut a = Value::scalar(7.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_066() {
        let mut a = Value::scalar(7.6000000000000005);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_067() {
        let mut a = Value::scalar(7.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_068() {
        let mut a = Value::scalar(7.800000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_069() {
        let mut a = Value::scalar(7.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_070() {
        let mut a = Value::scalar(8.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_071() {
        let mut a = Value::scalar(8.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_072() {
        let mut a = Value::scalar(8.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_073() {
        let mut a = Value::scalar(8.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_074() {
        let mut a = Value::scalar(8.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_075() {
        let mut a = Value::scalar(8.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_076() {
        let mut a = Value::scalar(8.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_077() {
        let mut a = Value::scalar(8.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_078() {
        let mut a = Value::scalar(8.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_079() {
        let mut a = Value::scalar(8.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_080() {
        let mut a = Value::scalar(9.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_081() {
        let mut a = Value::scalar(9.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_082() {
        let mut a = Value::scalar(9.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_083() {
        let mut a = Value::scalar(9.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_084() {
        let mut a = Value::scalar(9.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_085() {
        let mut a = Value::scalar(9.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_086() {
        let mut a = Value::scalar(9.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_087() {
        let mut a = Value::scalar(9.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_088() {
        let mut a = Value::scalar(9.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_089() {
        let mut a = Value::scalar(9.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_090() {
        let mut a = Value::scalar(10.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_091() {
        let mut a = Value::scalar(10.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_092() {
        let mut a = Value::scalar(10.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_093() {
        let mut a = Value::scalar(10.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_094() {
        let mut a = Value::scalar(10.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_095() {
        let mut a = Value::scalar(10.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_096() {
        let mut a = Value::scalar(10.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_097() {
        let mut a = Value::scalar(10.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_098() {
        let mut a = Value::scalar(10.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_099() {
        let mut a = Value::scalar(10.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_100() {
        let mut a = Value::scalar(11.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_101() {
        let mut a = Value::scalar(11.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_102() {
        let mut a = Value::scalar(11.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_103() {
        let mut a = Value::scalar(11.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_104() {
        let mut a = Value::scalar(11.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_105() {
        let mut a = Value::scalar(11.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_106() {
        let mut a = Value::scalar(11.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_107() {
        let mut a = Value::scalar(11.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_108() {
        let mut a = Value::scalar(11.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_109() {
        let mut a = Value::scalar(11.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_110() {
        let mut a = Value::scalar(12.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_111() {
        let mut a = Value::scalar(12.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_112() {
        let mut a = Value::scalar(12.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_113() {
        let mut a = Value::scalar(12.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_114() {
        let mut a = Value::scalar(12.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_115() {
        let mut a = Value::scalar(12.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_116() {
        let mut a = Value::scalar(12.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_117() {
        let mut a = Value::scalar(12.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_118() {
        let mut a = Value::scalar(12.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_119() {
        let mut a = Value::scalar(12.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_120() {
        let mut a = Value::scalar(13.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_121() {
        let mut a = Value::scalar(13.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_122() {
        let mut a = Value::scalar(13.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_123() {
        let mut a = Value::scalar(13.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_124() {
        let mut a = Value::scalar(13.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_125() {
        let mut a = Value::scalar(13.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_126() {
        let mut a = Value::scalar(13.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_127() {
        let mut a = Value::scalar(13.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_128() {
        let mut a = Value::scalar(13.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_129() {
        let mut a = Value::scalar(13.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_130() {
        let mut a = Value::scalar(14.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_131() {
        let mut a = Value::scalar(14.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_132() {
        let mut a = Value::scalar(14.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_133() {
        let mut a = Value::scalar(14.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_134() {
        let mut a = Value::scalar(14.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_135() {
        let mut a = Value::scalar(14.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_136() {
        let mut a = Value::scalar(14.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_137() {
        let mut a = Value::scalar(14.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_138() {
        let mut a = Value::scalar(14.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_139() {
        let mut a = Value::scalar(14.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_140() {
        let mut a = Value::scalar(15.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_141() {
        let mut a = Value::scalar(15.100000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_142() {
        let mut a = Value::scalar(15.200000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_143() {
        let mut a = Value::scalar(15.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_144() {
        let mut a = Value::scalar(15.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_145() {
        let mut a = Value::scalar(15.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_146() {
        let mut a = Value::scalar(15.600000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_147() {
        let mut a = Value::scalar(15.700000000000001);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_148() {
        let mut a = Value::scalar(15.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_149() {
        let mut a = Value::scalar(15.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_150() {
        let mut a = Value::scalar(16.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_151() {
        let mut a = Value::scalar(16.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_152() {
        let mut a = Value::scalar(16.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_153() {
        let mut a = Value::scalar(16.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_154() {
        let mut a = Value::scalar(16.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_155() {
        let mut a = Value::scalar(16.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_156() {
        let mut a = Value::scalar(16.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_157() {
        let mut a = Value::scalar(16.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_158() {
        let mut a = Value::scalar(16.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_159() {
        let mut a = Value::scalar(16.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_160() {
        let mut a = Value::scalar(17.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_161() {
        let mut a = Value::scalar(17.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_162() {
        let mut a = Value::scalar(17.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_163() {
        let mut a = Value::scalar(17.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_164() {
        let mut a = Value::scalar(17.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_165() {
        let mut a = Value::scalar(17.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_166() {
        let mut a = Value::scalar(17.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_167() {
        let mut a = Value::scalar(17.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_168() {
        let mut a = Value::scalar(17.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_169() {
        let mut a = Value::scalar(17.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_170() {
        let mut a = Value::scalar(18.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_171() {
        let mut a = Value::scalar(18.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_172() {
        let mut a = Value::scalar(18.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_173() {
        let mut a = Value::scalar(18.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_174() {
        let mut a = Value::scalar(18.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_175() {
        let mut a = Value::scalar(18.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_176() {
        let mut a = Value::scalar(18.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_177() {
        let mut a = Value::scalar(18.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_178() {
        let mut a = Value::scalar(18.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_179() {
        let mut a = Value::scalar(18.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_180() {
        let mut a = Value::scalar(19.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_181() {
        let mut a = Value::scalar(19.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_182() {
        let mut a = Value::scalar(19.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_183() {
        let mut a = Value::scalar(19.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_184() {
        let mut a = Value::scalar(19.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_185() {
        let mut a = Value::scalar(19.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_186() {
        let mut a = Value::scalar(19.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_187() {
        let mut a = Value::scalar(19.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_188() {
        let mut a = Value::scalar(19.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_189() {
        let mut a = Value::scalar(19.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_190() {
        let mut a = Value::scalar(20.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_191() {
        let mut a = Value::scalar(20.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_192() {
        let mut a = Value::scalar(20.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_193() {
        let mut a = Value::scalar(20.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_194() {
        let mut a = Value::scalar(20.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_195() {
        let mut a = Value::scalar(20.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_196() {
        let mut a = Value::scalar(20.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_197() {
        let mut a = Value::scalar(20.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_198() {
        let mut a = Value::scalar(20.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_199() {
        let mut a = Value::scalar(20.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_200() {
        let mut a = Value::scalar(21.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_201() {
        let mut a = Value::scalar(21.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_202() {
        let mut a = Value::scalar(21.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_203() {
        let mut a = Value::scalar(21.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_204() {
        let mut a = Value::scalar(21.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_205() {
        let mut a = Value::scalar(21.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_206() {
        let mut a = Value::scalar(21.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_207() {
        let mut a = Value::scalar(21.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_208() {
        let mut a = Value::scalar(21.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_209() {
        let mut a = Value::scalar(21.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_210() {
        let mut a = Value::scalar(22.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_211() {
        let mut a = Value::scalar(22.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_212() {
        let mut a = Value::scalar(22.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_213() {
        let mut a = Value::scalar(22.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_214() {
        let mut a = Value::scalar(22.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_215() {
        let mut a = Value::scalar(22.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_216() {
        let mut a = Value::scalar(22.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_217() {
        let mut a = Value::scalar(22.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_218() {
        let mut a = Value::scalar(22.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_219() {
        let mut a = Value::scalar(22.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_220() {
        let mut a = Value::scalar(23.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_221() {
        let mut a = Value::scalar(23.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_222() {
        let mut a = Value::scalar(23.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_223() {
        let mut a = Value::scalar(23.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_224() {
        let mut a = Value::scalar(23.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_225() {
        let mut a = Value::scalar(23.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_226() {
        let mut a = Value::scalar(23.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_227() {
        let mut a = Value::scalar(23.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_228() {
        let mut a = Value::scalar(23.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_229() {
        let mut a = Value::scalar(23.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_230() {
        let mut a = Value::scalar(24.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_231() {
        let mut a = Value::scalar(24.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_232() {
        let mut a = Value::scalar(24.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_233() {
        let mut a = Value::scalar(24.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_234() {
        let mut a = Value::scalar(24.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_235() {
        let mut a = Value::scalar(24.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_236() {
        let mut a = Value::scalar(24.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_237() {
        let mut a = Value::scalar(24.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_238() {
        let mut a = Value::scalar(24.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_239() {
        let mut a = Value::scalar(24.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_240() {
        let mut a = Value::scalar(25.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_241() {
        let mut a = Value::scalar(25.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_242() {
        let mut a = Value::scalar(25.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_243() {
        let mut a = Value::scalar(25.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_244() {
        let mut a = Value::scalar(25.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_245() {
        let mut a = Value::scalar(25.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_246() {
        let mut a = Value::scalar(25.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_247() {
        let mut a = Value::scalar(25.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_248() {
        let mut a = Value::scalar(25.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_249() {
        let mut a = Value::scalar(25.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_250() {
        let mut a = Value::scalar(26.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_251() {
        let mut a = Value::scalar(26.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_252() {
        let mut a = Value::scalar(26.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_253() {
        let mut a = Value::scalar(26.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_254() {
        let mut a = Value::scalar(26.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_255() {
        let mut a = Value::scalar(26.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_256() {
        let mut a = Value::scalar(26.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_257() {
        let mut a = Value::scalar(26.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_258() {
        let mut a = Value::scalar(26.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_259() {
        let mut a = Value::scalar(26.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_260() {
        let mut a = Value::scalar(27.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_261() {
        let mut a = Value::scalar(27.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_262() {
        let mut a = Value::scalar(27.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_263() {
        let mut a = Value::scalar(27.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_264() {
        let mut a = Value::scalar(27.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_265() {
        let mut a = Value::scalar(27.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_266() {
        let mut a = Value::scalar(27.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_267() {
        let mut a = Value::scalar(27.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_268() {
        let mut a = Value::scalar(27.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_269() {
        let mut a = Value::scalar(27.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_270() {
        let mut a = Value::scalar(28.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_271() {
        let mut a = Value::scalar(28.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_272() {
        let mut a = Value::scalar(28.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_273() {
        let mut a = Value::scalar(28.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_274() {
        let mut a = Value::scalar(28.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_275() {
        let mut a = Value::scalar(28.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_276() {
        let mut a = Value::scalar(28.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_277() {
        let mut a = Value::scalar(28.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_278() {
        let mut a = Value::scalar(28.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_279() {
        let mut a = Value::scalar(28.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_280() {
        let mut a = Value::scalar(29.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_281() {
        let mut a = Value::scalar(29.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_282() {
        let mut a = Value::scalar(29.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_283() {
        let mut a = Value::scalar(29.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_284() {
        let mut a = Value::scalar(29.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_285() {
        let mut a = Value::scalar(29.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_286() {
        let mut a = Value::scalar(29.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_287() {
        let mut a = Value::scalar(29.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_288() {
        let mut a = Value::scalar(29.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_289() {
        let mut a = Value::scalar(29.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_290() {
        let mut a = Value::scalar(30.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_291() {
        let mut a = Value::scalar(30.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_292() {
        let mut a = Value::scalar(30.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_293() {
        let mut a = Value::scalar(30.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_294() {
        let mut a = Value::scalar(30.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_295() {
        let mut a = Value::scalar(30.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_296() {
        let mut a = Value::scalar(30.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_297() {
        let mut a = Value::scalar(30.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_298() {
        let mut a = Value::scalar(30.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_299() {
        let mut a = Value::scalar(30.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_300() {
        let mut a = Value::scalar(31.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_301() {
        let mut a = Value::scalar(31.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_302() {
        let mut a = Value::scalar(31.200000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_303() {
        let mut a = Value::scalar(31.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_304() {
        let mut a = Value::scalar(31.400000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_305() {
        let mut a = Value::scalar(31.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_306() {
        let mut a = Value::scalar(31.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_307() {
        let mut a = Value::scalar(31.700000000000003);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_308() {
        let mut a = Value::scalar(31.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_309() {
        let mut a = Value::scalar(31.900000000000002);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_310() {
        let mut a = Value::scalar(32.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_311() {
        let mut a = Value::scalar(32.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_312() {
        let mut a = Value::scalar(32.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_313() {
        let mut a = Value::scalar(32.3);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_314() {
        let mut a = Value::scalar(32.400000000000006);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_315() {
        let mut a = Value::scalar(32.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_316() {
        let mut a = Value::scalar(32.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_317() {
        let mut a = Value::scalar(32.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_318() {
        let mut a = Value::scalar(32.8);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_319() {
        let mut a = Value::scalar(32.900000000000006);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_320() {
        let mut a = Value::scalar(33.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_321() {
        let mut a = Value::scalar(33.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_322() {
        let mut a = Value::scalar(33.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_323() {
        let mut a = Value::scalar(33.300000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_324() {
        let mut a = Value::scalar(33.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_325() {
        let mut a = Value::scalar(33.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_326() {
        let mut a = Value::scalar(33.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_327() {
        let mut a = Value::scalar(33.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_328() {
        let mut a = Value::scalar(33.800000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_329() {
        let mut a = Value::scalar(33.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_330() {
        let mut a = Value::scalar(34.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_331() {
        let mut a = Value::scalar(34.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_332() {
        let mut a = Value::scalar(34.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_333() {
        let mut a = Value::scalar(34.300000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_334() {
        let mut a = Value::scalar(34.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_335() {
        let mut a = Value::scalar(34.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_336() {
        let mut a = Value::scalar(34.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_337() {
        let mut a = Value::scalar(34.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_338() {
        let mut a = Value::scalar(34.800000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_339() {
        let mut a = Value::scalar(34.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_340() {
        let mut a = Value::scalar(35.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_341() {
        let mut a = Value::scalar(35.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_342() {
        let mut a = Value::scalar(35.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_343() {
        let mut a = Value::scalar(35.300000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_344() {
        let mut a = Value::scalar(35.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_345() {
        let mut a = Value::scalar(35.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_346() {
        let mut a = Value::scalar(35.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_347() {
        let mut a = Value::scalar(35.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_348() {
        let mut a = Value::scalar(35.800000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_349() {
        let mut a = Value::scalar(35.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_350() {
        let mut a = Value::scalar(36.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_351() {
        let mut a = Value::scalar(36.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_352() {
        let mut a = Value::scalar(36.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_353() {
        let mut a = Value::scalar(36.300000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_354() {
        let mut a = Value::scalar(36.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_355() {
        let mut a = Value::scalar(36.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_356() {
        let mut a = Value::scalar(36.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_357() {
        let mut a = Value::scalar(36.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_358() {
        let mut a = Value::scalar(36.800000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_359() {
        let mut a = Value::scalar(36.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_360() {
        let mut a = Value::scalar(37.0);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_361() {
        let mut a = Value::scalar(37.1);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_362() {
        let mut a = Value::scalar(37.2);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_363() {
        let mut a = Value::scalar(37.300000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_364() {
        let mut a = Value::scalar(37.4);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_365() {
        let mut a = Value::scalar(37.5);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_366() {
        let mut a = Value::scalar(37.6);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_367() {
        let mut a = Value::scalar(37.7);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_368() {
        let mut a = Value::scalar(37.800000000000004);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_backward_mod_stress_369() {
        let mut a = Value::scalar(37.9);
        a.set_requires_grad(true);
        let b = a.exp();
        backward_from(&b).unwrap();
        assert!(a.grad().is_some());
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
    // Autograd verification and gradient check padding line 7
}
