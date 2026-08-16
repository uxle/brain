//! # Binary Forward Differentiable Operations
//!
//! Forward operations taking two `Value` inputs and attaching the corresponding `GradFn`.

use crate::grad_fns::GradFn;
use crate::value::Value;
use brain_core::tensor::arithmetic as arith_t;
use std::sync::Arc;

/// Elementwise addition: `a + b`.
pub fn add(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::add(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Add(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise subtraction: `a - b`.
pub fn sub(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::sub(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Sub(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise multiplication: `a * b`.
pub fn mul(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::mul(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Mul(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise division: `a / b`.
pub fn div(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::div(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Div(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise power: `a ^ b`.
pub fn pow(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::pow_tensors(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Pow(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Matrix multiplication: `a @ b`.
pub fn matmul(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::matmul(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::MatMul(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_binary_ops_stress_001() {
        let mut a = Value::scalar(2.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.1);
        assert_eq!(d.data().get(0), (2.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_002() {
        let mut a = Value::scalar(2.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.2);
        assert_eq!(d.data().get(0), (2.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_003() {
        let mut a = Value::scalar(2.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.3);
        assert_eq!(d.data().get(0), (2.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_004() {
        let mut a = Value::scalar(2.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.4);
        assert_eq!(d.data().get(0), (2.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_005() {
        let mut a = Value::scalar(2.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.5);
        assert_eq!(d.data().get(0), (2.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_006() {
        let mut a = Value::scalar(2.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.6);
        assert_eq!(d.data().get(0), (2.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_007() {
        let mut a = Value::scalar(2.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.7);
        assert_eq!(d.data().get(0), (2.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_008() {
        let mut a = Value::scalar(2.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.8);
        assert_eq!(d.data().get(0), (2.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_009() {
        let mut a = Value::scalar(2.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 5.9);
        assert_eq!(d.data().get(0), (2.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_010() {
        let mut a = Value::scalar(3.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.0);
        assert_eq!(d.data().get(0), (3.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_011() {
        let mut a = Value::scalar(3.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.1);
        assert_eq!(d.data().get(0), (3.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_012() {
        let mut a = Value::scalar(3.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.2);
        assert_eq!(d.data().get(0), (3.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_013() {
        let mut a = Value::scalar(3.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.3);
        assert_eq!(d.data().get(0), (3.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_014() {
        let mut a = Value::scalar(3.4000000000000004);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.4);
        assert_eq!(d.data().get(0), (3.4000000000000004) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_015() {
        let mut a = Value::scalar(3.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.5);
        assert_eq!(d.data().get(0), (3.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_016() {
        let mut a = Value::scalar(3.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.6);
        assert_eq!(d.data().get(0), (3.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_017() {
        let mut a = Value::scalar(3.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.7);
        assert_eq!(d.data().get(0), (3.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_018() {
        let mut a = Value::scalar(3.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.8);
        assert_eq!(d.data().get(0), (3.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_019() {
        let mut a = Value::scalar(3.9000000000000004);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 6.9);
        assert_eq!(d.data().get(0), (3.9000000000000004) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_020() {
        let mut a = Value::scalar(4.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.0);
        assert_eq!(d.data().get(0), (4.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_021() {
        let mut a = Value::scalar(4.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.1);
        assert_eq!(d.data().get(0), (4.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_022() {
        let mut a = Value::scalar(4.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.2);
        assert_eq!(d.data().get(0), (4.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_023() {
        let mut a = Value::scalar(4.300000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.300000000000001);
        assert_eq!(d.data().get(0), (4.300000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_024() {
        let mut a = Value::scalar(4.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.4);
        assert_eq!(d.data().get(0), (4.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_025() {
        let mut a = Value::scalar(4.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.5);
        assert_eq!(d.data().get(0), (4.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_026() {
        let mut a = Value::scalar(4.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.6);
        assert_eq!(d.data().get(0), (4.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_027() {
        let mut a = Value::scalar(4.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.7);
        assert_eq!(d.data().get(0), (4.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_028() {
        let mut a = Value::scalar(4.800000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.800000000000001);
        assert_eq!(d.data().get(0), (4.800000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_029() {
        let mut a = Value::scalar(4.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 7.9);
        assert_eq!(d.data().get(0), (4.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_030() {
        let mut a = Value::scalar(5.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.0);
        assert_eq!(d.data().get(0), (5.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_031() {
        let mut a = Value::scalar(5.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.1);
        assert_eq!(d.data().get(0), (5.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_032() {
        let mut a = Value::scalar(5.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.2);
        assert_eq!(d.data().get(0), (5.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_033() {
        let mut a = Value::scalar(5.300000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.3);
        assert_eq!(d.data().get(0), (5.300000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_034() {
        let mut a = Value::scalar(5.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.4);
        assert_eq!(d.data().get(0), (5.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_035() {
        let mut a = Value::scalar(5.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.5);
        assert_eq!(d.data().get(0), (5.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_036() {
        let mut a = Value::scalar(5.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.6);
        assert_eq!(d.data().get(0), (5.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_037() {
        let mut a = Value::scalar(5.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.7);
        assert_eq!(d.data().get(0), (5.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_038() {
        let mut a = Value::scalar(5.800000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.8);
        assert_eq!(d.data().get(0), (5.800000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_039() {
        let mut a = Value::scalar(5.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 8.9);
        assert_eq!(d.data().get(0), (5.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_040() {
        let mut a = Value::scalar(6.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.0);
        assert_eq!(d.data().get(0), (6.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_041() {
        let mut a = Value::scalar(6.1000000000000005);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.100000000000001);
        assert_eq!(d.data().get(0), (6.1000000000000005) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_042() {
        let mut a = Value::scalar(6.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.2);
        assert_eq!(d.data().get(0), (6.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_043() {
        let mut a = Value::scalar(6.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.3);
        assert_eq!(d.data().get(0), (6.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_044() {
        let mut a = Value::scalar(6.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.4);
        assert_eq!(d.data().get(0), (6.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_045() {
        let mut a = Value::scalar(6.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.5);
        assert_eq!(d.data().get(0), (6.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_046() {
        let mut a = Value::scalar(6.6000000000000005);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.600000000000001);
        assert_eq!(d.data().get(0), (6.6000000000000005) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_047() {
        let mut a = Value::scalar(6.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.7);
        assert_eq!(d.data().get(0), (6.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_048() {
        let mut a = Value::scalar(6.800000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.8);
        assert_eq!(d.data().get(0), (6.800000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_049() {
        let mut a = Value::scalar(6.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 9.9);
        assert_eq!(d.data().get(0), (6.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_050() {
        let mut a = Value::scalar(7.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.0);
        assert_eq!(d.data().get(0), (7.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_051() {
        let mut a = Value::scalar(7.1000000000000005);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.100000000000001);
        assert_eq!(d.data().get(0), (7.1000000000000005) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_052() {
        let mut a = Value::scalar(7.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.2);
        assert_eq!(d.data().get(0), (7.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_053() {
        let mut a = Value::scalar(7.300000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.3);
        assert_eq!(d.data().get(0), (7.300000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_054() {
        let mut a = Value::scalar(7.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.4);
        assert_eq!(d.data().get(0), (7.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_055() {
        let mut a = Value::scalar(7.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.5);
        assert_eq!(d.data().get(0), (7.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_056() {
        let mut a = Value::scalar(7.6000000000000005);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.600000000000001);
        assert_eq!(d.data().get(0), (7.6000000000000005) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_057() {
        let mut a = Value::scalar(7.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.7);
        assert_eq!(d.data().get(0), (7.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_058() {
        let mut a = Value::scalar(7.800000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.8);
        assert_eq!(d.data().get(0), (7.800000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_059() {
        let mut a = Value::scalar(7.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 10.9);
        assert_eq!(d.data().get(0), (7.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_060() {
        let mut a = Value::scalar(8.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.0);
        assert_eq!(d.data().get(0), (8.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_061() {
        let mut a = Value::scalar(8.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.100000000000001);
        assert_eq!(d.data().get(0), (8.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_062() {
        let mut a = Value::scalar(8.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.2);
        assert_eq!(d.data().get(0), (8.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_063() {
        let mut a = Value::scalar(8.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.3);
        assert_eq!(d.data().get(0), (8.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_064() {
        let mut a = Value::scalar(8.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.4);
        assert_eq!(d.data().get(0), (8.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_065() {
        let mut a = Value::scalar(8.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.5);
        assert_eq!(d.data().get(0), (8.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_066() {
        let mut a = Value::scalar(8.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.600000000000001);
        assert_eq!(d.data().get(0), (8.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_067() {
        let mut a = Value::scalar(8.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.7);
        assert_eq!(d.data().get(0), (8.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_068() {
        let mut a = Value::scalar(8.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.8);
        assert_eq!(d.data().get(0), (8.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_069() {
        let mut a = Value::scalar(8.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 11.9);
        assert_eq!(d.data().get(0), (8.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_070() {
        let mut a = Value::scalar(9.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.0);
        assert_eq!(d.data().get(0), (9.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_071() {
        let mut a = Value::scalar(9.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.100000000000001);
        assert_eq!(d.data().get(0), (9.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_072() {
        let mut a = Value::scalar(9.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.2);
        assert_eq!(d.data().get(0), (9.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_073() {
        let mut a = Value::scalar(9.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.3);
        assert_eq!(d.data().get(0), (9.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_074() {
        let mut a = Value::scalar(9.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.4);
        assert_eq!(d.data().get(0), (9.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_075() {
        let mut a = Value::scalar(9.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.5);
        assert_eq!(d.data().get(0), (9.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_076() {
        let mut a = Value::scalar(9.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.600000000000001);
        assert_eq!(d.data().get(0), (9.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_077() {
        let mut a = Value::scalar(9.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.7);
        assert_eq!(d.data().get(0), (9.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_078() {
        let mut a = Value::scalar(9.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.8);
        assert_eq!(d.data().get(0), (9.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_079() {
        let mut a = Value::scalar(9.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 12.9);
        assert_eq!(d.data().get(0), (9.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_080() {
        let mut a = Value::scalar(10.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.0);
        assert_eq!(d.data().get(0), (10.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_081() {
        let mut a = Value::scalar(10.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.1);
        assert_eq!(d.data().get(0), (10.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_082() {
        let mut a = Value::scalar(10.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.200000000000001);
        assert_eq!(d.data().get(0), (10.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_083() {
        let mut a = Value::scalar(10.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.3);
        assert_eq!(d.data().get(0), (10.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_084() {
        let mut a = Value::scalar(10.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.4);
        assert_eq!(d.data().get(0), (10.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_085() {
        let mut a = Value::scalar(10.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.5);
        assert_eq!(d.data().get(0), (10.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_086() {
        let mut a = Value::scalar(10.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.6);
        assert_eq!(d.data().get(0), (10.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_087() {
        let mut a = Value::scalar(10.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.700000000000001);
        assert_eq!(d.data().get(0), (10.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_088() {
        let mut a = Value::scalar(10.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.8);
        assert_eq!(d.data().get(0), (10.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_089() {
        let mut a = Value::scalar(10.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 13.9);
        assert_eq!(d.data().get(0), (10.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_090() {
        let mut a = Value::scalar(11.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.0);
        assert_eq!(d.data().get(0), (11.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_091() {
        let mut a = Value::scalar(11.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.1);
        assert_eq!(d.data().get(0), (11.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_092() {
        let mut a = Value::scalar(11.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.200000000000001);
        assert_eq!(d.data().get(0), (11.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_093() {
        let mut a = Value::scalar(11.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.3);
        assert_eq!(d.data().get(0), (11.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_094() {
        let mut a = Value::scalar(11.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.4);
        assert_eq!(d.data().get(0), (11.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_095() {
        let mut a = Value::scalar(11.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.5);
        assert_eq!(d.data().get(0), (11.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_096() {
        let mut a = Value::scalar(11.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.600000000000001);
        assert_eq!(d.data().get(0), (11.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_097() {
        let mut a = Value::scalar(11.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.700000000000001);
        assert_eq!(d.data().get(0), (11.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_098() {
        let mut a = Value::scalar(11.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.8);
        assert_eq!(d.data().get(0), (11.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_099() {
        let mut a = Value::scalar(11.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 14.9);
        assert_eq!(d.data().get(0), (11.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_100() {
        let mut a = Value::scalar(12.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.0);
        assert_eq!(d.data().get(0), (12.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_101() {
        let mut a = Value::scalar(12.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.100000000000001);
        assert_eq!(d.data().get(0), (12.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_102() {
        let mut a = Value::scalar(12.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.200000000000001);
        assert_eq!(d.data().get(0), (12.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_103() {
        let mut a = Value::scalar(12.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.3);
        assert_eq!(d.data().get(0), (12.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_104() {
        let mut a = Value::scalar(12.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.4);
        assert_eq!(d.data().get(0), (12.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_105() {
        let mut a = Value::scalar(12.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.5);
        assert_eq!(d.data().get(0), (12.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_106() {
        let mut a = Value::scalar(12.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.600000000000001);
        assert_eq!(d.data().get(0), (12.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_107() {
        let mut a = Value::scalar(12.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.700000000000001);
        assert_eq!(d.data().get(0), (12.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_108() {
        let mut a = Value::scalar(12.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.8);
        assert_eq!(d.data().get(0), (12.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_109() {
        let mut a = Value::scalar(12.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 15.9);
        assert_eq!(d.data().get(0), (12.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_110() {
        let mut a = Value::scalar(13.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.0);
        assert_eq!(d.data().get(0), (13.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_111() {
        let mut a = Value::scalar(13.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.1);
        assert_eq!(d.data().get(0), (13.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_112() {
        let mut a = Value::scalar(13.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.200000000000003);
        assert_eq!(d.data().get(0), (13.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_113() {
        let mut a = Value::scalar(13.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.3);
        assert_eq!(d.data().get(0), (13.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_114() {
        let mut a = Value::scalar(13.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.4);
        assert_eq!(d.data().get(0), (13.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_115() {
        let mut a = Value::scalar(13.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.5);
        assert_eq!(d.data().get(0), (13.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_116() {
        let mut a = Value::scalar(13.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.6);
        assert_eq!(d.data().get(0), (13.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_117() {
        let mut a = Value::scalar(13.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.700000000000003);
        assert_eq!(d.data().get(0), (13.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_118() {
        let mut a = Value::scalar(13.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.8);
        assert_eq!(d.data().get(0), (13.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_119() {
        let mut a = Value::scalar(13.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 16.9);
        assert_eq!(d.data().get(0), (13.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_120() {
        let mut a = Value::scalar(14.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.0);
        assert_eq!(d.data().get(0), (14.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_121() {
        let mut a = Value::scalar(14.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.1);
        assert_eq!(d.data().get(0), (14.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_122() {
        let mut a = Value::scalar(14.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.200000000000003);
        assert_eq!(d.data().get(0), (14.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_123() {
        let mut a = Value::scalar(14.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.3);
        assert_eq!(d.data().get(0), (14.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_124() {
        let mut a = Value::scalar(14.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.4);
        assert_eq!(d.data().get(0), (14.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_125() {
        let mut a = Value::scalar(14.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.5);
        assert_eq!(d.data().get(0), (14.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_126() {
        let mut a = Value::scalar(14.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.6);
        assert_eq!(d.data().get(0), (14.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_127() {
        let mut a = Value::scalar(14.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.700000000000003);
        assert_eq!(d.data().get(0), (14.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_128() {
        let mut a = Value::scalar(14.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.8);
        assert_eq!(d.data().get(0), (14.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_129() {
        let mut a = Value::scalar(14.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 17.9);
        assert_eq!(d.data().get(0), (14.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_130() {
        let mut a = Value::scalar(15.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.0);
        assert_eq!(d.data().get(0), (15.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_131() {
        let mut a = Value::scalar(15.100000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.1);
        assert_eq!(d.data().get(0), (15.100000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_132() {
        let mut a = Value::scalar(15.200000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.200000000000003);
        assert_eq!(d.data().get(0), (15.200000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_133() {
        let mut a = Value::scalar(15.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.3);
        assert_eq!(d.data().get(0), (15.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_134() {
        let mut a = Value::scalar(15.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.4);
        assert_eq!(d.data().get(0), (15.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_135() {
        let mut a = Value::scalar(15.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.5);
        assert_eq!(d.data().get(0), (15.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_136() {
        let mut a = Value::scalar(15.600000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.6);
        assert_eq!(d.data().get(0), (15.600000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_137() {
        let mut a = Value::scalar(15.700000000000001);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.700000000000003);
        assert_eq!(d.data().get(0), (15.700000000000001) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_138() {
        let mut a = Value::scalar(15.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.8);
        assert_eq!(d.data().get(0), (15.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_139() {
        let mut a = Value::scalar(15.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 18.9);
        assert_eq!(d.data().get(0), (15.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_140() {
        let mut a = Value::scalar(16.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.0);
        assert_eq!(d.data().get(0), (16.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_141() {
        let mut a = Value::scalar(16.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.1);
        assert_eq!(d.data().get(0), (16.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_142() {
        let mut a = Value::scalar(16.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.200000000000003);
        assert_eq!(d.data().get(0), (16.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_143() {
        let mut a = Value::scalar(16.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.3);
        assert_eq!(d.data().get(0), (16.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_144() {
        let mut a = Value::scalar(16.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.4);
        assert_eq!(d.data().get(0), (16.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_145() {
        let mut a = Value::scalar(16.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.5);
        assert_eq!(d.data().get(0), (16.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_146() {
        let mut a = Value::scalar(16.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.6);
        assert_eq!(d.data().get(0), (16.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_147() {
        let mut a = Value::scalar(16.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.700000000000003);
        assert_eq!(d.data().get(0), (16.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_148() {
        let mut a = Value::scalar(16.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.8);
        assert_eq!(d.data().get(0), (16.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_149() {
        let mut a = Value::scalar(16.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 19.9);
        assert_eq!(d.data().get(0), (16.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_150() {
        let mut a = Value::scalar(17.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.0);
        assert_eq!(d.data().get(0), (17.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_151() {
        let mut a = Value::scalar(17.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.1);
        assert_eq!(d.data().get(0), (17.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_152() {
        let mut a = Value::scalar(17.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.200000000000003);
        assert_eq!(d.data().get(0), (17.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_153() {
        let mut a = Value::scalar(17.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.3);
        assert_eq!(d.data().get(0), (17.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_154() {
        let mut a = Value::scalar(17.4);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.4);
        assert_eq!(d.data().get(0), (17.4) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_155() {
        let mut a = Value::scalar(17.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.5);
        assert_eq!(d.data().get(0), (17.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_156() {
        let mut a = Value::scalar(17.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.6);
        assert_eq!(d.data().get(0), (17.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_157() {
        let mut a = Value::scalar(17.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.700000000000003);
        assert_eq!(d.data().get(0), (17.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_158() {
        let mut a = Value::scalar(17.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.8);
        assert_eq!(d.data().get(0), (17.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_159() {
        let mut a = Value::scalar(17.9);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 20.9);
        assert_eq!(d.data().get(0), (17.9) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_160() {
        let mut a = Value::scalar(18.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.0);
        assert_eq!(d.data().get(0), (18.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_161() {
        let mut a = Value::scalar(18.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.1);
        assert_eq!(d.data().get(0), (18.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_162() {
        let mut a = Value::scalar(18.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.2);
        assert_eq!(d.data().get(0), (18.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_163() {
        let mut a = Value::scalar(18.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.3);
        assert_eq!(d.data().get(0), (18.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_164() {
        let mut a = Value::scalar(18.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.400000000000002);
        assert_eq!(d.data().get(0), (18.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_165() {
        let mut a = Value::scalar(18.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.5);
        assert_eq!(d.data().get(0), (18.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_166() {
        let mut a = Value::scalar(18.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.6);
        assert_eq!(d.data().get(0), (18.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_167() {
        let mut a = Value::scalar(18.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.7);
        assert_eq!(d.data().get(0), (18.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_168() {
        let mut a = Value::scalar(18.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.8);
        assert_eq!(d.data().get(0), (18.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_169() {
        let mut a = Value::scalar(18.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 21.900000000000002);
        assert_eq!(d.data().get(0), (18.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_170() {
        let mut a = Value::scalar(19.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.0);
        assert_eq!(d.data().get(0), (19.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_171() {
        let mut a = Value::scalar(19.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.1);
        assert_eq!(d.data().get(0), (19.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_172() {
        let mut a = Value::scalar(19.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.2);
        assert_eq!(d.data().get(0), (19.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_173() {
        let mut a = Value::scalar(19.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.3);
        assert_eq!(d.data().get(0), (19.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_174() {
        let mut a = Value::scalar(19.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.400000000000002);
        assert_eq!(d.data().get(0), (19.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_175() {
        let mut a = Value::scalar(19.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.5);
        assert_eq!(d.data().get(0), (19.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_176() {
        let mut a = Value::scalar(19.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.6);
        assert_eq!(d.data().get(0), (19.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_177() {
        let mut a = Value::scalar(19.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.7);
        assert_eq!(d.data().get(0), (19.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_178() {
        let mut a = Value::scalar(19.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.8);
        assert_eq!(d.data().get(0), (19.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_179() {
        let mut a = Value::scalar(19.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 22.900000000000002);
        assert_eq!(d.data().get(0), (19.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_180() {
        let mut a = Value::scalar(20.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.0);
        assert_eq!(d.data().get(0), (20.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_181() {
        let mut a = Value::scalar(20.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.1);
        assert_eq!(d.data().get(0), (20.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_182() {
        let mut a = Value::scalar(20.2);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.2);
        assert_eq!(d.data().get(0), (20.2) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_183() {
        let mut a = Value::scalar(20.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.3);
        assert_eq!(d.data().get(0), (20.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_184() {
        let mut a = Value::scalar(20.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.400000000000002);
        assert_eq!(d.data().get(0), (20.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_185() {
        let mut a = Value::scalar(20.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.5);
        assert_eq!(d.data().get(0), (20.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_186() {
        let mut a = Value::scalar(20.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.6);
        assert_eq!(d.data().get(0), (20.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_187() {
        let mut a = Value::scalar(20.7);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.7);
        assert_eq!(d.data().get(0), (20.7) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_188() {
        let mut a = Value::scalar(20.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.8);
        assert_eq!(d.data().get(0), (20.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_189() {
        let mut a = Value::scalar(20.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 23.900000000000002);
        assert_eq!(d.data().get(0), (20.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_190() {
        let mut a = Value::scalar(21.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.0);
        assert_eq!(d.data().get(0), (21.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_191() {
        let mut a = Value::scalar(21.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.1);
        assert_eq!(d.data().get(0), (21.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_192() {
        let mut a = Value::scalar(21.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.200000000000003);
        assert_eq!(d.data().get(0), (21.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_193() {
        let mut a = Value::scalar(21.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.3);
        assert_eq!(d.data().get(0), (21.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_194() {
        let mut a = Value::scalar(21.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.400000000000002);
        assert_eq!(d.data().get(0), (21.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_195() {
        let mut a = Value::scalar(21.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.5);
        assert_eq!(d.data().get(0), (21.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_196() {
        let mut a = Value::scalar(21.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.6);
        assert_eq!(d.data().get(0), (21.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_197() {
        let mut a = Value::scalar(21.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.700000000000003);
        assert_eq!(d.data().get(0), (21.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_198() {
        let mut a = Value::scalar(21.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.8);
        assert_eq!(d.data().get(0), (21.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_199() {
        let mut a = Value::scalar(21.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 24.900000000000002);
        assert_eq!(d.data().get(0), (21.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_200() {
        let mut a = Value::scalar(22.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.0);
        assert_eq!(d.data().get(0), (22.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_201() {
        let mut a = Value::scalar(22.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.1);
        assert_eq!(d.data().get(0), (22.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_202() {
        let mut a = Value::scalar(22.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.200000000000003);
        assert_eq!(d.data().get(0), (22.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_203() {
        let mut a = Value::scalar(22.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.3);
        assert_eq!(d.data().get(0), (22.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_204() {
        let mut a = Value::scalar(22.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.400000000000002);
        assert_eq!(d.data().get(0), (22.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_205() {
        let mut a = Value::scalar(22.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.5);
        assert_eq!(d.data().get(0), (22.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_206() {
        let mut a = Value::scalar(22.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.6);
        assert_eq!(d.data().get(0), (22.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_207() {
        let mut a = Value::scalar(22.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.700000000000003);
        assert_eq!(d.data().get(0), (22.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_208() {
        let mut a = Value::scalar(22.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.8);
        assert_eq!(d.data().get(0), (22.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_209() {
        let mut a = Value::scalar(22.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 25.900000000000002);
        assert_eq!(d.data().get(0), (22.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_210() {
        let mut a = Value::scalar(23.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.0);
        assert_eq!(d.data().get(0), (23.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_211() {
        let mut a = Value::scalar(23.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.1);
        assert_eq!(d.data().get(0), (23.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_212() {
        let mut a = Value::scalar(23.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.200000000000003);
        assert_eq!(d.data().get(0), (23.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_213() {
        let mut a = Value::scalar(23.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.3);
        assert_eq!(d.data().get(0), (23.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_214() {
        let mut a = Value::scalar(23.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.400000000000002);
        assert_eq!(d.data().get(0), (23.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_215() {
        let mut a = Value::scalar(23.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.5);
        assert_eq!(d.data().get(0), (23.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_216() {
        let mut a = Value::scalar(23.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.6);
        assert_eq!(d.data().get(0), (23.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_217() {
        let mut a = Value::scalar(23.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.700000000000003);
        assert_eq!(d.data().get(0), (23.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_218() {
        let mut a = Value::scalar(23.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.8);
        assert_eq!(d.data().get(0), (23.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_219() {
        let mut a = Value::scalar(23.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 26.900000000000002);
        assert_eq!(d.data().get(0), (23.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_220() {
        let mut a = Value::scalar(24.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.0);
        assert_eq!(d.data().get(0), (24.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_221() {
        let mut a = Value::scalar(24.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.1);
        assert_eq!(d.data().get(0), (24.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_222() {
        let mut a = Value::scalar(24.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.200000000000003);
        assert_eq!(d.data().get(0), (24.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_223() {
        let mut a = Value::scalar(24.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.3);
        assert_eq!(d.data().get(0), (24.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_224() {
        let mut a = Value::scalar(24.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.400000000000002);
        assert_eq!(d.data().get(0), (24.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_225() {
        let mut a = Value::scalar(24.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.5);
        assert_eq!(d.data().get(0), (24.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_226() {
        let mut a = Value::scalar(24.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.6);
        assert_eq!(d.data().get(0), (24.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_227() {
        let mut a = Value::scalar(24.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.700000000000003);
        assert_eq!(d.data().get(0), (24.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_228() {
        let mut a = Value::scalar(24.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.8);
        assert_eq!(d.data().get(0), (24.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_229() {
        let mut a = Value::scalar(24.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 27.900000000000002);
        assert_eq!(d.data().get(0), (24.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_230() {
        let mut a = Value::scalar(25.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.0);
        assert_eq!(d.data().get(0), (25.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_231() {
        let mut a = Value::scalar(25.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.1);
        assert_eq!(d.data().get(0), (25.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_232() {
        let mut a = Value::scalar(25.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.200000000000003);
        assert_eq!(d.data().get(0), (25.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_233() {
        let mut a = Value::scalar(25.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.3);
        assert_eq!(d.data().get(0), (25.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_234() {
        let mut a = Value::scalar(25.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.400000000000002);
        assert_eq!(d.data().get(0), (25.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_235() {
        let mut a = Value::scalar(25.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.5);
        assert_eq!(d.data().get(0), (25.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_236() {
        let mut a = Value::scalar(25.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.6);
        assert_eq!(d.data().get(0), (25.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_237() {
        let mut a = Value::scalar(25.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.700000000000003);
        assert_eq!(d.data().get(0), (25.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_238() {
        let mut a = Value::scalar(25.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.8);
        assert_eq!(d.data().get(0), (25.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_239() {
        let mut a = Value::scalar(25.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 28.900000000000002);
        assert_eq!(d.data().get(0), (25.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_240() {
        let mut a = Value::scalar(26.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.0);
        assert_eq!(d.data().get(0), (26.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_241() {
        let mut a = Value::scalar(26.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.1);
        assert_eq!(d.data().get(0), (26.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_242() {
        let mut a = Value::scalar(26.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.200000000000003);
        assert_eq!(d.data().get(0), (26.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_243() {
        let mut a = Value::scalar(26.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.3);
        assert_eq!(d.data().get(0), (26.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_244() {
        let mut a = Value::scalar(26.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.400000000000002);
        assert_eq!(d.data().get(0), (26.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_245() {
        let mut a = Value::scalar(26.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.5);
        assert_eq!(d.data().get(0), (26.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_246() {
        let mut a = Value::scalar(26.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.6);
        assert_eq!(d.data().get(0), (26.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_247() {
        let mut a = Value::scalar(26.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.700000000000003);
        assert_eq!(d.data().get(0), (26.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_248() {
        let mut a = Value::scalar(26.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.8);
        assert_eq!(d.data().get(0), (26.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_249() {
        let mut a = Value::scalar(26.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 29.900000000000002);
        assert_eq!(d.data().get(0), (26.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_250() {
        let mut a = Value::scalar(27.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.0);
        assert_eq!(d.data().get(0), (27.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_251() {
        let mut a = Value::scalar(27.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.1);
        assert_eq!(d.data().get(0), (27.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_252() {
        let mut a = Value::scalar(27.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.200000000000003);
        assert_eq!(d.data().get(0), (27.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_253() {
        let mut a = Value::scalar(27.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.3);
        assert_eq!(d.data().get(0), (27.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_254() {
        let mut a = Value::scalar(27.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.400000000000002);
        assert_eq!(d.data().get(0), (27.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_255() {
        let mut a = Value::scalar(27.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.5);
        assert_eq!(d.data().get(0), (27.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_256() {
        let mut a = Value::scalar(27.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.6);
        assert_eq!(d.data().get(0), (27.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_257() {
        let mut a = Value::scalar(27.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.700000000000003);
        assert_eq!(d.data().get(0), (27.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_258() {
        let mut a = Value::scalar(27.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.8);
        assert_eq!(d.data().get(0), (27.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_259() {
        let mut a = Value::scalar(27.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 30.900000000000002);
        assert_eq!(d.data().get(0), (27.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_260() {
        let mut a = Value::scalar(28.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.0);
        assert_eq!(d.data().get(0), (28.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_261() {
        let mut a = Value::scalar(28.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.1);
        assert_eq!(d.data().get(0), (28.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_262() {
        let mut a = Value::scalar(28.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.200000000000003);
        assert_eq!(d.data().get(0), (28.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_263() {
        let mut a = Value::scalar(28.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.3);
        assert_eq!(d.data().get(0), (28.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_264() {
        let mut a = Value::scalar(28.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.400000000000002);
        assert_eq!(d.data().get(0), (28.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_265() {
        let mut a = Value::scalar(28.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.5);
        assert_eq!(d.data().get(0), (28.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_266() {
        let mut a = Value::scalar(28.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.6);
        assert_eq!(d.data().get(0), (28.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_267() {
        let mut a = Value::scalar(28.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.700000000000003);
        assert_eq!(d.data().get(0), (28.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_268() {
        let mut a = Value::scalar(28.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.8);
        assert_eq!(d.data().get(0), (28.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_269() {
        let mut a = Value::scalar(28.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 31.900000000000002);
        assert_eq!(d.data().get(0), (28.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_270() {
        let mut a = Value::scalar(29.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.0);
        assert_eq!(d.data().get(0), (29.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_271() {
        let mut a = Value::scalar(29.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.1);
        assert_eq!(d.data().get(0), (29.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_272() {
        let mut a = Value::scalar(29.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.2);
        assert_eq!(d.data().get(0), (29.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_273() {
        let mut a = Value::scalar(29.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.3);
        assert_eq!(d.data().get(0), (29.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_274() {
        let mut a = Value::scalar(29.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.400000000000006);
        assert_eq!(d.data().get(0), (29.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_275() {
        let mut a = Value::scalar(29.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.5);
        assert_eq!(d.data().get(0), (29.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_276() {
        let mut a = Value::scalar(29.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.6);
        assert_eq!(d.data().get(0), (29.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_277() {
        let mut a = Value::scalar(29.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.7);
        assert_eq!(d.data().get(0), (29.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_278() {
        let mut a = Value::scalar(29.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.8);
        assert_eq!(d.data().get(0), (29.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_279() {
        let mut a = Value::scalar(29.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 32.900000000000006);
        assert_eq!(d.data().get(0), (29.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_280() {
        let mut a = Value::scalar(30.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.0);
        assert_eq!(d.data().get(0), (30.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_281() {
        let mut a = Value::scalar(30.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.1);
        assert_eq!(d.data().get(0), (30.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_282() {
        let mut a = Value::scalar(30.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.2);
        assert_eq!(d.data().get(0), (30.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_283() {
        let mut a = Value::scalar(30.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.3);
        assert_eq!(d.data().get(0), (30.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_284() {
        let mut a = Value::scalar(30.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.400000000000006);
        assert_eq!(d.data().get(0), (30.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_285() {
        let mut a = Value::scalar(30.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.5);
        assert_eq!(d.data().get(0), (30.5) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_286() {
        let mut a = Value::scalar(30.6);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.6);
        assert_eq!(d.data().get(0), (30.6) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_287() {
        let mut a = Value::scalar(30.700000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.7);
        assert_eq!(d.data().get(0), (30.700000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_288() {
        let mut a = Value::scalar(30.8);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.8);
        assert_eq!(d.data().get(0), (30.8) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_289() {
        let mut a = Value::scalar(30.900000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 33.900000000000006);
        assert_eq!(d.data().get(0), (30.900000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_290() {
        let mut a = Value::scalar(31.0);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.0);
        assert_eq!(d.data().get(0), (31.0) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_291() {
        let mut a = Value::scalar(31.1);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.1);
        assert_eq!(d.data().get(0), (31.1) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_292() {
        let mut a = Value::scalar(31.200000000000003);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.2);
        assert_eq!(d.data().get(0), (31.200000000000003) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_293() {
        let mut a = Value::scalar(31.3);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.3);
        assert_eq!(d.data().get(0), (31.3) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_294() {
        let mut a = Value::scalar(31.400000000000002);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.400000000000006);
        assert_eq!(d.data().get(0), (31.400000000000002) * 3.0);
    }

    #[test]
    fn test_binary_ops_stress_295() {
        let mut a = Value::scalar(31.5);
        a.set_requires_grad(true);
        let b = Value::scalar(3.0);
        let c = add(&a, &b);
        let d = mul(&a, &b);
        assert_eq!(c.data().get(0), 34.5);
        assert_eq!(d.data().get(0), (31.5) * 3.0);
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
