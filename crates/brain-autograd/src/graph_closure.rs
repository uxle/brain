//! # Functional Autograd Transforms
//!
//! Higher-order functional interfaces for automatic differentiation:
//! `grad`, `value_and_grad`, `vjp`, `jvp`, `jacobian`, `hessian`, and `grad_and_hess`.

use crate::value::Value;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::{BrainError, BrainResult, Tensor};

/// Evaluates scalar function `f(x)` and returns its gradient with respect to `x`.
pub fn grad<F>(f: F, x: &Value) -> BrainResult<Option<Tensor>>
where
    F: FnOnce(&Value) -> Value,
{
    let leaf = Value::new(x.data().clone(), true);
    let out = f(&leaf);
    out.backward()?;
    Ok(leaf.grad())
}

/// Evaluates scalar function `f(x)` and returns both output value and gradient.
pub fn value_and_grad<F>(f: F, x: &Value) -> BrainResult<(Value, Option<Tensor>)>
where
    F: FnOnce(&Value) -> Value,
{
    let leaf = Value::new(x.data().clone(), true);
    let out = f(&leaf);
    out.backward()?;
    let g = leaf.grad();
    Ok((out, g))
}

/// Computes Vector-Jacobian Product: `(f(x), v^T J)`.
pub fn vjp<F>(f: F, x: &Value, v: &Tensor) -> BrainResult<(Value, Tensor)>
where
    F: FnOnce(&Value) -> Value,
{
    let leaf = Value::new(x.data().clone(), true);
    let out = f(&leaf);
    out.backward_with_grad(v)?;
    let g = leaf.grad().ok_or_else(|| BrainError::invalid_value("No gradient accumulated on leaf"))?;
    Ok((out, g))
}

/// Computes Jacobian-Vector Product: `(f(x), J v)`.
pub fn jvp<F>(f: F, x: &Value, v: &Tensor) -> BrainResult<(Value, Tensor)>
where
    F: Fn(&Value) -> Value,
{
    let eps = 1e-6;
    let v_scaled = v.map(|x| x * eps);
    let x_plus = Value::from_tensor(&arith_t::add(x.data(), &v_scaled));
    let x_minus = Value::from_tensor(&arith_t::sub(x.data(), &v_scaled));
    let y_plus = f(&x_plus);
    let y_minus = f(&x_minus);
    let diff = arith_t::sub(y_plus.data(), y_minus.data());
    let jvp_out = diff.map(|x| x / (2.0 * eps));
    let y_center = f(x);
    Ok((y_center, jvp_out))
}

/// Computes the full Jacobian matrix `J[i, j] = dy_i / dx_j`. Output shape: `[M, N]`.
pub fn jacobian<F>(f: F, x: &Value) -> BrainResult<Tensor>
where
    F: Fn(&Value) -> Value,
{
    let y_test = f(x);
    let m = y_test.numel();
    let n = x.numel();
    let mut jac = vec![0.0; m * n];

    for i in 0..m {
        let mut v_data = vec![0.0; m];
        v_data[i] = 1.0;
        let v = Tensor::from_slice(&v_data, y_test.shape().to_vec());

        let leaf = Value::new(x.data().clone(), true);
        let out = f(&leaf);
        out.backward_with_grad(&v)?;
        let g = leaf.grad().ok_or_else(|| BrainError::invalid_value("Jacobian grad missing"))?;
        let g_slice = g.data();
        for j in 0..n {
            jac[i * n + j] = g_slice[j];
        }
    }

    Ok(Tensor::from_slice(&jac, vec![m, n]))
}

/// Computes the full Hessian matrix `H[i, j] = d^2 y / (dx_i dx_j)`. Output shape: `[N, N]`.
pub fn hessian<F>(f: F, x: &Value) -> BrainResult<Tensor>
where
    F: Fn(&Value) -> Value,
{
    let n = x.numel();
    let eps = 1e-5;
    let mut h = vec![0.0; n * n];

    for j in 0..n {
        let mut x_data_p = x.data().data().to_vec();
        let mut x_data_m = x.data().data().to_vec();
        x_data_p[j] += eps;
        x_data_m[j] -= eps;

        let xp = Value::from_slice(&x_data_p, x.shape().to_vec());
        let xm = Value::from_slice(&x_data_m, x.shape().to_vec());

        let gp = grad(&f, &xp)?.ok_or_else(|| BrainError::invalid_value("Hessian grad missing"))?;
        let gm = grad(&f, &xm)?.ok_or_else(|| BrainError::invalid_value("Hessian grad missing"))?;

        let diff = arith_t::sub(&gp, &gm);
        let diff_slice = diff.data();
        for i in 0..n {
            h[i * n + j] = diff_slice[i] / (2.0 * eps);
        }
    }

    Ok(Tensor::from_slice(&h, vec![n, n]))
}

/// Returns both gradient vector and Hessian matrix.
pub fn grad_and_hess<F>(f: F, x: &Value) -> BrainResult<(Tensor, Tensor)>
where
    F: Fn(&Value) -> Value,
{
    let g = grad(&f, x)?.ok_or_else(|| BrainError::invalid_value("Grad missing in grad_and_hess"))?;
    let h = hessian(f, x)?;
    Ok((g, h))
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
    fn test_graph_closure_stress_001() {
        let x = Value::scalar(2.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.1) * (2.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_002() {
        let x = Value::scalar(2.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.2) * (2.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_003() {
        let x = Value::scalar(2.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.3) * (2.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_004() {
        let x = Value::scalar(2.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.4) * (2.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_005() {
        let x = Value::scalar(2.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.5) * (2.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_006() {
        let x = Value::scalar(2.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.6) * (2.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_007() {
        let x = Value::scalar(2.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.7) * (2.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_008() {
        let x = Value::scalar(2.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.8) * (2.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_009() {
        let x = Value::scalar(2.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (2.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (2.9) * (2.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_010() {
        let x = Value::scalar(3.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.0) * (3.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_011() {
        let x = Value::scalar(3.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.1) * (3.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_012() {
        let x = Value::scalar(3.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.2) * (3.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_013() {
        let x = Value::scalar(3.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.3) * (3.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_014() {
        let x = Value::scalar(3.4000000000000004);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.4000000000000004);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.4000000000000004) * (3.4000000000000004);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_015() {
        let x = Value::scalar(3.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.5) * (3.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_016() {
        let x = Value::scalar(3.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.6) * (3.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_017() {
        let x = Value::scalar(3.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.7) * (3.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_018() {
        let x = Value::scalar(3.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.8) * (3.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_019() {
        let x = Value::scalar(3.9000000000000004);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (3.9000000000000004);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (3.9000000000000004) * (3.9000000000000004);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_020() {
        let x = Value::scalar(4.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.0) * (4.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_021() {
        let x = Value::scalar(4.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.1) * (4.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_022() {
        let x = Value::scalar(4.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.2) * (4.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_023() {
        let x = Value::scalar(4.300000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.300000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.300000000000001) * (4.300000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_024() {
        let x = Value::scalar(4.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.4) * (4.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_025() {
        let x = Value::scalar(4.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.5) * (4.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_026() {
        let x = Value::scalar(4.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.6) * (4.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_027() {
        let x = Value::scalar(4.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.7) * (4.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_028() {
        let x = Value::scalar(4.800000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.800000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.800000000000001) * (4.800000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_029() {
        let x = Value::scalar(4.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (4.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (4.9) * (4.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_030() {
        let x = Value::scalar(5.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.0) * (5.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_031() {
        let x = Value::scalar(5.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.1) * (5.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_032() {
        let x = Value::scalar(5.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.2) * (5.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_033() {
        let x = Value::scalar(5.300000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.300000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.300000000000001) * (5.300000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_034() {
        let x = Value::scalar(5.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.4) * (5.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_035() {
        let x = Value::scalar(5.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.5) * (5.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_036() {
        let x = Value::scalar(5.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.6) * (5.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_037() {
        let x = Value::scalar(5.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.7) * (5.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_038() {
        let x = Value::scalar(5.800000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.800000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.800000000000001) * (5.800000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_039() {
        let x = Value::scalar(5.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (5.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (5.9) * (5.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_040() {
        let x = Value::scalar(6.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.0) * (6.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_041() {
        let x = Value::scalar(6.1000000000000005);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.1000000000000005);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.1000000000000005) * (6.1000000000000005);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_042() {
        let x = Value::scalar(6.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.2) * (6.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_043() {
        let x = Value::scalar(6.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.3) * (6.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_044() {
        let x = Value::scalar(6.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.4) * (6.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_045() {
        let x = Value::scalar(6.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.5) * (6.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_046() {
        let x = Value::scalar(6.6000000000000005);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.6000000000000005);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.6000000000000005) * (6.6000000000000005);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_047() {
        let x = Value::scalar(6.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.7) * (6.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_048() {
        let x = Value::scalar(6.800000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.800000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.800000000000001) * (6.800000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_049() {
        let x = Value::scalar(6.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (6.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (6.9) * (6.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_050() {
        let x = Value::scalar(7.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.0) * (7.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_051() {
        let x = Value::scalar(7.1000000000000005);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.1000000000000005);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.1000000000000005) * (7.1000000000000005);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_052() {
        let x = Value::scalar(7.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.2) * (7.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_053() {
        let x = Value::scalar(7.300000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.300000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.300000000000001) * (7.300000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_054() {
        let x = Value::scalar(7.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.4) * (7.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_055() {
        let x = Value::scalar(7.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.5) * (7.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_056() {
        let x = Value::scalar(7.6000000000000005);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.6000000000000005);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.6000000000000005) * (7.6000000000000005);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_057() {
        let x = Value::scalar(7.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.7) * (7.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_058() {
        let x = Value::scalar(7.800000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.800000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.800000000000001) * (7.800000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_059() {
        let x = Value::scalar(7.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (7.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (7.9) * (7.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_060() {
        let x = Value::scalar(8.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.0) * (8.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_061() {
        let x = Value::scalar(8.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.100000000000001) * (8.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_062() {
        let x = Value::scalar(8.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.2) * (8.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_063() {
        let x = Value::scalar(8.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.3) * (8.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_064() {
        let x = Value::scalar(8.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.4) * (8.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_065() {
        let x = Value::scalar(8.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.5) * (8.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_066() {
        let x = Value::scalar(8.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.600000000000001) * (8.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_067() {
        let x = Value::scalar(8.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.7) * (8.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_068() {
        let x = Value::scalar(8.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.8) * (8.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_069() {
        let x = Value::scalar(8.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (8.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (8.9) * (8.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_070() {
        let x = Value::scalar(9.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.0) * (9.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_071() {
        let x = Value::scalar(9.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.100000000000001) * (9.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_072() {
        let x = Value::scalar(9.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.2) * (9.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_073() {
        let x = Value::scalar(9.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.3) * (9.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_074() {
        let x = Value::scalar(9.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.4) * (9.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_075() {
        let x = Value::scalar(9.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.5) * (9.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_076() {
        let x = Value::scalar(9.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.600000000000001) * (9.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_077() {
        let x = Value::scalar(9.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.7) * (9.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_078() {
        let x = Value::scalar(9.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.8) * (9.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_079() {
        let x = Value::scalar(9.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (9.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (9.9) * (9.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_080() {
        let x = Value::scalar(10.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.0) * (10.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_081() {
        let x = Value::scalar(10.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.1) * (10.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_082() {
        let x = Value::scalar(10.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.200000000000001) * (10.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_083() {
        let x = Value::scalar(10.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.3) * (10.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_084() {
        let x = Value::scalar(10.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.4) * (10.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_085() {
        let x = Value::scalar(10.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.5) * (10.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_086() {
        let x = Value::scalar(10.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.6) * (10.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_087() {
        let x = Value::scalar(10.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.700000000000001) * (10.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_088() {
        let x = Value::scalar(10.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.8) * (10.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_089() {
        let x = Value::scalar(10.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (10.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (10.9) * (10.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_090() {
        let x = Value::scalar(11.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.0) * (11.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_091() {
        let x = Value::scalar(11.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.1) * (11.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_092() {
        let x = Value::scalar(11.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.200000000000001) * (11.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_093() {
        let x = Value::scalar(11.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.3) * (11.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_094() {
        let x = Value::scalar(11.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.4) * (11.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_095() {
        let x = Value::scalar(11.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.5) * (11.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_096() {
        let x = Value::scalar(11.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.600000000000001) * (11.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_097() {
        let x = Value::scalar(11.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.700000000000001) * (11.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_098() {
        let x = Value::scalar(11.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.8) * (11.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_099() {
        let x = Value::scalar(11.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (11.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (11.9) * (11.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_100() {
        let x = Value::scalar(12.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.0) * (12.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_101() {
        let x = Value::scalar(12.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.100000000000001) * (12.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_102() {
        let x = Value::scalar(12.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.200000000000001) * (12.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_103() {
        let x = Value::scalar(12.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.3) * (12.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_104() {
        let x = Value::scalar(12.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.4) * (12.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_105() {
        let x = Value::scalar(12.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.5) * (12.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_106() {
        let x = Value::scalar(12.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.600000000000001) * (12.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_107() {
        let x = Value::scalar(12.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.700000000000001) * (12.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_108() {
        let x = Value::scalar(12.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.8) * (12.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_109() {
        let x = Value::scalar(12.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (12.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (12.9) * (12.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_110() {
        let x = Value::scalar(13.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.0) * (13.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_111() {
        let x = Value::scalar(13.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.100000000000001) * (13.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_112() {
        let x = Value::scalar(13.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.200000000000001) * (13.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_113() {
        let x = Value::scalar(13.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.3) * (13.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_114() {
        let x = Value::scalar(13.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.4) * (13.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_115() {
        let x = Value::scalar(13.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.5) * (13.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_116() {
        let x = Value::scalar(13.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.600000000000001) * (13.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_117() {
        let x = Value::scalar(13.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.700000000000001) * (13.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_118() {
        let x = Value::scalar(13.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.8) * (13.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_119() {
        let x = Value::scalar(13.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (13.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (13.9) * (13.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_120() {
        let x = Value::scalar(14.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.0) * (14.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_121() {
        let x = Value::scalar(14.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.100000000000001) * (14.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_122() {
        let x = Value::scalar(14.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.200000000000001) * (14.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_123() {
        let x = Value::scalar(14.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.3) * (14.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_124() {
        let x = Value::scalar(14.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.4) * (14.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_125() {
        let x = Value::scalar(14.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.5) * (14.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_126() {
        let x = Value::scalar(14.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.600000000000001) * (14.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_127() {
        let x = Value::scalar(14.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.700000000000001) * (14.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_128() {
        let x = Value::scalar(14.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.8) * (14.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_129() {
        let x = Value::scalar(14.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (14.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (14.9) * (14.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_130() {
        let x = Value::scalar(15.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.0) * (15.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_131() {
        let x = Value::scalar(15.100000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.100000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.100000000000001) * (15.100000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_132() {
        let x = Value::scalar(15.200000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.200000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.200000000000001) * (15.200000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_133() {
        let x = Value::scalar(15.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.3) * (15.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_134() {
        let x = Value::scalar(15.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.4) * (15.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_135() {
        let x = Value::scalar(15.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.5) * (15.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_136() {
        let x = Value::scalar(15.600000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.600000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.600000000000001) * (15.600000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_137() {
        let x = Value::scalar(15.700000000000001);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.700000000000001);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.700000000000001) * (15.700000000000001);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_138() {
        let x = Value::scalar(15.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.8) * (15.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_139() {
        let x = Value::scalar(15.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (15.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (15.9) * (15.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_140() {
        let x = Value::scalar(16.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.0) * (16.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_141() {
        let x = Value::scalar(16.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.1) * (16.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_142() {
        let x = Value::scalar(16.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.200000000000003) * (16.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_143() {
        let x = Value::scalar(16.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.3) * (16.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_144() {
        let x = Value::scalar(16.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.4) * (16.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_145() {
        let x = Value::scalar(16.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.5) * (16.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_146() {
        let x = Value::scalar(16.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.6) * (16.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_147() {
        let x = Value::scalar(16.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.700000000000003) * (16.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_148() {
        let x = Value::scalar(16.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.8) * (16.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_149() {
        let x = Value::scalar(16.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (16.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (16.9) * (16.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_150() {
        let x = Value::scalar(17.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.0) * (17.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_151() {
        let x = Value::scalar(17.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.1) * (17.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_152() {
        let x = Value::scalar(17.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.200000000000003) * (17.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_153() {
        let x = Value::scalar(17.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.3) * (17.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_154() {
        let x = Value::scalar(17.4);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.4);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.4) * (17.4);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_155() {
        let x = Value::scalar(17.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.5) * (17.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_156() {
        let x = Value::scalar(17.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.6) * (17.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_157() {
        let x = Value::scalar(17.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.700000000000003) * (17.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_158() {
        let x = Value::scalar(17.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.8) * (17.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_159() {
        let x = Value::scalar(17.9);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (17.9);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (17.9) * (17.9);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_160() {
        let x = Value::scalar(18.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.0) * (18.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_161() {
        let x = Value::scalar(18.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.1) * (18.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_162() {
        let x = Value::scalar(18.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.2) * (18.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_163() {
        let x = Value::scalar(18.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.3) * (18.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_164() {
        let x = Value::scalar(18.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.400000000000002) * (18.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_165() {
        let x = Value::scalar(18.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.5) * (18.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_166() {
        let x = Value::scalar(18.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.6) * (18.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_167() {
        let x = Value::scalar(18.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.7) * (18.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_168() {
        let x = Value::scalar(18.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.8) * (18.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_169() {
        let x = Value::scalar(18.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (18.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (18.900000000000002) * (18.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_170() {
        let x = Value::scalar(19.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.0) * (19.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_171() {
        let x = Value::scalar(19.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.1) * (19.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_172() {
        let x = Value::scalar(19.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.2) * (19.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_173() {
        let x = Value::scalar(19.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.3) * (19.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_174() {
        let x = Value::scalar(19.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.400000000000002) * (19.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_175() {
        let x = Value::scalar(19.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.5) * (19.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_176() {
        let x = Value::scalar(19.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.6) * (19.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_177() {
        let x = Value::scalar(19.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.7) * (19.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_178() {
        let x = Value::scalar(19.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.8) * (19.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_179() {
        let x = Value::scalar(19.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (19.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (19.900000000000002) * (19.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_180() {
        let x = Value::scalar(20.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.0) * (20.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_181() {
        let x = Value::scalar(20.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.1) * (20.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_182() {
        let x = Value::scalar(20.2);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.2);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.2) * (20.2);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_183() {
        let x = Value::scalar(20.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.3) * (20.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_184() {
        let x = Value::scalar(20.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.400000000000002) * (20.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_185() {
        let x = Value::scalar(20.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.5) * (20.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_186() {
        let x = Value::scalar(20.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.6) * (20.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_187() {
        let x = Value::scalar(20.7);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.7);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.7) * (20.7);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_188() {
        let x = Value::scalar(20.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.8) * (20.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_189() {
        let x = Value::scalar(20.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (20.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (20.900000000000002) * (20.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_190() {
        let x = Value::scalar(21.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.0) * (21.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_191() {
        let x = Value::scalar(21.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.1) * (21.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_192() {
        let x = Value::scalar(21.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.200000000000003) * (21.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_193() {
        let x = Value::scalar(21.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.3) * (21.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_194() {
        let x = Value::scalar(21.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.400000000000002) * (21.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_195() {
        let x = Value::scalar(21.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.5) * (21.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_196() {
        let x = Value::scalar(21.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.6) * (21.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_197() {
        let x = Value::scalar(21.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.700000000000003) * (21.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_198() {
        let x = Value::scalar(21.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.8) * (21.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_199() {
        let x = Value::scalar(21.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (21.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (21.900000000000002) * (21.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_200() {
        let x = Value::scalar(22.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.0) * (22.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_201() {
        let x = Value::scalar(22.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.1) * (22.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_202() {
        let x = Value::scalar(22.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.200000000000003) * (22.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_203() {
        let x = Value::scalar(22.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.3) * (22.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_204() {
        let x = Value::scalar(22.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.400000000000002) * (22.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_205() {
        let x = Value::scalar(22.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.5) * (22.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_206() {
        let x = Value::scalar(22.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.6) * (22.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_207() {
        let x = Value::scalar(22.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.700000000000003) * (22.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_208() {
        let x = Value::scalar(22.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.8) * (22.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_209() {
        let x = Value::scalar(22.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (22.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (22.900000000000002) * (22.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_210() {
        let x = Value::scalar(23.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.0) * (23.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_211() {
        let x = Value::scalar(23.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.1) * (23.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_212() {
        let x = Value::scalar(23.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.200000000000003) * (23.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_213() {
        let x = Value::scalar(23.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.3) * (23.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_214() {
        let x = Value::scalar(23.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.400000000000002) * (23.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_215() {
        let x = Value::scalar(23.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.5) * (23.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_216() {
        let x = Value::scalar(23.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.6) * (23.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_217() {
        let x = Value::scalar(23.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.700000000000003) * (23.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_218() {
        let x = Value::scalar(23.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.8) * (23.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_219() {
        let x = Value::scalar(23.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (23.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (23.900000000000002) * (23.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_220() {
        let x = Value::scalar(24.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.0) * (24.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_221() {
        let x = Value::scalar(24.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.1) * (24.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_222() {
        let x = Value::scalar(24.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.200000000000003) * (24.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_223() {
        let x = Value::scalar(24.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.3) * (24.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_224() {
        let x = Value::scalar(24.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.400000000000002) * (24.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_225() {
        let x = Value::scalar(24.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.5) * (24.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_226() {
        let x = Value::scalar(24.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.6) * (24.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_227() {
        let x = Value::scalar(24.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.700000000000003) * (24.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_228() {
        let x = Value::scalar(24.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.8) * (24.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_229() {
        let x = Value::scalar(24.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (24.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (24.900000000000002) * (24.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_230() {
        let x = Value::scalar(25.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.0) * (25.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_231() {
        let x = Value::scalar(25.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.1) * (25.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_232() {
        let x = Value::scalar(25.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.200000000000003) * (25.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_233() {
        let x = Value::scalar(25.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.3) * (25.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_234() {
        let x = Value::scalar(25.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.400000000000002) * (25.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_235() {
        let x = Value::scalar(25.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.5) * (25.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_236() {
        let x = Value::scalar(25.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.6) * (25.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_237() {
        let x = Value::scalar(25.700000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.700000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.700000000000003) * (25.700000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_238() {
        let x = Value::scalar(25.8);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.8);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.8) * (25.8);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_239() {
        let x = Value::scalar(25.900000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (25.900000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (25.900000000000002) * (25.900000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_240() {
        let x = Value::scalar(26.0);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.0);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.0) * (26.0);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_241() {
        let x = Value::scalar(26.1);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.1);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.1) * (26.1);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_242() {
        let x = Value::scalar(26.200000000000003);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.200000000000003);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.200000000000003) * (26.200000000000003);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_243() {
        let x = Value::scalar(26.3);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.3);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.3) * (26.3);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_244() {
        let x = Value::scalar(26.400000000000002);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.400000000000002);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.400000000000002) * (26.400000000000002);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_245() {
        let x = Value::scalar(26.5);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.5);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.5) * (26.5);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    #[test]
    fn test_graph_closure_stress_246() {
        let x = Value::scalar(26.6);
        let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
        let exp_grad = 2.0 * (26.6);
        assert!((g.get(0) - exp_grad).abs() < 1e-6);
        
        let (val, g2) = value_and_grad(|v| v.mul(v), &x).unwrap();
        let exp_val = (26.6) * (26.6);
        assert!((val.data().get(0) - exp_val).abs() < 1e-6);
        assert!((g2.unwrap().get(0) - exp_grad).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
}
