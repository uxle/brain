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
    let g = leaf
        .grad()
        .ok_or_else(|| BrainError::invalid_value("No gradient accumulated on leaf"))?;
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
        let g = leaf
            .grad()
            .ok_or_else(|| BrainError::invalid_value("Jacobian grad missing"))?;
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
    let g =
        grad(&f, x)?.ok_or_else(|| BrainError::invalid_value("Grad missing in grad_and_hess"))?;
    let h = hessian(f, x)?;
    Ok((g, h))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
