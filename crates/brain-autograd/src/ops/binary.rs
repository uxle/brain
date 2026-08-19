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

/// Elementwise minimum: `min(a, b)`.
pub fn min_elem(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::min_elem(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::MinElem(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise maximum: `max(a, b)`.
pub fn max_elem(a: &Value, b: &Value) -> Value {
    let out_data = arith_t::max_elem(a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::MaxElem(Arc::new(a.clone()), Arc::new(b.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Conditional elementwise select: `where(cond, a, b)`.
///
/// `cond` is a boolean mask (1.0 / 0.0); it receives no gradient.
pub fn where_cond(cond: &Value, a: &Value, b: &Value) -> Value {
    let out_data = arith_t::where_cond(cond.data(), a.data(), b.data());
    let req = a.requires_grad() || b.requires_grad();
    let grad_fn = if req {
        GradFn::Where {
            cond: Arc::new(cond.clone()),
            a: Arc::new(a.clone()),
            b: Arc::new(b.clone()),
        }
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
}
