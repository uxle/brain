//! # Unary Forward Differentiable Operations
//!
//! Forward operations taking a single `Value` and attaching the corresponding `GradFn`.

use crate::grad_fns::GradFn;
use crate::value::Value;
use brain_core::tensor::math as math_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::Tensor;
use std::sync::Arc;

/// Elementwise negation: `-a`.
pub fn neg(a: &Value) -> Value {
    let out_data = a.data().map(|x| -x);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Neg(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise exponential: `exp(a)`.
pub fn exp(a: &Value) -> Value {
    let out_data = math_t::exp(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Exp(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise natural logarithm: `log(a)`.
pub fn log(a: &Value) -> Value {
    let out_data = math_t::log(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Log(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise square root: `sqrt(a)`.
pub fn sqrt(a: &Value) -> Value {
    let out_data = math_t::sqrt(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Sqrt(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise ReLU: `max(0, a)`.
pub fn relu(a: &Value) -> Value {
    let out_data = math_t::relu(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Relu(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise Sigmoid: `1 / (1 + exp(-a))`.
pub fn sigmoid(a: &Value) -> Value {
    let out_data = math_t::sigmoid(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Sigmoid(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise Tanh: `tanh(a)`.
pub fn tanh(a: &Value) -> Value {
    let out_data = math_t::tanh(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Tanh(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Full sum reduction.
pub fn sum(a: &Value) -> Value {
    let val = red_t::sum(a.data());
    let out_data = Tensor::scalar(val);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Sum(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Full mean reduction.
pub fn mean(a: &Value) -> Value {
    let val = red_t::mean(a.data());
    let out_data = Tensor::scalar(val);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Mean(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Softmax along the last dimension.
pub fn softmax(a: &Value) -> Value {
    let last_dim = a.ndim().saturating_sub(1);
    let out_data = spec_t::softmax(a.data(), last_dim);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Softmax(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// LogSoftmax along the last dimension.
pub fn log_softmax(a: &Value) -> Value {
    let last_dim = a.ndim().saturating_sub(1);
    let out_data = spec_t::log_softmax(a.data(), last_dim);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::LogSoftmax(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise absolute value: `|a|`.
pub fn abs(a: &Value) -> Value {
    let out_data = math_t::abs(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Abs(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise clamp: `min(max(a, min_val), max_val)`.
pub fn clamp(a: &Value, min_val: f64, max_val: f64) -> Value {
    let out_data = math_t::clamp(a.data(), min_val, max_val);
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Clamp {
            input: Arc::new(a.clone()),
            min_val,
            max_val,
        }
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise sine: `sin(a)`.
pub fn sin(a: &Value) -> Value {
    let out_data = math_t::sin(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Sin(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise cosine: `cos(a)`.
pub fn cos(a: &Value) -> Value {
    let out_data = math_t::cos(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Cos(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise reciprocal: `1 / a`.
pub fn recip(a: &Value) -> Value {
    let out_data = math_t::recip(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Recip(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise square: `a * a`.
pub fn square(a: &Value) -> Value {
    let out_data = math_t::square(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Square(Arc::new(a.clone()))
    } else {
        GradFn::None
    };
    Value::from_op(out_data, grad_fn, req)
}

/// Elementwise sign: `-1.0, 0.0, 1.0` (piecewise constant, zero gradient).
pub fn sign(a: &Value) -> Value {
    let out_data = math_t::sign(a.data());
    let req = a.requires_grad();
    let grad_fn = if req {
        GradFn::Sign(Arc::new(a.clone()))
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
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
