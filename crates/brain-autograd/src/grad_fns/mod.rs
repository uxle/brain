//! # Differentiable Operator Gradient Functions
//!
//! Defines the [`GradFn`] graph links, parent tracking, and VJP calculation rules.

pub mod arith;
pub mod composite;
pub mod loss_grad;
pub mod nnops;

use crate::value::Value;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};
use std::sync::Arc;

/// Identifies which differentiable operation created a `Value` node.
#[derive(Debug, Clone)]
pub enum GradFn {
    None,
    Add(Arc<Value>, Arc<Value>),
    Sub(Arc<Value>, Arc<Value>),
    Mul(Arc<Value>, Arc<Value>),
    Div(Arc<Value>, Arc<Value>),
    Pow(Arc<Value>, Arc<Value>),
    MatMul(Arc<Value>, Arc<Value>),
    Neg(Arc<Value>),
    Exp(Arc<Value>),
    Log(Arc<Value>),
    Sqrt(Arc<Value>),
    Relu(Arc<Value>),
    Sigmoid(Arc<Value>),
    Tanh(Arc<Value>),
    Sum(Arc<Value>),
    Mean(Arc<Value>),
    Softmax(Arc<Value>),
    LogSoftmax(Arc<Value>),
    Abs(Arc<Value>),
    Clamp {
        input: Arc<Value>,
        min_val: f64,
        max_val: f64,
    },
    Sin(Arc<Value>),
    Cos(Arc<Value>),
    Recip(Arc<Value>),
    Square(Arc<Value>),
    Sign(Arc<Value>),
    MinElem(Arc<Value>, Arc<Value>),
    MaxElem(Arc<Value>, Arc<Value>),
    Where {
        cond: Arc<Value>,
        a: Arc<Value>,
        b: Arc<Value>,
    },
    Reshape(Arc<Value>, Vec<usize>),
    Transpose(Arc<Value>, usize, usize),
    Permute(Arc<Value>, Vec<usize>),
    BroadcastTo(Arc<Value>, Vec<usize>),
    Conv2d {
        input: Arc<Value>,
        weight: Arc<Value>,
        bias: Option<Arc<Value>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    },
    ConvTranspose2d {
        input: Arc<Value>,
        weight: Arc<Value>,
        bias: Option<Arc<Value>>,
        stride: (usize, usize),
        padding: (usize, usize),
    },
    MaxPool2d {
        input: Arc<Value>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    AvgPool2d {
        input: Arc<Value>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    AdaptiveAvgPool2d {
        input: Arc<Value>,
        out_size: (usize, usize),
    },
    AdaptiveMaxPool2d {
        input: Arc<Value>,
        out_size: (usize, usize),
    },
    Embedding {
        weight: Arc<Value>,
        indices: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

impl Default for GradFn {
    fn default() -> Self {
        GradFn::None
    }
}

impl GradFn {
    /// Returns the parent values this node depends on.
    pub fn parents(&self) -> Vec<&Arc<Value>> {
        use GradFn::*;
        match self {
            None => Vec::new(),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) | MatMul(a, b) => {
                vec![a, b]
            }
            Neg(a)
            | Exp(a)
            | Log(a)
            | Sqrt(a)
            | Relu(a)
            | Sigmoid(a)
            | Tanh(a)
            | Sum(a)
            | Mean(a)
            | Softmax(a)
            | LogSoftmax(a)
            | Abs(a)
            | Sin(a)
            | Cos(a)
            | Recip(a)
            | Square(a)
            | Sign(a)
            | Reshape(a, _)
            | Transpose(a, _, _)
            | Permute(a, _)
            | BroadcastTo(a, _)
            | MaxPool2d { input: a, .. }
            | AvgPool2d { input: a, .. }
            | AdaptiveAvgPool2d { input: a, .. }
            | AdaptiveMaxPool2d { input: a, .. } => vec![a],
            Clamp { input: a, .. } => vec![a],
            MinElem(a, b) | MaxElem(a, b) => vec![a, b],
            Where { cond, a, b } => vec![cond, a, b],
            Conv2d {
                input,
                weight,
                bias,
                ..
            }
            | ConvTranspose2d {
                input,
                weight,
                bias,
                ..
            } => {
                let mut p = vec![input, weight];
                if let Some(ref b) = bias {
                    p.push(b);
                }
                p
            }
            Embedding { weight, .. } => vec![weight],
        }
    }

    /// Disconnects and returns all parent `Arc<Value>` references, resetting to `GradFn::None`.
    pub fn take_parents(&mut self) -> Vec<Arc<Value>> {
        use GradFn::*;
        match std::mem::replace(self, GradFn::None) {
            None => Vec::new(),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) | MatMul(a, b) => {
                vec![a, b]
            }
            Neg(a)
            | Exp(a)
            | Log(a)
            | Sqrt(a)
            | Relu(a)
            | Sigmoid(a)
            | Tanh(a)
            | Sum(a)
            | Mean(a)
            | Softmax(a)
            | LogSoftmax(a)
            | Abs(a)
            | Sin(a)
            | Cos(a)
            | Recip(a)
            | Square(a)
            | Sign(a)
            | Reshape(a, _)
            | Transpose(a, _, _)
            | Permute(a, _)
            | BroadcastTo(a, _)
            | MaxPool2d { input: a, .. }
            | AvgPool2d { input: a, .. }
            | AdaptiveAvgPool2d { input: a, .. }
            | AdaptiveMaxPool2d { input: a, .. } => vec![a],
            Clamp { input: a, .. } => vec![a],
            MinElem(a, b) | MaxElem(a, b) => vec![a, b],
            Where { cond, a, b } => vec![cond, a, b],
            Conv2d {
                input,
                weight,
                bias,
                ..
            }
            | ConvTranspose2d {
                input,
                weight,
                bias,
                ..
            } => {
                let mut p = vec![input, weight];
                if let Some(b) = bias {
                    p.push(b);
                }
                p
            }
            Embedding { weight, .. } => vec![weight],
        }
    }

    /// Returns a static string representation of the operation.
    pub fn op_name(&self) -> &'static str {
        use GradFn::*;
        match self {
            None => "leaf",
            Add(..) => "add",
            Sub(..) => "sub",
            Mul(..) => "mul",
            Div(..) => "div",
            Pow(..) => "pow",
            MatMul(..) => "matmul",
            Neg(..) => "neg",
            Exp(..) => "exp",
            Log(..) => "log",
            Sqrt(..) => "sqrt",
            Relu(..) => "relu",
            Sigmoid(..) => "sigmoid",
            Tanh(..) => "tanh",
            Sum(..) => "sum",
            Mean(..) => "mean",
            Softmax(..) => "softmax",
            LogSoftmax(..) => "log_softmax",
            Abs(..) => "abs",
            Clamp { .. } => "clamp",
            Sin(..) => "sin",
            Cos(..) => "cos",
            Recip(..) => "recip",
            Square(..) => "square",
            Sign(..) => "sign",
            MinElem(..) => "min_elem",
            MaxElem(..) => "max_elem",
            Where { .. } => "where",
            Reshape(..) => "reshape",
            Transpose(..) => "transpose",
            Permute(..) => "permute",
            BroadcastTo(..) => "broadcast_to",
            Conv2d { .. } => "conv2d",
            ConvTranspose2d { .. } => "conv_transpose2d",
            MaxPool2d { .. } => "max_pool2d",
            AvgPool2d { .. } => "avg_pool2d",
            AdaptiveAvgPool2d { .. } => "adaptive_avg_pool2d",
            AdaptiveMaxPool2d { .. } => "adaptive_max_pool2d",
            Embedding { .. } => "embedding",
        }
    }

    /// Returns whether this gradient function has parent dependencies.
    pub fn is_op(&self) -> bool {
        !matches!(self, GradFn::None)
    }

    /// Applies the Vector-Jacobian Product (VJP) rule for this operation.
    pub fn apply_vjp(&self, out_grad: &Tensor) -> BrainResult<Vec<Tensor>> {
        use GradFn::*;
        match self {
            None => Ok(Vec::new()),
            Add(a, b) => {
                let ga = util::sum_to_shape(out_grad, a.shape())?;
                let gb = util::sum_to_shape(out_grad, b.shape())?;
                Ok(vec![ga, gb])
            }
            Sub(a, b) => {
                let ga = util::sum_to_shape(out_grad, a.shape())?;
                let neg_grad = out_grad.map(|x| -x);
                let gb = util::sum_to_shape(&neg_grad, b.shape())?;
                Ok(vec![ga, gb])
            }
            Mul(a, b) => {
                let ga_full = arith_t::mul(out_grad, b.data());
                let gb_full = arith_t::mul(out_grad, a.data());
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            Div(a, b) => {
                let ga_full = arith_t::div(out_grad, b.data());
                let b_sq = arith_t::mul(b.data(), b.data());
                let num = arith_t::mul(out_grad, a.data()).map(|x| -x);
                let gb_full = arith_t::div(&num, &b_sq);
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            Pow(a, b) => {
                // dy/da = b * a^(b-1) * out_grad
                let ones = Tensor::full(b.shape().to_vec(), 1.0);
                let b_minus_1 = arith_t::sub(b.data(), &ones);
                let a_pow_bm1 = arith_t::pow_tensors(a.data(), &b_minus_1);
                let da_term = arith_t::mul(b.data(), &a_pow_bm1);
                let ga_full = arith_t::mul(out_grad, &da_term);

                // dy/db = a^b * ln(a) * out_grad
                let a_pow_b = arith_t::pow_tensors(a.data(), b.data());
                let ln_a = math_t::log(a.data());
                let db_term = arith_t::mul(&a_pow_b, &ln_a);
                let gb_full = arith_t::mul(out_grad, &db_term);

                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            MatMul(a, b) => {
                let b_ndim = b.data().ndim();
                let a_ndim = a.data().ndim();
                let bt = if b_ndim >= 2 {
                    b.data().transpose(b_ndim - 2, b_ndim - 1)
                } else {
                    b.data().transpose(0, 1)
                };
                let at = if a_ndim >= 2 {
                    a.data().transpose(a_ndim - 2, a_ndim - 1)
                } else {
                    a.data().transpose(0, 1)
                };
                let ga_full = arith_t::matmul(out_grad, &bt);
                let gb_full = arith_t::matmul(&at, out_grad);
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            Neg(_a) => {
                let ga = out_grad.map(|x| -x);
                Ok(vec![ga])
            }
            Exp(a) => {
                let exp_a = math_t::exp(a.data());
                let ga = arith_t::mul(out_grad, &exp_a);
                Ok(vec![ga])
            }
            Log(a) => {
                let ga = arith_t::div(out_grad, a.data());
                Ok(vec![ga])
            }
            Sqrt(a) => {
                let sqrt_a = math_t::sqrt(a.data());
                let two_sqrt_a = sqrt_a.map(|x| x * 2.0);
                let ga = arith_t::div(out_grad, &two_sqrt_a);
                Ok(vec![ga])
            }
            Relu(a) => {
                let mut mask = vec![0.0; a.numel()];
                let data = a.data().data();
                for (i, &val) in data.iter().enumerate() {
                    if val > 0.0 {
                        mask[i] = 1.0;
                    }
                }
                let mask_t = Tensor::from_slice(&mask, a.shape().to_vec());
                let ga = arith_t::mul(out_grad, &mask_t);
                Ok(vec![ga])
            }
            Sigmoid(a) => {
                let sig = math_t::sigmoid(a.data());
                let ones = Tensor::full(sig.shape().to_vec(), 1.0);
                let one_minus_sig = arith_t::sub(&ones, &sig);
                let sig_grad = arith_t::mul(&sig, &one_minus_sig);
                let ga = arith_t::mul(out_grad, &sig_grad);
                Ok(vec![ga])
            }
            Tanh(a) => {
                let th = math_t::tanh(a.data());
                let th_sq = arith_t::mul(&th, &th);
                let ones = Tensor::full(th.shape().to_vec(), 1.0);
                let one_minus_th_sq = arith_t::sub(&ones, &th_sq);
                let ga = arith_t::mul(out_grad, &one_minus_th_sq);
                Ok(vec![ga])
            }
            Sum(a) => {
                let val = out_grad.get(0);
                let ga = Tensor::full(a.shape().to_vec(), val);
                Ok(vec![ga])
            }
            Mean(a) => {
                let n = a.numel() as f64;
                let val = out_grad.get(0) / n;
                let ga = Tensor::full(a.shape().to_vec(), val);
                Ok(vec![ga])
            }
            Softmax(a) => {
                let last_dim = a.ndim().saturating_sub(1);
                let sm = spec_t::softmax(a.data(), last_dim);
                let dot = arith_t::mul(out_grad, &sm);
                let dot_sum = red_t::sum_along_dim(&dot, last_dim, true);
                let sub = arith_t::sub(out_grad, &dot_sum);
                let ga = arith_t::mul(&sm, &sub);
                Ok(vec![ga])
            }
            LogSoftmax(a) => {
                let last_dim = a.ndim().saturating_sub(1);
                let sm = spec_t::softmax(a.data(), last_dim);
                let sum_out = red_t::sum_along_dim(out_grad, last_dim, true);
                let sm_sum = arith_t::mul(&sm, &sum_out);
                let ga = arith_t::sub(out_grad, &sm_sum);
                Ok(vec![ga])
            }
            Abs(a) => {
                // d|x|/dx = sign(x), 0 at x == 0
                let sign_a = math_t::sign(a.data());
                let ga = arith_t::mul(out_grad, &sign_a);
                Ok(vec![ga])
            }
            Clamp {
                input: a,
                min_val,
                max_val,
            } => {
                // Gradient flows only where min < x < max
                let data = a.data().data();
                let mut mask = vec![0.0; a.numel()];
                for (i, &x) in data.iter().enumerate() {
                    if x > *min_val && x < *max_val {
                        mask[i] = 1.0;
                    }
                }
                let mask_t = Tensor::from_slice(&mask, a.shape().to_vec());
                let ga = arith_t::mul(out_grad, &mask_t);
                Ok(vec![ga])
            }
            Sin(a) => {
                // d sin(x)/dx = cos(x)
                let cos_a = math_t::cos(a.data());
                let ga = arith_t::mul(out_grad, &cos_a);
                Ok(vec![ga])
            }
            Cos(a) => {
                // d cos(x)/dx = -sin(x)
                let sin_a = math_t::sin(a.data());
                let neg_sin = sin_a.map(|x| -x);
                let ga = arith_t::mul(out_grad, &neg_sin);
                Ok(vec![ga])
            }
            Recip(a) => {
                // d (1/x)/dx = -1/x^2
                let a_sq = math_t::square(a.data());
                let ones = Tensor::full(a_sq.shape().to_vec(), 1.0);
                let neg = arith_t::div(&ones, &a_sq).map(|x| -x);
                let ga = arith_t::mul(out_grad, &neg);
                Ok(vec![ga])
            }
            Square(a) => {
                // d (x^2)/dx = 2x
                let two_x = a.data().map(|x| 2.0 * x);
                let ga = arith_t::mul(out_grad, &two_x);
                Ok(vec![ga])
            }
            Sign(a) => {
                // Piecewise constant: zero gradient everywhere
                let zeros = Tensor::zeros(a.shape().to_vec());
                Ok(vec![zeros])
            }
            MinElem(a, b) => {
                // Ties split the gradient 0.5 / 0.5 (PyTorch convention)
                let ad = a.data().data();
                let bd = b.data().data();
                let mut mask_a = vec![0.5; ad.len()];
                let mut mask_b = vec![0.5; bd.len()];
                for (i, (&x, &y)) in ad.iter().zip(bd.iter()).enumerate() {
                    if x < y {
                        mask_a[i] = 1.0;
                        mask_b[i] = 0.0;
                    } else if x > y {
                        mask_a[i] = 0.0;
                        mask_b[i] = 1.0;
                    }
                }
                let mask_a_t = Tensor::from_slice(&mask_a, a.shape().to_vec());
                let mask_b_t = Tensor::from_slice(&mask_b, b.shape().to_vec());
                let ga_full = arith_t::mul(out_grad, &mask_a_t);
                let gb_full = arith_t::mul(out_grad, &mask_b_t);
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            MaxElem(a, b) => {
                let ad = a.data().data();
                let bd = b.data().data();
                let mut mask_a = vec![0.5; ad.len()];
                let mut mask_b = vec![0.5; bd.len()];
                for (i, (&x, &y)) in ad.iter().zip(bd.iter()).enumerate() {
                    if x > y {
                        mask_a[i] = 1.0;
                        mask_b[i] = 0.0;
                    } else if x < y {
                        mask_a[i] = 0.0;
                        mask_b[i] = 1.0;
                    }
                }
                let mask_a_t = Tensor::from_slice(&mask_a, a.shape().to_vec());
                let mask_b_t = Tensor::from_slice(&mask_b, b.shape().to_vec());
                let ga_full = arith_t::mul(out_grad, &mask_a_t);
                let gb_full = arith_t::mul(out_grad, &mask_b_t);
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                Ok(vec![ga, gb])
            }
            Where { cond, a, b } => {
                // cond is boolean-ish: gradient flows through selected branch only
                let cd = cond.data().data();
                let mut mask_a = vec![0.0; a.numel()];
                let mut mask_b = vec![0.0; b.numel()];
                for (i, &c) in cd.iter().enumerate() {
                    if c != 0.0 {
                        mask_a[i] = 1.0;
                    } else {
                        mask_b[i] = 1.0;
                    }
                }
                let mask_a_t = Tensor::from_slice(&mask_a, a.shape().to_vec());
                let mask_b_t = Tensor::from_slice(&mask_b, b.shape().to_vec());
                let ga_full = arith_t::mul(out_grad, &mask_a_t);
                let gb_full = arith_t::mul(out_grad, &mask_b_t);
                let ga = util::sum_to_shape(&ga_full, a.shape())?;
                let gb = util::sum_to_shape(&gb_full, b.shape())?;
                let gcond = Tensor::zeros(cond.shape().to_vec());
                Ok(vec![gcond, ga, gb])
            }
            Reshape(a, _) => {
                let ga = out_grad.reshape(a.shape().to_vec());
                Ok(vec![ga])
            }
            Transpose(_a, d0, d1) => {
                let ga = out_grad.transpose(*d0, *d1);
                Ok(vec![ga])
            }
            Permute(_a, dims) => {
                let mut inv_dims = vec![0; dims.len()];
                for (i, &d) in dims.iter().enumerate() {
                    inv_dims[d] = i;
                }
                let ga = out_grad.permute(&inv_dims);
                Ok(vec![ga])
            }
            BroadcastTo(a, _) => {
                let ga = util::sum_to_shape(out_grad, a.shape())?;
                Ok(vec![ga])
            }
            Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
            } => {
                let (di, dw, db) = crate::ops::conv_grad::grad_conv2d(
                    input.data(),
                    weight.data(),
                    out_grad,
                    *stride,
                    *padding,
                    *dilation,
                )?;
                let mut grads = vec![di, dw];
                if bias.is_some() {
                    grads.push(db.unwrap_or_else(|| Tensor::zeros(vec![weight.shape()[0]])));
                }
                Ok(grads)
            }
            ConvTranspose2d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => {
                let (di, dw, db) = crate::ops::conv_grad::grad_conv_transpose2d(
                    input.data(),
                    weight.data(),
                    out_grad,
                    *stride,
                    *padding,
                )?;
                let mut grads = vec![di, dw];
                if bias.is_some() {
                    grads.push(db.unwrap_or_else(|| Tensor::zeros(vec![weight.shape()[1]])));
                }
                Ok(grads)
            }
            MaxPool2d {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                let di = crate::ops::pool_grad::grad_max_pool2d(
                    input.data(),
                    out_grad,
                    *kernel_size,
                    *stride,
                    *padding,
                )?;
                Ok(vec![di])
            }
            AvgPool2d {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                let di = crate::ops::pool_grad::grad_avg_pool2d_ext(
                    input.shape(),
                    out_grad,
                    *kernel_size,
                    *stride,
                    *padding,
                )?;
                Ok(vec![di])
            }
            AdaptiveAvgPool2d { input, out_size } => {
                let di = crate::ops::pool_grad::grad_adaptive_avg_pool2d(
                    input.shape(),
                    out_grad,
                    out_size.0,
                    out_size.1,
                )?;
                Ok(vec![di])
            }
            AdaptiveMaxPool2d { input, out_size } => {
                let di = crate::ops::pool_grad::grad_adaptive_max_pool2d(
                    input.data(),
                    out_grad,
                    out_size.0,
                    out_size.1,
                )?;
                Ok(vec![di])
            }
            Embedding {
                weight, indices, ..
            } => {
                let w_shape = weight.shape();
                let num_embeddings = if !w_shape.is_empty() { w_shape[0] } else { 0 };
                let embedding_dim = if w_shape.len() > 1 { w_shape[1] } else { 1 };
                let dw = crate::ops::index_grad::grad_embedding(
                    out_grad,
                    num_embeddings,
                    embedding_dim,
                    indices,
                )?;
                Ok(vec![dw])
            }
        }
    }
}

/// Utility helpers for gradient reductions and unbroadcasting.
pub mod util {
    use brain_core::tensor::reduction as red_t;
    use brain_core::{BrainError, BrainResult, Tensor};

    /// Sums incoming gradient along expanded or prepended dimensions to match `target_shape`.
    pub fn sum_to_shape(grad: &Tensor, target_shape: &[usize]) -> BrainResult<Tensor> {
        let g_shape = grad.shape();
        if g_shape == target_shape {
            return Ok(grad.clone());
        }
        if target_shape.is_empty() {
            return Ok(Tensor::scalar(red_t::sum(grad)));
        }
        if target_shape == [1] && g_shape != [1] {
            return Ok(Tensor::from_slice(&[red_t::sum(grad)], vec![1]));
        }

        let mut curr = grad.clone();
        // Sum out leading prepended dimensions
        while curr.ndim() > target_shape.len() {
            curr = red_t::sum_along_dim(&curr, 0, false);
        }

        // Sum out singleton dimensions that were broadcast
        let g_dims: Vec<usize> = curr.shape().to_vec();
        for (i, (&g_dim, &t_dim)) in g_dims.iter().zip(target_shape.iter()).enumerate() {
            if g_dim != t_dim {
                if t_dim == 1 {
                    curr = red_t::sum_along_dim(&curr, i, true);
                } else {
                    return Err(BrainError::shape_mismatch(
                        format!("{:?}", target_shape),
                        format!("{:?}", g_shape),
                        "Cannot reduce grad shape to target",
                    ));
                }
            }
        }
        Ok(curr)
    }

    /// Alias for `sum_to_shape`.
    pub fn unbroadcast(grad: &Tensor, target_shape: &[usize]) -> BrainResult<Tensor> {
        sum_to_shape(grad, target_shape)
    }
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
