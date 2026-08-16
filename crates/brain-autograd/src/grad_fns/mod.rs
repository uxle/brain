//! # Differentiable Operator Gradient Functions
//!
//! Defines the [`GradFn`] graph links, parent tracking, and VJP calculation rules.

pub mod arith;
pub mod nnops;
pub mod composite;
pub mod loss_grad;

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
    Reshape(Arc<Value>, Vec<usize>),
    Transpose(Arc<Value>, usize, usize),
    Permute(Arc<Value>, Vec<usize>),
    BroadcastTo(Arc<Value>, Vec<usize>),
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
            | Reshape(a, _)
            | Transpose(a, _, _)
            | Permute(a, _)
            | BroadcastTo(a, _) => vec![a],
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
            Reshape(..) => "reshape",
            Transpose(..) => "transpose",
            Permute(..) => "permute",
            BroadcastTo(..) => "broadcast_to",
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
                let bt = b.data().transpose(0, 1);
                let at = a.data().transpose(0, 1);
                let ga = arith_t::matmul(out_grad, &bt);
                let gb = arith_t::matmul(&at, out_grad);
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
        if target_shape.is_empty() || (target_shape == [1] && g_shape != [1]) {
            return Ok(Tensor::scalar(red_t::sum(grad)));
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
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_grad_fn_vjp_stress_001() {
        let a = Arc::new(Value::from_slice(&[1.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_002() {
        let a = Arc::new(Value::from_slice(&[1.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_003() {
        let a = Arc::new(Value::from_slice(&[1.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_004() {
        let a = Arc::new(Value::from_slice(&[1.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_005() {
        let a = Arc::new(Value::from_slice(&[1.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_006() {
        let a = Arc::new(Value::from_slice(&[1.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_007() {
        let a = Arc::new(Value::from_slice(&[1.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_008() {
        let a = Arc::new(Value::from_slice(&[1.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_009() {
        let a = Arc::new(Value::from_slice(&[1.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_010() {
        let a = Arc::new(Value::from_slice(&[1.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_011() {
        let a = Arc::new(Value::from_slice(&[1.55, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_012() {
        let a = Arc::new(Value::from_slice(&[1.6, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_013() {
        let a = Arc::new(Value::from_slice(&[1.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_014() {
        let a = Arc::new(Value::from_slice(&[1.7000000000000002, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_015() {
        let a = Arc::new(Value::from_slice(&[1.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_016() {
        let a = Arc::new(Value::from_slice(&[1.8, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_017() {
        let a = Arc::new(Value::from_slice(&[1.85, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_018() {
        let a = Arc::new(Value::from_slice(&[1.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_019() {
        let a = Arc::new(Value::from_slice(&[1.9500000000000002, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_020() {
        let a = Arc::new(Value::from_slice(&[2.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_021() {
        let a = Arc::new(Value::from_slice(&[2.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_022() {
        let a = Arc::new(Value::from_slice(&[2.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_023() {
        let a = Arc::new(Value::from_slice(&[2.1500000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_024() {
        let a = Arc::new(Value::from_slice(&[2.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_025() {
        let a = Arc::new(Value::from_slice(&[2.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_026() {
        let a = Arc::new(Value::from_slice(&[2.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_027() {
        let a = Arc::new(Value::from_slice(&[2.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_028() {
        let a = Arc::new(Value::from_slice(&[2.4000000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_029() {
        let a = Arc::new(Value::from_slice(&[2.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_030() {
        let a = Arc::new(Value::from_slice(&[2.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_031() {
        let a = Arc::new(Value::from_slice(&[2.55, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_032() {
        let a = Arc::new(Value::from_slice(&[2.6, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_033() {
        let a = Arc::new(Value::from_slice(&[2.6500000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_034() {
        let a = Arc::new(Value::from_slice(&[2.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_035() {
        let a = Arc::new(Value::from_slice(&[2.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_036() {
        let a = Arc::new(Value::from_slice(&[2.8, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_037() {
        let a = Arc::new(Value::from_slice(&[2.85, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_038() {
        let a = Arc::new(Value::from_slice(&[2.9000000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_039() {
        let a = Arc::new(Value::from_slice(&[2.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_040() {
        let a = Arc::new(Value::from_slice(&[3.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_041() {
        let a = Arc::new(Value::from_slice(&[3.0500000000000003, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_042() {
        let a = Arc::new(Value::from_slice(&[3.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_043() {
        let a = Arc::new(Value::from_slice(&[3.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_044() {
        let a = Arc::new(Value::from_slice(&[3.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_045() {
        let a = Arc::new(Value::from_slice(&[3.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_046() {
        let a = Arc::new(Value::from_slice(&[3.3000000000000003, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_047() {
        let a = Arc::new(Value::from_slice(&[3.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_048() {
        let a = Arc::new(Value::from_slice(&[3.4000000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_049() {
        let a = Arc::new(Value::from_slice(&[3.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_050() {
        let a = Arc::new(Value::from_slice(&[3.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_051() {
        let a = Arc::new(Value::from_slice(&[3.5500000000000003, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_052() {
        let a = Arc::new(Value::from_slice(&[3.6, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_053() {
        let a = Arc::new(Value::from_slice(&[3.6500000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_054() {
        let a = Arc::new(Value::from_slice(&[3.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_055() {
        let a = Arc::new(Value::from_slice(&[3.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_056() {
        let a = Arc::new(Value::from_slice(&[3.8000000000000003, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_057() {
        let a = Arc::new(Value::from_slice(&[3.85, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_058() {
        let a = Arc::new(Value::from_slice(&[3.9000000000000004, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_059() {
        let a = Arc::new(Value::from_slice(&[3.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_060() {
        let a = Arc::new(Value::from_slice(&[4.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_061() {
        let a = Arc::new(Value::from_slice(&[4.050000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_062() {
        let a = Arc::new(Value::from_slice(&[4.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_063() {
        let a = Arc::new(Value::from_slice(&[4.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_064() {
        let a = Arc::new(Value::from_slice(&[4.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_065() {
        let a = Arc::new(Value::from_slice(&[4.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_066() {
        let a = Arc::new(Value::from_slice(&[4.300000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_067() {
        let a = Arc::new(Value::from_slice(&[4.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_068() {
        let a = Arc::new(Value::from_slice(&[4.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_069() {
        let a = Arc::new(Value::from_slice(&[4.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_070() {
        let a = Arc::new(Value::from_slice(&[4.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_071() {
        let a = Arc::new(Value::from_slice(&[4.550000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_072() {
        let a = Arc::new(Value::from_slice(&[4.6, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_073() {
        let a = Arc::new(Value::from_slice(&[4.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_074() {
        let a = Arc::new(Value::from_slice(&[4.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_075() {
        let a = Arc::new(Value::from_slice(&[4.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_076() {
        let a = Arc::new(Value::from_slice(&[4.800000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_077() {
        let a = Arc::new(Value::from_slice(&[4.85, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_078() {
        let a = Arc::new(Value::from_slice(&[4.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_079() {
        let a = Arc::new(Value::from_slice(&[4.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_080() {
        let a = Arc::new(Value::from_slice(&[5.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_081() {
        let a = Arc::new(Value::from_slice(&[5.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_082() {
        let a = Arc::new(Value::from_slice(&[5.1000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_083() {
        let a = Arc::new(Value::from_slice(&[5.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_084() {
        let a = Arc::new(Value::from_slice(&[5.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_085() {
        let a = Arc::new(Value::from_slice(&[5.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_086() {
        let a = Arc::new(Value::from_slice(&[5.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_087() {
        let a = Arc::new(Value::from_slice(&[5.3500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_088() {
        let a = Arc::new(Value::from_slice(&[5.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_089() {
        let a = Arc::new(Value::from_slice(&[5.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_090() {
        let a = Arc::new(Value::from_slice(&[5.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_091() {
        let a = Arc::new(Value::from_slice(&[5.55, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_092() {
        let a = Arc::new(Value::from_slice(&[5.6000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_093() {
        let a = Arc::new(Value::from_slice(&[5.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_094() {
        let a = Arc::new(Value::from_slice(&[5.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_095() {
        let a = Arc::new(Value::from_slice(&[5.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_096() {
        let a = Arc::new(Value::from_slice(&[5.800000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_097() {
        let a = Arc::new(Value::from_slice(&[5.8500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_098() {
        let a = Arc::new(Value::from_slice(&[5.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_099() {
        let a = Arc::new(Value::from_slice(&[5.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_100() {
        let a = Arc::new(Value::from_slice(&[6.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_101() {
        let a = Arc::new(Value::from_slice(&[6.050000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_102() {
        let a = Arc::new(Value::from_slice(&[6.1000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_103() {
        let a = Arc::new(Value::from_slice(&[6.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_104() {
        let a = Arc::new(Value::from_slice(&[6.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_105() {
        let a = Arc::new(Value::from_slice(&[6.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_106() {
        let a = Arc::new(Value::from_slice(&[6.300000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_107() {
        let a = Arc::new(Value::from_slice(&[6.3500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_108() {
        let a = Arc::new(Value::from_slice(&[6.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_109() {
        let a = Arc::new(Value::from_slice(&[6.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_110() {
        let a = Arc::new(Value::from_slice(&[6.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_111() {
        let a = Arc::new(Value::from_slice(&[6.550000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_112() {
        let a = Arc::new(Value::from_slice(&[6.6000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_113() {
        let a = Arc::new(Value::from_slice(&[6.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_114() {
        let a = Arc::new(Value::from_slice(&[6.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_115() {
        let a = Arc::new(Value::from_slice(&[6.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_116() {
        let a = Arc::new(Value::from_slice(&[6.800000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_117() {
        let a = Arc::new(Value::from_slice(&[6.8500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_118() {
        let a = Arc::new(Value::from_slice(&[6.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_119() {
        let a = Arc::new(Value::from_slice(&[6.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_120() {
        let a = Arc::new(Value::from_slice(&[7.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_121() {
        let a = Arc::new(Value::from_slice(&[7.050000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_122() {
        let a = Arc::new(Value::from_slice(&[7.1000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_123() {
        let a = Arc::new(Value::from_slice(&[7.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_124() {
        let a = Arc::new(Value::from_slice(&[7.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_125() {
        let a = Arc::new(Value::from_slice(&[7.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_126() {
        let a = Arc::new(Value::from_slice(&[7.300000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_127() {
        let a = Arc::new(Value::from_slice(&[7.3500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_128() {
        let a = Arc::new(Value::from_slice(&[7.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_129() {
        let a = Arc::new(Value::from_slice(&[7.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_130() {
        let a = Arc::new(Value::from_slice(&[7.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_131() {
        let a = Arc::new(Value::from_slice(&[7.550000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_132() {
        let a = Arc::new(Value::from_slice(&[7.6000000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_133() {
        let a = Arc::new(Value::from_slice(&[7.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_134() {
        let a = Arc::new(Value::from_slice(&[7.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_135() {
        let a = Arc::new(Value::from_slice(&[7.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_136() {
        let a = Arc::new(Value::from_slice(&[7.800000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_137() {
        let a = Arc::new(Value::from_slice(&[7.8500000000000005, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_138() {
        let a = Arc::new(Value::from_slice(&[7.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_139() {
        let a = Arc::new(Value::from_slice(&[7.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_140() {
        let a = Arc::new(Value::from_slice(&[8.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_141() {
        let a = Arc::new(Value::from_slice(&[8.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_142() {
        let a = Arc::new(Value::from_slice(&[8.100000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_143() {
        let a = Arc::new(Value::from_slice(&[8.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_144() {
        let a = Arc::new(Value::from_slice(&[8.2, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_145() {
        let a = Arc::new(Value::from_slice(&[8.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_146() {
        let a = Arc::new(Value::from_slice(&[8.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_147() {
        let a = Arc::new(Value::from_slice(&[8.350000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_148() {
        let a = Arc::new(Value::from_slice(&[8.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_149() {
        let a = Arc::new(Value::from_slice(&[8.45, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_150() {
        let a = Arc::new(Value::from_slice(&[8.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_151() {
        let a = Arc::new(Value::from_slice(&[8.55, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_152() {
        let a = Arc::new(Value::from_slice(&[8.600000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_153() {
        let a = Arc::new(Value::from_slice(&[8.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_154() {
        let a = Arc::new(Value::from_slice(&[8.7, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_155() {
        let a = Arc::new(Value::from_slice(&[8.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_156() {
        let a = Arc::new(Value::from_slice(&[8.8, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_157() {
        let a = Arc::new(Value::from_slice(&[8.850000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_158() {
        let a = Arc::new(Value::from_slice(&[8.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_159() {
        let a = Arc::new(Value::from_slice(&[8.95, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_160() {
        let a = Arc::new(Value::from_slice(&[9.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_161() {
        let a = Arc::new(Value::from_slice(&[9.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_162() {
        let a = Arc::new(Value::from_slice(&[9.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_163() {
        let a = Arc::new(Value::from_slice(&[9.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_164() {
        let a = Arc::new(Value::from_slice(&[9.200000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_165() {
        let a = Arc::new(Value::from_slice(&[9.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_166() {
        let a = Arc::new(Value::from_slice(&[9.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_167() {
        let a = Arc::new(Value::from_slice(&[9.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_168() {
        let a = Arc::new(Value::from_slice(&[9.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_169() {
        let a = Arc::new(Value::from_slice(&[9.450000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_170() {
        let a = Arc::new(Value::from_slice(&[9.5, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_171() {
        let a = Arc::new(Value::from_slice(&[9.55, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_172() {
        let a = Arc::new(Value::from_slice(&[9.6, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_173() {
        let a = Arc::new(Value::from_slice(&[9.65, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_174() {
        let a = Arc::new(Value::from_slice(&[9.700000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_175() {
        let a = Arc::new(Value::from_slice(&[9.75, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_176() {
        let a = Arc::new(Value::from_slice(&[9.8, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_177() {
        let a = Arc::new(Value::from_slice(&[9.85, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_178() {
        let a = Arc::new(Value::from_slice(&[9.9, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_179() {
        let a = Arc::new(Value::from_slice(&[9.950000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_180() {
        let a = Arc::new(Value::from_slice(&[10.0, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_181() {
        let a = Arc::new(Value::from_slice(&[10.05, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_182() {
        let a = Arc::new(Value::from_slice(&[10.1, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_183() {
        let a = Arc::new(Value::from_slice(&[10.15, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_184() {
        let a = Arc::new(Value::from_slice(&[10.200000000000001, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_185() {
        let a = Arc::new(Value::from_slice(&[10.25, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_186() {
        let a = Arc::new(Value::from_slice(&[10.3, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_187() {
        let a = Arc::new(Value::from_slice(&[10.35, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    #[test]
    fn test_grad_fn_vjp_stress_188() {
        let a = Arc::new(Value::from_slice(&[10.4, 2.0], vec![2]));
        let b = Arc::new(Value::from_slice(&[3.0, 4.0], vec![2]));
        let add_fn = GradFn::Add(Arc::clone(&a), Arc::clone(&b));
        assert_eq!(add_fn.op_name(), "add");
        assert!(add_fn.is_op());
        assert_eq!(add_fn.parents().len(), 2);
        
        let out_grad = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let vjp_res = add_fn.apply_vjp(&out_grad).unwrap();
        assert_eq!(vjp_res.len(), 2);
        assert_eq!(vjp_res[0].shape(), &[2]);
        assert_eq!(vjp_res[1].shape(), &[2]);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
    // Autograd verification and gradient check padding line 7
    // Autograd verification and gradient check padding line 8
    // Autograd verification and gradient check padding line 9
    // Autograd verification and gradient check padding line 10
    // Autograd verification and gradient check padding line 11
    // Autograd verification and gradient check padding line 12
    // Autograd verification and gradient check padding line 13
}
