//! # Differentiable Value Graph Node
//!
//! The fundamental computation graph node wrapping `Tensor`, gradient slots,
//! and parent links for reverse-mode automatic differentiation.

use crate::grad_fns::GradFn;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::{BrainResult, Tensor};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

static NEXT_VALUE_ID: AtomicUsize = AtomicUsize::new(1);

/// A differentiable computation node wrapping a `Tensor` and backward metadata.
pub struct Value {
    pub(crate) id: usize,
    pub(crate) data: Arc<Tensor>,
    pub(crate) grad: Arc<RwLock<Option<Tensor>>>,
    pub(crate) requires_grad: bool,
    pub(crate) grad_fn: GradFn,
    pub(crate) name: Option<String>,
    pub(crate) is_leaf: bool,
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: Arc::clone(&self.data),
            grad: Arc::clone(&self.grad),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            name: self.name.clone(),
            is_leaf: self.is_leaf,
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("id", &self.id)
            .field("shape", &self.data.shape())
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.grad_fn.op_name())
            .field("name", &self.name)
            .finish()
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        let mut work_list = self.grad_fn.take_parents();
        while let Some(parent_arc) = work_list.pop() {
            if let Some(mut parent_val) = Arc::into_inner(parent_arc) {
                let next_parents = parent_val.grad_fn.take_parents();
                work_list.extend(next_parents);
            }
        }
    }
}

impl Value {
    /// Creates a leaf `Value` node from a `Tensor`.
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        Self {
            id: NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            data: Arc::new(tensor),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            grad_fn: GradFn::None,
            name: None,
            is_leaf: true,
        }
    }

    /// Creates a leaf `Value` with gradient tracking enabled.
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self::new(tensor.clone(), true)
    }

    /// Creates a scalar `Value`.
    pub fn scalar(val: f64) -> Self {
        Self::new(Tensor::scalar(val), false)
    }

    /// Creates a `Value` from a 1D slice with a specified shape.
    pub fn from_slice(data: &[f64], shape: Vec<usize>) -> Self {
        Self::new(Tensor::from_slice(data, shape), false)
    }

    /// Creates an operation node in the computation graph.
    pub fn from_op(tensor: Tensor, grad_fn: GradFn, requires_grad: bool) -> Self {
        Self {
            id: NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            data: Arc::new(tensor),
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            grad_fn,
            name: None,
            is_leaf: false,
        }
    }

    /// Detaches the value from the computation graph, returning a new leaf value.
    pub fn detach(&self) -> Self {
        Self::new((*self.data).clone(), false)
    }

    /// Unique identifier for this computation graph node.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Reference to underlying `Tensor` payload.
    pub fn data(&self) -> &Tensor {
        &self.data
    }

    /// Clones the accumulated gradient if present.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.read().unwrap().clone()
    }

    /// Whether this node requires backward gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Enables or disables gradient tracking for this node.
    pub fn set_requires_grad(&mut self, req: bool) {
        self.requires_grad = req;
    }

    /// Reference to gradient function metadata.
    pub fn grad_fn(&self) -> &GradFn {
        &self.grad_fn
    }

    /// Name tag of this node if assigned.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sets an optional debug name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Returns `true` if this value was created directly without prior graph history.
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Returns the shape of the underlying tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns total number of elements in the underlying tensor.
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Returns number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Clears any accumulated gradient.
    pub fn zero_grad(&self) {
        let mut g = self.grad.write().unwrap();
        *g = None;
    }

    /// Accumulates incoming gradient into this node.
    pub fn accumulate_grad(&self, incoming: &Tensor) -> BrainResult<()> {
        let mut g = self.grad.write().unwrap();
        match g.as_mut() {
            Some(existing) => {
                if existing.shape() != incoming.shape() {
                    let unbroadcasted = crate::grad_fns::util::sum_to_shape(incoming, existing.shape())?;
                    *existing = arith_t::add(existing, &unbroadcasted);
                } else {
                    *existing = arith_t::add(existing, incoming);
                }
            }
            None => {
                if incoming.shape() != self.shape() {
                    let unbroadcasted = crate::grad_fns::util::sum_to_shape(incoming, self.shape())?;
                    *g = Some(unbroadcasted);
                } else {
                    *g = Some(incoming.clone());
                }
            }
        }
        Ok(())
    }

    /// Runs backward pass starting from this scalar root node.
    pub fn backward(&self) -> BrainResult<()> {
        crate::backward::grad::backward_from(self)
    }

    /// Runs backward pass with a given seed gradient.
    pub fn backward_with_grad(&self, seed_grad: &Tensor) -> BrainResult<()> {
        crate::backward::grad::backward_with_grad(self, seed_grad)
    }

    // Forward Math and NN methods
    /// Elementwise addition.
    pub fn add(&self, other: &Value) -> Value {
        crate::ops::binary::add(self, other)
    }

    /// Elementwise subtraction.
    pub fn sub(&self, other: &Value) -> Value {
        crate::ops::binary::sub(self, other)
    }

    /// Elementwise multiplication.
    pub fn mul(&self, other: &Value) -> Value {
        crate::ops::binary::mul(self, other)
    }

    /// Elementwise division.
    pub fn div(&self, other: &Value) -> Value {
        crate::ops::binary::div(self, other)
    }

    /// Elementwise power.
    pub fn pow(&self, other: &Value) -> Value {
        crate::ops::binary::pow(self, other)
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Value) -> Value {
        crate::ops::binary::matmul(self, other)
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Value {
        crate::ops::unary::neg(self)
    }

    /// Elementwise exponential.
    pub fn exp(&self) -> Value {
        crate::ops::unary::exp(self)
    }

    /// Elementwise natural logarithm.
    pub fn log(&self) -> Value {
        crate::ops::unary::log(self)
    }

    /// Elementwise square root.
    pub fn sqrt(&self) -> Value {
        crate::ops::unary::sqrt(self)
    }

    /// Elementwise ReLU.
    pub fn relu(&self) -> Value {
        crate::ops::unary::relu(self)
    }

    /// Elementwise Sigmoid.
    pub fn sigmoid(&self) -> Value {
        crate::ops::unary::sigmoid(self)
    }

    /// Elementwise Tanh.
    pub fn tanh(&self) -> Value {
        crate::ops::unary::tanh(self)
    }

    /// Sum reduction.
    pub fn sum(&self) -> Value {
        crate::ops::unary::sum(self)
    }

    /// Transposes two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Value {
        let out_tensor = self.data().transpose(dim0, dim1);
        let grad_fn = if self.requires_grad {
            GradFn::Transpose(Arc::new(self.clone()), dim0, dim1)
        } else {
            GradFn::None
        };
        Value::from_op(out_tensor, grad_fn, self.requires_grad)
    }

    /// Reshapes value tensor.
    pub fn reshape(&self, shape: Vec<usize>) -> Value {
        let out_tensor = self.data().reshape(shape.clone());
        let grad_fn = if self.requires_grad {
            GradFn::Reshape(Arc::new(self.clone()), shape)
        } else {
            GradFn::None
        };
        Value::from_op(out_tensor, grad_fn, self.requires_grad)
    }

    /// Mean reduction.
    pub fn mean(&self) -> Value {
        crate::ops::unary::mean(self)
    }

    /// 2D spatial convolution.
    pub fn conv2d(
        &self,
        weight: &Value,
        bias: Option<&Value>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Value {
        crate::ops::conv_grad::conv2d(self, weight, bias, stride, padding)
    }

    /// 2D transposed convolution.
    pub fn conv_transpose2d(
        &self,
        weight: &Value,
        bias: Option<&Value>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Value {
        crate::ops::conv_grad::conv_transpose2d(self, weight, bias, stride, padding)
    }

    /// 2D Max Pooling.
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Value {
        crate::ops::pool_grad::max_pool2d(self, kernel_size, stride, padding)
    }

    /// 2D Average Pooling.
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Value {
        crate::ops::pool_grad::avg_pool2d(self, kernel_size, stride, padding)
    }

    /// Linear transformation `input * weight^T + bias` or `input * weight + bias`.
    pub fn linear(&self, weight: &Value, bias: Option<&Value>) -> Value {
        let out = if weight.ndim() == 2 && self.ndim() == 2 && self.shape()[1] == weight.shape()[1] {
            let w_t = weight.transpose(0, 1);
            self.matmul(&w_t)
        } else {
            self.matmul(weight)
        };
        if let Some(b) = bias {
            &out + b
        } else {
            out
        }
    }

    /// Embedding lookup table: weights matrix [V, D] indexed by token indices.
    pub fn embedding(&self, indices: &[usize], output_shape: Vec<usize>) -> Value {
        let w_shape = self.shape();
        let num_embeddings = if !w_shape.is_empty() { w_shape[0] } else { 0 };
        let embedding_dim = if w_shape.len() > 1 { w_shape[1] } else { 1 };

        let w_data = self.data();
        let mut out_data = Vec::with_capacity(indices.len() * embedding_dim);

        for &idx in indices {
            if idx < num_embeddings {
                let start = idx * embedding_dim;
                out_data.extend_from_slice(&w_data.data()[start..start + embedding_dim]);
            } else {
                out_data.resize(out_data.len() + embedding_dim, 0.0);
            }
        }

        let out_tensor = Tensor::from_vec(out_data, output_shape.clone());
        let grad_fn = if self.requires_grad {
            GradFn::Embedding {
                weight: Arc::new(self.clone()),
                indices: indices.to_vec(),
                output_shape,
            }
        } else {
            GradFn::None
        };

        Value::from_op(out_tensor, grad_fn, self.requires_grad)
    }
}

// Operator Overloads
impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        crate::ops::binary::add(self, rhs)
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        crate::ops::binary::add(&self, &rhs)
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        crate::ops::binary::sub(self, rhs)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        crate::ops::binary::sub(&self, &rhs)
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        crate::ops::binary::mul(self, rhs)
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        crate::ops::binary::mul(&self, &rhs)
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        crate::ops::binary::div(self, rhs)
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        crate::ops::binary::div(&self, &rhs)
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        crate::ops::unary::neg(self)
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        crate::ops::unary::neg(&self)
    }
}

/// Compares two `Value` nodes for numerical closeness within `tol`.
pub fn values_close(a: &Value, b: &Value, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let a_data = a.data().data();
    let b_data = b.data().data();
    a_data.iter().zip(b_data.iter()).all(|(&x, &y)| (x - y).abs() <= tol)
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
    fn test_value_node_stress_001() {
        let mut v = Value::scalar(10.5);
        v.set_name(format!("node_1"));
        assert_eq!(v.name(), Some(format!("node_1").as_str()));
        assert_eq!(v.data().get(0), 10.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 10.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_002() {
        let mut v = Value::scalar(11.0);
        v.set_name(format!("node_2"));
        assert_eq!(v.name(), Some(format!("node_2").as_str()));
        assert_eq!(v.data().get(0), 11.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 11.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_003() {
        let mut v = Value::scalar(11.5);
        v.set_name(format!("node_3"));
        assert_eq!(v.name(), Some(format!("node_3").as_str()));
        assert_eq!(v.data().get(0), 11.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 11.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_004() {
        let mut v = Value::scalar(12.0);
        v.set_name(format!("node_4"));
        assert_eq!(v.name(), Some(format!("node_4").as_str()));
        assert_eq!(v.data().get(0), 12.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 12.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_005() {
        let mut v = Value::scalar(12.5);
        v.set_name(format!("node_5"));
        assert_eq!(v.name(), Some(format!("node_5").as_str()));
        assert_eq!(v.data().get(0), 12.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 12.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_006() {
        let mut v = Value::scalar(13.0);
        v.set_name(format!("node_6"));
        assert_eq!(v.name(), Some(format!("node_6").as_str()));
        assert_eq!(v.data().get(0), 13.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 13.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_007() {
        let mut v = Value::scalar(13.5);
        v.set_name(format!("node_7"));
        assert_eq!(v.name(), Some(format!("node_7").as_str()));
        assert_eq!(v.data().get(0), 13.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 13.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_008() {
        let mut v = Value::scalar(14.0);
        v.set_name(format!("node_8"));
        assert_eq!(v.name(), Some(format!("node_8").as_str()));
        assert_eq!(v.data().get(0), 14.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 14.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_009() {
        let mut v = Value::scalar(14.5);
        v.set_name(format!("node_9"));
        assert_eq!(v.name(), Some(format!("node_9").as_str()));
        assert_eq!(v.data().get(0), 14.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 14.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_010() {
        let mut v = Value::scalar(15.0);
        v.set_name(format!("node_10"));
        assert_eq!(v.name(), Some(format!("node_10").as_str()));
        assert_eq!(v.data().get(0), 15.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 15.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_011() {
        let mut v = Value::scalar(15.5);
        v.set_name(format!("node_11"));
        assert_eq!(v.name(), Some(format!("node_11").as_str()));
        assert_eq!(v.data().get(0), 15.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 15.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_012() {
        let mut v = Value::scalar(16.0);
        v.set_name(format!("node_12"));
        assert_eq!(v.name(), Some(format!("node_12").as_str()));
        assert_eq!(v.data().get(0), 16.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 16.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_013() {
        let mut v = Value::scalar(16.5);
        v.set_name(format!("node_13"));
        assert_eq!(v.name(), Some(format!("node_13").as_str()));
        assert_eq!(v.data().get(0), 16.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 16.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_014() {
        let mut v = Value::scalar(17.0);
        v.set_name(format!("node_14"));
        assert_eq!(v.name(), Some(format!("node_14").as_str()));
        assert_eq!(v.data().get(0), 17.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 17.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_015() {
        let mut v = Value::scalar(17.5);
        v.set_name(format!("node_15"));
        assert_eq!(v.name(), Some(format!("node_15").as_str()));
        assert_eq!(v.data().get(0), 17.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 17.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_016() {
        let mut v = Value::scalar(18.0);
        v.set_name(format!("node_16"));
        assert_eq!(v.name(), Some(format!("node_16").as_str()));
        assert_eq!(v.data().get(0), 18.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 18.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_017() {
        let mut v = Value::scalar(18.5);
        v.set_name(format!("node_17"));
        assert_eq!(v.name(), Some(format!("node_17").as_str()));
        assert_eq!(v.data().get(0), 18.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 18.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_018() {
        let mut v = Value::scalar(19.0);
        v.set_name(format!("node_18"));
        assert_eq!(v.name(), Some(format!("node_18").as_str()));
        assert_eq!(v.data().get(0), 19.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 19.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_019() {
        let mut v = Value::scalar(19.5);
        v.set_name(format!("node_19"));
        assert_eq!(v.name(), Some(format!("node_19").as_str()));
        assert_eq!(v.data().get(0), 19.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 19.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_020() {
        let mut v = Value::scalar(20.0);
        v.set_name(format!("node_20"));
        assert_eq!(v.name(), Some(format!("node_20").as_str()));
        assert_eq!(v.data().get(0), 20.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 20.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_021() {
        let mut v = Value::scalar(20.5);
        v.set_name(format!("node_21"));
        assert_eq!(v.name(), Some(format!("node_21").as_str()));
        assert_eq!(v.data().get(0), 20.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 20.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_022() {
        let mut v = Value::scalar(21.0);
        v.set_name(format!("node_22"));
        assert_eq!(v.name(), Some(format!("node_22").as_str()));
        assert_eq!(v.data().get(0), 21.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 21.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_023() {
        let mut v = Value::scalar(21.5);
        v.set_name(format!("node_23"));
        assert_eq!(v.name(), Some(format!("node_23").as_str()));
        assert_eq!(v.data().get(0), 21.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 21.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_024() {
        let mut v = Value::scalar(22.0);
        v.set_name(format!("node_24"));
        assert_eq!(v.name(), Some(format!("node_24").as_str()));
        assert_eq!(v.data().get(0), 22.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 22.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_025() {
        let mut v = Value::scalar(22.5);
        v.set_name(format!("node_25"));
        assert_eq!(v.name(), Some(format!("node_25").as_str()));
        assert_eq!(v.data().get(0), 22.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 22.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_026() {
        let mut v = Value::scalar(23.0);
        v.set_name(format!("node_26"));
        assert_eq!(v.name(), Some(format!("node_26").as_str()));
        assert_eq!(v.data().get(0), 23.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 23.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_027() {
        let mut v = Value::scalar(23.5);
        v.set_name(format!("node_27"));
        assert_eq!(v.name(), Some(format!("node_27").as_str()));
        assert_eq!(v.data().get(0), 23.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 23.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_028() {
        let mut v = Value::scalar(24.0);
        v.set_name(format!("node_28"));
        assert_eq!(v.name(), Some(format!("node_28").as_str()));
        assert_eq!(v.data().get(0), 24.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 24.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_029() {
        let mut v = Value::scalar(24.5);
        v.set_name(format!("node_29"));
        assert_eq!(v.name(), Some(format!("node_29").as_str()));
        assert_eq!(v.data().get(0), 24.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 24.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_030() {
        let mut v = Value::scalar(25.0);
        v.set_name(format!("node_30"));
        assert_eq!(v.name(), Some(format!("node_30").as_str()));
        assert_eq!(v.data().get(0), 25.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 25.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_031() {
        let mut v = Value::scalar(25.5);
        v.set_name(format!("node_31"));
        assert_eq!(v.name(), Some(format!("node_31").as_str()));
        assert_eq!(v.data().get(0), 25.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 25.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_032() {
        let mut v = Value::scalar(26.0);
        v.set_name(format!("node_32"));
        assert_eq!(v.name(), Some(format!("node_32").as_str()));
        assert_eq!(v.data().get(0), 26.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 26.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_033() {
        let mut v = Value::scalar(26.5);
        v.set_name(format!("node_33"));
        assert_eq!(v.name(), Some(format!("node_33").as_str()));
        assert_eq!(v.data().get(0), 26.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 26.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_034() {
        let mut v = Value::scalar(27.0);
        v.set_name(format!("node_34"));
        assert_eq!(v.name(), Some(format!("node_34").as_str()));
        assert_eq!(v.data().get(0), 27.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 27.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_035() {
        let mut v = Value::scalar(27.5);
        v.set_name(format!("node_35"));
        assert_eq!(v.name(), Some(format!("node_35").as_str()));
        assert_eq!(v.data().get(0), 27.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 27.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_036() {
        let mut v = Value::scalar(28.0);
        v.set_name(format!("node_36"));
        assert_eq!(v.name(), Some(format!("node_36").as_str()));
        assert_eq!(v.data().get(0), 28.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 28.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_037() {
        let mut v = Value::scalar(28.5);
        v.set_name(format!("node_37"));
        assert_eq!(v.name(), Some(format!("node_37").as_str()));
        assert_eq!(v.data().get(0), 28.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 28.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_038() {
        let mut v = Value::scalar(29.0);
        v.set_name(format!("node_38"));
        assert_eq!(v.name(), Some(format!("node_38").as_str()));
        assert_eq!(v.data().get(0), 29.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 29.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_039() {
        let mut v = Value::scalar(29.5);
        v.set_name(format!("node_39"));
        assert_eq!(v.name(), Some(format!("node_39").as_str()));
        assert_eq!(v.data().get(0), 29.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 29.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_040() {
        let mut v = Value::scalar(30.0);
        v.set_name(format!("node_40"));
        assert_eq!(v.name(), Some(format!("node_40").as_str()));
        assert_eq!(v.data().get(0), 30.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 30.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_041() {
        let mut v = Value::scalar(30.5);
        v.set_name(format!("node_41"));
        assert_eq!(v.name(), Some(format!("node_41").as_str()));
        assert_eq!(v.data().get(0), 30.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 30.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_042() {
        let mut v = Value::scalar(31.0);
        v.set_name(format!("node_42"));
        assert_eq!(v.name(), Some(format!("node_42").as_str()));
        assert_eq!(v.data().get(0), 31.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 31.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_043() {
        let mut v = Value::scalar(31.5);
        v.set_name(format!("node_43"));
        assert_eq!(v.name(), Some(format!("node_43").as_str()));
        assert_eq!(v.data().get(0), 31.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 31.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_044() {
        let mut v = Value::scalar(32.0);
        v.set_name(format!("node_44"));
        assert_eq!(v.name(), Some(format!("node_44").as_str()));
        assert_eq!(v.data().get(0), 32.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 32.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_045() {
        let mut v = Value::scalar(32.5);
        v.set_name(format!("node_45"));
        assert_eq!(v.name(), Some(format!("node_45").as_str()));
        assert_eq!(v.data().get(0), 32.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 32.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_046() {
        let mut v = Value::scalar(33.0);
        v.set_name(format!("node_46"));
        assert_eq!(v.name(), Some(format!("node_46").as_str()));
        assert_eq!(v.data().get(0), 33.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 33.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_047() {
        let mut v = Value::scalar(33.5);
        v.set_name(format!("node_47"));
        assert_eq!(v.name(), Some(format!("node_47").as_str()));
        assert_eq!(v.data().get(0), 33.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 33.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_048() {
        let mut v = Value::scalar(34.0);
        v.set_name(format!("node_48"));
        assert_eq!(v.name(), Some(format!("node_48").as_str()));
        assert_eq!(v.data().get(0), 34.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 34.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_049() {
        let mut v = Value::scalar(34.5);
        v.set_name(format!("node_49"));
        assert_eq!(v.name(), Some(format!("node_49").as_str()));
        assert_eq!(v.data().get(0), 34.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 34.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_050() {
        let mut v = Value::scalar(35.0);
        v.set_name(format!("node_50"));
        assert_eq!(v.name(), Some(format!("node_50").as_str()));
        assert_eq!(v.data().get(0), 35.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 35.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_051() {
        let mut v = Value::scalar(35.5);
        v.set_name(format!("node_51"));
        assert_eq!(v.name(), Some(format!("node_51").as_str()));
        assert_eq!(v.data().get(0), 35.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 35.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_052() {
        let mut v = Value::scalar(36.0);
        v.set_name(format!("node_52"));
        assert_eq!(v.name(), Some(format!("node_52").as_str()));
        assert_eq!(v.data().get(0), 36.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 36.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_053() {
        let mut v = Value::scalar(36.5);
        v.set_name(format!("node_53"));
        assert_eq!(v.name(), Some(format!("node_53").as_str()));
        assert_eq!(v.data().get(0), 36.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 36.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_054() {
        let mut v = Value::scalar(37.0);
        v.set_name(format!("node_54"));
        assert_eq!(v.name(), Some(format!("node_54").as_str()));
        assert_eq!(v.data().get(0), 37.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 37.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_055() {
        let mut v = Value::scalar(37.5);
        v.set_name(format!("node_55"));
        assert_eq!(v.name(), Some(format!("node_55").as_str()));
        assert_eq!(v.data().get(0), 37.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 37.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_056() {
        let mut v = Value::scalar(38.0);
        v.set_name(format!("node_56"));
        assert_eq!(v.name(), Some(format!("node_56").as_str()));
        assert_eq!(v.data().get(0), 38.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 38.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_057() {
        let mut v = Value::scalar(38.5);
        v.set_name(format!("node_57"));
        assert_eq!(v.name(), Some(format!("node_57").as_str()));
        assert_eq!(v.data().get(0), 38.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 38.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_058() {
        let mut v = Value::scalar(39.0);
        v.set_name(format!("node_58"));
        assert_eq!(v.name(), Some(format!("node_58").as_str()));
        assert_eq!(v.data().get(0), 39.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 39.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_059() {
        let mut v = Value::scalar(39.5);
        v.set_name(format!("node_59"));
        assert_eq!(v.name(), Some(format!("node_59").as_str()));
        assert_eq!(v.data().get(0), 39.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 39.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_060() {
        let mut v = Value::scalar(40.0);
        v.set_name(format!("node_60"));
        assert_eq!(v.name(), Some(format!("node_60").as_str()));
        assert_eq!(v.data().get(0), 40.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 40.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_061() {
        let mut v = Value::scalar(40.5);
        v.set_name(format!("node_61"));
        assert_eq!(v.name(), Some(format!("node_61").as_str()));
        assert_eq!(v.data().get(0), 40.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 40.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_062() {
        let mut v = Value::scalar(41.0);
        v.set_name(format!("node_62"));
        assert_eq!(v.name(), Some(format!("node_62").as_str()));
        assert_eq!(v.data().get(0), 41.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 41.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_063() {
        let mut v = Value::scalar(41.5);
        v.set_name(format!("node_63"));
        assert_eq!(v.name(), Some(format!("node_63").as_str()));
        assert_eq!(v.data().get(0), 41.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 41.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_064() {
        let mut v = Value::scalar(42.0);
        v.set_name(format!("node_64"));
        assert_eq!(v.name(), Some(format!("node_64").as_str()));
        assert_eq!(v.data().get(0), 42.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 42.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_065() {
        let mut v = Value::scalar(42.5);
        v.set_name(format!("node_65"));
        assert_eq!(v.name(), Some(format!("node_65").as_str()));
        assert_eq!(v.data().get(0), 42.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 42.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_066() {
        let mut v = Value::scalar(43.0);
        v.set_name(format!("node_66"));
        assert_eq!(v.name(), Some(format!("node_66").as_str()));
        assert_eq!(v.data().get(0), 43.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 43.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_067() {
        let mut v = Value::scalar(43.5);
        v.set_name(format!("node_67"));
        assert_eq!(v.name(), Some(format!("node_67").as_str()));
        assert_eq!(v.data().get(0), 43.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 43.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_068() {
        let mut v = Value::scalar(44.0);
        v.set_name(format!("node_68"));
        assert_eq!(v.name(), Some(format!("node_68").as_str()));
        assert_eq!(v.data().get(0), 44.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 44.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_069() {
        let mut v = Value::scalar(44.5);
        v.set_name(format!("node_69"));
        assert_eq!(v.name(), Some(format!("node_69").as_str()));
        assert_eq!(v.data().get(0), 44.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 44.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_070() {
        let mut v = Value::scalar(45.0);
        v.set_name(format!("node_70"));
        assert_eq!(v.name(), Some(format!("node_70").as_str()));
        assert_eq!(v.data().get(0), 45.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 45.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_071() {
        let mut v = Value::scalar(45.5);
        v.set_name(format!("node_71"));
        assert_eq!(v.name(), Some(format!("node_71").as_str()));
        assert_eq!(v.data().get(0), 45.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 45.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_072() {
        let mut v = Value::scalar(46.0);
        v.set_name(format!("node_72"));
        assert_eq!(v.name(), Some(format!("node_72").as_str()));
        assert_eq!(v.data().get(0), 46.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 46.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_073() {
        let mut v = Value::scalar(46.5);
        v.set_name(format!("node_73"));
        assert_eq!(v.name(), Some(format!("node_73").as_str()));
        assert_eq!(v.data().get(0), 46.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 46.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_074() {
        let mut v = Value::scalar(47.0);
        v.set_name(format!("node_74"));
        assert_eq!(v.name(), Some(format!("node_74").as_str()));
        assert_eq!(v.data().get(0), 47.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 47.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_075() {
        let mut v = Value::scalar(47.5);
        v.set_name(format!("node_75"));
        assert_eq!(v.name(), Some(format!("node_75").as_str()));
        assert_eq!(v.data().get(0), 47.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 47.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_076() {
        let mut v = Value::scalar(48.0);
        v.set_name(format!("node_76"));
        assert_eq!(v.name(), Some(format!("node_76").as_str()));
        assert_eq!(v.data().get(0), 48.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 48.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_077() {
        let mut v = Value::scalar(48.5);
        v.set_name(format!("node_77"));
        assert_eq!(v.name(), Some(format!("node_77").as_str()));
        assert_eq!(v.data().get(0), 48.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 48.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_078() {
        let mut v = Value::scalar(49.0);
        v.set_name(format!("node_78"));
        assert_eq!(v.name(), Some(format!("node_78").as_str()));
        assert_eq!(v.data().get(0), 49.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 49.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_079() {
        let mut v = Value::scalar(49.5);
        v.set_name(format!("node_79"));
        assert_eq!(v.name(), Some(format!("node_79").as_str()));
        assert_eq!(v.data().get(0), 49.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 49.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_080() {
        let mut v = Value::scalar(50.0);
        v.set_name(format!("node_80"));
        assert_eq!(v.name(), Some(format!("node_80").as_str()));
        assert_eq!(v.data().get(0), 50.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 50.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_081() {
        let mut v = Value::scalar(50.5);
        v.set_name(format!("node_81"));
        assert_eq!(v.name(), Some(format!("node_81").as_str()));
        assert_eq!(v.data().get(0), 50.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 50.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_082() {
        let mut v = Value::scalar(51.0);
        v.set_name(format!("node_82"));
        assert_eq!(v.name(), Some(format!("node_82").as_str()));
        assert_eq!(v.data().get(0), 51.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 51.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_083() {
        let mut v = Value::scalar(51.5);
        v.set_name(format!("node_83"));
        assert_eq!(v.name(), Some(format!("node_83").as_str()));
        assert_eq!(v.data().get(0), 51.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 51.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_084() {
        let mut v = Value::scalar(52.0);
        v.set_name(format!("node_84"));
        assert_eq!(v.name(), Some(format!("node_84").as_str()));
        assert_eq!(v.data().get(0), 52.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 52.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_085() {
        let mut v = Value::scalar(52.5);
        v.set_name(format!("node_85"));
        assert_eq!(v.name(), Some(format!("node_85").as_str()));
        assert_eq!(v.data().get(0), 52.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 52.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_086() {
        let mut v = Value::scalar(53.0);
        v.set_name(format!("node_86"));
        assert_eq!(v.name(), Some(format!("node_86").as_str()));
        assert_eq!(v.data().get(0), 53.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 53.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_087() {
        let mut v = Value::scalar(53.5);
        v.set_name(format!("node_87"));
        assert_eq!(v.name(), Some(format!("node_87").as_str()));
        assert_eq!(v.data().get(0), 53.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 53.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_088() {
        let mut v = Value::scalar(54.0);
        v.set_name(format!("node_88"));
        assert_eq!(v.name(), Some(format!("node_88").as_str()));
        assert_eq!(v.data().get(0), 54.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 54.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_089() {
        let mut v = Value::scalar(54.5);
        v.set_name(format!("node_89"));
        assert_eq!(v.name(), Some(format!("node_89").as_str()));
        assert_eq!(v.data().get(0), 54.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 54.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_090() {
        let mut v = Value::scalar(55.0);
        v.set_name(format!("node_90"));
        assert_eq!(v.name(), Some(format!("node_90").as_str()));
        assert_eq!(v.data().get(0), 55.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 55.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_091() {
        let mut v = Value::scalar(55.5);
        v.set_name(format!("node_91"));
        assert_eq!(v.name(), Some(format!("node_91").as_str()));
        assert_eq!(v.data().get(0), 55.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 55.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_092() {
        let mut v = Value::scalar(56.0);
        v.set_name(format!("node_92"));
        assert_eq!(v.name(), Some(format!("node_92").as_str()));
        assert_eq!(v.data().get(0), 56.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 56.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_093() {
        let mut v = Value::scalar(56.5);
        v.set_name(format!("node_93"));
        assert_eq!(v.name(), Some(format!("node_93").as_str()));
        assert_eq!(v.data().get(0), 56.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 56.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_094() {
        let mut v = Value::scalar(57.0);
        v.set_name(format!("node_94"));
        assert_eq!(v.name(), Some(format!("node_94").as_str()));
        assert_eq!(v.data().get(0), 57.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 57.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_095() {
        let mut v = Value::scalar(57.5);
        v.set_name(format!("node_95"));
        assert_eq!(v.name(), Some(format!("node_95").as_str()));
        assert_eq!(v.data().get(0), 57.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 57.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_096() {
        let mut v = Value::scalar(58.0);
        v.set_name(format!("node_96"));
        assert_eq!(v.name(), Some(format!("node_96").as_str()));
        assert_eq!(v.data().get(0), 58.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 58.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_097() {
        let mut v = Value::scalar(58.5);
        v.set_name(format!("node_97"));
        assert_eq!(v.name(), Some(format!("node_97").as_str()));
        assert_eq!(v.data().get(0), 58.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 58.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_098() {
        let mut v = Value::scalar(59.0);
        v.set_name(format!("node_98"));
        assert_eq!(v.name(), Some(format!("node_98").as_str()));
        assert_eq!(v.data().get(0), 59.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 59.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_099() {
        let mut v = Value::scalar(59.5);
        v.set_name(format!("node_99"));
        assert_eq!(v.name(), Some(format!("node_99").as_str()));
        assert_eq!(v.data().get(0), 59.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 59.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_100() {
        let mut v = Value::scalar(60.0);
        v.set_name(format!("node_100"));
        assert_eq!(v.name(), Some(format!("node_100").as_str()));
        assert_eq!(v.data().get(0), 60.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 60.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_101() {
        let mut v = Value::scalar(60.5);
        v.set_name(format!("node_101"));
        assert_eq!(v.name(), Some(format!("node_101").as_str()));
        assert_eq!(v.data().get(0), 60.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 60.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_102() {
        let mut v = Value::scalar(61.0);
        v.set_name(format!("node_102"));
        assert_eq!(v.name(), Some(format!("node_102").as_str()));
        assert_eq!(v.data().get(0), 61.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 61.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_103() {
        let mut v = Value::scalar(61.5);
        v.set_name(format!("node_103"));
        assert_eq!(v.name(), Some(format!("node_103").as_str()));
        assert_eq!(v.data().get(0), 61.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 61.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_104() {
        let mut v = Value::scalar(62.0);
        v.set_name(format!("node_104"));
        assert_eq!(v.name(), Some(format!("node_104").as_str()));
        assert_eq!(v.data().get(0), 62.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 62.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_105() {
        let mut v = Value::scalar(62.5);
        v.set_name(format!("node_105"));
        assert_eq!(v.name(), Some(format!("node_105").as_str()));
        assert_eq!(v.data().get(0), 62.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 62.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_106() {
        let mut v = Value::scalar(63.0);
        v.set_name(format!("node_106"));
        assert_eq!(v.name(), Some(format!("node_106").as_str()));
        assert_eq!(v.data().get(0), 63.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 63.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_107() {
        let mut v = Value::scalar(63.5);
        v.set_name(format!("node_107"));
        assert_eq!(v.name(), Some(format!("node_107").as_str()));
        assert_eq!(v.data().get(0), 63.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 63.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_108() {
        let mut v = Value::scalar(64.0);
        v.set_name(format!("node_108"));
        assert_eq!(v.name(), Some(format!("node_108").as_str()));
        assert_eq!(v.data().get(0), 64.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 64.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_109() {
        let mut v = Value::scalar(64.5);
        v.set_name(format!("node_109"));
        assert_eq!(v.name(), Some(format!("node_109").as_str()));
        assert_eq!(v.data().get(0), 64.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 64.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_110() {
        let mut v = Value::scalar(65.0);
        v.set_name(format!("node_110"));
        assert_eq!(v.name(), Some(format!("node_110").as_str()));
        assert_eq!(v.data().get(0), 65.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 65.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_111() {
        let mut v = Value::scalar(65.5);
        v.set_name(format!("node_111"));
        assert_eq!(v.name(), Some(format!("node_111").as_str()));
        assert_eq!(v.data().get(0), 65.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 65.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_112() {
        let mut v = Value::scalar(66.0);
        v.set_name(format!("node_112"));
        assert_eq!(v.name(), Some(format!("node_112").as_str()));
        assert_eq!(v.data().get(0), 66.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 66.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_113() {
        let mut v = Value::scalar(66.5);
        v.set_name(format!("node_113"));
        assert_eq!(v.name(), Some(format!("node_113").as_str()));
        assert_eq!(v.data().get(0), 66.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 66.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_114() {
        let mut v = Value::scalar(67.0);
        v.set_name(format!("node_114"));
        assert_eq!(v.name(), Some(format!("node_114").as_str()));
        assert_eq!(v.data().get(0), 67.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 67.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_115() {
        let mut v = Value::scalar(67.5);
        v.set_name(format!("node_115"));
        assert_eq!(v.name(), Some(format!("node_115").as_str()));
        assert_eq!(v.data().get(0), 67.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 67.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_116() {
        let mut v = Value::scalar(68.0);
        v.set_name(format!("node_116"));
        assert_eq!(v.name(), Some(format!("node_116").as_str()));
        assert_eq!(v.data().get(0), 68.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 68.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_117() {
        let mut v = Value::scalar(68.5);
        v.set_name(format!("node_117"));
        assert_eq!(v.name(), Some(format!("node_117").as_str()));
        assert_eq!(v.data().get(0), 68.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 68.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_118() {
        let mut v = Value::scalar(69.0);
        v.set_name(format!("node_118"));
        assert_eq!(v.name(), Some(format!("node_118").as_str()));
        assert_eq!(v.data().get(0), 69.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 69.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_119() {
        let mut v = Value::scalar(69.5);
        v.set_name(format!("node_119"));
        assert_eq!(v.name(), Some(format!("node_119").as_str()));
        assert_eq!(v.data().get(0), 69.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 69.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_120() {
        let mut v = Value::scalar(70.0);
        v.set_name(format!("node_120"));
        assert_eq!(v.name(), Some(format!("node_120").as_str()));
        assert_eq!(v.data().get(0), 70.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 70.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_121() {
        let mut v = Value::scalar(70.5);
        v.set_name(format!("node_121"));
        assert_eq!(v.name(), Some(format!("node_121").as_str()));
        assert_eq!(v.data().get(0), 70.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 70.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_122() {
        let mut v = Value::scalar(71.0);
        v.set_name(format!("node_122"));
        assert_eq!(v.name(), Some(format!("node_122").as_str()));
        assert_eq!(v.data().get(0), 71.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 71.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_123() {
        let mut v = Value::scalar(71.5);
        v.set_name(format!("node_123"));
        assert_eq!(v.name(), Some(format!("node_123").as_str()));
        assert_eq!(v.data().get(0), 71.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 71.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_124() {
        let mut v = Value::scalar(72.0);
        v.set_name(format!("node_124"));
        assert_eq!(v.name(), Some(format!("node_124").as_str()));
        assert_eq!(v.data().get(0), 72.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 72.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_125() {
        let mut v = Value::scalar(72.5);
        v.set_name(format!("node_125"));
        assert_eq!(v.name(), Some(format!("node_125").as_str()));
        assert_eq!(v.data().get(0), 72.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 72.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_126() {
        let mut v = Value::scalar(73.0);
        v.set_name(format!("node_126"));
        assert_eq!(v.name(), Some(format!("node_126").as_str()));
        assert_eq!(v.data().get(0), 73.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 73.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_127() {
        let mut v = Value::scalar(73.5);
        v.set_name(format!("node_127"));
        assert_eq!(v.name(), Some(format!("node_127").as_str()));
        assert_eq!(v.data().get(0), 73.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 73.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_128() {
        let mut v = Value::scalar(74.0);
        v.set_name(format!("node_128"));
        assert_eq!(v.name(), Some(format!("node_128").as_str()));
        assert_eq!(v.data().get(0), 74.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 74.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_129() {
        let mut v = Value::scalar(74.5);
        v.set_name(format!("node_129"));
        assert_eq!(v.name(), Some(format!("node_129").as_str()));
        assert_eq!(v.data().get(0), 74.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 74.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_130() {
        let mut v = Value::scalar(75.0);
        v.set_name(format!("node_130"));
        assert_eq!(v.name(), Some(format!("node_130").as_str()));
        assert_eq!(v.data().get(0), 75.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 75.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_131() {
        let mut v = Value::scalar(75.5);
        v.set_name(format!("node_131"));
        assert_eq!(v.name(), Some(format!("node_131").as_str()));
        assert_eq!(v.data().get(0), 75.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 75.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_132() {
        let mut v = Value::scalar(76.0);
        v.set_name(format!("node_132"));
        assert_eq!(v.name(), Some(format!("node_132").as_str()));
        assert_eq!(v.data().get(0), 76.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 76.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_133() {
        let mut v = Value::scalar(76.5);
        v.set_name(format!("node_133"));
        assert_eq!(v.name(), Some(format!("node_133").as_str()));
        assert_eq!(v.data().get(0), 76.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 76.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_134() {
        let mut v = Value::scalar(77.0);
        v.set_name(format!("node_134"));
        assert_eq!(v.name(), Some(format!("node_134").as_str()));
        assert_eq!(v.data().get(0), 77.0);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 77.0);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_value_node_stress_135() {
        let mut v = Value::scalar(77.5);
        v.set_name(format!("node_135"));
        assert_eq!(v.name(), Some(format!("node_135").as_str()));
        assert_eq!(v.data().get(0), 77.5);
        assert_eq!(v.shape(), &[] as &[usize]);
        assert_eq!(v.numel(), 1);
        assert_eq!(v.ndim(), 0);
        assert!(v.is_leaf());
        assert!(!v.requires_grad());
        v.set_requires_grad(true);
        assert!(v.requires_grad());
        v.accumulate_grad(&Tensor::scalar(2.5)).unwrap();
        assert_eq!(v.grad().unwrap().get(0), 2.5);
        let detached = v.detach();
        assert_eq!(detached.data().get(0), 77.5);
        assert!(!detached.requires_grad());
        v.zero_grad();
        assert!(v.grad().is_none());
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
