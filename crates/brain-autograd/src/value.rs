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

    /// Returns element at flat index from underlying tensor.
    pub fn get(&self, idx: usize) -> f64 {
        self.data.data()[idx]
    }

    /// Returns element at 4D index from underlying tensor.
    pub fn get_4d(&self, b: usize, c: usize, h: usize, w: usize) -> f64 {
        let shape = self.shape();
        if shape.len() == 4 {
            let idx = ((b * shape[1] + c) * shape[2] + h) * shape[3] + w;
            self.data.data()[idx]
        } else {
            0.0
        }
    }

    /// Returns copy of the underlying flattened vector of elements.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.to_vec()
    }

    /// In-place updates the underlying Tensor data payload.
    pub fn set_data(&mut self, tensor: Tensor) {
        self.data = Arc::new(tensor);
    }

    /// Accumulates incoming gradient into this node.
    pub fn accumulate_grad(&self, incoming: &Tensor) -> BrainResult<()> {
        let mut g = self.grad.write().unwrap();
        match g.as_mut() {
            Some(existing) => {
                if existing.shape() != incoming.shape() {
                    let unbroadcasted =
                        crate::grad_fns::util::sum_to_shape(incoming, existing.shape())?;
                    *existing = arith_t::add(existing, &unbroadcasted);
                } else {
                    *existing = arith_t::add(existing, incoming);
                }
            }
            None => {
                if incoming.shape() != self.shape() {
                    let unbroadcasted =
                        crate::grad_fns::util::sum_to_shape(incoming, self.shape())?;
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

    /// Elementwise absolute value.
    pub fn abs(&self) -> Value {
        crate::ops::unary::abs(self)
    }

    /// Elementwise clamp between `min_val` and `max_val`.
    pub fn clamp(&self, min_val: f64, max_val: f64) -> Value {
        crate::ops::unary::clamp(self, min_val, max_val)
    }

    /// Elementwise sine.
    pub fn sin(&self) -> Value {
        crate::ops::unary::sin(self)
    }

    /// Elementwise cosine.
    pub fn cos(&self) -> Value {
        crate::ops::unary::cos(self)
    }

    /// Elementwise reciprocal: `1 / x`.
    pub fn recip(&self) -> Value {
        crate::ops::unary::recip(self)
    }

    /// Elementwise square: `x * x`.
    pub fn square(&self) -> Value {
        crate::ops::unary::square(self)
    }

    /// Elementwise sign: `-1.0, 0.0, 1.0`.
    pub fn sign(&self) -> Value {
        crate::ops::unary::sign(self)
    }

    /// Elementwise minimum with another value.
    pub fn min_elem(&self, other: &Value) -> Value {
        crate::ops::binary::min_elem(self, other)
    }

    /// Elementwise maximum with another value.
    pub fn max_elem(&self, other: &Value) -> Value {
        crate::ops::binary::max_elem(self, other)
    }

    /// Conditional select: `where(cond, self, other)`.
    pub fn where_cond(&self, cond: &Value, other: &Value) -> Value {
        crate::ops::binary::where_cond(cond, self, other)
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
        let out = if weight.ndim() == 2 && self.ndim() == 2 && self.shape()[1] == weight.shape()[1]
        {
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
    a_data
        .iter()
        .zip(b_data.iter())
        .all(|(&x, &y)| (x - y).abs() <= tol)
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
