//! Autograd computation tape and differentiable mathematical operation nodes.
//!
//! This module provides backward gradient formulas and tape contexts for reverse-mode automatic differentiation.

use crate::tensor::Tensor;

/// Context object for holding intermediate forward activations for backward passes.
#[derive(Debug, Clone, Default)]
pub struct Context {
    saved_tensors: Vec<Tensor>,
}

impl Context {
    /// Creates a new empty context.
    pub fn new() -> Self {
        Context {
            saved_tensors: Vec::new(),
        }
    }

    /// Saves a tensor for backward computation.
    pub fn save_for_backward(&mut self, tensor: Tensor) {
        self.saved_tensors.push(tensor);
    }

    /// Retrieves a saved tensor by index.
    pub fn get_saved(&self, idx: usize) -> &Tensor {
        &self.saved_tensors[idx]
    }
}

/// Differentiable operation trait.
pub trait Function {
    /// Executes forward evaluation.
    fn forward(ctx: &mut Context, inputs: &[&Tensor]) -> Tensor;
    /// Executes reverse-mode gradient propagation.
    fn backward(ctx: &Context, grad_output: &Tensor) -> Vec<Tensor>;
}

/// Differentiable Addition: forward(a, b) = a + b; backward = (grad, grad).
pub struct AddOp;

impl Function for AddOp {
    fn forward(_ctx: &mut Context, inputs: &[&Tensor]) -> Tensor {
        crate::tensor::arithmetic::add(inputs[0], inputs[1])
    }

    fn backward(_ctx: &Context, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

/// Differentiable Multiplication: forward(a, b) = a * b; backward = (grad * b, grad * a).
pub struct MulOp;

impl Function for MulOp {
    fn forward(ctx: &mut Context, inputs: &[&Tensor]) -> Tensor {
        ctx.save_for_backward(inputs[0].clone());
        ctx.save_for_backward(inputs[1].clone());
        crate::tensor::arithmetic::mul(inputs[0], inputs[1])
    }

    fn backward(ctx: &Context, grad_output: &Tensor) -> Vec<Tensor> {
        let a = ctx.get_saved(0);
        let b = ctx.get_saved(1);
        let grad_a = crate::tensor::arithmetic::mul(grad_output, b);
        let grad_b = crate::tensor::arithmetic::mul(grad_output, a);
        vec![grad_a, grad_b]
    }
}

/// Differentiable ReLU: forward(x) = max(0, x); backward = grad * (x > 0).
pub struct ReluOp;

impl Function for ReluOp {
    fn forward(ctx: &mut Context, inputs: &[&Tensor]) -> Tensor {
        ctx.save_for_backward(inputs[0].clone());
        crate::tensor::math::relu(inputs[0])
    }

    fn backward(ctx: &Context, grad_output: &Tensor) -> Vec<Tensor> {
        let x = ctx.get_saved(0);
        let mask = crate::tensor::compare::gt_tensor(x, &Tensor::zeros(x.shape().to_vec()));
        let grad_x = crate::tensor::arithmetic::mul(grad_output, &mask);
        vec![grad_x]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autograd_add_and_mul() {
        let a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[4.0, 5.0], vec![2]);
        let mut ctx = Context::new();
        let out = MulOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.data(), &[8.0, 15.0]);

        let grad_out = Tensor::ones(vec![2]);
        let grads = MulOp::backward(&ctx, &grad_out);
        assert_eq!(grads[0].data(), &[4.0, 5.0]);
        assert_eq!(grads[1].data(), &[2.0, 3.0]);
    }

    #[test]
    fn test_op_forward() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let mut ctx = Context::new();
        let c = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(c.to_vec(), vec![4.0, 6.0]);
    }
}
