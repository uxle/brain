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
    fn test_function_stress_case_001() {
        let a = Tensor::from_slice(&[1.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 1.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_002() {
        let a = Tensor::from_slice(&[2.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 2.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_003() {
        let a = Tensor::from_slice(&[3.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 3.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_004() {
        let a = Tensor::from_slice(&[4.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 4.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_005() {
        let a = Tensor::from_slice(&[5.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 5.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_006() {
        let a = Tensor::from_slice(&[6.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 6.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_007() {
        let a = Tensor::from_slice(&[7.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 7.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_008() {
        let a = Tensor::from_slice(&[8.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 8.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_009() {
        let a = Tensor::from_slice(&[9.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 9.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_010() {
        let a = Tensor::from_slice(&[10.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 10.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_011() {
        let a = Tensor::from_slice(&[11.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 11.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_012() {
        let a = Tensor::from_slice(&[12.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 12.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_013() {
        let a = Tensor::from_slice(&[13.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 13.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_014() {
        let a = Tensor::from_slice(&[14.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 14.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_015() {
        let a = Tensor::from_slice(&[15.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 15.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_016() {
        let a = Tensor::from_slice(&[16.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 16.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_017() {
        let a = Tensor::from_slice(&[17.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 17.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_018() {
        let a = Tensor::from_slice(&[18.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 18.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_019() {
        let a = Tensor::from_slice(&[19.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 19.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_020() {
        let a = Tensor::from_slice(&[20.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 20.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_021() {
        let a = Tensor::from_slice(&[21.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 21.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_022() {
        let a = Tensor::from_slice(&[22.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 22.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_023() {
        let a = Tensor::from_slice(&[23.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 23.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_024() {
        let a = Tensor::from_slice(&[24.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 24.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_025() {
        let a = Tensor::from_slice(&[25.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 25.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_026() {
        let a = Tensor::from_slice(&[26.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 26.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_027() {
        let a = Tensor::from_slice(&[27.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 27.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_028() {
        let a = Tensor::from_slice(&[28.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 28.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_029() {
        let a = Tensor::from_slice(&[29.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 29.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_030() {
        let a = Tensor::from_slice(&[30.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 30.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_031() {
        let a = Tensor::from_slice(&[31.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 31.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_032() {
        let a = Tensor::from_slice(&[32.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 32.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_033() {
        let a = Tensor::from_slice(&[33.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 33.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_034() {
        let a = Tensor::from_slice(&[34.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 34.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_035() {
        let a = Tensor::from_slice(&[35.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 35.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_036() {
        let a = Tensor::from_slice(&[36.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 36.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_037() {
        let a = Tensor::from_slice(&[37.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 37.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_038() {
        let a = Tensor::from_slice(&[38.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 38.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_039() {
        let a = Tensor::from_slice(&[39.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 39.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_040() {
        let a = Tensor::from_slice(&[40.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 40.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_041() {
        let a = Tensor::from_slice(&[41.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 41.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_042() {
        let a = Tensor::from_slice(&[42.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 42.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_043() {
        let a = Tensor::from_slice(&[43.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 43.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_044() {
        let a = Tensor::from_slice(&[44.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 44.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_045() {
        let a = Tensor::from_slice(&[45.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 45.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_046() {
        let a = Tensor::from_slice(&[46.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 46.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_047() {
        let a = Tensor::from_slice(&[47.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 47.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_048() {
        let a = Tensor::from_slice(&[48.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 48.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_049() {
        let a = Tensor::from_slice(&[49.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 49.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_050() {
        let a = Tensor::from_slice(&[50.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 50.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_051() {
        let a = Tensor::from_slice(&[51.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 51.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_052() {
        let a = Tensor::from_slice(&[52.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 52.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_053() {
        let a = Tensor::from_slice(&[53.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 53.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_054() {
        let a = Tensor::from_slice(&[54.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 54.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_055() {
        let a = Tensor::from_slice(&[55.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 55.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_056() {
        let a = Tensor::from_slice(&[56.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 56.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_057() {
        let a = Tensor::from_slice(&[57.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 57.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_058() {
        let a = Tensor::from_slice(&[58.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 58.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_059() {
        let a = Tensor::from_slice(&[59.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 59.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_060() {
        let a = Tensor::from_slice(&[60.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 60.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_061() {
        let a = Tensor::from_slice(&[61.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 61.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_062() {
        let a = Tensor::from_slice(&[62.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 62.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_063() {
        let a = Tensor::from_slice(&[63.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 63.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_064() {
        let a = Tensor::from_slice(&[64.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 64.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_065() {
        let a = Tensor::from_slice(&[65.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 65.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_066() {
        let a = Tensor::from_slice(&[66.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 66.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_067() {
        let a = Tensor::from_slice(&[67.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 67.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_068() {
        let a = Tensor::from_slice(&[68.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 68.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_069() {
        let a = Tensor::from_slice(&[69.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 69.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_070() {
        let a = Tensor::from_slice(&[70.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 70.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_071() {
        let a = Tensor::from_slice(&[71.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 71.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_072() {
        let a = Tensor::from_slice(&[72.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 72.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_073() {
        let a = Tensor::from_slice(&[73.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 73.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_074() {
        let a = Tensor::from_slice(&[74.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 74.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_075() {
        let a = Tensor::from_slice(&[75.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 75.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_076() {
        let a = Tensor::from_slice(&[76.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 76.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_077() {
        let a = Tensor::from_slice(&[77.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 77.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_078() {
        let a = Tensor::from_slice(&[78.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 78.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_079() {
        let a = Tensor::from_slice(&[79.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 79.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_080() {
        let a = Tensor::from_slice(&[80.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 80.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_081() {
        let a = Tensor::from_slice(&[81.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 81.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_082() {
        let a = Tensor::from_slice(&[82.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 82.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_083() {
        let a = Tensor::from_slice(&[83.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 83.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_084() {
        let a = Tensor::from_slice(&[84.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 84.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_085() {
        let a = Tensor::from_slice(&[85.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 85.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_086() {
        let a = Tensor::from_slice(&[86.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 86.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_087() {
        let a = Tensor::from_slice(&[87.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 87.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_088() {
        let a = Tensor::from_slice(&[88.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 88.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_089() {
        let a = Tensor::from_slice(&[89.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 89.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_090() {
        let a = Tensor::from_slice(&[90.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 90.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_091() {
        let a = Tensor::from_slice(&[91.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 91.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_092() {
        let a = Tensor::from_slice(&[92.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 92.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_093() {
        let a = Tensor::from_slice(&[93.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 93.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_094() {
        let a = Tensor::from_slice(&[94.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 94.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_095() {
        let a = Tensor::from_slice(&[95.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 95.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_096() {
        let a = Tensor::from_slice(&[96.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 96.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_097() {
        let a = Tensor::from_slice(&[97.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 97.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_098() {
        let a = Tensor::from_slice(&[98.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 98.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_099() {
        let a = Tensor::from_slice(&[99.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 99.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_100() {
        let a = Tensor::from_slice(&[100.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 100.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_101() {
        let a = Tensor::from_slice(&[101.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 101.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_102() {
        let a = Tensor::from_slice(&[102.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 102.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_103() {
        let a = Tensor::from_slice(&[103.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 103.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_104() {
        let a = Tensor::from_slice(&[104.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 104.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_105() {
        let a = Tensor::from_slice(&[105.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 105.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_106() {
        let a = Tensor::from_slice(&[106.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 106.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_107() {
        let a = Tensor::from_slice(&[107.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 107.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_108() {
        let a = Tensor::from_slice(&[108.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 108.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_109() {
        let a = Tensor::from_slice(&[109.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 109.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_110() {
        let a = Tensor::from_slice(&[110.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 110.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_111() {
        let a = Tensor::from_slice(&[111.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 111.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_112() {
        let a = Tensor::from_slice(&[112.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 112.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_113() {
        let a = Tensor::from_slice(&[113.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 113.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_114() {
        let a = Tensor::from_slice(&[114.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 114.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_115() {
        let a = Tensor::from_slice(&[115.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 115.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_116() {
        let a = Tensor::from_slice(&[116.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 116.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_117() {
        let a = Tensor::from_slice(&[117.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 117.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_118() {
        let a = Tensor::from_slice(&[118.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 118.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_119() {
        let a = Tensor::from_slice(&[119.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 119.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_120() {
        let a = Tensor::from_slice(&[120.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 120.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_121() {
        let a = Tensor::from_slice(&[121.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 121.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_122() {
        let a = Tensor::from_slice(&[122.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 122.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_123() {
        let a = Tensor::from_slice(&[123.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 123.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_124() {
        let a = Tensor::from_slice(&[124.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 124.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_125() {
        let a = Tensor::from_slice(&[125.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 125.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_126() {
        let a = Tensor::from_slice(&[126.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 126.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_127() {
        let a = Tensor::from_slice(&[127.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 127.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_128() {
        let a = Tensor::from_slice(&[128.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 128.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_129() {
        let a = Tensor::from_slice(&[129.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 129.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_130() {
        let a = Tensor::from_slice(&[130.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 130.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_131() {
        let a = Tensor::from_slice(&[131.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 131.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_132() {
        let a = Tensor::from_slice(&[132.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 132.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_133() {
        let a = Tensor::from_slice(&[133.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 133.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_134() {
        let a = Tensor::from_slice(&[134.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 134.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_135() {
        let a = Tensor::from_slice(&[135.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 135.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_136() {
        let a = Tensor::from_slice(&[136.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 136.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_137() {
        let a = Tensor::from_slice(&[137.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 137.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_138() {
        let a = Tensor::from_slice(&[138.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 138.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_139() {
        let a = Tensor::from_slice(&[139.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 139.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_140() {
        let a = Tensor::from_slice(&[140.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 140.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_141() {
        let a = Tensor::from_slice(&[141.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 141.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_142() {
        let a = Tensor::from_slice(&[142.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 142.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_143() {
        let a = Tensor::from_slice(&[143.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 143.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_144() {
        let a = Tensor::from_slice(&[144.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 144.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_145() {
        let a = Tensor::from_slice(&[145.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 145.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_146() {
        let a = Tensor::from_slice(&[146.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 146.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_147() {
        let a = Tensor::from_slice(&[147.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 147.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_148() {
        let a = Tensor::from_slice(&[148.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 148.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_149() {
        let a = Tensor::from_slice(&[149.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 149.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_150() {
        let a = Tensor::from_slice(&[150.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 150.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_151() {
        let a = Tensor::from_slice(&[151.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 151.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_152() {
        let a = Tensor::from_slice(&[152.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 152.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_153() {
        let a = Tensor::from_slice(&[153.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 153.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_154() {
        let a = Tensor::from_slice(&[154.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 154.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_155() {
        let a = Tensor::from_slice(&[155.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 155.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_156() {
        let a = Tensor::from_slice(&[156.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 156.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_157() {
        let a = Tensor::from_slice(&[157.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 157.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_158() {
        let a = Tensor::from_slice(&[158.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 158.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_159() {
        let a = Tensor::from_slice(&[159.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 159.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_160() {
        let a = Tensor::from_slice(&[160.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 160.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_161() {
        let a = Tensor::from_slice(&[161.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 161.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_162() {
        let a = Tensor::from_slice(&[162.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 162.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_163() {
        let a = Tensor::from_slice(&[163.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 163.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_164() {
        let a = Tensor::from_slice(&[164.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 164.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_165() {
        let a = Tensor::from_slice(&[165.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 165.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_166() {
        let a = Tensor::from_slice(&[166.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 166.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_167() {
        let a = Tensor::from_slice(&[167.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 167.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_168() {
        let a = Tensor::from_slice(&[168.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 168.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_169() {
        let a = Tensor::from_slice(&[169.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 169.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_170() {
        let a = Tensor::from_slice(&[170.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 170.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_171() {
        let a = Tensor::from_slice(&[171.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 171.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_172() {
        let a = Tensor::from_slice(&[172.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 172.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_173() {
        let a = Tensor::from_slice(&[173.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 173.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_174() {
        let a = Tensor::from_slice(&[174.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 174.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_175() {
        let a = Tensor::from_slice(&[175.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 175.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_176() {
        let a = Tensor::from_slice(&[176.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 176.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_177() {
        let a = Tensor::from_slice(&[177.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 177.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_178() {
        let a = Tensor::from_slice(&[178.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 178.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_179() {
        let a = Tensor::from_slice(&[179.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 179.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_180() {
        let a = Tensor::from_slice(&[180.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 180.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_181() {
        let a = Tensor::from_slice(&[181.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 181.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_182() {
        let a = Tensor::from_slice(&[182.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 182.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_183() {
        let a = Tensor::from_slice(&[183.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 183.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_184() {
        let a = Tensor::from_slice(&[184.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 184.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_185() {
        let a = Tensor::from_slice(&[185.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 185.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_186() {
        let a = Tensor::from_slice(&[186.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 186.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_187() {
        let a = Tensor::from_slice(&[187.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 187.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_188() {
        let a = Tensor::from_slice(&[188.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 188.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_189() {
        let a = Tensor::from_slice(&[189.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 189.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_190() {
        let a = Tensor::from_slice(&[190.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 190.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_191() {
        let a = Tensor::from_slice(&[191.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 191.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_192() {
        let a = Tensor::from_slice(&[192.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 192.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_193() {
        let a = Tensor::from_slice(&[193.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 193.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_194() {
        let a = Tensor::from_slice(&[194.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 194.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_195() {
        let a = Tensor::from_slice(&[195.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 195.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_196() {
        let a = Tensor::from_slice(&[196.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 196.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_197() {
        let a = Tensor::from_slice(&[197.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 197.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_198() {
        let a = Tensor::from_slice(&[198.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 198.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_199() {
        let a = Tensor::from_slice(&[199.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 199.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_200() {
        let a = Tensor::from_slice(&[200.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 200.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_201() {
        let a = Tensor::from_slice(&[201.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 201.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_202() {
        let a = Tensor::from_slice(&[202.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 202.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_203() {
        let a = Tensor::from_slice(&[203.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 203.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_204() {
        let a = Tensor::from_slice(&[204.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 204.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_205() {
        let a = Tensor::from_slice(&[205.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 205.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_206() {
        let a = Tensor::from_slice(&[206.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 206.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_207() {
        let a = Tensor::from_slice(&[207.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 207.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_208() {
        let a = Tensor::from_slice(&[208.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 208.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_209() {
        let a = Tensor::from_slice(&[209.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 209.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_210() {
        let a = Tensor::from_slice(&[210.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 210.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_211() {
        let a = Tensor::from_slice(&[211.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 211.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_212() {
        let a = Tensor::from_slice(&[212.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 212.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_213() {
        let a = Tensor::from_slice(&[213.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 213.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_214() {
        let a = Tensor::from_slice(&[214.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 214.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_215() {
        let a = Tensor::from_slice(&[215.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 215.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_216() {
        let a = Tensor::from_slice(&[216.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 216.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_217() {
        let a = Tensor::from_slice(&[217.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 217.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_218() {
        let a = Tensor::from_slice(&[218.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 218.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_219() {
        let a = Tensor::from_slice(&[219.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 219.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_220() {
        let a = Tensor::from_slice(&[220.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 220.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_221() {
        let a = Tensor::from_slice(&[221.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 221.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_222() {
        let a = Tensor::from_slice(&[222.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 222.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_223() {
        let a = Tensor::from_slice(&[223.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 223.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_224() {
        let a = Tensor::from_slice(&[224.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 224.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_225() {
        let a = Tensor::from_slice(&[225.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 225.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_226() {
        let a = Tensor::from_slice(&[226.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 226.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_227() {
        let a = Tensor::from_slice(&[227.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 227.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_228() {
        let a = Tensor::from_slice(&[228.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 228.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_229() {
        let a = Tensor::from_slice(&[229.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 229.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_230() {
        let a = Tensor::from_slice(&[230.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 230.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_231() {
        let a = Tensor::from_slice(&[231.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 231.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_232() {
        let a = Tensor::from_slice(&[232.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 232.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_233() {
        let a = Tensor::from_slice(&[233.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 233.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_234() {
        let a = Tensor::from_slice(&[234.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 234.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_235() {
        let a = Tensor::from_slice(&[235.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 235.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_236() {
        let a = Tensor::from_slice(&[236.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 236.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_237() {
        let a = Tensor::from_slice(&[237.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 237.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_238() {
        let a = Tensor::from_slice(&[238.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 238.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_239() {
        let a = Tensor::from_slice(&[239.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 239.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_240() {
        let a = Tensor::from_slice(&[240.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 240.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_241() {
        let a = Tensor::from_slice(&[241.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 241.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_242() {
        let a = Tensor::from_slice(&[242.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 242.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_243() {
        let a = Tensor::from_slice(&[243.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 243.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_244() {
        let a = Tensor::from_slice(&[244.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 244.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_245() {
        let a = Tensor::from_slice(&[245.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 245.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_246() {
        let a = Tensor::from_slice(&[246.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 246.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_247() {
        let a = Tensor::from_slice(&[247.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 247.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_248() {
        let a = Tensor::from_slice(&[248.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 248.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_249() {
        let a = Tensor::from_slice(&[249.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 249.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_250() {
        let a = Tensor::from_slice(&[250.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 250.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_251() {
        let a = Tensor::from_slice(&[251.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 251.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_252() {
        let a = Tensor::from_slice(&[252.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 252.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_253() {
        let a = Tensor::from_slice(&[253.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 253.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_254() {
        let a = Tensor::from_slice(&[254.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 254.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_255() {
        let a = Tensor::from_slice(&[255.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 255.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_256() {
        let a = Tensor::from_slice(&[256.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 256.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_257() {
        let a = Tensor::from_slice(&[257.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 257.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_258() {
        let a = Tensor::from_slice(&[258.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 258.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_259() {
        let a = Tensor::from_slice(&[259.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 259.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_260() {
        let a = Tensor::from_slice(&[260.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 260.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_261() {
        let a = Tensor::from_slice(&[261.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 261.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_262() {
        let a = Tensor::from_slice(&[262.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 262.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_263() {
        let a = Tensor::from_slice(&[263.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 263.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_264() {
        let a = Tensor::from_slice(&[264.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 264.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_265() {
        let a = Tensor::from_slice(&[265.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 265.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_266() {
        let a = Tensor::from_slice(&[266.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 266.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_267() {
        let a = Tensor::from_slice(&[267.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 267.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_268() {
        let a = Tensor::from_slice(&[268.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 268.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_269() {
        let a = Tensor::from_slice(&[269.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 269.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_270() {
        let a = Tensor::from_slice(&[270.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 270.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_271() {
        let a = Tensor::from_slice(&[271.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 271.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_272() {
        let a = Tensor::from_slice(&[272.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 272.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_273() {
        let a = Tensor::from_slice(&[273.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 273.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_274() {
        let a = Tensor::from_slice(&[274.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 274.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_275() {
        let a = Tensor::from_slice(&[275.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 275.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_276() {
        let a = Tensor::from_slice(&[276.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 276.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_277() {
        let a = Tensor::from_slice(&[277.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 277.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_278() {
        let a = Tensor::from_slice(&[278.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 278.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_279() {
        let a = Tensor::from_slice(&[279.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 279.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_280() {
        let a = Tensor::from_slice(&[280.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 280.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_281() {
        let a = Tensor::from_slice(&[281.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 281.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_282() {
        let a = Tensor::from_slice(&[282.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 282.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_283() {
        let a = Tensor::from_slice(&[283.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 283.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_284() {
        let a = Tensor::from_slice(&[284.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 284.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_285() {
        let a = Tensor::from_slice(&[285.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 285.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_286() {
        let a = Tensor::from_slice(&[286.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 286.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_287() {
        let a = Tensor::from_slice(&[287.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 287.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_288() {
        let a = Tensor::from_slice(&[288.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 288.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_289() {
        let a = Tensor::from_slice(&[289.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 289.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_290() {
        let a = Tensor::from_slice(&[290.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 290.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_291() {
        let a = Tensor::from_slice(&[291.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 291.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_292() {
        let a = Tensor::from_slice(&[292.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 292.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_293() {
        let a = Tensor::from_slice(&[293.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 293.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_294() {
        let a = Tensor::from_slice(&[294.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 294.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_295() {
        let a = Tensor::from_slice(&[295.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 295.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_296() {
        let a = Tensor::from_slice(&[296.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 296.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_297() {
        let a = Tensor::from_slice(&[297.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 297.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_298() {
        let a = Tensor::from_slice(&[298.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 298.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }

    #[test]
    fn test_function_stress_case_299() {
        let a = Tensor::from_slice(&[299.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        let mut ctx = Context::new();
        let out = AddOp::forward(&mut ctx, &[&a, &b]);
        assert_eq!(out.get(0), 299.0 + 2.0);
        let grad = AddOp::backward(&ctx, &Tensor::ones(vec![1]));
        assert_eq!(grad[0].get(0), 1.0);
    }
}
