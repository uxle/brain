//! # Reverse-Mode Backward Driver and Gradient Accumulation
//!
//! Drives the reverse topological sweep, dispatches VJPs, and accumulates
//! gradients into graph leaves.

use crate::backward::topo::topological_sort;
use crate::value::Value;
use brain_core::{BrainError, BrainResult, Tensor};

/// Runs reverse-mode automatic differentiation from the specified root node.
pub fn backward_from(root: &Value) -> BrainResult<()> {
    if !root.requires_grad() && root.is_leaf() {
        return Ok(());
    }

    let initial_grad = if let Some(existing) = root.grad() {
        existing
    } else {
        Tensor::ones(root.shape().to_vec())
    };

    backward_with_grad(root, &initial_grad)
}

/// Runs reverse-mode automatic differentiation with a supplied seed gradient.
pub fn backward_with_grad(root: &Value, seed_grad: &Tensor) -> BrainResult<()> {
    if seed_grad.shape() != root.shape() {
        return Err(BrainError::shape_mismatch(
            format!("{:?}", root.shape()),
            format!("{:?}", seed_grad.shape()),
            "Root gradient shape does not match value shape",
        ));
    }

    root.accumulate_grad(seed_grad)?;

    let mut order = topological_sort(root)?;
    order.reverse();

    for node in order {
        let current_grad = match node.grad() {
            Some(g) => g,
            None => continue,
        };

        if !node.grad_fn().is_op() {
            continue;
        }

        let parent_vjps = node.grad_fn().apply_vjp(&current_grad)?;
        let parents = node.grad_fn().parents();

        for (parent, vjp) in parents.iter().zip(parent_vjps.iter()) {
            if parent.requires_grad() {
                parent.accumulate_grad(vjp)?;
            }
        }
    }

    Ok(())
}

/// Clears all accumulated gradients along the graph reachable from `root`.
pub fn zero_grad_from(root: &Value) -> BrainResult<()> {
    let order = topological_sort(root)?;
    for node in order {
        node.zero_grad();
    }
    root.zero_grad();
    Ok(())
}

/// Computes gradient of scalar function `f` with respect to `x`.
pub fn grad_of(x: &Value) -> Option<Tensor> {
    x.grad()
}

/// Evaluates `f(x)` and returns `(f(x), grad(f)(x))`.
pub fn value_and_grad<F>(f: F, x: &Value) -> BrainResult<(Value, Option<Tensor>)>
where
    F: FnOnce(&Value) -> Value,
{
    crate::graph_closure::value_and_grad(f, x)
}

/// Evaluates `f(x)` and returns `grad(f)(x)`.
pub fn grad<F>(f: F, x: &Value) -> BrainResult<Option<Tensor>>
where
    F: FnOnce(&Value) -> Value,
{
    crate::graph_closure::grad(f, x)
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
    fn test_backward_driver_stress_001() {
        let mut x = Value::scalar(1.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_002() {
        let mut x = Value::scalar(1.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_003() {
        let mut x = Value::scalar(1.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_004() {
        let mut x = Value::scalar(1.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_005() {
        let mut x = Value::scalar(1.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_006() {
        let mut x = Value::scalar(1.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_007() {
        let mut x = Value::scalar(1.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_008() {
        let mut x = Value::scalar(1.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_009() {
        let mut x = Value::scalar(1.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (1.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_010() {
        let mut x = Value::scalar(2.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_011() {
        let mut x = Value::scalar(2.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_012() {
        let mut x = Value::scalar(2.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_013() {
        let mut x = Value::scalar(2.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_014() {
        let mut x = Value::scalar(2.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_015() {
        let mut x = Value::scalar(2.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_016() {
        let mut x = Value::scalar(2.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_017() {
        let mut x = Value::scalar(2.35);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.35);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_018() {
        let mut x = Value::scalar(2.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_019() {
        let mut x = Value::scalar(2.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_020() {
        let mut x = Value::scalar(2.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_021() {
        let mut x = Value::scalar(2.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_022() {
        let mut x = Value::scalar(2.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_023() {
        let mut x = Value::scalar(2.6500000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.6500000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_024() {
        let mut x = Value::scalar(2.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_025() {
        let mut x = Value::scalar(2.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_026() {
        let mut x = Value::scalar(2.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_027() {
        let mut x = Value::scalar(2.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_028() {
        let mut x = Value::scalar(2.9000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.9000000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_029() {
        let mut x = Value::scalar(2.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (2.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_030() {
        let mut x = Value::scalar(3.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_031() {
        let mut x = Value::scalar(3.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_032() {
        let mut x = Value::scalar(3.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_033() {
        let mut x = Value::scalar(3.1500000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.1500000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_034() {
        let mut x = Value::scalar(3.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_035() {
        let mut x = Value::scalar(3.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_036() {
        let mut x = Value::scalar(3.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_037() {
        let mut x = Value::scalar(3.35);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.35);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_038() {
        let mut x = Value::scalar(3.4000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.4000000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_039() {
        let mut x = Value::scalar(3.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_040() {
        let mut x = Value::scalar(3.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_041() {
        let mut x = Value::scalar(3.5500000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.5500000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_042() {
        let mut x = Value::scalar(3.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_043() {
        let mut x = Value::scalar(3.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_044() {
        let mut x = Value::scalar(3.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_045() {
        let mut x = Value::scalar(3.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_046() {
        let mut x = Value::scalar(3.8000000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.8000000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_047() {
        let mut x = Value::scalar(3.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_048() {
        let mut x = Value::scalar(3.9000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.9000000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_049() {
        let mut x = Value::scalar(3.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (3.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_050() {
        let mut x = Value::scalar(4.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_051() {
        let mut x = Value::scalar(4.050000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.050000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_052() {
        let mut x = Value::scalar(4.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_053() {
        let mut x = Value::scalar(4.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_054() {
        let mut x = Value::scalar(4.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_055() {
        let mut x = Value::scalar(4.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_056() {
        let mut x = Value::scalar(4.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_057() {
        let mut x = Value::scalar(4.35);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.35);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_058() {
        let mut x = Value::scalar(4.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_059() {
        let mut x = Value::scalar(4.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_060() {
        let mut x = Value::scalar(4.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_061() {
        let mut x = Value::scalar(4.550000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.550000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_062() {
        let mut x = Value::scalar(4.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_063() {
        let mut x = Value::scalar(4.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_064() {
        let mut x = Value::scalar(4.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_065() {
        let mut x = Value::scalar(4.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_066() {
        let mut x = Value::scalar(4.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_067() {
        let mut x = Value::scalar(4.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_068() {
        let mut x = Value::scalar(4.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_069() {
        let mut x = Value::scalar(4.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (4.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_070() {
        let mut x = Value::scalar(5.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_071() {
        let mut x = Value::scalar(5.050000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.050000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_072() {
        let mut x = Value::scalar(5.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_073() {
        let mut x = Value::scalar(5.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_074() {
        let mut x = Value::scalar(5.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_075() {
        let mut x = Value::scalar(5.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_076() {
        let mut x = Value::scalar(5.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_077() {
        let mut x = Value::scalar(5.35);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.35);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_078() {
        let mut x = Value::scalar(5.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_079() {
        let mut x = Value::scalar(5.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_080() {
        let mut x = Value::scalar(5.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_081() {
        let mut x = Value::scalar(5.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_082() {
        let mut x = Value::scalar(5.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.6000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_083() {
        let mut x = Value::scalar(5.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_084() {
        let mut x = Value::scalar(5.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_085() {
        let mut x = Value::scalar(5.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_086() {
        let mut x = Value::scalar(5.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_087() {
        let mut x = Value::scalar(5.8500000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.8500000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_088() {
        let mut x = Value::scalar(5.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_089() {
        let mut x = Value::scalar(5.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (5.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_090() {
        let mut x = Value::scalar(6.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_091() {
        let mut x = Value::scalar(6.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_092() {
        let mut x = Value::scalar(6.1000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.1000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_093() {
        let mut x = Value::scalar(6.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_094() {
        let mut x = Value::scalar(6.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_095() {
        let mut x = Value::scalar(6.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_096() {
        let mut x = Value::scalar(6.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_097() {
        let mut x = Value::scalar(6.3500000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.3500000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_098() {
        let mut x = Value::scalar(6.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_099() {
        let mut x = Value::scalar(6.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_100() {
        let mut x = Value::scalar(6.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_101() {
        let mut x = Value::scalar(6.550000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.550000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_102() {
        let mut x = Value::scalar(6.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.6000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_103() {
        let mut x = Value::scalar(6.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_104() {
        let mut x = Value::scalar(6.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_105() {
        let mut x = Value::scalar(6.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_106() {
        let mut x = Value::scalar(6.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_107() {
        let mut x = Value::scalar(6.8500000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.8500000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_108() {
        let mut x = Value::scalar(6.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_109() {
        let mut x = Value::scalar(6.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (6.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_110() {
        let mut x = Value::scalar(7.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_111() {
        let mut x = Value::scalar(7.050000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.050000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_112() {
        let mut x = Value::scalar(7.1000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.1000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_113() {
        let mut x = Value::scalar(7.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_114() {
        let mut x = Value::scalar(7.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_115() {
        let mut x = Value::scalar(7.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_116() {
        let mut x = Value::scalar(7.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_117() {
        let mut x = Value::scalar(7.3500000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.3500000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_118() {
        let mut x = Value::scalar(7.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_119() {
        let mut x = Value::scalar(7.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_120() {
        let mut x = Value::scalar(7.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_121() {
        let mut x = Value::scalar(7.550000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.550000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_122() {
        let mut x = Value::scalar(7.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.6000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_123() {
        let mut x = Value::scalar(7.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_124() {
        let mut x = Value::scalar(7.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_125() {
        let mut x = Value::scalar(7.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_126() {
        let mut x = Value::scalar(7.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_127() {
        let mut x = Value::scalar(7.8500000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.8500000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_128() {
        let mut x = Value::scalar(7.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_129() {
        let mut x = Value::scalar(7.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (7.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_130() {
        let mut x = Value::scalar(8.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_131() {
        let mut x = Value::scalar(8.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_132() {
        let mut x = Value::scalar(8.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_133() {
        let mut x = Value::scalar(8.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_134() {
        let mut x = Value::scalar(8.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_135() {
        let mut x = Value::scalar(8.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_136() {
        let mut x = Value::scalar(8.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_137() {
        let mut x = Value::scalar(8.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_138() {
        let mut x = Value::scalar(8.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_139() {
        let mut x = Value::scalar(8.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_140() {
        let mut x = Value::scalar(8.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_141() {
        let mut x = Value::scalar(8.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_142() {
        let mut x = Value::scalar(8.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_143() {
        let mut x = Value::scalar(8.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_144() {
        let mut x = Value::scalar(8.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_145() {
        let mut x = Value::scalar(8.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_146() {
        let mut x = Value::scalar(8.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_147() {
        let mut x = Value::scalar(8.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_148() {
        let mut x = Value::scalar(8.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_149() {
        let mut x = Value::scalar(8.95);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (8.95);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_150() {
        let mut x = Value::scalar(9.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_151() {
        let mut x = Value::scalar(9.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_152() {
        let mut x = Value::scalar(9.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_153() {
        let mut x = Value::scalar(9.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_154() {
        let mut x = Value::scalar(9.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_155() {
        let mut x = Value::scalar(9.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_156() {
        let mut x = Value::scalar(9.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_157() {
        let mut x = Value::scalar(9.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_158() {
        let mut x = Value::scalar(9.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_159() {
        let mut x = Value::scalar(9.45);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.45);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_160() {
        let mut x = Value::scalar(9.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_161() {
        let mut x = Value::scalar(9.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_162() {
        let mut x = Value::scalar(9.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_163() {
        let mut x = Value::scalar(9.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_164() {
        let mut x = Value::scalar(9.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_165() {
        let mut x = Value::scalar(9.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_166() {
        let mut x = Value::scalar(9.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_167() {
        let mut x = Value::scalar(9.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_168() {
        let mut x = Value::scalar(9.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_169() {
        let mut x = Value::scalar(9.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (9.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_170() {
        let mut x = Value::scalar(10.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_171() {
        let mut x = Value::scalar(10.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_172() {
        let mut x = Value::scalar(10.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_173() {
        let mut x = Value::scalar(10.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_174() {
        let mut x = Value::scalar(10.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_175() {
        let mut x = Value::scalar(10.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_176() {
        let mut x = Value::scalar(10.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_177() {
        let mut x = Value::scalar(10.35);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.35);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_178() {
        let mut x = Value::scalar(10.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_179() {
        let mut x = Value::scalar(10.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_180() {
        let mut x = Value::scalar(10.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_181() {
        let mut x = Value::scalar(10.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_182() {
        let mut x = Value::scalar(10.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_183() {
        let mut x = Value::scalar(10.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_184() {
        let mut x = Value::scalar(10.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_185() {
        let mut x = Value::scalar(10.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_186() {
        let mut x = Value::scalar(10.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_187() {
        let mut x = Value::scalar(10.85);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.85);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_188() {
        let mut x = Value::scalar(10.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_189() {
        let mut x = Value::scalar(10.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (10.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_190() {
        let mut x = Value::scalar(11.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_191() {
        let mut x = Value::scalar(11.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_192() {
        let mut x = Value::scalar(11.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_193() {
        let mut x = Value::scalar(11.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_194() {
        let mut x = Value::scalar(11.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_195() {
        let mut x = Value::scalar(11.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_196() {
        let mut x = Value::scalar(11.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_197() {
        let mut x = Value::scalar(11.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_198() {
        let mut x = Value::scalar(11.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_199() {
        let mut x = Value::scalar(11.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_200() {
        let mut x = Value::scalar(11.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_201() {
        let mut x = Value::scalar(11.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_202() {
        let mut x = Value::scalar(11.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_203() {
        let mut x = Value::scalar(11.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_204() {
        let mut x = Value::scalar(11.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_205() {
        let mut x = Value::scalar(11.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_206() {
        let mut x = Value::scalar(11.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_207() {
        let mut x = Value::scalar(11.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_208() {
        let mut x = Value::scalar(11.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_209() {
        let mut x = Value::scalar(11.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (11.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_210() {
        let mut x = Value::scalar(12.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_211() {
        let mut x = Value::scalar(12.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_212() {
        let mut x = Value::scalar(12.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_213() {
        let mut x = Value::scalar(12.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_214() {
        let mut x = Value::scalar(12.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_215() {
        let mut x = Value::scalar(12.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_216() {
        let mut x = Value::scalar(12.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_217() {
        let mut x = Value::scalar(12.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_218() {
        let mut x = Value::scalar(12.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_219() {
        let mut x = Value::scalar(12.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_220() {
        let mut x = Value::scalar(12.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_221() {
        let mut x = Value::scalar(12.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_222() {
        let mut x = Value::scalar(12.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_223() {
        let mut x = Value::scalar(12.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_224() {
        let mut x = Value::scalar(12.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_225() {
        let mut x = Value::scalar(12.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_226() {
        let mut x = Value::scalar(12.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_227() {
        let mut x = Value::scalar(12.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_228() {
        let mut x = Value::scalar(12.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_229() {
        let mut x = Value::scalar(12.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (12.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_230() {
        let mut x = Value::scalar(13.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_231() {
        let mut x = Value::scalar(13.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_232() {
        let mut x = Value::scalar(13.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_233() {
        let mut x = Value::scalar(13.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_234() {
        let mut x = Value::scalar(13.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_235() {
        let mut x = Value::scalar(13.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_236() {
        let mut x = Value::scalar(13.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_237() {
        let mut x = Value::scalar(13.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_238() {
        let mut x = Value::scalar(13.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_239() {
        let mut x = Value::scalar(13.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_240() {
        let mut x = Value::scalar(13.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_241() {
        let mut x = Value::scalar(13.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_242() {
        let mut x = Value::scalar(13.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_243() {
        let mut x = Value::scalar(13.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_244() {
        let mut x = Value::scalar(13.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_245() {
        let mut x = Value::scalar(13.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_246() {
        let mut x = Value::scalar(13.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_247() {
        let mut x = Value::scalar(13.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_248() {
        let mut x = Value::scalar(13.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_249() {
        let mut x = Value::scalar(13.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (13.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_250() {
        let mut x = Value::scalar(14.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_251() {
        let mut x = Value::scalar(14.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_252() {
        let mut x = Value::scalar(14.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_253() {
        let mut x = Value::scalar(14.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_254() {
        let mut x = Value::scalar(14.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_255() {
        let mut x = Value::scalar(14.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_256() {
        let mut x = Value::scalar(14.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_257() {
        let mut x = Value::scalar(14.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_258() {
        let mut x = Value::scalar(14.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_259() {
        let mut x = Value::scalar(14.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_260() {
        let mut x = Value::scalar(14.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_261() {
        let mut x = Value::scalar(14.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_262() {
        let mut x = Value::scalar(14.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_263() {
        let mut x = Value::scalar(14.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_264() {
        let mut x = Value::scalar(14.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_265() {
        let mut x = Value::scalar(14.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_266() {
        let mut x = Value::scalar(14.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_267() {
        let mut x = Value::scalar(14.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_268() {
        let mut x = Value::scalar(14.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_269() {
        let mut x = Value::scalar(14.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (14.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_270() {
        let mut x = Value::scalar(15.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_271() {
        let mut x = Value::scalar(15.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_272() {
        let mut x = Value::scalar(15.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_273() {
        let mut x = Value::scalar(15.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_274() {
        let mut x = Value::scalar(15.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_275() {
        let mut x = Value::scalar(15.25);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.25);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_276() {
        let mut x = Value::scalar(15.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_277() {
        let mut x = Value::scalar(15.350000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.350000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_278() {
        let mut x = Value::scalar(15.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_279() {
        let mut x = Value::scalar(15.450000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.450000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_280() {
        let mut x = Value::scalar(15.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_281() {
        let mut x = Value::scalar(15.55);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.55);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_282() {
        let mut x = Value::scalar(15.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_283() {
        let mut x = Value::scalar(15.65);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.65);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_284() {
        let mut x = Value::scalar(15.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_285() {
        let mut x = Value::scalar(15.75);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.75);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_286() {
        let mut x = Value::scalar(15.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_287() {
        let mut x = Value::scalar(15.850000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.850000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_288() {
        let mut x = Value::scalar(15.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_289() {
        let mut x = Value::scalar(15.950000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (15.950000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_290() {
        let mut x = Value::scalar(16.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (16.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_291() {
        let mut x = Value::scalar(16.05);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (16.05);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_292() {
        let mut x = Value::scalar(16.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (16.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_293() {
        let mut x = Value::scalar(16.15);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (16.15);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_backward_driver_stress_294() {
        let mut x = Value::scalar(16.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        backward_from(&y).unwrap();
        let g = x.grad().unwrap();
        let exp = 2.0 * (16.200000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
}
