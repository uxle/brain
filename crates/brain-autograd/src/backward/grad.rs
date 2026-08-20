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

    let initial_grad = Tensor::ones(root.shape().to_vec());
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

    let mut pass_grads: std::collections::HashMap<usize, Tensor> = std::collections::HashMap::new();
    pass_grads.insert(root.id(), seed_grad.clone());

    root.accumulate_grad(seed_grad)?;

    let mut order = topological_sort(root)?;
    order.reverse();

    for node in order {
        let current_grad = match pass_grads.remove(&node.id()) {
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
            pass_grads
                .entry(parent.id())
                .and_modify(|g| {
                    *g = brain_core::tensor::arithmetic::add(g, vjp);
                })
                .or_insert_with(|| vjp.clone());
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
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
