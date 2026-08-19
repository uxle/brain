//! # Reverse-Mode Backward Subsystem
//!
//! High-performance topological ordering, gradient dispatch, and memory management.

pub mod grad;
pub mod topo;

pub use grad::{backward_from, backward_with_grad, grad, grad_of, value_and_grad, zero_grad_from};
pub use topo::topological_sort;
