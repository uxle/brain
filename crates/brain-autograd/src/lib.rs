//! # Brain Autograd — High-Performance Reverse-Mode Automatic Differentiation
//!
//! Production-grade tape-free autograd engine with gradient checkpointing,
//! parallel reverse sweeps, and complete neural network operator gradient rules.
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_autograd::prelude::*;
//!
//! let mut x = Value::scalar(3.0);
//! x.set_requires_grad(true);
//! let y = x.mul(&x);
//! y.backward().unwrap();
//! assert_eq!(x.grad().unwrap().get(0), 6.0);
//! ```
//!
//! ## Higher-Order Functional Transforms
//!
//! ```rust
//! use brain_autograd::prelude::*;
//!
//! let x = Value::scalar(4.0);
//! let g = grad(|v| v.mul(v), &x).unwrap().unwrap();
//! assert_eq!(g.get(0), 8.0);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod backward;
pub mod checkpoint;
pub mod engine;
pub mod grad_fns;
pub mod graph_closure;
pub mod ops;
pub mod tape;
pub mod value;

// Re-exports
pub use backward::grad::{backward_from, backward_with_grad, zero_grad_from};
pub use grad_fns::GradFn;
pub use graph_closure::{grad, grad_and_hess, hessian, jacobian, jvp, value_and_grad, vjp};
pub use tape::{OpRecord, Tape};
pub use value::{
    is_grad_enabled, set_grad_enabled, values_close, with_enable_grad, with_no_grad, NoGradGuard,
    Value,
};

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_autograd::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_autograd::version_string;
/// assert_eq!(version_string(), "brain-autograd v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-autograd v{}", VERSION)
}

/// Common prelude imports.
///
/// ```rust
/// use brain_autograd::prelude::*;
/// let x = Value::scalar(2.0);
/// assert_eq!(x.data().get(0), 2.0);
/// ```
pub mod prelude {
    pub use crate::backward::grad::{backward_from, backward_with_grad, zero_grad_from};
    pub use crate::grad_fns::GradFn;
    pub use crate::graph_closure::{
        grad, grad_and_hess, hessian, jacobian, jvp, value_and_grad, vjp,
    };
    pub use crate::ops;
    pub use crate::tape::{start_recording, stop_recording, with_tape, Tape};
    pub use crate::value::{values_close, Value};
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
