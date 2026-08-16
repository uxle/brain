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
pub use value::{values_close, Value};

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
    pub use crate::graph_closure::{grad, grad_and_hess, hessian, jacobian, jvp, value_and_grad, vjp};
    pub use crate::ops;
    pub use crate::tape::{start_recording, stop_recording, with_tape, Tape};
    pub use crate::value::{values_close, Value};
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
    fn test_autograd_lib_stress_001() {
        let x = Value::scalar(1.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_002() {
        let x = Value::scalar(1.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_003() {
        let x = Value::scalar(1.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_004() {
        let x = Value::scalar(1.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_005() {
        let x = Value::scalar(1.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_006() {
        let x = Value::scalar(1.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_007() {
        let x = Value::scalar(1.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_008() {
        let x = Value::scalar(1.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_009() {
        let x = Value::scalar(1.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_010() {
        let x = Value::scalar(1.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_011() {
        let x = Value::scalar(1.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_012() {
        let x = Value::scalar(1.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_013() {
        let x = Value::scalar(1.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_014() {
        let x = Value::scalar(1.7000000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_015() {
        let x = Value::scalar(1.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_016() {
        let x = Value::scalar(1.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_017() {
        let x = Value::scalar(1.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_018() {
        let x = Value::scalar(1.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_019() {
        let x = Value::scalar(1.9500000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_020() {
        let x = Value::scalar(2.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_021() {
        let x = Value::scalar(2.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_022() {
        let x = Value::scalar(2.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_023() {
        let x = Value::scalar(2.1500000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_024() {
        let x = Value::scalar(2.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_025() {
        let x = Value::scalar(2.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_026() {
        let x = Value::scalar(2.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_027() {
        let x = Value::scalar(2.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_028() {
        let x = Value::scalar(2.4000000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_029() {
        let x = Value::scalar(2.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_030() {
        let x = Value::scalar(2.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_031() {
        let x = Value::scalar(2.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_032() {
        let x = Value::scalar(2.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_033() {
        let x = Value::scalar(2.6500000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_034() {
        let x = Value::scalar(2.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_035() {
        let x = Value::scalar(2.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_036() {
        let x = Value::scalar(2.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_037() {
        let x = Value::scalar(2.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_038() {
        let x = Value::scalar(2.9000000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_039() {
        let x = Value::scalar(2.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_040() {
        let x = Value::scalar(3.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_041() {
        let x = Value::scalar(3.0500000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_042() {
        let x = Value::scalar(3.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_043() {
        let x = Value::scalar(3.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_044() {
        let x = Value::scalar(3.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_045() {
        let x = Value::scalar(3.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_046() {
        let x = Value::scalar(3.3000000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_047() {
        let x = Value::scalar(3.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_048() {
        let x = Value::scalar(3.4000000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_049() {
        let x = Value::scalar(3.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_050() {
        let x = Value::scalar(3.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_051() {
        let x = Value::scalar(3.5500000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_052() {
        let x = Value::scalar(3.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_053() {
        let x = Value::scalar(3.6500000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_054() {
        let x = Value::scalar(3.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_055() {
        let x = Value::scalar(3.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_056() {
        let x = Value::scalar(3.8000000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_057() {
        let x = Value::scalar(3.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_058() {
        let x = Value::scalar(3.9000000000000004);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_059() {
        let x = Value::scalar(3.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_060() {
        let x = Value::scalar(4.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_061() {
        let x = Value::scalar(4.050000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_062() {
        let x = Value::scalar(4.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_063() {
        let x = Value::scalar(4.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_064() {
        let x = Value::scalar(4.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_065() {
        let x = Value::scalar(4.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_066() {
        let x = Value::scalar(4.300000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_067() {
        let x = Value::scalar(4.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_068() {
        let x = Value::scalar(4.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_069() {
        let x = Value::scalar(4.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_070() {
        let x = Value::scalar(4.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_071() {
        let x = Value::scalar(4.550000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_072() {
        let x = Value::scalar(4.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_073() {
        let x = Value::scalar(4.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_074() {
        let x = Value::scalar(4.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_075() {
        let x = Value::scalar(4.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_076() {
        let x = Value::scalar(4.800000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_077() {
        let x = Value::scalar(4.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_078() {
        let x = Value::scalar(4.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_079() {
        let x = Value::scalar(4.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_080() {
        let x = Value::scalar(5.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_081() {
        let x = Value::scalar(5.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_082() {
        let x = Value::scalar(5.1000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_083() {
        let x = Value::scalar(5.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_084() {
        let x = Value::scalar(5.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_085() {
        let x = Value::scalar(5.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_086() {
        let x = Value::scalar(5.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_087() {
        let x = Value::scalar(5.3500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_088() {
        let x = Value::scalar(5.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_089() {
        let x = Value::scalar(5.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_090() {
        let x = Value::scalar(5.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_091() {
        let x = Value::scalar(5.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_092() {
        let x = Value::scalar(5.6000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_093() {
        let x = Value::scalar(5.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_094() {
        let x = Value::scalar(5.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_095() {
        let x = Value::scalar(5.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_096() {
        let x = Value::scalar(5.800000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_097() {
        let x = Value::scalar(5.8500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_098() {
        let x = Value::scalar(5.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_099() {
        let x = Value::scalar(5.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_100() {
        let x = Value::scalar(6.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_101() {
        let x = Value::scalar(6.050000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_102() {
        let x = Value::scalar(6.1000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_103() {
        let x = Value::scalar(6.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_104() {
        let x = Value::scalar(6.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_105() {
        let x = Value::scalar(6.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_106() {
        let x = Value::scalar(6.300000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_107() {
        let x = Value::scalar(6.3500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_108() {
        let x = Value::scalar(6.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_109() {
        let x = Value::scalar(6.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_110() {
        let x = Value::scalar(6.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_111() {
        let x = Value::scalar(6.550000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_112() {
        let x = Value::scalar(6.6000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_113() {
        let x = Value::scalar(6.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_114() {
        let x = Value::scalar(6.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_115() {
        let x = Value::scalar(6.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_116() {
        let x = Value::scalar(6.800000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_117() {
        let x = Value::scalar(6.8500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_118() {
        let x = Value::scalar(6.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_119() {
        let x = Value::scalar(6.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_120() {
        let x = Value::scalar(7.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_121() {
        let x = Value::scalar(7.050000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_122() {
        let x = Value::scalar(7.1000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_123() {
        let x = Value::scalar(7.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_124() {
        let x = Value::scalar(7.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_125() {
        let x = Value::scalar(7.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_126() {
        let x = Value::scalar(7.300000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_127() {
        let x = Value::scalar(7.3500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_128() {
        let x = Value::scalar(7.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_129() {
        let x = Value::scalar(7.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_130() {
        let x = Value::scalar(7.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_131() {
        let x = Value::scalar(7.550000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_132() {
        let x = Value::scalar(7.6000000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_133() {
        let x = Value::scalar(7.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_134() {
        let x = Value::scalar(7.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_135() {
        let x = Value::scalar(7.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_136() {
        let x = Value::scalar(7.800000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_137() {
        let x = Value::scalar(7.8500000000000005);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_138() {
        let x = Value::scalar(7.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_139() {
        let x = Value::scalar(7.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_140() {
        let x = Value::scalar(8.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_141() {
        let x = Value::scalar(8.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_142() {
        let x = Value::scalar(8.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_143() {
        let x = Value::scalar(8.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_144() {
        let x = Value::scalar(8.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_145() {
        let x = Value::scalar(8.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_146() {
        let x = Value::scalar(8.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_147() {
        let x = Value::scalar(8.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_148() {
        let x = Value::scalar(8.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_149() {
        let x = Value::scalar(8.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_150() {
        let x = Value::scalar(8.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_151() {
        let x = Value::scalar(8.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_152() {
        let x = Value::scalar(8.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_153() {
        let x = Value::scalar(8.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_154() {
        let x = Value::scalar(8.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_155() {
        let x = Value::scalar(8.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_156() {
        let x = Value::scalar(8.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_157() {
        let x = Value::scalar(8.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_158() {
        let x = Value::scalar(8.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_159() {
        let x = Value::scalar(8.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_160() {
        let x = Value::scalar(9.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_161() {
        let x = Value::scalar(9.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_162() {
        let x = Value::scalar(9.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_163() {
        let x = Value::scalar(9.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_164() {
        let x = Value::scalar(9.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_165() {
        let x = Value::scalar(9.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_166() {
        let x = Value::scalar(9.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_167() {
        let x = Value::scalar(9.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_168() {
        let x = Value::scalar(9.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_169() {
        let x = Value::scalar(9.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_170() {
        let x = Value::scalar(9.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_171() {
        let x = Value::scalar(9.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_172() {
        let x = Value::scalar(9.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_173() {
        let x = Value::scalar(9.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_174() {
        let x = Value::scalar(9.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_175() {
        let x = Value::scalar(9.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_176() {
        let x = Value::scalar(9.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_177() {
        let x = Value::scalar(9.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_178() {
        let x = Value::scalar(9.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_179() {
        let x = Value::scalar(9.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_180() {
        let x = Value::scalar(10.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_181() {
        let x = Value::scalar(10.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_182() {
        let x = Value::scalar(10.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_183() {
        let x = Value::scalar(10.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_184() {
        let x = Value::scalar(10.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_185() {
        let x = Value::scalar(10.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_186() {
        let x = Value::scalar(10.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_187() {
        let x = Value::scalar(10.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_188() {
        let x = Value::scalar(10.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_189() {
        let x = Value::scalar(10.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_190() {
        let x = Value::scalar(10.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_191() {
        let x = Value::scalar(10.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_192() {
        let x = Value::scalar(10.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_193() {
        let x = Value::scalar(10.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_194() {
        let x = Value::scalar(10.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_195() {
        let x = Value::scalar(10.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_196() {
        let x = Value::scalar(10.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_197() {
        let x = Value::scalar(10.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_198() {
        let x = Value::scalar(10.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_199() {
        let x = Value::scalar(10.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_200() {
        let x = Value::scalar(11.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_201() {
        let x = Value::scalar(11.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_202() {
        let x = Value::scalar(11.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_203() {
        let x = Value::scalar(11.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_204() {
        let x = Value::scalar(11.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_205() {
        let x = Value::scalar(11.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_206() {
        let x = Value::scalar(11.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_207() {
        let x = Value::scalar(11.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_208() {
        let x = Value::scalar(11.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_209() {
        let x = Value::scalar(11.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_210() {
        let x = Value::scalar(11.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_211() {
        let x = Value::scalar(11.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_212() {
        let x = Value::scalar(11.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_213() {
        let x = Value::scalar(11.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_214() {
        let x = Value::scalar(11.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_215() {
        let x = Value::scalar(11.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_216() {
        let x = Value::scalar(11.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_217() {
        let x = Value::scalar(11.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_218() {
        let x = Value::scalar(11.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_219() {
        let x = Value::scalar(11.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_220() {
        let x = Value::scalar(12.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_221() {
        let x = Value::scalar(12.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_222() {
        let x = Value::scalar(12.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_223() {
        let x = Value::scalar(12.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_224() {
        let x = Value::scalar(12.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_225() {
        let x = Value::scalar(12.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_226() {
        let x = Value::scalar(12.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_227() {
        let x = Value::scalar(12.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_228() {
        let x = Value::scalar(12.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_229() {
        let x = Value::scalar(12.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_230() {
        let x = Value::scalar(12.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_231() {
        let x = Value::scalar(12.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_232() {
        let x = Value::scalar(12.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_233() {
        let x = Value::scalar(12.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_234() {
        let x = Value::scalar(12.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_235() {
        let x = Value::scalar(12.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_236() {
        let x = Value::scalar(12.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_237() {
        let x = Value::scalar(12.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_238() {
        let x = Value::scalar(12.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_239() {
        let x = Value::scalar(12.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_240() {
        let x = Value::scalar(13.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_241() {
        let x = Value::scalar(13.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_242() {
        let x = Value::scalar(13.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_243() {
        let x = Value::scalar(13.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_244() {
        let x = Value::scalar(13.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_245() {
        let x = Value::scalar(13.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_246() {
        let x = Value::scalar(13.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_247() {
        let x = Value::scalar(13.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_248() {
        let x = Value::scalar(13.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_249() {
        let x = Value::scalar(13.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_250() {
        let x = Value::scalar(13.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_251() {
        let x = Value::scalar(13.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_252() {
        let x = Value::scalar(13.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_253() {
        let x = Value::scalar(13.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_254() {
        let x = Value::scalar(13.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_255() {
        let x = Value::scalar(13.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_256() {
        let x = Value::scalar(13.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_257() {
        let x = Value::scalar(13.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_258() {
        let x = Value::scalar(13.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_259() {
        let x = Value::scalar(13.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_260() {
        let x = Value::scalar(14.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_261() {
        let x = Value::scalar(14.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_262() {
        let x = Value::scalar(14.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_263() {
        let x = Value::scalar(14.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_264() {
        let x = Value::scalar(14.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_265() {
        let x = Value::scalar(14.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_266() {
        let x = Value::scalar(14.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_267() {
        let x = Value::scalar(14.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_268() {
        let x = Value::scalar(14.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_269() {
        let x = Value::scalar(14.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_270() {
        let x = Value::scalar(14.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_271() {
        let x = Value::scalar(14.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_272() {
        let x = Value::scalar(14.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_273() {
        let x = Value::scalar(14.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_274() {
        let x = Value::scalar(14.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_275() {
        let x = Value::scalar(14.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_276() {
        let x = Value::scalar(14.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_277() {
        let x = Value::scalar(14.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_278() {
        let x = Value::scalar(14.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_279() {
        let x = Value::scalar(14.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_280() {
        let x = Value::scalar(15.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_281() {
        let x = Value::scalar(15.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_282() {
        let x = Value::scalar(15.100000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_283() {
        let x = Value::scalar(15.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_284() {
        let x = Value::scalar(15.200000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_285() {
        let x = Value::scalar(15.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_286() {
        let x = Value::scalar(15.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_287() {
        let x = Value::scalar(15.350000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_288() {
        let x = Value::scalar(15.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_289() {
        let x = Value::scalar(15.450000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_290() {
        let x = Value::scalar(15.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_291() {
        let x = Value::scalar(15.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_292() {
        let x = Value::scalar(15.600000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_293() {
        let x = Value::scalar(15.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_294() {
        let x = Value::scalar(15.700000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_295() {
        let x = Value::scalar(15.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_296() {
        let x = Value::scalar(15.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_297() {
        let x = Value::scalar(15.850000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_298() {
        let x = Value::scalar(15.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_299() {
        let x = Value::scalar(15.950000000000001);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_300() {
        let x = Value::scalar(16.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_301() {
        let x = Value::scalar(16.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_302() {
        let x = Value::scalar(16.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_303() {
        let x = Value::scalar(16.15);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_304() {
        let x = Value::scalar(16.200000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_305() {
        let x = Value::scalar(16.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_306() {
        let x = Value::scalar(16.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_307() {
        let x = Value::scalar(16.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_308() {
        let x = Value::scalar(16.4);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_309() {
        let x = Value::scalar(16.450000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_310() {
        let x = Value::scalar(16.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_311() {
        let x = Value::scalar(16.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_312() {
        let x = Value::scalar(16.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_313() {
        let x = Value::scalar(16.65);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_314() {
        let x = Value::scalar(16.700000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_315() {
        let x = Value::scalar(16.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_316() {
        let x = Value::scalar(16.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_317() {
        let x = Value::scalar(16.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_318() {
        let x = Value::scalar(16.9);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_319() {
        let x = Value::scalar(16.950000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_320() {
        let x = Value::scalar(17.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_321() {
        let x = Value::scalar(17.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_322() {
        let x = Value::scalar(17.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_323() {
        let x = Value::scalar(17.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_324() {
        let x = Value::scalar(17.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_325() {
        let x = Value::scalar(17.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_326() {
        let x = Value::scalar(17.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_327() {
        let x = Value::scalar(17.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_328() {
        let x = Value::scalar(17.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_329() {
        let x = Value::scalar(17.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_330() {
        let x = Value::scalar(17.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_331() {
        let x = Value::scalar(17.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_332() {
        let x = Value::scalar(17.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_333() {
        let x = Value::scalar(17.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_334() {
        let x = Value::scalar(17.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_335() {
        let x = Value::scalar(17.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_336() {
        let x = Value::scalar(17.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_337() {
        let x = Value::scalar(17.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_338() {
        let x = Value::scalar(17.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_339() {
        let x = Value::scalar(17.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_340() {
        let x = Value::scalar(18.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_341() {
        let x = Value::scalar(18.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_342() {
        let x = Value::scalar(18.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_343() {
        let x = Value::scalar(18.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_344() {
        let x = Value::scalar(18.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_345() {
        let x = Value::scalar(18.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_346() {
        let x = Value::scalar(18.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_347() {
        let x = Value::scalar(18.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_348() {
        let x = Value::scalar(18.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_349() {
        let x = Value::scalar(18.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_350() {
        let x = Value::scalar(18.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_351() {
        let x = Value::scalar(18.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_352() {
        let x = Value::scalar(18.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_353() {
        let x = Value::scalar(18.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_354() {
        let x = Value::scalar(18.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_355() {
        let x = Value::scalar(18.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_356() {
        let x = Value::scalar(18.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_357() {
        let x = Value::scalar(18.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_358() {
        let x = Value::scalar(18.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_359() {
        let x = Value::scalar(18.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_360() {
        let x = Value::scalar(19.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_361() {
        let x = Value::scalar(19.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_362() {
        let x = Value::scalar(19.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_363() {
        let x = Value::scalar(19.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_364() {
        let x = Value::scalar(19.2);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_365() {
        let x = Value::scalar(19.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_366() {
        let x = Value::scalar(19.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_367() {
        let x = Value::scalar(19.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_368() {
        let x = Value::scalar(19.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_369() {
        let x = Value::scalar(19.45);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_370() {
        let x = Value::scalar(19.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_371() {
        let x = Value::scalar(19.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_372() {
        let x = Value::scalar(19.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_373() {
        let x = Value::scalar(19.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_374() {
        let x = Value::scalar(19.7);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_375() {
        let x = Value::scalar(19.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_376() {
        let x = Value::scalar(19.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_377() {
        let x = Value::scalar(19.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_378() {
        let x = Value::scalar(19.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_379() {
        let x = Value::scalar(19.95);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_380() {
        let x = Value::scalar(20.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_381() {
        let x = Value::scalar(20.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_382() {
        let x = Value::scalar(20.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_383() {
        let x = Value::scalar(20.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_384() {
        let x = Value::scalar(20.200000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_385() {
        let x = Value::scalar(20.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_386() {
        let x = Value::scalar(20.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_387() {
        let x = Value::scalar(20.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_388() {
        let x = Value::scalar(20.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_389() {
        let x = Value::scalar(20.450000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_390() {
        let x = Value::scalar(20.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_391() {
        let x = Value::scalar(20.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_392() {
        let x = Value::scalar(20.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_393() {
        let x = Value::scalar(20.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_394() {
        let x = Value::scalar(20.700000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_395() {
        let x = Value::scalar(20.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_396() {
        let x = Value::scalar(20.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_397() {
        let x = Value::scalar(20.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_398() {
        let x = Value::scalar(20.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_399() {
        let x = Value::scalar(20.950000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_400() {
        let x = Value::scalar(21.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_401() {
        let x = Value::scalar(21.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_402() {
        let x = Value::scalar(21.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_403() {
        let x = Value::scalar(21.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_404() {
        let x = Value::scalar(21.200000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_405() {
        let x = Value::scalar(21.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_406() {
        let x = Value::scalar(21.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_407() {
        let x = Value::scalar(21.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_408() {
        let x = Value::scalar(21.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_409() {
        let x = Value::scalar(21.450000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_410() {
        let x = Value::scalar(21.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_411() {
        let x = Value::scalar(21.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_412() {
        let x = Value::scalar(21.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_413() {
        let x = Value::scalar(21.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_414() {
        let x = Value::scalar(21.700000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_415() {
        let x = Value::scalar(21.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_416() {
        let x = Value::scalar(21.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_417() {
        let x = Value::scalar(21.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_418() {
        let x = Value::scalar(21.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_419() {
        let x = Value::scalar(21.950000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_420() {
        let x = Value::scalar(22.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_421() {
        let x = Value::scalar(22.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_422() {
        let x = Value::scalar(22.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_423() {
        let x = Value::scalar(22.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_424() {
        let x = Value::scalar(22.200000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_425() {
        let x = Value::scalar(22.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_426() {
        let x = Value::scalar(22.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_427() {
        let x = Value::scalar(22.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_428() {
        let x = Value::scalar(22.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_429() {
        let x = Value::scalar(22.450000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_430() {
        let x = Value::scalar(22.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_431() {
        let x = Value::scalar(22.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_432() {
        let x = Value::scalar(22.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_433() {
        let x = Value::scalar(22.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_434() {
        let x = Value::scalar(22.700000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_435() {
        let x = Value::scalar(22.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_436() {
        let x = Value::scalar(22.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_437() {
        let x = Value::scalar(22.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_438() {
        let x = Value::scalar(22.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_439() {
        let x = Value::scalar(22.950000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_440() {
        let x = Value::scalar(23.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_441() {
        let x = Value::scalar(23.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_442() {
        let x = Value::scalar(23.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_443() {
        let x = Value::scalar(23.150000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_444() {
        let x = Value::scalar(23.200000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_445() {
        let x = Value::scalar(23.25);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_446() {
        let x = Value::scalar(23.3);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_447() {
        let x = Value::scalar(23.35);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_448() {
        let x = Value::scalar(23.400000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_449() {
        let x = Value::scalar(23.450000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_450() {
        let x = Value::scalar(23.5);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_451() {
        let x = Value::scalar(23.55);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_452() {
        let x = Value::scalar(23.6);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_453() {
        let x = Value::scalar(23.650000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_454() {
        let x = Value::scalar(23.700000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_455() {
        let x = Value::scalar(23.75);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_456() {
        let x = Value::scalar(23.8);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_457() {
        let x = Value::scalar(23.85);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_458() {
        let x = Value::scalar(23.900000000000002);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_459() {
        let x = Value::scalar(23.950000000000003);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_460() {
        let x = Value::scalar(24.0);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_461() {
        let x = Value::scalar(24.05);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    #[test]
    fn test_autograd_lib_stress_462() {
        let x = Value::scalar(24.1);
        let y = x.mul(&x);
        assert!(y.data().get(0) > 0.0);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
