//! # Container Modules
//!
//! Containers for orchestrating layer pipelines: `Sequential`, `SequentialNamed`, and `ModuleList`.
#![allow(missing_docs)]

pub mod seq;
pub mod sequential2;

pub use seq::Sequential;
pub use sequential2::{SequentialNamed, NamedModule};
pub use crate::module::{ModuleList, ModuleDict};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
