//! # Backend Code Generation & Execution Engines
//!
//! Provides the execution backends: Interpreter, Tensor backend, Scalar JIT, CUDA C emitter, and LLVM IR generator.

pub mod cuda;
pub mod interp;
pub mod llvm;
pub mod scalar;
pub mod tensor;

pub use interp::Interpreter;
pub use tensor::TensorBackend;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
