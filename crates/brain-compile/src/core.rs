//! # Core Compilation Data Types & Targets
//!
//! Provides the primary primitives for JIT compilation options, target backends,
//! optimization levels, and compilation diagnostic results.

/// Target backend for lowered execution and code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetBackend {
    #[default]
    Interpreter,
    Tensor,
    Scalar,
    Cuda,
    Llvm,
}

/// Optimization pass optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum OptimizationLevel {
    O0,
    O1,
    #[default]
    O2,
    O3,
}

/// Configuration options for the compilation pipeline.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub target: TargetBackend,
    pub opt_level: OptimizationLevel,
    pub enable_fusion: bool,
    pub enable_constant_folding: bool,
    pub enable_dce: bool,
    pub max_fusion_group_size: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            target: TargetBackend::Interpreter,
            opt_level: OptimizationLevel::O2,
            enable_fusion: true,
            enable_constant_folding: true,
            enable_dce: true,
            max_fusion_group_size: 16,
        }
    }
}

impl CompileOptions {
    /// Creates a new `CompileOptions` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target execution backend.
    pub fn with_target(mut self, target: TargetBackend) -> Self {
        self.target = target;
        self
    }

    /// Sets the optimization level.
    pub fn with_opt_level(mut self, level: OptimizationLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Enables or disables kernel fusion.
    pub fn with_fusion(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }
}

/// Diagnostic compilation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError {
    VerificationFailed(String),
    UnsupportedOp(String),
    TypeMismatch(String),
    BackendError(String),
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::UnsupportedOp(msg) => write!(f, "Unsupported op: {}", msg),
            Self::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            Self::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for CompilationError {}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
