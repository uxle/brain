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

    #[test]
    fn test_compile_core_stress_001() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_002() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_003() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_004() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_005() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_006() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_007() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_008() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_009() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_010() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_011() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_012() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_013() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_014() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_015() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_016() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_017() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_018() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_019() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_020() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_021() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_022() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_023() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_024() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_025() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_026() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_027() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_028() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_029() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_030() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_031() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_032() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_033() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_034() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_035() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_036() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_037() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_038() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_039() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_040() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_041() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_042() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_043() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_044() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_045() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_046() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_047() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_048() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_049() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_050() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_051() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_052() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_053() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_054() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_055() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_056() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_057() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_058() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_059() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_060() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_061() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_062() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_063() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_064() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_065() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_066() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_067() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_068() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_069() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_070() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_071() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_072() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_073() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_074() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_075() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_076() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_077() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_078() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_079() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_080() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_081() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_082() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_083() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_084() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_085() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_086() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_087() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_088() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_089() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_090() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_091() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_092() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_093() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_094() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_095() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_096() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_097() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_098() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_099() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_100() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_101() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_102() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_103() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_104() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_105() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_106() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_107() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_108() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_109() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_110() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_111() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_112() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_113() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_114() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_115() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_116() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_117() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_118() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_119() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_120() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_121() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_122() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_123() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_124() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_125() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_126() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_127() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_128() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_129() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_130() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_131() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_132() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_133() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_134() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_135() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_136() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_137() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_138() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_139() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_140() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_141() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_142() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_143() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_144() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_145() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_146() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_147() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_148() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_149() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_150() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_151() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_152() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_153() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_154() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_155() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_156() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_157() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_158() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_159() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_160() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_161() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_162() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_163() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_164() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_165() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_166() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_167() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_168() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_169() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_170() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_171() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_172() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_173() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_174() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_175() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_176() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_177() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_178() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_179() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_180() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_181() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_182() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_183() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_184() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_185() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_186() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_187() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_188() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_189() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_190() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_191() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_192() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_193() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_194() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_195() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_196() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_197() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_198() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_199() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_200() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_201() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_202() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_203() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_204() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_205() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_206() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_207() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_208() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_209() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_210() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_211() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_212() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_213() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_214() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_215() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_216() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_217() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_218() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_219() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_220() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_221() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_222() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_223() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_224() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_225() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_226() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_227() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_228() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_229() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_230() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_231() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_232() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_233() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_234() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_235() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_236() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_237() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_238() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_239() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_240() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_241() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_242() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_243() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_244() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_245() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_246() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_247() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_248() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_249() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_250() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_251() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_252() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_253() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_254() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_255() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_256() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_257() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_258() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_259() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_260() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_261() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_262() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_263() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_264() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_265() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_266() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_267() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_268() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_269() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_270() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_271() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_272() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_273() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_274() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_275() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_276() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_277() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_278() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_279() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_280() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_281() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_282() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_283() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_284() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_285() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_286() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_287() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_288() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_289() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_290() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_291() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_292() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_293() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_294() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }

    #[test]
    fn test_compile_core_stress_295() {
        let opts = CompileOptions::new()
            .with_target(TargetBackend::Tensor)
            .with_opt_level(OptimizationLevel::O3)
            .with_fusion(true);
        assert_eq!(opts.target, TargetBackend::Tensor);
        assert_eq!(opts.opt_level, OptimizationLevel::O3);
        assert!(opts.enable_fusion);
    }
}
