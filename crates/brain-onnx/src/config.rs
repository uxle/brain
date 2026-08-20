//! # ONNX Configurations
//!
//! Settings for model import, optimization levels, graph lowering, and execution evaluation.
#![allow(missing_docs)]

/// Policy for handling unknown or unsupported operators during import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownOpPolicy {
    #[default]
    Error,
    Skip,
    CustomFallback,
}

/// Import configuration controlling opset alignment and shape inference.
#[derive(Debug, Clone)]
pub struct ImportConfig {
    pub target_opset: Option<i64>,
    pub unknown_op_policy: UnknownOpPolicy,
    pub infer_shapes: bool,
    pub fold_constants: bool,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            target_opset: Some(17),
            unknown_op_policy: UnknownOpPolicy::Error,
            infer_shapes: true,
            fold_constants: true,
        }
    }
}

/// Optimization configuration levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    None,
    Basic,
    #[default]
    Extended,
    All,
}

/// Configuration for ONNX graph optimization.
#[derive(Debug, Clone, Default)]
pub struct OptimizeConfig {
    pub level: OptimizationLevel,
    pub fuse_bn_relu: bool,
    pub fuse_conv_relu: bool,
    pub fuse_gemm: bool,
}

/// Configuration for ONNX graph evaluation.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub tolerance: f64,
    pub verbose: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            verbose: false,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
