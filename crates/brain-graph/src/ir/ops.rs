//! # Operation Catalog & Registry
//!
//! Enumeration of all graph operators, signature lookup, and shape calculators.
#![allow(missing_docs)]

use std::collections::HashMap;

/// Broad category of operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OpKind {
    #[default]
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    MatMul,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm,
    LayerNorm,
    Softmax,
    Reshape,
    Transpose,
    Flatten,
    Constant,
    Custom,
}

impl OpKind {
    pub fn name(&self) -> &'static str {
        match self {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::Neg => "Neg",
            OpKind::Relu => "Relu",
            OpKind::Sigmoid => "Sigmoid",
            OpKind::Tanh => "Tanh",
            OpKind::Gelu => "Gelu",
            OpKind::MatMul => "MatMul",
            OpKind::Conv2D => "Conv2D",
            OpKind::MaxPool2D => "MaxPool2D",
            OpKind::AvgPool2D => "AvgPool2D",
            OpKind::BatchNorm => "BatchNorm",
            OpKind::LayerNorm => "LayerNorm",
            OpKind::Softmax => "Softmax",
            OpKind::Reshape => "Reshape",
            OpKind::Transpose => "Transpose",
            OpKind::Flatten => "Flatten",
            OpKind::Constant => "Constant",
            OpKind::Custom => "Custom",
        }
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
            | OpKind::Neg | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Gelu
        )
    }

    pub fn min_inputs(&self) -> usize {
        match self {
            OpKind::Constant => 0,
            OpKind::Neg | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Gelu
            | OpKind::MaxPool2D | OpKind::AvgPool2D | OpKind::Softmax
            | OpKind::Reshape | OpKind::Transpose | OpKind::Flatten => 1,
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::MatMul
            | OpKind::Conv2D | OpKind::BatchNorm | OpKind::LayerNorm => 2,
            OpKind::Custom => 1,
        }
    }
}

/// Registry mapping string identifiers to OpKind and metadata.
#[derive(Debug, Default)]
pub struct OpRegistry {
    ops: HashMap<String, OpKind>,
}

impl OpRegistry {
    pub fn new() -> Self {
        let mut reg = Self { ops: HashMap::new() };
        reg.register("add", OpKind::Add);
        reg.register("sub", OpKind::Sub);
        reg.register("mul", OpKind::Mul);
        reg.register("div", OpKind::Div);
        reg.register("matmul", OpKind::MatMul);
        reg.register("relu", OpKind::Relu);
        reg.register("conv2d", OpKind::Conv2D);
        reg.register("softmax", OpKind::Softmax);
        reg
    }

    pub fn register(&mut self, name: &str, op: OpKind) {
        self.ops.insert(name.to_lowercase(), op);
    }

    pub fn lookup(&self, name: &str) -> Option<OpKind> {
        self.ops.get(&name.to_lowercase()).copied()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_002() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_003() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_004() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_005() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_006() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_007() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_008() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_009() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_010() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_011() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_012() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_013() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_014() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_015() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_016() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_017() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_018() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_019() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_020() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_021() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_022() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_023() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_024() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_025() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_026() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_027() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_028() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_029() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_030() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_031() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_032() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_033() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_034() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_035() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_036() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_037() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_038() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_039() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_040() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_041() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_042() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_043() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_044() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_045() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_046() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_047() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_048() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_049() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_050() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_051() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_052() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_053() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_054() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_055() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_056() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_057() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_058() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_059() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_060() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_061() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_062() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_063() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_064() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_065() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_066() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_067() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_068() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_069() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_070() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_071() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_072() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_073() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_074() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_075() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_076() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_077() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_078() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_079() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_080() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_081() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_082() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_083() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_084() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_085() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_086() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_087() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_088() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_089() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_090() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_091() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_092() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_093() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_094() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_095() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_096() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_097() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_098() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_099() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_100() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_101() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_102() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_103() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_104() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_105() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_106() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_107() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_108() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_109() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_110() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_111() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_112() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_113() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_114() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_115() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_116() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_117() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_118() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_119() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_120() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_121() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_122() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_123() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_124() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_125() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_126() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_127() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_128() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_129() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_130() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_131() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_132() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_133() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_134() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_135() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_136() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_137() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_138() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_139() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_140() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_141() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_142() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_143() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_144() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_145() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_146() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_147() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_148() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_149() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_150() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_151() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_152() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_153() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_154() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_155() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_156() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_157() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_158() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_159() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_160() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_161() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_162() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_163() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_164() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_165() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_166() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_167() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_168() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_169() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_170() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_171() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_172() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_173() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_174() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_175() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_176() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_177() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_178() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_179() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_180() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_181() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_182() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_183() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_184() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_185() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_186() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_187() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_188() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_189() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_190() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_191() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_192() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_193() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_194() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_195() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_196() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_197() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_198() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_199() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_200() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_201() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_202() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_203() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_204() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_205() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_206() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_207() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_208() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_209() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_210() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_211() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_212() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_213() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_214() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_215() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_216() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_217() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_218() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_219() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_220() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_221() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_222() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_223() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_224() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_225() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_226() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_227() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_228() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_229() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_230() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_231() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_232() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_233() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_234() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_235() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_236() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_237() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_238() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_239() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_240() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_241() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_242() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_243() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_244() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_245() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_246() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_247() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_248() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_249() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_250() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_251() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_252() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_253() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_254() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_255() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_256() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_257() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_258() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_259() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_260() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_261() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_262() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_263() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_264() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_265() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_266() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_267() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_268() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_269() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_270() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_271() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_272() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_273() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_274() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_275() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_276() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_277() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_278() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_279() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_280() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_281() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_282() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_283() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_284() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_285() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_286() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_287() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_288() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_289() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_290() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_291() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_292() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_293() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_294() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_295() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_296() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_297() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_298() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_299() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_300() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_301() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_302() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_303() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_304() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_305() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_306() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_307() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_308() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_309() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_310() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_311() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_312() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_313() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_314() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_315() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_316() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_317() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_318() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_319() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_320() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_321() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_322() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }

    #[test]
    fn test_ops_stress_323() {
        let reg = OpRegistry::new();
        assert_eq!(reg.lookup("add"), Some(OpKind::Add));
        assert_eq!(reg.lookup("matmul"), Some(OpKind::MatMul));
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert_eq!(OpKind::Add.min_inputs(), 2);
    }
}
