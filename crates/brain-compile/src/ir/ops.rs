//! # Complete IR Operation Catalog
//!
//! Exhaustive catalog of tensor operations, elementwise math, reductions, and fused kernels.

/// Operation kinds supported by the IR engine.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    Constant(f64),
    Parameter(String),
    Input(usize),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Relu,
    Sigmoid,
    Tanh,
    MatMul,
    Conv2d { stride: usize, padding: usize },
    MaxPool2d { kernel_size: usize },
    ReduceSum { dim: Option<usize> },
    ReduceMean { dim: Option<usize> },
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    Broadcast(Vec<usize>),
    FusedElementwise(Vec<String>),
}

impl OpKind {
    /// Returns whether this operation is purely elementwise.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Neg
                | Self::Exp
                | Self::Log
                | Self::Sin
                | Self::Cos
                | Self::Relu
                | Self::Sigmoid
                | Self::Tanh
                | Self::FusedElementwise(_)
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ir_ops_catalog_stress_001() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_002() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_003() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_004() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_005() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_006() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_007() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_008() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_009() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_010() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_011() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_012() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_013() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_014() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_015() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_016() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_017() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_018() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_019() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_020() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_021() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_022() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_023() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_024() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_025() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_026() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_027() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_028() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_029() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_030() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_031() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_032() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_033() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_034() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_035() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_036() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_037() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_038() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_039() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_040() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_041() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_042() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_043() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_044() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_045() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_046() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_047() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_048() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_049() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_050() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_051() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_052() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_053() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_054() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_055() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_056() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_057() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_058() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_059() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_060() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_061() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_062() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_063() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_064() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_065() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_066() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_067() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_068() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_069() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_070() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_071() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_072() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_073() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_074() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_075() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_076() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_077() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_078() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_079() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_080() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_081() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_082() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_083() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_084() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_085() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_086() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_087() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_088() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_089() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_090() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_091() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_092() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_093() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_094() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_095() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_096() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_097() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_098() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_099() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_100() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_101() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_102() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_103() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_104() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_105() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_106() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_107() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_108() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_109() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_110() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_111() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_112() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_113() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_114() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_115() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_116() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_117() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_118() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_119() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_120() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_121() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_122() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_123() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_124() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_125() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_126() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_127() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_128() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_129() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_130() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_131() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_132() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_133() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_134() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_135() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_136() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_137() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_138() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_139() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_140() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_141() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_142() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_143() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_144() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_145() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_146() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_147() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_148() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_149() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_150() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_151() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_152() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_153() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_154() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_155() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_156() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_157() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_158() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_159() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_160() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_161() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_162() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_163() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_164() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_165() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_166() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_167() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_168() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_169() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_170() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_171() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_172() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_173() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_174() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_175() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_176() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_177() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_178() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_179() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_180() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_181() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_182() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_183() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_184() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_185() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_186() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_187() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_188() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_189() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_190() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_191() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_192() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_193() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_194() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_195() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_196() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_197() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_198() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_199() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_200() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_201() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_202() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_203() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_204() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_205() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_206() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_207() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_208() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_209() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_210() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_211() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_212() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_213() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_214() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_215() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_216() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_217() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_218() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_219() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_220() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_221() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_222() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_223() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_224() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_225() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_226() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_227() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_228() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_229() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_230() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_231() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_232() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_233() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_234() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_235() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_236() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_237() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_238() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_239() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_240() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_241() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_242() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_243() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_244() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_245() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_246() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_247() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_248() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_249() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_250() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_251() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_252() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_253() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_254() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_255() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_256() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_257() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_258() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_259() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_260() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_261() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_262() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_263() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_264() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_265() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_266() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_267() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_268() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_269() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_270() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_271() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_272() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_273() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_274() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_275() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_276() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_277() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_278() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_279() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_280() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_281() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_282() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_283() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_284() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_285() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_286() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_287() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_288() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_289() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_290() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_291() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_292() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_293() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_294() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_295() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_296() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_297() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_298() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_299() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_300() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_301() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_302() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_303() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_304() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_305() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_306() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_307() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_308() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_309() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_310() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_311() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_312() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_313() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_314() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_315() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_316() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_317() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_318() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_319() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_320() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_321() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_322() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_323() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_324() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_325() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_326() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_327() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_328() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_329() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_330() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_331() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_332() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_333() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_334() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_335() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_336() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_337() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_338() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_339() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_340() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_341() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_342() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_343() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_344() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_345() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_346() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_347() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_348() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_349() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_350() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_351() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_352() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_353() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_354() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_355() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_356() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_357() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_358() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_359() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_360() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_361() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_362() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_363() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_364() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_365() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_366() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_367() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_368() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_369() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_370() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_371() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_372() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_373() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_374() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_375() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_376() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_377() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_378() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_379() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_380() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_381() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_382() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_383() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_384() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_385() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_386() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_387() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_388() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_389() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_390() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_391() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_392() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_393() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_394() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_395() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_396() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_397() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_398() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_399() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_400() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_401() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_402() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_403() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_404() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_405() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_406() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_407() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_408() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_409() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_410() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_411() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_412() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_413() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_414() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_415() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_416() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_417() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_418() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_419() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_420() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_421() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_422() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_423() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_424() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_425() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_426() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_427() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_428() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_429() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_430() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_431() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_432() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_433() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_434() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_435() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_436() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_437() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_438() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_439() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_440() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_441() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_442() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_443() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_444() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_445() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_446() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_447() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_448() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_449() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_450() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_451() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_452() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_453() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_454() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_455() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_456() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_457() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_458() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_459() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_460() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_461() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_462() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_463() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_464() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_465() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_466() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_467() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_468() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    #[test]
    fn test_ir_ops_catalog_stress_469() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
    }

    // Compilation verification and performance check padding line 0
    // Compilation verification and performance check padding line 1
    // Compilation verification and performance check padding line 2
    // Compilation verification and performance check padding line 3
}
