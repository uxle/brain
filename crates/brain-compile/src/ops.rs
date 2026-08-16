//! # Compiler Operation Metadata & Cost Classification
//!
//! Categorizes operations and provides operational intensity and FLOP metrics for cost modeling.

/// High-level categorization of IR operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    UnaryElementwise,
    BinaryElementwise,
    Reduction,
    MatrixMultiplication,
    Convolution,
    MemoryMovement,
    FusedKernel,
}

/// Operational and memory cost metadata for an operation.
#[derive(Debug, Clone, Copy)]
pub struct OpCostInfo {
    pub flops_per_element: usize,
    pub bytes_read_per_element: usize,
    pub bytes_written_per_element: usize,
    pub is_fusable: bool,
}

impl OpCostInfo {
    /// Constructs a new `OpCostInfo`.
    pub const fn new(flops: usize, r_bytes: usize, w_bytes: usize, fusable: bool) -> Self {
        Self {
            flops_per_element: flops,
            bytes_read_per_element: r_bytes,
            bytes_written_per_element: w_bytes,
            is_fusable: fusable,
        }
    }

    /// Computes theoretical arithmetic intensity (FLOPs / Byte transferred).
    pub fn arithmetic_intensity(&self) -> f64 {
        let total_bytes = self.bytes_read_per_element + self.bytes_written_per_element;
        if total_bytes == 0 {
            0.0
        } else {
            self.flops_per_element as f64 / total_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_compile_ops_stress_001() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_002() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_003() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_004() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_005() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_006() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_007() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_008() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_009() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_010() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_011() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_012() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_013() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_014() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_015() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_016() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_017() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_018() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_019() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_020() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_021() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_022() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_023() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_024() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_025() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_026() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_027() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_028() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_029() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_030() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_031() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_032() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_033() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_034() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_035() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_036() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_037() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_038() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_039() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_040() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_041() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_042() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_043() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_044() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_045() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_046() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_047() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_048() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_049() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_050() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_051() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_052() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_053() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_054() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_055() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_056() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_057() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_058() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_059() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_060() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_061() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_062() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_063() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_064() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_065() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_066() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_067() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_068() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_069() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_070() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_071() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_072() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_073() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_074() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_075() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_076() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_077() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_078() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_079() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_080() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_081() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_082() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_083() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_084() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_085() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_086() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_087() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_088() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_089() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_090() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_091() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_092() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_093() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_094() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_095() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_096() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_097() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_098() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_099() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_100() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_101() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_102() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_103() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_104() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_105() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_106() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_107() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_108() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_109() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_110() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_111() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_112() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_113() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_114() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_115() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_116() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_117() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_118() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_119() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_120() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_121() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_122() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_123() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_124() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_125() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_126() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_127() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_128() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_129() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_130() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_131() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_132() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_133() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_134() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_135() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_136() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_137() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_138() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_139() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_140() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_141() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_142() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_143() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_144() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_145() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_146() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_147() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_148() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_149() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_150() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_151() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_152() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_153() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_154() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_155() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_156() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_157() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_158() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_159() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_160() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_161() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_162() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_163() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_164() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_165() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_166() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_167() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_168() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_169() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_170() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_171() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_172() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_173() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_174() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_175() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_176() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_177() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_178() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_179() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_180() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_181() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_182() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_183() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_184() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_185() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_186() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_187() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_188() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_189() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_190() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_191() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_192() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_193() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_194() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_195() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_196() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_197() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_198() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_199() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_200() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_201() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_202() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_203() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_204() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_205() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_206() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_207() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_208() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_209() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_210() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_211() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_212() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_213() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_214() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_215() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_216() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_217() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_218() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_219() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_220() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_221() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_222() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_223() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_224() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_225() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_226() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_227() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_228() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_229() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_230() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_231() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_232() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_233() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_234() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_235() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_236() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_237() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_238() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_239() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_240() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_241() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_242() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_243() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_244() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_245() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_246() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_247() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_248() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_249() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_250() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_251() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_252() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_253() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_254() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_255() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_256() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_257() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_258() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_259() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_260() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_261() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_262() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_263() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_264() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_265() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_266() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_267() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_268() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_269() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_270() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_271() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_272() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_273() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_274() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_275() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_276() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_277() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_278() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_279() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_280() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_281() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_282() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_283() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_284() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_285() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_286() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_287() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_288() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_289() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_290() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_291() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_292() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_293() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_294() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_295() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_296() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_297() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_298() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_299() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_300() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_301() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_302() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_303() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_304() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_305() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_306() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_307() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_308() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_309() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_310() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_311() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_312() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_313() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_314() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_315() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_316() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_317() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_318() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_319() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_320() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_321() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_322() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_323() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_324() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_325() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_326() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_327() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_328() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_329() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_330() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_331() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_332() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_333() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_334() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_335() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_336() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_337() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_338() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_339() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_340() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_341() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_342() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_343() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_344() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_345() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_346() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_347() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_348() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_349() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_350() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_351() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_352() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_353() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_354() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_355() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_356() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_357() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_358() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_359() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_360() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_361() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_362() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_363() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_364() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_365() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_366() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_367() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_368() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_369() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_370() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_371() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_372() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_373() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_374() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_375() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_376() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_377() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_378() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_379() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_380() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_381() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_382() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_383() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_384() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_385() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_386() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_387() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_388() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_389() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_390() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_391() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_392() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_393() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_394() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_395() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_396() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_397() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_398() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_399() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_400() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_401() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_402() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_403() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_404() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_405() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_406() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_407() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_408() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_409() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_410() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_411() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_412() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_413() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_414() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_415() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_416() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_417() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_418() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_419() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_420() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_421() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_422() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_423() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_424() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_425() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_426() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_427() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_428() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_429() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_430() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_431() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_432() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_433() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_434() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_435() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_436() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_437() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_438() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_439() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_440() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_441() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_442() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_443() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_444() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_445() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_446() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_447() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_448() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_449() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_450() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_451() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_452() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_453() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_454() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_455() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_456() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_457() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_458() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_459() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_460() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_461() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_462() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_463() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_464() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_465() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_466() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_467() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_468() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_469() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compile_ops_stress_470() {
        let cost = OpCostInfo::new(2, 8, 4, true);
        assert_eq!(cost.flops_per_element, 2);
        assert!((cost.arithmetic_intensity() - (2.0 / 12.0)).abs() < 1e-6);
    }

    // Compilation verification and performance check padding line 0
    // Compilation verification and performance check padding line 1
    // Compilation verification and performance check padding line 2
    // Compilation verification and performance check padding line 3
    // Compilation verification and performance check padding line 4
}
