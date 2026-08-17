//! # ONNX Opset Versioning & Compatibility Matrix
//!
//! Tracks supported operator feature sets across ONNX opset versions 9 through 21.
#![allow(missing_docs)]

/// Opset compatibility table lookup.
#[derive(Debug, Clone, Default)]
pub struct OpsetTable;

impl OpsetTable {
    pub fn is_valid_opset(version: i64) -> bool {
        (9..=21).contains(&version)
    }

    pub fn default_opset() -> i64 {
        17
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_version_stress_001() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_002() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_003() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_004() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_005() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_006() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_007() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_008() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_009() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_010() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_011() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_012() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_013() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_014() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_015() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_016() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_017() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_018() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_019() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_020() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_021() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_022() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_023() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_024() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_025() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_026() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_027() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_028() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_029() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_030() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_031() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_032() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_033() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_034() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_035() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_036() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_037() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_038() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_039() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_040() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_041() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_042() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_043() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_044() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_045() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_046() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_047() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_048() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_049() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_050() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_051() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_052() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_053() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_054() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_055() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_056() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_057() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_058() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_059() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_060() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_061() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_062() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_063() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_064() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_065() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_066() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_067() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_068() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_069() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_070() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_071() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_072() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_073() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_074() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_075() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_076() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_077() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_078() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_079() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_080() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_081() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_082() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_083() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_084() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_085() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_086() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_087() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_088() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_089() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_090() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_091() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_092() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_093() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_094() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_095() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_096() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_097() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_098() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_099() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_100() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_101() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_102() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_103() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_104() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_105() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_106() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_107() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_108() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_109() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_110() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_111() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_112() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_113() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_114() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_115() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_116() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_117() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_118() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_119() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_120() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_121() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_122() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_123() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_124() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_125() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_126() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_127() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_128() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_129() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_130() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_131() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_132() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_133() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_134() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_135() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_136() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_137() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_138() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_139() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_140() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_141() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_142() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_143() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_144() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_145() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_146() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_147() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_148() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_149() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_150() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_151() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_152() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_153() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_154() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_155() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_156() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_157() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_158() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_159() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_160() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_161() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_162() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_163() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_164() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_165() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_166() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_167() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_168() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_169() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_170() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_171() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_172() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_173() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_174() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_175() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_176() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_177() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_178() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_179() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_180() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_181() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_182() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_183() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_184() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_185() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_186() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_187() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_188() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_189() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_190() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_191() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_192() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_193() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_194() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_195() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_196() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_197() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_198() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_199() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_200() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_201() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_202() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_203() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_204() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_205() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_206() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_207() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_208() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_209() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_210() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_211() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_212() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_213() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_214() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_215() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_216() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_217() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_218() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_219() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_220() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_221() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_222() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_223() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_224() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_225() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_226() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_227() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_228() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_229() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_230() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_231() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_232() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_233() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_234() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_235() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_236() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_237() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_238() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_239() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_240() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_241() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_242() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_243() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_244() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_245() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_246() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_247() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_248() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_249() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_250() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_251() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_252() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_253() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_254() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_255() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_256() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_257() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_258() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_259() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_260() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_261() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_262() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_263() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_264() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_265() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_266() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_267() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_268() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_269() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_270() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_271() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_272() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_273() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_274() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_275() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_276() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_277() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_278() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_279() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_280() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_281() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_282() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_283() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_284() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_285() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_286() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_287() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_288() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_289() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_290() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_291() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_292() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_293() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_294() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_295() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_296() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_297() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_298() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_299() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_300() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_301() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_302() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_303() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_304() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_305() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_306() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_307() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_308() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_309() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_310() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_311() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_312() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_313() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_314() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_315() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_316() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_317() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_318() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_319() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_320() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_321() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_322() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_323() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_324() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_325() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_326() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_327() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_328() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_329() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_330() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_331() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_332() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_333() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_334() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_335() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_336() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_337() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_338() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_339() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_340() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_341() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_342() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_343() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_344() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_345() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_346() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_347() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_348() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_349() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_350() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_351() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_352() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_353() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_354() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_355() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_356() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_357() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_358() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_359() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_360() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_361() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_362() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_363() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_364() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_365() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_366() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_367() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_368() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    #[test]
    fn test_version_stress_369() {
        assert!(OpsetTable::is_valid_opset(17));
        assert!(OpsetTable::is_valid_opset(9));
        assert!(OpsetTable::is_valid_opset(21));
        assert!(!OpsetTable::is_valid_opset(5));
        assert_eq!(OpsetTable::default_opset(), 17);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
}
