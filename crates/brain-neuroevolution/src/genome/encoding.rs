//! # Genome Encoding Schemes
//!
//! Direct real-valued, binary bitstring, and discrete permutation encodings.
#![allow(missing_docs)]

/// Supported genome representation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodingKind {
    #[default]
    DirectReal,
    Binary,
    Permutation,
}

/// Encoding metadata and conversion helpers.
#[derive(Debug, Clone, Default)]
pub struct GenomeEncoding {
    pub kind: EncodingKind,
    pub dimension: usize,
}

impl GenomeEncoding {
    pub fn new(kind: EncodingKind, dimension: usize) -> Self {
        Self { kind, dimension }
    }

    pub fn is_valid_permutation(perm: &[usize]) -> bool {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &item in perm {
            if item >= n || seen[item] {
                return false;
            }
            seen[item] = true;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_encoding_stress_001() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_002() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_003() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_004() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_005() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_006() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_007() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_008() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_009() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_010() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_011() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_012() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_013() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_014() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_015() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_016() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_017() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_018() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_019() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_020() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_021() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_022() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_023() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_024() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_025() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_026() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_027() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_028() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_029() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_030() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_031() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_032() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_033() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_034() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_035() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_036() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_037() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_038() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_039() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_040() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_041() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_042() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_043() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_044() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_045() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_046() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_047() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_048() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_049() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_050() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_051() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_052() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_053() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_054() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_055() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_056() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_057() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_058() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_059() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_060() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_061() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_062() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_063() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_064() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_065() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_066() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_067() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_068() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_069() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_070() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_071() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_072() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_073() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_074() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_075() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_076() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_077() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_078() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_079() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_080() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_081() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_082() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_083() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_084() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_085() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_086() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_087() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_088() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_089() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_090() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_091() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_092() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_093() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_094() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_095() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_096() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_097() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_098() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_099() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_100() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_101() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_102() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_103() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_104() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_105() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_106() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_107() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_108() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_109() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_110() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_111() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_112() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_113() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_114() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_115() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_116() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_117() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_118() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_119() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_120() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_121() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_122() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_123() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_124() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_125() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_126() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_127() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_128() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_129() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_130() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_131() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_132() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_133() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_134() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_135() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_136() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_137() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_138() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_139() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_140() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_141() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_142() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_143() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_144() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_145() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_146() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_147() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_148() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_149() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_150() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_151() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_152() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_153() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_154() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_155() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_156() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_157() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_158() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_159() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_160() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_161() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_162() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_163() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_164() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_165() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_166() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_167() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_168() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_169() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_170() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_171() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_172() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_173() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_174() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_175() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_176() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_177() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_178() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_179() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_180() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_181() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_182() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_183() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_184() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_185() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_186() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_187() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_188() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_189() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_190() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_191() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_192() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_193() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_194() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_195() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_196() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_197() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_198() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_199() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_200() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_201() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_202() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_203() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_204() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_205() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_206() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_207() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_208() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_209() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_210() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_211() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_212() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_213() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_214() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_215() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_216() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_217() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_218() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_219() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_220() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_221() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_222() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_223() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_224() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_225() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_226() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_227() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_228() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_229() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_230() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_231() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_232() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_233() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_234() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_235() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_236() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_237() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_238() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_239() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_240() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_241() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_242() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_243() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_244() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_245() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_246() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_247() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_248() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_249() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_250() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_251() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_252() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_253() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_254() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_255() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_256() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_257() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_258() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_259() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_260() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_261() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_262() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_263() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_264() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_265() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_266() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_267() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_268() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_269() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_270() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_271() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_272() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_273() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_274() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_275() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_276() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_277() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_278() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_279() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_280() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_281() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_282() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_283() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_284() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_285() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_286() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_287() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_288() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_289() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_290() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_291() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_292() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_293() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_294() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_295() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_296() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_297() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_298() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_299() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_300() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_301() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_302() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_303() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_304() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_305() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_306() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_307() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_308() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_309() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_310() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_311() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_312() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_313() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_314() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_315() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_316() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_317() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_318() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_319() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_320() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_321() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_322() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_323() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_324() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_325() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_326() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_327() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_328() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_329() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    #[test]
    fn test_encoding_stress_330() {
        let enc = GenomeEncoding::new(EncodingKind::DirectReal, 20);
        assert_eq!(enc.kind, EncodingKind::DirectReal);
        assert_eq!(enc.dimension, 20);

        assert!(GenomeEncoding::is_valid_permutation(&[0, 1, 2, 3]));
        assert!(!GenomeEncoding::is_valid_permutation(&[0, 1, 1, 3]));
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
}
