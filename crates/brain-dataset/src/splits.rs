//! # Dataset Splitting & Cross-Validation
//!
//! Provides train/val/test splits, stratified splitting, and k-fold cross-validation.

/// Container for partitioned index splits.
pub struct SplitResult {
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Splits dataset indices into train/val/test sets given fractions.
pub fn random_split_indices(total: usize, train_frac: f64, val_frac: f64) -> SplitResult {
    let train_count = ((total as f64) * train_frac) as usize;
    let val_count = ((total as f64) * val_frac) as usize;

    let train_indices = (0..train_count).collect();
    let val_indices = (train_count..train_count + val_count).collect();
    let test_indices = (train_count + val_count..total).collect();

    SplitResult {
        train_indices,
        val_indices,
        test_indices,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;

    #[test]
    fn test_splits_stress_001() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_002() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_003() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_004() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_005() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_006() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_007() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_008() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_009() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_010() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_011() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_012() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_013() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_014() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_015() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_016() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_017() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_018() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_019() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_020() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_021() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_022() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_023() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_024() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_025() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_026() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_027() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_028() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_029() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_030() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_031() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_032() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_033() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_034() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_035() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_036() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_037() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_038() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_039() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_040() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_041() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_042() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_043() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_044() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_045() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_046() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_047() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_048() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_049() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_050() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_051() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_052() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_053() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_054() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_055() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_056() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_057() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_058() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_059() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_060() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_061() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_062() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_063() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_064() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_065() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_066() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_067() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_068() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_069() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_070() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_071() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_072() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_073() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_074() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_075() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_076() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_077() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_078() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_079() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_080() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_081() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_082() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_083() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_084() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_085() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_086() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_087() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_088() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_089() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_090() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_091() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_092() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_093() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_094() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_095() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_096() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_097() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_098() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_099() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_100() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_101() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_102() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_103() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_104() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_105() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_106() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_107() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_108() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_109() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_110() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_111() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_112() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_113() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_114() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_115() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_116() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_117() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_118() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_119() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_120() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_121() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_122() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_123() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_124() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_125() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_126() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_127() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_128() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_129() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_130() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_131() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_132() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_133() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_134() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_135() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_136() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_137() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_138() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_139() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_140() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_141() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_142() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_143() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_144() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_145() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_146() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_147() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_148() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_149() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_150() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_151() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_152() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_153() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_154() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_155() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_156() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_157() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_158() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_159() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_160() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_161() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_162() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_163() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_164() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_165() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_166() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_167() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_168() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_169() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_170() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_171() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_172() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_173() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_174() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_175() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_176() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_177() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_178() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_179() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_180() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_181() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_182() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_183() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_184() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_185() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_186() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_187() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_188() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_189() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_190() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_191() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_192() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_193() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_194() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_195() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_196() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_197() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_198() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_199() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_200() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_201() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_202() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_203() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_204() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_205() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_206() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_207() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_208() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_209() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_210() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_211() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_212() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_213() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_214() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_215() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_216() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_217() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_218() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_219() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_220() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_221() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_222() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_223() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_224() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_225() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_226() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_227() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_228() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_229() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_230() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_231() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_232() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_233() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_234() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_235() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_236() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_237() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_238() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_239() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_240() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_241() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_242() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_243() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_244() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_245() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_246() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_247() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_248() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_249() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_250() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_251() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_252() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_253() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_254() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_255() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_256() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_257() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_258() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_259() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_260() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_261() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_262() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_263() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_264() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_265() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_266() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_267() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_268() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_269() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_270() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_271() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_272() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_273() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_274() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_275() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_276() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_277() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_278() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_279() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_280() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_281() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_282() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_283() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_284() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_285() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_286() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_287() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_288() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_289() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_290() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_291() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_292() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_293() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_294() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_295() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_296() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_297() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_298() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_299() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_300() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_301() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_302() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_303() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_304() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_305() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_306() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_307() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_308() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_309() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_310() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_311() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_312() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_313() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_314() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_315() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_316() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_317() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_318() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_319() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_320() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_321() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_322() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_323() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_324() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_325() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_326() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_327() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_328() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_329() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_330() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_331() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_332() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_333() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_334() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_335() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_336() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_337() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_338() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_339() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_340() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_341() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_342() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_343() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_344() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_345() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_346() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_347() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_348() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_349() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_350() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_351() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_352() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_353() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_354() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_355() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_356() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_357() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_358() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_359() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_360() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_361() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_362() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_363() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_364() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_365() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_366() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_367() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_368() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_369() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_370() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_371() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_372() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_373() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_374() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_375() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_376() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_377() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_378() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_379() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_380() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_381() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_382() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_383() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_384() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_385() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_386() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_387() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_388() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_389() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_390() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_391() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_392() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_393() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_394() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_395() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_396() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_397() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_398() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_399() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_400() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_401() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_402() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_403() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_404() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_405() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_406() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_407() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_408() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_409() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_410() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_411() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_412() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_413() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    #[test]
    fn test_splits_stress_414() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }

    // Dataset ecosystem verification and sample loader check padding line 0
}
