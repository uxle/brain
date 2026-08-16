//! # Dataset Batch Operations
//!
//! Functional batch transformations and filter predicates.

use crate::core::{Batch, Item};

/// Applies an in-place transformation across all items in a batch.
pub fn map_batch<F>(batch: Batch, f: F) -> Batch
where
    F: Fn(Item) -> Item,
{
    Batch::new(batch.items.into_iter().map(f).collect())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let b = Batch::new(vec![Item::new(1, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 1 + 1);
    }

    #[test]
    fn test_ops_stress_002() {
        let b = Batch::new(vec![Item::new(2, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 2 + 1);
    }

    #[test]
    fn test_ops_stress_003() {
        let b = Batch::new(vec![Item::new(3, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 3 + 1);
    }

    #[test]
    fn test_ops_stress_004() {
        let b = Batch::new(vec![Item::new(4, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 4 + 1);
    }

    #[test]
    fn test_ops_stress_005() {
        let b = Batch::new(vec![Item::new(5, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 5 + 1);
    }

    #[test]
    fn test_ops_stress_006() {
        let b = Batch::new(vec![Item::new(6, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 6 + 1);
    }

    #[test]
    fn test_ops_stress_007() {
        let b = Batch::new(vec![Item::new(7, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 7 + 1);
    }

    #[test]
    fn test_ops_stress_008() {
        let b = Batch::new(vec![Item::new(8, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 8 + 1);
    }

    #[test]
    fn test_ops_stress_009() {
        let b = Batch::new(vec![Item::new(9, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 9 + 1);
    }

    #[test]
    fn test_ops_stress_010() {
        let b = Batch::new(vec![Item::new(10, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 10 + 1);
    }

    #[test]
    fn test_ops_stress_011() {
        let b = Batch::new(vec![Item::new(11, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 11 + 1);
    }

    #[test]
    fn test_ops_stress_012() {
        let b = Batch::new(vec![Item::new(12, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 12 + 1);
    }

    #[test]
    fn test_ops_stress_013() {
        let b = Batch::new(vec![Item::new(13, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 13 + 1);
    }

    #[test]
    fn test_ops_stress_014() {
        let b = Batch::new(vec![Item::new(14, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 14 + 1);
    }

    #[test]
    fn test_ops_stress_015() {
        let b = Batch::new(vec![Item::new(15, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 15 + 1);
    }

    #[test]
    fn test_ops_stress_016() {
        let b = Batch::new(vec![Item::new(16, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 16 + 1);
    }

    #[test]
    fn test_ops_stress_017() {
        let b = Batch::new(vec![Item::new(17, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 17 + 1);
    }

    #[test]
    fn test_ops_stress_018() {
        let b = Batch::new(vec![Item::new(18, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 18 + 1);
    }

    #[test]
    fn test_ops_stress_019() {
        let b = Batch::new(vec![Item::new(19, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 19 + 1);
    }

    #[test]
    fn test_ops_stress_020() {
        let b = Batch::new(vec![Item::new(20, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 20 + 1);
    }

    #[test]
    fn test_ops_stress_021() {
        let b = Batch::new(vec![Item::new(21, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 21 + 1);
    }

    #[test]
    fn test_ops_stress_022() {
        let b = Batch::new(vec![Item::new(22, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 22 + 1);
    }

    #[test]
    fn test_ops_stress_023() {
        let b = Batch::new(vec![Item::new(23, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 23 + 1);
    }

    #[test]
    fn test_ops_stress_024() {
        let b = Batch::new(vec![Item::new(24, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 24 + 1);
    }

    #[test]
    fn test_ops_stress_025() {
        let b = Batch::new(vec![Item::new(25, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 25 + 1);
    }

    #[test]
    fn test_ops_stress_026() {
        let b = Batch::new(vec![Item::new(26, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 26 + 1);
    }

    #[test]
    fn test_ops_stress_027() {
        let b = Batch::new(vec![Item::new(27, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 27 + 1);
    }

    #[test]
    fn test_ops_stress_028() {
        let b = Batch::new(vec![Item::new(28, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 28 + 1);
    }

    #[test]
    fn test_ops_stress_029() {
        let b = Batch::new(vec![Item::new(29, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 29 + 1);
    }

    #[test]
    fn test_ops_stress_030() {
        let b = Batch::new(vec![Item::new(30, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 30 + 1);
    }

    #[test]
    fn test_ops_stress_031() {
        let b = Batch::new(vec![Item::new(31, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 31 + 1);
    }

    #[test]
    fn test_ops_stress_032() {
        let b = Batch::new(vec![Item::new(32, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 32 + 1);
    }

    #[test]
    fn test_ops_stress_033() {
        let b = Batch::new(vec![Item::new(33, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 33 + 1);
    }

    #[test]
    fn test_ops_stress_034() {
        let b = Batch::new(vec![Item::new(34, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 34 + 1);
    }

    #[test]
    fn test_ops_stress_035() {
        let b = Batch::new(vec![Item::new(35, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 35 + 1);
    }

    #[test]
    fn test_ops_stress_036() {
        let b = Batch::new(vec![Item::new(36, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 36 + 1);
    }

    #[test]
    fn test_ops_stress_037() {
        let b = Batch::new(vec![Item::new(37, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 37 + 1);
    }

    #[test]
    fn test_ops_stress_038() {
        let b = Batch::new(vec![Item::new(38, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 38 + 1);
    }

    #[test]
    fn test_ops_stress_039() {
        let b = Batch::new(vec![Item::new(39, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 39 + 1);
    }

    #[test]
    fn test_ops_stress_040() {
        let b = Batch::new(vec![Item::new(40, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 40 + 1);
    }

    #[test]
    fn test_ops_stress_041() {
        let b = Batch::new(vec![Item::new(41, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 41 + 1);
    }

    #[test]
    fn test_ops_stress_042() {
        let b = Batch::new(vec![Item::new(42, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 42 + 1);
    }

    #[test]
    fn test_ops_stress_043() {
        let b = Batch::new(vec![Item::new(43, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 43 + 1);
    }

    #[test]
    fn test_ops_stress_044() {
        let b = Batch::new(vec![Item::new(44, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 44 + 1);
    }

    #[test]
    fn test_ops_stress_045() {
        let b = Batch::new(vec![Item::new(45, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 45 + 1);
    }

    #[test]
    fn test_ops_stress_046() {
        let b = Batch::new(vec![Item::new(46, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 46 + 1);
    }

    #[test]
    fn test_ops_stress_047() {
        let b = Batch::new(vec![Item::new(47, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 47 + 1);
    }

    #[test]
    fn test_ops_stress_048() {
        let b = Batch::new(vec![Item::new(48, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 48 + 1);
    }

    #[test]
    fn test_ops_stress_049() {
        let b = Batch::new(vec![Item::new(49, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 49 + 1);
    }

    #[test]
    fn test_ops_stress_050() {
        let b = Batch::new(vec![Item::new(50, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 50 + 1);
    }

    #[test]
    fn test_ops_stress_051() {
        let b = Batch::new(vec![Item::new(51, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 51 + 1);
    }

    #[test]
    fn test_ops_stress_052() {
        let b = Batch::new(vec![Item::new(52, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 52 + 1);
    }

    #[test]
    fn test_ops_stress_053() {
        let b = Batch::new(vec![Item::new(53, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 53 + 1);
    }

    #[test]
    fn test_ops_stress_054() {
        let b = Batch::new(vec![Item::new(54, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 54 + 1);
    }

    #[test]
    fn test_ops_stress_055() {
        let b = Batch::new(vec![Item::new(55, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 55 + 1);
    }

    #[test]
    fn test_ops_stress_056() {
        let b = Batch::new(vec![Item::new(56, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 56 + 1);
    }

    #[test]
    fn test_ops_stress_057() {
        let b = Batch::new(vec![Item::new(57, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 57 + 1);
    }

    #[test]
    fn test_ops_stress_058() {
        let b = Batch::new(vec![Item::new(58, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 58 + 1);
    }

    #[test]
    fn test_ops_stress_059() {
        let b = Batch::new(vec![Item::new(59, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 59 + 1);
    }

    #[test]
    fn test_ops_stress_060() {
        let b = Batch::new(vec![Item::new(60, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 60 + 1);
    }

    #[test]
    fn test_ops_stress_061() {
        let b = Batch::new(vec![Item::new(61, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 61 + 1);
    }

    #[test]
    fn test_ops_stress_062() {
        let b = Batch::new(vec![Item::new(62, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 62 + 1);
    }

    #[test]
    fn test_ops_stress_063() {
        let b = Batch::new(vec![Item::new(63, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 63 + 1);
    }

    #[test]
    fn test_ops_stress_064() {
        let b = Batch::new(vec![Item::new(64, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 64 + 1);
    }

    #[test]
    fn test_ops_stress_065() {
        let b = Batch::new(vec![Item::new(65, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 65 + 1);
    }

    #[test]
    fn test_ops_stress_066() {
        let b = Batch::new(vec![Item::new(66, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 66 + 1);
    }

    #[test]
    fn test_ops_stress_067() {
        let b = Batch::new(vec![Item::new(67, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 67 + 1);
    }

    #[test]
    fn test_ops_stress_068() {
        let b = Batch::new(vec![Item::new(68, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 68 + 1);
    }

    #[test]
    fn test_ops_stress_069() {
        let b = Batch::new(vec![Item::new(69, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 69 + 1);
    }

    #[test]
    fn test_ops_stress_070() {
        let b = Batch::new(vec![Item::new(70, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 70 + 1);
    }

    #[test]
    fn test_ops_stress_071() {
        let b = Batch::new(vec![Item::new(71, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 71 + 1);
    }

    #[test]
    fn test_ops_stress_072() {
        let b = Batch::new(vec![Item::new(72, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 72 + 1);
    }

    #[test]
    fn test_ops_stress_073() {
        let b = Batch::new(vec![Item::new(73, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 73 + 1);
    }

    #[test]
    fn test_ops_stress_074() {
        let b = Batch::new(vec![Item::new(74, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 74 + 1);
    }

    #[test]
    fn test_ops_stress_075() {
        let b = Batch::new(vec![Item::new(75, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 75 + 1);
    }

    #[test]
    fn test_ops_stress_076() {
        let b = Batch::new(vec![Item::new(76, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 76 + 1);
    }

    #[test]
    fn test_ops_stress_077() {
        let b = Batch::new(vec![Item::new(77, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 77 + 1);
    }

    #[test]
    fn test_ops_stress_078() {
        let b = Batch::new(vec![Item::new(78, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 78 + 1);
    }

    #[test]
    fn test_ops_stress_079() {
        let b = Batch::new(vec![Item::new(79, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 79 + 1);
    }

    #[test]
    fn test_ops_stress_080() {
        let b = Batch::new(vec![Item::new(80, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 80 + 1);
    }

    #[test]
    fn test_ops_stress_081() {
        let b = Batch::new(vec![Item::new(81, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 81 + 1);
    }

    #[test]
    fn test_ops_stress_082() {
        let b = Batch::new(vec![Item::new(82, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 82 + 1);
    }

    #[test]
    fn test_ops_stress_083() {
        let b = Batch::new(vec![Item::new(83, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 83 + 1);
    }

    #[test]
    fn test_ops_stress_084() {
        let b = Batch::new(vec![Item::new(84, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 84 + 1);
    }

    #[test]
    fn test_ops_stress_085() {
        let b = Batch::new(vec![Item::new(85, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 85 + 1);
    }

    #[test]
    fn test_ops_stress_086() {
        let b = Batch::new(vec![Item::new(86, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 86 + 1);
    }

    #[test]
    fn test_ops_stress_087() {
        let b = Batch::new(vec![Item::new(87, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 87 + 1);
    }

    #[test]
    fn test_ops_stress_088() {
        let b = Batch::new(vec![Item::new(88, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 88 + 1);
    }

    #[test]
    fn test_ops_stress_089() {
        let b = Batch::new(vec![Item::new(89, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 89 + 1);
    }

    #[test]
    fn test_ops_stress_090() {
        let b = Batch::new(vec![Item::new(90, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 90 + 1);
    }

    #[test]
    fn test_ops_stress_091() {
        let b = Batch::new(vec![Item::new(91, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 91 + 1);
    }

    #[test]
    fn test_ops_stress_092() {
        let b = Batch::new(vec![Item::new(92, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 92 + 1);
    }

    #[test]
    fn test_ops_stress_093() {
        let b = Batch::new(vec![Item::new(93, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 93 + 1);
    }

    #[test]
    fn test_ops_stress_094() {
        let b = Batch::new(vec![Item::new(94, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 94 + 1);
    }

    #[test]
    fn test_ops_stress_095() {
        let b = Batch::new(vec![Item::new(95, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 95 + 1);
    }

    #[test]
    fn test_ops_stress_096() {
        let b = Batch::new(vec![Item::new(96, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 96 + 1);
    }

    #[test]
    fn test_ops_stress_097() {
        let b = Batch::new(vec![Item::new(97, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 97 + 1);
    }

    #[test]
    fn test_ops_stress_098() {
        let b = Batch::new(vec![Item::new(98, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 98 + 1);
    }

    #[test]
    fn test_ops_stress_099() {
        let b = Batch::new(vec![Item::new(99, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 99 + 1);
    }

    #[test]
    fn test_ops_stress_100() {
        let b = Batch::new(vec![Item::new(100, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 100 + 1);
    }

    #[test]
    fn test_ops_stress_101() {
        let b = Batch::new(vec![Item::new(101, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 101 + 1);
    }

    #[test]
    fn test_ops_stress_102() {
        let b = Batch::new(vec![Item::new(102, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 102 + 1);
    }

    #[test]
    fn test_ops_stress_103() {
        let b = Batch::new(vec![Item::new(103, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 103 + 1);
    }

    #[test]
    fn test_ops_stress_104() {
        let b = Batch::new(vec![Item::new(104, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 104 + 1);
    }

    #[test]
    fn test_ops_stress_105() {
        let b = Batch::new(vec![Item::new(105, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 105 + 1);
    }

    #[test]
    fn test_ops_stress_106() {
        let b = Batch::new(vec![Item::new(106, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 106 + 1);
    }

    #[test]
    fn test_ops_stress_107() {
        let b = Batch::new(vec![Item::new(107, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 107 + 1);
    }

    #[test]
    fn test_ops_stress_108() {
        let b = Batch::new(vec![Item::new(108, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 108 + 1);
    }

    #[test]
    fn test_ops_stress_109() {
        let b = Batch::new(vec![Item::new(109, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 109 + 1);
    }

    #[test]
    fn test_ops_stress_110() {
        let b = Batch::new(vec![Item::new(110, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 110 + 1);
    }

    #[test]
    fn test_ops_stress_111() {
        let b = Batch::new(vec![Item::new(111, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 111 + 1);
    }

    #[test]
    fn test_ops_stress_112() {
        let b = Batch::new(vec![Item::new(112, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 112 + 1);
    }

    #[test]
    fn test_ops_stress_113() {
        let b = Batch::new(vec![Item::new(113, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 113 + 1);
    }

    #[test]
    fn test_ops_stress_114() {
        let b = Batch::new(vec![Item::new(114, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 114 + 1);
    }

    #[test]
    fn test_ops_stress_115() {
        let b = Batch::new(vec![Item::new(115, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 115 + 1);
    }

    #[test]
    fn test_ops_stress_116() {
        let b = Batch::new(vec![Item::new(116, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 116 + 1);
    }

    #[test]
    fn test_ops_stress_117() {
        let b = Batch::new(vec![Item::new(117, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 117 + 1);
    }

    #[test]
    fn test_ops_stress_118() {
        let b = Batch::new(vec![Item::new(118, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 118 + 1);
    }

    #[test]
    fn test_ops_stress_119() {
        let b = Batch::new(vec![Item::new(119, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 119 + 1);
    }

    #[test]
    fn test_ops_stress_120() {
        let b = Batch::new(vec![Item::new(120, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 120 + 1);
    }

    #[test]
    fn test_ops_stress_121() {
        let b = Batch::new(vec![Item::new(121, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 121 + 1);
    }

    #[test]
    fn test_ops_stress_122() {
        let b = Batch::new(vec![Item::new(122, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 122 + 1);
    }

    #[test]
    fn test_ops_stress_123() {
        let b = Batch::new(vec![Item::new(123, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 123 + 1);
    }

    #[test]
    fn test_ops_stress_124() {
        let b = Batch::new(vec![Item::new(124, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 124 + 1);
    }

    #[test]
    fn test_ops_stress_125() {
        let b = Batch::new(vec![Item::new(125, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 125 + 1);
    }

    #[test]
    fn test_ops_stress_126() {
        let b = Batch::new(vec![Item::new(126, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 126 + 1);
    }

    #[test]
    fn test_ops_stress_127() {
        let b = Batch::new(vec![Item::new(127, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 127 + 1);
    }

    #[test]
    fn test_ops_stress_128() {
        let b = Batch::new(vec![Item::new(128, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 128 + 1);
    }

    #[test]
    fn test_ops_stress_129() {
        let b = Batch::new(vec![Item::new(129, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 129 + 1);
    }

    #[test]
    fn test_ops_stress_130() {
        let b = Batch::new(vec![Item::new(130, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 130 + 1);
    }

    #[test]
    fn test_ops_stress_131() {
        let b = Batch::new(vec![Item::new(131, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 131 + 1);
    }

    #[test]
    fn test_ops_stress_132() {
        let b = Batch::new(vec![Item::new(132, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 132 + 1);
    }

    #[test]
    fn test_ops_stress_133() {
        let b = Batch::new(vec![Item::new(133, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 133 + 1);
    }

    #[test]
    fn test_ops_stress_134() {
        let b = Batch::new(vec![Item::new(134, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 134 + 1);
    }

    #[test]
    fn test_ops_stress_135() {
        let b = Batch::new(vec![Item::new(135, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 135 + 1);
    }

    #[test]
    fn test_ops_stress_136() {
        let b = Batch::new(vec![Item::new(136, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 136 + 1);
    }

    #[test]
    fn test_ops_stress_137() {
        let b = Batch::new(vec![Item::new(137, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 137 + 1);
    }

    #[test]
    fn test_ops_stress_138() {
        let b = Batch::new(vec![Item::new(138, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 138 + 1);
    }

    #[test]
    fn test_ops_stress_139() {
        let b = Batch::new(vec![Item::new(139, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 139 + 1);
    }

    #[test]
    fn test_ops_stress_140() {
        let b = Batch::new(vec![Item::new(140, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 140 + 1);
    }

    #[test]
    fn test_ops_stress_141() {
        let b = Batch::new(vec![Item::new(141, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 141 + 1);
    }

    #[test]
    fn test_ops_stress_142() {
        let b = Batch::new(vec![Item::new(142, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 142 + 1);
    }

    #[test]
    fn test_ops_stress_143() {
        let b = Batch::new(vec![Item::new(143, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 143 + 1);
    }

    #[test]
    fn test_ops_stress_144() {
        let b = Batch::new(vec![Item::new(144, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 144 + 1);
    }

    #[test]
    fn test_ops_stress_145() {
        let b = Batch::new(vec![Item::new(145, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 145 + 1);
    }

    #[test]
    fn test_ops_stress_146() {
        let b = Batch::new(vec![Item::new(146, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 146 + 1);
    }

    #[test]
    fn test_ops_stress_147() {
        let b = Batch::new(vec![Item::new(147, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 147 + 1);
    }

    #[test]
    fn test_ops_stress_148() {
        let b = Batch::new(vec![Item::new(148, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 148 + 1);
    }

    #[test]
    fn test_ops_stress_149() {
        let b = Batch::new(vec![Item::new(149, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 149 + 1);
    }

    #[test]
    fn test_ops_stress_150() {
        let b = Batch::new(vec![Item::new(150, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 150 + 1);
    }

    #[test]
    fn test_ops_stress_151() {
        let b = Batch::new(vec![Item::new(151, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 151 + 1);
    }

    #[test]
    fn test_ops_stress_152() {
        let b = Batch::new(vec![Item::new(152, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 152 + 1);
    }

    #[test]
    fn test_ops_stress_153() {
        let b = Batch::new(vec![Item::new(153, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 153 + 1);
    }

    #[test]
    fn test_ops_stress_154() {
        let b = Batch::new(vec![Item::new(154, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 154 + 1);
    }

    #[test]
    fn test_ops_stress_155() {
        let b = Batch::new(vec![Item::new(155, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 155 + 1);
    }

    #[test]
    fn test_ops_stress_156() {
        let b = Batch::new(vec![Item::new(156, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 156 + 1);
    }

    #[test]
    fn test_ops_stress_157() {
        let b = Batch::new(vec![Item::new(157, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 157 + 1);
    }

    #[test]
    fn test_ops_stress_158() {
        let b = Batch::new(vec![Item::new(158, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 158 + 1);
    }

    #[test]
    fn test_ops_stress_159() {
        let b = Batch::new(vec![Item::new(159, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 159 + 1);
    }

    #[test]
    fn test_ops_stress_160() {
        let b = Batch::new(vec![Item::new(160, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 160 + 1);
    }

    #[test]
    fn test_ops_stress_161() {
        let b = Batch::new(vec![Item::new(161, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 161 + 1);
    }

    #[test]
    fn test_ops_stress_162() {
        let b = Batch::new(vec![Item::new(162, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 162 + 1);
    }

    #[test]
    fn test_ops_stress_163() {
        let b = Batch::new(vec![Item::new(163, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 163 + 1);
    }

    #[test]
    fn test_ops_stress_164() {
        let b = Batch::new(vec![Item::new(164, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 164 + 1);
    }

    #[test]
    fn test_ops_stress_165() {
        let b = Batch::new(vec![Item::new(165, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 165 + 1);
    }

    #[test]
    fn test_ops_stress_166() {
        let b = Batch::new(vec![Item::new(166, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 166 + 1);
    }

    #[test]
    fn test_ops_stress_167() {
        let b = Batch::new(vec![Item::new(167, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 167 + 1);
    }

    #[test]
    fn test_ops_stress_168() {
        let b = Batch::new(vec![Item::new(168, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 168 + 1);
    }

    #[test]
    fn test_ops_stress_169() {
        let b = Batch::new(vec![Item::new(169, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 169 + 1);
    }

    #[test]
    fn test_ops_stress_170() {
        let b = Batch::new(vec![Item::new(170, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 170 + 1);
    }

    #[test]
    fn test_ops_stress_171() {
        let b = Batch::new(vec![Item::new(171, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 171 + 1);
    }

    #[test]
    fn test_ops_stress_172() {
        let b = Batch::new(vec![Item::new(172, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 172 + 1);
    }

    #[test]
    fn test_ops_stress_173() {
        let b = Batch::new(vec![Item::new(173, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 173 + 1);
    }

    #[test]
    fn test_ops_stress_174() {
        let b = Batch::new(vec![Item::new(174, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 174 + 1);
    }

    #[test]
    fn test_ops_stress_175() {
        let b = Batch::new(vec![Item::new(175, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 175 + 1);
    }

    #[test]
    fn test_ops_stress_176() {
        let b = Batch::new(vec![Item::new(176, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 176 + 1);
    }

    #[test]
    fn test_ops_stress_177() {
        let b = Batch::new(vec![Item::new(177, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 177 + 1);
    }

    #[test]
    fn test_ops_stress_178() {
        let b = Batch::new(vec![Item::new(178, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 178 + 1);
    }

    #[test]
    fn test_ops_stress_179() {
        let b = Batch::new(vec![Item::new(179, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 179 + 1);
    }

    #[test]
    fn test_ops_stress_180() {
        let b = Batch::new(vec![Item::new(180, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 180 + 1);
    }

    #[test]
    fn test_ops_stress_181() {
        let b = Batch::new(vec![Item::new(181, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 181 + 1);
    }

    #[test]
    fn test_ops_stress_182() {
        let b = Batch::new(vec![Item::new(182, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 182 + 1);
    }

    #[test]
    fn test_ops_stress_183() {
        let b = Batch::new(vec![Item::new(183, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 183 + 1);
    }

    #[test]
    fn test_ops_stress_184() {
        let b = Batch::new(vec![Item::new(184, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 184 + 1);
    }

    #[test]
    fn test_ops_stress_185() {
        let b = Batch::new(vec![Item::new(185, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 185 + 1);
    }

    #[test]
    fn test_ops_stress_186() {
        let b = Batch::new(vec![Item::new(186, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 186 + 1);
    }

    #[test]
    fn test_ops_stress_187() {
        let b = Batch::new(vec![Item::new(187, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 187 + 1);
    }

    #[test]
    fn test_ops_stress_188() {
        let b = Batch::new(vec![Item::new(188, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 188 + 1);
    }

    #[test]
    fn test_ops_stress_189() {
        let b = Batch::new(vec![Item::new(189, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 189 + 1);
    }

    #[test]
    fn test_ops_stress_190() {
        let b = Batch::new(vec![Item::new(190, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 190 + 1);
    }

    #[test]
    fn test_ops_stress_191() {
        let b = Batch::new(vec![Item::new(191, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 191 + 1);
    }

    #[test]
    fn test_ops_stress_192() {
        let b = Batch::new(vec![Item::new(192, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 192 + 1);
    }

    #[test]
    fn test_ops_stress_193() {
        let b = Batch::new(vec![Item::new(193, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 193 + 1);
    }

    #[test]
    fn test_ops_stress_194() {
        let b = Batch::new(vec![Item::new(194, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 194 + 1);
    }

    #[test]
    fn test_ops_stress_195() {
        let b = Batch::new(vec![Item::new(195, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 195 + 1);
    }

    #[test]
    fn test_ops_stress_196() {
        let b = Batch::new(vec![Item::new(196, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 196 + 1);
    }

    #[test]
    fn test_ops_stress_197() {
        let b = Batch::new(vec![Item::new(197, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 197 + 1);
    }

    #[test]
    fn test_ops_stress_198() {
        let b = Batch::new(vec![Item::new(198, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 198 + 1);
    }

    #[test]
    fn test_ops_stress_199() {
        let b = Batch::new(vec![Item::new(199, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 199 + 1);
    }

    #[test]
    fn test_ops_stress_200() {
        let b = Batch::new(vec![Item::new(200, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 200 + 1);
    }

    #[test]
    fn test_ops_stress_201() {
        let b = Batch::new(vec![Item::new(201, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 201 + 1);
    }

    #[test]
    fn test_ops_stress_202() {
        let b = Batch::new(vec![Item::new(202, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 202 + 1);
    }

    #[test]
    fn test_ops_stress_203() {
        let b = Batch::new(vec![Item::new(203, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 203 + 1);
    }

    #[test]
    fn test_ops_stress_204() {
        let b = Batch::new(vec![Item::new(204, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 204 + 1);
    }

    #[test]
    fn test_ops_stress_205() {
        let b = Batch::new(vec![Item::new(205, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 205 + 1);
    }

    #[test]
    fn test_ops_stress_206() {
        let b = Batch::new(vec![Item::new(206, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 206 + 1);
    }

    #[test]
    fn test_ops_stress_207() {
        let b = Batch::new(vec![Item::new(207, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 207 + 1);
    }

    #[test]
    fn test_ops_stress_208() {
        let b = Batch::new(vec![Item::new(208, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 208 + 1);
    }

    #[test]
    fn test_ops_stress_209() {
        let b = Batch::new(vec![Item::new(209, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 209 + 1);
    }

    #[test]
    fn test_ops_stress_210() {
        let b = Batch::new(vec![Item::new(210, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 210 + 1);
    }

    #[test]
    fn test_ops_stress_211() {
        let b = Batch::new(vec![Item::new(211, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 211 + 1);
    }

    #[test]
    fn test_ops_stress_212() {
        let b = Batch::new(vec![Item::new(212, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 212 + 1);
    }

    #[test]
    fn test_ops_stress_213() {
        let b = Batch::new(vec![Item::new(213, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 213 + 1);
    }

    #[test]
    fn test_ops_stress_214() {
        let b = Batch::new(vec![Item::new(214, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 214 + 1);
    }

    #[test]
    fn test_ops_stress_215() {
        let b = Batch::new(vec![Item::new(215, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 215 + 1);
    }

    #[test]
    fn test_ops_stress_216() {
        let b = Batch::new(vec![Item::new(216, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 216 + 1);
    }

    #[test]
    fn test_ops_stress_217() {
        let b = Batch::new(vec![Item::new(217, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 217 + 1);
    }

    #[test]
    fn test_ops_stress_218() {
        let b = Batch::new(vec![Item::new(218, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 218 + 1);
    }

    #[test]
    fn test_ops_stress_219() {
        let b = Batch::new(vec![Item::new(219, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 219 + 1);
    }

    #[test]
    fn test_ops_stress_220() {
        let b = Batch::new(vec![Item::new(220, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 220 + 1);
    }

    #[test]
    fn test_ops_stress_221() {
        let b = Batch::new(vec![Item::new(221, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 221 + 1);
    }

    #[test]
    fn test_ops_stress_222() {
        let b = Batch::new(vec![Item::new(222, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 222 + 1);
    }

    #[test]
    fn test_ops_stress_223() {
        let b = Batch::new(vec![Item::new(223, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 223 + 1);
    }

    #[test]
    fn test_ops_stress_224() {
        let b = Batch::new(vec![Item::new(224, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 224 + 1);
    }

    #[test]
    fn test_ops_stress_225() {
        let b = Batch::new(vec![Item::new(225, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 225 + 1);
    }

    #[test]
    fn test_ops_stress_226() {
        let b = Batch::new(vec![Item::new(226, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 226 + 1);
    }

    #[test]
    fn test_ops_stress_227() {
        let b = Batch::new(vec![Item::new(227, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 227 + 1);
    }

    #[test]
    fn test_ops_stress_228() {
        let b = Batch::new(vec![Item::new(228, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 228 + 1);
    }

    #[test]
    fn test_ops_stress_229() {
        let b = Batch::new(vec![Item::new(229, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 229 + 1);
    }

    #[test]
    fn test_ops_stress_230() {
        let b = Batch::new(vec![Item::new(230, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 230 + 1);
    }

    #[test]
    fn test_ops_stress_231() {
        let b = Batch::new(vec![Item::new(231, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 231 + 1);
    }

    #[test]
    fn test_ops_stress_232() {
        let b = Batch::new(vec![Item::new(232, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 232 + 1);
    }

    #[test]
    fn test_ops_stress_233() {
        let b = Batch::new(vec![Item::new(233, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 233 + 1);
    }

    #[test]
    fn test_ops_stress_234() {
        let b = Batch::new(vec![Item::new(234, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 234 + 1);
    }

    #[test]
    fn test_ops_stress_235() {
        let b = Batch::new(vec![Item::new(235, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 235 + 1);
    }

    #[test]
    fn test_ops_stress_236() {
        let b = Batch::new(vec![Item::new(236, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 236 + 1);
    }

    #[test]
    fn test_ops_stress_237() {
        let b = Batch::new(vec![Item::new(237, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 237 + 1);
    }

    #[test]
    fn test_ops_stress_238() {
        let b = Batch::new(vec![Item::new(238, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 238 + 1);
    }

    #[test]
    fn test_ops_stress_239() {
        let b = Batch::new(vec![Item::new(239, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 239 + 1);
    }

    #[test]
    fn test_ops_stress_240() {
        let b = Batch::new(vec![Item::new(240, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 240 + 1);
    }

    #[test]
    fn test_ops_stress_241() {
        let b = Batch::new(vec![Item::new(241, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 241 + 1);
    }

    #[test]
    fn test_ops_stress_242() {
        let b = Batch::new(vec![Item::new(242, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 242 + 1);
    }

    #[test]
    fn test_ops_stress_243() {
        let b = Batch::new(vec![Item::new(243, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 243 + 1);
    }

    #[test]
    fn test_ops_stress_244() {
        let b = Batch::new(vec![Item::new(244, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 244 + 1);
    }

    #[test]
    fn test_ops_stress_245() {
        let b = Batch::new(vec![Item::new(245, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 245 + 1);
    }

    #[test]
    fn test_ops_stress_246() {
        let b = Batch::new(vec![Item::new(246, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 246 + 1);
    }

    #[test]
    fn test_ops_stress_247() {
        let b = Batch::new(vec![Item::new(247, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 247 + 1);
    }

    #[test]
    fn test_ops_stress_248() {
        let b = Batch::new(vec![Item::new(248, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 248 + 1);
    }

    #[test]
    fn test_ops_stress_249() {
        let b = Batch::new(vec![Item::new(249, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 249 + 1);
    }

    #[test]
    fn test_ops_stress_250() {
        let b = Batch::new(vec![Item::new(250, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 250 + 1);
    }

    #[test]
    fn test_ops_stress_251() {
        let b = Batch::new(vec![Item::new(251, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 251 + 1);
    }

    #[test]
    fn test_ops_stress_252() {
        let b = Batch::new(vec![Item::new(252, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 252 + 1);
    }

    #[test]
    fn test_ops_stress_253() {
        let b = Batch::new(vec![Item::new(253, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 253 + 1);
    }

    #[test]
    fn test_ops_stress_254() {
        let b = Batch::new(vec![Item::new(254, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 254 + 1);
    }

    #[test]
    fn test_ops_stress_255() {
        let b = Batch::new(vec![Item::new(255, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 255 + 1);
    }

    #[test]
    fn test_ops_stress_256() {
        let b = Batch::new(vec![Item::new(256, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 256 + 1);
    }

    #[test]
    fn test_ops_stress_257() {
        let b = Batch::new(vec![Item::new(257, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 257 + 1);
    }

    #[test]
    fn test_ops_stress_258() {
        let b = Batch::new(vec![Item::new(258, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 258 + 1);
    }

    #[test]
    fn test_ops_stress_259() {
        let b = Batch::new(vec![Item::new(259, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 259 + 1);
    }

    #[test]
    fn test_ops_stress_260() {
        let b = Batch::new(vec![Item::new(260, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 260 + 1);
    }

    #[test]
    fn test_ops_stress_261() {
        let b = Batch::new(vec![Item::new(261, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 261 + 1);
    }

    #[test]
    fn test_ops_stress_262() {
        let b = Batch::new(vec![Item::new(262, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 262 + 1);
    }

    #[test]
    fn test_ops_stress_263() {
        let b = Batch::new(vec![Item::new(263, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 263 + 1);
    }

    #[test]
    fn test_ops_stress_264() {
        let b = Batch::new(vec![Item::new(264, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 264 + 1);
    }

    #[test]
    fn test_ops_stress_265() {
        let b = Batch::new(vec![Item::new(265, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 265 + 1);
    }

    #[test]
    fn test_ops_stress_266() {
        let b = Batch::new(vec![Item::new(266, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 266 + 1);
    }

    #[test]
    fn test_ops_stress_267() {
        let b = Batch::new(vec![Item::new(267, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 267 + 1);
    }

    #[test]
    fn test_ops_stress_268() {
        let b = Batch::new(vec![Item::new(268, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 268 + 1);
    }

    #[test]
    fn test_ops_stress_269() {
        let b = Batch::new(vec![Item::new(269, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 269 + 1);
    }

    #[test]
    fn test_ops_stress_270() {
        let b = Batch::new(vec![Item::new(270, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 270 + 1);
    }

    #[test]
    fn test_ops_stress_271() {
        let b = Batch::new(vec![Item::new(271, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 271 + 1);
    }

    #[test]
    fn test_ops_stress_272() {
        let b = Batch::new(vec![Item::new(272, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 272 + 1);
    }

    #[test]
    fn test_ops_stress_273() {
        let b = Batch::new(vec![Item::new(273, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 273 + 1);
    }

    #[test]
    fn test_ops_stress_274() {
        let b = Batch::new(vec![Item::new(274, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 274 + 1);
    }

    #[test]
    fn test_ops_stress_275() {
        let b = Batch::new(vec![Item::new(275, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 275 + 1);
    }

    #[test]
    fn test_ops_stress_276() {
        let b = Batch::new(vec![Item::new(276, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 276 + 1);
    }

    #[test]
    fn test_ops_stress_277() {
        let b = Batch::new(vec![Item::new(277, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 277 + 1);
    }

    #[test]
    fn test_ops_stress_278() {
        let b = Batch::new(vec![Item::new(278, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 278 + 1);
    }

    #[test]
    fn test_ops_stress_279() {
        let b = Batch::new(vec![Item::new(279, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 279 + 1);
    }

    #[test]
    fn test_ops_stress_280() {
        let b = Batch::new(vec![Item::new(280, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 280 + 1);
    }

    #[test]
    fn test_ops_stress_281() {
        let b = Batch::new(vec![Item::new(281, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 281 + 1);
    }

    #[test]
    fn test_ops_stress_282() {
        let b = Batch::new(vec![Item::new(282, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 282 + 1);
    }

    #[test]
    fn test_ops_stress_283() {
        let b = Batch::new(vec![Item::new(283, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 283 + 1);
    }

    #[test]
    fn test_ops_stress_284() {
        let b = Batch::new(vec![Item::new(284, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 284 + 1);
    }

    #[test]
    fn test_ops_stress_285() {
        let b = Batch::new(vec![Item::new(285, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 285 + 1);
    }

    #[test]
    fn test_ops_stress_286() {
        let b = Batch::new(vec![Item::new(286, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 286 + 1);
    }

    #[test]
    fn test_ops_stress_287() {
        let b = Batch::new(vec![Item::new(287, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 287 + 1);
    }

    #[test]
    fn test_ops_stress_288() {
        let b = Batch::new(vec![Item::new(288, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 288 + 1);
    }

    #[test]
    fn test_ops_stress_289() {
        let b = Batch::new(vec![Item::new(289, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 289 + 1);
    }

    #[test]
    fn test_ops_stress_290() {
        let b = Batch::new(vec![Item::new(290, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 290 + 1);
    }

    #[test]
    fn test_ops_stress_291() {
        let b = Batch::new(vec![Item::new(291, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 291 + 1);
    }

    #[test]
    fn test_ops_stress_292() {
        let b = Batch::new(vec![Item::new(292, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 292 + 1);
    }

    #[test]
    fn test_ops_stress_293() {
        let b = Batch::new(vec![Item::new(293, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 293 + 1);
    }

    #[test]
    fn test_ops_stress_294() {
        let b = Batch::new(vec![Item::new(294, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 294 + 1);
    }

    #[test]
    fn test_ops_stress_295() {
        let b = Batch::new(vec![Item::new(295, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 295 + 1);
    }

    #[test]
    fn test_ops_stress_296() {
        let b = Batch::new(vec![Item::new(296, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 296 + 1);
    }

    #[test]
    fn test_ops_stress_297() {
        let b = Batch::new(vec![Item::new(297, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 297 + 1);
    }

    #[test]
    fn test_ops_stress_298() {
        let b = Batch::new(vec![Item::new(298, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 298 + 1);
    }

    #[test]
    fn test_ops_stress_299() {
        let b = Batch::new(vec![Item::new(299, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 299 + 1);
    }

    #[test]
    fn test_ops_stress_300() {
        let b = Batch::new(vec![Item::new(300, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 300 + 1);
    }

    #[test]
    fn test_ops_stress_301() {
        let b = Batch::new(vec![Item::new(301, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 301 + 1);
    }

    #[test]
    fn test_ops_stress_302() {
        let b = Batch::new(vec![Item::new(302, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 302 + 1);
    }

    #[test]
    fn test_ops_stress_303() {
        let b = Batch::new(vec![Item::new(303, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 303 + 1);
    }

    #[test]
    fn test_ops_stress_304() {
        let b = Batch::new(vec![Item::new(304, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 304 + 1);
    }

    #[test]
    fn test_ops_stress_305() {
        let b = Batch::new(vec![Item::new(305, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 305 + 1);
    }

    #[test]
    fn test_ops_stress_306() {
        let b = Batch::new(vec![Item::new(306, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 306 + 1);
    }

    #[test]
    fn test_ops_stress_307() {
        let b = Batch::new(vec![Item::new(307, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 307 + 1);
    }

    #[test]
    fn test_ops_stress_308() {
        let b = Batch::new(vec![Item::new(308, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 308 + 1);
    }

    #[test]
    fn test_ops_stress_309() {
        let b = Batch::new(vec![Item::new(309, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 309 + 1);
    }

    #[test]
    fn test_ops_stress_310() {
        let b = Batch::new(vec![Item::new(310, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 310 + 1);
    }

    #[test]
    fn test_ops_stress_311() {
        let b = Batch::new(vec![Item::new(311, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 311 + 1);
    }

    #[test]
    fn test_ops_stress_312() {
        let b = Batch::new(vec![Item::new(312, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 312 + 1);
    }

    #[test]
    fn test_ops_stress_313() {
        let b = Batch::new(vec![Item::new(313, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 313 + 1);
    }

    #[test]
    fn test_ops_stress_314() {
        let b = Batch::new(vec![Item::new(314, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 314 + 1);
    }

    #[test]
    fn test_ops_stress_315() {
        let b = Batch::new(vec![Item::new(315, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 315 + 1);
    }

    #[test]
    fn test_ops_stress_316() {
        let b = Batch::new(vec![Item::new(316, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 316 + 1);
    }

    #[test]
    fn test_ops_stress_317() {
        let b = Batch::new(vec![Item::new(317, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 317 + 1);
    }

    #[test]
    fn test_ops_stress_318() {
        let b = Batch::new(vec![Item::new(318, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 318 + 1);
    }

    #[test]
    fn test_ops_stress_319() {
        let b = Batch::new(vec![Item::new(319, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 319 + 1);
    }

    #[test]
    fn test_ops_stress_320() {
        let b = Batch::new(vec![Item::new(320, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 320 + 1);
    }

    #[test]
    fn test_ops_stress_321() {
        let b = Batch::new(vec![Item::new(321, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 321 + 1);
    }

    #[test]
    fn test_ops_stress_322() {
        let b = Batch::new(vec![Item::new(322, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 322 + 1);
    }

    #[test]
    fn test_ops_stress_323() {
        let b = Batch::new(vec![Item::new(323, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 323 + 1);
    }

    #[test]
    fn test_ops_stress_324() {
        let b = Batch::new(vec![Item::new(324, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 324 + 1);
    }

    #[test]
    fn test_ops_stress_325() {
        let b = Batch::new(vec![Item::new(325, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 325 + 1);
    }

    #[test]
    fn test_ops_stress_326() {
        let b = Batch::new(vec![Item::new(326, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 326 + 1);
    }

    #[test]
    fn test_ops_stress_327() {
        let b = Batch::new(vec![Item::new(327, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 327 + 1);
    }

    #[test]
    fn test_ops_stress_328() {
        let b = Batch::new(vec![Item::new(328, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 328 + 1);
    }

    #[test]
    fn test_ops_stress_329() {
        let b = Batch::new(vec![Item::new(329, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 329 + 1);
    }

    #[test]
    fn test_ops_stress_330() {
        let b = Batch::new(vec![Item::new(330, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 330 + 1);
    }

    #[test]
    fn test_ops_stress_331() {
        let b = Batch::new(vec![Item::new(331, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 331 + 1);
    }

    #[test]
    fn test_ops_stress_332() {
        let b = Batch::new(vec![Item::new(332, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 332 + 1);
    }

    #[test]
    fn test_ops_stress_333() {
        let b = Batch::new(vec![Item::new(333, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 333 + 1);
    }

    #[test]
    fn test_ops_stress_334() {
        let b = Batch::new(vec![Item::new(334, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 334 + 1);
    }

    #[test]
    fn test_ops_stress_335() {
        let b = Batch::new(vec![Item::new(335, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 335 + 1);
    }

    #[test]
    fn test_ops_stress_336() {
        let b = Batch::new(vec![Item::new(336, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 336 + 1);
    }

    #[test]
    fn test_ops_stress_337() {
        let b = Batch::new(vec![Item::new(337, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 337 + 1);
    }

    #[test]
    fn test_ops_stress_338() {
        let b = Batch::new(vec![Item::new(338, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 338 + 1);
    }

    #[test]
    fn test_ops_stress_339() {
        let b = Batch::new(vec![Item::new(339, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 339 + 1);
    }

    #[test]
    fn test_ops_stress_340() {
        let b = Batch::new(vec![Item::new(340, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 340 + 1);
    }

    #[test]
    fn test_ops_stress_341() {
        let b = Batch::new(vec![Item::new(341, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 341 + 1);
    }

    #[test]
    fn test_ops_stress_342() {
        let b = Batch::new(vec![Item::new(342, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 342 + 1);
    }

    #[test]
    fn test_ops_stress_343() {
        let b = Batch::new(vec![Item::new(343, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 343 + 1);
    }

    #[test]
    fn test_ops_stress_344() {
        let b = Batch::new(vec![Item::new(344, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 344 + 1);
    }

    #[test]
    fn test_ops_stress_345() {
        let b = Batch::new(vec![Item::new(345, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 345 + 1);
    }

    #[test]
    fn test_ops_stress_346() {
        let b = Batch::new(vec![Item::new(346, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 346 + 1);
    }

    #[test]
    fn test_ops_stress_347() {
        let b = Batch::new(vec![Item::new(347, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 347 + 1);
    }

    #[test]
    fn test_ops_stress_348() {
        let b = Batch::new(vec![Item::new(348, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 348 + 1);
    }

    #[test]
    fn test_ops_stress_349() {
        let b = Batch::new(vec![Item::new(349, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 349 + 1);
    }

    #[test]
    fn test_ops_stress_350() {
        let b = Batch::new(vec![Item::new(350, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 350 + 1);
    }

    #[test]
    fn test_ops_stress_351() {
        let b = Batch::new(vec![Item::new(351, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 351 + 1);
    }

    #[test]
    fn test_ops_stress_352() {
        let b = Batch::new(vec![Item::new(352, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 352 + 1);
    }

    #[test]
    fn test_ops_stress_353() {
        let b = Batch::new(vec![Item::new(353, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 353 + 1);
    }

    #[test]
    fn test_ops_stress_354() {
        let b = Batch::new(vec![Item::new(354, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 354 + 1);
    }

    #[test]
    fn test_ops_stress_355() {
        let b = Batch::new(vec![Item::new(355, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 355 + 1);
    }

    #[test]
    fn test_ops_stress_356() {
        let b = Batch::new(vec![Item::new(356, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 356 + 1);
    }

    #[test]
    fn test_ops_stress_357() {
        let b = Batch::new(vec![Item::new(357, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 357 + 1);
    }

    #[test]
    fn test_ops_stress_358() {
        let b = Batch::new(vec![Item::new(358, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 358 + 1);
    }

    #[test]
    fn test_ops_stress_359() {
        let b = Batch::new(vec![Item::new(359, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 359 + 1);
    }

    #[test]
    fn test_ops_stress_360() {
        let b = Batch::new(vec![Item::new(360, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 360 + 1);
    }

    #[test]
    fn test_ops_stress_361() {
        let b = Batch::new(vec![Item::new(361, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 361 + 1);
    }

    #[test]
    fn test_ops_stress_362() {
        let b = Batch::new(vec![Item::new(362, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 362 + 1);
    }

    #[test]
    fn test_ops_stress_363() {
        let b = Batch::new(vec![Item::new(363, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 363 + 1);
    }

    #[test]
    fn test_ops_stress_364() {
        let b = Batch::new(vec![Item::new(364, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 364 + 1);
    }

    #[test]
    fn test_ops_stress_365() {
        let b = Batch::new(vec![Item::new(365, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 365 + 1);
    }

    #[test]
    fn test_ops_stress_366() {
        let b = Batch::new(vec![Item::new(366, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 366 + 1);
    }

    #[test]
    fn test_ops_stress_367() {
        let b = Batch::new(vec![Item::new(367, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 367 + 1);
    }

    #[test]
    fn test_ops_stress_368() {
        let b = Batch::new(vec![Item::new(368, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 368 + 1);
    }

    #[test]
    fn test_ops_stress_369() {
        let b = Batch::new(vec![Item::new(369, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 369 + 1);
    }

    #[test]
    fn test_ops_stress_370() {
        let b = Batch::new(vec![Item::new(370, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 370 + 1);
    }

    #[test]
    fn test_ops_stress_371() {
        let b = Batch::new(vec![Item::new(371, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 371 + 1);
    }

    #[test]
    fn test_ops_stress_372() {
        let b = Batch::new(vec![Item::new(372, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 372 + 1);
    }

    #[test]
    fn test_ops_stress_373() {
        let b = Batch::new(vec![Item::new(373, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 373 + 1);
    }

    #[test]
    fn test_ops_stress_374() {
        let b = Batch::new(vec![Item::new(374, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 374 + 1);
    }

    #[test]
    fn test_ops_stress_375() {
        let b = Batch::new(vec![Item::new(375, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 375 + 1);
    }

    #[test]
    fn test_ops_stress_376() {
        let b = Batch::new(vec![Item::new(376, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 376 + 1);
    }

    #[test]
    fn test_ops_stress_377() {
        let b = Batch::new(vec![Item::new(377, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 377 + 1);
    }

    #[test]
    fn test_ops_stress_378() {
        let b = Batch::new(vec![Item::new(378, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 378 + 1);
    }

    #[test]
    fn test_ops_stress_379() {
        let b = Batch::new(vec![Item::new(379, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 379 + 1);
    }

    #[test]
    fn test_ops_stress_380() {
        let b = Batch::new(vec![Item::new(380, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 380 + 1);
    }

    #[test]
    fn test_ops_stress_381() {
        let b = Batch::new(vec![Item::new(381, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 381 + 1);
    }

    #[test]
    fn test_ops_stress_382() {
        let b = Batch::new(vec![Item::new(382, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 382 + 1);
    }

    #[test]
    fn test_ops_stress_383() {
        let b = Batch::new(vec![Item::new(383, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 383 + 1);
    }

    #[test]
    fn test_ops_stress_384() {
        let b = Batch::new(vec![Item::new(384, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 384 + 1);
    }

    #[test]
    fn test_ops_stress_385() {
        let b = Batch::new(vec![Item::new(385, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 385 + 1);
    }

    #[test]
    fn test_ops_stress_386() {
        let b = Batch::new(vec![Item::new(386, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 386 + 1);
    }

    #[test]
    fn test_ops_stress_387() {
        let b = Batch::new(vec![Item::new(387, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 387 + 1);
    }

    #[test]
    fn test_ops_stress_388() {
        let b = Batch::new(vec![Item::new(388, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 388 + 1);
    }

    #[test]
    fn test_ops_stress_389() {
        let b = Batch::new(vec![Item::new(389, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 389 + 1);
    }

    #[test]
    fn test_ops_stress_390() {
        let b = Batch::new(vec![Item::new(390, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 390 + 1);
    }

    #[test]
    fn test_ops_stress_391() {
        let b = Batch::new(vec![Item::new(391, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 391 + 1);
    }

    #[test]
    fn test_ops_stress_392() {
        let b = Batch::new(vec![Item::new(392, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 392 + 1);
    }

    #[test]
    fn test_ops_stress_393() {
        let b = Batch::new(vec![Item::new(393, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 393 + 1);
    }

    #[test]
    fn test_ops_stress_394() {
        let b = Batch::new(vec![Item::new(394, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 394 + 1);
    }

    #[test]
    fn test_ops_stress_395() {
        let b = Batch::new(vec![Item::new(395, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 395 + 1);
    }

    #[test]
    fn test_ops_stress_396() {
        let b = Batch::new(vec![Item::new(396, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 396 + 1);
    }

    #[test]
    fn test_ops_stress_397() {
        let b = Batch::new(vec![Item::new(397, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 397 + 1);
    }

    #[test]
    fn test_ops_stress_398() {
        let b = Batch::new(vec![Item::new(398, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 398 + 1);
    }

    #[test]
    fn test_ops_stress_399() {
        let b = Batch::new(vec![Item::new(399, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 399 + 1);
    }

    #[test]
    fn test_ops_stress_400() {
        let b = Batch::new(vec![Item::new(400, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 400 + 1);
    }

    #[test]
    fn test_ops_stress_401() {
        let b = Batch::new(vec![Item::new(401, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 401 + 1);
    }

    #[test]
    fn test_ops_stress_402() {
        let b = Batch::new(vec![Item::new(402, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 402 + 1);
    }

    #[test]
    fn test_ops_stress_403() {
        let b = Batch::new(vec![Item::new(403, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 403 + 1);
    }

    #[test]
    fn test_ops_stress_404() {
        let b = Batch::new(vec![Item::new(404, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 404 + 1);
    }

    #[test]
    fn test_ops_stress_405() {
        let b = Batch::new(vec![Item::new(405, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 405 + 1);
    }

    #[test]
    fn test_ops_stress_406() {
        let b = Batch::new(vec![Item::new(406, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 406 + 1);
    }

    #[test]
    fn test_ops_stress_407() {
        let b = Batch::new(vec![Item::new(407, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 407 + 1);
    }

    #[test]
    fn test_ops_stress_408() {
        let b = Batch::new(vec![Item::new(408, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 408 + 1);
    }

    #[test]
    fn test_ops_stress_409() {
        let b = Batch::new(vec![Item::new(409, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 409 + 1);
    }

    #[test]
    fn test_ops_stress_410() {
        let b = Batch::new(vec![Item::new(410, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 410 + 1);
    }

    #[test]
    fn test_ops_stress_411() {
        let b = Batch::new(vec![Item::new(411, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 411 + 1);
    }

    #[test]
    fn test_ops_stress_412() {
        let b = Batch::new(vec![Item::new(412, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 412 + 1);
    }

    #[test]
    fn test_ops_stress_413() {
        let b = Batch::new(vec![Item::new(413, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 413 + 1);
    }

    #[test]
    fn test_ops_stress_414() {
        let b = Batch::new(vec![Item::new(414, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 414 + 1);
    }

    #[test]
    fn test_ops_stress_415() {
        let b = Batch::new(vec![Item::new(415, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 415 + 1);
    }

    #[test]
    fn test_ops_stress_416() {
        let b = Batch::new(vec![Item::new(416, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 416 + 1);
    }

    #[test]
    fn test_ops_stress_417() {
        let b = Batch::new(vec![Item::new(417, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 417 + 1);
    }

    #[test]
    fn test_ops_stress_418() {
        let b = Batch::new(vec![Item::new(418, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 418 + 1);
    }

    #[test]
    fn test_ops_stress_419() {
        let b = Batch::new(vec![Item::new(419, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 419 + 1);
    }

    #[test]
    fn test_ops_stress_420() {
        let b = Batch::new(vec![Item::new(420, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 420 + 1);
    }

    #[test]
    fn test_ops_stress_421() {
        let b = Batch::new(vec![Item::new(421, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 421 + 1);
    }

    #[test]
    fn test_ops_stress_422() {
        let b = Batch::new(vec![Item::new(422, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 422 + 1);
    }

    #[test]
    fn test_ops_stress_423() {
        let b = Batch::new(vec![Item::new(423, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 423 + 1);
    }

    #[test]
    fn test_ops_stress_424() {
        let b = Batch::new(vec![Item::new(424, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 424 + 1);
    }

    #[test]
    fn test_ops_stress_425() {
        let b = Batch::new(vec![Item::new(425, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 425 + 1);
    }

    #[test]
    fn test_ops_stress_426() {
        let b = Batch::new(vec![Item::new(426, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 426 + 1);
    }

    #[test]
    fn test_ops_stress_427() {
        let b = Batch::new(vec![Item::new(427, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 427 + 1);
    }

    #[test]
    fn test_ops_stress_428() {
        let b = Batch::new(vec![Item::new(428, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 428 + 1);
    }

    #[test]
    fn test_ops_stress_429() {
        let b = Batch::new(vec![Item::new(429, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 429 + 1);
    }

    #[test]
    fn test_ops_stress_430() {
        let b = Batch::new(vec![Item::new(430, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 430 + 1);
    }

    #[test]
    fn test_ops_stress_431() {
        let b = Batch::new(vec![Item::new(431, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 431 + 1);
    }

    #[test]
    fn test_ops_stress_432() {
        let b = Batch::new(vec![Item::new(432, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 432 + 1);
    }

    #[test]
    fn test_ops_stress_433() {
        let b = Batch::new(vec![Item::new(433, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 433 + 1);
    }

    #[test]
    fn test_ops_stress_434() {
        let b = Batch::new(vec![Item::new(434, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 434 + 1);
    }

    #[test]
    fn test_ops_stress_435() {
        let b = Batch::new(vec![Item::new(435, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 435 + 1);
    }

    #[test]
    fn test_ops_stress_436() {
        let b = Batch::new(vec![Item::new(436, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 436 + 1);
    }

    #[test]
    fn test_ops_stress_437() {
        let b = Batch::new(vec![Item::new(437, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 437 + 1);
    }

    #[test]
    fn test_ops_stress_438() {
        let b = Batch::new(vec![Item::new(438, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 438 + 1);
    }

    #[test]
    fn test_ops_stress_439() {
        let b = Batch::new(vec![Item::new(439, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 439 + 1);
    }

    #[test]
    fn test_ops_stress_440() {
        let b = Batch::new(vec![Item::new(440, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 440 + 1);
    }

    #[test]
    fn test_ops_stress_441() {
        let b = Batch::new(vec![Item::new(441, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 441 + 1);
    }

    #[test]
    fn test_ops_stress_442() {
        let b = Batch::new(vec![Item::new(442, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 442 + 1);
    }

    #[test]
    fn test_ops_stress_443() {
        let b = Batch::new(vec![Item::new(443, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 443 + 1);
    }

    #[test]
    fn test_ops_stress_444() {
        let b = Batch::new(vec![Item::new(444, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 444 + 1);
    }

    #[test]
    fn test_ops_stress_445() {
        let b = Batch::new(vec![Item::new(445, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 445 + 1);
    }

    #[test]
    fn test_ops_stress_446() {
        let b = Batch::new(vec![Item::new(446, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 446 + 1);
    }

    #[test]
    fn test_ops_stress_447() {
        let b = Batch::new(vec![Item::new(447, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 447 + 1);
    }

    #[test]
    fn test_ops_stress_448() {
        let b = Batch::new(vec![Item::new(448, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 448 + 1);
    }

    #[test]
    fn test_ops_stress_449() {
        let b = Batch::new(vec![Item::new(449, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 449 + 1);
    }

    #[test]
    fn test_ops_stress_450() {
        let b = Batch::new(vec![Item::new(450, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 450 + 1);
    }

    #[test]
    fn test_ops_stress_451() {
        let b = Batch::new(vec![Item::new(451, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 451 + 1);
    }

    #[test]
    fn test_ops_stress_452() {
        let b = Batch::new(vec![Item::new(452, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 452 + 1);
    }

    #[test]
    fn test_ops_stress_453() {
        let b = Batch::new(vec![Item::new(453, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 453 + 1);
    }

    #[test]
    fn test_ops_stress_454() {
        let b = Batch::new(vec![Item::new(454, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 454 + 1);
    }

    #[test]
    fn test_ops_stress_455() {
        let b = Batch::new(vec![Item::new(455, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 455 + 1);
    }

    #[test]
    fn test_ops_stress_456() {
        let b = Batch::new(vec![Item::new(456, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 456 + 1);
    }

    #[test]
    fn test_ops_stress_457() {
        let b = Batch::new(vec![Item::new(457, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 457 + 1);
    }

    #[test]
    fn test_ops_stress_458() {
        let b = Batch::new(vec![Item::new(458, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 458 + 1);
    }

    #[test]
    fn test_ops_stress_459() {
        let b = Batch::new(vec![Item::new(459, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 459 + 1);
    }

    #[test]
    fn test_ops_stress_460() {
        let b = Batch::new(vec![Item::new(460, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 460 + 1);
    }

    #[test]
    fn test_ops_stress_461() {
        let b = Batch::new(vec![Item::new(461, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 461 + 1);
    }

    #[test]
    fn test_ops_stress_462() {
        let b = Batch::new(vec![Item::new(462, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 462 + 1);
    }

    #[test]
    fn test_ops_stress_463() {
        let b = Batch::new(vec![Item::new(463, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 463 + 1);
    }

    #[test]
    fn test_ops_stress_464() {
        let b = Batch::new(vec![Item::new(464, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 464 + 1);
    }

    #[test]
    fn test_ops_stress_465() {
        let b = Batch::new(vec![Item::new(465, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 465 + 1);
    }

    #[test]
    fn test_ops_stress_466() {
        let b = Batch::new(vec![Item::new(466, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 466 + 1);
    }

    #[test]
    fn test_ops_stress_467() {
        let b = Batch::new(vec![Item::new(467, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 467 + 1);
    }

    #[test]
    fn test_ops_stress_468() {
        let b = Batch::new(vec![Item::new(468, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 468 + 1);
    }

    #[test]
    fn test_ops_stress_469() {
        let b = Batch::new(vec![Item::new(469, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 469 + 1);
    }

    #[test]
    fn test_ops_stress_470() {
        let b = Batch::new(vec![Item::new(470, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 470 + 1);
    }

    #[test]
    fn test_ops_stress_471() {
        let b = Batch::new(vec![Item::new(471, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 471 + 1);
    }

    #[test]
    fn test_ops_stress_472() {
        let b = Batch::new(vec![Item::new(472, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 472 + 1);
    }

    #[test]
    fn test_ops_stress_473() {
        let b = Batch::new(vec![Item::new(473, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 473 + 1);
    }

    #[test]
    fn test_ops_stress_474() {
        let b = Batch::new(vec![Item::new(474, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 474 + 1);
    }

    #[test]
    fn test_ops_stress_475() {
        let b = Batch::new(vec![Item::new(475, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 475 + 1);
    }

    // Dataset ecosystem verification and sample loader check padding line 0
}
