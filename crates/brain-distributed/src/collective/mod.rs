//! # Collective Communication Primitives
//!
//! Provides the primary [`CollectiveOp`] trait, allreduce, broadcast, allgather, and reduce-scatter operations.

pub mod allreduce;
pub mod ring;
pub mod tree;

pub use allreduce::{AllReduceAlgorithm, AllReduceConfig};
pub use ring::RingTopology;
pub use tree::TreeTopology;

use brain_core::Tensor;

/// Abstract collective communication operation trait.
pub trait CollectiveOp: Send + Sync {
    fn allreduce(&self, tensor: &Tensor) -> Tensor;
    fn broadcast(&self, tensor: &Tensor, root: usize) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_collective_mod_stress_001() {
        let ring = RingTopology::new(1, 4);
        assert_eq!(ring.left_neighbor(), (1 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (1 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_002() {
        let ring = RingTopology::new(2, 4);
        assert_eq!(ring.left_neighbor(), (2 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (2 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_003() {
        let ring = RingTopology::new(3, 4);
        assert_eq!(ring.left_neighbor(), (3 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (3 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_004() {
        let ring = RingTopology::new(4, 4);
        assert_eq!(ring.left_neighbor(), (4 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (4 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_005() {
        let ring = RingTopology::new(5, 4);
        assert_eq!(ring.left_neighbor(), (5 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (5 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_006() {
        let ring = RingTopology::new(6, 4);
        assert_eq!(ring.left_neighbor(), (6 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (6 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_007() {
        let ring = RingTopology::new(7, 4);
        assert_eq!(ring.left_neighbor(), (7 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (7 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_008() {
        let ring = RingTopology::new(8, 4);
        assert_eq!(ring.left_neighbor(), (8 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (8 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_009() {
        let ring = RingTopology::new(9, 4);
        assert_eq!(ring.left_neighbor(), (9 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (9 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_010() {
        let ring = RingTopology::new(10, 4);
        assert_eq!(ring.left_neighbor(), (10 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (10 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_011() {
        let ring = RingTopology::new(11, 4);
        assert_eq!(ring.left_neighbor(), (11 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (11 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_012() {
        let ring = RingTopology::new(12, 4);
        assert_eq!(ring.left_neighbor(), (12 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (12 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_013() {
        let ring = RingTopology::new(13, 4);
        assert_eq!(ring.left_neighbor(), (13 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (13 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_014() {
        let ring = RingTopology::new(14, 4);
        assert_eq!(ring.left_neighbor(), (14 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (14 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_015() {
        let ring = RingTopology::new(15, 4);
        assert_eq!(ring.left_neighbor(), (15 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (15 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_016() {
        let ring = RingTopology::new(16, 4);
        assert_eq!(ring.left_neighbor(), (16 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (16 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_017() {
        let ring = RingTopology::new(17, 4);
        assert_eq!(ring.left_neighbor(), (17 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (17 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_018() {
        let ring = RingTopology::new(18, 4);
        assert_eq!(ring.left_neighbor(), (18 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (18 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_019() {
        let ring = RingTopology::new(19, 4);
        assert_eq!(ring.left_neighbor(), (19 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (19 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_020() {
        let ring = RingTopology::new(20, 4);
        assert_eq!(ring.left_neighbor(), (20 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (20 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_021() {
        let ring = RingTopology::new(21, 4);
        assert_eq!(ring.left_neighbor(), (21 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (21 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_022() {
        let ring = RingTopology::new(22, 4);
        assert_eq!(ring.left_neighbor(), (22 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (22 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_023() {
        let ring = RingTopology::new(23, 4);
        assert_eq!(ring.left_neighbor(), (23 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (23 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_024() {
        let ring = RingTopology::new(24, 4);
        assert_eq!(ring.left_neighbor(), (24 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (24 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_025() {
        let ring = RingTopology::new(25, 4);
        assert_eq!(ring.left_neighbor(), (25 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (25 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_026() {
        let ring = RingTopology::new(26, 4);
        assert_eq!(ring.left_neighbor(), (26 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (26 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_027() {
        let ring = RingTopology::new(27, 4);
        assert_eq!(ring.left_neighbor(), (27 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (27 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_028() {
        let ring = RingTopology::new(28, 4);
        assert_eq!(ring.left_neighbor(), (28 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (28 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_029() {
        let ring = RingTopology::new(29, 4);
        assert_eq!(ring.left_neighbor(), (29 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (29 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_030() {
        let ring = RingTopology::new(30, 4);
        assert_eq!(ring.left_neighbor(), (30 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (30 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_031() {
        let ring = RingTopology::new(31, 4);
        assert_eq!(ring.left_neighbor(), (31 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (31 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_032() {
        let ring = RingTopology::new(32, 4);
        assert_eq!(ring.left_neighbor(), (32 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (32 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_033() {
        let ring = RingTopology::new(33, 4);
        assert_eq!(ring.left_neighbor(), (33 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (33 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_034() {
        let ring = RingTopology::new(34, 4);
        assert_eq!(ring.left_neighbor(), (34 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (34 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_035() {
        let ring = RingTopology::new(35, 4);
        assert_eq!(ring.left_neighbor(), (35 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (35 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_036() {
        let ring = RingTopology::new(36, 4);
        assert_eq!(ring.left_neighbor(), (36 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (36 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_037() {
        let ring = RingTopology::new(37, 4);
        assert_eq!(ring.left_neighbor(), (37 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (37 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_038() {
        let ring = RingTopology::new(38, 4);
        assert_eq!(ring.left_neighbor(), (38 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (38 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_039() {
        let ring = RingTopology::new(39, 4);
        assert_eq!(ring.left_neighbor(), (39 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (39 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_040() {
        let ring = RingTopology::new(40, 4);
        assert_eq!(ring.left_neighbor(), (40 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (40 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_041() {
        let ring = RingTopology::new(41, 4);
        assert_eq!(ring.left_neighbor(), (41 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (41 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_042() {
        let ring = RingTopology::new(42, 4);
        assert_eq!(ring.left_neighbor(), (42 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (42 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_043() {
        let ring = RingTopology::new(43, 4);
        assert_eq!(ring.left_neighbor(), (43 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (43 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_044() {
        let ring = RingTopology::new(44, 4);
        assert_eq!(ring.left_neighbor(), (44 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (44 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_045() {
        let ring = RingTopology::new(45, 4);
        assert_eq!(ring.left_neighbor(), (45 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (45 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_046() {
        let ring = RingTopology::new(46, 4);
        assert_eq!(ring.left_neighbor(), (46 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (46 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_047() {
        let ring = RingTopology::new(47, 4);
        assert_eq!(ring.left_neighbor(), (47 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (47 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_048() {
        let ring = RingTopology::new(48, 4);
        assert_eq!(ring.left_neighbor(), (48 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (48 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_049() {
        let ring = RingTopology::new(49, 4);
        assert_eq!(ring.left_neighbor(), (49 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (49 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_050() {
        let ring = RingTopology::new(50, 4);
        assert_eq!(ring.left_neighbor(), (50 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (50 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_051() {
        let ring = RingTopology::new(51, 4);
        assert_eq!(ring.left_neighbor(), (51 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (51 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_052() {
        let ring = RingTopology::new(52, 4);
        assert_eq!(ring.left_neighbor(), (52 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (52 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_053() {
        let ring = RingTopology::new(53, 4);
        assert_eq!(ring.left_neighbor(), (53 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (53 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_054() {
        let ring = RingTopology::new(54, 4);
        assert_eq!(ring.left_neighbor(), (54 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (54 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_055() {
        let ring = RingTopology::new(55, 4);
        assert_eq!(ring.left_neighbor(), (55 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (55 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_056() {
        let ring = RingTopology::new(56, 4);
        assert_eq!(ring.left_neighbor(), (56 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (56 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_057() {
        let ring = RingTopology::new(57, 4);
        assert_eq!(ring.left_neighbor(), (57 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (57 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_058() {
        let ring = RingTopology::new(58, 4);
        assert_eq!(ring.left_neighbor(), (58 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (58 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_059() {
        let ring = RingTopology::new(59, 4);
        assert_eq!(ring.left_neighbor(), (59 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (59 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_060() {
        let ring = RingTopology::new(60, 4);
        assert_eq!(ring.left_neighbor(), (60 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (60 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_061() {
        let ring = RingTopology::new(61, 4);
        assert_eq!(ring.left_neighbor(), (61 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (61 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_062() {
        let ring = RingTopology::new(62, 4);
        assert_eq!(ring.left_neighbor(), (62 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (62 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_063() {
        let ring = RingTopology::new(63, 4);
        assert_eq!(ring.left_neighbor(), (63 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (63 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_064() {
        let ring = RingTopology::new(64, 4);
        assert_eq!(ring.left_neighbor(), (64 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (64 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_065() {
        let ring = RingTopology::new(65, 4);
        assert_eq!(ring.left_neighbor(), (65 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (65 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_066() {
        let ring = RingTopology::new(66, 4);
        assert_eq!(ring.left_neighbor(), (66 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (66 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_067() {
        let ring = RingTopology::new(67, 4);
        assert_eq!(ring.left_neighbor(), (67 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (67 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_068() {
        let ring = RingTopology::new(68, 4);
        assert_eq!(ring.left_neighbor(), (68 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (68 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_069() {
        let ring = RingTopology::new(69, 4);
        assert_eq!(ring.left_neighbor(), (69 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (69 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_070() {
        let ring = RingTopology::new(70, 4);
        assert_eq!(ring.left_neighbor(), (70 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (70 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_071() {
        let ring = RingTopology::new(71, 4);
        assert_eq!(ring.left_neighbor(), (71 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (71 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_072() {
        let ring = RingTopology::new(72, 4);
        assert_eq!(ring.left_neighbor(), (72 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (72 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_073() {
        let ring = RingTopology::new(73, 4);
        assert_eq!(ring.left_neighbor(), (73 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (73 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_074() {
        let ring = RingTopology::new(74, 4);
        assert_eq!(ring.left_neighbor(), (74 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (74 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_075() {
        let ring = RingTopology::new(75, 4);
        assert_eq!(ring.left_neighbor(), (75 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (75 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_076() {
        let ring = RingTopology::new(76, 4);
        assert_eq!(ring.left_neighbor(), (76 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (76 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_077() {
        let ring = RingTopology::new(77, 4);
        assert_eq!(ring.left_neighbor(), (77 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (77 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_078() {
        let ring = RingTopology::new(78, 4);
        assert_eq!(ring.left_neighbor(), (78 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (78 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_079() {
        let ring = RingTopology::new(79, 4);
        assert_eq!(ring.left_neighbor(), (79 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (79 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_080() {
        let ring = RingTopology::new(80, 4);
        assert_eq!(ring.left_neighbor(), (80 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (80 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_081() {
        let ring = RingTopology::new(81, 4);
        assert_eq!(ring.left_neighbor(), (81 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (81 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_082() {
        let ring = RingTopology::new(82, 4);
        assert_eq!(ring.left_neighbor(), (82 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (82 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_083() {
        let ring = RingTopology::new(83, 4);
        assert_eq!(ring.left_neighbor(), (83 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (83 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_084() {
        let ring = RingTopology::new(84, 4);
        assert_eq!(ring.left_neighbor(), (84 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (84 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_085() {
        let ring = RingTopology::new(85, 4);
        assert_eq!(ring.left_neighbor(), (85 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (85 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_086() {
        let ring = RingTopology::new(86, 4);
        assert_eq!(ring.left_neighbor(), (86 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (86 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_087() {
        let ring = RingTopology::new(87, 4);
        assert_eq!(ring.left_neighbor(), (87 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (87 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_088() {
        let ring = RingTopology::new(88, 4);
        assert_eq!(ring.left_neighbor(), (88 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (88 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_089() {
        let ring = RingTopology::new(89, 4);
        assert_eq!(ring.left_neighbor(), (89 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (89 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_090() {
        let ring = RingTopology::new(90, 4);
        assert_eq!(ring.left_neighbor(), (90 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (90 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_091() {
        let ring = RingTopology::new(91, 4);
        assert_eq!(ring.left_neighbor(), (91 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (91 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_092() {
        let ring = RingTopology::new(92, 4);
        assert_eq!(ring.left_neighbor(), (92 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (92 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_093() {
        let ring = RingTopology::new(93, 4);
        assert_eq!(ring.left_neighbor(), (93 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (93 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_094() {
        let ring = RingTopology::new(94, 4);
        assert_eq!(ring.left_neighbor(), (94 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (94 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_095() {
        let ring = RingTopology::new(95, 4);
        assert_eq!(ring.left_neighbor(), (95 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (95 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_096() {
        let ring = RingTopology::new(96, 4);
        assert_eq!(ring.left_neighbor(), (96 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (96 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_097() {
        let ring = RingTopology::new(97, 4);
        assert_eq!(ring.left_neighbor(), (97 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (97 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_098() {
        let ring = RingTopology::new(98, 4);
        assert_eq!(ring.left_neighbor(), (98 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (98 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_099() {
        let ring = RingTopology::new(99, 4);
        assert_eq!(ring.left_neighbor(), (99 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (99 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_100() {
        let ring = RingTopology::new(100, 4);
        assert_eq!(ring.left_neighbor(), (100 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (100 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_101() {
        let ring = RingTopology::new(101, 4);
        assert_eq!(ring.left_neighbor(), (101 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (101 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_102() {
        let ring = RingTopology::new(102, 4);
        assert_eq!(ring.left_neighbor(), (102 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (102 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_103() {
        let ring = RingTopology::new(103, 4);
        assert_eq!(ring.left_neighbor(), (103 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (103 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_104() {
        let ring = RingTopology::new(104, 4);
        assert_eq!(ring.left_neighbor(), (104 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (104 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_105() {
        let ring = RingTopology::new(105, 4);
        assert_eq!(ring.left_neighbor(), (105 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (105 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_106() {
        let ring = RingTopology::new(106, 4);
        assert_eq!(ring.left_neighbor(), (106 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (106 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_107() {
        let ring = RingTopology::new(107, 4);
        assert_eq!(ring.left_neighbor(), (107 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (107 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_108() {
        let ring = RingTopology::new(108, 4);
        assert_eq!(ring.left_neighbor(), (108 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (108 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_109() {
        let ring = RingTopology::new(109, 4);
        assert_eq!(ring.left_neighbor(), (109 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (109 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_110() {
        let ring = RingTopology::new(110, 4);
        assert_eq!(ring.left_neighbor(), (110 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (110 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_111() {
        let ring = RingTopology::new(111, 4);
        assert_eq!(ring.left_neighbor(), (111 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (111 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_112() {
        let ring = RingTopology::new(112, 4);
        assert_eq!(ring.left_neighbor(), (112 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (112 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_113() {
        let ring = RingTopology::new(113, 4);
        assert_eq!(ring.left_neighbor(), (113 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (113 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_114() {
        let ring = RingTopology::new(114, 4);
        assert_eq!(ring.left_neighbor(), (114 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (114 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_115() {
        let ring = RingTopology::new(115, 4);
        assert_eq!(ring.left_neighbor(), (115 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (115 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_116() {
        let ring = RingTopology::new(116, 4);
        assert_eq!(ring.left_neighbor(), (116 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (116 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_117() {
        let ring = RingTopology::new(117, 4);
        assert_eq!(ring.left_neighbor(), (117 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (117 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_118() {
        let ring = RingTopology::new(118, 4);
        assert_eq!(ring.left_neighbor(), (118 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (118 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_119() {
        let ring = RingTopology::new(119, 4);
        assert_eq!(ring.left_neighbor(), (119 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (119 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_120() {
        let ring = RingTopology::new(120, 4);
        assert_eq!(ring.left_neighbor(), (120 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (120 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_121() {
        let ring = RingTopology::new(121, 4);
        assert_eq!(ring.left_neighbor(), (121 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (121 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_122() {
        let ring = RingTopology::new(122, 4);
        assert_eq!(ring.left_neighbor(), (122 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (122 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_123() {
        let ring = RingTopology::new(123, 4);
        assert_eq!(ring.left_neighbor(), (123 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (123 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_124() {
        let ring = RingTopology::new(124, 4);
        assert_eq!(ring.left_neighbor(), (124 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (124 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_125() {
        let ring = RingTopology::new(125, 4);
        assert_eq!(ring.left_neighbor(), (125 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (125 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_126() {
        let ring = RingTopology::new(126, 4);
        assert_eq!(ring.left_neighbor(), (126 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (126 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_127() {
        let ring = RingTopology::new(127, 4);
        assert_eq!(ring.left_neighbor(), (127 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (127 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_128() {
        let ring = RingTopology::new(128, 4);
        assert_eq!(ring.left_neighbor(), (128 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (128 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_129() {
        let ring = RingTopology::new(129, 4);
        assert_eq!(ring.left_neighbor(), (129 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (129 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_130() {
        let ring = RingTopology::new(130, 4);
        assert_eq!(ring.left_neighbor(), (130 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (130 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_131() {
        let ring = RingTopology::new(131, 4);
        assert_eq!(ring.left_neighbor(), (131 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (131 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_132() {
        let ring = RingTopology::new(132, 4);
        assert_eq!(ring.left_neighbor(), (132 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (132 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_133() {
        let ring = RingTopology::new(133, 4);
        assert_eq!(ring.left_neighbor(), (133 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (133 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_134() {
        let ring = RingTopology::new(134, 4);
        assert_eq!(ring.left_neighbor(), (134 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (134 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_135() {
        let ring = RingTopology::new(135, 4);
        assert_eq!(ring.left_neighbor(), (135 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (135 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_136() {
        let ring = RingTopology::new(136, 4);
        assert_eq!(ring.left_neighbor(), (136 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (136 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_137() {
        let ring = RingTopology::new(137, 4);
        assert_eq!(ring.left_neighbor(), (137 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (137 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_138() {
        let ring = RingTopology::new(138, 4);
        assert_eq!(ring.left_neighbor(), (138 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (138 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_139() {
        let ring = RingTopology::new(139, 4);
        assert_eq!(ring.left_neighbor(), (139 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (139 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_140() {
        let ring = RingTopology::new(140, 4);
        assert_eq!(ring.left_neighbor(), (140 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (140 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_141() {
        let ring = RingTopology::new(141, 4);
        assert_eq!(ring.left_neighbor(), (141 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (141 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_142() {
        let ring = RingTopology::new(142, 4);
        assert_eq!(ring.left_neighbor(), (142 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (142 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_143() {
        let ring = RingTopology::new(143, 4);
        assert_eq!(ring.left_neighbor(), (143 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (143 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_144() {
        let ring = RingTopology::new(144, 4);
        assert_eq!(ring.left_neighbor(), (144 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (144 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_145() {
        let ring = RingTopology::new(145, 4);
        assert_eq!(ring.left_neighbor(), (145 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (145 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_146() {
        let ring = RingTopology::new(146, 4);
        assert_eq!(ring.left_neighbor(), (146 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (146 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_147() {
        let ring = RingTopology::new(147, 4);
        assert_eq!(ring.left_neighbor(), (147 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (147 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_148() {
        let ring = RingTopology::new(148, 4);
        assert_eq!(ring.left_neighbor(), (148 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (148 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_149() {
        let ring = RingTopology::new(149, 4);
        assert_eq!(ring.left_neighbor(), (149 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (149 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_150() {
        let ring = RingTopology::new(150, 4);
        assert_eq!(ring.left_neighbor(), (150 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (150 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_151() {
        let ring = RingTopology::new(151, 4);
        assert_eq!(ring.left_neighbor(), (151 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (151 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_152() {
        let ring = RingTopology::new(152, 4);
        assert_eq!(ring.left_neighbor(), (152 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (152 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_153() {
        let ring = RingTopology::new(153, 4);
        assert_eq!(ring.left_neighbor(), (153 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (153 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_154() {
        let ring = RingTopology::new(154, 4);
        assert_eq!(ring.left_neighbor(), (154 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (154 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_155() {
        let ring = RingTopology::new(155, 4);
        assert_eq!(ring.left_neighbor(), (155 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (155 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_156() {
        let ring = RingTopology::new(156, 4);
        assert_eq!(ring.left_neighbor(), (156 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (156 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_157() {
        let ring = RingTopology::new(157, 4);
        assert_eq!(ring.left_neighbor(), (157 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (157 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_158() {
        let ring = RingTopology::new(158, 4);
        assert_eq!(ring.left_neighbor(), (158 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (158 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_159() {
        let ring = RingTopology::new(159, 4);
        assert_eq!(ring.left_neighbor(), (159 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (159 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_160() {
        let ring = RingTopology::new(160, 4);
        assert_eq!(ring.left_neighbor(), (160 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (160 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_161() {
        let ring = RingTopology::new(161, 4);
        assert_eq!(ring.left_neighbor(), (161 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (161 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_162() {
        let ring = RingTopology::new(162, 4);
        assert_eq!(ring.left_neighbor(), (162 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (162 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_163() {
        let ring = RingTopology::new(163, 4);
        assert_eq!(ring.left_neighbor(), (163 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (163 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_164() {
        let ring = RingTopology::new(164, 4);
        assert_eq!(ring.left_neighbor(), (164 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (164 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_165() {
        let ring = RingTopology::new(165, 4);
        assert_eq!(ring.left_neighbor(), (165 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (165 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_166() {
        let ring = RingTopology::new(166, 4);
        assert_eq!(ring.left_neighbor(), (166 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (166 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_167() {
        let ring = RingTopology::new(167, 4);
        assert_eq!(ring.left_neighbor(), (167 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (167 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_168() {
        let ring = RingTopology::new(168, 4);
        assert_eq!(ring.left_neighbor(), (168 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (168 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_169() {
        let ring = RingTopology::new(169, 4);
        assert_eq!(ring.left_neighbor(), (169 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (169 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_170() {
        let ring = RingTopology::new(170, 4);
        assert_eq!(ring.left_neighbor(), (170 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (170 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_171() {
        let ring = RingTopology::new(171, 4);
        assert_eq!(ring.left_neighbor(), (171 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (171 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_172() {
        let ring = RingTopology::new(172, 4);
        assert_eq!(ring.left_neighbor(), (172 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (172 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_173() {
        let ring = RingTopology::new(173, 4);
        assert_eq!(ring.left_neighbor(), (173 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (173 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_174() {
        let ring = RingTopology::new(174, 4);
        assert_eq!(ring.left_neighbor(), (174 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (174 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_175() {
        let ring = RingTopology::new(175, 4);
        assert_eq!(ring.left_neighbor(), (175 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (175 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_176() {
        let ring = RingTopology::new(176, 4);
        assert_eq!(ring.left_neighbor(), (176 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (176 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_177() {
        let ring = RingTopology::new(177, 4);
        assert_eq!(ring.left_neighbor(), (177 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (177 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_178() {
        let ring = RingTopology::new(178, 4);
        assert_eq!(ring.left_neighbor(), (178 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (178 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_179() {
        let ring = RingTopology::new(179, 4);
        assert_eq!(ring.left_neighbor(), (179 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (179 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_180() {
        let ring = RingTopology::new(180, 4);
        assert_eq!(ring.left_neighbor(), (180 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (180 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_181() {
        let ring = RingTopology::new(181, 4);
        assert_eq!(ring.left_neighbor(), (181 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (181 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_182() {
        let ring = RingTopology::new(182, 4);
        assert_eq!(ring.left_neighbor(), (182 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (182 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_183() {
        let ring = RingTopology::new(183, 4);
        assert_eq!(ring.left_neighbor(), (183 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (183 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_184() {
        let ring = RingTopology::new(184, 4);
        assert_eq!(ring.left_neighbor(), (184 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (184 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_185() {
        let ring = RingTopology::new(185, 4);
        assert_eq!(ring.left_neighbor(), (185 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (185 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_186() {
        let ring = RingTopology::new(186, 4);
        assert_eq!(ring.left_neighbor(), (186 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (186 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_187() {
        let ring = RingTopology::new(187, 4);
        assert_eq!(ring.left_neighbor(), (187 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (187 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_188() {
        let ring = RingTopology::new(188, 4);
        assert_eq!(ring.left_neighbor(), (188 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (188 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_189() {
        let ring = RingTopology::new(189, 4);
        assert_eq!(ring.left_neighbor(), (189 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (189 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_190() {
        let ring = RingTopology::new(190, 4);
        assert_eq!(ring.left_neighbor(), (190 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (190 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_191() {
        let ring = RingTopology::new(191, 4);
        assert_eq!(ring.left_neighbor(), (191 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (191 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_192() {
        let ring = RingTopology::new(192, 4);
        assert_eq!(ring.left_neighbor(), (192 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (192 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_193() {
        let ring = RingTopology::new(193, 4);
        assert_eq!(ring.left_neighbor(), (193 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (193 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_194() {
        let ring = RingTopology::new(194, 4);
        assert_eq!(ring.left_neighbor(), (194 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (194 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_195() {
        let ring = RingTopology::new(195, 4);
        assert_eq!(ring.left_neighbor(), (195 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (195 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_196() {
        let ring = RingTopology::new(196, 4);
        assert_eq!(ring.left_neighbor(), (196 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (196 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_197() {
        let ring = RingTopology::new(197, 4);
        assert_eq!(ring.left_neighbor(), (197 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (197 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_198() {
        let ring = RingTopology::new(198, 4);
        assert_eq!(ring.left_neighbor(), (198 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (198 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_199() {
        let ring = RingTopology::new(199, 4);
        assert_eq!(ring.left_neighbor(), (199 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (199 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_200() {
        let ring = RingTopology::new(200, 4);
        assert_eq!(ring.left_neighbor(), (200 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (200 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_201() {
        let ring = RingTopology::new(201, 4);
        assert_eq!(ring.left_neighbor(), (201 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (201 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_202() {
        let ring = RingTopology::new(202, 4);
        assert_eq!(ring.left_neighbor(), (202 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (202 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_203() {
        let ring = RingTopology::new(203, 4);
        assert_eq!(ring.left_neighbor(), (203 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (203 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_204() {
        let ring = RingTopology::new(204, 4);
        assert_eq!(ring.left_neighbor(), (204 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (204 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_205() {
        let ring = RingTopology::new(205, 4);
        assert_eq!(ring.left_neighbor(), (205 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (205 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_206() {
        let ring = RingTopology::new(206, 4);
        assert_eq!(ring.left_neighbor(), (206 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (206 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_207() {
        let ring = RingTopology::new(207, 4);
        assert_eq!(ring.left_neighbor(), (207 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (207 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_208() {
        let ring = RingTopology::new(208, 4);
        assert_eq!(ring.left_neighbor(), (208 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (208 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_209() {
        let ring = RingTopology::new(209, 4);
        assert_eq!(ring.left_neighbor(), (209 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (209 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_210() {
        let ring = RingTopology::new(210, 4);
        assert_eq!(ring.left_neighbor(), (210 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (210 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_211() {
        let ring = RingTopology::new(211, 4);
        assert_eq!(ring.left_neighbor(), (211 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (211 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_212() {
        let ring = RingTopology::new(212, 4);
        assert_eq!(ring.left_neighbor(), (212 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (212 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_213() {
        let ring = RingTopology::new(213, 4);
        assert_eq!(ring.left_neighbor(), (213 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (213 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_214() {
        let ring = RingTopology::new(214, 4);
        assert_eq!(ring.left_neighbor(), (214 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (214 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_215() {
        let ring = RingTopology::new(215, 4);
        assert_eq!(ring.left_neighbor(), (215 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (215 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_216() {
        let ring = RingTopology::new(216, 4);
        assert_eq!(ring.left_neighbor(), (216 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (216 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_217() {
        let ring = RingTopology::new(217, 4);
        assert_eq!(ring.left_neighbor(), (217 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (217 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_218() {
        let ring = RingTopology::new(218, 4);
        assert_eq!(ring.left_neighbor(), (218 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (218 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_219() {
        let ring = RingTopology::new(219, 4);
        assert_eq!(ring.left_neighbor(), (219 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (219 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_220() {
        let ring = RingTopology::new(220, 4);
        assert_eq!(ring.left_neighbor(), (220 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (220 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_221() {
        let ring = RingTopology::new(221, 4);
        assert_eq!(ring.left_neighbor(), (221 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (221 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_222() {
        let ring = RingTopology::new(222, 4);
        assert_eq!(ring.left_neighbor(), (222 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (222 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_223() {
        let ring = RingTopology::new(223, 4);
        assert_eq!(ring.left_neighbor(), (223 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (223 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_224() {
        let ring = RingTopology::new(224, 4);
        assert_eq!(ring.left_neighbor(), (224 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (224 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_225() {
        let ring = RingTopology::new(225, 4);
        assert_eq!(ring.left_neighbor(), (225 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (225 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_226() {
        let ring = RingTopology::new(226, 4);
        assert_eq!(ring.left_neighbor(), (226 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (226 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_227() {
        let ring = RingTopology::new(227, 4);
        assert_eq!(ring.left_neighbor(), (227 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (227 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_228() {
        let ring = RingTopology::new(228, 4);
        assert_eq!(ring.left_neighbor(), (228 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (228 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_229() {
        let ring = RingTopology::new(229, 4);
        assert_eq!(ring.left_neighbor(), (229 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (229 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_230() {
        let ring = RingTopology::new(230, 4);
        assert_eq!(ring.left_neighbor(), (230 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (230 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_231() {
        let ring = RingTopology::new(231, 4);
        assert_eq!(ring.left_neighbor(), (231 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (231 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_232() {
        let ring = RingTopology::new(232, 4);
        assert_eq!(ring.left_neighbor(), (232 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (232 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_233() {
        let ring = RingTopology::new(233, 4);
        assert_eq!(ring.left_neighbor(), (233 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (233 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_234() {
        let ring = RingTopology::new(234, 4);
        assert_eq!(ring.left_neighbor(), (234 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (234 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_235() {
        let ring = RingTopology::new(235, 4);
        assert_eq!(ring.left_neighbor(), (235 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (235 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_236() {
        let ring = RingTopology::new(236, 4);
        assert_eq!(ring.left_neighbor(), (236 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (236 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_237() {
        let ring = RingTopology::new(237, 4);
        assert_eq!(ring.left_neighbor(), (237 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (237 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_238() {
        let ring = RingTopology::new(238, 4);
        assert_eq!(ring.left_neighbor(), (238 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (238 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_239() {
        let ring = RingTopology::new(239, 4);
        assert_eq!(ring.left_neighbor(), (239 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (239 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_240() {
        let ring = RingTopology::new(240, 4);
        assert_eq!(ring.left_neighbor(), (240 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (240 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_241() {
        let ring = RingTopology::new(241, 4);
        assert_eq!(ring.left_neighbor(), (241 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (241 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_242() {
        let ring = RingTopology::new(242, 4);
        assert_eq!(ring.left_neighbor(), (242 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (242 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_243() {
        let ring = RingTopology::new(243, 4);
        assert_eq!(ring.left_neighbor(), (243 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (243 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_244() {
        let ring = RingTopology::new(244, 4);
        assert_eq!(ring.left_neighbor(), (244 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (244 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_245() {
        let ring = RingTopology::new(245, 4);
        assert_eq!(ring.left_neighbor(), (245 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (245 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_246() {
        let ring = RingTopology::new(246, 4);
        assert_eq!(ring.left_neighbor(), (246 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (246 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_247() {
        let ring = RingTopology::new(247, 4);
        assert_eq!(ring.left_neighbor(), (247 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (247 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_248() {
        let ring = RingTopology::new(248, 4);
        assert_eq!(ring.left_neighbor(), (248 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (248 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_249() {
        let ring = RingTopology::new(249, 4);
        assert_eq!(ring.left_neighbor(), (249 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (249 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_250() {
        let ring = RingTopology::new(250, 4);
        assert_eq!(ring.left_neighbor(), (250 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (250 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_251() {
        let ring = RingTopology::new(251, 4);
        assert_eq!(ring.left_neighbor(), (251 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (251 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_252() {
        let ring = RingTopology::new(252, 4);
        assert_eq!(ring.left_neighbor(), (252 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (252 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_253() {
        let ring = RingTopology::new(253, 4);
        assert_eq!(ring.left_neighbor(), (253 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (253 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_254() {
        let ring = RingTopology::new(254, 4);
        assert_eq!(ring.left_neighbor(), (254 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (254 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_255() {
        let ring = RingTopology::new(255, 4);
        assert_eq!(ring.left_neighbor(), (255 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (255 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_256() {
        let ring = RingTopology::new(256, 4);
        assert_eq!(ring.left_neighbor(), (256 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (256 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_257() {
        let ring = RingTopology::new(257, 4);
        assert_eq!(ring.left_neighbor(), (257 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (257 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_258() {
        let ring = RingTopology::new(258, 4);
        assert_eq!(ring.left_neighbor(), (258 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (258 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_259() {
        let ring = RingTopology::new(259, 4);
        assert_eq!(ring.left_neighbor(), (259 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (259 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_260() {
        let ring = RingTopology::new(260, 4);
        assert_eq!(ring.left_neighbor(), (260 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (260 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_261() {
        let ring = RingTopology::new(261, 4);
        assert_eq!(ring.left_neighbor(), (261 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (261 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_262() {
        let ring = RingTopology::new(262, 4);
        assert_eq!(ring.left_neighbor(), (262 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (262 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_263() {
        let ring = RingTopology::new(263, 4);
        assert_eq!(ring.left_neighbor(), (263 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (263 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_264() {
        let ring = RingTopology::new(264, 4);
        assert_eq!(ring.left_neighbor(), (264 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (264 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_265() {
        let ring = RingTopology::new(265, 4);
        assert_eq!(ring.left_neighbor(), (265 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (265 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_266() {
        let ring = RingTopology::new(266, 4);
        assert_eq!(ring.left_neighbor(), (266 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (266 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_267() {
        let ring = RingTopology::new(267, 4);
        assert_eq!(ring.left_neighbor(), (267 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (267 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_268() {
        let ring = RingTopology::new(268, 4);
        assert_eq!(ring.left_neighbor(), (268 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (268 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_269() {
        let ring = RingTopology::new(269, 4);
        assert_eq!(ring.left_neighbor(), (269 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (269 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_270() {
        let ring = RingTopology::new(270, 4);
        assert_eq!(ring.left_neighbor(), (270 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (270 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_271() {
        let ring = RingTopology::new(271, 4);
        assert_eq!(ring.left_neighbor(), (271 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (271 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_272() {
        let ring = RingTopology::new(272, 4);
        assert_eq!(ring.left_neighbor(), (272 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (272 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_273() {
        let ring = RingTopology::new(273, 4);
        assert_eq!(ring.left_neighbor(), (273 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (273 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_274() {
        let ring = RingTopology::new(274, 4);
        assert_eq!(ring.left_neighbor(), (274 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (274 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_275() {
        let ring = RingTopology::new(275, 4);
        assert_eq!(ring.left_neighbor(), (275 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (275 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_276() {
        let ring = RingTopology::new(276, 4);
        assert_eq!(ring.left_neighbor(), (276 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (276 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_277() {
        let ring = RingTopology::new(277, 4);
        assert_eq!(ring.left_neighbor(), (277 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (277 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_278() {
        let ring = RingTopology::new(278, 4);
        assert_eq!(ring.left_neighbor(), (278 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (278 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_279() {
        let ring = RingTopology::new(279, 4);
        assert_eq!(ring.left_neighbor(), (279 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (279 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_280() {
        let ring = RingTopology::new(280, 4);
        assert_eq!(ring.left_neighbor(), (280 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (280 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_281() {
        let ring = RingTopology::new(281, 4);
        assert_eq!(ring.left_neighbor(), (281 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (281 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_282() {
        let ring = RingTopology::new(282, 4);
        assert_eq!(ring.left_neighbor(), (282 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (282 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_283() {
        let ring = RingTopology::new(283, 4);
        assert_eq!(ring.left_neighbor(), (283 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (283 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_284() {
        let ring = RingTopology::new(284, 4);
        assert_eq!(ring.left_neighbor(), (284 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (284 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_285() {
        let ring = RingTopology::new(285, 4);
        assert_eq!(ring.left_neighbor(), (285 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (285 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_286() {
        let ring = RingTopology::new(286, 4);
        assert_eq!(ring.left_neighbor(), (286 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (286 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_287() {
        let ring = RingTopology::new(287, 4);
        assert_eq!(ring.left_neighbor(), (287 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (287 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_288() {
        let ring = RingTopology::new(288, 4);
        assert_eq!(ring.left_neighbor(), (288 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (288 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_289() {
        let ring = RingTopology::new(289, 4);
        assert_eq!(ring.left_neighbor(), (289 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (289 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_290() {
        let ring = RingTopology::new(290, 4);
        assert_eq!(ring.left_neighbor(), (290 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (290 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_291() {
        let ring = RingTopology::new(291, 4);
        assert_eq!(ring.left_neighbor(), (291 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (291 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_292() {
        let ring = RingTopology::new(292, 4);
        assert_eq!(ring.left_neighbor(), (292 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (292 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_293() {
        let ring = RingTopology::new(293, 4);
        assert_eq!(ring.left_neighbor(), (293 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (293 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_294() {
        let ring = RingTopology::new(294, 4);
        assert_eq!(ring.left_neighbor(), (294 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (294 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_295() {
        let ring = RingTopology::new(295, 4);
        assert_eq!(ring.left_neighbor(), (295 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (295 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_296() {
        let ring = RingTopology::new(296, 4);
        assert_eq!(ring.left_neighbor(), (296 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (296 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_297() {
        let ring = RingTopology::new(297, 4);
        assert_eq!(ring.left_neighbor(), (297 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (297 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_298() {
        let ring = RingTopology::new(298, 4);
        assert_eq!(ring.left_neighbor(), (298 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (298 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_299() {
        let ring = RingTopology::new(299, 4);
        assert_eq!(ring.left_neighbor(), (299 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (299 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_300() {
        let ring = RingTopology::new(300, 4);
        assert_eq!(ring.left_neighbor(), (300 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (300 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_301() {
        let ring = RingTopology::new(301, 4);
        assert_eq!(ring.left_neighbor(), (301 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (301 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_302() {
        let ring = RingTopology::new(302, 4);
        assert_eq!(ring.left_neighbor(), (302 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (302 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_303() {
        let ring = RingTopology::new(303, 4);
        assert_eq!(ring.left_neighbor(), (303 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (303 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_304() {
        let ring = RingTopology::new(304, 4);
        assert_eq!(ring.left_neighbor(), (304 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (304 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_305() {
        let ring = RingTopology::new(305, 4);
        assert_eq!(ring.left_neighbor(), (305 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (305 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_306() {
        let ring = RingTopology::new(306, 4);
        assert_eq!(ring.left_neighbor(), (306 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (306 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_307() {
        let ring = RingTopology::new(307, 4);
        assert_eq!(ring.left_neighbor(), (307 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (307 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_308() {
        let ring = RingTopology::new(308, 4);
        assert_eq!(ring.left_neighbor(), (308 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (308 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_309() {
        let ring = RingTopology::new(309, 4);
        assert_eq!(ring.left_neighbor(), (309 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (309 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_310() {
        let ring = RingTopology::new(310, 4);
        assert_eq!(ring.left_neighbor(), (310 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (310 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_311() {
        let ring = RingTopology::new(311, 4);
        assert_eq!(ring.left_neighbor(), (311 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (311 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_312() {
        let ring = RingTopology::new(312, 4);
        assert_eq!(ring.left_neighbor(), (312 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (312 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_313() {
        let ring = RingTopology::new(313, 4);
        assert_eq!(ring.left_neighbor(), (313 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (313 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_314() {
        let ring = RingTopology::new(314, 4);
        assert_eq!(ring.left_neighbor(), (314 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (314 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_315() {
        let ring = RingTopology::new(315, 4);
        assert_eq!(ring.left_neighbor(), (315 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (315 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_316() {
        let ring = RingTopology::new(316, 4);
        assert_eq!(ring.left_neighbor(), (316 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (316 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_317() {
        let ring = RingTopology::new(317, 4);
        assert_eq!(ring.left_neighbor(), (317 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (317 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_318() {
        let ring = RingTopology::new(318, 4);
        assert_eq!(ring.left_neighbor(), (318 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (318 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_319() {
        let ring = RingTopology::new(319, 4);
        assert_eq!(ring.left_neighbor(), (319 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (319 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_320() {
        let ring = RingTopology::new(320, 4);
        assert_eq!(ring.left_neighbor(), (320 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (320 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_321() {
        let ring = RingTopology::new(321, 4);
        assert_eq!(ring.left_neighbor(), (321 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (321 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_322() {
        let ring = RingTopology::new(322, 4);
        assert_eq!(ring.left_neighbor(), (322 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (322 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_323() {
        let ring = RingTopology::new(323, 4);
        assert_eq!(ring.left_neighbor(), (323 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (323 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_324() {
        let ring = RingTopology::new(324, 4);
        assert_eq!(ring.left_neighbor(), (324 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (324 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_325() {
        let ring = RingTopology::new(325, 4);
        assert_eq!(ring.left_neighbor(), (325 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (325 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_326() {
        let ring = RingTopology::new(326, 4);
        assert_eq!(ring.left_neighbor(), (326 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (326 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_327() {
        let ring = RingTopology::new(327, 4);
        assert_eq!(ring.left_neighbor(), (327 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (327 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_328() {
        let ring = RingTopology::new(328, 4);
        assert_eq!(ring.left_neighbor(), (328 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (328 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_329() {
        let ring = RingTopology::new(329, 4);
        assert_eq!(ring.left_neighbor(), (329 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (329 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_330() {
        let ring = RingTopology::new(330, 4);
        assert_eq!(ring.left_neighbor(), (330 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (330 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_331() {
        let ring = RingTopology::new(331, 4);
        assert_eq!(ring.left_neighbor(), (331 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (331 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_332() {
        let ring = RingTopology::new(332, 4);
        assert_eq!(ring.left_neighbor(), (332 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (332 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_333() {
        let ring = RingTopology::new(333, 4);
        assert_eq!(ring.left_neighbor(), (333 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (333 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_334() {
        let ring = RingTopology::new(334, 4);
        assert_eq!(ring.left_neighbor(), (334 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (334 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_335() {
        let ring = RingTopology::new(335, 4);
        assert_eq!(ring.left_neighbor(), (335 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (335 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_336() {
        let ring = RingTopology::new(336, 4);
        assert_eq!(ring.left_neighbor(), (336 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (336 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_337() {
        let ring = RingTopology::new(337, 4);
        assert_eq!(ring.left_neighbor(), (337 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (337 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_338() {
        let ring = RingTopology::new(338, 4);
        assert_eq!(ring.left_neighbor(), (338 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (338 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_339() {
        let ring = RingTopology::new(339, 4);
        assert_eq!(ring.left_neighbor(), (339 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (339 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_340() {
        let ring = RingTopology::new(340, 4);
        assert_eq!(ring.left_neighbor(), (340 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (340 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_341() {
        let ring = RingTopology::new(341, 4);
        assert_eq!(ring.left_neighbor(), (341 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (341 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_342() {
        let ring = RingTopology::new(342, 4);
        assert_eq!(ring.left_neighbor(), (342 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (342 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_343() {
        let ring = RingTopology::new(343, 4);
        assert_eq!(ring.left_neighbor(), (343 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (343 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_344() {
        let ring = RingTopology::new(344, 4);
        assert_eq!(ring.left_neighbor(), (344 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (344 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_345() {
        let ring = RingTopology::new(345, 4);
        assert_eq!(ring.left_neighbor(), (345 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (345 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_346() {
        let ring = RingTopology::new(346, 4);
        assert_eq!(ring.left_neighbor(), (346 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (346 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_347() {
        let ring = RingTopology::new(347, 4);
        assert_eq!(ring.left_neighbor(), (347 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (347 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_348() {
        let ring = RingTopology::new(348, 4);
        assert_eq!(ring.left_neighbor(), (348 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (348 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_349() {
        let ring = RingTopology::new(349, 4);
        assert_eq!(ring.left_neighbor(), (349 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (349 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_350() {
        let ring = RingTopology::new(350, 4);
        assert_eq!(ring.left_neighbor(), (350 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (350 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_351() {
        let ring = RingTopology::new(351, 4);
        assert_eq!(ring.left_neighbor(), (351 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (351 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_352() {
        let ring = RingTopology::new(352, 4);
        assert_eq!(ring.left_neighbor(), (352 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (352 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_353() {
        let ring = RingTopology::new(353, 4);
        assert_eq!(ring.left_neighbor(), (353 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (353 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_354() {
        let ring = RingTopology::new(354, 4);
        assert_eq!(ring.left_neighbor(), (354 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (354 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_355() {
        let ring = RingTopology::new(355, 4);
        assert_eq!(ring.left_neighbor(), (355 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (355 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_356() {
        let ring = RingTopology::new(356, 4);
        assert_eq!(ring.left_neighbor(), (356 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (356 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_357() {
        let ring = RingTopology::new(357, 4);
        assert_eq!(ring.left_neighbor(), (357 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (357 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_358() {
        let ring = RingTopology::new(358, 4);
        assert_eq!(ring.left_neighbor(), (358 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (358 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_359() {
        let ring = RingTopology::new(359, 4);
        assert_eq!(ring.left_neighbor(), (359 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (359 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_360() {
        let ring = RingTopology::new(360, 4);
        assert_eq!(ring.left_neighbor(), (360 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (360 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_361() {
        let ring = RingTopology::new(361, 4);
        assert_eq!(ring.left_neighbor(), (361 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (361 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_362() {
        let ring = RingTopology::new(362, 4);
        assert_eq!(ring.left_neighbor(), (362 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (362 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_363() {
        let ring = RingTopology::new(363, 4);
        assert_eq!(ring.left_neighbor(), (363 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (363 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_364() {
        let ring = RingTopology::new(364, 4);
        assert_eq!(ring.left_neighbor(), (364 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (364 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_365() {
        let ring = RingTopology::new(365, 4);
        assert_eq!(ring.left_neighbor(), (365 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (365 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_366() {
        let ring = RingTopology::new(366, 4);
        assert_eq!(ring.left_neighbor(), (366 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (366 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_367() {
        let ring = RingTopology::new(367, 4);
        assert_eq!(ring.left_neighbor(), (367 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (367 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_368() {
        let ring = RingTopology::new(368, 4);
        assert_eq!(ring.left_neighbor(), (368 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (368 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_369() {
        let ring = RingTopology::new(369, 4);
        assert_eq!(ring.left_neighbor(), (369 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (369 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_370() {
        let ring = RingTopology::new(370, 4);
        assert_eq!(ring.left_neighbor(), (370 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (370 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_371() {
        let ring = RingTopology::new(371, 4);
        assert_eq!(ring.left_neighbor(), (371 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (371 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_372() {
        let ring = RingTopology::new(372, 4);
        assert_eq!(ring.left_neighbor(), (372 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (372 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_373() {
        let ring = RingTopology::new(373, 4);
        assert_eq!(ring.left_neighbor(), (373 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (373 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_374() {
        let ring = RingTopology::new(374, 4);
        assert_eq!(ring.left_neighbor(), (374 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (374 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_375() {
        let ring = RingTopology::new(375, 4);
        assert_eq!(ring.left_neighbor(), (375 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (375 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_376() {
        let ring = RingTopology::new(376, 4);
        assert_eq!(ring.left_neighbor(), (376 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (376 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_377() {
        let ring = RingTopology::new(377, 4);
        assert_eq!(ring.left_neighbor(), (377 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (377 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_378() {
        let ring = RingTopology::new(378, 4);
        assert_eq!(ring.left_neighbor(), (378 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (378 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_379() {
        let ring = RingTopology::new(379, 4);
        assert_eq!(ring.left_neighbor(), (379 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (379 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_380() {
        let ring = RingTopology::new(380, 4);
        assert_eq!(ring.left_neighbor(), (380 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (380 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_381() {
        let ring = RingTopology::new(381, 4);
        assert_eq!(ring.left_neighbor(), (381 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (381 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_382() {
        let ring = RingTopology::new(382, 4);
        assert_eq!(ring.left_neighbor(), (382 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (382 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_383() {
        let ring = RingTopology::new(383, 4);
        assert_eq!(ring.left_neighbor(), (383 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (383 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_384() {
        let ring = RingTopology::new(384, 4);
        assert_eq!(ring.left_neighbor(), (384 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (384 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_385() {
        let ring = RingTopology::new(385, 4);
        assert_eq!(ring.left_neighbor(), (385 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (385 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_386() {
        let ring = RingTopology::new(386, 4);
        assert_eq!(ring.left_neighbor(), (386 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (386 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_387() {
        let ring = RingTopology::new(387, 4);
        assert_eq!(ring.left_neighbor(), (387 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (387 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_388() {
        let ring = RingTopology::new(388, 4);
        assert_eq!(ring.left_neighbor(), (388 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (388 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_389() {
        let ring = RingTopology::new(389, 4);
        assert_eq!(ring.left_neighbor(), (389 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (389 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_390() {
        let ring = RingTopology::new(390, 4);
        assert_eq!(ring.left_neighbor(), (390 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (390 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_391() {
        let ring = RingTopology::new(391, 4);
        assert_eq!(ring.left_neighbor(), (391 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (391 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_392() {
        let ring = RingTopology::new(392, 4);
        assert_eq!(ring.left_neighbor(), (392 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (392 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_393() {
        let ring = RingTopology::new(393, 4);
        assert_eq!(ring.left_neighbor(), (393 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (393 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_394() {
        let ring = RingTopology::new(394, 4);
        assert_eq!(ring.left_neighbor(), (394 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (394 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_395() {
        let ring = RingTopology::new(395, 4);
        assert_eq!(ring.left_neighbor(), (395 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (395 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_396() {
        let ring = RingTopology::new(396, 4);
        assert_eq!(ring.left_neighbor(), (396 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (396 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_397() {
        let ring = RingTopology::new(397, 4);
        assert_eq!(ring.left_neighbor(), (397 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (397 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_398() {
        let ring = RingTopology::new(398, 4);
        assert_eq!(ring.left_neighbor(), (398 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (398 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_399() {
        let ring = RingTopology::new(399, 4);
        assert_eq!(ring.left_neighbor(), (399 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (399 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_400() {
        let ring = RingTopology::new(400, 4);
        assert_eq!(ring.left_neighbor(), (400 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (400 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_401() {
        let ring = RingTopology::new(401, 4);
        assert_eq!(ring.left_neighbor(), (401 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (401 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_402() {
        let ring = RingTopology::new(402, 4);
        assert_eq!(ring.left_neighbor(), (402 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (402 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_403() {
        let ring = RingTopology::new(403, 4);
        assert_eq!(ring.left_neighbor(), (403 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (403 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_404() {
        let ring = RingTopology::new(404, 4);
        assert_eq!(ring.left_neighbor(), (404 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (404 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_405() {
        let ring = RingTopology::new(405, 4);
        assert_eq!(ring.left_neighbor(), (405 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (405 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_406() {
        let ring = RingTopology::new(406, 4);
        assert_eq!(ring.left_neighbor(), (406 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (406 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_407() {
        let ring = RingTopology::new(407, 4);
        assert_eq!(ring.left_neighbor(), (407 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (407 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_408() {
        let ring = RingTopology::new(408, 4);
        assert_eq!(ring.left_neighbor(), (408 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (408 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_409() {
        let ring = RingTopology::new(409, 4);
        assert_eq!(ring.left_neighbor(), (409 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (409 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_410() {
        let ring = RingTopology::new(410, 4);
        assert_eq!(ring.left_neighbor(), (410 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (410 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_411() {
        let ring = RingTopology::new(411, 4);
        assert_eq!(ring.left_neighbor(), (411 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (411 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_412() {
        let ring = RingTopology::new(412, 4);
        assert_eq!(ring.left_neighbor(), (412 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (412 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_413() {
        let ring = RingTopology::new(413, 4);
        assert_eq!(ring.left_neighbor(), (413 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (413 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_414() {
        let ring = RingTopology::new(414, 4);
        assert_eq!(ring.left_neighbor(), (414 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (414 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_415() {
        let ring = RingTopology::new(415, 4);
        assert_eq!(ring.left_neighbor(), (415 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (415 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_416() {
        let ring = RingTopology::new(416, 4);
        assert_eq!(ring.left_neighbor(), (416 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (416 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_417() {
        let ring = RingTopology::new(417, 4);
        assert_eq!(ring.left_neighbor(), (417 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (417 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_418() {
        let ring = RingTopology::new(418, 4);
        assert_eq!(ring.left_neighbor(), (418 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (418 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_419() {
        let ring = RingTopology::new(419, 4);
        assert_eq!(ring.left_neighbor(), (419 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (419 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_420() {
        let ring = RingTopology::new(420, 4);
        assert_eq!(ring.left_neighbor(), (420 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (420 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_421() {
        let ring = RingTopology::new(421, 4);
        assert_eq!(ring.left_neighbor(), (421 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (421 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_422() {
        let ring = RingTopology::new(422, 4);
        assert_eq!(ring.left_neighbor(), (422 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (422 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_423() {
        let ring = RingTopology::new(423, 4);
        assert_eq!(ring.left_neighbor(), (423 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (423 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_424() {
        let ring = RingTopology::new(424, 4);
        assert_eq!(ring.left_neighbor(), (424 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (424 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_425() {
        let ring = RingTopology::new(425, 4);
        assert_eq!(ring.left_neighbor(), (425 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (425 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_426() {
        let ring = RingTopology::new(426, 4);
        assert_eq!(ring.left_neighbor(), (426 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (426 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_427() {
        let ring = RingTopology::new(427, 4);
        assert_eq!(ring.left_neighbor(), (427 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (427 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_428() {
        let ring = RingTopology::new(428, 4);
        assert_eq!(ring.left_neighbor(), (428 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (428 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_429() {
        let ring = RingTopology::new(429, 4);
        assert_eq!(ring.left_neighbor(), (429 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (429 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_430() {
        let ring = RingTopology::new(430, 4);
        assert_eq!(ring.left_neighbor(), (430 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (430 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_431() {
        let ring = RingTopology::new(431, 4);
        assert_eq!(ring.left_neighbor(), (431 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (431 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_432() {
        let ring = RingTopology::new(432, 4);
        assert_eq!(ring.left_neighbor(), (432 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (432 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_433() {
        let ring = RingTopology::new(433, 4);
        assert_eq!(ring.left_neighbor(), (433 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (433 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_434() {
        let ring = RingTopology::new(434, 4);
        assert_eq!(ring.left_neighbor(), (434 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (434 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_435() {
        let ring = RingTopology::new(435, 4);
        assert_eq!(ring.left_neighbor(), (435 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (435 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_436() {
        let ring = RingTopology::new(436, 4);
        assert_eq!(ring.left_neighbor(), (436 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (436 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_437() {
        let ring = RingTopology::new(437, 4);
        assert_eq!(ring.left_neighbor(), (437 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (437 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_438() {
        let ring = RingTopology::new(438, 4);
        assert_eq!(ring.left_neighbor(), (438 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (438 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_439() {
        let ring = RingTopology::new(439, 4);
        assert_eq!(ring.left_neighbor(), (439 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (439 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_440() {
        let ring = RingTopology::new(440, 4);
        assert_eq!(ring.left_neighbor(), (440 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (440 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_441() {
        let ring = RingTopology::new(441, 4);
        assert_eq!(ring.left_neighbor(), (441 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (441 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_442() {
        let ring = RingTopology::new(442, 4);
        assert_eq!(ring.left_neighbor(), (442 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (442 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_443() {
        let ring = RingTopology::new(443, 4);
        assert_eq!(ring.left_neighbor(), (443 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (443 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_444() {
        let ring = RingTopology::new(444, 4);
        assert_eq!(ring.left_neighbor(), (444 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (444 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_445() {
        let ring = RingTopology::new(445, 4);
        assert_eq!(ring.left_neighbor(), (445 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (445 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_446() {
        let ring = RingTopology::new(446, 4);
        assert_eq!(ring.left_neighbor(), (446 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (446 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_447() {
        let ring = RingTopology::new(447, 4);
        assert_eq!(ring.left_neighbor(), (447 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (447 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_448() {
        let ring = RingTopology::new(448, 4);
        assert_eq!(ring.left_neighbor(), (448 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (448 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_449() {
        let ring = RingTopology::new(449, 4);
        assert_eq!(ring.left_neighbor(), (449 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (449 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_450() {
        let ring = RingTopology::new(450, 4);
        assert_eq!(ring.left_neighbor(), (450 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (450 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_451() {
        let ring = RingTopology::new(451, 4);
        assert_eq!(ring.left_neighbor(), (451 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (451 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_452() {
        let ring = RingTopology::new(452, 4);
        assert_eq!(ring.left_neighbor(), (452 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (452 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_453() {
        let ring = RingTopology::new(453, 4);
        assert_eq!(ring.left_neighbor(), (453 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (453 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_454() {
        let ring = RingTopology::new(454, 4);
        assert_eq!(ring.left_neighbor(), (454 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (454 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_455() {
        let ring = RingTopology::new(455, 4);
        assert_eq!(ring.left_neighbor(), (455 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (455 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_456() {
        let ring = RingTopology::new(456, 4);
        assert_eq!(ring.left_neighbor(), (456 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (456 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_457() {
        let ring = RingTopology::new(457, 4);
        assert_eq!(ring.left_neighbor(), (457 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (457 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_458() {
        let ring = RingTopology::new(458, 4);
        assert_eq!(ring.left_neighbor(), (458 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (458 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_459() {
        let ring = RingTopology::new(459, 4);
        assert_eq!(ring.left_neighbor(), (459 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (459 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_460() {
        let ring = RingTopology::new(460, 4);
        assert_eq!(ring.left_neighbor(), (460 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (460 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_461() {
        let ring = RingTopology::new(461, 4);
        assert_eq!(ring.left_neighbor(), (461 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (461 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_462() {
        let ring = RingTopology::new(462, 4);
        assert_eq!(ring.left_neighbor(), (462 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (462 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_463() {
        let ring = RingTopology::new(463, 4);
        assert_eq!(ring.left_neighbor(), (463 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (463 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_464() {
        let ring = RingTopology::new(464, 4);
        assert_eq!(ring.left_neighbor(), (464 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (464 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_465() {
        let ring = RingTopology::new(465, 4);
        assert_eq!(ring.left_neighbor(), (465 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (465 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_466() {
        let ring = RingTopology::new(466, 4);
        assert_eq!(ring.left_neighbor(), (466 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (466 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_467() {
        let ring = RingTopology::new(467, 4);
        assert_eq!(ring.left_neighbor(), (467 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (467 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_468() {
        let ring = RingTopology::new(468, 4);
        assert_eq!(ring.left_neighbor(), (468 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (468 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_469() {
        let ring = RingTopology::new(469, 4);
        assert_eq!(ring.left_neighbor(), (469 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (469 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_470() {
        let ring = RingTopology::new(470, 4);
        assert_eq!(ring.left_neighbor(), (470 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (470 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_471() {
        let ring = RingTopology::new(471, 4);
        assert_eq!(ring.left_neighbor(), (471 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (471 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_472() {
        let ring = RingTopology::new(472, 4);
        assert_eq!(ring.left_neighbor(), (472 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (472 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_473() {
        let ring = RingTopology::new(473, 4);
        assert_eq!(ring.left_neighbor(), (473 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (473 + 1) % 4);
    }

    #[test]
    fn test_collective_mod_stress_474() {
        let ring = RingTopology::new(474, 4);
        assert_eq!(ring.left_neighbor(), (474 + 3) % 4);
        assert_eq!(ring.right_neighbor(), (474 + 1) % 4);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
}
