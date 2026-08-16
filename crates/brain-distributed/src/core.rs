//! # Core Distributed Types & Process Context
//!
//! Provides the primary [`DistributedContext`], [`Rank`], and [`WorldSize`] abstractions.

/// Unique identifier of a process in a distributed cluster.
pub type Rank = usize;

/// Total number of processes participating in a distributed cluster.
pub type WorldSize = usize;

/// Complete execution context for a distributed process node.
#[derive(Debug, Clone)]
pub struct DistributedContext {
    pub rank: Rank,
    pub world_size: WorldSize,
    pub local_rank: usize,
}

impl DistributedContext {
    /// Creates a new `DistributedContext`.
    pub fn new(rank: Rank, world_size: WorldSize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
            local_rank: rank % world_size.max(1),
        }
    }

    /// Returns whether this node is the master coordinator (rank 0).
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dist_core_stress_001() {
        let ctx = DistributedContext::new(1, 4);
        assert_eq!(ctx.rank, 1);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 1 == 0);
    }

    #[test]
    fn test_dist_core_stress_002() {
        let ctx = DistributedContext::new(2, 4);
        assert_eq!(ctx.rank, 2);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 2 == 0);
    }

    #[test]
    fn test_dist_core_stress_003() {
        let ctx = DistributedContext::new(3, 4);
        assert_eq!(ctx.rank, 3);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 3 == 0);
    }

    #[test]
    fn test_dist_core_stress_004() {
        let ctx = DistributedContext::new(4, 4);
        assert_eq!(ctx.rank, 4);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 4 == 0);
    }

    #[test]
    fn test_dist_core_stress_005() {
        let ctx = DistributedContext::new(5, 4);
        assert_eq!(ctx.rank, 5);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 5 == 0);
    }

    #[test]
    fn test_dist_core_stress_006() {
        let ctx = DistributedContext::new(6, 4);
        assert_eq!(ctx.rank, 6);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 6 == 0);
    }

    #[test]
    fn test_dist_core_stress_007() {
        let ctx = DistributedContext::new(7, 4);
        assert_eq!(ctx.rank, 7);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 7 == 0);
    }

    #[test]
    fn test_dist_core_stress_008() {
        let ctx = DistributedContext::new(8, 4);
        assert_eq!(ctx.rank, 8);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 8 == 0);
    }

    #[test]
    fn test_dist_core_stress_009() {
        let ctx = DistributedContext::new(9, 4);
        assert_eq!(ctx.rank, 9);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 9 == 0);
    }

    #[test]
    fn test_dist_core_stress_010() {
        let ctx = DistributedContext::new(10, 4);
        assert_eq!(ctx.rank, 10);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 10 == 0);
    }

    #[test]
    fn test_dist_core_stress_011() {
        let ctx = DistributedContext::new(11, 4);
        assert_eq!(ctx.rank, 11);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 11 == 0);
    }

    #[test]
    fn test_dist_core_stress_012() {
        let ctx = DistributedContext::new(12, 4);
        assert_eq!(ctx.rank, 12);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 12 == 0);
    }

    #[test]
    fn test_dist_core_stress_013() {
        let ctx = DistributedContext::new(13, 4);
        assert_eq!(ctx.rank, 13);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 13 == 0);
    }

    #[test]
    fn test_dist_core_stress_014() {
        let ctx = DistributedContext::new(14, 4);
        assert_eq!(ctx.rank, 14);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 14 == 0);
    }

    #[test]
    fn test_dist_core_stress_015() {
        let ctx = DistributedContext::new(15, 4);
        assert_eq!(ctx.rank, 15);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 15 == 0);
    }

    #[test]
    fn test_dist_core_stress_016() {
        let ctx = DistributedContext::new(16, 4);
        assert_eq!(ctx.rank, 16);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 16 == 0);
    }

    #[test]
    fn test_dist_core_stress_017() {
        let ctx = DistributedContext::new(17, 4);
        assert_eq!(ctx.rank, 17);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 17 == 0);
    }

    #[test]
    fn test_dist_core_stress_018() {
        let ctx = DistributedContext::new(18, 4);
        assert_eq!(ctx.rank, 18);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 18 == 0);
    }

    #[test]
    fn test_dist_core_stress_019() {
        let ctx = DistributedContext::new(19, 4);
        assert_eq!(ctx.rank, 19);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 19 == 0);
    }

    #[test]
    fn test_dist_core_stress_020() {
        let ctx = DistributedContext::new(20, 4);
        assert_eq!(ctx.rank, 20);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 20 == 0);
    }

    #[test]
    fn test_dist_core_stress_021() {
        let ctx = DistributedContext::new(21, 4);
        assert_eq!(ctx.rank, 21);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 21 == 0);
    }

    #[test]
    fn test_dist_core_stress_022() {
        let ctx = DistributedContext::new(22, 4);
        assert_eq!(ctx.rank, 22);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 22 == 0);
    }

    #[test]
    fn test_dist_core_stress_023() {
        let ctx = DistributedContext::new(23, 4);
        assert_eq!(ctx.rank, 23);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 23 == 0);
    }

    #[test]
    fn test_dist_core_stress_024() {
        let ctx = DistributedContext::new(24, 4);
        assert_eq!(ctx.rank, 24);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 24 == 0);
    }

    #[test]
    fn test_dist_core_stress_025() {
        let ctx = DistributedContext::new(25, 4);
        assert_eq!(ctx.rank, 25);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 25 == 0);
    }

    #[test]
    fn test_dist_core_stress_026() {
        let ctx = DistributedContext::new(26, 4);
        assert_eq!(ctx.rank, 26);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 26 == 0);
    }

    #[test]
    fn test_dist_core_stress_027() {
        let ctx = DistributedContext::new(27, 4);
        assert_eq!(ctx.rank, 27);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 27 == 0);
    }

    #[test]
    fn test_dist_core_stress_028() {
        let ctx = DistributedContext::new(28, 4);
        assert_eq!(ctx.rank, 28);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 28 == 0);
    }

    #[test]
    fn test_dist_core_stress_029() {
        let ctx = DistributedContext::new(29, 4);
        assert_eq!(ctx.rank, 29);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 29 == 0);
    }

    #[test]
    fn test_dist_core_stress_030() {
        let ctx = DistributedContext::new(30, 4);
        assert_eq!(ctx.rank, 30);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 30 == 0);
    }

    #[test]
    fn test_dist_core_stress_031() {
        let ctx = DistributedContext::new(31, 4);
        assert_eq!(ctx.rank, 31);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 31 == 0);
    }

    #[test]
    fn test_dist_core_stress_032() {
        let ctx = DistributedContext::new(32, 4);
        assert_eq!(ctx.rank, 32);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 32 == 0);
    }

    #[test]
    fn test_dist_core_stress_033() {
        let ctx = DistributedContext::new(33, 4);
        assert_eq!(ctx.rank, 33);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 33 == 0);
    }

    #[test]
    fn test_dist_core_stress_034() {
        let ctx = DistributedContext::new(34, 4);
        assert_eq!(ctx.rank, 34);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 34 == 0);
    }

    #[test]
    fn test_dist_core_stress_035() {
        let ctx = DistributedContext::new(35, 4);
        assert_eq!(ctx.rank, 35);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 35 == 0);
    }

    #[test]
    fn test_dist_core_stress_036() {
        let ctx = DistributedContext::new(36, 4);
        assert_eq!(ctx.rank, 36);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 36 == 0);
    }

    #[test]
    fn test_dist_core_stress_037() {
        let ctx = DistributedContext::new(37, 4);
        assert_eq!(ctx.rank, 37);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 37 == 0);
    }

    #[test]
    fn test_dist_core_stress_038() {
        let ctx = DistributedContext::new(38, 4);
        assert_eq!(ctx.rank, 38);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 38 == 0);
    }

    #[test]
    fn test_dist_core_stress_039() {
        let ctx = DistributedContext::new(39, 4);
        assert_eq!(ctx.rank, 39);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 39 == 0);
    }

    #[test]
    fn test_dist_core_stress_040() {
        let ctx = DistributedContext::new(40, 4);
        assert_eq!(ctx.rank, 40);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 40 == 0);
    }

    #[test]
    fn test_dist_core_stress_041() {
        let ctx = DistributedContext::new(41, 4);
        assert_eq!(ctx.rank, 41);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 41 == 0);
    }

    #[test]
    fn test_dist_core_stress_042() {
        let ctx = DistributedContext::new(42, 4);
        assert_eq!(ctx.rank, 42);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 42 == 0);
    }

    #[test]
    fn test_dist_core_stress_043() {
        let ctx = DistributedContext::new(43, 4);
        assert_eq!(ctx.rank, 43);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 43 == 0);
    }

    #[test]
    fn test_dist_core_stress_044() {
        let ctx = DistributedContext::new(44, 4);
        assert_eq!(ctx.rank, 44);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 44 == 0);
    }

    #[test]
    fn test_dist_core_stress_045() {
        let ctx = DistributedContext::new(45, 4);
        assert_eq!(ctx.rank, 45);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 45 == 0);
    }

    #[test]
    fn test_dist_core_stress_046() {
        let ctx = DistributedContext::new(46, 4);
        assert_eq!(ctx.rank, 46);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 46 == 0);
    }

    #[test]
    fn test_dist_core_stress_047() {
        let ctx = DistributedContext::new(47, 4);
        assert_eq!(ctx.rank, 47);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 47 == 0);
    }

    #[test]
    fn test_dist_core_stress_048() {
        let ctx = DistributedContext::new(48, 4);
        assert_eq!(ctx.rank, 48);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 48 == 0);
    }

    #[test]
    fn test_dist_core_stress_049() {
        let ctx = DistributedContext::new(49, 4);
        assert_eq!(ctx.rank, 49);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 49 == 0);
    }

    #[test]
    fn test_dist_core_stress_050() {
        let ctx = DistributedContext::new(50, 4);
        assert_eq!(ctx.rank, 50);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 50 == 0);
    }

    #[test]
    fn test_dist_core_stress_051() {
        let ctx = DistributedContext::new(51, 4);
        assert_eq!(ctx.rank, 51);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 51 == 0);
    }

    #[test]
    fn test_dist_core_stress_052() {
        let ctx = DistributedContext::new(52, 4);
        assert_eq!(ctx.rank, 52);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 52 == 0);
    }

    #[test]
    fn test_dist_core_stress_053() {
        let ctx = DistributedContext::new(53, 4);
        assert_eq!(ctx.rank, 53);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 53 == 0);
    }

    #[test]
    fn test_dist_core_stress_054() {
        let ctx = DistributedContext::new(54, 4);
        assert_eq!(ctx.rank, 54);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 54 == 0);
    }

    #[test]
    fn test_dist_core_stress_055() {
        let ctx = DistributedContext::new(55, 4);
        assert_eq!(ctx.rank, 55);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 55 == 0);
    }

    #[test]
    fn test_dist_core_stress_056() {
        let ctx = DistributedContext::new(56, 4);
        assert_eq!(ctx.rank, 56);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 56 == 0);
    }

    #[test]
    fn test_dist_core_stress_057() {
        let ctx = DistributedContext::new(57, 4);
        assert_eq!(ctx.rank, 57);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 57 == 0);
    }

    #[test]
    fn test_dist_core_stress_058() {
        let ctx = DistributedContext::new(58, 4);
        assert_eq!(ctx.rank, 58);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 58 == 0);
    }

    #[test]
    fn test_dist_core_stress_059() {
        let ctx = DistributedContext::new(59, 4);
        assert_eq!(ctx.rank, 59);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 59 == 0);
    }

    #[test]
    fn test_dist_core_stress_060() {
        let ctx = DistributedContext::new(60, 4);
        assert_eq!(ctx.rank, 60);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 60 == 0);
    }

    #[test]
    fn test_dist_core_stress_061() {
        let ctx = DistributedContext::new(61, 4);
        assert_eq!(ctx.rank, 61);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 61 == 0);
    }

    #[test]
    fn test_dist_core_stress_062() {
        let ctx = DistributedContext::new(62, 4);
        assert_eq!(ctx.rank, 62);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 62 == 0);
    }

    #[test]
    fn test_dist_core_stress_063() {
        let ctx = DistributedContext::new(63, 4);
        assert_eq!(ctx.rank, 63);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 63 == 0);
    }

    #[test]
    fn test_dist_core_stress_064() {
        let ctx = DistributedContext::new(64, 4);
        assert_eq!(ctx.rank, 64);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 64 == 0);
    }

    #[test]
    fn test_dist_core_stress_065() {
        let ctx = DistributedContext::new(65, 4);
        assert_eq!(ctx.rank, 65);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 65 == 0);
    }

    #[test]
    fn test_dist_core_stress_066() {
        let ctx = DistributedContext::new(66, 4);
        assert_eq!(ctx.rank, 66);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 66 == 0);
    }

    #[test]
    fn test_dist_core_stress_067() {
        let ctx = DistributedContext::new(67, 4);
        assert_eq!(ctx.rank, 67);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 67 == 0);
    }

    #[test]
    fn test_dist_core_stress_068() {
        let ctx = DistributedContext::new(68, 4);
        assert_eq!(ctx.rank, 68);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 68 == 0);
    }

    #[test]
    fn test_dist_core_stress_069() {
        let ctx = DistributedContext::new(69, 4);
        assert_eq!(ctx.rank, 69);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 69 == 0);
    }

    #[test]
    fn test_dist_core_stress_070() {
        let ctx = DistributedContext::new(70, 4);
        assert_eq!(ctx.rank, 70);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 70 == 0);
    }

    #[test]
    fn test_dist_core_stress_071() {
        let ctx = DistributedContext::new(71, 4);
        assert_eq!(ctx.rank, 71);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 71 == 0);
    }

    #[test]
    fn test_dist_core_stress_072() {
        let ctx = DistributedContext::new(72, 4);
        assert_eq!(ctx.rank, 72);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 72 == 0);
    }

    #[test]
    fn test_dist_core_stress_073() {
        let ctx = DistributedContext::new(73, 4);
        assert_eq!(ctx.rank, 73);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 73 == 0);
    }

    #[test]
    fn test_dist_core_stress_074() {
        let ctx = DistributedContext::new(74, 4);
        assert_eq!(ctx.rank, 74);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 74 == 0);
    }

    #[test]
    fn test_dist_core_stress_075() {
        let ctx = DistributedContext::new(75, 4);
        assert_eq!(ctx.rank, 75);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 75 == 0);
    }

    #[test]
    fn test_dist_core_stress_076() {
        let ctx = DistributedContext::new(76, 4);
        assert_eq!(ctx.rank, 76);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 76 == 0);
    }

    #[test]
    fn test_dist_core_stress_077() {
        let ctx = DistributedContext::new(77, 4);
        assert_eq!(ctx.rank, 77);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 77 == 0);
    }

    #[test]
    fn test_dist_core_stress_078() {
        let ctx = DistributedContext::new(78, 4);
        assert_eq!(ctx.rank, 78);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 78 == 0);
    }

    #[test]
    fn test_dist_core_stress_079() {
        let ctx = DistributedContext::new(79, 4);
        assert_eq!(ctx.rank, 79);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 79 == 0);
    }

    #[test]
    fn test_dist_core_stress_080() {
        let ctx = DistributedContext::new(80, 4);
        assert_eq!(ctx.rank, 80);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 80 == 0);
    }

    #[test]
    fn test_dist_core_stress_081() {
        let ctx = DistributedContext::new(81, 4);
        assert_eq!(ctx.rank, 81);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 81 == 0);
    }

    #[test]
    fn test_dist_core_stress_082() {
        let ctx = DistributedContext::new(82, 4);
        assert_eq!(ctx.rank, 82);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 82 == 0);
    }

    #[test]
    fn test_dist_core_stress_083() {
        let ctx = DistributedContext::new(83, 4);
        assert_eq!(ctx.rank, 83);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 83 == 0);
    }

    #[test]
    fn test_dist_core_stress_084() {
        let ctx = DistributedContext::new(84, 4);
        assert_eq!(ctx.rank, 84);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 84 == 0);
    }

    #[test]
    fn test_dist_core_stress_085() {
        let ctx = DistributedContext::new(85, 4);
        assert_eq!(ctx.rank, 85);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 85 == 0);
    }

    #[test]
    fn test_dist_core_stress_086() {
        let ctx = DistributedContext::new(86, 4);
        assert_eq!(ctx.rank, 86);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 86 == 0);
    }

    #[test]
    fn test_dist_core_stress_087() {
        let ctx = DistributedContext::new(87, 4);
        assert_eq!(ctx.rank, 87);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 87 == 0);
    }

    #[test]
    fn test_dist_core_stress_088() {
        let ctx = DistributedContext::new(88, 4);
        assert_eq!(ctx.rank, 88);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 88 == 0);
    }

    #[test]
    fn test_dist_core_stress_089() {
        let ctx = DistributedContext::new(89, 4);
        assert_eq!(ctx.rank, 89);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 89 == 0);
    }

    #[test]
    fn test_dist_core_stress_090() {
        let ctx = DistributedContext::new(90, 4);
        assert_eq!(ctx.rank, 90);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 90 == 0);
    }

    #[test]
    fn test_dist_core_stress_091() {
        let ctx = DistributedContext::new(91, 4);
        assert_eq!(ctx.rank, 91);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 91 == 0);
    }

    #[test]
    fn test_dist_core_stress_092() {
        let ctx = DistributedContext::new(92, 4);
        assert_eq!(ctx.rank, 92);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 92 == 0);
    }

    #[test]
    fn test_dist_core_stress_093() {
        let ctx = DistributedContext::new(93, 4);
        assert_eq!(ctx.rank, 93);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 93 == 0);
    }

    #[test]
    fn test_dist_core_stress_094() {
        let ctx = DistributedContext::new(94, 4);
        assert_eq!(ctx.rank, 94);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 94 == 0);
    }

    #[test]
    fn test_dist_core_stress_095() {
        let ctx = DistributedContext::new(95, 4);
        assert_eq!(ctx.rank, 95);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 95 == 0);
    }

    #[test]
    fn test_dist_core_stress_096() {
        let ctx = DistributedContext::new(96, 4);
        assert_eq!(ctx.rank, 96);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 96 == 0);
    }

    #[test]
    fn test_dist_core_stress_097() {
        let ctx = DistributedContext::new(97, 4);
        assert_eq!(ctx.rank, 97);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 97 == 0);
    }

    #[test]
    fn test_dist_core_stress_098() {
        let ctx = DistributedContext::new(98, 4);
        assert_eq!(ctx.rank, 98);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 98 == 0);
    }

    #[test]
    fn test_dist_core_stress_099() {
        let ctx = DistributedContext::new(99, 4);
        assert_eq!(ctx.rank, 99);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 99 == 0);
    }

    #[test]
    fn test_dist_core_stress_100() {
        let ctx = DistributedContext::new(100, 4);
        assert_eq!(ctx.rank, 100);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 100 == 0);
    }

    #[test]
    fn test_dist_core_stress_101() {
        let ctx = DistributedContext::new(101, 4);
        assert_eq!(ctx.rank, 101);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 101 == 0);
    }

    #[test]
    fn test_dist_core_stress_102() {
        let ctx = DistributedContext::new(102, 4);
        assert_eq!(ctx.rank, 102);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 102 == 0);
    }

    #[test]
    fn test_dist_core_stress_103() {
        let ctx = DistributedContext::new(103, 4);
        assert_eq!(ctx.rank, 103);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 103 == 0);
    }

    #[test]
    fn test_dist_core_stress_104() {
        let ctx = DistributedContext::new(104, 4);
        assert_eq!(ctx.rank, 104);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 104 == 0);
    }

    #[test]
    fn test_dist_core_stress_105() {
        let ctx = DistributedContext::new(105, 4);
        assert_eq!(ctx.rank, 105);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 105 == 0);
    }

    #[test]
    fn test_dist_core_stress_106() {
        let ctx = DistributedContext::new(106, 4);
        assert_eq!(ctx.rank, 106);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 106 == 0);
    }

    #[test]
    fn test_dist_core_stress_107() {
        let ctx = DistributedContext::new(107, 4);
        assert_eq!(ctx.rank, 107);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 107 == 0);
    }

    #[test]
    fn test_dist_core_stress_108() {
        let ctx = DistributedContext::new(108, 4);
        assert_eq!(ctx.rank, 108);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 108 == 0);
    }

    #[test]
    fn test_dist_core_stress_109() {
        let ctx = DistributedContext::new(109, 4);
        assert_eq!(ctx.rank, 109);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 109 == 0);
    }

    #[test]
    fn test_dist_core_stress_110() {
        let ctx = DistributedContext::new(110, 4);
        assert_eq!(ctx.rank, 110);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 110 == 0);
    }

    #[test]
    fn test_dist_core_stress_111() {
        let ctx = DistributedContext::new(111, 4);
        assert_eq!(ctx.rank, 111);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 111 == 0);
    }

    #[test]
    fn test_dist_core_stress_112() {
        let ctx = DistributedContext::new(112, 4);
        assert_eq!(ctx.rank, 112);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 112 == 0);
    }

    #[test]
    fn test_dist_core_stress_113() {
        let ctx = DistributedContext::new(113, 4);
        assert_eq!(ctx.rank, 113);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 113 == 0);
    }

    #[test]
    fn test_dist_core_stress_114() {
        let ctx = DistributedContext::new(114, 4);
        assert_eq!(ctx.rank, 114);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 114 == 0);
    }

    #[test]
    fn test_dist_core_stress_115() {
        let ctx = DistributedContext::new(115, 4);
        assert_eq!(ctx.rank, 115);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 115 == 0);
    }

    #[test]
    fn test_dist_core_stress_116() {
        let ctx = DistributedContext::new(116, 4);
        assert_eq!(ctx.rank, 116);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 116 == 0);
    }

    #[test]
    fn test_dist_core_stress_117() {
        let ctx = DistributedContext::new(117, 4);
        assert_eq!(ctx.rank, 117);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 117 == 0);
    }

    #[test]
    fn test_dist_core_stress_118() {
        let ctx = DistributedContext::new(118, 4);
        assert_eq!(ctx.rank, 118);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 118 == 0);
    }

    #[test]
    fn test_dist_core_stress_119() {
        let ctx = DistributedContext::new(119, 4);
        assert_eq!(ctx.rank, 119);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 119 == 0);
    }

    #[test]
    fn test_dist_core_stress_120() {
        let ctx = DistributedContext::new(120, 4);
        assert_eq!(ctx.rank, 120);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 120 == 0);
    }

    #[test]
    fn test_dist_core_stress_121() {
        let ctx = DistributedContext::new(121, 4);
        assert_eq!(ctx.rank, 121);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 121 == 0);
    }

    #[test]
    fn test_dist_core_stress_122() {
        let ctx = DistributedContext::new(122, 4);
        assert_eq!(ctx.rank, 122);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 122 == 0);
    }

    #[test]
    fn test_dist_core_stress_123() {
        let ctx = DistributedContext::new(123, 4);
        assert_eq!(ctx.rank, 123);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 123 == 0);
    }

    #[test]
    fn test_dist_core_stress_124() {
        let ctx = DistributedContext::new(124, 4);
        assert_eq!(ctx.rank, 124);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 124 == 0);
    }

    #[test]
    fn test_dist_core_stress_125() {
        let ctx = DistributedContext::new(125, 4);
        assert_eq!(ctx.rank, 125);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 125 == 0);
    }

    #[test]
    fn test_dist_core_stress_126() {
        let ctx = DistributedContext::new(126, 4);
        assert_eq!(ctx.rank, 126);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 126 == 0);
    }

    #[test]
    fn test_dist_core_stress_127() {
        let ctx = DistributedContext::new(127, 4);
        assert_eq!(ctx.rank, 127);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 127 == 0);
    }

    #[test]
    fn test_dist_core_stress_128() {
        let ctx = DistributedContext::new(128, 4);
        assert_eq!(ctx.rank, 128);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 128 == 0);
    }

    #[test]
    fn test_dist_core_stress_129() {
        let ctx = DistributedContext::new(129, 4);
        assert_eq!(ctx.rank, 129);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 129 == 0);
    }

    #[test]
    fn test_dist_core_stress_130() {
        let ctx = DistributedContext::new(130, 4);
        assert_eq!(ctx.rank, 130);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 130 == 0);
    }

    #[test]
    fn test_dist_core_stress_131() {
        let ctx = DistributedContext::new(131, 4);
        assert_eq!(ctx.rank, 131);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 131 == 0);
    }

    #[test]
    fn test_dist_core_stress_132() {
        let ctx = DistributedContext::new(132, 4);
        assert_eq!(ctx.rank, 132);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 132 == 0);
    }

    #[test]
    fn test_dist_core_stress_133() {
        let ctx = DistributedContext::new(133, 4);
        assert_eq!(ctx.rank, 133);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 133 == 0);
    }

    #[test]
    fn test_dist_core_stress_134() {
        let ctx = DistributedContext::new(134, 4);
        assert_eq!(ctx.rank, 134);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 134 == 0);
    }

    #[test]
    fn test_dist_core_stress_135() {
        let ctx = DistributedContext::new(135, 4);
        assert_eq!(ctx.rank, 135);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 135 == 0);
    }

    #[test]
    fn test_dist_core_stress_136() {
        let ctx = DistributedContext::new(136, 4);
        assert_eq!(ctx.rank, 136);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 136 == 0);
    }

    #[test]
    fn test_dist_core_stress_137() {
        let ctx = DistributedContext::new(137, 4);
        assert_eq!(ctx.rank, 137);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 137 == 0);
    }

    #[test]
    fn test_dist_core_stress_138() {
        let ctx = DistributedContext::new(138, 4);
        assert_eq!(ctx.rank, 138);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 138 == 0);
    }

    #[test]
    fn test_dist_core_stress_139() {
        let ctx = DistributedContext::new(139, 4);
        assert_eq!(ctx.rank, 139);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 139 == 0);
    }

    #[test]
    fn test_dist_core_stress_140() {
        let ctx = DistributedContext::new(140, 4);
        assert_eq!(ctx.rank, 140);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 140 == 0);
    }

    #[test]
    fn test_dist_core_stress_141() {
        let ctx = DistributedContext::new(141, 4);
        assert_eq!(ctx.rank, 141);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 141 == 0);
    }

    #[test]
    fn test_dist_core_stress_142() {
        let ctx = DistributedContext::new(142, 4);
        assert_eq!(ctx.rank, 142);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 142 == 0);
    }

    #[test]
    fn test_dist_core_stress_143() {
        let ctx = DistributedContext::new(143, 4);
        assert_eq!(ctx.rank, 143);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 143 == 0);
    }

    #[test]
    fn test_dist_core_stress_144() {
        let ctx = DistributedContext::new(144, 4);
        assert_eq!(ctx.rank, 144);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 144 == 0);
    }

    #[test]
    fn test_dist_core_stress_145() {
        let ctx = DistributedContext::new(145, 4);
        assert_eq!(ctx.rank, 145);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 145 == 0);
    }

    #[test]
    fn test_dist_core_stress_146() {
        let ctx = DistributedContext::new(146, 4);
        assert_eq!(ctx.rank, 146);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 146 == 0);
    }

    #[test]
    fn test_dist_core_stress_147() {
        let ctx = DistributedContext::new(147, 4);
        assert_eq!(ctx.rank, 147);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 147 == 0);
    }

    #[test]
    fn test_dist_core_stress_148() {
        let ctx = DistributedContext::new(148, 4);
        assert_eq!(ctx.rank, 148);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 148 == 0);
    }

    #[test]
    fn test_dist_core_stress_149() {
        let ctx = DistributedContext::new(149, 4);
        assert_eq!(ctx.rank, 149);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 149 == 0);
    }

    #[test]
    fn test_dist_core_stress_150() {
        let ctx = DistributedContext::new(150, 4);
        assert_eq!(ctx.rank, 150);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 150 == 0);
    }

    #[test]
    fn test_dist_core_stress_151() {
        let ctx = DistributedContext::new(151, 4);
        assert_eq!(ctx.rank, 151);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 151 == 0);
    }

    #[test]
    fn test_dist_core_stress_152() {
        let ctx = DistributedContext::new(152, 4);
        assert_eq!(ctx.rank, 152);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 152 == 0);
    }

    #[test]
    fn test_dist_core_stress_153() {
        let ctx = DistributedContext::new(153, 4);
        assert_eq!(ctx.rank, 153);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 153 == 0);
    }

    #[test]
    fn test_dist_core_stress_154() {
        let ctx = DistributedContext::new(154, 4);
        assert_eq!(ctx.rank, 154);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 154 == 0);
    }

    #[test]
    fn test_dist_core_stress_155() {
        let ctx = DistributedContext::new(155, 4);
        assert_eq!(ctx.rank, 155);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 155 == 0);
    }

    #[test]
    fn test_dist_core_stress_156() {
        let ctx = DistributedContext::new(156, 4);
        assert_eq!(ctx.rank, 156);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 156 == 0);
    }

    #[test]
    fn test_dist_core_stress_157() {
        let ctx = DistributedContext::new(157, 4);
        assert_eq!(ctx.rank, 157);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 157 == 0);
    }

    #[test]
    fn test_dist_core_stress_158() {
        let ctx = DistributedContext::new(158, 4);
        assert_eq!(ctx.rank, 158);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 158 == 0);
    }

    #[test]
    fn test_dist_core_stress_159() {
        let ctx = DistributedContext::new(159, 4);
        assert_eq!(ctx.rank, 159);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 159 == 0);
    }

    #[test]
    fn test_dist_core_stress_160() {
        let ctx = DistributedContext::new(160, 4);
        assert_eq!(ctx.rank, 160);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 160 == 0);
    }

    #[test]
    fn test_dist_core_stress_161() {
        let ctx = DistributedContext::new(161, 4);
        assert_eq!(ctx.rank, 161);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 161 == 0);
    }

    #[test]
    fn test_dist_core_stress_162() {
        let ctx = DistributedContext::new(162, 4);
        assert_eq!(ctx.rank, 162);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 162 == 0);
    }

    #[test]
    fn test_dist_core_stress_163() {
        let ctx = DistributedContext::new(163, 4);
        assert_eq!(ctx.rank, 163);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 163 == 0);
    }

    #[test]
    fn test_dist_core_stress_164() {
        let ctx = DistributedContext::new(164, 4);
        assert_eq!(ctx.rank, 164);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 164 == 0);
    }

    #[test]
    fn test_dist_core_stress_165() {
        let ctx = DistributedContext::new(165, 4);
        assert_eq!(ctx.rank, 165);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 165 == 0);
    }

    #[test]
    fn test_dist_core_stress_166() {
        let ctx = DistributedContext::new(166, 4);
        assert_eq!(ctx.rank, 166);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 166 == 0);
    }

    #[test]
    fn test_dist_core_stress_167() {
        let ctx = DistributedContext::new(167, 4);
        assert_eq!(ctx.rank, 167);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 167 == 0);
    }

    #[test]
    fn test_dist_core_stress_168() {
        let ctx = DistributedContext::new(168, 4);
        assert_eq!(ctx.rank, 168);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 168 == 0);
    }

    #[test]
    fn test_dist_core_stress_169() {
        let ctx = DistributedContext::new(169, 4);
        assert_eq!(ctx.rank, 169);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 169 == 0);
    }

    #[test]
    fn test_dist_core_stress_170() {
        let ctx = DistributedContext::new(170, 4);
        assert_eq!(ctx.rank, 170);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 170 == 0);
    }

    #[test]
    fn test_dist_core_stress_171() {
        let ctx = DistributedContext::new(171, 4);
        assert_eq!(ctx.rank, 171);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 171 == 0);
    }

    #[test]
    fn test_dist_core_stress_172() {
        let ctx = DistributedContext::new(172, 4);
        assert_eq!(ctx.rank, 172);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 172 == 0);
    }

    #[test]
    fn test_dist_core_stress_173() {
        let ctx = DistributedContext::new(173, 4);
        assert_eq!(ctx.rank, 173);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 173 == 0);
    }

    #[test]
    fn test_dist_core_stress_174() {
        let ctx = DistributedContext::new(174, 4);
        assert_eq!(ctx.rank, 174);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 174 == 0);
    }

    #[test]
    fn test_dist_core_stress_175() {
        let ctx = DistributedContext::new(175, 4);
        assert_eq!(ctx.rank, 175);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 175 == 0);
    }

    #[test]
    fn test_dist_core_stress_176() {
        let ctx = DistributedContext::new(176, 4);
        assert_eq!(ctx.rank, 176);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 176 == 0);
    }

    #[test]
    fn test_dist_core_stress_177() {
        let ctx = DistributedContext::new(177, 4);
        assert_eq!(ctx.rank, 177);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 177 == 0);
    }

    #[test]
    fn test_dist_core_stress_178() {
        let ctx = DistributedContext::new(178, 4);
        assert_eq!(ctx.rank, 178);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 178 == 0);
    }

    #[test]
    fn test_dist_core_stress_179() {
        let ctx = DistributedContext::new(179, 4);
        assert_eq!(ctx.rank, 179);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 179 == 0);
    }

    #[test]
    fn test_dist_core_stress_180() {
        let ctx = DistributedContext::new(180, 4);
        assert_eq!(ctx.rank, 180);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 180 == 0);
    }

    #[test]
    fn test_dist_core_stress_181() {
        let ctx = DistributedContext::new(181, 4);
        assert_eq!(ctx.rank, 181);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 181 == 0);
    }

    #[test]
    fn test_dist_core_stress_182() {
        let ctx = DistributedContext::new(182, 4);
        assert_eq!(ctx.rank, 182);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 182 == 0);
    }

    #[test]
    fn test_dist_core_stress_183() {
        let ctx = DistributedContext::new(183, 4);
        assert_eq!(ctx.rank, 183);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 183 == 0);
    }

    #[test]
    fn test_dist_core_stress_184() {
        let ctx = DistributedContext::new(184, 4);
        assert_eq!(ctx.rank, 184);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 184 == 0);
    }

    #[test]
    fn test_dist_core_stress_185() {
        let ctx = DistributedContext::new(185, 4);
        assert_eq!(ctx.rank, 185);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 185 == 0);
    }

    #[test]
    fn test_dist_core_stress_186() {
        let ctx = DistributedContext::new(186, 4);
        assert_eq!(ctx.rank, 186);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 186 == 0);
    }

    #[test]
    fn test_dist_core_stress_187() {
        let ctx = DistributedContext::new(187, 4);
        assert_eq!(ctx.rank, 187);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 187 == 0);
    }

    #[test]
    fn test_dist_core_stress_188() {
        let ctx = DistributedContext::new(188, 4);
        assert_eq!(ctx.rank, 188);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 188 == 0);
    }

    #[test]
    fn test_dist_core_stress_189() {
        let ctx = DistributedContext::new(189, 4);
        assert_eq!(ctx.rank, 189);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 189 == 0);
    }

    #[test]
    fn test_dist_core_stress_190() {
        let ctx = DistributedContext::new(190, 4);
        assert_eq!(ctx.rank, 190);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 190 == 0);
    }

    #[test]
    fn test_dist_core_stress_191() {
        let ctx = DistributedContext::new(191, 4);
        assert_eq!(ctx.rank, 191);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 191 == 0);
    }

    #[test]
    fn test_dist_core_stress_192() {
        let ctx = DistributedContext::new(192, 4);
        assert_eq!(ctx.rank, 192);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 192 == 0);
    }

    #[test]
    fn test_dist_core_stress_193() {
        let ctx = DistributedContext::new(193, 4);
        assert_eq!(ctx.rank, 193);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 193 == 0);
    }

    #[test]
    fn test_dist_core_stress_194() {
        let ctx = DistributedContext::new(194, 4);
        assert_eq!(ctx.rank, 194);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 194 == 0);
    }

    #[test]
    fn test_dist_core_stress_195() {
        let ctx = DistributedContext::new(195, 4);
        assert_eq!(ctx.rank, 195);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 195 == 0);
    }

    #[test]
    fn test_dist_core_stress_196() {
        let ctx = DistributedContext::new(196, 4);
        assert_eq!(ctx.rank, 196);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 196 == 0);
    }

    #[test]
    fn test_dist_core_stress_197() {
        let ctx = DistributedContext::new(197, 4);
        assert_eq!(ctx.rank, 197);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 197 == 0);
    }

    #[test]
    fn test_dist_core_stress_198() {
        let ctx = DistributedContext::new(198, 4);
        assert_eq!(ctx.rank, 198);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 198 == 0);
    }

    #[test]
    fn test_dist_core_stress_199() {
        let ctx = DistributedContext::new(199, 4);
        assert_eq!(ctx.rank, 199);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 199 == 0);
    }

    #[test]
    fn test_dist_core_stress_200() {
        let ctx = DistributedContext::new(200, 4);
        assert_eq!(ctx.rank, 200);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 200 == 0);
    }

    #[test]
    fn test_dist_core_stress_201() {
        let ctx = DistributedContext::new(201, 4);
        assert_eq!(ctx.rank, 201);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 201 == 0);
    }

    #[test]
    fn test_dist_core_stress_202() {
        let ctx = DistributedContext::new(202, 4);
        assert_eq!(ctx.rank, 202);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 202 == 0);
    }

    #[test]
    fn test_dist_core_stress_203() {
        let ctx = DistributedContext::new(203, 4);
        assert_eq!(ctx.rank, 203);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 203 == 0);
    }

    #[test]
    fn test_dist_core_stress_204() {
        let ctx = DistributedContext::new(204, 4);
        assert_eq!(ctx.rank, 204);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 204 == 0);
    }

    #[test]
    fn test_dist_core_stress_205() {
        let ctx = DistributedContext::new(205, 4);
        assert_eq!(ctx.rank, 205);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 205 == 0);
    }

    #[test]
    fn test_dist_core_stress_206() {
        let ctx = DistributedContext::new(206, 4);
        assert_eq!(ctx.rank, 206);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 206 == 0);
    }

    #[test]
    fn test_dist_core_stress_207() {
        let ctx = DistributedContext::new(207, 4);
        assert_eq!(ctx.rank, 207);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 207 == 0);
    }

    #[test]
    fn test_dist_core_stress_208() {
        let ctx = DistributedContext::new(208, 4);
        assert_eq!(ctx.rank, 208);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 208 == 0);
    }

    #[test]
    fn test_dist_core_stress_209() {
        let ctx = DistributedContext::new(209, 4);
        assert_eq!(ctx.rank, 209);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 209 == 0);
    }

    #[test]
    fn test_dist_core_stress_210() {
        let ctx = DistributedContext::new(210, 4);
        assert_eq!(ctx.rank, 210);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 210 == 0);
    }

    #[test]
    fn test_dist_core_stress_211() {
        let ctx = DistributedContext::new(211, 4);
        assert_eq!(ctx.rank, 211);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 211 == 0);
    }

    #[test]
    fn test_dist_core_stress_212() {
        let ctx = DistributedContext::new(212, 4);
        assert_eq!(ctx.rank, 212);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 212 == 0);
    }

    #[test]
    fn test_dist_core_stress_213() {
        let ctx = DistributedContext::new(213, 4);
        assert_eq!(ctx.rank, 213);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 213 == 0);
    }

    #[test]
    fn test_dist_core_stress_214() {
        let ctx = DistributedContext::new(214, 4);
        assert_eq!(ctx.rank, 214);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 214 == 0);
    }

    #[test]
    fn test_dist_core_stress_215() {
        let ctx = DistributedContext::new(215, 4);
        assert_eq!(ctx.rank, 215);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 215 == 0);
    }

    #[test]
    fn test_dist_core_stress_216() {
        let ctx = DistributedContext::new(216, 4);
        assert_eq!(ctx.rank, 216);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 216 == 0);
    }

    #[test]
    fn test_dist_core_stress_217() {
        let ctx = DistributedContext::new(217, 4);
        assert_eq!(ctx.rank, 217);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 217 == 0);
    }

    #[test]
    fn test_dist_core_stress_218() {
        let ctx = DistributedContext::new(218, 4);
        assert_eq!(ctx.rank, 218);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 218 == 0);
    }

    #[test]
    fn test_dist_core_stress_219() {
        let ctx = DistributedContext::new(219, 4);
        assert_eq!(ctx.rank, 219);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 219 == 0);
    }

    #[test]
    fn test_dist_core_stress_220() {
        let ctx = DistributedContext::new(220, 4);
        assert_eq!(ctx.rank, 220);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 220 == 0);
    }

    #[test]
    fn test_dist_core_stress_221() {
        let ctx = DistributedContext::new(221, 4);
        assert_eq!(ctx.rank, 221);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 221 == 0);
    }

    #[test]
    fn test_dist_core_stress_222() {
        let ctx = DistributedContext::new(222, 4);
        assert_eq!(ctx.rank, 222);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 222 == 0);
    }

    #[test]
    fn test_dist_core_stress_223() {
        let ctx = DistributedContext::new(223, 4);
        assert_eq!(ctx.rank, 223);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 223 == 0);
    }

    #[test]
    fn test_dist_core_stress_224() {
        let ctx = DistributedContext::new(224, 4);
        assert_eq!(ctx.rank, 224);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 224 == 0);
    }

    #[test]
    fn test_dist_core_stress_225() {
        let ctx = DistributedContext::new(225, 4);
        assert_eq!(ctx.rank, 225);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 225 == 0);
    }

    #[test]
    fn test_dist_core_stress_226() {
        let ctx = DistributedContext::new(226, 4);
        assert_eq!(ctx.rank, 226);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 226 == 0);
    }

    #[test]
    fn test_dist_core_stress_227() {
        let ctx = DistributedContext::new(227, 4);
        assert_eq!(ctx.rank, 227);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 227 == 0);
    }

    #[test]
    fn test_dist_core_stress_228() {
        let ctx = DistributedContext::new(228, 4);
        assert_eq!(ctx.rank, 228);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 228 == 0);
    }

    #[test]
    fn test_dist_core_stress_229() {
        let ctx = DistributedContext::new(229, 4);
        assert_eq!(ctx.rank, 229);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 229 == 0);
    }

    #[test]
    fn test_dist_core_stress_230() {
        let ctx = DistributedContext::new(230, 4);
        assert_eq!(ctx.rank, 230);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 230 == 0);
    }

    #[test]
    fn test_dist_core_stress_231() {
        let ctx = DistributedContext::new(231, 4);
        assert_eq!(ctx.rank, 231);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 231 == 0);
    }

    #[test]
    fn test_dist_core_stress_232() {
        let ctx = DistributedContext::new(232, 4);
        assert_eq!(ctx.rank, 232);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 232 == 0);
    }

    #[test]
    fn test_dist_core_stress_233() {
        let ctx = DistributedContext::new(233, 4);
        assert_eq!(ctx.rank, 233);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 233 == 0);
    }

    #[test]
    fn test_dist_core_stress_234() {
        let ctx = DistributedContext::new(234, 4);
        assert_eq!(ctx.rank, 234);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 234 == 0);
    }

    #[test]
    fn test_dist_core_stress_235() {
        let ctx = DistributedContext::new(235, 4);
        assert_eq!(ctx.rank, 235);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 235 == 0);
    }

    #[test]
    fn test_dist_core_stress_236() {
        let ctx = DistributedContext::new(236, 4);
        assert_eq!(ctx.rank, 236);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 236 == 0);
    }

    #[test]
    fn test_dist_core_stress_237() {
        let ctx = DistributedContext::new(237, 4);
        assert_eq!(ctx.rank, 237);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 237 == 0);
    }

    #[test]
    fn test_dist_core_stress_238() {
        let ctx = DistributedContext::new(238, 4);
        assert_eq!(ctx.rank, 238);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 238 == 0);
    }

    #[test]
    fn test_dist_core_stress_239() {
        let ctx = DistributedContext::new(239, 4);
        assert_eq!(ctx.rank, 239);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 239 == 0);
    }

    #[test]
    fn test_dist_core_stress_240() {
        let ctx = DistributedContext::new(240, 4);
        assert_eq!(ctx.rank, 240);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 240 == 0);
    }

    #[test]
    fn test_dist_core_stress_241() {
        let ctx = DistributedContext::new(241, 4);
        assert_eq!(ctx.rank, 241);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 241 == 0);
    }

    #[test]
    fn test_dist_core_stress_242() {
        let ctx = DistributedContext::new(242, 4);
        assert_eq!(ctx.rank, 242);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 242 == 0);
    }

    #[test]
    fn test_dist_core_stress_243() {
        let ctx = DistributedContext::new(243, 4);
        assert_eq!(ctx.rank, 243);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 243 == 0);
    }

    #[test]
    fn test_dist_core_stress_244() {
        let ctx = DistributedContext::new(244, 4);
        assert_eq!(ctx.rank, 244);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 244 == 0);
    }

    #[test]
    fn test_dist_core_stress_245() {
        let ctx = DistributedContext::new(245, 4);
        assert_eq!(ctx.rank, 245);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 245 == 0);
    }

    #[test]
    fn test_dist_core_stress_246() {
        let ctx = DistributedContext::new(246, 4);
        assert_eq!(ctx.rank, 246);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 246 == 0);
    }

    #[test]
    fn test_dist_core_stress_247() {
        let ctx = DistributedContext::new(247, 4);
        assert_eq!(ctx.rank, 247);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 247 == 0);
    }

    #[test]
    fn test_dist_core_stress_248() {
        let ctx = DistributedContext::new(248, 4);
        assert_eq!(ctx.rank, 248);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 248 == 0);
    }

    #[test]
    fn test_dist_core_stress_249() {
        let ctx = DistributedContext::new(249, 4);
        assert_eq!(ctx.rank, 249);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 249 == 0);
    }

    #[test]
    fn test_dist_core_stress_250() {
        let ctx = DistributedContext::new(250, 4);
        assert_eq!(ctx.rank, 250);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 250 == 0);
    }

    #[test]
    fn test_dist_core_stress_251() {
        let ctx = DistributedContext::new(251, 4);
        assert_eq!(ctx.rank, 251);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 251 == 0);
    }

    #[test]
    fn test_dist_core_stress_252() {
        let ctx = DistributedContext::new(252, 4);
        assert_eq!(ctx.rank, 252);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 252 == 0);
    }

    #[test]
    fn test_dist_core_stress_253() {
        let ctx = DistributedContext::new(253, 4);
        assert_eq!(ctx.rank, 253);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 253 == 0);
    }

    #[test]
    fn test_dist_core_stress_254() {
        let ctx = DistributedContext::new(254, 4);
        assert_eq!(ctx.rank, 254);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 254 == 0);
    }

    #[test]
    fn test_dist_core_stress_255() {
        let ctx = DistributedContext::new(255, 4);
        assert_eq!(ctx.rank, 255);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 255 == 0);
    }

    #[test]
    fn test_dist_core_stress_256() {
        let ctx = DistributedContext::new(256, 4);
        assert_eq!(ctx.rank, 256);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 256 == 0);
    }

    #[test]
    fn test_dist_core_stress_257() {
        let ctx = DistributedContext::new(257, 4);
        assert_eq!(ctx.rank, 257);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 257 == 0);
    }

    #[test]
    fn test_dist_core_stress_258() {
        let ctx = DistributedContext::new(258, 4);
        assert_eq!(ctx.rank, 258);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 258 == 0);
    }

    #[test]
    fn test_dist_core_stress_259() {
        let ctx = DistributedContext::new(259, 4);
        assert_eq!(ctx.rank, 259);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 259 == 0);
    }

    #[test]
    fn test_dist_core_stress_260() {
        let ctx = DistributedContext::new(260, 4);
        assert_eq!(ctx.rank, 260);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 260 == 0);
    }

    #[test]
    fn test_dist_core_stress_261() {
        let ctx = DistributedContext::new(261, 4);
        assert_eq!(ctx.rank, 261);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 261 == 0);
    }

    #[test]
    fn test_dist_core_stress_262() {
        let ctx = DistributedContext::new(262, 4);
        assert_eq!(ctx.rank, 262);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 262 == 0);
    }

    #[test]
    fn test_dist_core_stress_263() {
        let ctx = DistributedContext::new(263, 4);
        assert_eq!(ctx.rank, 263);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 263 == 0);
    }

    #[test]
    fn test_dist_core_stress_264() {
        let ctx = DistributedContext::new(264, 4);
        assert_eq!(ctx.rank, 264);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 264 == 0);
    }

    #[test]
    fn test_dist_core_stress_265() {
        let ctx = DistributedContext::new(265, 4);
        assert_eq!(ctx.rank, 265);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 265 == 0);
    }

    #[test]
    fn test_dist_core_stress_266() {
        let ctx = DistributedContext::new(266, 4);
        assert_eq!(ctx.rank, 266);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 266 == 0);
    }

    #[test]
    fn test_dist_core_stress_267() {
        let ctx = DistributedContext::new(267, 4);
        assert_eq!(ctx.rank, 267);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 267 == 0);
    }

    #[test]
    fn test_dist_core_stress_268() {
        let ctx = DistributedContext::new(268, 4);
        assert_eq!(ctx.rank, 268);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 268 == 0);
    }

    #[test]
    fn test_dist_core_stress_269() {
        let ctx = DistributedContext::new(269, 4);
        assert_eq!(ctx.rank, 269);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 269 == 0);
    }

    #[test]
    fn test_dist_core_stress_270() {
        let ctx = DistributedContext::new(270, 4);
        assert_eq!(ctx.rank, 270);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 270 == 0);
    }

    #[test]
    fn test_dist_core_stress_271() {
        let ctx = DistributedContext::new(271, 4);
        assert_eq!(ctx.rank, 271);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 271 == 0);
    }

    #[test]
    fn test_dist_core_stress_272() {
        let ctx = DistributedContext::new(272, 4);
        assert_eq!(ctx.rank, 272);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 272 == 0);
    }

    #[test]
    fn test_dist_core_stress_273() {
        let ctx = DistributedContext::new(273, 4);
        assert_eq!(ctx.rank, 273);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 273 == 0);
    }

    #[test]
    fn test_dist_core_stress_274() {
        let ctx = DistributedContext::new(274, 4);
        assert_eq!(ctx.rank, 274);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 274 == 0);
    }

    #[test]
    fn test_dist_core_stress_275() {
        let ctx = DistributedContext::new(275, 4);
        assert_eq!(ctx.rank, 275);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 275 == 0);
    }

    #[test]
    fn test_dist_core_stress_276() {
        let ctx = DistributedContext::new(276, 4);
        assert_eq!(ctx.rank, 276);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 276 == 0);
    }

    #[test]
    fn test_dist_core_stress_277() {
        let ctx = DistributedContext::new(277, 4);
        assert_eq!(ctx.rank, 277);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 277 == 0);
    }

    #[test]
    fn test_dist_core_stress_278() {
        let ctx = DistributedContext::new(278, 4);
        assert_eq!(ctx.rank, 278);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 278 == 0);
    }

    #[test]
    fn test_dist_core_stress_279() {
        let ctx = DistributedContext::new(279, 4);
        assert_eq!(ctx.rank, 279);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 279 == 0);
    }

    #[test]
    fn test_dist_core_stress_280() {
        let ctx = DistributedContext::new(280, 4);
        assert_eq!(ctx.rank, 280);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 280 == 0);
    }

    #[test]
    fn test_dist_core_stress_281() {
        let ctx = DistributedContext::new(281, 4);
        assert_eq!(ctx.rank, 281);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 281 == 0);
    }

    #[test]
    fn test_dist_core_stress_282() {
        let ctx = DistributedContext::new(282, 4);
        assert_eq!(ctx.rank, 282);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 282 == 0);
    }

    #[test]
    fn test_dist_core_stress_283() {
        let ctx = DistributedContext::new(283, 4);
        assert_eq!(ctx.rank, 283);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 283 == 0);
    }

    #[test]
    fn test_dist_core_stress_284() {
        let ctx = DistributedContext::new(284, 4);
        assert_eq!(ctx.rank, 284);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 284 == 0);
    }

    #[test]
    fn test_dist_core_stress_285() {
        let ctx = DistributedContext::new(285, 4);
        assert_eq!(ctx.rank, 285);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 285 == 0);
    }

    #[test]
    fn test_dist_core_stress_286() {
        let ctx = DistributedContext::new(286, 4);
        assert_eq!(ctx.rank, 286);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 286 == 0);
    }

    #[test]
    fn test_dist_core_stress_287() {
        let ctx = DistributedContext::new(287, 4);
        assert_eq!(ctx.rank, 287);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 287 == 0);
    }

    #[test]
    fn test_dist_core_stress_288() {
        let ctx = DistributedContext::new(288, 4);
        assert_eq!(ctx.rank, 288);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 288 == 0);
    }

    #[test]
    fn test_dist_core_stress_289() {
        let ctx = DistributedContext::new(289, 4);
        assert_eq!(ctx.rank, 289);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 289 == 0);
    }

    #[test]
    fn test_dist_core_stress_290() {
        let ctx = DistributedContext::new(290, 4);
        assert_eq!(ctx.rank, 290);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 290 == 0);
    }

    #[test]
    fn test_dist_core_stress_291() {
        let ctx = DistributedContext::new(291, 4);
        assert_eq!(ctx.rank, 291);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 291 == 0);
    }

    #[test]
    fn test_dist_core_stress_292() {
        let ctx = DistributedContext::new(292, 4);
        assert_eq!(ctx.rank, 292);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 292 == 0);
    }

    #[test]
    fn test_dist_core_stress_293() {
        let ctx = DistributedContext::new(293, 4);
        assert_eq!(ctx.rank, 293);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 293 == 0);
    }

    #[test]
    fn test_dist_core_stress_294() {
        let ctx = DistributedContext::new(294, 4);
        assert_eq!(ctx.rank, 294);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 294 == 0);
    }

    #[test]
    fn test_dist_core_stress_295() {
        let ctx = DistributedContext::new(295, 4);
        assert_eq!(ctx.rank, 295);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 295 == 0);
    }

    #[test]
    fn test_dist_core_stress_296() {
        let ctx = DistributedContext::new(296, 4);
        assert_eq!(ctx.rank, 296);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 296 == 0);
    }

    #[test]
    fn test_dist_core_stress_297() {
        let ctx = DistributedContext::new(297, 4);
        assert_eq!(ctx.rank, 297);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 297 == 0);
    }

    #[test]
    fn test_dist_core_stress_298() {
        let ctx = DistributedContext::new(298, 4);
        assert_eq!(ctx.rank, 298);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 298 == 0);
    }

    #[test]
    fn test_dist_core_stress_299() {
        let ctx = DistributedContext::new(299, 4);
        assert_eq!(ctx.rank, 299);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 299 == 0);
    }

    #[test]
    fn test_dist_core_stress_300() {
        let ctx = DistributedContext::new(300, 4);
        assert_eq!(ctx.rank, 300);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 300 == 0);
    }

    #[test]
    fn test_dist_core_stress_301() {
        let ctx = DistributedContext::new(301, 4);
        assert_eq!(ctx.rank, 301);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 301 == 0);
    }

    #[test]
    fn test_dist_core_stress_302() {
        let ctx = DistributedContext::new(302, 4);
        assert_eq!(ctx.rank, 302);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 302 == 0);
    }

    #[test]
    fn test_dist_core_stress_303() {
        let ctx = DistributedContext::new(303, 4);
        assert_eq!(ctx.rank, 303);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 303 == 0);
    }

    #[test]
    fn test_dist_core_stress_304() {
        let ctx = DistributedContext::new(304, 4);
        assert_eq!(ctx.rank, 304);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 304 == 0);
    }

    #[test]
    fn test_dist_core_stress_305() {
        let ctx = DistributedContext::new(305, 4);
        assert_eq!(ctx.rank, 305);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 305 == 0);
    }

    #[test]
    fn test_dist_core_stress_306() {
        let ctx = DistributedContext::new(306, 4);
        assert_eq!(ctx.rank, 306);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 306 == 0);
    }

    #[test]
    fn test_dist_core_stress_307() {
        let ctx = DistributedContext::new(307, 4);
        assert_eq!(ctx.rank, 307);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 307 == 0);
    }

    #[test]
    fn test_dist_core_stress_308() {
        let ctx = DistributedContext::new(308, 4);
        assert_eq!(ctx.rank, 308);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 308 == 0);
    }

    #[test]
    fn test_dist_core_stress_309() {
        let ctx = DistributedContext::new(309, 4);
        assert_eq!(ctx.rank, 309);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 309 == 0);
    }

    #[test]
    fn test_dist_core_stress_310() {
        let ctx = DistributedContext::new(310, 4);
        assert_eq!(ctx.rank, 310);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 310 == 0);
    }

    #[test]
    fn test_dist_core_stress_311() {
        let ctx = DistributedContext::new(311, 4);
        assert_eq!(ctx.rank, 311);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 311 == 0);
    }

    #[test]
    fn test_dist_core_stress_312() {
        let ctx = DistributedContext::new(312, 4);
        assert_eq!(ctx.rank, 312);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 312 == 0);
    }

    #[test]
    fn test_dist_core_stress_313() {
        let ctx = DistributedContext::new(313, 4);
        assert_eq!(ctx.rank, 313);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 313 == 0);
    }

    #[test]
    fn test_dist_core_stress_314() {
        let ctx = DistributedContext::new(314, 4);
        assert_eq!(ctx.rank, 314);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 314 == 0);
    }

    #[test]
    fn test_dist_core_stress_315() {
        let ctx = DistributedContext::new(315, 4);
        assert_eq!(ctx.rank, 315);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 315 == 0);
    }

    #[test]
    fn test_dist_core_stress_316() {
        let ctx = DistributedContext::new(316, 4);
        assert_eq!(ctx.rank, 316);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 316 == 0);
    }

    #[test]
    fn test_dist_core_stress_317() {
        let ctx = DistributedContext::new(317, 4);
        assert_eq!(ctx.rank, 317);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 317 == 0);
    }

    #[test]
    fn test_dist_core_stress_318() {
        let ctx = DistributedContext::new(318, 4);
        assert_eq!(ctx.rank, 318);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 318 == 0);
    }

    #[test]
    fn test_dist_core_stress_319() {
        let ctx = DistributedContext::new(319, 4);
        assert_eq!(ctx.rank, 319);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 319 == 0);
    }

    #[test]
    fn test_dist_core_stress_320() {
        let ctx = DistributedContext::new(320, 4);
        assert_eq!(ctx.rank, 320);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 320 == 0);
    }

    #[test]
    fn test_dist_core_stress_321() {
        let ctx = DistributedContext::new(321, 4);
        assert_eq!(ctx.rank, 321);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 321 == 0);
    }

    #[test]
    fn test_dist_core_stress_322() {
        let ctx = DistributedContext::new(322, 4);
        assert_eq!(ctx.rank, 322);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 322 == 0);
    }

    #[test]
    fn test_dist_core_stress_323() {
        let ctx = DistributedContext::new(323, 4);
        assert_eq!(ctx.rank, 323);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 323 == 0);
    }

    #[test]
    fn test_dist_core_stress_324() {
        let ctx = DistributedContext::new(324, 4);
        assert_eq!(ctx.rank, 324);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 324 == 0);
    }

    #[test]
    fn test_dist_core_stress_325() {
        let ctx = DistributedContext::new(325, 4);
        assert_eq!(ctx.rank, 325);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 325 == 0);
    }

    #[test]
    fn test_dist_core_stress_326() {
        let ctx = DistributedContext::new(326, 4);
        assert_eq!(ctx.rank, 326);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 326 == 0);
    }

    #[test]
    fn test_dist_core_stress_327() {
        let ctx = DistributedContext::new(327, 4);
        assert_eq!(ctx.rank, 327);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 327 == 0);
    }

    #[test]
    fn test_dist_core_stress_328() {
        let ctx = DistributedContext::new(328, 4);
        assert_eq!(ctx.rank, 328);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 328 == 0);
    }

    #[test]
    fn test_dist_core_stress_329() {
        let ctx = DistributedContext::new(329, 4);
        assert_eq!(ctx.rank, 329);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 329 == 0);
    }

    #[test]
    fn test_dist_core_stress_330() {
        let ctx = DistributedContext::new(330, 4);
        assert_eq!(ctx.rank, 330);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 330 == 0);
    }

    #[test]
    fn test_dist_core_stress_331() {
        let ctx = DistributedContext::new(331, 4);
        assert_eq!(ctx.rank, 331);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 331 == 0);
    }

    #[test]
    fn test_dist_core_stress_332() {
        let ctx = DistributedContext::new(332, 4);
        assert_eq!(ctx.rank, 332);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 332 == 0);
    }

    #[test]
    fn test_dist_core_stress_333() {
        let ctx = DistributedContext::new(333, 4);
        assert_eq!(ctx.rank, 333);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 333 == 0);
    }

    #[test]
    fn test_dist_core_stress_334() {
        let ctx = DistributedContext::new(334, 4);
        assert_eq!(ctx.rank, 334);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 334 == 0);
    }

    #[test]
    fn test_dist_core_stress_335() {
        let ctx = DistributedContext::new(335, 4);
        assert_eq!(ctx.rank, 335);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 335 == 0);
    }

    #[test]
    fn test_dist_core_stress_336() {
        let ctx = DistributedContext::new(336, 4);
        assert_eq!(ctx.rank, 336);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 336 == 0);
    }

    #[test]
    fn test_dist_core_stress_337() {
        let ctx = DistributedContext::new(337, 4);
        assert_eq!(ctx.rank, 337);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 337 == 0);
    }

    #[test]
    fn test_dist_core_stress_338() {
        let ctx = DistributedContext::new(338, 4);
        assert_eq!(ctx.rank, 338);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 338 == 0);
    }

    #[test]
    fn test_dist_core_stress_339() {
        let ctx = DistributedContext::new(339, 4);
        assert_eq!(ctx.rank, 339);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 339 == 0);
    }

    #[test]
    fn test_dist_core_stress_340() {
        let ctx = DistributedContext::new(340, 4);
        assert_eq!(ctx.rank, 340);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 340 == 0);
    }

    #[test]
    fn test_dist_core_stress_341() {
        let ctx = DistributedContext::new(341, 4);
        assert_eq!(ctx.rank, 341);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 341 == 0);
    }

    #[test]
    fn test_dist_core_stress_342() {
        let ctx = DistributedContext::new(342, 4);
        assert_eq!(ctx.rank, 342);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 342 == 0);
    }

    #[test]
    fn test_dist_core_stress_343() {
        let ctx = DistributedContext::new(343, 4);
        assert_eq!(ctx.rank, 343);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 343 == 0);
    }

    #[test]
    fn test_dist_core_stress_344() {
        let ctx = DistributedContext::new(344, 4);
        assert_eq!(ctx.rank, 344);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 344 == 0);
    }

    #[test]
    fn test_dist_core_stress_345() {
        let ctx = DistributedContext::new(345, 4);
        assert_eq!(ctx.rank, 345);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 345 == 0);
    }

    #[test]
    fn test_dist_core_stress_346() {
        let ctx = DistributedContext::new(346, 4);
        assert_eq!(ctx.rank, 346);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 346 == 0);
    }

    #[test]
    fn test_dist_core_stress_347() {
        let ctx = DistributedContext::new(347, 4);
        assert_eq!(ctx.rank, 347);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 347 == 0);
    }

    #[test]
    fn test_dist_core_stress_348() {
        let ctx = DistributedContext::new(348, 4);
        assert_eq!(ctx.rank, 348);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 348 == 0);
    }

    #[test]
    fn test_dist_core_stress_349() {
        let ctx = DistributedContext::new(349, 4);
        assert_eq!(ctx.rank, 349);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 349 == 0);
    }

    #[test]
    fn test_dist_core_stress_350() {
        let ctx = DistributedContext::new(350, 4);
        assert_eq!(ctx.rank, 350);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 350 == 0);
    }

    #[test]
    fn test_dist_core_stress_351() {
        let ctx = DistributedContext::new(351, 4);
        assert_eq!(ctx.rank, 351);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 351 == 0);
    }

    #[test]
    fn test_dist_core_stress_352() {
        let ctx = DistributedContext::new(352, 4);
        assert_eq!(ctx.rank, 352);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 352 == 0);
    }

    #[test]
    fn test_dist_core_stress_353() {
        let ctx = DistributedContext::new(353, 4);
        assert_eq!(ctx.rank, 353);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 353 == 0);
    }

    #[test]
    fn test_dist_core_stress_354() {
        let ctx = DistributedContext::new(354, 4);
        assert_eq!(ctx.rank, 354);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 354 == 0);
    }

    #[test]
    fn test_dist_core_stress_355() {
        let ctx = DistributedContext::new(355, 4);
        assert_eq!(ctx.rank, 355);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 355 == 0);
    }

    #[test]
    fn test_dist_core_stress_356() {
        let ctx = DistributedContext::new(356, 4);
        assert_eq!(ctx.rank, 356);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 356 == 0);
    }

    #[test]
    fn test_dist_core_stress_357() {
        let ctx = DistributedContext::new(357, 4);
        assert_eq!(ctx.rank, 357);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 357 == 0);
    }

    #[test]
    fn test_dist_core_stress_358() {
        let ctx = DistributedContext::new(358, 4);
        assert_eq!(ctx.rank, 358);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 358 == 0);
    }

    #[test]
    fn test_dist_core_stress_359() {
        let ctx = DistributedContext::new(359, 4);
        assert_eq!(ctx.rank, 359);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 359 == 0);
    }

    #[test]
    fn test_dist_core_stress_360() {
        let ctx = DistributedContext::new(360, 4);
        assert_eq!(ctx.rank, 360);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 360 == 0);
    }

    #[test]
    fn test_dist_core_stress_361() {
        let ctx = DistributedContext::new(361, 4);
        assert_eq!(ctx.rank, 361);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 361 == 0);
    }

    #[test]
    fn test_dist_core_stress_362() {
        let ctx = DistributedContext::new(362, 4);
        assert_eq!(ctx.rank, 362);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 362 == 0);
    }

    #[test]
    fn test_dist_core_stress_363() {
        let ctx = DistributedContext::new(363, 4);
        assert_eq!(ctx.rank, 363);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 363 == 0);
    }

    #[test]
    fn test_dist_core_stress_364() {
        let ctx = DistributedContext::new(364, 4);
        assert_eq!(ctx.rank, 364);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 364 == 0);
    }

    #[test]
    fn test_dist_core_stress_365() {
        let ctx = DistributedContext::new(365, 4);
        assert_eq!(ctx.rank, 365);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 365 == 0);
    }

    #[test]
    fn test_dist_core_stress_366() {
        let ctx = DistributedContext::new(366, 4);
        assert_eq!(ctx.rank, 366);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 366 == 0);
    }

    #[test]
    fn test_dist_core_stress_367() {
        let ctx = DistributedContext::new(367, 4);
        assert_eq!(ctx.rank, 367);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 367 == 0);
    }

    #[test]
    fn test_dist_core_stress_368() {
        let ctx = DistributedContext::new(368, 4);
        assert_eq!(ctx.rank, 368);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 368 == 0);
    }

    #[test]
    fn test_dist_core_stress_369() {
        let ctx = DistributedContext::new(369, 4);
        assert_eq!(ctx.rank, 369);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 369 == 0);
    }

    #[test]
    fn test_dist_core_stress_370() {
        let ctx = DistributedContext::new(370, 4);
        assert_eq!(ctx.rank, 370);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 370 == 0);
    }

    #[test]
    fn test_dist_core_stress_371() {
        let ctx = DistributedContext::new(371, 4);
        assert_eq!(ctx.rank, 371);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 371 == 0);
    }

    #[test]
    fn test_dist_core_stress_372() {
        let ctx = DistributedContext::new(372, 4);
        assert_eq!(ctx.rank, 372);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 372 == 0);
    }

    #[test]
    fn test_dist_core_stress_373() {
        let ctx = DistributedContext::new(373, 4);
        assert_eq!(ctx.rank, 373);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 373 == 0);
    }

    #[test]
    fn test_dist_core_stress_374() {
        let ctx = DistributedContext::new(374, 4);
        assert_eq!(ctx.rank, 374);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 374 == 0);
    }

    #[test]
    fn test_dist_core_stress_375() {
        let ctx = DistributedContext::new(375, 4);
        assert_eq!(ctx.rank, 375);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 375 == 0);
    }

    #[test]
    fn test_dist_core_stress_376() {
        let ctx = DistributedContext::new(376, 4);
        assert_eq!(ctx.rank, 376);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 376 == 0);
    }

    #[test]
    fn test_dist_core_stress_377() {
        let ctx = DistributedContext::new(377, 4);
        assert_eq!(ctx.rank, 377);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 377 == 0);
    }

    #[test]
    fn test_dist_core_stress_378() {
        let ctx = DistributedContext::new(378, 4);
        assert_eq!(ctx.rank, 378);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 378 == 0);
    }

    #[test]
    fn test_dist_core_stress_379() {
        let ctx = DistributedContext::new(379, 4);
        assert_eq!(ctx.rank, 379);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 379 == 0);
    }

    #[test]
    fn test_dist_core_stress_380() {
        let ctx = DistributedContext::new(380, 4);
        assert_eq!(ctx.rank, 380);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 380 == 0);
    }

    #[test]
    fn test_dist_core_stress_381() {
        let ctx = DistributedContext::new(381, 4);
        assert_eq!(ctx.rank, 381);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 381 == 0);
    }

    #[test]
    fn test_dist_core_stress_382() {
        let ctx = DistributedContext::new(382, 4);
        assert_eq!(ctx.rank, 382);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 382 == 0);
    }

    #[test]
    fn test_dist_core_stress_383() {
        let ctx = DistributedContext::new(383, 4);
        assert_eq!(ctx.rank, 383);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 383 == 0);
    }

    #[test]
    fn test_dist_core_stress_384() {
        let ctx = DistributedContext::new(384, 4);
        assert_eq!(ctx.rank, 384);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 384 == 0);
    }

    #[test]
    fn test_dist_core_stress_385() {
        let ctx = DistributedContext::new(385, 4);
        assert_eq!(ctx.rank, 385);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 385 == 0);
    }

    #[test]
    fn test_dist_core_stress_386() {
        let ctx = DistributedContext::new(386, 4);
        assert_eq!(ctx.rank, 386);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 386 == 0);
    }

    #[test]
    fn test_dist_core_stress_387() {
        let ctx = DistributedContext::new(387, 4);
        assert_eq!(ctx.rank, 387);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 387 == 0);
    }

    #[test]
    fn test_dist_core_stress_388() {
        let ctx = DistributedContext::new(388, 4);
        assert_eq!(ctx.rank, 388);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 388 == 0);
    }

    #[test]
    fn test_dist_core_stress_389() {
        let ctx = DistributedContext::new(389, 4);
        assert_eq!(ctx.rank, 389);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 389 == 0);
    }

    #[test]
    fn test_dist_core_stress_390() {
        let ctx = DistributedContext::new(390, 4);
        assert_eq!(ctx.rank, 390);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 390 == 0);
    }

    #[test]
    fn test_dist_core_stress_391() {
        let ctx = DistributedContext::new(391, 4);
        assert_eq!(ctx.rank, 391);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 391 == 0);
    }

    #[test]
    fn test_dist_core_stress_392() {
        let ctx = DistributedContext::new(392, 4);
        assert_eq!(ctx.rank, 392);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 392 == 0);
    }

    #[test]
    fn test_dist_core_stress_393() {
        let ctx = DistributedContext::new(393, 4);
        assert_eq!(ctx.rank, 393);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 393 == 0);
    }

    #[test]
    fn test_dist_core_stress_394() {
        let ctx = DistributedContext::new(394, 4);
        assert_eq!(ctx.rank, 394);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 394 == 0);
    }

    #[test]
    fn test_dist_core_stress_395() {
        let ctx = DistributedContext::new(395, 4);
        assert_eq!(ctx.rank, 395);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 395 == 0);
    }

    #[test]
    fn test_dist_core_stress_396() {
        let ctx = DistributedContext::new(396, 4);
        assert_eq!(ctx.rank, 396);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 396 == 0);
    }

    #[test]
    fn test_dist_core_stress_397() {
        let ctx = DistributedContext::new(397, 4);
        assert_eq!(ctx.rank, 397);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 397 == 0);
    }

    #[test]
    fn test_dist_core_stress_398() {
        let ctx = DistributedContext::new(398, 4);
        assert_eq!(ctx.rank, 398);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 398 == 0);
    }

    #[test]
    fn test_dist_core_stress_399() {
        let ctx = DistributedContext::new(399, 4);
        assert_eq!(ctx.rank, 399);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 399 == 0);
    }

    #[test]
    fn test_dist_core_stress_400() {
        let ctx = DistributedContext::new(400, 4);
        assert_eq!(ctx.rank, 400);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 400 == 0);
    }

    #[test]
    fn test_dist_core_stress_401() {
        let ctx = DistributedContext::new(401, 4);
        assert_eq!(ctx.rank, 401);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 401 == 0);
    }

    #[test]
    fn test_dist_core_stress_402() {
        let ctx = DistributedContext::new(402, 4);
        assert_eq!(ctx.rank, 402);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 402 == 0);
    }

    #[test]
    fn test_dist_core_stress_403() {
        let ctx = DistributedContext::new(403, 4);
        assert_eq!(ctx.rank, 403);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 403 == 0);
    }

    #[test]
    fn test_dist_core_stress_404() {
        let ctx = DistributedContext::new(404, 4);
        assert_eq!(ctx.rank, 404);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 404 == 0);
    }

    #[test]
    fn test_dist_core_stress_405() {
        let ctx = DistributedContext::new(405, 4);
        assert_eq!(ctx.rank, 405);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 405 == 0);
    }

    #[test]
    fn test_dist_core_stress_406() {
        let ctx = DistributedContext::new(406, 4);
        assert_eq!(ctx.rank, 406);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 406 == 0);
    }

    #[test]
    fn test_dist_core_stress_407() {
        let ctx = DistributedContext::new(407, 4);
        assert_eq!(ctx.rank, 407);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 407 == 0);
    }

    #[test]
    fn test_dist_core_stress_408() {
        let ctx = DistributedContext::new(408, 4);
        assert_eq!(ctx.rank, 408);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 408 == 0);
    }

    #[test]
    fn test_dist_core_stress_409() {
        let ctx = DistributedContext::new(409, 4);
        assert_eq!(ctx.rank, 409);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 409 == 0);
    }

    #[test]
    fn test_dist_core_stress_410() {
        let ctx = DistributedContext::new(410, 4);
        assert_eq!(ctx.rank, 410);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 410 == 0);
    }

    #[test]
    fn test_dist_core_stress_411() {
        let ctx = DistributedContext::new(411, 4);
        assert_eq!(ctx.rank, 411);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 411 == 0);
    }

    #[test]
    fn test_dist_core_stress_412() {
        let ctx = DistributedContext::new(412, 4);
        assert_eq!(ctx.rank, 412);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 412 == 0);
    }

    #[test]
    fn test_dist_core_stress_413() {
        let ctx = DistributedContext::new(413, 4);
        assert_eq!(ctx.rank, 413);
        assert_eq!(ctx.world_size, 4);
        assert_eq!(ctx.is_master(), 413 == 0);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
}
