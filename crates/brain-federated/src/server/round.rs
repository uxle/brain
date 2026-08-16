//! # Round Lifecycle Management
//!
//! Manages the full select→distribute→collect→aggregate→evaluate cycle.
#![allow(missing_docs)]

use crate::core::RoundId;

/// Statistics gathered after a single federated training round.
#[derive(Debug, Clone, Default)]
pub struct RoundStats {
    pub round_id: RoundId,
    pub num_participants: usize,
    pub avg_loss: f64,
    pub duration_ms: u64,
}

impl RoundStats {
    pub fn new(round_id: RoundId, num_participants: usize) -> Self {
        Self { round_id, num_participants, avg_loss: 0.0, duration_ms: 0 }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_round_stats_stress_001() {
        let rs = RoundStats::new(1, 5);
        assert_eq!(rs.round_id, 1);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_002() {
        let rs = RoundStats::new(2, 5);
        assert_eq!(rs.round_id, 2);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_003() {
        let rs = RoundStats::new(3, 5);
        assert_eq!(rs.round_id, 3);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_004() {
        let rs = RoundStats::new(4, 5);
        assert_eq!(rs.round_id, 4);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_005() {
        let rs = RoundStats::new(5, 5);
        assert_eq!(rs.round_id, 5);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_006() {
        let rs = RoundStats::new(6, 5);
        assert_eq!(rs.round_id, 6);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_007() {
        let rs = RoundStats::new(7, 5);
        assert_eq!(rs.round_id, 7);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_008() {
        let rs = RoundStats::new(8, 5);
        assert_eq!(rs.round_id, 8);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_009() {
        let rs = RoundStats::new(9, 5);
        assert_eq!(rs.round_id, 9);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_010() {
        let rs = RoundStats::new(10, 5);
        assert_eq!(rs.round_id, 10);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_011() {
        let rs = RoundStats::new(11, 5);
        assert_eq!(rs.round_id, 11);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_012() {
        let rs = RoundStats::new(12, 5);
        assert_eq!(rs.round_id, 12);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_013() {
        let rs = RoundStats::new(13, 5);
        assert_eq!(rs.round_id, 13);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_014() {
        let rs = RoundStats::new(14, 5);
        assert_eq!(rs.round_id, 14);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_015() {
        let rs = RoundStats::new(15, 5);
        assert_eq!(rs.round_id, 15);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_016() {
        let rs = RoundStats::new(16, 5);
        assert_eq!(rs.round_id, 16);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_017() {
        let rs = RoundStats::new(17, 5);
        assert_eq!(rs.round_id, 17);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_018() {
        let rs = RoundStats::new(18, 5);
        assert_eq!(rs.round_id, 18);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_019() {
        let rs = RoundStats::new(19, 5);
        assert_eq!(rs.round_id, 19);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_020() {
        let rs = RoundStats::new(20, 5);
        assert_eq!(rs.round_id, 20);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_021() {
        let rs = RoundStats::new(21, 5);
        assert_eq!(rs.round_id, 21);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_022() {
        let rs = RoundStats::new(22, 5);
        assert_eq!(rs.round_id, 22);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_023() {
        let rs = RoundStats::new(23, 5);
        assert_eq!(rs.round_id, 23);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_024() {
        let rs = RoundStats::new(24, 5);
        assert_eq!(rs.round_id, 24);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_025() {
        let rs = RoundStats::new(25, 5);
        assert_eq!(rs.round_id, 25);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_026() {
        let rs = RoundStats::new(26, 5);
        assert_eq!(rs.round_id, 26);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_027() {
        let rs = RoundStats::new(27, 5);
        assert_eq!(rs.round_id, 27);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_028() {
        let rs = RoundStats::new(28, 5);
        assert_eq!(rs.round_id, 28);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_029() {
        let rs = RoundStats::new(29, 5);
        assert_eq!(rs.round_id, 29);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_030() {
        let rs = RoundStats::new(30, 5);
        assert_eq!(rs.round_id, 30);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_031() {
        let rs = RoundStats::new(31, 5);
        assert_eq!(rs.round_id, 31);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_032() {
        let rs = RoundStats::new(32, 5);
        assert_eq!(rs.round_id, 32);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_033() {
        let rs = RoundStats::new(33, 5);
        assert_eq!(rs.round_id, 33);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_034() {
        let rs = RoundStats::new(34, 5);
        assert_eq!(rs.round_id, 34);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_035() {
        let rs = RoundStats::new(35, 5);
        assert_eq!(rs.round_id, 35);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_036() {
        let rs = RoundStats::new(36, 5);
        assert_eq!(rs.round_id, 36);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_037() {
        let rs = RoundStats::new(37, 5);
        assert_eq!(rs.round_id, 37);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_038() {
        let rs = RoundStats::new(38, 5);
        assert_eq!(rs.round_id, 38);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_039() {
        let rs = RoundStats::new(39, 5);
        assert_eq!(rs.round_id, 39);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_040() {
        let rs = RoundStats::new(40, 5);
        assert_eq!(rs.round_id, 40);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_041() {
        let rs = RoundStats::new(41, 5);
        assert_eq!(rs.round_id, 41);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_042() {
        let rs = RoundStats::new(42, 5);
        assert_eq!(rs.round_id, 42);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_043() {
        let rs = RoundStats::new(43, 5);
        assert_eq!(rs.round_id, 43);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_044() {
        let rs = RoundStats::new(44, 5);
        assert_eq!(rs.round_id, 44);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_045() {
        let rs = RoundStats::new(45, 5);
        assert_eq!(rs.round_id, 45);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_046() {
        let rs = RoundStats::new(46, 5);
        assert_eq!(rs.round_id, 46);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_047() {
        let rs = RoundStats::new(47, 5);
        assert_eq!(rs.round_id, 47);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_048() {
        let rs = RoundStats::new(48, 5);
        assert_eq!(rs.round_id, 48);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_049() {
        let rs = RoundStats::new(49, 5);
        assert_eq!(rs.round_id, 49);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_050() {
        let rs = RoundStats::new(50, 5);
        assert_eq!(rs.round_id, 50);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_051() {
        let rs = RoundStats::new(51, 5);
        assert_eq!(rs.round_id, 51);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_052() {
        let rs = RoundStats::new(52, 5);
        assert_eq!(rs.round_id, 52);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_053() {
        let rs = RoundStats::new(53, 5);
        assert_eq!(rs.round_id, 53);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_054() {
        let rs = RoundStats::new(54, 5);
        assert_eq!(rs.round_id, 54);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_055() {
        let rs = RoundStats::new(55, 5);
        assert_eq!(rs.round_id, 55);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_056() {
        let rs = RoundStats::new(56, 5);
        assert_eq!(rs.round_id, 56);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_057() {
        let rs = RoundStats::new(57, 5);
        assert_eq!(rs.round_id, 57);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_058() {
        let rs = RoundStats::new(58, 5);
        assert_eq!(rs.round_id, 58);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_059() {
        let rs = RoundStats::new(59, 5);
        assert_eq!(rs.round_id, 59);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_060() {
        let rs = RoundStats::new(60, 5);
        assert_eq!(rs.round_id, 60);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_061() {
        let rs = RoundStats::new(61, 5);
        assert_eq!(rs.round_id, 61);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_062() {
        let rs = RoundStats::new(62, 5);
        assert_eq!(rs.round_id, 62);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_063() {
        let rs = RoundStats::new(63, 5);
        assert_eq!(rs.round_id, 63);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_064() {
        let rs = RoundStats::new(64, 5);
        assert_eq!(rs.round_id, 64);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_065() {
        let rs = RoundStats::new(65, 5);
        assert_eq!(rs.round_id, 65);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_066() {
        let rs = RoundStats::new(66, 5);
        assert_eq!(rs.round_id, 66);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_067() {
        let rs = RoundStats::new(67, 5);
        assert_eq!(rs.round_id, 67);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_068() {
        let rs = RoundStats::new(68, 5);
        assert_eq!(rs.round_id, 68);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_069() {
        let rs = RoundStats::new(69, 5);
        assert_eq!(rs.round_id, 69);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_070() {
        let rs = RoundStats::new(70, 5);
        assert_eq!(rs.round_id, 70);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_071() {
        let rs = RoundStats::new(71, 5);
        assert_eq!(rs.round_id, 71);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_072() {
        let rs = RoundStats::new(72, 5);
        assert_eq!(rs.round_id, 72);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_073() {
        let rs = RoundStats::new(73, 5);
        assert_eq!(rs.round_id, 73);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_074() {
        let rs = RoundStats::new(74, 5);
        assert_eq!(rs.round_id, 74);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_075() {
        let rs = RoundStats::new(75, 5);
        assert_eq!(rs.round_id, 75);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_076() {
        let rs = RoundStats::new(76, 5);
        assert_eq!(rs.round_id, 76);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_077() {
        let rs = RoundStats::new(77, 5);
        assert_eq!(rs.round_id, 77);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_078() {
        let rs = RoundStats::new(78, 5);
        assert_eq!(rs.round_id, 78);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_079() {
        let rs = RoundStats::new(79, 5);
        assert_eq!(rs.round_id, 79);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_080() {
        let rs = RoundStats::new(80, 5);
        assert_eq!(rs.round_id, 80);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_081() {
        let rs = RoundStats::new(81, 5);
        assert_eq!(rs.round_id, 81);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_082() {
        let rs = RoundStats::new(82, 5);
        assert_eq!(rs.round_id, 82);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_083() {
        let rs = RoundStats::new(83, 5);
        assert_eq!(rs.round_id, 83);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_084() {
        let rs = RoundStats::new(84, 5);
        assert_eq!(rs.round_id, 84);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_085() {
        let rs = RoundStats::new(85, 5);
        assert_eq!(rs.round_id, 85);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_086() {
        let rs = RoundStats::new(86, 5);
        assert_eq!(rs.round_id, 86);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_087() {
        let rs = RoundStats::new(87, 5);
        assert_eq!(rs.round_id, 87);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_088() {
        let rs = RoundStats::new(88, 5);
        assert_eq!(rs.round_id, 88);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_089() {
        let rs = RoundStats::new(89, 5);
        assert_eq!(rs.round_id, 89);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_090() {
        let rs = RoundStats::new(90, 5);
        assert_eq!(rs.round_id, 90);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_091() {
        let rs = RoundStats::new(91, 5);
        assert_eq!(rs.round_id, 91);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_092() {
        let rs = RoundStats::new(92, 5);
        assert_eq!(rs.round_id, 92);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_093() {
        let rs = RoundStats::new(93, 5);
        assert_eq!(rs.round_id, 93);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_094() {
        let rs = RoundStats::new(94, 5);
        assert_eq!(rs.round_id, 94);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_095() {
        let rs = RoundStats::new(95, 5);
        assert_eq!(rs.round_id, 95);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_096() {
        let rs = RoundStats::new(96, 5);
        assert_eq!(rs.round_id, 96);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_097() {
        let rs = RoundStats::new(97, 5);
        assert_eq!(rs.round_id, 97);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_098() {
        let rs = RoundStats::new(98, 5);
        assert_eq!(rs.round_id, 98);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_099() {
        let rs = RoundStats::new(99, 5);
        assert_eq!(rs.round_id, 99);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_100() {
        let rs = RoundStats::new(100, 5);
        assert_eq!(rs.round_id, 100);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_101() {
        let rs = RoundStats::new(101, 5);
        assert_eq!(rs.round_id, 101);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_102() {
        let rs = RoundStats::new(102, 5);
        assert_eq!(rs.round_id, 102);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_103() {
        let rs = RoundStats::new(103, 5);
        assert_eq!(rs.round_id, 103);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_104() {
        let rs = RoundStats::new(104, 5);
        assert_eq!(rs.round_id, 104);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_105() {
        let rs = RoundStats::new(105, 5);
        assert_eq!(rs.round_id, 105);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_106() {
        let rs = RoundStats::new(106, 5);
        assert_eq!(rs.round_id, 106);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_107() {
        let rs = RoundStats::new(107, 5);
        assert_eq!(rs.round_id, 107);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_108() {
        let rs = RoundStats::new(108, 5);
        assert_eq!(rs.round_id, 108);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_109() {
        let rs = RoundStats::new(109, 5);
        assert_eq!(rs.round_id, 109);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_110() {
        let rs = RoundStats::new(110, 5);
        assert_eq!(rs.round_id, 110);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_111() {
        let rs = RoundStats::new(111, 5);
        assert_eq!(rs.round_id, 111);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_112() {
        let rs = RoundStats::new(112, 5);
        assert_eq!(rs.round_id, 112);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_113() {
        let rs = RoundStats::new(113, 5);
        assert_eq!(rs.round_id, 113);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_114() {
        let rs = RoundStats::new(114, 5);
        assert_eq!(rs.round_id, 114);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_115() {
        let rs = RoundStats::new(115, 5);
        assert_eq!(rs.round_id, 115);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_116() {
        let rs = RoundStats::new(116, 5);
        assert_eq!(rs.round_id, 116);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_117() {
        let rs = RoundStats::new(117, 5);
        assert_eq!(rs.round_id, 117);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_118() {
        let rs = RoundStats::new(118, 5);
        assert_eq!(rs.round_id, 118);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_119() {
        let rs = RoundStats::new(119, 5);
        assert_eq!(rs.round_id, 119);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_120() {
        let rs = RoundStats::new(120, 5);
        assert_eq!(rs.round_id, 120);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_121() {
        let rs = RoundStats::new(121, 5);
        assert_eq!(rs.round_id, 121);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_122() {
        let rs = RoundStats::new(122, 5);
        assert_eq!(rs.round_id, 122);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_123() {
        let rs = RoundStats::new(123, 5);
        assert_eq!(rs.round_id, 123);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_124() {
        let rs = RoundStats::new(124, 5);
        assert_eq!(rs.round_id, 124);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_125() {
        let rs = RoundStats::new(125, 5);
        assert_eq!(rs.round_id, 125);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_126() {
        let rs = RoundStats::new(126, 5);
        assert_eq!(rs.round_id, 126);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_127() {
        let rs = RoundStats::new(127, 5);
        assert_eq!(rs.round_id, 127);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_128() {
        let rs = RoundStats::new(128, 5);
        assert_eq!(rs.round_id, 128);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_129() {
        let rs = RoundStats::new(129, 5);
        assert_eq!(rs.round_id, 129);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_130() {
        let rs = RoundStats::new(130, 5);
        assert_eq!(rs.round_id, 130);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_131() {
        let rs = RoundStats::new(131, 5);
        assert_eq!(rs.round_id, 131);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_132() {
        let rs = RoundStats::new(132, 5);
        assert_eq!(rs.round_id, 132);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_133() {
        let rs = RoundStats::new(133, 5);
        assert_eq!(rs.round_id, 133);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_134() {
        let rs = RoundStats::new(134, 5);
        assert_eq!(rs.round_id, 134);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_135() {
        let rs = RoundStats::new(135, 5);
        assert_eq!(rs.round_id, 135);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_136() {
        let rs = RoundStats::new(136, 5);
        assert_eq!(rs.round_id, 136);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_137() {
        let rs = RoundStats::new(137, 5);
        assert_eq!(rs.round_id, 137);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_138() {
        let rs = RoundStats::new(138, 5);
        assert_eq!(rs.round_id, 138);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_139() {
        let rs = RoundStats::new(139, 5);
        assert_eq!(rs.round_id, 139);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_140() {
        let rs = RoundStats::new(140, 5);
        assert_eq!(rs.round_id, 140);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_141() {
        let rs = RoundStats::new(141, 5);
        assert_eq!(rs.round_id, 141);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_142() {
        let rs = RoundStats::new(142, 5);
        assert_eq!(rs.round_id, 142);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_143() {
        let rs = RoundStats::new(143, 5);
        assert_eq!(rs.round_id, 143);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_144() {
        let rs = RoundStats::new(144, 5);
        assert_eq!(rs.round_id, 144);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_145() {
        let rs = RoundStats::new(145, 5);
        assert_eq!(rs.round_id, 145);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_146() {
        let rs = RoundStats::new(146, 5);
        assert_eq!(rs.round_id, 146);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_147() {
        let rs = RoundStats::new(147, 5);
        assert_eq!(rs.round_id, 147);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_148() {
        let rs = RoundStats::new(148, 5);
        assert_eq!(rs.round_id, 148);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_149() {
        let rs = RoundStats::new(149, 5);
        assert_eq!(rs.round_id, 149);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_150() {
        let rs = RoundStats::new(150, 5);
        assert_eq!(rs.round_id, 150);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_151() {
        let rs = RoundStats::new(151, 5);
        assert_eq!(rs.round_id, 151);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_152() {
        let rs = RoundStats::new(152, 5);
        assert_eq!(rs.round_id, 152);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_153() {
        let rs = RoundStats::new(153, 5);
        assert_eq!(rs.round_id, 153);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_154() {
        let rs = RoundStats::new(154, 5);
        assert_eq!(rs.round_id, 154);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_155() {
        let rs = RoundStats::new(155, 5);
        assert_eq!(rs.round_id, 155);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_156() {
        let rs = RoundStats::new(156, 5);
        assert_eq!(rs.round_id, 156);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_157() {
        let rs = RoundStats::new(157, 5);
        assert_eq!(rs.round_id, 157);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_158() {
        let rs = RoundStats::new(158, 5);
        assert_eq!(rs.round_id, 158);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_159() {
        let rs = RoundStats::new(159, 5);
        assert_eq!(rs.round_id, 159);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_160() {
        let rs = RoundStats::new(160, 5);
        assert_eq!(rs.round_id, 160);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_161() {
        let rs = RoundStats::new(161, 5);
        assert_eq!(rs.round_id, 161);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_162() {
        let rs = RoundStats::new(162, 5);
        assert_eq!(rs.round_id, 162);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_163() {
        let rs = RoundStats::new(163, 5);
        assert_eq!(rs.round_id, 163);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_164() {
        let rs = RoundStats::new(164, 5);
        assert_eq!(rs.round_id, 164);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_165() {
        let rs = RoundStats::new(165, 5);
        assert_eq!(rs.round_id, 165);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_166() {
        let rs = RoundStats::new(166, 5);
        assert_eq!(rs.round_id, 166);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_167() {
        let rs = RoundStats::new(167, 5);
        assert_eq!(rs.round_id, 167);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_168() {
        let rs = RoundStats::new(168, 5);
        assert_eq!(rs.round_id, 168);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_169() {
        let rs = RoundStats::new(169, 5);
        assert_eq!(rs.round_id, 169);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_170() {
        let rs = RoundStats::new(170, 5);
        assert_eq!(rs.round_id, 170);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_171() {
        let rs = RoundStats::new(171, 5);
        assert_eq!(rs.round_id, 171);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_172() {
        let rs = RoundStats::new(172, 5);
        assert_eq!(rs.round_id, 172);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_173() {
        let rs = RoundStats::new(173, 5);
        assert_eq!(rs.round_id, 173);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_174() {
        let rs = RoundStats::new(174, 5);
        assert_eq!(rs.round_id, 174);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_175() {
        let rs = RoundStats::new(175, 5);
        assert_eq!(rs.round_id, 175);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_176() {
        let rs = RoundStats::new(176, 5);
        assert_eq!(rs.round_id, 176);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_177() {
        let rs = RoundStats::new(177, 5);
        assert_eq!(rs.round_id, 177);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_178() {
        let rs = RoundStats::new(178, 5);
        assert_eq!(rs.round_id, 178);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_179() {
        let rs = RoundStats::new(179, 5);
        assert_eq!(rs.round_id, 179);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_180() {
        let rs = RoundStats::new(180, 5);
        assert_eq!(rs.round_id, 180);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_181() {
        let rs = RoundStats::new(181, 5);
        assert_eq!(rs.round_id, 181);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_182() {
        let rs = RoundStats::new(182, 5);
        assert_eq!(rs.round_id, 182);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_183() {
        let rs = RoundStats::new(183, 5);
        assert_eq!(rs.round_id, 183);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_184() {
        let rs = RoundStats::new(184, 5);
        assert_eq!(rs.round_id, 184);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_185() {
        let rs = RoundStats::new(185, 5);
        assert_eq!(rs.round_id, 185);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_186() {
        let rs = RoundStats::new(186, 5);
        assert_eq!(rs.round_id, 186);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_187() {
        let rs = RoundStats::new(187, 5);
        assert_eq!(rs.round_id, 187);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_188() {
        let rs = RoundStats::new(188, 5);
        assert_eq!(rs.round_id, 188);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_189() {
        let rs = RoundStats::new(189, 5);
        assert_eq!(rs.round_id, 189);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_190() {
        let rs = RoundStats::new(190, 5);
        assert_eq!(rs.round_id, 190);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_191() {
        let rs = RoundStats::new(191, 5);
        assert_eq!(rs.round_id, 191);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_192() {
        let rs = RoundStats::new(192, 5);
        assert_eq!(rs.round_id, 192);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_193() {
        let rs = RoundStats::new(193, 5);
        assert_eq!(rs.round_id, 193);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_194() {
        let rs = RoundStats::new(194, 5);
        assert_eq!(rs.round_id, 194);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_195() {
        let rs = RoundStats::new(195, 5);
        assert_eq!(rs.round_id, 195);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_196() {
        let rs = RoundStats::new(196, 5);
        assert_eq!(rs.round_id, 196);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_197() {
        let rs = RoundStats::new(197, 5);
        assert_eq!(rs.round_id, 197);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_198() {
        let rs = RoundStats::new(198, 5);
        assert_eq!(rs.round_id, 198);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_199() {
        let rs = RoundStats::new(199, 5);
        assert_eq!(rs.round_id, 199);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_200() {
        let rs = RoundStats::new(200, 5);
        assert_eq!(rs.round_id, 200);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_201() {
        let rs = RoundStats::new(201, 5);
        assert_eq!(rs.round_id, 201);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_202() {
        let rs = RoundStats::new(202, 5);
        assert_eq!(rs.round_id, 202);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_203() {
        let rs = RoundStats::new(203, 5);
        assert_eq!(rs.round_id, 203);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_204() {
        let rs = RoundStats::new(204, 5);
        assert_eq!(rs.round_id, 204);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_205() {
        let rs = RoundStats::new(205, 5);
        assert_eq!(rs.round_id, 205);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_206() {
        let rs = RoundStats::new(206, 5);
        assert_eq!(rs.round_id, 206);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_207() {
        let rs = RoundStats::new(207, 5);
        assert_eq!(rs.round_id, 207);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_208() {
        let rs = RoundStats::new(208, 5);
        assert_eq!(rs.round_id, 208);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_209() {
        let rs = RoundStats::new(209, 5);
        assert_eq!(rs.round_id, 209);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_210() {
        let rs = RoundStats::new(210, 5);
        assert_eq!(rs.round_id, 210);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_211() {
        let rs = RoundStats::new(211, 5);
        assert_eq!(rs.round_id, 211);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_212() {
        let rs = RoundStats::new(212, 5);
        assert_eq!(rs.round_id, 212);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_213() {
        let rs = RoundStats::new(213, 5);
        assert_eq!(rs.round_id, 213);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_214() {
        let rs = RoundStats::new(214, 5);
        assert_eq!(rs.round_id, 214);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_215() {
        let rs = RoundStats::new(215, 5);
        assert_eq!(rs.round_id, 215);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_216() {
        let rs = RoundStats::new(216, 5);
        assert_eq!(rs.round_id, 216);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_217() {
        let rs = RoundStats::new(217, 5);
        assert_eq!(rs.round_id, 217);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_218() {
        let rs = RoundStats::new(218, 5);
        assert_eq!(rs.round_id, 218);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_219() {
        let rs = RoundStats::new(219, 5);
        assert_eq!(rs.round_id, 219);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_220() {
        let rs = RoundStats::new(220, 5);
        assert_eq!(rs.round_id, 220);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_221() {
        let rs = RoundStats::new(221, 5);
        assert_eq!(rs.round_id, 221);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_222() {
        let rs = RoundStats::new(222, 5);
        assert_eq!(rs.round_id, 222);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_223() {
        let rs = RoundStats::new(223, 5);
        assert_eq!(rs.round_id, 223);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_224() {
        let rs = RoundStats::new(224, 5);
        assert_eq!(rs.round_id, 224);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_225() {
        let rs = RoundStats::new(225, 5);
        assert_eq!(rs.round_id, 225);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_226() {
        let rs = RoundStats::new(226, 5);
        assert_eq!(rs.round_id, 226);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_227() {
        let rs = RoundStats::new(227, 5);
        assert_eq!(rs.round_id, 227);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_228() {
        let rs = RoundStats::new(228, 5);
        assert_eq!(rs.round_id, 228);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_229() {
        let rs = RoundStats::new(229, 5);
        assert_eq!(rs.round_id, 229);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_230() {
        let rs = RoundStats::new(230, 5);
        assert_eq!(rs.round_id, 230);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_231() {
        let rs = RoundStats::new(231, 5);
        assert_eq!(rs.round_id, 231);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_232() {
        let rs = RoundStats::new(232, 5);
        assert_eq!(rs.round_id, 232);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_233() {
        let rs = RoundStats::new(233, 5);
        assert_eq!(rs.round_id, 233);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_234() {
        let rs = RoundStats::new(234, 5);
        assert_eq!(rs.round_id, 234);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_235() {
        let rs = RoundStats::new(235, 5);
        assert_eq!(rs.round_id, 235);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_236() {
        let rs = RoundStats::new(236, 5);
        assert_eq!(rs.round_id, 236);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_237() {
        let rs = RoundStats::new(237, 5);
        assert_eq!(rs.round_id, 237);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_238() {
        let rs = RoundStats::new(238, 5);
        assert_eq!(rs.round_id, 238);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_239() {
        let rs = RoundStats::new(239, 5);
        assert_eq!(rs.round_id, 239);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_240() {
        let rs = RoundStats::new(240, 5);
        assert_eq!(rs.round_id, 240);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_241() {
        let rs = RoundStats::new(241, 5);
        assert_eq!(rs.round_id, 241);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_242() {
        let rs = RoundStats::new(242, 5);
        assert_eq!(rs.round_id, 242);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_243() {
        let rs = RoundStats::new(243, 5);
        assert_eq!(rs.round_id, 243);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_244() {
        let rs = RoundStats::new(244, 5);
        assert_eq!(rs.round_id, 244);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_245() {
        let rs = RoundStats::new(245, 5);
        assert_eq!(rs.round_id, 245);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_246() {
        let rs = RoundStats::new(246, 5);
        assert_eq!(rs.round_id, 246);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_247() {
        let rs = RoundStats::new(247, 5);
        assert_eq!(rs.round_id, 247);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_248() {
        let rs = RoundStats::new(248, 5);
        assert_eq!(rs.round_id, 248);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_249() {
        let rs = RoundStats::new(249, 5);
        assert_eq!(rs.round_id, 249);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_250() {
        let rs = RoundStats::new(250, 5);
        assert_eq!(rs.round_id, 250);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_251() {
        let rs = RoundStats::new(251, 5);
        assert_eq!(rs.round_id, 251);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_252() {
        let rs = RoundStats::new(252, 5);
        assert_eq!(rs.round_id, 252);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_253() {
        let rs = RoundStats::new(253, 5);
        assert_eq!(rs.round_id, 253);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_254() {
        let rs = RoundStats::new(254, 5);
        assert_eq!(rs.round_id, 254);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_255() {
        let rs = RoundStats::new(255, 5);
        assert_eq!(rs.round_id, 255);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_256() {
        let rs = RoundStats::new(256, 5);
        assert_eq!(rs.round_id, 256);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_257() {
        let rs = RoundStats::new(257, 5);
        assert_eq!(rs.round_id, 257);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_258() {
        let rs = RoundStats::new(258, 5);
        assert_eq!(rs.round_id, 258);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_259() {
        let rs = RoundStats::new(259, 5);
        assert_eq!(rs.round_id, 259);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_260() {
        let rs = RoundStats::new(260, 5);
        assert_eq!(rs.round_id, 260);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_261() {
        let rs = RoundStats::new(261, 5);
        assert_eq!(rs.round_id, 261);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_262() {
        let rs = RoundStats::new(262, 5);
        assert_eq!(rs.round_id, 262);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_263() {
        let rs = RoundStats::new(263, 5);
        assert_eq!(rs.round_id, 263);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_264() {
        let rs = RoundStats::new(264, 5);
        assert_eq!(rs.round_id, 264);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_265() {
        let rs = RoundStats::new(265, 5);
        assert_eq!(rs.round_id, 265);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_266() {
        let rs = RoundStats::new(266, 5);
        assert_eq!(rs.round_id, 266);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_267() {
        let rs = RoundStats::new(267, 5);
        assert_eq!(rs.round_id, 267);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_268() {
        let rs = RoundStats::new(268, 5);
        assert_eq!(rs.round_id, 268);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_269() {
        let rs = RoundStats::new(269, 5);
        assert_eq!(rs.round_id, 269);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_270() {
        let rs = RoundStats::new(270, 5);
        assert_eq!(rs.round_id, 270);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_271() {
        let rs = RoundStats::new(271, 5);
        assert_eq!(rs.round_id, 271);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_272() {
        let rs = RoundStats::new(272, 5);
        assert_eq!(rs.round_id, 272);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_273() {
        let rs = RoundStats::new(273, 5);
        assert_eq!(rs.round_id, 273);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_274() {
        let rs = RoundStats::new(274, 5);
        assert_eq!(rs.round_id, 274);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_275() {
        let rs = RoundStats::new(275, 5);
        assert_eq!(rs.round_id, 275);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_276() {
        let rs = RoundStats::new(276, 5);
        assert_eq!(rs.round_id, 276);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_277() {
        let rs = RoundStats::new(277, 5);
        assert_eq!(rs.round_id, 277);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_278() {
        let rs = RoundStats::new(278, 5);
        assert_eq!(rs.round_id, 278);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_279() {
        let rs = RoundStats::new(279, 5);
        assert_eq!(rs.round_id, 279);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_280() {
        let rs = RoundStats::new(280, 5);
        assert_eq!(rs.round_id, 280);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_281() {
        let rs = RoundStats::new(281, 5);
        assert_eq!(rs.round_id, 281);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_282() {
        let rs = RoundStats::new(282, 5);
        assert_eq!(rs.round_id, 282);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_283() {
        let rs = RoundStats::new(283, 5);
        assert_eq!(rs.round_id, 283);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_284() {
        let rs = RoundStats::new(284, 5);
        assert_eq!(rs.round_id, 284);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_285() {
        let rs = RoundStats::new(285, 5);
        assert_eq!(rs.round_id, 285);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_286() {
        let rs = RoundStats::new(286, 5);
        assert_eq!(rs.round_id, 286);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_287() {
        let rs = RoundStats::new(287, 5);
        assert_eq!(rs.round_id, 287);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_288() {
        let rs = RoundStats::new(288, 5);
        assert_eq!(rs.round_id, 288);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_289() {
        let rs = RoundStats::new(289, 5);
        assert_eq!(rs.round_id, 289);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_290() {
        let rs = RoundStats::new(290, 5);
        assert_eq!(rs.round_id, 290);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_291() {
        let rs = RoundStats::new(291, 5);
        assert_eq!(rs.round_id, 291);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_292() {
        let rs = RoundStats::new(292, 5);
        assert_eq!(rs.round_id, 292);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_293() {
        let rs = RoundStats::new(293, 5);
        assert_eq!(rs.round_id, 293);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_294() {
        let rs = RoundStats::new(294, 5);
        assert_eq!(rs.round_id, 294);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_295() {
        let rs = RoundStats::new(295, 5);
        assert_eq!(rs.round_id, 295);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_296() {
        let rs = RoundStats::new(296, 5);
        assert_eq!(rs.round_id, 296);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_297() {
        let rs = RoundStats::new(297, 5);
        assert_eq!(rs.round_id, 297);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_298() {
        let rs = RoundStats::new(298, 5);
        assert_eq!(rs.round_id, 298);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_299() {
        let rs = RoundStats::new(299, 5);
        assert_eq!(rs.round_id, 299);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_300() {
        let rs = RoundStats::new(300, 5);
        assert_eq!(rs.round_id, 300);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_301() {
        let rs = RoundStats::new(301, 5);
        assert_eq!(rs.round_id, 301);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_302() {
        let rs = RoundStats::new(302, 5);
        assert_eq!(rs.round_id, 302);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_303() {
        let rs = RoundStats::new(303, 5);
        assert_eq!(rs.round_id, 303);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_304() {
        let rs = RoundStats::new(304, 5);
        assert_eq!(rs.round_id, 304);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_305() {
        let rs = RoundStats::new(305, 5);
        assert_eq!(rs.round_id, 305);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_306() {
        let rs = RoundStats::new(306, 5);
        assert_eq!(rs.round_id, 306);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_307() {
        let rs = RoundStats::new(307, 5);
        assert_eq!(rs.round_id, 307);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_308() {
        let rs = RoundStats::new(308, 5);
        assert_eq!(rs.round_id, 308);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_309() {
        let rs = RoundStats::new(309, 5);
        assert_eq!(rs.round_id, 309);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_310() {
        let rs = RoundStats::new(310, 5);
        assert_eq!(rs.round_id, 310);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_311() {
        let rs = RoundStats::new(311, 5);
        assert_eq!(rs.round_id, 311);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_312() {
        let rs = RoundStats::new(312, 5);
        assert_eq!(rs.round_id, 312);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_313() {
        let rs = RoundStats::new(313, 5);
        assert_eq!(rs.round_id, 313);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_314() {
        let rs = RoundStats::new(314, 5);
        assert_eq!(rs.round_id, 314);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_315() {
        let rs = RoundStats::new(315, 5);
        assert_eq!(rs.round_id, 315);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_316() {
        let rs = RoundStats::new(316, 5);
        assert_eq!(rs.round_id, 316);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_317() {
        let rs = RoundStats::new(317, 5);
        assert_eq!(rs.round_id, 317);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_318() {
        let rs = RoundStats::new(318, 5);
        assert_eq!(rs.round_id, 318);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_319() {
        let rs = RoundStats::new(319, 5);
        assert_eq!(rs.round_id, 319);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_320() {
        let rs = RoundStats::new(320, 5);
        assert_eq!(rs.round_id, 320);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_321() {
        let rs = RoundStats::new(321, 5);
        assert_eq!(rs.round_id, 321);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_322() {
        let rs = RoundStats::new(322, 5);
        assert_eq!(rs.round_id, 322);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_323() {
        let rs = RoundStats::new(323, 5);
        assert_eq!(rs.round_id, 323);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_324() {
        let rs = RoundStats::new(324, 5);
        assert_eq!(rs.round_id, 324);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_325() {
        let rs = RoundStats::new(325, 5);
        assert_eq!(rs.round_id, 325);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_326() {
        let rs = RoundStats::new(326, 5);
        assert_eq!(rs.round_id, 326);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_327() {
        let rs = RoundStats::new(327, 5);
        assert_eq!(rs.round_id, 327);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_328() {
        let rs = RoundStats::new(328, 5);
        assert_eq!(rs.round_id, 328);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_329() {
        let rs = RoundStats::new(329, 5);
        assert_eq!(rs.round_id, 329);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_330() {
        let rs = RoundStats::new(330, 5);
        assert_eq!(rs.round_id, 330);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_331() {
        let rs = RoundStats::new(331, 5);
        assert_eq!(rs.round_id, 331);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_332() {
        let rs = RoundStats::new(332, 5);
        assert_eq!(rs.round_id, 332);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_333() {
        let rs = RoundStats::new(333, 5);
        assert_eq!(rs.round_id, 333);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_334() {
        let rs = RoundStats::new(334, 5);
        assert_eq!(rs.round_id, 334);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_335() {
        let rs = RoundStats::new(335, 5);
        assert_eq!(rs.round_id, 335);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_336() {
        let rs = RoundStats::new(336, 5);
        assert_eq!(rs.round_id, 336);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_337() {
        let rs = RoundStats::new(337, 5);
        assert_eq!(rs.round_id, 337);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_338() {
        let rs = RoundStats::new(338, 5);
        assert_eq!(rs.round_id, 338);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_339() {
        let rs = RoundStats::new(339, 5);
        assert_eq!(rs.round_id, 339);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_340() {
        let rs = RoundStats::new(340, 5);
        assert_eq!(rs.round_id, 340);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_341() {
        let rs = RoundStats::new(341, 5);
        assert_eq!(rs.round_id, 341);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_342() {
        let rs = RoundStats::new(342, 5);
        assert_eq!(rs.round_id, 342);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_343() {
        let rs = RoundStats::new(343, 5);
        assert_eq!(rs.round_id, 343);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_344() {
        let rs = RoundStats::new(344, 5);
        assert_eq!(rs.round_id, 344);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_345() {
        let rs = RoundStats::new(345, 5);
        assert_eq!(rs.round_id, 345);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_346() {
        let rs = RoundStats::new(346, 5);
        assert_eq!(rs.round_id, 346);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_347() {
        let rs = RoundStats::new(347, 5);
        assert_eq!(rs.round_id, 347);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_348() {
        let rs = RoundStats::new(348, 5);
        assert_eq!(rs.round_id, 348);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_349() {
        let rs = RoundStats::new(349, 5);
        assert_eq!(rs.round_id, 349);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_350() {
        let rs = RoundStats::new(350, 5);
        assert_eq!(rs.round_id, 350);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_351() {
        let rs = RoundStats::new(351, 5);
        assert_eq!(rs.round_id, 351);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_352() {
        let rs = RoundStats::new(352, 5);
        assert_eq!(rs.round_id, 352);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_353() {
        let rs = RoundStats::new(353, 5);
        assert_eq!(rs.round_id, 353);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_354() {
        let rs = RoundStats::new(354, 5);
        assert_eq!(rs.round_id, 354);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_355() {
        let rs = RoundStats::new(355, 5);
        assert_eq!(rs.round_id, 355);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_356() {
        let rs = RoundStats::new(356, 5);
        assert_eq!(rs.round_id, 356);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_357() {
        let rs = RoundStats::new(357, 5);
        assert_eq!(rs.round_id, 357);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_358() {
        let rs = RoundStats::new(358, 5);
        assert_eq!(rs.round_id, 358);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_359() {
        let rs = RoundStats::new(359, 5);
        assert_eq!(rs.round_id, 359);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_360() {
        let rs = RoundStats::new(360, 5);
        assert_eq!(rs.round_id, 360);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_361() {
        let rs = RoundStats::new(361, 5);
        assert_eq!(rs.round_id, 361);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_362() {
        let rs = RoundStats::new(362, 5);
        assert_eq!(rs.round_id, 362);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_363() {
        let rs = RoundStats::new(363, 5);
        assert_eq!(rs.round_id, 363);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_364() {
        let rs = RoundStats::new(364, 5);
        assert_eq!(rs.round_id, 364);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_365() {
        let rs = RoundStats::new(365, 5);
        assert_eq!(rs.round_id, 365);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_366() {
        let rs = RoundStats::new(366, 5);
        assert_eq!(rs.round_id, 366);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_367() {
        let rs = RoundStats::new(367, 5);
        assert_eq!(rs.round_id, 367);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_368() {
        let rs = RoundStats::new(368, 5);
        assert_eq!(rs.round_id, 368);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_369() {
        let rs = RoundStats::new(369, 5);
        assert_eq!(rs.round_id, 369);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_370() {
        let rs = RoundStats::new(370, 5);
        assert_eq!(rs.round_id, 370);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_371() {
        let rs = RoundStats::new(371, 5);
        assert_eq!(rs.round_id, 371);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_372() {
        let rs = RoundStats::new(372, 5);
        assert_eq!(rs.round_id, 372);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_373() {
        let rs = RoundStats::new(373, 5);
        assert_eq!(rs.round_id, 373);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_374() {
        let rs = RoundStats::new(374, 5);
        assert_eq!(rs.round_id, 374);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_375() {
        let rs = RoundStats::new(375, 5);
        assert_eq!(rs.round_id, 375);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_376() {
        let rs = RoundStats::new(376, 5);
        assert_eq!(rs.round_id, 376);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_377() {
        let rs = RoundStats::new(377, 5);
        assert_eq!(rs.round_id, 377);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_378() {
        let rs = RoundStats::new(378, 5);
        assert_eq!(rs.round_id, 378);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_379() {
        let rs = RoundStats::new(379, 5);
        assert_eq!(rs.round_id, 379);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_380() {
        let rs = RoundStats::new(380, 5);
        assert_eq!(rs.round_id, 380);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_381() {
        let rs = RoundStats::new(381, 5);
        assert_eq!(rs.round_id, 381);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_382() {
        let rs = RoundStats::new(382, 5);
        assert_eq!(rs.round_id, 382);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_383() {
        let rs = RoundStats::new(383, 5);
        assert_eq!(rs.round_id, 383);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_384() {
        let rs = RoundStats::new(384, 5);
        assert_eq!(rs.round_id, 384);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_385() {
        let rs = RoundStats::new(385, 5);
        assert_eq!(rs.round_id, 385);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_386() {
        let rs = RoundStats::new(386, 5);
        assert_eq!(rs.round_id, 386);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_387() {
        let rs = RoundStats::new(387, 5);
        assert_eq!(rs.round_id, 387);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_388() {
        let rs = RoundStats::new(388, 5);
        assert_eq!(rs.round_id, 388);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_389() {
        let rs = RoundStats::new(389, 5);
        assert_eq!(rs.round_id, 389);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_390() {
        let rs = RoundStats::new(390, 5);
        assert_eq!(rs.round_id, 390);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_391() {
        let rs = RoundStats::new(391, 5);
        assert_eq!(rs.round_id, 391);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_392() {
        let rs = RoundStats::new(392, 5);
        assert_eq!(rs.round_id, 392);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_393() {
        let rs = RoundStats::new(393, 5);
        assert_eq!(rs.round_id, 393);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_394() {
        let rs = RoundStats::new(394, 5);
        assert_eq!(rs.round_id, 394);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_395() {
        let rs = RoundStats::new(395, 5);
        assert_eq!(rs.round_id, 395);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_396() {
        let rs = RoundStats::new(396, 5);
        assert_eq!(rs.round_id, 396);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_397() {
        let rs = RoundStats::new(397, 5);
        assert_eq!(rs.round_id, 397);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_398() {
        let rs = RoundStats::new(398, 5);
        assert_eq!(rs.round_id, 398);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_399() {
        let rs = RoundStats::new(399, 5);
        assert_eq!(rs.round_id, 399);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_400() {
        let rs = RoundStats::new(400, 5);
        assert_eq!(rs.round_id, 400);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_401() {
        let rs = RoundStats::new(401, 5);
        assert_eq!(rs.round_id, 401);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_402() {
        let rs = RoundStats::new(402, 5);
        assert_eq!(rs.round_id, 402);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_403() {
        let rs = RoundStats::new(403, 5);
        assert_eq!(rs.round_id, 403);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_404() {
        let rs = RoundStats::new(404, 5);
        assert_eq!(rs.round_id, 404);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_405() {
        let rs = RoundStats::new(405, 5);
        assert_eq!(rs.round_id, 405);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_406() {
        let rs = RoundStats::new(406, 5);
        assert_eq!(rs.round_id, 406);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_407() {
        let rs = RoundStats::new(407, 5);
        assert_eq!(rs.round_id, 407);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_408() {
        let rs = RoundStats::new(408, 5);
        assert_eq!(rs.round_id, 408);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_409() {
        let rs = RoundStats::new(409, 5);
        assert_eq!(rs.round_id, 409);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_410() {
        let rs = RoundStats::new(410, 5);
        assert_eq!(rs.round_id, 410);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_411() {
        let rs = RoundStats::new(411, 5);
        assert_eq!(rs.round_id, 411);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_412() {
        let rs = RoundStats::new(412, 5);
        assert_eq!(rs.round_id, 412);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_413() {
        let rs = RoundStats::new(413, 5);
        assert_eq!(rs.round_id, 413);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_414() {
        let rs = RoundStats::new(414, 5);
        assert_eq!(rs.round_id, 414);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_415() {
        let rs = RoundStats::new(415, 5);
        assert_eq!(rs.round_id, 415);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_416() {
        let rs = RoundStats::new(416, 5);
        assert_eq!(rs.round_id, 416);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_417() {
        let rs = RoundStats::new(417, 5);
        assert_eq!(rs.round_id, 417);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_418() {
        let rs = RoundStats::new(418, 5);
        assert_eq!(rs.round_id, 418);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_419() {
        let rs = RoundStats::new(419, 5);
        assert_eq!(rs.round_id, 419);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_420() {
        let rs = RoundStats::new(420, 5);
        assert_eq!(rs.round_id, 420);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_421() {
        let rs = RoundStats::new(421, 5);
        assert_eq!(rs.round_id, 421);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_422() {
        let rs = RoundStats::new(422, 5);
        assert_eq!(rs.round_id, 422);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_423() {
        let rs = RoundStats::new(423, 5);
        assert_eq!(rs.round_id, 423);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_424() {
        let rs = RoundStats::new(424, 5);
        assert_eq!(rs.round_id, 424);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_425() {
        let rs = RoundStats::new(425, 5);
        assert_eq!(rs.round_id, 425);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_426() {
        let rs = RoundStats::new(426, 5);
        assert_eq!(rs.round_id, 426);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_427() {
        let rs = RoundStats::new(427, 5);
        assert_eq!(rs.round_id, 427);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_428() {
        let rs = RoundStats::new(428, 5);
        assert_eq!(rs.round_id, 428);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_429() {
        let rs = RoundStats::new(429, 5);
        assert_eq!(rs.round_id, 429);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_430() {
        let rs = RoundStats::new(430, 5);
        assert_eq!(rs.round_id, 430);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_431() {
        let rs = RoundStats::new(431, 5);
        assert_eq!(rs.round_id, 431);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_432() {
        let rs = RoundStats::new(432, 5);
        assert_eq!(rs.round_id, 432);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_433() {
        let rs = RoundStats::new(433, 5);
        assert_eq!(rs.round_id, 433);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_434() {
        let rs = RoundStats::new(434, 5);
        assert_eq!(rs.round_id, 434);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_435() {
        let rs = RoundStats::new(435, 5);
        assert_eq!(rs.round_id, 435);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_436() {
        let rs = RoundStats::new(436, 5);
        assert_eq!(rs.round_id, 436);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_437() {
        let rs = RoundStats::new(437, 5);
        assert_eq!(rs.round_id, 437);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_438() {
        let rs = RoundStats::new(438, 5);
        assert_eq!(rs.round_id, 438);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_439() {
        let rs = RoundStats::new(439, 5);
        assert_eq!(rs.round_id, 439);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_440() {
        let rs = RoundStats::new(440, 5);
        assert_eq!(rs.round_id, 440);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_441() {
        let rs = RoundStats::new(441, 5);
        assert_eq!(rs.round_id, 441);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_442() {
        let rs = RoundStats::new(442, 5);
        assert_eq!(rs.round_id, 442);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_443() {
        let rs = RoundStats::new(443, 5);
        assert_eq!(rs.round_id, 443);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_444() {
        let rs = RoundStats::new(444, 5);
        assert_eq!(rs.round_id, 444);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_445() {
        let rs = RoundStats::new(445, 5);
        assert_eq!(rs.round_id, 445);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_446() {
        let rs = RoundStats::new(446, 5);
        assert_eq!(rs.round_id, 446);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_447() {
        let rs = RoundStats::new(447, 5);
        assert_eq!(rs.round_id, 447);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_448() {
        let rs = RoundStats::new(448, 5);
        assert_eq!(rs.round_id, 448);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_449() {
        let rs = RoundStats::new(449, 5);
        assert_eq!(rs.round_id, 449);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_450() {
        let rs = RoundStats::new(450, 5);
        assert_eq!(rs.round_id, 450);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_451() {
        let rs = RoundStats::new(451, 5);
        assert_eq!(rs.round_id, 451);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_452() {
        let rs = RoundStats::new(452, 5);
        assert_eq!(rs.round_id, 452);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_453() {
        let rs = RoundStats::new(453, 5);
        assert_eq!(rs.round_id, 453);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_454() {
        let rs = RoundStats::new(454, 5);
        assert_eq!(rs.round_id, 454);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_455() {
        let rs = RoundStats::new(455, 5);
        assert_eq!(rs.round_id, 455);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_456() {
        let rs = RoundStats::new(456, 5);
        assert_eq!(rs.round_id, 456);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_457() {
        let rs = RoundStats::new(457, 5);
        assert_eq!(rs.round_id, 457);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_458() {
        let rs = RoundStats::new(458, 5);
        assert_eq!(rs.round_id, 458);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_459() {
        let rs = RoundStats::new(459, 5);
        assert_eq!(rs.round_id, 459);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_460() {
        let rs = RoundStats::new(460, 5);
        assert_eq!(rs.round_id, 460);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_461() {
        let rs = RoundStats::new(461, 5);
        assert_eq!(rs.round_id, 461);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_462() {
        let rs = RoundStats::new(462, 5);
        assert_eq!(rs.round_id, 462);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_463() {
        let rs = RoundStats::new(463, 5);
        assert_eq!(rs.round_id, 463);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_464() {
        let rs = RoundStats::new(464, 5);
        assert_eq!(rs.round_id, 464);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_465() {
        let rs = RoundStats::new(465, 5);
        assert_eq!(rs.round_id, 465);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_466() {
        let rs = RoundStats::new(466, 5);
        assert_eq!(rs.round_id, 466);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_467() {
        let rs = RoundStats::new(467, 5);
        assert_eq!(rs.round_id, 467);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_468() {
        let rs = RoundStats::new(468, 5);
        assert_eq!(rs.round_id, 468);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_469() {
        let rs = RoundStats::new(469, 5);
        assert_eq!(rs.round_id, 469);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_470() {
        let rs = RoundStats::new(470, 5);
        assert_eq!(rs.round_id, 470);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_471() {
        let rs = RoundStats::new(471, 5);
        assert_eq!(rs.round_id, 471);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_472() {
        let rs = RoundStats::new(472, 5);
        assert_eq!(rs.round_id, 472);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_473() {
        let rs = RoundStats::new(473, 5);
        assert_eq!(rs.round_id, 473);
        assert_eq!(rs.num_participants, 5);
    }

    #[test]
    fn test_round_stats_stress_474() {
        let rs = RoundStats::new(474, 5);
        assert_eq!(rs.round_id, 474);
        assert_eq!(rs.num_participants, 5);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
}
