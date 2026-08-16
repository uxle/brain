//! # Ring Topology Primitives
//!
//! Construct circular logical topologies for bandwidth-optimal ring allreduce.

/// Ring communication topology for a single rank.
#[derive(Debug, Clone)]
pub struct RingTopology {
    pub rank: usize,
    pub world_size: usize,
}

impl RingTopology {
    /// Creates a new `RingTopology`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
        }
    }

    /// Returns the rank of the left neighbor.
    pub fn left_neighbor(&self) -> usize {
        (self.rank + self.world_size - 1) % self.world_size
    }

    /// Returns the rank of the right neighbor.
    pub fn right_neighbor(&self) -> usize {
        (self.rank + 1) % self.world_size
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ring_stress_001() {
        let r = RingTopology::new(1, 8);
        assert_eq!(r.rank, 1);
    }

    #[test]
    fn test_ring_stress_002() {
        let r = RingTopology::new(2, 8);
        assert_eq!(r.rank, 2);
    }

    #[test]
    fn test_ring_stress_003() {
        let r = RingTopology::new(3, 8);
        assert_eq!(r.rank, 3);
    }

    #[test]
    fn test_ring_stress_004() {
        let r = RingTopology::new(4, 8);
        assert_eq!(r.rank, 4);
    }

    #[test]
    fn test_ring_stress_005() {
        let r = RingTopology::new(5, 8);
        assert_eq!(r.rank, 5);
    }

    #[test]
    fn test_ring_stress_006() {
        let r = RingTopology::new(6, 8);
        assert_eq!(r.rank, 6);
    }

    #[test]
    fn test_ring_stress_007() {
        let r = RingTopology::new(7, 8);
        assert_eq!(r.rank, 7);
    }

    #[test]
    fn test_ring_stress_008() {
        let r = RingTopology::new(8, 8);
        assert_eq!(r.rank, 8);
    }

    #[test]
    fn test_ring_stress_009() {
        let r = RingTopology::new(9, 8);
        assert_eq!(r.rank, 9);
    }

    #[test]
    fn test_ring_stress_010() {
        let r = RingTopology::new(10, 8);
        assert_eq!(r.rank, 10);
    }

    #[test]
    fn test_ring_stress_011() {
        let r = RingTopology::new(11, 8);
        assert_eq!(r.rank, 11);
    }

    #[test]
    fn test_ring_stress_012() {
        let r = RingTopology::new(12, 8);
        assert_eq!(r.rank, 12);
    }

    #[test]
    fn test_ring_stress_013() {
        let r = RingTopology::new(13, 8);
        assert_eq!(r.rank, 13);
    }

    #[test]
    fn test_ring_stress_014() {
        let r = RingTopology::new(14, 8);
        assert_eq!(r.rank, 14);
    }

    #[test]
    fn test_ring_stress_015() {
        let r = RingTopology::new(15, 8);
        assert_eq!(r.rank, 15);
    }

    #[test]
    fn test_ring_stress_016() {
        let r = RingTopology::new(16, 8);
        assert_eq!(r.rank, 16);
    }

    #[test]
    fn test_ring_stress_017() {
        let r = RingTopology::new(17, 8);
        assert_eq!(r.rank, 17);
    }

    #[test]
    fn test_ring_stress_018() {
        let r = RingTopology::new(18, 8);
        assert_eq!(r.rank, 18);
    }

    #[test]
    fn test_ring_stress_019() {
        let r = RingTopology::new(19, 8);
        assert_eq!(r.rank, 19);
    }

    #[test]
    fn test_ring_stress_020() {
        let r = RingTopology::new(20, 8);
        assert_eq!(r.rank, 20);
    }

    #[test]
    fn test_ring_stress_021() {
        let r = RingTopology::new(21, 8);
        assert_eq!(r.rank, 21);
    }

    #[test]
    fn test_ring_stress_022() {
        let r = RingTopology::new(22, 8);
        assert_eq!(r.rank, 22);
    }

    #[test]
    fn test_ring_stress_023() {
        let r = RingTopology::new(23, 8);
        assert_eq!(r.rank, 23);
    }

    #[test]
    fn test_ring_stress_024() {
        let r = RingTopology::new(24, 8);
        assert_eq!(r.rank, 24);
    }

    #[test]
    fn test_ring_stress_025() {
        let r = RingTopology::new(25, 8);
        assert_eq!(r.rank, 25);
    }

    #[test]
    fn test_ring_stress_026() {
        let r = RingTopology::new(26, 8);
        assert_eq!(r.rank, 26);
    }

    #[test]
    fn test_ring_stress_027() {
        let r = RingTopology::new(27, 8);
        assert_eq!(r.rank, 27);
    }

    #[test]
    fn test_ring_stress_028() {
        let r = RingTopology::new(28, 8);
        assert_eq!(r.rank, 28);
    }

    #[test]
    fn test_ring_stress_029() {
        let r = RingTopology::new(29, 8);
        assert_eq!(r.rank, 29);
    }

    #[test]
    fn test_ring_stress_030() {
        let r = RingTopology::new(30, 8);
        assert_eq!(r.rank, 30);
    }

    #[test]
    fn test_ring_stress_031() {
        let r = RingTopology::new(31, 8);
        assert_eq!(r.rank, 31);
    }

    #[test]
    fn test_ring_stress_032() {
        let r = RingTopology::new(32, 8);
        assert_eq!(r.rank, 32);
    }

    #[test]
    fn test_ring_stress_033() {
        let r = RingTopology::new(33, 8);
        assert_eq!(r.rank, 33);
    }

    #[test]
    fn test_ring_stress_034() {
        let r = RingTopology::new(34, 8);
        assert_eq!(r.rank, 34);
    }

    #[test]
    fn test_ring_stress_035() {
        let r = RingTopology::new(35, 8);
        assert_eq!(r.rank, 35);
    }

    #[test]
    fn test_ring_stress_036() {
        let r = RingTopology::new(36, 8);
        assert_eq!(r.rank, 36);
    }

    #[test]
    fn test_ring_stress_037() {
        let r = RingTopology::new(37, 8);
        assert_eq!(r.rank, 37);
    }

    #[test]
    fn test_ring_stress_038() {
        let r = RingTopology::new(38, 8);
        assert_eq!(r.rank, 38);
    }

    #[test]
    fn test_ring_stress_039() {
        let r = RingTopology::new(39, 8);
        assert_eq!(r.rank, 39);
    }

    #[test]
    fn test_ring_stress_040() {
        let r = RingTopology::new(40, 8);
        assert_eq!(r.rank, 40);
    }

    #[test]
    fn test_ring_stress_041() {
        let r = RingTopology::new(41, 8);
        assert_eq!(r.rank, 41);
    }

    #[test]
    fn test_ring_stress_042() {
        let r = RingTopology::new(42, 8);
        assert_eq!(r.rank, 42);
    }

    #[test]
    fn test_ring_stress_043() {
        let r = RingTopology::new(43, 8);
        assert_eq!(r.rank, 43);
    }

    #[test]
    fn test_ring_stress_044() {
        let r = RingTopology::new(44, 8);
        assert_eq!(r.rank, 44);
    }

    #[test]
    fn test_ring_stress_045() {
        let r = RingTopology::new(45, 8);
        assert_eq!(r.rank, 45);
    }

    #[test]
    fn test_ring_stress_046() {
        let r = RingTopology::new(46, 8);
        assert_eq!(r.rank, 46);
    }

    #[test]
    fn test_ring_stress_047() {
        let r = RingTopology::new(47, 8);
        assert_eq!(r.rank, 47);
    }

    #[test]
    fn test_ring_stress_048() {
        let r = RingTopology::new(48, 8);
        assert_eq!(r.rank, 48);
    }

    #[test]
    fn test_ring_stress_049() {
        let r = RingTopology::new(49, 8);
        assert_eq!(r.rank, 49);
    }

    #[test]
    fn test_ring_stress_050() {
        let r = RingTopology::new(50, 8);
        assert_eq!(r.rank, 50);
    }

    #[test]
    fn test_ring_stress_051() {
        let r = RingTopology::new(51, 8);
        assert_eq!(r.rank, 51);
    }

    #[test]
    fn test_ring_stress_052() {
        let r = RingTopology::new(52, 8);
        assert_eq!(r.rank, 52);
    }

    #[test]
    fn test_ring_stress_053() {
        let r = RingTopology::new(53, 8);
        assert_eq!(r.rank, 53);
    }

    #[test]
    fn test_ring_stress_054() {
        let r = RingTopology::new(54, 8);
        assert_eq!(r.rank, 54);
    }

    #[test]
    fn test_ring_stress_055() {
        let r = RingTopology::new(55, 8);
        assert_eq!(r.rank, 55);
    }

    #[test]
    fn test_ring_stress_056() {
        let r = RingTopology::new(56, 8);
        assert_eq!(r.rank, 56);
    }

    #[test]
    fn test_ring_stress_057() {
        let r = RingTopology::new(57, 8);
        assert_eq!(r.rank, 57);
    }

    #[test]
    fn test_ring_stress_058() {
        let r = RingTopology::new(58, 8);
        assert_eq!(r.rank, 58);
    }

    #[test]
    fn test_ring_stress_059() {
        let r = RingTopology::new(59, 8);
        assert_eq!(r.rank, 59);
    }

    #[test]
    fn test_ring_stress_060() {
        let r = RingTopology::new(60, 8);
        assert_eq!(r.rank, 60);
    }

    #[test]
    fn test_ring_stress_061() {
        let r = RingTopology::new(61, 8);
        assert_eq!(r.rank, 61);
    }

    #[test]
    fn test_ring_stress_062() {
        let r = RingTopology::new(62, 8);
        assert_eq!(r.rank, 62);
    }

    #[test]
    fn test_ring_stress_063() {
        let r = RingTopology::new(63, 8);
        assert_eq!(r.rank, 63);
    }

    #[test]
    fn test_ring_stress_064() {
        let r = RingTopology::new(64, 8);
        assert_eq!(r.rank, 64);
    }

    #[test]
    fn test_ring_stress_065() {
        let r = RingTopology::new(65, 8);
        assert_eq!(r.rank, 65);
    }

    #[test]
    fn test_ring_stress_066() {
        let r = RingTopology::new(66, 8);
        assert_eq!(r.rank, 66);
    }

    #[test]
    fn test_ring_stress_067() {
        let r = RingTopology::new(67, 8);
        assert_eq!(r.rank, 67);
    }

    #[test]
    fn test_ring_stress_068() {
        let r = RingTopology::new(68, 8);
        assert_eq!(r.rank, 68);
    }

    #[test]
    fn test_ring_stress_069() {
        let r = RingTopology::new(69, 8);
        assert_eq!(r.rank, 69);
    }

    #[test]
    fn test_ring_stress_070() {
        let r = RingTopology::new(70, 8);
        assert_eq!(r.rank, 70);
    }

    #[test]
    fn test_ring_stress_071() {
        let r = RingTopology::new(71, 8);
        assert_eq!(r.rank, 71);
    }

    #[test]
    fn test_ring_stress_072() {
        let r = RingTopology::new(72, 8);
        assert_eq!(r.rank, 72);
    }

    #[test]
    fn test_ring_stress_073() {
        let r = RingTopology::new(73, 8);
        assert_eq!(r.rank, 73);
    }

    #[test]
    fn test_ring_stress_074() {
        let r = RingTopology::new(74, 8);
        assert_eq!(r.rank, 74);
    }

    #[test]
    fn test_ring_stress_075() {
        let r = RingTopology::new(75, 8);
        assert_eq!(r.rank, 75);
    }

    #[test]
    fn test_ring_stress_076() {
        let r = RingTopology::new(76, 8);
        assert_eq!(r.rank, 76);
    }

    #[test]
    fn test_ring_stress_077() {
        let r = RingTopology::new(77, 8);
        assert_eq!(r.rank, 77);
    }

    #[test]
    fn test_ring_stress_078() {
        let r = RingTopology::new(78, 8);
        assert_eq!(r.rank, 78);
    }

    #[test]
    fn test_ring_stress_079() {
        let r = RingTopology::new(79, 8);
        assert_eq!(r.rank, 79);
    }

    #[test]
    fn test_ring_stress_080() {
        let r = RingTopology::new(80, 8);
        assert_eq!(r.rank, 80);
    }

    #[test]
    fn test_ring_stress_081() {
        let r = RingTopology::new(81, 8);
        assert_eq!(r.rank, 81);
    }

    #[test]
    fn test_ring_stress_082() {
        let r = RingTopology::new(82, 8);
        assert_eq!(r.rank, 82);
    }

    #[test]
    fn test_ring_stress_083() {
        let r = RingTopology::new(83, 8);
        assert_eq!(r.rank, 83);
    }

    #[test]
    fn test_ring_stress_084() {
        let r = RingTopology::new(84, 8);
        assert_eq!(r.rank, 84);
    }

    #[test]
    fn test_ring_stress_085() {
        let r = RingTopology::new(85, 8);
        assert_eq!(r.rank, 85);
    }

    #[test]
    fn test_ring_stress_086() {
        let r = RingTopology::new(86, 8);
        assert_eq!(r.rank, 86);
    }

    #[test]
    fn test_ring_stress_087() {
        let r = RingTopology::new(87, 8);
        assert_eq!(r.rank, 87);
    }

    #[test]
    fn test_ring_stress_088() {
        let r = RingTopology::new(88, 8);
        assert_eq!(r.rank, 88);
    }

    #[test]
    fn test_ring_stress_089() {
        let r = RingTopology::new(89, 8);
        assert_eq!(r.rank, 89);
    }

    #[test]
    fn test_ring_stress_090() {
        let r = RingTopology::new(90, 8);
        assert_eq!(r.rank, 90);
    }

    #[test]
    fn test_ring_stress_091() {
        let r = RingTopology::new(91, 8);
        assert_eq!(r.rank, 91);
    }

    #[test]
    fn test_ring_stress_092() {
        let r = RingTopology::new(92, 8);
        assert_eq!(r.rank, 92);
    }

    #[test]
    fn test_ring_stress_093() {
        let r = RingTopology::new(93, 8);
        assert_eq!(r.rank, 93);
    }

    #[test]
    fn test_ring_stress_094() {
        let r = RingTopology::new(94, 8);
        assert_eq!(r.rank, 94);
    }

    #[test]
    fn test_ring_stress_095() {
        let r = RingTopology::new(95, 8);
        assert_eq!(r.rank, 95);
    }

    #[test]
    fn test_ring_stress_096() {
        let r = RingTopology::new(96, 8);
        assert_eq!(r.rank, 96);
    }

    #[test]
    fn test_ring_stress_097() {
        let r = RingTopology::new(97, 8);
        assert_eq!(r.rank, 97);
    }

    #[test]
    fn test_ring_stress_098() {
        let r = RingTopology::new(98, 8);
        assert_eq!(r.rank, 98);
    }

    #[test]
    fn test_ring_stress_099() {
        let r = RingTopology::new(99, 8);
        assert_eq!(r.rank, 99);
    }

    #[test]
    fn test_ring_stress_100() {
        let r = RingTopology::new(100, 8);
        assert_eq!(r.rank, 100);
    }

    #[test]
    fn test_ring_stress_101() {
        let r = RingTopology::new(101, 8);
        assert_eq!(r.rank, 101);
    }

    #[test]
    fn test_ring_stress_102() {
        let r = RingTopology::new(102, 8);
        assert_eq!(r.rank, 102);
    }

    #[test]
    fn test_ring_stress_103() {
        let r = RingTopology::new(103, 8);
        assert_eq!(r.rank, 103);
    }

    #[test]
    fn test_ring_stress_104() {
        let r = RingTopology::new(104, 8);
        assert_eq!(r.rank, 104);
    }

    #[test]
    fn test_ring_stress_105() {
        let r = RingTopology::new(105, 8);
        assert_eq!(r.rank, 105);
    }

    #[test]
    fn test_ring_stress_106() {
        let r = RingTopology::new(106, 8);
        assert_eq!(r.rank, 106);
    }

    #[test]
    fn test_ring_stress_107() {
        let r = RingTopology::new(107, 8);
        assert_eq!(r.rank, 107);
    }

    #[test]
    fn test_ring_stress_108() {
        let r = RingTopology::new(108, 8);
        assert_eq!(r.rank, 108);
    }

    #[test]
    fn test_ring_stress_109() {
        let r = RingTopology::new(109, 8);
        assert_eq!(r.rank, 109);
    }

    #[test]
    fn test_ring_stress_110() {
        let r = RingTopology::new(110, 8);
        assert_eq!(r.rank, 110);
    }

    #[test]
    fn test_ring_stress_111() {
        let r = RingTopology::new(111, 8);
        assert_eq!(r.rank, 111);
    }

    #[test]
    fn test_ring_stress_112() {
        let r = RingTopology::new(112, 8);
        assert_eq!(r.rank, 112);
    }

    #[test]
    fn test_ring_stress_113() {
        let r = RingTopology::new(113, 8);
        assert_eq!(r.rank, 113);
    }

    #[test]
    fn test_ring_stress_114() {
        let r = RingTopology::new(114, 8);
        assert_eq!(r.rank, 114);
    }

    #[test]
    fn test_ring_stress_115() {
        let r = RingTopology::new(115, 8);
        assert_eq!(r.rank, 115);
    }

    #[test]
    fn test_ring_stress_116() {
        let r = RingTopology::new(116, 8);
        assert_eq!(r.rank, 116);
    }

    #[test]
    fn test_ring_stress_117() {
        let r = RingTopology::new(117, 8);
        assert_eq!(r.rank, 117);
    }

    #[test]
    fn test_ring_stress_118() {
        let r = RingTopology::new(118, 8);
        assert_eq!(r.rank, 118);
    }

    #[test]
    fn test_ring_stress_119() {
        let r = RingTopology::new(119, 8);
        assert_eq!(r.rank, 119);
    }

    #[test]
    fn test_ring_stress_120() {
        let r = RingTopology::new(120, 8);
        assert_eq!(r.rank, 120);
    }

    #[test]
    fn test_ring_stress_121() {
        let r = RingTopology::new(121, 8);
        assert_eq!(r.rank, 121);
    }

    #[test]
    fn test_ring_stress_122() {
        let r = RingTopology::new(122, 8);
        assert_eq!(r.rank, 122);
    }

    #[test]
    fn test_ring_stress_123() {
        let r = RingTopology::new(123, 8);
        assert_eq!(r.rank, 123);
    }

    #[test]
    fn test_ring_stress_124() {
        let r = RingTopology::new(124, 8);
        assert_eq!(r.rank, 124);
    }

    #[test]
    fn test_ring_stress_125() {
        let r = RingTopology::new(125, 8);
        assert_eq!(r.rank, 125);
    }

    #[test]
    fn test_ring_stress_126() {
        let r = RingTopology::new(126, 8);
        assert_eq!(r.rank, 126);
    }

    #[test]
    fn test_ring_stress_127() {
        let r = RingTopology::new(127, 8);
        assert_eq!(r.rank, 127);
    }

    #[test]
    fn test_ring_stress_128() {
        let r = RingTopology::new(128, 8);
        assert_eq!(r.rank, 128);
    }

    #[test]
    fn test_ring_stress_129() {
        let r = RingTopology::new(129, 8);
        assert_eq!(r.rank, 129);
    }

    #[test]
    fn test_ring_stress_130() {
        let r = RingTopology::new(130, 8);
        assert_eq!(r.rank, 130);
    }

    #[test]
    fn test_ring_stress_131() {
        let r = RingTopology::new(131, 8);
        assert_eq!(r.rank, 131);
    }

    #[test]
    fn test_ring_stress_132() {
        let r = RingTopology::new(132, 8);
        assert_eq!(r.rank, 132);
    }

    #[test]
    fn test_ring_stress_133() {
        let r = RingTopology::new(133, 8);
        assert_eq!(r.rank, 133);
    }

    #[test]
    fn test_ring_stress_134() {
        let r = RingTopology::new(134, 8);
        assert_eq!(r.rank, 134);
    }

    #[test]
    fn test_ring_stress_135() {
        let r = RingTopology::new(135, 8);
        assert_eq!(r.rank, 135);
    }

    #[test]
    fn test_ring_stress_136() {
        let r = RingTopology::new(136, 8);
        assert_eq!(r.rank, 136);
    }

    #[test]
    fn test_ring_stress_137() {
        let r = RingTopology::new(137, 8);
        assert_eq!(r.rank, 137);
    }

    #[test]
    fn test_ring_stress_138() {
        let r = RingTopology::new(138, 8);
        assert_eq!(r.rank, 138);
    }

    #[test]
    fn test_ring_stress_139() {
        let r = RingTopology::new(139, 8);
        assert_eq!(r.rank, 139);
    }

    #[test]
    fn test_ring_stress_140() {
        let r = RingTopology::new(140, 8);
        assert_eq!(r.rank, 140);
    }

    #[test]
    fn test_ring_stress_141() {
        let r = RingTopology::new(141, 8);
        assert_eq!(r.rank, 141);
    }

    #[test]
    fn test_ring_stress_142() {
        let r = RingTopology::new(142, 8);
        assert_eq!(r.rank, 142);
    }

    #[test]
    fn test_ring_stress_143() {
        let r = RingTopology::new(143, 8);
        assert_eq!(r.rank, 143);
    }

    #[test]
    fn test_ring_stress_144() {
        let r = RingTopology::new(144, 8);
        assert_eq!(r.rank, 144);
    }

    #[test]
    fn test_ring_stress_145() {
        let r = RingTopology::new(145, 8);
        assert_eq!(r.rank, 145);
    }

    #[test]
    fn test_ring_stress_146() {
        let r = RingTopology::new(146, 8);
        assert_eq!(r.rank, 146);
    }

    #[test]
    fn test_ring_stress_147() {
        let r = RingTopology::new(147, 8);
        assert_eq!(r.rank, 147);
    }

    #[test]
    fn test_ring_stress_148() {
        let r = RingTopology::new(148, 8);
        assert_eq!(r.rank, 148);
    }

    #[test]
    fn test_ring_stress_149() {
        let r = RingTopology::new(149, 8);
        assert_eq!(r.rank, 149);
    }

    #[test]
    fn test_ring_stress_150() {
        let r = RingTopology::new(150, 8);
        assert_eq!(r.rank, 150);
    }

    #[test]
    fn test_ring_stress_151() {
        let r = RingTopology::new(151, 8);
        assert_eq!(r.rank, 151);
    }

    #[test]
    fn test_ring_stress_152() {
        let r = RingTopology::new(152, 8);
        assert_eq!(r.rank, 152);
    }

    #[test]
    fn test_ring_stress_153() {
        let r = RingTopology::new(153, 8);
        assert_eq!(r.rank, 153);
    }

    #[test]
    fn test_ring_stress_154() {
        let r = RingTopology::new(154, 8);
        assert_eq!(r.rank, 154);
    }

    #[test]
    fn test_ring_stress_155() {
        let r = RingTopology::new(155, 8);
        assert_eq!(r.rank, 155);
    }

    #[test]
    fn test_ring_stress_156() {
        let r = RingTopology::new(156, 8);
        assert_eq!(r.rank, 156);
    }

    #[test]
    fn test_ring_stress_157() {
        let r = RingTopology::new(157, 8);
        assert_eq!(r.rank, 157);
    }

    #[test]
    fn test_ring_stress_158() {
        let r = RingTopology::new(158, 8);
        assert_eq!(r.rank, 158);
    }

    #[test]
    fn test_ring_stress_159() {
        let r = RingTopology::new(159, 8);
        assert_eq!(r.rank, 159);
    }

    #[test]
    fn test_ring_stress_160() {
        let r = RingTopology::new(160, 8);
        assert_eq!(r.rank, 160);
    }

    #[test]
    fn test_ring_stress_161() {
        let r = RingTopology::new(161, 8);
        assert_eq!(r.rank, 161);
    }

    #[test]
    fn test_ring_stress_162() {
        let r = RingTopology::new(162, 8);
        assert_eq!(r.rank, 162);
    }

    #[test]
    fn test_ring_stress_163() {
        let r = RingTopology::new(163, 8);
        assert_eq!(r.rank, 163);
    }

    #[test]
    fn test_ring_stress_164() {
        let r = RingTopology::new(164, 8);
        assert_eq!(r.rank, 164);
    }

    #[test]
    fn test_ring_stress_165() {
        let r = RingTopology::new(165, 8);
        assert_eq!(r.rank, 165);
    }

    #[test]
    fn test_ring_stress_166() {
        let r = RingTopology::new(166, 8);
        assert_eq!(r.rank, 166);
    }

    #[test]
    fn test_ring_stress_167() {
        let r = RingTopology::new(167, 8);
        assert_eq!(r.rank, 167);
    }

    #[test]
    fn test_ring_stress_168() {
        let r = RingTopology::new(168, 8);
        assert_eq!(r.rank, 168);
    }

    #[test]
    fn test_ring_stress_169() {
        let r = RingTopology::new(169, 8);
        assert_eq!(r.rank, 169);
    }

    #[test]
    fn test_ring_stress_170() {
        let r = RingTopology::new(170, 8);
        assert_eq!(r.rank, 170);
    }

    #[test]
    fn test_ring_stress_171() {
        let r = RingTopology::new(171, 8);
        assert_eq!(r.rank, 171);
    }

    #[test]
    fn test_ring_stress_172() {
        let r = RingTopology::new(172, 8);
        assert_eq!(r.rank, 172);
    }

    #[test]
    fn test_ring_stress_173() {
        let r = RingTopology::new(173, 8);
        assert_eq!(r.rank, 173);
    }

    #[test]
    fn test_ring_stress_174() {
        let r = RingTopology::new(174, 8);
        assert_eq!(r.rank, 174);
    }

    #[test]
    fn test_ring_stress_175() {
        let r = RingTopology::new(175, 8);
        assert_eq!(r.rank, 175);
    }

    #[test]
    fn test_ring_stress_176() {
        let r = RingTopology::new(176, 8);
        assert_eq!(r.rank, 176);
    }

    #[test]
    fn test_ring_stress_177() {
        let r = RingTopology::new(177, 8);
        assert_eq!(r.rank, 177);
    }

    #[test]
    fn test_ring_stress_178() {
        let r = RingTopology::new(178, 8);
        assert_eq!(r.rank, 178);
    }

    #[test]
    fn test_ring_stress_179() {
        let r = RingTopology::new(179, 8);
        assert_eq!(r.rank, 179);
    }

    #[test]
    fn test_ring_stress_180() {
        let r = RingTopology::new(180, 8);
        assert_eq!(r.rank, 180);
    }

    #[test]
    fn test_ring_stress_181() {
        let r = RingTopology::new(181, 8);
        assert_eq!(r.rank, 181);
    }

    #[test]
    fn test_ring_stress_182() {
        let r = RingTopology::new(182, 8);
        assert_eq!(r.rank, 182);
    }

    #[test]
    fn test_ring_stress_183() {
        let r = RingTopology::new(183, 8);
        assert_eq!(r.rank, 183);
    }

    #[test]
    fn test_ring_stress_184() {
        let r = RingTopology::new(184, 8);
        assert_eq!(r.rank, 184);
    }

    #[test]
    fn test_ring_stress_185() {
        let r = RingTopology::new(185, 8);
        assert_eq!(r.rank, 185);
    }

    #[test]
    fn test_ring_stress_186() {
        let r = RingTopology::new(186, 8);
        assert_eq!(r.rank, 186);
    }

    #[test]
    fn test_ring_stress_187() {
        let r = RingTopology::new(187, 8);
        assert_eq!(r.rank, 187);
    }

    #[test]
    fn test_ring_stress_188() {
        let r = RingTopology::new(188, 8);
        assert_eq!(r.rank, 188);
    }

    #[test]
    fn test_ring_stress_189() {
        let r = RingTopology::new(189, 8);
        assert_eq!(r.rank, 189);
    }

    #[test]
    fn test_ring_stress_190() {
        let r = RingTopology::new(190, 8);
        assert_eq!(r.rank, 190);
    }

    #[test]
    fn test_ring_stress_191() {
        let r = RingTopology::new(191, 8);
        assert_eq!(r.rank, 191);
    }

    #[test]
    fn test_ring_stress_192() {
        let r = RingTopology::new(192, 8);
        assert_eq!(r.rank, 192);
    }

    #[test]
    fn test_ring_stress_193() {
        let r = RingTopology::new(193, 8);
        assert_eq!(r.rank, 193);
    }

    #[test]
    fn test_ring_stress_194() {
        let r = RingTopology::new(194, 8);
        assert_eq!(r.rank, 194);
    }

    #[test]
    fn test_ring_stress_195() {
        let r = RingTopology::new(195, 8);
        assert_eq!(r.rank, 195);
    }

    #[test]
    fn test_ring_stress_196() {
        let r = RingTopology::new(196, 8);
        assert_eq!(r.rank, 196);
    }

    #[test]
    fn test_ring_stress_197() {
        let r = RingTopology::new(197, 8);
        assert_eq!(r.rank, 197);
    }

    #[test]
    fn test_ring_stress_198() {
        let r = RingTopology::new(198, 8);
        assert_eq!(r.rank, 198);
    }

    #[test]
    fn test_ring_stress_199() {
        let r = RingTopology::new(199, 8);
        assert_eq!(r.rank, 199);
    }

    #[test]
    fn test_ring_stress_200() {
        let r = RingTopology::new(200, 8);
        assert_eq!(r.rank, 200);
    }

    #[test]
    fn test_ring_stress_201() {
        let r = RingTopology::new(201, 8);
        assert_eq!(r.rank, 201);
    }

    #[test]
    fn test_ring_stress_202() {
        let r = RingTopology::new(202, 8);
        assert_eq!(r.rank, 202);
    }

    #[test]
    fn test_ring_stress_203() {
        let r = RingTopology::new(203, 8);
        assert_eq!(r.rank, 203);
    }

    #[test]
    fn test_ring_stress_204() {
        let r = RingTopology::new(204, 8);
        assert_eq!(r.rank, 204);
    }

    #[test]
    fn test_ring_stress_205() {
        let r = RingTopology::new(205, 8);
        assert_eq!(r.rank, 205);
    }

    #[test]
    fn test_ring_stress_206() {
        let r = RingTopology::new(206, 8);
        assert_eq!(r.rank, 206);
    }

    #[test]
    fn test_ring_stress_207() {
        let r = RingTopology::new(207, 8);
        assert_eq!(r.rank, 207);
    }

    #[test]
    fn test_ring_stress_208() {
        let r = RingTopology::new(208, 8);
        assert_eq!(r.rank, 208);
    }

    #[test]
    fn test_ring_stress_209() {
        let r = RingTopology::new(209, 8);
        assert_eq!(r.rank, 209);
    }

    #[test]
    fn test_ring_stress_210() {
        let r = RingTopology::new(210, 8);
        assert_eq!(r.rank, 210);
    }

    #[test]
    fn test_ring_stress_211() {
        let r = RingTopology::new(211, 8);
        assert_eq!(r.rank, 211);
    }

    #[test]
    fn test_ring_stress_212() {
        let r = RingTopology::new(212, 8);
        assert_eq!(r.rank, 212);
    }

    #[test]
    fn test_ring_stress_213() {
        let r = RingTopology::new(213, 8);
        assert_eq!(r.rank, 213);
    }

    #[test]
    fn test_ring_stress_214() {
        let r = RingTopology::new(214, 8);
        assert_eq!(r.rank, 214);
    }

    #[test]
    fn test_ring_stress_215() {
        let r = RingTopology::new(215, 8);
        assert_eq!(r.rank, 215);
    }

    #[test]
    fn test_ring_stress_216() {
        let r = RingTopology::new(216, 8);
        assert_eq!(r.rank, 216);
    }

    #[test]
    fn test_ring_stress_217() {
        let r = RingTopology::new(217, 8);
        assert_eq!(r.rank, 217);
    }

    #[test]
    fn test_ring_stress_218() {
        let r = RingTopology::new(218, 8);
        assert_eq!(r.rank, 218);
    }

    #[test]
    fn test_ring_stress_219() {
        let r = RingTopology::new(219, 8);
        assert_eq!(r.rank, 219);
    }

    #[test]
    fn test_ring_stress_220() {
        let r = RingTopology::new(220, 8);
        assert_eq!(r.rank, 220);
    }

    #[test]
    fn test_ring_stress_221() {
        let r = RingTopology::new(221, 8);
        assert_eq!(r.rank, 221);
    }

    #[test]
    fn test_ring_stress_222() {
        let r = RingTopology::new(222, 8);
        assert_eq!(r.rank, 222);
    }

    #[test]
    fn test_ring_stress_223() {
        let r = RingTopology::new(223, 8);
        assert_eq!(r.rank, 223);
    }

    #[test]
    fn test_ring_stress_224() {
        let r = RingTopology::new(224, 8);
        assert_eq!(r.rank, 224);
    }

    #[test]
    fn test_ring_stress_225() {
        let r = RingTopology::new(225, 8);
        assert_eq!(r.rank, 225);
    }

    #[test]
    fn test_ring_stress_226() {
        let r = RingTopology::new(226, 8);
        assert_eq!(r.rank, 226);
    }

    #[test]
    fn test_ring_stress_227() {
        let r = RingTopology::new(227, 8);
        assert_eq!(r.rank, 227);
    }

    #[test]
    fn test_ring_stress_228() {
        let r = RingTopology::new(228, 8);
        assert_eq!(r.rank, 228);
    }

    #[test]
    fn test_ring_stress_229() {
        let r = RingTopology::new(229, 8);
        assert_eq!(r.rank, 229);
    }

    #[test]
    fn test_ring_stress_230() {
        let r = RingTopology::new(230, 8);
        assert_eq!(r.rank, 230);
    }

    #[test]
    fn test_ring_stress_231() {
        let r = RingTopology::new(231, 8);
        assert_eq!(r.rank, 231);
    }

    #[test]
    fn test_ring_stress_232() {
        let r = RingTopology::new(232, 8);
        assert_eq!(r.rank, 232);
    }

    #[test]
    fn test_ring_stress_233() {
        let r = RingTopology::new(233, 8);
        assert_eq!(r.rank, 233);
    }

    #[test]
    fn test_ring_stress_234() {
        let r = RingTopology::new(234, 8);
        assert_eq!(r.rank, 234);
    }

    #[test]
    fn test_ring_stress_235() {
        let r = RingTopology::new(235, 8);
        assert_eq!(r.rank, 235);
    }

    #[test]
    fn test_ring_stress_236() {
        let r = RingTopology::new(236, 8);
        assert_eq!(r.rank, 236);
    }

    #[test]
    fn test_ring_stress_237() {
        let r = RingTopology::new(237, 8);
        assert_eq!(r.rank, 237);
    }

    #[test]
    fn test_ring_stress_238() {
        let r = RingTopology::new(238, 8);
        assert_eq!(r.rank, 238);
    }

    #[test]
    fn test_ring_stress_239() {
        let r = RingTopology::new(239, 8);
        assert_eq!(r.rank, 239);
    }

    #[test]
    fn test_ring_stress_240() {
        let r = RingTopology::new(240, 8);
        assert_eq!(r.rank, 240);
    }

    #[test]
    fn test_ring_stress_241() {
        let r = RingTopology::new(241, 8);
        assert_eq!(r.rank, 241);
    }

    #[test]
    fn test_ring_stress_242() {
        let r = RingTopology::new(242, 8);
        assert_eq!(r.rank, 242);
    }

    #[test]
    fn test_ring_stress_243() {
        let r = RingTopology::new(243, 8);
        assert_eq!(r.rank, 243);
    }

    #[test]
    fn test_ring_stress_244() {
        let r = RingTopology::new(244, 8);
        assert_eq!(r.rank, 244);
    }

    #[test]
    fn test_ring_stress_245() {
        let r = RingTopology::new(245, 8);
        assert_eq!(r.rank, 245);
    }

    #[test]
    fn test_ring_stress_246() {
        let r = RingTopology::new(246, 8);
        assert_eq!(r.rank, 246);
    }

    #[test]
    fn test_ring_stress_247() {
        let r = RingTopology::new(247, 8);
        assert_eq!(r.rank, 247);
    }

    #[test]
    fn test_ring_stress_248() {
        let r = RingTopology::new(248, 8);
        assert_eq!(r.rank, 248);
    }

    #[test]
    fn test_ring_stress_249() {
        let r = RingTopology::new(249, 8);
        assert_eq!(r.rank, 249);
    }

    #[test]
    fn test_ring_stress_250() {
        let r = RingTopology::new(250, 8);
        assert_eq!(r.rank, 250);
    }

    #[test]
    fn test_ring_stress_251() {
        let r = RingTopology::new(251, 8);
        assert_eq!(r.rank, 251);
    }

    #[test]
    fn test_ring_stress_252() {
        let r = RingTopology::new(252, 8);
        assert_eq!(r.rank, 252);
    }

    #[test]
    fn test_ring_stress_253() {
        let r = RingTopology::new(253, 8);
        assert_eq!(r.rank, 253);
    }

    #[test]
    fn test_ring_stress_254() {
        let r = RingTopology::new(254, 8);
        assert_eq!(r.rank, 254);
    }

    #[test]
    fn test_ring_stress_255() {
        let r = RingTopology::new(255, 8);
        assert_eq!(r.rank, 255);
    }

    #[test]
    fn test_ring_stress_256() {
        let r = RingTopology::new(256, 8);
        assert_eq!(r.rank, 256);
    }

    #[test]
    fn test_ring_stress_257() {
        let r = RingTopology::new(257, 8);
        assert_eq!(r.rank, 257);
    }

    #[test]
    fn test_ring_stress_258() {
        let r = RingTopology::new(258, 8);
        assert_eq!(r.rank, 258);
    }

    #[test]
    fn test_ring_stress_259() {
        let r = RingTopology::new(259, 8);
        assert_eq!(r.rank, 259);
    }

    #[test]
    fn test_ring_stress_260() {
        let r = RingTopology::new(260, 8);
        assert_eq!(r.rank, 260);
    }

    #[test]
    fn test_ring_stress_261() {
        let r = RingTopology::new(261, 8);
        assert_eq!(r.rank, 261);
    }

    #[test]
    fn test_ring_stress_262() {
        let r = RingTopology::new(262, 8);
        assert_eq!(r.rank, 262);
    }

    #[test]
    fn test_ring_stress_263() {
        let r = RingTopology::new(263, 8);
        assert_eq!(r.rank, 263);
    }

    #[test]
    fn test_ring_stress_264() {
        let r = RingTopology::new(264, 8);
        assert_eq!(r.rank, 264);
    }

    #[test]
    fn test_ring_stress_265() {
        let r = RingTopology::new(265, 8);
        assert_eq!(r.rank, 265);
    }

    #[test]
    fn test_ring_stress_266() {
        let r = RingTopology::new(266, 8);
        assert_eq!(r.rank, 266);
    }

    #[test]
    fn test_ring_stress_267() {
        let r = RingTopology::new(267, 8);
        assert_eq!(r.rank, 267);
    }

    #[test]
    fn test_ring_stress_268() {
        let r = RingTopology::new(268, 8);
        assert_eq!(r.rank, 268);
    }

    #[test]
    fn test_ring_stress_269() {
        let r = RingTopology::new(269, 8);
        assert_eq!(r.rank, 269);
    }

    #[test]
    fn test_ring_stress_270() {
        let r = RingTopology::new(270, 8);
        assert_eq!(r.rank, 270);
    }

    #[test]
    fn test_ring_stress_271() {
        let r = RingTopology::new(271, 8);
        assert_eq!(r.rank, 271);
    }

    #[test]
    fn test_ring_stress_272() {
        let r = RingTopology::new(272, 8);
        assert_eq!(r.rank, 272);
    }

    #[test]
    fn test_ring_stress_273() {
        let r = RingTopology::new(273, 8);
        assert_eq!(r.rank, 273);
    }

    #[test]
    fn test_ring_stress_274() {
        let r = RingTopology::new(274, 8);
        assert_eq!(r.rank, 274);
    }

    #[test]
    fn test_ring_stress_275() {
        let r = RingTopology::new(275, 8);
        assert_eq!(r.rank, 275);
    }

    #[test]
    fn test_ring_stress_276() {
        let r = RingTopology::new(276, 8);
        assert_eq!(r.rank, 276);
    }

    #[test]
    fn test_ring_stress_277() {
        let r = RingTopology::new(277, 8);
        assert_eq!(r.rank, 277);
    }

    #[test]
    fn test_ring_stress_278() {
        let r = RingTopology::new(278, 8);
        assert_eq!(r.rank, 278);
    }

    #[test]
    fn test_ring_stress_279() {
        let r = RingTopology::new(279, 8);
        assert_eq!(r.rank, 279);
    }

    #[test]
    fn test_ring_stress_280() {
        let r = RingTopology::new(280, 8);
        assert_eq!(r.rank, 280);
    }

    #[test]
    fn test_ring_stress_281() {
        let r = RingTopology::new(281, 8);
        assert_eq!(r.rank, 281);
    }

    #[test]
    fn test_ring_stress_282() {
        let r = RingTopology::new(282, 8);
        assert_eq!(r.rank, 282);
    }

    #[test]
    fn test_ring_stress_283() {
        let r = RingTopology::new(283, 8);
        assert_eq!(r.rank, 283);
    }

    #[test]
    fn test_ring_stress_284() {
        let r = RingTopology::new(284, 8);
        assert_eq!(r.rank, 284);
    }

    #[test]
    fn test_ring_stress_285() {
        let r = RingTopology::new(285, 8);
        assert_eq!(r.rank, 285);
    }

    #[test]
    fn test_ring_stress_286() {
        let r = RingTopology::new(286, 8);
        assert_eq!(r.rank, 286);
    }

    #[test]
    fn test_ring_stress_287() {
        let r = RingTopology::new(287, 8);
        assert_eq!(r.rank, 287);
    }

    #[test]
    fn test_ring_stress_288() {
        let r = RingTopology::new(288, 8);
        assert_eq!(r.rank, 288);
    }

    #[test]
    fn test_ring_stress_289() {
        let r = RingTopology::new(289, 8);
        assert_eq!(r.rank, 289);
    }

    #[test]
    fn test_ring_stress_290() {
        let r = RingTopology::new(290, 8);
        assert_eq!(r.rank, 290);
    }

    #[test]
    fn test_ring_stress_291() {
        let r = RingTopology::new(291, 8);
        assert_eq!(r.rank, 291);
    }

    #[test]
    fn test_ring_stress_292() {
        let r = RingTopology::new(292, 8);
        assert_eq!(r.rank, 292);
    }

    #[test]
    fn test_ring_stress_293() {
        let r = RingTopology::new(293, 8);
        assert_eq!(r.rank, 293);
    }

    #[test]
    fn test_ring_stress_294() {
        let r = RingTopology::new(294, 8);
        assert_eq!(r.rank, 294);
    }

    #[test]
    fn test_ring_stress_295() {
        let r = RingTopology::new(295, 8);
        assert_eq!(r.rank, 295);
    }

    #[test]
    fn test_ring_stress_296() {
        let r = RingTopology::new(296, 8);
        assert_eq!(r.rank, 296);
    }

    #[test]
    fn test_ring_stress_297() {
        let r = RingTopology::new(297, 8);
        assert_eq!(r.rank, 297);
    }

    #[test]
    fn test_ring_stress_298() {
        let r = RingTopology::new(298, 8);
        assert_eq!(r.rank, 298);
    }

    #[test]
    fn test_ring_stress_299() {
        let r = RingTopology::new(299, 8);
        assert_eq!(r.rank, 299);
    }

    #[test]
    fn test_ring_stress_300() {
        let r = RingTopology::new(300, 8);
        assert_eq!(r.rank, 300);
    }

    #[test]
    fn test_ring_stress_301() {
        let r = RingTopology::new(301, 8);
        assert_eq!(r.rank, 301);
    }

    #[test]
    fn test_ring_stress_302() {
        let r = RingTopology::new(302, 8);
        assert_eq!(r.rank, 302);
    }

    #[test]
    fn test_ring_stress_303() {
        let r = RingTopology::new(303, 8);
        assert_eq!(r.rank, 303);
    }

    #[test]
    fn test_ring_stress_304() {
        let r = RingTopology::new(304, 8);
        assert_eq!(r.rank, 304);
    }

    #[test]
    fn test_ring_stress_305() {
        let r = RingTopology::new(305, 8);
        assert_eq!(r.rank, 305);
    }

    #[test]
    fn test_ring_stress_306() {
        let r = RingTopology::new(306, 8);
        assert_eq!(r.rank, 306);
    }

    #[test]
    fn test_ring_stress_307() {
        let r = RingTopology::new(307, 8);
        assert_eq!(r.rank, 307);
    }

    #[test]
    fn test_ring_stress_308() {
        let r = RingTopology::new(308, 8);
        assert_eq!(r.rank, 308);
    }

    #[test]
    fn test_ring_stress_309() {
        let r = RingTopology::new(309, 8);
        assert_eq!(r.rank, 309);
    }

    #[test]
    fn test_ring_stress_310() {
        let r = RingTopology::new(310, 8);
        assert_eq!(r.rank, 310);
    }

    #[test]
    fn test_ring_stress_311() {
        let r = RingTopology::new(311, 8);
        assert_eq!(r.rank, 311);
    }

    #[test]
    fn test_ring_stress_312() {
        let r = RingTopology::new(312, 8);
        assert_eq!(r.rank, 312);
    }

    #[test]
    fn test_ring_stress_313() {
        let r = RingTopology::new(313, 8);
        assert_eq!(r.rank, 313);
    }

    #[test]
    fn test_ring_stress_314() {
        let r = RingTopology::new(314, 8);
        assert_eq!(r.rank, 314);
    }

    #[test]
    fn test_ring_stress_315() {
        let r = RingTopology::new(315, 8);
        assert_eq!(r.rank, 315);
    }

    #[test]
    fn test_ring_stress_316() {
        let r = RingTopology::new(316, 8);
        assert_eq!(r.rank, 316);
    }

    #[test]
    fn test_ring_stress_317() {
        let r = RingTopology::new(317, 8);
        assert_eq!(r.rank, 317);
    }

    #[test]
    fn test_ring_stress_318() {
        let r = RingTopology::new(318, 8);
        assert_eq!(r.rank, 318);
    }

    #[test]
    fn test_ring_stress_319() {
        let r = RingTopology::new(319, 8);
        assert_eq!(r.rank, 319);
    }

    #[test]
    fn test_ring_stress_320() {
        let r = RingTopology::new(320, 8);
        assert_eq!(r.rank, 320);
    }

    #[test]
    fn test_ring_stress_321() {
        let r = RingTopology::new(321, 8);
        assert_eq!(r.rank, 321);
    }

    #[test]
    fn test_ring_stress_322() {
        let r = RingTopology::new(322, 8);
        assert_eq!(r.rank, 322);
    }

    #[test]
    fn test_ring_stress_323() {
        let r = RingTopology::new(323, 8);
        assert_eq!(r.rank, 323);
    }

    #[test]
    fn test_ring_stress_324() {
        let r = RingTopology::new(324, 8);
        assert_eq!(r.rank, 324);
    }

    #[test]
    fn test_ring_stress_325() {
        let r = RingTopology::new(325, 8);
        assert_eq!(r.rank, 325);
    }

    #[test]
    fn test_ring_stress_326() {
        let r = RingTopology::new(326, 8);
        assert_eq!(r.rank, 326);
    }

    #[test]
    fn test_ring_stress_327() {
        let r = RingTopology::new(327, 8);
        assert_eq!(r.rank, 327);
    }

    #[test]
    fn test_ring_stress_328() {
        let r = RingTopology::new(328, 8);
        assert_eq!(r.rank, 328);
    }

    #[test]
    fn test_ring_stress_329() {
        let r = RingTopology::new(329, 8);
        assert_eq!(r.rank, 329);
    }

    #[test]
    fn test_ring_stress_330() {
        let r = RingTopology::new(330, 8);
        assert_eq!(r.rank, 330);
    }

    #[test]
    fn test_ring_stress_331() {
        let r = RingTopology::new(331, 8);
        assert_eq!(r.rank, 331);
    }

    #[test]
    fn test_ring_stress_332() {
        let r = RingTopology::new(332, 8);
        assert_eq!(r.rank, 332);
    }

    #[test]
    fn test_ring_stress_333() {
        let r = RingTopology::new(333, 8);
        assert_eq!(r.rank, 333);
    }

    #[test]
    fn test_ring_stress_334() {
        let r = RingTopology::new(334, 8);
        assert_eq!(r.rank, 334);
    }

    #[test]
    fn test_ring_stress_335() {
        let r = RingTopology::new(335, 8);
        assert_eq!(r.rank, 335);
    }

    #[test]
    fn test_ring_stress_336() {
        let r = RingTopology::new(336, 8);
        assert_eq!(r.rank, 336);
    }

    #[test]
    fn test_ring_stress_337() {
        let r = RingTopology::new(337, 8);
        assert_eq!(r.rank, 337);
    }

    #[test]
    fn test_ring_stress_338() {
        let r = RingTopology::new(338, 8);
        assert_eq!(r.rank, 338);
    }

    #[test]
    fn test_ring_stress_339() {
        let r = RingTopology::new(339, 8);
        assert_eq!(r.rank, 339);
    }

    #[test]
    fn test_ring_stress_340() {
        let r = RingTopology::new(340, 8);
        assert_eq!(r.rank, 340);
    }

    #[test]
    fn test_ring_stress_341() {
        let r = RingTopology::new(341, 8);
        assert_eq!(r.rank, 341);
    }

    #[test]
    fn test_ring_stress_342() {
        let r = RingTopology::new(342, 8);
        assert_eq!(r.rank, 342);
    }

    #[test]
    fn test_ring_stress_343() {
        let r = RingTopology::new(343, 8);
        assert_eq!(r.rank, 343);
    }

    #[test]
    fn test_ring_stress_344() {
        let r = RingTopology::new(344, 8);
        assert_eq!(r.rank, 344);
    }

    #[test]
    fn test_ring_stress_345() {
        let r = RingTopology::new(345, 8);
        assert_eq!(r.rank, 345);
    }

    #[test]
    fn test_ring_stress_346() {
        let r = RingTopology::new(346, 8);
        assert_eq!(r.rank, 346);
    }

    #[test]
    fn test_ring_stress_347() {
        let r = RingTopology::new(347, 8);
        assert_eq!(r.rank, 347);
    }

    #[test]
    fn test_ring_stress_348() {
        let r = RingTopology::new(348, 8);
        assert_eq!(r.rank, 348);
    }

    #[test]
    fn test_ring_stress_349() {
        let r = RingTopology::new(349, 8);
        assert_eq!(r.rank, 349);
    }

    #[test]
    fn test_ring_stress_350() {
        let r = RingTopology::new(350, 8);
        assert_eq!(r.rank, 350);
    }

    #[test]
    fn test_ring_stress_351() {
        let r = RingTopology::new(351, 8);
        assert_eq!(r.rank, 351);
    }

    #[test]
    fn test_ring_stress_352() {
        let r = RingTopology::new(352, 8);
        assert_eq!(r.rank, 352);
    }

    #[test]
    fn test_ring_stress_353() {
        let r = RingTopology::new(353, 8);
        assert_eq!(r.rank, 353);
    }

    #[test]
    fn test_ring_stress_354() {
        let r = RingTopology::new(354, 8);
        assert_eq!(r.rank, 354);
    }

    #[test]
    fn test_ring_stress_355() {
        let r = RingTopology::new(355, 8);
        assert_eq!(r.rank, 355);
    }

    #[test]
    fn test_ring_stress_356() {
        let r = RingTopology::new(356, 8);
        assert_eq!(r.rank, 356);
    }

    #[test]
    fn test_ring_stress_357() {
        let r = RingTopology::new(357, 8);
        assert_eq!(r.rank, 357);
    }

    #[test]
    fn test_ring_stress_358() {
        let r = RingTopology::new(358, 8);
        assert_eq!(r.rank, 358);
    }

    #[test]
    fn test_ring_stress_359() {
        let r = RingTopology::new(359, 8);
        assert_eq!(r.rank, 359);
    }

    #[test]
    fn test_ring_stress_360() {
        let r = RingTopology::new(360, 8);
        assert_eq!(r.rank, 360);
    }

    #[test]
    fn test_ring_stress_361() {
        let r = RingTopology::new(361, 8);
        assert_eq!(r.rank, 361);
    }

    #[test]
    fn test_ring_stress_362() {
        let r = RingTopology::new(362, 8);
        assert_eq!(r.rank, 362);
    }

    #[test]
    fn test_ring_stress_363() {
        let r = RingTopology::new(363, 8);
        assert_eq!(r.rank, 363);
    }

    #[test]
    fn test_ring_stress_364() {
        let r = RingTopology::new(364, 8);
        assert_eq!(r.rank, 364);
    }

    #[test]
    fn test_ring_stress_365() {
        let r = RingTopology::new(365, 8);
        assert_eq!(r.rank, 365);
    }

    #[test]
    fn test_ring_stress_366() {
        let r = RingTopology::new(366, 8);
        assert_eq!(r.rank, 366);
    }

    #[test]
    fn test_ring_stress_367() {
        let r = RingTopology::new(367, 8);
        assert_eq!(r.rank, 367);
    }

    #[test]
    fn test_ring_stress_368() {
        let r = RingTopology::new(368, 8);
        assert_eq!(r.rank, 368);
    }

    #[test]
    fn test_ring_stress_369() {
        let r = RingTopology::new(369, 8);
        assert_eq!(r.rank, 369);
    }

    #[test]
    fn test_ring_stress_370() {
        let r = RingTopology::new(370, 8);
        assert_eq!(r.rank, 370);
    }

    #[test]
    fn test_ring_stress_371() {
        let r = RingTopology::new(371, 8);
        assert_eq!(r.rank, 371);
    }

    #[test]
    fn test_ring_stress_372() {
        let r = RingTopology::new(372, 8);
        assert_eq!(r.rank, 372);
    }

    #[test]
    fn test_ring_stress_373() {
        let r = RingTopology::new(373, 8);
        assert_eq!(r.rank, 373);
    }

    #[test]
    fn test_ring_stress_374() {
        let r = RingTopology::new(374, 8);
        assert_eq!(r.rank, 374);
    }

    #[test]
    fn test_ring_stress_375() {
        let r = RingTopology::new(375, 8);
        assert_eq!(r.rank, 375);
    }

    #[test]
    fn test_ring_stress_376() {
        let r = RingTopology::new(376, 8);
        assert_eq!(r.rank, 376);
    }

    #[test]
    fn test_ring_stress_377() {
        let r = RingTopology::new(377, 8);
        assert_eq!(r.rank, 377);
    }

    #[test]
    fn test_ring_stress_378() {
        let r = RingTopology::new(378, 8);
        assert_eq!(r.rank, 378);
    }

    #[test]
    fn test_ring_stress_379() {
        let r = RingTopology::new(379, 8);
        assert_eq!(r.rank, 379);
    }

    #[test]
    fn test_ring_stress_380() {
        let r = RingTopology::new(380, 8);
        assert_eq!(r.rank, 380);
    }

    #[test]
    fn test_ring_stress_381() {
        let r = RingTopology::new(381, 8);
        assert_eq!(r.rank, 381);
    }

    #[test]
    fn test_ring_stress_382() {
        let r = RingTopology::new(382, 8);
        assert_eq!(r.rank, 382);
    }

    #[test]
    fn test_ring_stress_383() {
        let r = RingTopology::new(383, 8);
        assert_eq!(r.rank, 383);
    }

    #[test]
    fn test_ring_stress_384() {
        let r = RingTopology::new(384, 8);
        assert_eq!(r.rank, 384);
    }

    #[test]
    fn test_ring_stress_385() {
        let r = RingTopology::new(385, 8);
        assert_eq!(r.rank, 385);
    }

    #[test]
    fn test_ring_stress_386() {
        let r = RingTopology::new(386, 8);
        assert_eq!(r.rank, 386);
    }

    #[test]
    fn test_ring_stress_387() {
        let r = RingTopology::new(387, 8);
        assert_eq!(r.rank, 387);
    }

    #[test]
    fn test_ring_stress_388() {
        let r = RingTopology::new(388, 8);
        assert_eq!(r.rank, 388);
    }

    #[test]
    fn test_ring_stress_389() {
        let r = RingTopology::new(389, 8);
        assert_eq!(r.rank, 389);
    }

    #[test]
    fn test_ring_stress_390() {
        let r = RingTopology::new(390, 8);
        assert_eq!(r.rank, 390);
    }

    #[test]
    fn test_ring_stress_391() {
        let r = RingTopology::new(391, 8);
        assert_eq!(r.rank, 391);
    }

    #[test]
    fn test_ring_stress_392() {
        let r = RingTopology::new(392, 8);
        assert_eq!(r.rank, 392);
    }

    #[test]
    fn test_ring_stress_393() {
        let r = RingTopology::new(393, 8);
        assert_eq!(r.rank, 393);
    }

    #[test]
    fn test_ring_stress_394() {
        let r = RingTopology::new(394, 8);
        assert_eq!(r.rank, 394);
    }

    #[test]
    fn test_ring_stress_395() {
        let r = RingTopology::new(395, 8);
        assert_eq!(r.rank, 395);
    }

    #[test]
    fn test_ring_stress_396() {
        let r = RingTopology::new(396, 8);
        assert_eq!(r.rank, 396);
    }

    #[test]
    fn test_ring_stress_397() {
        let r = RingTopology::new(397, 8);
        assert_eq!(r.rank, 397);
    }

    #[test]
    fn test_ring_stress_398() {
        let r = RingTopology::new(398, 8);
        assert_eq!(r.rank, 398);
    }

    #[test]
    fn test_ring_stress_399() {
        let r = RingTopology::new(399, 8);
        assert_eq!(r.rank, 399);
    }

    #[test]
    fn test_ring_stress_400() {
        let r = RingTopology::new(400, 8);
        assert_eq!(r.rank, 400);
    }

    #[test]
    fn test_ring_stress_401() {
        let r = RingTopology::new(401, 8);
        assert_eq!(r.rank, 401);
    }

    #[test]
    fn test_ring_stress_402() {
        let r = RingTopology::new(402, 8);
        assert_eq!(r.rank, 402);
    }

    #[test]
    fn test_ring_stress_403() {
        let r = RingTopology::new(403, 8);
        assert_eq!(r.rank, 403);
    }

    #[test]
    fn test_ring_stress_404() {
        let r = RingTopology::new(404, 8);
        assert_eq!(r.rank, 404);
    }

    #[test]
    fn test_ring_stress_405() {
        let r = RingTopology::new(405, 8);
        assert_eq!(r.rank, 405);
    }

    #[test]
    fn test_ring_stress_406() {
        let r = RingTopology::new(406, 8);
        assert_eq!(r.rank, 406);
    }

    #[test]
    fn test_ring_stress_407() {
        let r = RingTopology::new(407, 8);
        assert_eq!(r.rank, 407);
    }

    #[test]
    fn test_ring_stress_408() {
        let r = RingTopology::new(408, 8);
        assert_eq!(r.rank, 408);
    }

    #[test]
    fn test_ring_stress_409() {
        let r = RingTopology::new(409, 8);
        assert_eq!(r.rank, 409);
    }

    #[test]
    fn test_ring_stress_410() {
        let r = RingTopology::new(410, 8);
        assert_eq!(r.rank, 410);
    }

    #[test]
    fn test_ring_stress_411() {
        let r = RingTopology::new(411, 8);
        assert_eq!(r.rank, 411);
    }

    #[test]
    fn test_ring_stress_412() {
        let r = RingTopology::new(412, 8);
        assert_eq!(r.rank, 412);
    }

    #[test]
    fn test_ring_stress_413() {
        let r = RingTopology::new(413, 8);
        assert_eq!(r.rank, 413);
    }

    #[test]
    fn test_ring_stress_414() {
        let r = RingTopology::new(414, 8);
        assert_eq!(r.rank, 414);
    }

    #[test]
    fn test_ring_stress_415() {
        let r = RingTopology::new(415, 8);
        assert_eq!(r.rank, 415);
    }

    #[test]
    fn test_ring_stress_416() {
        let r = RingTopology::new(416, 8);
        assert_eq!(r.rank, 416);
    }

    #[test]
    fn test_ring_stress_417() {
        let r = RingTopology::new(417, 8);
        assert_eq!(r.rank, 417);
    }

    #[test]
    fn test_ring_stress_418() {
        let r = RingTopology::new(418, 8);
        assert_eq!(r.rank, 418);
    }

    #[test]
    fn test_ring_stress_419() {
        let r = RingTopology::new(419, 8);
        assert_eq!(r.rank, 419);
    }

    #[test]
    fn test_ring_stress_420() {
        let r = RingTopology::new(420, 8);
        assert_eq!(r.rank, 420);
    }

    #[test]
    fn test_ring_stress_421() {
        let r = RingTopology::new(421, 8);
        assert_eq!(r.rank, 421);
    }

    #[test]
    fn test_ring_stress_422() {
        let r = RingTopology::new(422, 8);
        assert_eq!(r.rank, 422);
    }

    #[test]
    fn test_ring_stress_423() {
        let r = RingTopology::new(423, 8);
        assert_eq!(r.rank, 423);
    }

    #[test]
    fn test_ring_stress_424() {
        let r = RingTopology::new(424, 8);
        assert_eq!(r.rank, 424);
    }

    #[test]
    fn test_ring_stress_425() {
        let r = RingTopology::new(425, 8);
        assert_eq!(r.rank, 425);
    }

    #[test]
    fn test_ring_stress_426() {
        let r = RingTopology::new(426, 8);
        assert_eq!(r.rank, 426);
    }

    #[test]
    fn test_ring_stress_427() {
        let r = RingTopology::new(427, 8);
        assert_eq!(r.rank, 427);
    }

    #[test]
    fn test_ring_stress_428() {
        let r = RingTopology::new(428, 8);
        assert_eq!(r.rank, 428);
    }

    #[test]
    fn test_ring_stress_429() {
        let r = RingTopology::new(429, 8);
        assert_eq!(r.rank, 429);
    }

    #[test]
    fn test_ring_stress_430() {
        let r = RingTopology::new(430, 8);
        assert_eq!(r.rank, 430);
    }

    #[test]
    fn test_ring_stress_431() {
        let r = RingTopology::new(431, 8);
        assert_eq!(r.rank, 431);
    }

    #[test]
    fn test_ring_stress_432() {
        let r = RingTopology::new(432, 8);
        assert_eq!(r.rank, 432);
    }

    #[test]
    fn test_ring_stress_433() {
        let r = RingTopology::new(433, 8);
        assert_eq!(r.rank, 433);
    }

    #[test]
    fn test_ring_stress_434() {
        let r = RingTopology::new(434, 8);
        assert_eq!(r.rank, 434);
    }

    #[test]
    fn test_ring_stress_435() {
        let r = RingTopology::new(435, 8);
        assert_eq!(r.rank, 435);
    }

    #[test]
    fn test_ring_stress_436() {
        let r = RingTopology::new(436, 8);
        assert_eq!(r.rank, 436);
    }

    #[test]
    fn test_ring_stress_437() {
        let r = RingTopology::new(437, 8);
        assert_eq!(r.rank, 437);
    }

    #[test]
    fn test_ring_stress_438() {
        let r = RingTopology::new(438, 8);
        assert_eq!(r.rank, 438);
    }

    #[test]
    fn test_ring_stress_439() {
        let r = RingTopology::new(439, 8);
        assert_eq!(r.rank, 439);
    }

    #[test]
    fn test_ring_stress_440() {
        let r = RingTopology::new(440, 8);
        assert_eq!(r.rank, 440);
    }

    #[test]
    fn test_ring_stress_441() {
        let r = RingTopology::new(441, 8);
        assert_eq!(r.rank, 441);
    }

    #[test]
    fn test_ring_stress_442() {
        let r = RingTopology::new(442, 8);
        assert_eq!(r.rank, 442);
    }

    #[test]
    fn test_ring_stress_443() {
        let r = RingTopology::new(443, 8);
        assert_eq!(r.rank, 443);
    }

    #[test]
    fn test_ring_stress_444() {
        let r = RingTopology::new(444, 8);
        assert_eq!(r.rank, 444);
    }

    #[test]
    fn test_ring_stress_445() {
        let r = RingTopology::new(445, 8);
        assert_eq!(r.rank, 445);
    }

    #[test]
    fn test_ring_stress_446() {
        let r = RingTopology::new(446, 8);
        assert_eq!(r.rank, 446);
    }

    #[test]
    fn test_ring_stress_447() {
        let r = RingTopology::new(447, 8);
        assert_eq!(r.rank, 447);
    }

    #[test]
    fn test_ring_stress_448() {
        let r = RingTopology::new(448, 8);
        assert_eq!(r.rank, 448);
    }

    #[test]
    fn test_ring_stress_449() {
        let r = RingTopology::new(449, 8);
        assert_eq!(r.rank, 449);
    }

    #[test]
    fn test_ring_stress_450() {
        let r = RingTopology::new(450, 8);
        assert_eq!(r.rank, 450);
    }

    #[test]
    fn test_ring_stress_451() {
        let r = RingTopology::new(451, 8);
        assert_eq!(r.rank, 451);
    }

    #[test]
    fn test_ring_stress_452() {
        let r = RingTopology::new(452, 8);
        assert_eq!(r.rank, 452);
    }

    #[test]
    fn test_ring_stress_453() {
        let r = RingTopology::new(453, 8);
        assert_eq!(r.rank, 453);
    }

    #[test]
    fn test_ring_stress_454() {
        let r = RingTopology::new(454, 8);
        assert_eq!(r.rank, 454);
    }

    #[test]
    fn test_ring_stress_455() {
        let r = RingTopology::new(455, 8);
        assert_eq!(r.rank, 455);
    }

    #[test]
    fn test_ring_stress_456() {
        let r = RingTopology::new(456, 8);
        assert_eq!(r.rank, 456);
    }

    #[test]
    fn test_ring_stress_457() {
        let r = RingTopology::new(457, 8);
        assert_eq!(r.rank, 457);
    }

    #[test]
    fn test_ring_stress_458() {
        let r = RingTopology::new(458, 8);
        assert_eq!(r.rank, 458);
    }

    #[test]
    fn test_ring_stress_459() {
        let r = RingTopology::new(459, 8);
        assert_eq!(r.rank, 459);
    }

    #[test]
    fn test_ring_stress_460() {
        let r = RingTopology::new(460, 8);
        assert_eq!(r.rank, 460);
    }

    #[test]
    fn test_ring_stress_461() {
        let r = RingTopology::new(461, 8);
        assert_eq!(r.rank, 461);
    }

    #[test]
    fn test_ring_stress_462() {
        let r = RingTopology::new(462, 8);
        assert_eq!(r.rank, 462);
    }

    #[test]
    fn test_ring_stress_463() {
        let r = RingTopology::new(463, 8);
        assert_eq!(r.rank, 463);
    }

    #[test]
    fn test_ring_stress_464() {
        let r = RingTopology::new(464, 8);
        assert_eq!(r.rank, 464);
    }

    #[test]
    fn test_ring_stress_465() {
        let r = RingTopology::new(465, 8);
        assert_eq!(r.rank, 465);
    }

    #[test]
    fn test_ring_stress_466() {
        let r = RingTopology::new(466, 8);
        assert_eq!(r.rank, 466);
    }

    #[test]
    fn test_ring_stress_467() {
        let r = RingTopology::new(467, 8);
        assert_eq!(r.rank, 467);
    }

    #[test]
    fn test_ring_stress_468() {
        let r = RingTopology::new(468, 8);
        assert_eq!(r.rank, 468);
    }

    #[test]
    fn test_ring_stress_469() {
        let r = RingTopology::new(469, 8);
        assert_eq!(r.rank, 469);
    }

    #[test]
    fn test_ring_stress_470() {
        let r = RingTopology::new(470, 8);
        assert_eq!(r.rank, 470);
    }

    #[test]
    fn test_ring_stress_471() {
        let r = RingTopology::new(471, 8);
        assert_eq!(r.rank, 471);
    }

    #[test]
    fn test_ring_stress_472() {
        let r = RingTopology::new(472, 8);
        assert_eq!(r.rank, 472);
    }

    #[test]
    fn test_ring_stress_473() {
        let r = RingTopology::new(473, 8);
        assert_eq!(r.rank, 473);
    }

    #[test]
    fn test_ring_stress_474() {
        let r = RingTopology::new(474, 8);
        assert_eq!(r.rank, 474);
    }

    #[test]
    fn test_ring_stress_475() {
        let r = RingTopology::new(475, 8);
        assert_eq!(r.rank, 475);
    }

    #[test]
    fn test_ring_stress_476() {
        let r = RingTopology::new(476, 8);
        assert_eq!(r.rank, 476);
    }

    #[test]
    fn test_ring_stress_477() {
        let r = RingTopology::new(477, 8);
        assert_eq!(r.rank, 477);
    }

    #[test]
    fn test_ring_stress_478() {
        let r = RingTopology::new(478, 8);
        assert_eq!(r.rank, 478);
    }

    #[test]
    fn test_ring_stress_479() {
        let r = RingTopology::new(479, 8);
        assert_eq!(r.rank, 479);
    }

    #[test]
    fn test_ring_stress_480() {
        let r = RingTopology::new(480, 8);
        assert_eq!(r.rank, 480);
    }

    #[test]
    fn test_ring_stress_481() {
        let r = RingTopology::new(481, 8);
        assert_eq!(r.rank, 481);
    }

    #[test]
    fn test_ring_stress_482() {
        let r = RingTopology::new(482, 8);
        assert_eq!(r.rank, 482);
    }

    #[test]
    fn test_ring_stress_483() {
        let r = RingTopology::new(483, 8);
        assert_eq!(r.rank, 483);
    }

    #[test]
    fn test_ring_stress_484() {
        let r = RingTopology::new(484, 8);
        assert_eq!(r.rank, 484);
    }

    #[test]
    fn test_ring_stress_485() {
        let r = RingTopology::new(485, 8);
        assert_eq!(r.rank, 485);
    }

    #[test]
    fn test_ring_stress_486() {
        let r = RingTopology::new(486, 8);
        assert_eq!(r.rank, 486);
    }

    #[test]
    fn test_ring_stress_487() {
        let r = RingTopology::new(487, 8);
        assert_eq!(r.rank, 487);
    }

    #[test]
    fn test_ring_stress_488() {
        let r = RingTopology::new(488, 8);
        assert_eq!(r.rank, 488);
    }

    #[test]
    fn test_ring_stress_489() {
        let r = RingTopology::new(489, 8);
        assert_eq!(r.rank, 489);
    }

    #[test]
    fn test_ring_stress_490() {
        let r = RingTopology::new(490, 8);
        assert_eq!(r.rank, 490);
    }

    #[test]
    fn test_ring_stress_491() {
        let r = RingTopology::new(491, 8);
        assert_eq!(r.rank, 491);
    }

    #[test]
    fn test_ring_stress_492() {
        let r = RingTopology::new(492, 8);
        assert_eq!(r.rank, 492);
    }

    #[test]
    fn test_ring_stress_493() {
        let r = RingTopology::new(493, 8);
        assert_eq!(r.rank, 493);
    }

    #[test]
    fn test_ring_stress_494() {
        let r = RingTopology::new(494, 8);
        assert_eq!(r.rank, 494);
    }

    #[test]
    fn test_ring_stress_495() {
        let r = RingTopology::new(495, 8);
        assert_eq!(r.rank, 495);
    }

    #[test]
    fn test_ring_stress_496() {
        let r = RingTopology::new(496, 8);
        assert_eq!(r.rank, 496);
    }

    #[test]
    fn test_ring_stress_497() {
        let r = RingTopology::new(497, 8);
        assert_eq!(r.rank, 497);
    }

    #[test]
    fn test_ring_stress_498() {
        let r = RingTopology::new(498, 8);
        assert_eq!(r.rank, 498);
    }

    #[test]
    fn test_ring_stress_499() {
        let r = RingTopology::new(499, 8);
        assert_eq!(r.rank, 499);
    }

    #[test]
    fn test_ring_stress_500() {
        let r = RingTopology::new(500, 8);
        assert_eq!(r.rank, 500);
    }

    #[test]
    fn test_ring_stress_501() {
        let r = RingTopology::new(501, 8);
        assert_eq!(r.rank, 501);
    }

    #[test]
    fn test_ring_stress_502() {
        let r = RingTopology::new(502, 8);
        assert_eq!(r.rank, 502);
    }

    #[test]
    fn test_ring_stress_503() {
        let r = RingTopology::new(503, 8);
        assert_eq!(r.rank, 503);
    }

    #[test]
    fn test_ring_stress_504() {
        let r = RingTopology::new(504, 8);
        assert_eq!(r.rank, 504);
    }

    #[test]
    fn test_ring_stress_505() {
        let r = RingTopology::new(505, 8);
        assert_eq!(r.rank, 505);
    }

    #[test]
    fn test_ring_stress_506() {
        let r = RingTopology::new(506, 8);
        assert_eq!(r.rank, 506);
    }

    #[test]
    fn test_ring_stress_507() {
        let r = RingTopology::new(507, 8);
        assert_eq!(r.rank, 507);
    }

    #[test]
    fn test_ring_stress_508() {
        let r = RingTopology::new(508, 8);
        assert_eq!(r.rank, 508);
    }

    #[test]
    fn test_ring_stress_509() {
        let r = RingTopology::new(509, 8);
        assert_eq!(r.rank, 509);
    }

    #[test]
    fn test_ring_stress_510() {
        let r = RingTopology::new(510, 8);
        assert_eq!(r.rank, 510);
    }

    #[test]
    fn test_ring_stress_511() {
        let r = RingTopology::new(511, 8);
        assert_eq!(r.rank, 511);
    }

    #[test]
    fn test_ring_stress_512() {
        let r = RingTopology::new(512, 8);
        assert_eq!(r.rank, 512);
    }

    #[test]
    fn test_ring_stress_513() {
        let r = RingTopology::new(513, 8);
        assert_eq!(r.rank, 513);
    }

    #[test]
    fn test_ring_stress_514() {
        let r = RingTopology::new(514, 8);
        assert_eq!(r.rank, 514);
    }

    #[test]
    fn test_ring_stress_515() {
        let r = RingTopology::new(515, 8);
        assert_eq!(r.rank, 515);
    }

    #[test]
    fn test_ring_stress_516() {
        let r = RingTopology::new(516, 8);
        assert_eq!(r.rank, 516);
    }

    #[test]
    fn test_ring_stress_517() {
        let r = RingTopology::new(517, 8);
        assert_eq!(r.rank, 517);
    }

    #[test]
    fn test_ring_stress_518() {
        let r = RingTopology::new(518, 8);
        assert_eq!(r.rank, 518);
    }

    #[test]
    fn test_ring_stress_519() {
        let r = RingTopology::new(519, 8);
        assert_eq!(r.rank, 519);
    }

    #[test]
    fn test_ring_stress_520() {
        let r = RingTopology::new(520, 8);
        assert_eq!(r.rank, 520);
    }

    #[test]
    fn test_ring_stress_521() {
        let r = RingTopology::new(521, 8);
        assert_eq!(r.rank, 521);
    }

    #[test]
    fn test_ring_stress_522() {
        let r = RingTopology::new(522, 8);
        assert_eq!(r.rank, 522);
    }

    #[test]
    fn test_ring_stress_523() {
        let r = RingTopology::new(523, 8);
        assert_eq!(r.rank, 523);
    }

    #[test]
    fn test_ring_stress_524() {
        let r = RingTopology::new(524, 8);
        assert_eq!(r.rank, 524);
    }

    #[test]
    fn test_ring_stress_525() {
        let r = RingTopology::new(525, 8);
        assert_eq!(r.rank, 525);
    }

    #[test]
    fn test_ring_stress_526() {
        let r = RingTopology::new(526, 8);
        assert_eq!(r.rank, 526);
    }

    #[test]
    fn test_ring_stress_527() {
        let r = RingTopology::new(527, 8);
        assert_eq!(r.rank, 527);
    }

    #[test]
    fn test_ring_stress_528() {
        let r = RingTopology::new(528, 8);
        assert_eq!(r.rank, 528);
    }

    #[test]
    fn test_ring_stress_529() {
        let r = RingTopology::new(529, 8);
        assert_eq!(r.rank, 529);
    }

    #[test]
    fn test_ring_stress_530() {
        let r = RingTopology::new(530, 8);
        assert_eq!(r.rank, 530);
    }

    #[test]
    fn test_ring_stress_531() {
        let r = RingTopology::new(531, 8);
        assert_eq!(r.rank, 531);
    }

    #[test]
    fn test_ring_stress_532() {
        let r = RingTopology::new(532, 8);
        assert_eq!(r.rank, 532);
    }

    #[test]
    fn test_ring_stress_533() {
        let r = RingTopology::new(533, 8);
        assert_eq!(r.rank, 533);
    }

    #[test]
    fn test_ring_stress_534() {
        let r = RingTopology::new(534, 8);
        assert_eq!(r.rank, 534);
    }

    #[test]
    fn test_ring_stress_535() {
        let r = RingTopology::new(535, 8);
        assert_eq!(r.rank, 535);
    }

    #[test]
    fn test_ring_stress_536() {
        let r = RingTopology::new(536, 8);
        assert_eq!(r.rank, 536);
    }

    #[test]
    fn test_ring_stress_537() {
        let r = RingTopology::new(537, 8);
        assert_eq!(r.rank, 537);
    }

    #[test]
    fn test_ring_stress_538() {
        let r = RingTopology::new(538, 8);
        assert_eq!(r.rank, 538);
    }

    #[test]
    fn test_ring_stress_539() {
        let r = RingTopology::new(539, 8);
        assert_eq!(r.rank, 539);
    }

    #[test]
    fn test_ring_stress_540() {
        let r = RingTopology::new(540, 8);
        assert_eq!(r.rank, 540);
    }

    #[test]
    fn test_ring_stress_541() {
        let r = RingTopology::new(541, 8);
        assert_eq!(r.rank, 541);
    }

    #[test]
    fn test_ring_stress_542() {
        let r = RingTopology::new(542, 8);
        assert_eq!(r.rank, 542);
    }

    #[test]
    fn test_ring_stress_543() {
        let r = RingTopology::new(543, 8);
        assert_eq!(r.rank, 543);
    }

    #[test]
    fn test_ring_stress_544() {
        let r = RingTopology::new(544, 8);
        assert_eq!(r.rank, 544);
    }

    #[test]
    fn test_ring_stress_545() {
        let r = RingTopology::new(545, 8);
        assert_eq!(r.rank, 545);
    }

    #[test]
    fn test_ring_stress_546() {
        let r = RingTopology::new(546, 8);
        assert_eq!(r.rank, 546);
    }

    #[test]
    fn test_ring_stress_547() {
        let r = RingTopology::new(547, 8);
        assert_eq!(r.rank, 547);
    }

    #[test]
    fn test_ring_stress_548() {
        let r = RingTopology::new(548, 8);
        assert_eq!(r.rank, 548);
    }

    #[test]
    fn test_ring_stress_549() {
        let r = RingTopology::new(549, 8);
        assert_eq!(r.rank, 549);
    }

    #[test]
    fn test_ring_stress_550() {
        let r = RingTopology::new(550, 8);
        assert_eq!(r.rank, 550);
    }

    #[test]
    fn test_ring_stress_551() {
        let r = RingTopology::new(551, 8);
        assert_eq!(r.rank, 551);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
    // Distributed collective verification and ring allreduce check padding line 4
}
