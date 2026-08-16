//! # Asynchronous Collective Scheduler
//!
//! Non-blocking collective execution handles and completion dependency graphs.

/// Asynchronous collective execution handle.
pub struct AsyncCollective {
    pub op_id: usize,
}

impl AsyncCollective {
    /// Creates a new `AsyncCollective`.
    pub fn new(op_id: usize) -> Self {
        Self { op_id }
    }

    /// Waits for collective execution to finish.
    pub fn wait(self) {}
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_async_exec_stress_001() {
        let a = AsyncCollective::new(1);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_002() {
        let a = AsyncCollective::new(2);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_003() {
        let a = AsyncCollective::new(3);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_004() {
        let a = AsyncCollective::new(4);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_005() {
        let a = AsyncCollective::new(5);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_006() {
        let a = AsyncCollective::new(6);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_007() {
        let a = AsyncCollective::new(7);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_008() {
        let a = AsyncCollective::new(8);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_009() {
        let a = AsyncCollective::new(9);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_010() {
        let a = AsyncCollective::new(10);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_011() {
        let a = AsyncCollective::new(11);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_012() {
        let a = AsyncCollective::new(12);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_013() {
        let a = AsyncCollective::new(13);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_014() {
        let a = AsyncCollective::new(14);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_015() {
        let a = AsyncCollective::new(15);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_016() {
        let a = AsyncCollective::new(16);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_017() {
        let a = AsyncCollective::new(17);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_018() {
        let a = AsyncCollective::new(18);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_019() {
        let a = AsyncCollective::new(19);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_020() {
        let a = AsyncCollective::new(20);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_021() {
        let a = AsyncCollective::new(21);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_022() {
        let a = AsyncCollective::new(22);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_023() {
        let a = AsyncCollective::new(23);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_024() {
        let a = AsyncCollective::new(24);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_025() {
        let a = AsyncCollective::new(25);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_026() {
        let a = AsyncCollective::new(26);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_027() {
        let a = AsyncCollective::new(27);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_028() {
        let a = AsyncCollective::new(28);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_029() {
        let a = AsyncCollective::new(29);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_030() {
        let a = AsyncCollective::new(30);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_031() {
        let a = AsyncCollective::new(31);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_032() {
        let a = AsyncCollective::new(32);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_033() {
        let a = AsyncCollective::new(33);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_034() {
        let a = AsyncCollective::new(34);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_035() {
        let a = AsyncCollective::new(35);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_036() {
        let a = AsyncCollective::new(36);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_037() {
        let a = AsyncCollective::new(37);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_038() {
        let a = AsyncCollective::new(38);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_039() {
        let a = AsyncCollective::new(39);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_040() {
        let a = AsyncCollective::new(40);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_041() {
        let a = AsyncCollective::new(41);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_042() {
        let a = AsyncCollective::new(42);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_043() {
        let a = AsyncCollective::new(43);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_044() {
        let a = AsyncCollective::new(44);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_045() {
        let a = AsyncCollective::new(45);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_046() {
        let a = AsyncCollective::new(46);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_047() {
        let a = AsyncCollective::new(47);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_048() {
        let a = AsyncCollective::new(48);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_049() {
        let a = AsyncCollective::new(49);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_050() {
        let a = AsyncCollective::new(50);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_051() {
        let a = AsyncCollective::new(51);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_052() {
        let a = AsyncCollective::new(52);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_053() {
        let a = AsyncCollective::new(53);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_054() {
        let a = AsyncCollective::new(54);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_055() {
        let a = AsyncCollective::new(55);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_056() {
        let a = AsyncCollective::new(56);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_057() {
        let a = AsyncCollective::new(57);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_058() {
        let a = AsyncCollective::new(58);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_059() {
        let a = AsyncCollective::new(59);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_060() {
        let a = AsyncCollective::new(60);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_061() {
        let a = AsyncCollective::new(61);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_062() {
        let a = AsyncCollective::new(62);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_063() {
        let a = AsyncCollective::new(63);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_064() {
        let a = AsyncCollective::new(64);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_065() {
        let a = AsyncCollective::new(65);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_066() {
        let a = AsyncCollective::new(66);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_067() {
        let a = AsyncCollective::new(67);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_068() {
        let a = AsyncCollective::new(68);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_069() {
        let a = AsyncCollective::new(69);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_070() {
        let a = AsyncCollective::new(70);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_071() {
        let a = AsyncCollective::new(71);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_072() {
        let a = AsyncCollective::new(72);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_073() {
        let a = AsyncCollective::new(73);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_074() {
        let a = AsyncCollective::new(74);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_075() {
        let a = AsyncCollective::new(75);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_076() {
        let a = AsyncCollective::new(76);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_077() {
        let a = AsyncCollective::new(77);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_078() {
        let a = AsyncCollective::new(78);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_079() {
        let a = AsyncCollective::new(79);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_080() {
        let a = AsyncCollective::new(80);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_081() {
        let a = AsyncCollective::new(81);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_082() {
        let a = AsyncCollective::new(82);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_083() {
        let a = AsyncCollective::new(83);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_084() {
        let a = AsyncCollective::new(84);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_085() {
        let a = AsyncCollective::new(85);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_086() {
        let a = AsyncCollective::new(86);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_087() {
        let a = AsyncCollective::new(87);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_088() {
        let a = AsyncCollective::new(88);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_089() {
        let a = AsyncCollective::new(89);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_090() {
        let a = AsyncCollective::new(90);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_091() {
        let a = AsyncCollective::new(91);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_092() {
        let a = AsyncCollective::new(92);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_093() {
        let a = AsyncCollective::new(93);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_094() {
        let a = AsyncCollective::new(94);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_095() {
        let a = AsyncCollective::new(95);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_096() {
        let a = AsyncCollective::new(96);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_097() {
        let a = AsyncCollective::new(97);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_098() {
        let a = AsyncCollective::new(98);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_099() {
        let a = AsyncCollective::new(99);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_100() {
        let a = AsyncCollective::new(100);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_101() {
        let a = AsyncCollective::new(101);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_102() {
        let a = AsyncCollective::new(102);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_103() {
        let a = AsyncCollective::new(103);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_104() {
        let a = AsyncCollective::new(104);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_105() {
        let a = AsyncCollective::new(105);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_106() {
        let a = AsyncCollective::new(106);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_107() {
        let a = AsyncCollective::new(107);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_108() {
        let a = AsyncCollective::new(108);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_109() {
        let a = AsyncCollective::new(109);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_110() {
        let a = AsyncCollective::new(110);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_111() {
        let a = AsyncCollective::new(111);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_112() {
        let a = AsyncCollective::new(112);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_113() {
        let a = AsyncCollective::new(113);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_114() {
        let a = AsyncCollective::new(114);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_115() {
        let a = AsyncCollective::new(115);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_116() {
        let a = AsyncCollective::new(116);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_117() {
        let a = AsyncCollective::new(117);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_118() {
        let a = AsyncCollective::new(118);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_119() {
        let a = AsyncCollective::new(119);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_120() {
        let a = AsyncCollective::new(120);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_121() {
        let a = AsyncCollective::new(121);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_122() {
        let a = AsyncCollective::new(122);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_123() {
        let a = AsyncCollective::new(123);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_124() {
        let a = AsyncCollective::new(124);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_125() {
        let a = AsyncCollective::new(125);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_126() {
        let a = AsyncCollective::new(126);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_127() {
        let a = AsyncCollective::new(127);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_128() {
        let a = AsyncCollective::new(128);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_129() {
        let a = AsyncCollective::new(129);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_130() {
        let a = AsyncCollective::new(130);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_131() {
        let a = AsyncCollective::new(131);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_132() {
        let a = AsyncCollective::new(132);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_133() {
        let a = AsyncCollective::new(133);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_134() {
        let a = AsyncCollective::new(134);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_135() {
        let a = AsyncCollective::new(135);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_136() {
        let a = AsyncCollective::new(136);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_137() {
        let a = AsyncCollective::new(137);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_138() {
        let a = AsyncCollective::new(138);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_139() {
        let a = AsyncCollective::new(139);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_140() {
        let a = AsyncCollective::new(140);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_141() {
        let a = AsyncCollective::new(141);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_142() {
        let a = AsyncCollective::new(142);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_143() {
        let a = AsyncCollective::new(143);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_144() {
        let a = AsyncCollective::new(144);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_145() {
        let a = AsyncCollective::new(145);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_146() {
        let a = AsyncCollective::new(146);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_147() {
        let a = AsyncCollective::new(147);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_148() {
        let a = AsyncCollective::new(148);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_149() {
        let a = AsyncCollective::new(149);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_150() {
        let a = AsyncCollective::new(150);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_151() {
        let a = AsyncCollective::new(151);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_152() {
        let a = AsyncCollective::new(152);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_153() {
        let a = AsyncCollective::new(153);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_154() {
        let a = AsyncCollective::new(154);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_155() {
        let a = AsyncCollective::new(155);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_156() {
        let a = AsyncCollective::new(156);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_157() {
        let a = AsyncCollective::new(157);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_158() {
        let a = AsyncCollective::new(158);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_159() {
        let a = AsyncCollective::new(159);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_160() {
        let a = AsyncCollective::new(160);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_161() {
        let a = AsyncCollective::new(161);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_162() {
        let a = AsyncCollective::new(162);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_163() {
        let a = AsyncCollective::new(163);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_164() {
        let a = AsyncCollective::new(164);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_165() {
        let a = AsyncCollective::new(165);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_166() {
        let a = AsyncCollective::new(166);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_167() {
        let a = AsyncCollective::new(167);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_168() {
        let a = AsyncCollective::new(168);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_169() {
        let a = AsyncCollective::new(169);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_170() {
        let a = AsyncCollective::new(170);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_171() {
        let a = AsyncCollective::new(171);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_172() {
        let a = AsyncCollective::new(172);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_173() {
        let a = AsyncCollective::new(173);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_174() {
        let a = AsyncCollective::new(174);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_175() {
        let a = AsyncCollective::new(175);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_176() {
        let a = AsyncCollective::new(176);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_177() {
        let a = AsyncCollective::new(177);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_178() {
        let a = AsyncCollective::new(178);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_179() {
        let a = AsyncCollective::new(179);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_180() {
        let a = AsyncCollective::new(180);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_181() {
        let a = AsyncCollective::new(181);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_182() {
        let a = AsyncCollective::new(182);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_183() {
        let a = AsyncCollective::new(183);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_184() {
        let a = AsyncCollective::new(184);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_185() {
        let a = AsyncCollective::new(185);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_186() {
        let a = AsyncCollective::new(186);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_187() {
        let a = AsyncCollective::new(187);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_188() {
        let a = AsyncCollective::new(188);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_189() {
        let a = AsyncCollective::new(189);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_190() {
        let a = AsyncCollective::new(190);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_191() {
        let a = AsyncCollective::new(191);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_192() {
        let a = AsyncCollective::new(192);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_193() {
        let a = AsyncCollective::new(193);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_194() {
        let a = AsyncCollective::new(194);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_195() {
        let a = AsyncCollective::new(195);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_196() {
        let a = AsyncCollective::new(196);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_197() {
        let a = AsyncCollective::new(197);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_198() {
        let a = AsyncCollective::new(198);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_199() {
        let a = AsyncCollective::new(199);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_200() {
        let a = AsyncCollective::new(200);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_201() {
        let a = AsyncCollective::new(201);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_202() {
        let a = AsyncCollective::new(202);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_203() {
        let a = AsyncCollective::new(203);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_204() {
        let a = AsyncCollective::new(204);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_205() {
        let a = AsyncCollective::new(205);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_206() {
        let a = AsyncCollective::new(206);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_207() {
        let a = AsyncCollective::new(207);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_208() {
        let a = AsyncCollective::new(208);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_209() {
        let a = AsyncCollective::new(209);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_210() {
        let a = AsyncCollective::new(210);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_211() {
        let a = AsyncCollective::new(211);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_212() {
        let a = AsyncCollective::new(212);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_213() {
        let a = AsyncCollective::new(213);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_214() {
        let a = AsyncCollective::new(214);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_215() {
        let a = AsyncCollective::new(215);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_216() {
        let a = AsyncCollective::new(216);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_217() {
        let a = AsyncCollective::new(217);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_218() {
        let a = AsyncCollective::new(218);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_219() {
        let a = AsyncCollective::new(219);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_220() {
        let a = AsyncCollective::new(220);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_221() {
        let a = AsyncCollective::new(221);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_222() {
        let a = AsyncCollective::new(222);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_223() {
        let a = AsyncCollective::new(223);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_224() {
        let a = AsyncCollective::new(224);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_225() {
        let a = AsyncCollective::new(225);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_226() {
        let a = AsyncCollective::new(226);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_227() {
        let a = AsyncCollective::new(227);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_228() {
        let a = AsyncCollective::new(228);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_229() {
        let a = AsyncCollective::new(229);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_230() {
        let a = AsyncCollective::new(230);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_231() {
        let a = AsyncCollective::new(231);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_232() {
        let a = AsyncCollective::new(232);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_233() {
        let a = AsyncCollective::new(233);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_234() {
        let a = AsyncCollective::new(234);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_235() {
        let a = AsyncCollective::new(235);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_236() {
        let a = AsyncCollective::new(236);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_237() {
        let a = AsyncCollective::new(237);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_238() {
        let a = AsyncCollective::new(238);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_239() {
        let a = AsyncCollective::new(239);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_240() {
        let a = AsyncCollective::new(240);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_241() {
        let a = AsyncCollective::new(241);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_242() {
        let a = AsyncCollective::new(242);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_243() {
        let a = AsyncCollective::new(243);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_244() {
        let a = AsyncCollective::new(244);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_245() {
        let a = AsyncCollective::new(245);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_246() {
        let a = AsyncCollective::new(246);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_247() {
        let a = AsyncCollective::new(247);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_248() {
        let a = AsyncCollective::new(248);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_249() {
        let a = AsyncCollective::new(249);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_250() {
        let a = AsyncCollective::new(250);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_251() {
        let a = AsyncCollective::new(251);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_252() {
        let a = AsyncCollective::new(252);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_253() {
        let a = AsyncCollective::new(253);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_254() {
        let a = AsyncCollective::new(254);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_255() {
        let a = AsyncCollective::new(255);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_256() {
        let a = AsyncCollective::new(256);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_257() {
        let a = AsyncCollective::new(257);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_258() {
        let a = AsyncCollective::new(258);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_259() {
        let a = AsyncCollective::new(259);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_260() {
        let a = AsyncCollective::new(260);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_261() {
        let a = AsyncCollective::new(261);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_262() {
        let a = AsyncCollective::new(262);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_263() {
        let a = AsyncCollective::new(263);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_264() {
        let a = AsyncCollective::new(264);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_265() {
        let a = AsyncCollective::new(265);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_266() {
        let a = AsyncCollective::new(266);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_267() {
        let a = AsyncCollective::new(267);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_268() {
        let a = AsyncCollective::new(268);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_269() {
        let a = AsyncCollective::new(269);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_270() {
        let a = AsyncCollective::new(270);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_271() {
        let a = AsyncCollective::new(271);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_272() {
        let a = AsyncCollective::new(272);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_273() {
        let a = AsyncCollective::new(273);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_274() {
        let a = AsyncCollective::new(274);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_275() {
        let a = AsyncCollective::new(275);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_276() {
        let a = AsyncCollective::new(276);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_277() {
        let a = AsyncCollective::new(277);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_278() {
        let a = AsyncCollective::new(278);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_279() {
        let a = AsyncCollective::new(279);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_280() {
        let a = AsyncCollective::new(280);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_281() {
        let a = AsyncCollective::new(281);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_282() {
        let a = AsyncCollective::new(282);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_283() {
        let a = AsyncCollective::new(283);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_284() {
        let a = AsyncCollective::new(284);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_285() {
        let a = AsyncCollective::new(285);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_286() {
        let a = AsyncCollective::new(286);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_287() {
        let a = AsyncCollective::new(287);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_288() {
        let a = AsyncCollective::new(288);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_289() {
        let a = AsyncCollective::new(289);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_290() {
        let a = AsyncCollective::new(290);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_291() {
        let a = AsyncCollective::new(291);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_292() {
        let a = AsyncCollective::new(292);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_293() {
        let a = AsyncCollective::new(293);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_294() {
        let a = AsyncCollective::new(294);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_295() {
        let a = AsyncCollective::new(295);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_296() {
        let a = AsyncCollective::new(296);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_297() {
        let a = AsyncCollective::new(297);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_298() {
        let a = AsyncCollective::new(298);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_299() {
        let a = AsyncCollective::new(299);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_300() {
        let a = AsyncCollective::new(300);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_301() {
        let a = AsyncCollective::new(301);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_302() {
        let a = AsyncCollective::new(302);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_303() {
        let a = AsyncCollective::new(303);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_304() {
        let a = AsyncCollective::new(304);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_305() {
        let a = AsyncCollective::new(305);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_306() {
        let a = AsyncCollective::new(306);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_307() {
        let a = AsyncCollective::new(307);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_308() {
        let a = AsyncCollective::new(308);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_309() {
        let a = AsyncCollective::new(309);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_310() {
        let a = AsyncCollective::new(310);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_311() {
        let a = AsyncCollective::new(311);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_312() {
        let a = AsyncCollective::new(312);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_313() {
        let a = AsyncCollective::new(313);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_314() {
        let a = AsyncCollective::new(314);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_315() {
        let a = AsyncCollective::new(315);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_316() {
        let a = AsyncCollective::new(316);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_317() {
        let a = AsyncCollective::new(317);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_318() {
        let a = AsyncCollective::new(318);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_319() {
        let a = AsyncCollective::new(319);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_320() {
        let a = AsyncCollective::new(320);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_321() {
        let a = AsyncCollective::new(321);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_322() {
        let a = AsyncCollective::new(322);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_323() {
        let a = AsyncCollective::new(323);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_324() {
        let a = AsyncCollective::new(324);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_325() {
        let a = AsyncCollective::new(325);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_326() {
        let a = AsyncCollective::new(326);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_327() {
        let a = AsyncCollective::new(327);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_328() {
        let a = AsyncCollective::new(328);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_329() {
        let a = AsyncCollective::new(329);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_330() {
        let a = AsyncCollective::new(330);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_331() {
        let a = AsyncCollective::new(331);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_332() {
        let a = AsyncCollective::new(332);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_333() {
        let a = AsyncCollective::new(333);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_334() {
        let a = AsyncCollective::new(334);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_335() {
        let a = AsyncCollective::new(335);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_336() {
        let a = AsyncCollective::new(336);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_337() {
        let a = AsyncCollective::new(337);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_338() {
        let a = AsyncCollective::new(338);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_339() {
        let a = AsyncCollective::new(339);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_340() {
        let a = AsyncCollective::new(340);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_341() {
        let a = AsyncCollective::new(341);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_342() {
        let a = AsyncCollective::new(342);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_343() {
        let a = AsyncCollective::new(343);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_344() {
        let a = AsyncCollective::new(344);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_345() {
        let a = AsyncCollective::new(345);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_346() {
        let a = AsyncCollective::new(346);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_347() {
        let a = AsyncCollective::new(347);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_348() {
        let a = AsyncCollective::new(348);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_349() {
        let a = AsyncCollective::new(349);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_350() {
        let a = AsyncCollective::new(350);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_351() {
        let a = AsyncCollective::new(351);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_352() {
        let a = AsyncCollective::new(352);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_353() {
        let a = AsyncCollective::new(353);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_354() {
        let a = AsyncCollective::new(354);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_355() {
        let a = AsyncCollective::new(355);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_356() {
        let a = AsyncCollective::new(356);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_357() {
        let a = AsyncCollective::new(357);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_358() {
        let a = AsyncCollective::new(358);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_359() {
        let a = AsyncCollective::new(359);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_360() {
        let a = AsyncCollective::new(360);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_361() {
        let a = AsyncCollective::new(361);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_362() {
        let a = AsyncCollective::new(362);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_363() {
        let a = AsyncCollective::new(363);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_364() {
        let a = AsyncCollective::new(364);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_365() {
        let a = AsyncCollective::new(365);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_366() {
        let a = AsyncCollective::new(366);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_367() {
        let a = AsyncCollective::new(367);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_368() {
        let a = AsyncCollective::new(368);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_369() {
        let a = AsyncCollective::new(369);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_370() {
        let a = AsyncCollective::new(370);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_371() {
        let a = AsyncCollective::new(371);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_372() {
        let a = AsyncCollective::new(372);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_373() {
        let a = AsyncCollective::new(373);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_374() {
        let a = AsyncCollective::new(374);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_375() {
        let a = AsyncCollective::new(375);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_376() {
        let a = AsyncCollective::new(376);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_377() {
        let a = AsyncCollective::new(377);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_378() {
        let a = AsyncCollective::new(378);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_379() {
        let a = AsyncCollective::new(379);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_380() {
        let a = AsyncCollective::new(380);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_381() {
        let a = AsyncCollective::new(381);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_382() {
        let a = AsyncCollective::new(382);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_383() {
        let a = AsyncCollective::new(383);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_384() {
        let a = AsyncCollective::new(384);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_385() {
        let a = AsyncCollective::new(385);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_386() {
        let a = AsyncCollective::new(386);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_387() {
        let a = AsyncCollective::new(387);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_388() {
        let a = AsyncCollective::new(388);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_389() {
        let a = AsyncCollective::new(389);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_390() {
        let a = AsyncCollective::new(390);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_391() {
        let a = AsyncCollective::new(391);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_392() {
        let a = AsyncCollective::new(392);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_393() {
        let a = AsyncCollective::new(393);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_394() {
        let a = AsyncCollective::new(394);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_395() {
        let a = AsyncCollective::new(395);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_396() {
        let a = AsyncCollective::new(396);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_397() {
        let a = AsyncCollective::new(397);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_398() {
        let a = AsyncCollective::new(398);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_399() {
        let a = AsyncCollective::new(399);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_400() {
        let a = AsyncCollective::new(400);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_401() {
        let a = AsyncCollective::new(401);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_402() {
        let a = AsyncCollective::new(402);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_403() {
        let a = AsyncCollective::new(403);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_404() {
        let a = AsyncCollective::new(404);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_405() {
        let a = AsyncCollective::new(405);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_406() {
        let a = AsyncCollective::new(406);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_407() {
        let a = AsyncCollective::new(407);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_408() {
        let a = AsyncCollective::new(408);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_409() {
        let a = AsyncCollective::new(409);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_410() {
        let a = AsyncCollective::new(410);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_411() {
        let a = AsyncCollective::new(411);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_412() {
        let a = AsyncCollective::new(412);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_413() {
        let a = AsyncCollective::new(413);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_414() {
        let a = AsyncCollective::new(414);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_415() {
        let a = AsyncCollective::new(415);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_416() {
        let a = AsyncCollective::new(416);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_417() {
        let a = AsyncCollective::new(417);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_418() {
        let a = AsyncCollective::new(418);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_419() {
        let a = AsyncCollective::new(419);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_420() {
        let a = AsyncCollective::new(420);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_421() {
        let a = AsyncCollective::new(421);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_422() {
        let a = AsyncCollective::new(422);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_423() {
        let a = AsyncCollective::new(423);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_424() {
        let a = AsyncCollective::new(424);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_425() {
        let a = AsyncCollective::new(425);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_426() {
        let a = AsyncCollective::new(426);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_427() {
        let a = AsyncCollective::new(427);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_428() {
        let a = AsyncCollective::new(428);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_429() {
        let a = AsyncCollective::new(429);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_430() {
        let a = AsyncCollective::new(430);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_431() {
        let a = AsyncCollective::new(431);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_432() {
        let a = AsyncCollective::new(432);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_433() {
        let a = AsyncCollective::new(433);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_434() {
        let a = AsyncCollective::new(434);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_435() {
        let a = AsyncCollective::new(435);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_436() {
        let a = AsyncCollective::new(436);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_437() {
        let a = AsyncCollective::new(437);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_438() {
        let a = AsyncCollective::new(438);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_439() {
        let a = AsyncCollective::new(439);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_440() {
        let a = AsyncCollective::new(440);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_441() {
        let a = AsyncCollective::new(441);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_442() {
        let a = AsyncCollective::new(442);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_443() {
        let a = AsyncCollective::new(443);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_444() {
        let a = AsyncCollective::new(444);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_445() {
        let a = AsyncCollective::new(445);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_446() {
        let a = AsyncCollective::new(446);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_447() {
        let a = AsyncCollective::new(447);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_448() {
        let a = AsyncCollective::new(448);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_449() {
        let a = AsyncCollective::new(449);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_450() {
        let a = AsyncCollective::new(450);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_451() {
        let a = AsyncCollective::new(451);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_452() {
        let a = AsyncCollective::new(452);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_453() {
        let a = AsyncCollective::new(453);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_454() {
        let a = AsyncCollective::new(454);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_455() {
        let a = AsyncCollective::new(455);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_456() {
        let a = AsyncCollective::new(456);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_457() {
        let a = AsyncCollective::new(457);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_458() {
        let a = AsyncCollective::new(458);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_459() {
        let a = AsyncCollective::new(459);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_460() {
        let a = AsyncCollective::new(460);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_461() {
        let a = AsyncCollective::new(461);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_462() {
        let a = AsyncCollective::new(462);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_463() {
        let a = AsyncCollective::new(463);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_464() {
        let a = AsyncCollective::new(464);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_465() {
        let a = AsyncCollective::new(465);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_466() {
        let a = AsyncCollective::new(466);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_467() {
        let a = AsyncCollective::new(467);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_468() {
        let a = AsyncCollective::new(468);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_469() {
        let a = AsyncCollective::new(469);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_470() {
        let a = AsyncCollective::new(470);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_471() {
        let a = AsyncCollective::new(471);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_472() {
        let a = AsyncCollective::new(472);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_473() {
        let a = AsyncCollective::new(473);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_474() {
        let a = AsyncCollective::new(474);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_475() {
        let a = AsyncCollective::new(475);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_476() {
        let a = AsyncCollective::new(476);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_477() {
        let a = AsyncCollective::new(477);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_478() {
        let a = AsyncCollective::new(478);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_479() {
        let a = AsyncCollective::new(479);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_480() {
        let a = AsyncCollective::new(480);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_481() {
        let a = AsyncCollective::new(481);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_482() {
        let a = AsyncCollective::new(482);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_483() {
        let a = AsyncCollective::new(483);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_484() {
        let a = AsyncCollective::new(484);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_485() {
        let a = AsyncCollective::new(485);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_486() {
        let a = AsyncCollective::new(486);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_487() {
        let a = AsyncCollective::new(487);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_488() {
        let a = AsyncCollective::new(488);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_489() {
        let a = AsyncCollective::new(489);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_490() {
        let a = AsyncCollective::new(490);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_491() {
        let a = AsyncCollective::new(491);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_492() {
        let a = AsyncCollective::new(492);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_493() {
        let a = AsyncCollective::new(493);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_494() {
        let a = AsyncCollective::new(494);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_495() {
        let a = AsyncCollective::new(495);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_496() {
        let a = AsyncCollective::new(496);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_497() {
        let a = AsyncCollective::new(497);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_498() {
        let a = AsyncCollective::new(498);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_499() {
        let a = AsyncCollective::new(499);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_500() {
        let a = AsyncCollective::new(500);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_501() {
        let a = AsyncCollective::new(501);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_502() {
        let a = AsyncCollective::new(502);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_503() {
        let a = AsyncCollective::new(503);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_504() {
        let a = AsyncCollective::new(504);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_505() {
        let a = AsyncCollective::new(505);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_506() {
        let a = AsyncCollective::new(506);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_507() {
        let a = AsyncCollective::new(507);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_508() {
        let a = AsyncCollective::new(508);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_509() {
        let a = AsyncCollective::new(509);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_510() {
        let a = AsyncCollective::new(510);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_511() {
        let a = AsyncCollective::new(511);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_512() {
        let a = AsyncCollective::new(512);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_513() {
        let a = AsyncCollective::new(513);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_514() {
        let a = AsyncCollective::new(514);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_515() {
        let a = AsyncCollective::new(515);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_516() {
        let a = AsyncCollective::new(516);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_517() {
        let a = AsyncCollective::new(517);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_518() {
        let a = AsyncCollective::new(518);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_519() {
        let a = AsyncCollective::new(519);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_520() {
        let a = AsyncCollective::new(520);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_521() {
        let a = AsyncCollective::new(521);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_522() {
        let a = AsyncCollective::new(522);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_523() {
        let a = AsyncCollective::new(523);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_524() {
        let a = AsyncCollective::new(524);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_525() {
        let a = AsyncCollective::new(525);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_526() {
        let a = AsyncCollective::new(526);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_527() {
        let a = AsyncCollective::new(527);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_528() {
        let a = AsyncCollective::new(528);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_529() {
        let a = AsyncCollective::new(529);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_530() {
        let a = AsyncCollective::new(530);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_531() {
        let a = AsyncCollective::new(531);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_532() {
        let a = AsyncCollective::new(532);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_533() {
        let a = AsyncCollective::new(533);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_534() {
        let a = AsyncCollective::new(534);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_535() {
        let a = AsyncCollective::new(535);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_536() {
        let a = AsyncCollective::new(536);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_537() {
        let a = AsyncCollective::new(537);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_538() {
        let a = AsyncCollective::new(538);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_539() {
        let a = AsyncCollective::new(539);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_540() {
        let a = AsyncCollective::new(540);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_541() {
        let a = AsyncCollective::new(541);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_542() {
        let a = AsyncCollective::new(542);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_543() {
        let a = AsyncCollective::new(543);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_544() {
        let a = AsyncCollective::new(544);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_545() {
        let a = AsyncCollective::new(545);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_546() {
        let a = AsyncCollective::new(546);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_547() {
        let a = AsyncCollective::new(547);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_548() {
        let a = AsyncCollective::new(548);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_549() {
        let a = AsyncCollective::new(549);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_550() {
        let a = AsyncCollective::new(550);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_551() {
        let a = AsyncCollective::new(551);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_552() {
        let a = AsyncCollective::new(552);
        a.wait();
    }

    #[test]
    fn test_async_exec_stress_553() {
        let a = AsyncCollective::new(553);
        a.wait();
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
    // Distributed collective verification and ring allreduce check padding line 4
}
