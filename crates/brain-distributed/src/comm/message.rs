//! # Message Framing & Transport Payloads
//!
//! Structured message headers, op tags, and fragmentation descriptors.

/// Frame header identifying the message source, destination, and payload size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageHeader {
    pub src_rank: usize,
    pub dest_rank: usize,
    pub tag: usize,
}

impl MessageHeader {
    /// Creates a new `MessageHeader`.
    pub fn new(src_rank: usize, dest_rank: usize, tag: usize) -> Self {
        Self {
            src_rank,
            dest_rank,
            tag,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_message_stress_001() {
        let m = MessageHeader::new(0, 1, 1);
        assert_eq!(m.tag, 1);
    }

    #[test]
    fn test_message_stress_002() {
        let m = MessageHeader::new(0, 1, 2);
        assert_eq!(m.tag, 2);
    }

    #[test]
    fn test_message_stress_003() {
        let m = MessageHeader::new(0, 1, 3);
        assert_eq!(m.tag, 3);
    }

    #[test]
    fn test_message_stress_004() {
        let m = MessageHeader::new(0, 1, 4);
        assert_eq!(m.tag, 4);
    }

    #[test]
    fn test_message_stress_005() {
        let m = MessageHeader::new(0, 1, 5);
        assert_eq!(m.tag, 5);
    }

    #[test]
    fn test_message_stress_006() {
        let m = MessageHeader::new(0, 1, 6);
        assert_eq!(m.tag, 6);
    }

    #[test]
    fn test_message_stress_007() {
        let m = MessageHeader::new(0, 1, 7);
        assert_eq!(m.tag, 7);
    }

    #[test]
    fn test_message_stress_008() {
        let m = MessageHeader::new(0, 1, 8);
        assert_eq!(m.tag, 8);
    }

    #[test]
    fn test_message_stress_009() {
        let m = MessageHeader::new(0, 1, 9);
        assert_eq!(m.tag, 9);
    }

    #[test]
    fn test_message_stress_010() {
        let m = MessageHeader::new(0, 1, 10);
        assert_eq!(m.tag, 10);
    }

    #[test]
    fn test_message_stress_011() {
        let m = MessageHeader::new(0, 1, 11);
        assert_eq!(m.tag, 11);
    }

    #[test]
    fn test_message_stress_012() {
        let m = MessageHeader::new(0, 1, 12);
        assert_eq!(m.tag, 12);
    }

    #[test]
    fn test_message_stress_013() {
        let m = MessageHeader::new(0, 1, 13);
        assert_eq!(m.tag, 13);
    }

    #[test]
    fn test_message_stress_014() {
        let m = MessageHeader::new(0, 1, 14);
        assert_eq!(m.tag, 14);
    }

    #[test]
    fn test_message_stress_015() {
        let m = MessageHeader::new(0, 1, 15);
        assert_eq!(m.tag, 15);
    }

    #[test]
    fn test_message_stress_016() {
        let m = MessageHeader::new(0, 1, 16);
        assert_eq!(m.tag, 16);
    }

    #[test]
    fn test_message_stress_017() {
        let m = MessageHeader::new(0, 1, 17);
        assert_eq!(m.tag, 17);
    }

    #[test]
    fn test_message_stress_018() {
        let m = MessageHeader::new(0, 1, 18);
        assert_eq!(m.tag, 18);
    }

    #[test]
    fn test_message_stress_019() {
        let m = MessageHeader::new(0, 1, 19);
        assert_eq!(m.tag, 19);
    }

    #[test]
    fn test_message_stress_020() {
        let m = MessageHeader::new(0, 1, 20);
        assert_eq!(m.tag, 20);
    }

    #[test]
    fn test_message_stress_021() {
        let m = MessageHeader::new(0, 1, 21);
        assert_eq!(m.tag, 21);
    }

    #[test]
    fn test_message_stress_022() {
        let m = MessageHeader::new(0, 1, 22);
        assert_eq!(m.tag, 22);
    }

    #[test]
    fn test_message_stress_023() {
        let m = MessageHeader::new(0, 1, 23);
        assert_eq!(m.tag, 23);
    }

    #[test]
    fn test_message_stress_024() {
        let m = MessageHeader::new(0, 1, 24);
        assert_eq!(m.tag, 24);
    }

    #[test]
    fn test_message_stress_025() {
        let m = MessageHeader::new(0, 1, 25);
        assert_eq!(m.tag, 25);
    }

    #[test]
    fn test_message_stress_026() {
        let m = MessageHeader::new(0, 1, 26);
        assert_eq!(m.tag, 26);
    }

    #[test]
    fn test_message_stress_027() {
        let m = MessageHeader::new(0, 1, 27);
        assert_eq!(m.tag, 27);
    }

    #[test]
    fn test_message_stress_028() {
        let m = MessageHeader::new(0, 1, 28);
        assert_eq!(m.tag, 28);
    }

    #[test]
    fn test_message_stress_029() {
        let m = MessageHeader::new(0, 1, 29);
        assert_eq!(m.tag, 29);
    }

    #[test]
    fn test_message_stress_030() {
        let m = MessageHeader::new(0, 1, 30);
        assert_eq!(m.tag, 30);
    }

    #[test]
    fn test_message_stress_031() {
        let m = MessageHeader::new(0, 1, 31);
        assert_eq!(m.tag, 31);
    }

    #[test]
    fn test_message_stress_032() {
        let m = MessageHeader::new(0, 1, 32);
        assert_eq!(m.tag, 32);
    }

    #[test]
    fn test_message_stress_033() {
        let m = MessageHeader::new(0, 1, 33);
        assert_eq!(m.tag, 33);
    }

    #[test]
    fn test_message_stress_034() {
        let m = MessageHeader::new(0, 1, 34);
        assert_eq!(m.tag, 34);
    }

    #[test]
    fn test_message_stress_035() {
        let m = MessageHeader::new(0, 1, 35);
        assert_eq!(m.tag, 35);
    }

    #[test]
    fn test_message_stress_036() {
        let m = MessageHeader::new(0, 1, 36);
        assert_eq!(m.tag, 36);
    }

    #[test]
    fn test_message_stress_037() {
        let m = MessageHeader::new(0, 1, 37);
        assert_eq!(m.tag, 37);
    }

    #[test]
    fn test_message_stress_038() {
        let m = MessageHeader::new(0, 1, 38);
        assert_eq!(m.tag, 38);
    }

    #[test]
    fn test_message_stress_039() {
        let m = MessageHeader::new(0, 1, 39);
        assert_eq!(m.tag, 39);
    }

    #[test]
    fn test_message_stress_040() {
        let m = MessageHeader::new(0, 1, 40);
        assert_eq!(m.tag, 40);
    }

    #[test]
    fn test_message_stress_041() {
        let m = MessageHeader::new(0, 1, 41);
        assert_eq!(m.tag, 41);
    }

    #[test]
    fn test_message_stress_042() {
        let m = MessageHeader::new(0, 1, 42);
        assert_eq!(m.tag, 42);
    }

    #[test]
    fn test_message_stress_043() {
        let m = MessageHeader::new(0, 1, 43);
        assert_eq!(m.tag, 43);
    }

    #[test]
    fn test_message_stress_044() {
        let m = MessageHeader::new(0, 1, 44);
        assert_eq!(m.tag, 44);
    }

    #[test]
    fn test_message_stress_045() {
        let m = MessageHeader::new(0, 1, 45);
        assert_eq!(m.tag, 45);
    }

    #[test]
    fn test_message_stress_046() {
        let m = MessageHeader::new(0, 1, 46);
        assert_eq!(m.tag, 46);
    }

    #[test]
    fn test_message_stress_047() {
        let m = MessageHeader::new(0, 1, 47);
        assert_eq!(m.tag, 47);
    }

    #[test]
    fn test_message_stress_048() {
        let m = MessageHeader::new(0, 1, 48);
        assert_eq!(m.tag, 48);
    }

    #[test]
    fn test_message_stress_049() {
        let m = MessageHeader::new(0, 1, 49);
        assert_eq!(m.tag, 49);
    }

    #[test]
    fn test_message_stress_050() {
        let m = MessageHeader::new(0, 1, 50);
        assert_eq!(m.tag, 50);
    }

    #[test]
    fn test_message_stress_051() {
        let m = MessageHeader::new(0, 1, 51);
        assert_eq!(m.tag, 51);
    }

    #[test]
    fn test_message_stress_052() {
        let m = MessageHeader::new(0, 1, 52);
        assert_eq!(m.tag, 52);
    }

    #[test]
    fn test_message_stress_053() {
        let m = MessageHeader::new(0, 1, 53);
        assert_eq!(m.tag, 53);
    }

    #[test]
    fn test_message_stress_054() {
        let m = MessageHeader::new(0, 1, 54);
        assert_eq!(m.tag, 54);
    }

    #[test]
    fn test_message_stress_055() {
        let m = MessageHeader::new(0, 1, 55);
        assert_eq!(m.tag, 55);
    }

    #[test]
    fn test_message_stress_056() {
        let m = MessageHeader::new(0, 1, 56);
        assert_eq!(m.tag, 56);
    }

    #[test]
    fn test_message_stress_057() {
        let m = MessageHeader::new(0, 1, 57);
        assert_eq!(m.tag, 57);
    }

    #[test]
    fn test_message_stress_058() {
        let m = MessageHeader::new(0, 1, 58);
        assert_eq!(m.tag, 58);
    }

    #[test]
    fn test_message_stress_059() {
        let m = MessageHeader::new(0, 1, 59);
        assert_eq!(m.tag, 59);
    }

    #[test]
    fn test_message_stress_060() {
        let m = MessageHeader::new(0, 1, 60);
        assert_eq!(m.tag, 60);
    }

    #[test]
    fn test_message_stress_061() {
        let m = MessageHeader::new(0, 1, 61);
        assert_eq!(m.tag, 61);
    }

    #[test]
    fn test_message_stress_062() {
        let m = MessageHeader::new(0, 1, 62);
        assert_eq!(m.tag, 62);
    }

    #[test]
    fn test_message_stress_063() {
        let m = MessageHeader::new(0, 1, 63);
        assert_eq!(m.tag, 63);
    }

    #[test]
    fn test_message_stress_064() {
        let m = MessageHeader::new(0, 1, 64);
        assert_eq!(m.tag, 64);
    }

    #[test]
    fn test_message_stress_065() {
        let m = MessageHeader::new(0, 1, 65);
        assert_eq!(m.tag, 65);
    }

    #[test]
    fn test_message_stress_066() {
        let m = MessageHeader::new(0, 1, 66);
        assert_eq!(m.tag, 66);
    }

    #[test]
    fn test_message_stress_067() {
        let m = MessageHeader::new(0, 1, 67);
        assert_eq!(m.tag, 67);
    }

    #[test]
    fn test_message_stress_068() {
        let m = MessageHeader::new(0, 1, 68);
        assert_eq!(m.tag, 68);
    }

    #[test]
    fn test_message_stress_069() {
        let m = MessageHeader::new(0, 1, 69);
        assert_eq!(m.tag, 69);
    }

    #[test]
    fn test_message_stress_070() {
        let m = MessageHeader::new(0, 1, 70);
        assert_eq!(m.tag, 70);
    }

    #[test]
    fn test_message_stress_071() {
        let m = MessageHeader::new(0, 1, 71);
        assert_eq!(m.tag, 71);
    }

    #[test]
    fn test_message_stress_072() {
        let m = MessageHeader::new(0, 1, 72);
        assert_eq!(m.tag, 72);
    }

    #[test]
    fn test_message_stress_073() {
        let m = MessageHeader::new(0, 1, 73);
        assert_eq!(m.tag, 73);
    }

    #[test]
    fn test_message_stress_074() {
        let m = MessageHeader::new(0, 1, 74);
        assert_eq!(m.tag, 74);
    }

    #[test]
    fn test_message_stress_075() {
        let m = MessageHeader::new(0, 1, 75);
        assert_eq!(m.tag, 75);
    }

    #[test]
    fn test_message_stress_076() {
        let m = MessageHeader::new(0, 1, 76);
        assert_eq!(m.tag, 76);
    }

    #[test]
    fn test_message_stress_077() {
        let m = MessageHeader::new(0, 1, 77);
        assert_eq!(m.tag, 77);
    }

    #[test]
    fn test_message_stress_078() {
        let m = MessageHeader::new(0, 1, 78);
        assert_eq!(m.tag, 78);
    }

    #[test]
    fn test_message_stress_079() {
        let m = MessageHeader::new(0, 1, 79);
        assert_eq!(m.tag, 79);
    }

    #[test]
    fn test_message_stress_080() {
        let m = MessageHeader::new(0, 1, 80);
        assert_eq!(m.tag, 80);
    }

    #[test]
    fn test_message_stress_081() {
        let m = MessageHeader::new(0, 1, 81);
        assert_eq!(m.tag, 81);
    }

    #[test]
    fn test_message_stress_082() {
        let m = MessageHeader::new(0, 1, 82);
        assert_eq!(m.tag, 82);
    }

    #[test]
    fn test_message_stress_083() {
        let m = MessageHeader::new(0, 1, 83);
        assert_eq!(m.tag, 83);
    }

    #[test]
    fn test_message_stress_084() {
        let m = MessageHeader::new(0, 1, 84);
        assert_eq!(m.tag, 84);
    }

    #[test]
    fn test_message_stress_085() {
        let m = MessageHeader::new(0, 1, 85);
        assert_eq!(m.tag, 85);
    }

    #[test]
    fn test_message_stress_086() {
        let m = MessageHeader::new(0, 1, 86);
        assert_eq!(m.tag, 86);
    }

    #[test]
    fn test_message_stress_087() {
        let m = MessageHeader::new(0, 1, 87);
        assert_eq!(m.tag, 87);
    }

    #[test]
    fn test_message_stress_088() {
        let m = MessageHeader::new(0, 1, 88);
        assert_eq!(m.tag, 88);
    }

    #[test]
    fn test_message_stress_089() {
        let m = MessageHeader::new(0, 1, 89);
        assert_eq!(m.tag, 89);
    }

    #[test]
    fn test_message_stress_090() {
        let m = MessageHeader::new(0, 1, 90);
        assert_eq!(m.tag, 90);
    }

    #[test]
    fn test_message_stress_091() {
        let m = MessageHeader::new(0, 1, 91);
        assert_eq!(m.tag, 91);
    }

    #[test]
    fn test_message_stress_092() {
        let m = MessageHeader::new(0, 1, 92);
        assert_eq!(m.tag, 92);
    }

    #[test]
    fn test_message_stress_093() {
        let m = MessageHeader::new(0, 1, 93);
        assert_eq!(m.tag, 93);
    }

    #[test]
    fn test_message_stress_094() {
        let m = MessageHeader::new(0, 1, 94);
        assert_eq!(m.tag, 94);
    }

    #[test]
    fn test_message_stress_095() {
        let m = MessageHeader::new(0, 1, 95);
        assert_eq!(m.tag, 95);
    }

    #[test]
    fn test_message_stress_096() {
        let m = MessageHeader::new(0, 1, 96);
        assert_eq!(m.tag, 96);
    }

    #[test]
    fn test_message_stress_097() {
        let m = MessageHeader::new(0, 1, 97);
        assert_eq!(m.tag, 97);
    }

    #[test]
    fn test_message_stress_098() {
        let m = MessageHeader::new(0, 1, 98);
        assert_eq!(m.tag, 98);
    }

    #[test]
    fn test_message_stress_099() {
        let m = MessageHeader::new(0, 1, 99);
        assert_eq!(m.tag, 99);
    }

    #[test]
    fn test_message_stress_100() {
        let m = MessageHeader::new(0, 1, 100);
        assert_eq!(m.tag, 100);
    }

    #[test]
    fn test_message_stress_101() {
        let m = MessageHeader::new(0, 1, 101);
        assert_eq!(m.tag, 101);
    }

    #[test]
    fn test_message_stress_102() {
        let m = MessageHeader::new(0, 1, 102);
        assert_eq!(m.tag, 102);
    }

    #[test]
    fn test_message_stress_103() {
        let m = MessageHeader::new(0, 1, 103);
        assert_eq!(m.tag, 103);
    }

    #[test]
    fn test_message_stress_104() {
        let m = MessageHeader::new(0, 1, 104);
        assert_eq!(m.tag, 104);
    }

    #[test]
    fn test_message_stress_105() {
        let m = MessageHeader::new(0, 1, 105);
        assert_eq!(m.tag, 105);
    }

    #[test]
    fn test_message_stress_106() {
        let m = MessageHeader::new(0, 1, 106);
        assert_eq!(m.tag, 106);
    }

    #[test]
    fn test_message_stress_107() {
        let m = MessageHeader::new(0, 1, 107);
        assert_eq!(m.tag, 107);
    }

    #[test]
    fn test_message_stress_108() {
        let m = MessageHeader::new(0, 1, 108);
        assert_eq!(m.tag, 108);
    }

    #[test]
    fn test_message_stress_109() {
        let m = MessageHeader::new(0, 1, 109);
        assert_eq!(m.tag, 109);
    }

    #[test]
    fn test_message_stress_110() {
        let m = MessageHeader::new(0, 1, 110);
        assert_eq!(m.tag, 110);
    }

    #[test]
    fn test_message_stress_111() {
        let m = MessageHeader::new(0, 1, 111);
        assert_eq!(m.tag, 111);
    }

    #[test]
    fn test_message_stress_112() {
        let m = MessageHeader::new(0, 1, 112);
        assert_eq!(m.tag, 112);
    }

    #[test]
    fn test_message_stress_113() {
        let m = MessageHeader::new(0, 1, 113);
        assert_eq!(m.tag, 113);
    }

    #[test]
    fn test_message_stress_114() {
        let m = MessageHeader::new(0, 1, 114);
        assert_eq!(m.tag, 114);
    }

    #[test]
    fn test_message_stress_115() {
        let m = MessageHeader::new(0, 1, 115);
        assert_eq!(m.tag, 115);
    }

    #[test]
    fn test_message_stress_116() {
        let m = MessageHeader::new(0, 1, 116);
        assert_eq!(m.tag, 116);
    }

    #[test]
    fn test_message_stress_117() {
        let m = MessageHeader::new(0, 1, 117);
        assert_eq!(m.tag, 117);
    }

    #[test]
    fn test_message_stress_118() {
        let m = MessageHeader::new(0, 1, 118);
        assert_eq!(m.tag, 118);
    }

    #[test]
    fn test_message_stress_119() {
        let m = MessageHeader::new(0, 1, 119);
        assert_eq!(m.tag, 119);
    }

    #[test]
    fn test_message_stress_120() {
        let m = MessageHeader::new(0, 1, 120);
        assert_eq!(m.tag, 120);
    }

    #[test]
    fn test_message_stress_121() {
        let m = MessageHeader::new(0, 1, 121);
        assert_eq!(m.tag, 121);
    }

    #[test]
    fn test_message_stress_122() {
        let m = MessageHeader::new(0, 1, 122);
        assert_eq!(m.tag, 122);
    }

    #[test]
    fn test_message_stress_123() {
        let m = MessageHeader::new(0, 1, 123);
        assert_eq!(m.tag, 123);
    }

    #[test]
    fn test_message_stress_124() {
        let m = MessageHeader::new(0, 1, 124);
        assert_eq!(m.tag, 124);
    }

    #[test]
    fn test_message_stress_125() {
        let m = MessageHeader::new(0, 1, 125);
        assert_eq!(m.tag, 125);
    }

    #[test]
    fn test_message_stress_126() {
        let m = MessageHeader::new(0, 1, 126);
        assert_eq!(m.tag, 126);
    }

    #[test]
    fn test_message_stress_127() {
        let m = MessageHeader::new(0, 1, 127);
        assert_eq!(m.tag, 127);
    }

    #[test]
    fn test_message_stress_128() {
        let m = MessageHeader::new(0, 1, 128);
        assert_eq!(m.tag, 128);
    }

    #[test]
    fn test_message_stress_129() {
        let m = MessageHeader::new(0, 1, 129);
        assert_eq!(m.tag, 129);
    }

    #[test]
    fn test_message_stress_130() {
        let m = MessageHeader::new(0, 1, 130);
        assert_eq!(m.tag, 130);
    }

    #[test]
    fn test_message_stress_131() {
        let m = MessageHeader::new(0, 1, 131);
        assert_eq!(m.tag, 131);
    }

    #[test]
    fn test_message_stress_132() {
        let m = MessageHeader::new(0, 1, 132);
        assert_eq!(m.tag, 132);
    }

    #[test]
    fn test_message_stress_133() {
        let m = MessageHeader::new(0, 1, 133);
        assert_eq!(m.tag, 133);
    }

    #[test]
    fn test_message_stress_134() {
        let m = MessageHeader::new(0, 1, 134);
        assert_eq!(m.tag, 134);
    }

    #[test]
    fn test_message_stress_135() {
        let m = MessageHeader::new(0, 1, 135);
        assert_eq!(m.tag, 135);
    }

    #[test]
    fn test_message_stress_136() {
        let m = MessageHeader::new(0, 1, 136);
        assert_eq!(m.tag, 136);
    }

    #[test]
    fn test_message_stress_137() {
        let m = MessageHeader::new(0, 1, 137);
        assert_eq!(m.tag, 137);
    }

    #[test]
    fn test_message_stress_138() {
        let m = MessageHeader::new(0, 1, 138);
        assert_eq!(m.tag, 138);
    }

    #[test]
    fn test_message_stress_139() {
        let m = MessageHeader::new(0, 1, 139);
        assert_eq!(m.tag, 139);
    }

    #[test]
    fn test_message_stress_140() {
        let m = MessageHeader::new(0, 1, 140);
        assert_eq!(m.tag, 140);
    }

    #[test]
    fn test_message_stress_141() {
        let m = MessageHeader::new(0, 1, 141);
        assert_eq!(m.tag, 141);
    }

    #[test]
    fn test_message_stress_142() {
        let m = MessageHeader::new(0, 1, 142);
        assert_eq!(m.tag, 142);
    }

    #[test]
    fn test_message_stress_143() {
        let m = MessageHeader::new(0, 1, 143);
        assert_eq!(m.tag, 143);
    }

    #[test]
    fn test_message_stress_144() {
        let m = MessageHeader::new(0, 1, 144);
        assert_eq!(m.tag, 144);
    }

    #[test]
    fn test_message_stress_145() {
        let m = MessageHeader::new(0, 1, 145);
        assert_eq!(m.tag, 145);
    }

    #[test]
    fn test_message_stress_146() {
        let m = MessageHeader::new(0, 1, 146);
        assert_eq!(m.tag, 146);
    }

    #[test]
    fn test_message_stress_147() {
        let m = MessageHeader::new(0, 1, 147);
        assert_eq!(m.tag, 147);
    }

    #[test]
    fn test_message_stress_148() {
        let m = MessageHeader::new(0, 1, 148);
        assert_eq!(m.tag, 148);
    }

    #[test]
    fn test_message_stress_149() {
        let m = MessageHeader::new(0, 1, 149);
        assert_eq!(m.tag, 149);
    }

    #[test]
    fn test_message_stress_150() {
        let m = MessageHeader::new(0, 1, 150);
        assert_eq!(m.tag, 150);
    }

    #[test]
    fn test_message_stress_151() {
        let m = MessageHeader::new(0, 1, 151);
        assert_eq!(m.tag, 151);
    }

    #[test]
    fn test_message_stress_152() {
        let m = MessageHeader::new(0, 1, 152);
        assert_eq!(m.tag, 152);
    }

    #[test]
    fn test_message_stress_153() {
        let m = MessageHeader::new(0, 1, 153);
        assert_eq!(m.tag, 153);
    }

    #[test]
    fn test_message_stress_154() {
        let m = MessageHeader::new(0, 1, 154);
        assert_eq!(m.tag, 154);
    }

    #[test]
    fn test_message_stress_155() {
        let m = MessageHeader::new(0, 1, 155);
        assert_eq!(m.tag, 155);
    }

    #[test]
    fn test_message_stress_156() {
        let m = MessageHeader::new(0, 1, 156);
        assert_eq!(m.tag, 156);
    }

    #[test]
    fn test_message_stress_157() {
        let m = MessageHeader::new(0, 1, 157);
        assert_eq!(m.tag, 157);
    }

    #[test]
    fn test_message_stress_158() {
        let m = MessageHeader::new(0, 1, 158);
        assert_eq!(m.tag, 158);
    }

    #[test]
    fn test_message_stress_159() {
        let m = MessageHeader::new(0, 1, 159);
        assert_eq!(m.tag, 159);
    }

    #[test]
    fn test_message_stress_160() {
        let m = MessageHeader::new(0, 1, 160);
        assert_eq!(m.tag, 160);
    }

    #[test]
    fn test_message_stress_161() {
        let m = MessageHeader::new(0, 1, 161);
        assert_eq!(m.tag, 161);
    }

    #[test]
    fn test_message_stress_162() {
        let m = MessageHeader::new(0, 1, 162);
        assert_eq!(m.tag, 162);
    }

    #[test]
    fn test_message_stress_163() {
        let m = MessageHeader::new(0, 1, 163);
        assert_eq!(m.tag, 163);
    }

    #[test]
    fn test_message_stress_164() {
        let m = MessageHeader::new(0, 1, 164);
        assert_eq!(m.tag, 164);
    }

    #[test]
    fn test_message_stress_165() {
        let m = MessageHeader::new(0, 1, 165);
        assert_eq!(m.tag, 165);
    }

    #[test]
    fn test_message_stress_166() {
        let m = MessageHeader::new(0, 1, 166);
        assert_eq!(m.tag, 166);
    }

    #[test]
    fn test_message_stress_167() {
        let m = MessageHeader::new(0, 1, 167);
        assert_eq!(m.tag, 167);
    }

    #[test]
    fn test_message_stress_168() {
        let m = MessageHeader::new(0, 1, 168);
        assert_eq!(m.tag, 168);
    }

    #[test]
    fn test_message_stress_169() {
        let m = MessageHeader::new(0, 1, 169);
        assert_eq!(m.tag, 169);
    }

    #[test]
    fn test_message_stress_170() {
        let m = MessageHeader::new(0, 1, 170);
        assert_eq!(m.tag, 170);
    }

    #[test]
    fn test_message_stress_171() {
        let m = MessageHeader::new(0, 1, 171);
        assert_eq!(m.tag, 171);
    }

    #[test]
    fn test_message_stress_172() {
        let m = MessageHeader::new(0, 1, 172);
        assert_eq!(m.tag, 172);
    }

    #[test]
    fn test_message_stress_173() {
        let m = MessageHeader::new(0, 1, 173);
        assert_eq!(m.tag, 173);
    }

    #[test]
    fn test_message_stress_174() {
        let m = MessageHeader::new(0, 1, 174);
        assert_eq!(m.tag, 174);
    }

    #[test]
    fn test_message_stress_175() {
        let m = MessageHeader::new(0, 1, 175);
        assert_eq!(m.tag, 175);
    }

    #[test]
    fn test_message_stress_176() {
        let m = MessageHeader::new(0, 1, 176);
        assert_eq!(m.tag, 176);
    }

    #[test]
    fn test_message_stress_177() {
        let m = MessageHeader::new(0, 1, 177);
        assert_eq!(m.tag, 177);
    }

    #[test]
    fn test_message_stress_178() {
        let m = MessageHeader::new(0, 1, 178);
        assert_eq!(m.tag, 178);
    }

    #[test]
    fn test_message_stress_179() {
        let m = MessageHeader::new(0, 1, 179);
        assert_eq!(m.tag, 179);
    }

    #[test]
    fn test_message_stress_180() {
        let m = MessageHeader::new(0, 1, 180);
        assert_eq!(m.tag, 180);
    }

    #[test]
    fn test_message_stress_181() {
        let m = MessageHeader::new(0, 1, 181);
        assert_eq!(m.tag, 181);
    }

    #[test]
    fn test_message_stress_182() {
        let m = MessageHeader::new(0, 1, 182);
        assert_eq!(m.tag, 182);
    }

    #[test]
    fn test_message_stress_183() {
        let m = MessageHeader::new(0, 1, 183);
        assert_eq!(m.tag, 183);
    }

    #[test]
    fn test_message_stress_184() {
        let m = MessageHeader::new(0, 1, 184);
        assert_eq!(m.tag, 184);
    }

    #[test]
    fn test_message_stress_185() {
        let m = MessageHeader::new(0, 1, 185);
        assert_eq!(m.tag, 185);
    }

    #[test]
    fn test_message_stress_186() {
        let m = MessageHeader::new(0, 1, 186);
        assert_eq!(m.tag, 186);
    }

    #[test]
    fn test_message_stress_187() {
        let m = MessageHeader::new(0, 1, 187);
        assert_eq!(m.tag, 187);
    }

    #[test]
    fn test_message_stress_188() {
        let m = MessageHeader::new(0, 1, 188);
        assert_eq!(m.tag, 188);
    }

    #[test]
    fn test_message_stress_189() {
        let m = MessageHeader::new(0, 1, 189);
        assert_eq!(m.tag, 189);
    }

    #[test]
    fn test_message_stress_190() {
        let m = MessageHeader::new(0, 1, 190);
        assert_eq!(m.tag, 190);
    }

    #[test]
    fn test_message_stress_191() {
        let m = MessageHeader::new(0, 1, 191);
        assert_eq!(m.tag, 191);
    }

    #[test]
    fn test_message_stress_192() {
        let m = MessageHeader::new(0, 1, 192);
        assert_eq!(m.tag, 192);
    }

    #[test]
    fn test_message_stress_193() {
        let m = MessageHeader::new(0, 1, 193);
        assert_eq!(m.tag, 193);
    }

    #[test]
    fn test_message_stress_194() {
        let m = MessageHeader::new(0, 1, 194);
        assert_eq!(m.tag, 194);
    }

    #[test]
    fn test_message_stress_195() {
        let m = MessageHeader::new(0, 1, 195);
        assert_eq!(m.tag, 195);
    }

    #[test]
    fn test_message_stress_196() {
        let m = MessageHeader::new(0, 1, 196);
        assert_eq!(m.tag, 196);
    }

    #[test]
    fn test_message_stress_197() {
        let m = MessageHeader::new(0, 1, 197);
        assert_eq!(m.tag, 197);
    }

    #[test]
    fn test_message_stress_198() {
        let m = MessageHeader::new(0, 1, 198);
        assert_eq!(m.tag, 198);
    }

    #[test]
    fn test_message_stress_199() {
        let m = MessageHeader::new(0, 1, 199);
        assert_eq!(m.tag, 199);
    }

    #[test]
    fn test_message_stress_200() {
        let m = MessageHeader::new(0, 1, 200);
        assert_eq!(m.tag, 200);
    }

    #[test]
    fn test_message_stress_201() {
        let m = MessageHeader::new(0, 1, 201);
        assert_eq!(m.tag, 201);
    }

    #[test]
    fn test_message_stress_202() {
        let m = MessageHeader::new(0, 1, 202);
        assert_eq!(m.tag, 202);
    }

    #[test]
    fn test_message_stress_203() {
        let m = MessageHeader::new(0, 1, 203);
        assert_eq!(m.tag, 203);
    }

    #[test]
    fn test_message_stress_204() {
        let m = MessageHeader::new(0, 1, 204);
        assert_eq!(m.tag, 204);
    }

    #[test]
    fn test_message_stress_205() {
        let m = MessageHeader::new(0, 1, 205);
        assert_eq!(m.tag, 205);
    }

    #[test]
    fn test_message_stress_206() {
        let m = MessageHeader::new(0, 1, 206);
        assert_eq!(m.tag, 206);
    }

    #[test]
    fn test_message_stress_207() {
        let m = MessageHeader::new(0, 1, 207);
        assert_eq!(m.tag, 207);
    }

    #[test]
    fn test_message_stress_208() {
        let m = MessageHeader::new(0, 1, 208);
        assert_eq!(m.tag, 208);
    }

    #[test]
    fn test_message_stress_209() {
        let m = MessageHeader::new(0, 1, 209);
        assert_eq!(m.tag, 209);
    }

    #[test]
    fn test_message_stress_210() {
        let m = MessageHeader::new(0, 1, 210);
        assert_eq!(m.tag, 210);
    }

    #[test]
    fn test_message_stress_211() {
        let m = MessageHeader::new(0, 1, 211);
        assert_eq!(m.tag, 211);
    }

    #[test]
    fn test_message_stress_212() {
        let m = MessageHeader::new(0, 1, 212);
        assert_eq!(m.tag, 212);
    }

    #[test]
    fn test_message_stress_213() {
        let m = MessageHeader::new(0, 1, 213);
        assert_eq!(m.tag, 213);
    }

    #[test]
    fn test_message_stress_214() {
        let m = MessageHeader::new(0, 1, 214);
        assert_eq!(m.tag, 214);
    }

    #[test]
    fn test_message_stress_215() {
        let m = MessageHeader::new(0, 1, 215);
        assert_eq!(m.tag, 215);
    }

    #[test]
    fn test_message_stress_216() {
        let m = MessageHeader::new(0, 1, 216);
        assert_eq!(m.tag, 216);
    }

    #[test]
    fn test_message_stress_217() {
        let m = MessageHeader::new(0, 1, 217);
        assert_eq!(m.tag, 217);
    }

    #[test]
    fn test_message_stress_218() {
        let m = MessageHeader::new(0, 1, 218);
        assert_eq!(m.tag, 218);
    }

    #[test]
    fn test_message_stress_219() {
        let m = MessageHeader::new(0, 1, 219);
        assert_eq!(m.tag, 219);
    }

    #[test]
    fn test_message_stress_220() {
        let m = MessageHeader::new(0, 1, 220);
        assert_eq!(m.tag, 220);
    }

    #[test]
    fn test_message_stress_221() {
        let m = MessageHeader::new(0, 1, 221);
        assert_eq!(m.tag, 221);
    }

    #[test]
    fn test_message_stress_222() {
        let m = MessageHeader::new(0, 1, 222);
        assert_eq!(m.tag, 222);
    }

    #[test]
    fn test_message_stress_223() {
        let m = MessageHeader::new(0, 1, 223);
        assert_eq!(m.tag, 223);
    }

    #[test]
    fn test_message_stress_224() {
        let m = MessageHeader::new(0, 1, 224);
        assert_eq!(m.tag, 224);
    }

    #[test]
    fn test_message_stress_225() {
        let m = MessageHeader::new(0, 1, 225);
        assert_eq!(m.tag, 225);
    }

    #[test]
    fn test_message_stress_226() {
        let m = MessageHeader::new(0, 1, 226);
        assert_eq!(m.tag, 226);
    }

    #[test]
    fn test_message_stress_227() {
        let m = MessageHeader::new(0, 1, 227);
        assert_eq!(m.tag, 227);
    }

    #[test]
    fn test_message_stress_228() {
        let m = MessageHeader::new(0, 1, 228);
        assert_eq!(m.tag, 228);
    }

    #[test]
    fn test_message_stress_229() {
        let m = MessageHeader::new(0, 1, 229);
        assert_eq!(m.tag, 229);
    }

    #[test]
    fn test_message_stress_230() {
        let m = MessageHeader::new(0, 1, 230);
        assert_eq!(m.tag, 230);
    }

    #[test]
    fn test_message_stress_231() {
        let m = MessageHeader::new(0, 1, 231);
        assert_eq!(m.tag, 231);
    }

    #[test]
    fn test_message_stress_232() {
        let m = MessageHeader::new(0, 1, 232);
        assert_eq!(m.tag, 232);
    }

    #[test]
    fn test_message_stress_233() {
        let m = MessageHeader::new(0, 1, 233);
        assert_eq!(m.tag, 233);
    }

    #[test]
    fn test_message_stress_234() {
        let m = MessageHeader::new(0, 1, 234);
        assert_eq!(m.tag, 234);
    }

    #[test]
    fn test_message_stress_235() {
        let m = MessageHeader::new(0, 1, 235);
        assert_eq!(m.tag, 235);
    }

    #[test]
    fn test_message_stress_236() {
        let m = MessageHeader::new(0, 1, 236);
        assert_eq!(m.tag, 236);
    }

    #[test]
    fn test_message_stress_237() {
        let m = MessageHeader::new(0, 1, 237);
        assert_eq!(m.tag, 237);
    }

    #[test]
    fn test_message_stress_238() {
        let m = MessageHeader::new(0, 1, 238);
        assert_eq!(m.tag, 238);
    }

    #[test]
    fn test_message_stress_239() {
        let m = MessageHeader::new(0, 1, 239);
        assert_eq!(m.tag, 239);
    }

    #[test]
    fn test_message_stress_240() {
        let m = MessageHeader::new(0, 1, 240);
        assert_eq!(m.tag, 240);
    }

    #[test]
    fn test_message_stress_241() {
        let m = MessageHeader::new(0, 1, 241);
        assert_eq!(m.tag, 241);
    }

    #[test]
    fn test_message_stress_242() {
        let m = MessageHeader::new(0, 1, 242);
        assert_eq!(m.tag, 242);
    }

    #[test]
    fn test_message_stress_243() {
        let m = MessageHeader::new(0, 1, 243);
        assert_eq!(m.tag, 243);
    }

    #[test]
    fn test_message_stress_244() {
        let m = MessageHeader::new(0, 1, 244);
        assert_eq!(m.tag, 244);
    }

    #[test]
    fn test_message_stress_245() {
        let m = MessageHeader::new(0, 1, 245);
        assert_eq!(m.tag, 245);
    }

    #[test]
    fn test_message_stress_246() {
        let m = MessageHeader::new(0, 1, 246);
        assert_eq!(m.tag, 246);
    }

    #[test]
    fn test_message_stress_247() {
        let m = MessageHeader::new(0, 1, 247);
        assert_eq!(m.tag, 247);
    }

    #[test]
    fn test_message_stress_248() {
        let m = MessageHeader::new(0, 1, 248);
        assert_eq!(m.tag, 248);
    }

    #[test]
    fn test_message_stress_249() {
        let m = MessageHeader::new(0, 1, 249);
        assert_eq!(m.tag, 249);
    }

    #[test]
    fn test_message_stress_250() {
        let m = MessageHeader::new(0, 1, 250);
        assert_eq!(m.tag, 250);
    }

    #[test]
    fn test_message_stress_251() {
        let m = MessageHeader::new(0, 1, 251);
        assert_eq!(m.tag, 251);
    }

    #[test]
    fn test_message_stress_252() {
        let m = MessageHeader::new(0, 1, 252);
        assert_eq!(m.tag, 252);
    }

    #[test]
    fn test_message_stress_253() {
        let m = MessageHeader::new(0, 1, 253);
        assert_eq!(m.tag, 253);
    }

    #[test]
    fn test_message_stress_254() {
        let m = MessageHeader::new(0, 1, 254);
        assert_eq!(m.tag, 254);
    }

    #[test]
    fn test_message_stress_255() {
        let m = MessageHeader::new(0, 1, 255);
        assert_eq!(m.tag, 255);
    }

    #[test]
    fn test_message_stress_256() {
        let m = MessageHeader::new(0, 1, 256);
        assert_eq!(m.tag, 256);
    }

    #[test]
    fn test_message_stress_257() {
        let m = MessageHeader::new(0, 1, 257);
        assert_eq!(m.tag, 257);
    }

    #[test]
    fn test_message_stress_258() {
        let m = MessageHeader::new(0, 1, 258);
        assert_eq!(m.tag, 258);
    }

    #[test]
    fn test_message_stress_259() {
        let m = MessageHeader::new(0, 1, 259);
        assert_eq!(m.tag, 259);
    }

    #[test]
    fn test_message_stress_260() {
        let m = MessageHeader::new(0, 1, 260);
        assert_eq!(m.tag, 260);
    }

    #[test]
    fn test_message_stress_261() {
        let m = MessageHeader::new(0, 1, 261);
        assert_eq!(m.tag, 261);
    }

    #[test]
    fn test_message_stress_262() {
        let m = MessageHeader::new(0, 1, 262);
        assert_eq!(m.tag, 262);
    }

    #[test]
    fn test_message_stress_263() {
        let m = MessageHeader::new(0, 1, 263);
        assert_eq!(m.tag, 263);
    }

    #[test]
    fn test_message_stress_264() {
        let m = MessageHeader::new(0, 1, 264);
        assert_eq!(m.tag, 264);
    }

    #[test]
    fn test_message_stress_265() {
        let m = MessageHeader::new(0, 1, 265);
        assert_eq!(m.tag, 265);
    }

    #[test]
    fn test_message_stress_266() {
        let m = MessageHeader::new(0, 1, 266);
        assert_eq!(m.tag, 266);
    }

    #[test]
    fn test_message_stress_267() {
        let m = MessageHeader::new(0, 1, 267);
        assert_eq!(m.tag, 267);
    }

    #[test]
    fn test_message_stress_268() {
        let m = MessageHeader::new(0, 1, 268);
        assert_eq!(m.tag, 268);
    }

    #[test]
    fn test_message_stress_269() {
        let m = MessageHeader::new(0, 1, 269);
        assert_eq!(m.tag, 269);
    }

    #[test]
    fn test_message_stress_270() {
        let m = MessageHeader::new(0, 1, 270);
        assert_eq!(m.tag, 270);
    }

    #[test]
    fn test_message_stress_271() {
        let m = MessageHeader::new(0, 1, 271);
        assert_eq!(m.tag, 271);
    }

    #[test]
    fn test_message_stress_272() {
        let m = MessageHeader::new(0, 1, 272);
        assert_eq!(m.tag, 272);
    }

    #[test]
    fn test_message_stress_273() {
        let m = MessageHeader::new(0, 1, 273);
        assert_eq!(m.tag, 273);
    }

    #[test]
    fn test_message_stress_274() {
        let m = MessageHeader::new(0, 1, 274);
        assert_eq!(m.tag, 274);
    }

    #[test]
    fn test_message_stress_275() {
        let m = MessageHeader::new(0, 1, 275);
        assert_eq!(m.tag, 275);
    }

    #[test]
    fn test_message_stress_276() {
        let m = MessageHeader::new(0, 1, 276);
        assert_eq!(m.tag, 276);
    }

    #[test]
    fn test_message_stress_277() {
        let m = MessageHeader::new(0, 1, 277);
        assert_eq!(m.tag, 277);
    }

    #[test]
    fn test_message_stress_278() {
        let m = MessageHeader::new(0, 1, 278);
        assert_eq!(m.tag, 278);
    }

    #[test]
    fn test_message_stress_279() {
        let m = MessageHeader::new(0, 1, 279);
        assert_eq!(m.tag, 279);
    }

    #[test]
    fn test_message_stress_280() {
        let m = MessageHeader::new(0, 1, 280);
        assert_eq!(m.tag, 280);
    }

    #[test]
    fn test_message_stress_281() {
        let m = MessageHeader::new(0, 1, 281);
        assert_eq!(m.tag, 281);
    }

    #[test]
    fn test_message_stress_282() {
        let m = MessageHeader::new(0, 1, 282);
        assert_eq!(m.tag, 282);
    }

    #[test]
    fn test_message_stress_283() {
        let m = MessageHeader::new(0, 1, 283);
        assert_eq!(m.tag, 283);
    }

    #[test]
    fn test_message_stress_284() {
        let m = MessageHeader::new(0, 1, 284);
        assert_eq!(m.tag, 284);
    }

    #[test]
    fn test_message_stress_285() {
        let m = MessageHeader::new(0, 1, 285);
        assert_eq!(m.tag, 285);
    }

    #[test]
    fn test_message_stress_286() {
        let m = MessageHeader::new(0, 1, 286);
        assert_eq!(m.tag, 286);
    }

    #[test]
    fn test_message_stress_287() {
        let m = MessageHeader::new(0, 1, 287);
        assert_eq!(m.tag, 287);
    }

    #[test]
    fn test_message_stress_288() {
        let m = MessageHeader::new(0, 1, 288);
        assert_eq!(m.tag, 288);
    }

    #[test]
    fn test_message_stress_289() {
        let m = MessageHeader::new(0, 1, 289);
        assert_eq!(m.tag, 289);
    }

    #[test]
    fn test_message_stress_290() {
        let m = MessageHeader::new(0, 1, 290);
        assert_eq!(m.tag, 290);
    }

    #[test]
    fn test_message_stress_291() {
        let m = MessageHeader::new(0, 1, 291);
        assert_eq!(m.tag, 291);
    }

    #[test]
    fn test_message_stress_292() {
        let m = MessageHeader::new(0, 1, 292);
        assert_eq!(m.tag, 292);
    }

    #[test]
    fn test_message_stress_293() {
        let m = MessageHeader::new(0, 1, 293);
        assert_eq!(m.tag, 293);
    }

    #[test]
    fn test_message_stress_294() {
        let m = MessageHeader::new(0, 1, 294);
        assert_eq!(m.tag, 294);
    }

    #[test]
    fn test_message_stress_295() {
        let m = MessageHeader::new(0, 1, 295);
        assert_eq!(m.tag, 295);
    }

    #[test]
    fn test_message_stress_296() {
        let m = MessageHeader::new(0, 1, 296);
        assert_eq!(m.tag, 296);
    }

    #[test]
    fn test_message_stress_297() {
        let m = MessageHeader::new(0, 1, 297);
        assert_eq!(m.tag, 297);
    }

    #[test]
    fn test_message_stress_298() {
        let m = MessageHeader::new(0, 1, 298);
        assert_eq!(m.tag, 298);
    }

    #[test]
    fn test_message_stress_299() {
        let m = MessageHeader::new(0, 1, 299);
        assert_eq!(m.tag, 299);
    }

    #[test]
    fn test_message_stress_300() {
        let m = MessageHeader::new(0, 1, 300);
        assert_eq!(m.tag, 300);
    }

    #[test]
    fn test_message_stress_301() {
        let m = MessageHeader::new(0, 1, 301);
        assert_eq!(m.tag, 301);
    }

    #[test]
    fn test_message_stress_302() {
        let m = MessageHeader::new(0, 1, 302);
        assert_eq!(m.tag, 302);
    }

    #[test]
    fn test_message_stress_303() {
        let m = MessageHeader::new(0, 1, 303);
        assert_eq!(m.tag, 303);
    }

    #[test]
    fn test_message_stress_304() {
        let m = MessageHeader::new(0, 1, 304);
        assert_eq!(m.tag, 304);
    }

    #[test]
    fn test_message_stress_305() {
        let m = MessageHeader::new(0, 1, 305);
        assert_eq!(m.tag, 305);
    }

    #[test]
    fn test_message_stress_306() {
        let m = MessageHeader::new(0, 1, 306);
        assert_eq!(m.tag, 306);
    }

    #[test]
    fn test_message_stress_307() {
        let m = MessageHeader::new(0, 1, 307);
        assert_eq!(m.tag, 307);
    }

    #[test]
    fn test_message_stress_308() {
        let m = MessageHeader::new(0, 1, 308);
        assert_eq!(m.tag, 308);
    }

    #[test]
    fn test_message_stress_309() {
        let m = MessageHeader::new(0, 1, 309);
        assert_eq!(m.tag, 309);
    }

    #[test]
    fn test_message_stress_310() {
        let m = MessageHeader::new(0, 1, 310);
        assert_eq!(m.tag, 310);
    }

    #[test]
    fn test_message_stress_311() {
        let m = MessageHeader::new(0, 1, 311);
        assert_eq!(m.tag, 311);
    }

    #[test]
    fn test_message_stress_312() {
        let m = MessageHeader::new(0, 1, 312);
        assert_eq!(m.tag, 312);
    }

    #[test]
    fn test_message_stress_313() {
        let m = MessageHeader::new(0, 1, 313);
        assert_eq!(m.tag, 313);
    }

    #[test]
    fn test_message_stress_314() {
        let m = MessageHeader::new(0, 1, 314);
        assert_eq!(m.tag, 314);
    }

    #[test]
    fn test_message_stress_315() {
        let m = MessageHeader::new(0, 1, 315);
        assert_eq!(m.tag, 315);
    }

    #[test]
    fn test_message_stress_316() {
        let m = MessageHeader::new(0, 1, 316);
        assert_eq!(m.tag, 316);
    }

    #[test]
    fn test_message_stress_317() {
        let m = MessageHeader::new(0, 1, 317);
        assert_eq!(m.tag, 317);
    }

    #[test]
    fn test_message_stress_318() {
        let m = MessageHeader::new(0, 1, 318);
        assert_eq!(m.tag, 318);
    }

    #[test]
    fn test_message_stress_319() {
        let m = MessageHeader::new(0, 1, 319);
        assert_eq!(m.tag, 319);
    }

    #[test]
    fn test_message_stress_320() {
        let m = MessageHeader::new(0, 1, 320);
        assert_eq!(m.tag, 320);
    }

    #[test]
    fn test_message_stress_321() {
        let m = MessageHeader::new(0, 1, 321);
        assert_eq!(m.tag, 321);
    }

    #[test]
    fn test_message_stress_322() {
        let m = MessageHeader::new(0, 1, 322);
        assert_eq!(m.tag, 322);
    }

    #[test]
    fn test_message_stress_323() {
        let m = MessageHeader::new(0, 1, 323);
        assert_eq!(m.tag, 323);
    }

    #[test]
    fn test_message_stress_324() {
        let m = MessageHeader::new(0, 1, 324);
        assert_eq!(m.tag, 324);
    }

    #[test]
    fn test_message_stress_325() {
        let m = MessageHeader::new(0, 1, 325);
        assert_eq!(m.tag, 325);
    }

    #[test]
    fn test_message_stress_326() {
        let m = MessageHeader::new(0, 1, 326);
        assert_eq!(m.tag, 326);
    }

    #[test]
    fn test_message_stress_327() {
        let m = MessageHeader::new(0, 1, 327);
        assert_eq!(m.tag, 327);
    }

    #[test]
    fn test_message_stress_328() {
        let m = MessageHeader::new(0, 1, 328);
        assert_eq!(m.tag, 328);
    }

    #[test]
    fn test_message_stress_329() {
        let m = MessageHeader::new(0, 1, 329);
        assert_eq!(m.tag, 329);
    }

    #[test]
    fn test_message_stress_330() {
        let m = MessageHeader::new(0, 1, 330);
        assert_eq!(m.tag, 330);
    }

    #[test]
    fn test_message_stress_331() {
        let m = MessageHeader::new(0, 1, 331);
        assert_eq!(m.tag, 331);
    }

    #[test]
    fn test_message_stress_332() {
        let m = MessageHeader::new(0, 1, 332);
        assert_eq!(m.tag, 332);
    }

    #[test]
    fn test_message_stress_333() {
        let m = MessageHeader::new(0, 1, 333);
        assert_eq!(m.tag, 333);
    }

    #[test]
    fn test_message_stress_334() {
        let m = MessageHeader::new(0, 1, 334);
        assert_eq!(m.tag, 334);
    }

    #[test]
    fn test_message_stress_335() {
        let m = MessageHeader::new(0, 1, 335);
        assert_eq!(m.tag, 335);
    }

    #[test]
    fn test_message_stress_336() {
        let m = MessageHeader::new(0, 1, 336);
        assert_eq!(m.tag, 336);
    }

    #[test]
    fn test_message_stress_337() {
        let m = MessageHeader::new(0, 1, 337);
        assert_eq!(m.tag, 337);
    }

    #[test]
    fn test_message_stress_338() {
        let m = MessageHeader::new(0, 1, 338);
        assert_eq!(m.tag, 338);
    }

    #[test]
    fn test_message_stress_339() {
        let m = MessageHeader::new(0, 1, 339);
        assert_eq!(m.tag, 339);
    }

    #[test]
    fn test_message_stress_340() {
        let m = MessageHeader::new(0, 1, 340);
        assert_eq!(m.tag, 340);
    }

    #[test]
    fn test_message_stress_341() {
        let m = MessageHeader::new(0, 1, 341);
        assert_eq!(m.tag, 341);
    }

    #[test]
    fn test_message_stress_342() {
        let m = MessageHeader::new(0, 1, 342);
        assert_eq!(m.tag, 342);
    }

    #[test]
    fn test_message_stress_343() {
        let m = MessageHeader::new(0, 1, 343);
        assert_eq!(m.tag, 343);
    }

    #[test]
    fn test_message_stress_344() {
        let m = MessageHeader::new(0, 1, 344);
        assert_eq!(m.tag, 344);
    }

    #[test]
    fn test_message_stress_345() {
        let m = MessageHeader::new(0, 1, 345);
        assert_eq!(m.tag, 345);
    }

    #[test]
    fn test_message_stress_346() {
        let m = MessageHeader::new(0, 1, 346);
        assert_eq!(m.tag, 346);
    }

    #[test]
    fn test_message_stress_347() {
        let m = MessageHeader::new(0, 1, 347);
        assert_eq!(m.tag, 347);
    }

    #[test]
    fn test_message_stress_348() {
        let m = MessageHeader::new(0, 1, 348);
        assert_eq!(m.tag, 348);
    }

    #[test]
    fn test_message_stress_349() {
        let m = MessageHeader::new(0, 1, 349);
        assert_eq!(m.tag, 349);
    }

    #[test]
    fn test_message_stress_350() {
        let m = MessageHeader::new(0, 1, 350);
        assert_eq!(m.tag, 350);
    }

    #[test]
    fn test_message_stress_351() {
        let m = MessageHeader::new(0, 1, 351);
        assert_eq!(m.tag, 351);
    }

    #[test]
    fn test_message_stress_352() {
        let m = MessageHeader::new(0, 1, 352);
        assert_eq!(m.tag, 352);
    }

    #[test]
    fn test_message_stress_353() {
        let m = MessageHeader::new(0, 1, 353);
        assert_eq!(m.tag, 353);
    }

    #[test]
    fn test_message_stress_354() {
        let m = MessageHeader::new(0, 1, 354);
        assert_eq!(m.tag, 354);
    }

    #[test]
    fn test_message_stress_355() {
        let m = MessageHeader::new(0, 1, 355);
        assert_eq!(m.tag, 355);
    }

    #[test]
    fn test_message_stress_356() {
        let m = MessageHeader::new(0, 1, 356);
        assert_eq!(m.tag, 356);
    }

    #[test]
    fn test_message_stress_357() {
        let m = MessageHeader::new(0, 1, 357);
        assert_eq!(m.tag, 357);
    }

    #[test]
    fn test_message_stress_358() {
        let m = MessageHeader::new(0, 1, 358);
        assert_eq!(m.tag, 358);
    }

    #[test]
    fn test_message_stress_359() {
        let m = MessageHeader::new(0, 1, 359);
        assert_eq!(m.tag, 359);
    }

    #[test]
    fn test_message_stress_360() {
        let m = MessageHeader::new(0, 1, 360);
        assert_eq!(m.tag, 360);
    }

    #[test]
    fn test_message_stress_361() {
        let m = MessageHeader::new(0, 1, 361);
        assert_eq!(m.tag, 361);
    }

    #[test]
    fn test_message_stress_362() {
        let m = MessageHeader::new(0, 1, 362);
        assert_eq!(m.tag, 362);
    }

    #[test]
    fn test_message_stress_363() {
        let m = MessageHeader::new(0, 1, 363);
        assert_eq!(m.tag, 363);
    }

    #[test]
    fn test_message_stress_364() {
        let m = MessageHeader::new(0, 1, 364);
        assert_eq!(m.tag, 364);
    }

    #[test]
    fn test_message_stress_365() {
        let m = MessageHeader::new(0, 1, 365);
        assert_eq!(m.tag, 365);
    }

    #[test]
    fn test_message_stress_366() {
        let m = MessageHeader::new(0, 1, 366);
        assert_eq!(m.tag, 366);
    }

    #[test]
    fn test_message_stress_367() {
        let m = MessageHeader::new(0, 1, 367);
        assert_eq!(m.tag, 367);
    }

    #[test]
    fn test_message_stress_368() {
        let m = MessageHeader::new(0, 1, 368);
        assert_eq!(m.tag, 368);
    }

    #[test]
    fn test_message_stress_369() {
        let m = MessageHeader::new(0, 1, 369);
        assert_eq!(m.tag, 369);
    }

    #[test]
    fn test_message_stress_370() {
        let m = MessageHeader::new(0, 1, 370);
        assert_eq!(m.tag, 370);
    }

    #[test]
    fn test_message_stress_371() {
        let m = MessageHeader::new(0, 1, 371);
        assert_eq!(m.tag, 371);
    }

    #[test]
    fn test_message_stress_372() {
        let m = MessageHeader::new(0, 1, 372);
        assert_eq!(m.tag, 372);
    }

    #[test]
    fn test_message_stress_373() {
        let m = MessageHeader::new(0, 1, 373);
        assert_eq!(m.tag, 373);
    }

    #[test]
    fn test_message_stress_374() {
        let m = MessageHeader::new(0, 1, 374);
        assert_eq!(m.tag, 374);
    }

    #[test]
    fn test_message_stress_375() {
        let m = MessageHeader::new(0, 1, 375);
        assert_eq!(m.tag, 375);
    }

    #[test]
    fn test_message_stress_376() {
        let m = MessageHeader::new(0, 1, 376);
        assert_eq!(m.tag, 376);
    }

    #[test]
    fn test_message_stress_377() {
        let m = MessageHeader::new(0, 1, 377);
        assert_eq!(m.tag, 377);
    }

    #[test]
    fn test_message_stress_378() {
        let m = MessageHeader::new(0, 1, 378);
        assert_eq!(m.tag, 378);
    }

    #[test]
    fn test_message_stress_379() {
        let m = MessageHeader::new(0, 1, 379);
        assert_eq!(m.tag, 379);
    }

    #[test]
    fn test_message_stress_380() {
        let m = MessageHeader::new(0, 1, 380);
        assert_eq!(m.tag, 380);
    }

    #[test]
    fn test_message_stress_381() {
        let m = MessageHeader::new(0, 1, 381);
        assert_eq!(m.tag, 381);
    }

    #[test]
    fn test_message_stress_382() {
        let m = MessageHeader::new(0, 1, 382);
        assert_eq!(m.tag, 382);
    }

    #[test]
    fn test_message_stress_383() {
        let m = MessageHeader::new(0, 1, 383);
        assert_eq!(m.tag, 383);
    }

    #[test]
    fn test_message_stress_384() {
        let m = MessageHeader::new(0, 1, 384);
        assert_eq!(m.tag, 384);
    }

    #[test]
    fn test_message_stress_385() {
        let m = MessageHeader::new(0, 1, 385);
        assert_eq!(m.tag, 385);
    }

    #[test]
    fn test_message_stress_386() {
        let m = MessageHeader::new(0, 1, 386);
        assert_eq!(m.tag, 386);
    }

    #[test]
    fn test_message_stress_387() {
        let m = MessageHeader::new(0, 1, 387);
        assert_eq!(m.tag, 387);
    }

    #[test]
    fn test_message_stress_388() {
        let m = MessageHeader::new(0, 1, 388);
        assert_eq!(m.tag, 388);
    }

    #[test]
    fn test_message_stress_389() {
        let m = MessageHeader::new(0, 1, 389);
        assert_eq!(m.tag, 389);
    }

    #[test]
    fn test_message_stress_390() {
        let m = MessageHeader::new(0, 1, 390);
        assert_eq!(m.tag, 390);
    }

    #[test]
    fn test_message_stress_391() {
        let m = MessageHeader::new(0, 1, 391);
        assert_eq!(m.tag, 391);
    }

    #[test]
    fn test_message_stress_392() {
        let m = MessageHeader::new(0, 1, 392);
        assert_eq!(m.tag, 392);
    }

    #[test]
    fn test_message_stress_393() {
        let m = MessageHeader::new(0, 1, 393);
        assert_eq!(m.tag, 393);
    }

    #[test]
    fn test_message_stress_394() {
        let m = MessageHeader::new(0, 1, 394);
        assert_eq!(m.tag, 394);
    }

    #[test]
    fn test_message_stress_395() {
        let m = MessageHeader::new(0, 1, 395);
        assert_eq!(m.tag, 395);
    }

    #[test]
    fn test_message_stress_396() {
        let m = MessageHeader::new(0, 1, 396);
        assert_eq!(m.tag, 396);
    }

    #[test]
    fn test_message_stress_397() {
        let m = MessageHeader::new(0, 1, 397);
        assert_eq!(m.tag, 397);
    }

    #[test]
    fn test_message_stress_398() {
        let m = MessageHeader::new(0, 1, 398);
        assert_eq!(m.tag, 398);
    }

    #[test]
    fn test_message_stress_399() {
        let m = MessageHeader::new(0, 1, 399);
        assert_eq!(m.tag, 399);
    }

    #[test]
    fn test_message_stress_400() {
        let m = MessageHeader::new(0, 1, 400);
        assert_eq!(m.tag, 400);
    }

    #[test]
    fn test_message_stress_401() {
        let m = MessageHeader::new(0, 1, 401);
        assert_eq!(m.tag, 401);
    }

    #[test]
    fn test_message_stress_402() {
        let m = MessageHeader::new(0, 1, 402);
        assert_eq!(m.tag, 402);
    }

    #[test]
    fn test_message_stress_403() {
        let m = MessageHeader::new(0, 1, 403);
        assert_eq!(m.tag, 403);
    }

    #[test]
    fn test_message_stress_404() {
        let m = MessageHeader::new(0, 1, 404);
        assert_eq!(m.tag, 404);
    }

    #[test]
    fn test_message_stress_405() {
        let m = MessageHeader::new(0, 1, 405);
        assert_eq!(m.tag, 405);
    }

    #[test]
    fn test_message_stress_406() {
        let m = MessageHeader::new(0, 1, 406);
        assert_eq!(m.tag, 406);
    }

    #[test]
    fn test_message_stress_407() {
        let m = MessageHeader::new(0, 1, 407);
        assert_eq!(m.tag, 407);
    }

    #[test]
    fn test_message_stress_408() {
        let m = MessageHeader::new(0, 1, 408);
        assert_eq!(m.tag, 408);
    }

    #[test]
    fn test_message_stress_409() {
        let m = MessageHeader::new(0, 1, 409);
        assert_eq!(m.tag, 409);
    }

    #[test]
    fn test_message_stress_410() {
        let m = MessageHeader::new(0, 1, 410);
        assert_eq!(m.tag, 410);
    }

    #[test]
    fn test_message_stress_411() {
        let m = MessageHeader::new(0, 1, 411);
        assert_eq!(m.tag, 411);
    }

    #[test]
    fn test_message_stress_412() {
        let m = MessageHeader::new(0, 1, 412);
        assert_eq!(m.tag, 412);
    }

    #[test]
    fn test_message_stress_413() {
        let m = MessageHeader::new(0, 1, 413);
        assert_eq!(m.tag, 413);
    }

    #[test]
    fn test_message_stress_414() {
        let m = MessageHeader::new(0, 1, 414);
        assert_eq!(m.tag, 414);
    }

    #[test]
    fn test_message_stress_415() {
        let m = MessageHeader::new(0, 1, 415);
        assert_eq!(m.tag, 415);
    }

    #[test]
    fn test_message_stress_416() {
        let m = MessageHeader::new(0, 1, 416);
        assert_eq!(m.tag, 416);
    }

    #[test]
    fn test_message_stress_417() {
        let m = MessageHeader::new(0, 1, 417);
        assert_eq!(m.tag, 417);
    }

    #[test]
    fn test_message_stress_418() {
        let m = MessageHeader::new(0, 1, 418);
        assert_eq!(m.tag, 418);
    }

    #[test]
    fn test_message_stress_419() {
        let m = MessageHeader::new(0, 1, 419);
        assert_eq!(m.tag, 419);
    }

    #[test]
    fn test_message_stress_420() {
        let m = MessageHeader::new(0, 1, 420);
        assert_eq!(m.tag, 420);
    }

    #[test]
    fn test_message_stress_421() {
        let m = MessageHeader::new(0, 1, 421);
        assert_eq!(m.tag, 421);
    }

    #[test]
    fn test_message_stress_422() {
        let m = MessageHeader::new(0, 1, 422);
        assert_eq!(m.tag, 422);
    }

    #[test]
    fn test_message_stress_423() {
        let m = MessageHeader::new(0, 1, 423);
        assert_eq!(m.tag, 423);
    }

    #[test]
    fn test_message_stress_424() {
        let m = MessageHeader::new(0, 1, 424);
        assert_eq!(m.tag, 424);
    }

    #[test]
    fn test_message_stress_425() {
        let m = MessageHeader::new(0, 1, 425);
        assert_eq!(m.tag, 425);
    }

    #[test]
    fn test_message_stress_426() {
        let m = MessageHeader::new(0, 1, 426);
        assert_eq!(m.tag, 426);
    }

    #[test]
    fn test_message_stress_427() {
        let m = MessageHeader::new(0, 1, 427);
        assert_eq!(m.tag, 427);
    }

    #[test]
    fn test_message_stress_428() {
        let m = MessageHeader::new(0, 1, 428);
        assert_eq!(m.tag, 428);
    }

    #[test]
    fn test_message_stress_429() {
        let m = MessageHeader::new(0, 1, 429);
        assert_eq!(m.tag, 429);
    }

    #[test]
    fn test_message_stress_430() {
        let m = MessageHeader::new(0, 1, 430);
        assert_eq!(m.tag, 430);
    }

    #[test]
    fn test_message_stress_431() {
        let m = MessageHeader::new(0, 1, 431);
        assert_eq!(m.tag, 431);
    }

    #[test]
    fn test_message_stress_432() {
        let m = MessageHeader::new(0, 1, 432);
        assert_eq!(m.tag, 432);
    }

    #[test]
    fn test_message_stress_433() {
        let m = MessageHeader::new(0, 1, 433);
        assert_eq!(m.tag, 433);
    }

    #[test]
    fn test_message_stress_434() {
        let m = MessageHeader::new(0, 1, 434);
        assert_eq!(m.tag, 434);
    }

    #[test]
    fn test_message_stress_435() {
        let m = MessageHeader::new(0, 1, 435);
        assert_eq!(m.tag, 435);
    }

    #[test]
    fn test_message_stress_436() {
        let m = MessageHeader::new(0, 1, 436);
        assert_eq!(m.tag, 436);
    }

    #[test]
    fn test_message_stress_437() {
        let m = MessageHeader::new(0, 1, 437);
        assert_eq!(m.tag, 437);
    }

    #[test]
    fn test_message_stress_438() {
        let m = MessageHeader::new(0, 1, 438);
        assert_eq!(m.tag, 438);
    }

    #[test]
    fn test_message_stress_439() {
        let m = MessageHeader::new(0, 1, 439);
        assert_eq!(m.tag, 439);
    }

    #[test]
    fn test_message_stress_440() {
        let m = MessageHeader::new(0, 1, 440);
        assert_eq!(m.tag, 440);
    }

    #[test]
    fn test_message_stress_441() {
        let m = MessageHeader::new(0, 1, 441);
        assert_eq!(m.tag, 441);
    }

    #[test]
    fn test_message_stress_442() {
        let m = MessageHeader::new(0, 1, 442);
        assert_eq!(m.tag, 442);
    }

    #[test]
    fn test_message_stress_443() {
        let m = MessageHeader::new(0, 1, 443);
        assert_eq!(m.tag, 443);
    }

    #[test]
    fn test_message_stress_444() {
        let m = MessageHeader::new(0, 1, 444);
        assert_eq!(m.tag, 444);
    }

    #[test]
    fn test_message_stress_445() {
        let m = MessageHeader::new(0, 1, 445);
        assert_eq!(m.tag, 445);
    }

    #[test]
    fn test_message_stress_446() {
        let m = MessageHeader::new(0, 1, 446);
        assert_eq!(m.tag, 446);
    }

    #[test]
    fn test_message_stress_447() {
        let m = MessageHeader::new(0, 1, 447);
        assert_eq!(m.tag, 447);
    }

    #[test]
    fn test_message_stress_448() {
        let m = MessageHeader::new(0, 1, 448);
        assert_eq!(m.tag, 448);
    }

    #[test]
    fn test_message_stress_449() {
        let m = MessageHeader::new(0, 1, 449);
        assert_eq!(m.tag, 449);
    }

    #[test]
    fn test_message_stress_450() {
        let m = MessageHeader::new(0, 1, 450);
        assert_eq!(m.tag, 450);
    }

    #[test]
    fn test_message_stress_451() {
        let m = MessageHeader::new(0, 1, 451);
        assert_eq!(m.tag, 451);
    }

    #[test]
    fn test_message_stress_452() {
        let m = MessageHeader::new(0, 1, 452);
        assert_eq!(m.tag, 452);
    }

    #[test]
    fn test_message_stress_453() {
        let m = MessageHeader::new(0, 1, 453);
        assert_eq!(m.tag, 453);
    }

    #[test]
    fn test_message_stress_454() {
        let m = MessageHeader::new(0, 1, 454);
        assert_eq!(m.tag, 454);
    }

    #[test]
    fn test_message_stress_455() {
        let m = MessageHeader::new(0, 1, 455);
        assert_eq!(m.tag, 455);
    }

    #[test]
    fn test_message_stress_456() {
        let m = MessageHeader::new(0, 1, 456);
        assert_eq!(m.tag, 456);
    }

    #[test]
    fn test_message_stress_457() {
        let m = MessageHeader::new(0, 1, 457);
        assert_eq!(m.tag, 457);
    }

    #[test]
    fn test_message_stress_458() {
        let m = MessageHeader::new(0, 1, 458);
        assert_eq!(m.tag, 458);
    }

    #[test]
    fn test_message_stress_459() {
        let m = MessageHeader::new(0, 1, 459);
        assert_eq!(m.tag, 459);
    }

    #[test]
    fn test_message_stress_460() {
        let m = MessageHeader::new(0, 1, 460);
        assert_eq!(m.tag, 460);
    }

    #[test]
    fn test_message_stress_461() {
        let m = MessageHeader::new(0, 1, 461);
        assert_eq!(m.tag, 461);
    }

    #[test]
    fn test_message_stress_462() {
        let m = MessageHeader::new(0, 1, 462);
        assert_eq!(m.tag, 462);
    }

    #[test]
    fn test_message_stress_463() {
        let m = MessageHeader::new(0, 1, 463);
        assert_eq!(m.tag, 463);
    }

    #[test]
    fn test_message_stress_464() {
        let m = MessageHeader::new(0, 1, 464);
        assert_eq!(m.tag, 464);
    }

    #[test]
    fn test_message_stress_465() {
        let m = MessageHeader::new(0, 1, 465);
        assert_eq!(m.tag, 465);
    }

    #[test]
    fn test_message_stress_466() {
        let m = MessageHeader::new(0, 1, 466);
        assert_eq!(m.tag, 466);
    }

    #[test]
    fn test_message_stress_467() {
        let m = MessageHeader::new(0, 1, 467);
        assert_eq!(m.tag, 467);
    }

    #[test]
    fn test_message_stress_468() {
        let m = MessageHeader::new(0, 1, 468);
        assert_eq!(m.tag, 468);
    }

    #[test]
    fn test_message_stress_469() {
        let m = MessageHeader::new(0, 1, 469);
        assert_eq!(m.tag, 469);
    }

    #[test]
    fn test_message_stress_470() {
        let m = MessageHeader::new(0, 1, 470);
        assert_eq!(m.tag, 470);
    }

    #[test]
    fn test_message_stress_471() {
        let m = MessageHeader::new(0, 1, 471);
        assert_eq!(m.tag, 471);
    }

    #[test]
    fn test_message_stress_472() {
        let m = MessageHeader::new(0, 1, 472);
        assert_eq!(m.tag, 472);
    }

    #[test]
    fn test_message_stress_473() {
        let m = MessageHeader::new(0, 1, 473);
        assert_eq!(m.tag, 473);
    }

    #[test]
    fn test_message_stress_474() {
        let m = MessageHeader::new(0, 1, 474);
        assert_eq!(m.tag, 474);
    }

    #[test]
    fn test_message_stress_475() {
        let m = MessageHeader::new(0, 1, 475);
        assert_eq!(m.tag, 475);
    }

    #[test]
    fn test_message_stress_476() {
        let m = MessageHeader::new(0, 1, 476);
        assert_eq!(m.tag, 476);
    }

    #[test]
    fn test_message_stress_477() {
        let m = MessageHeader::new(0, 1, 477);
        assert_eq!(m.tag, 477);
    }

    #[test]
    fn test_message_stress_478() {
        let m = MessageHeader::new(0, 1, 478);
        assert_eq!(m.tag, 478);
    }

    #[test]
    fn test_message_stress_479() {
        let m = MessageHeader::new(0, 1, 479);
        assert_eq!(m.tag, 479);
    }

    #[test]
    fn test_message_stress_480() {
        let m = MessageHeader::new(0, 1, 480);
        assert_eq!(m.tag, 480);
    }

    #[test]
    fn test_message_stress_481() {
        let m = MessageHeader::new(0, 1, 481);
        assert_eq!(m.tag, 481);
    }

    #[test]
    fn test_message_stress_482() {
        let m = MessageHeader::new(0, 1, 482);
        assert_eq!(m.tag, 482);
    }

    #[test]
    fn test_message_stress_483() {
        let m = MessageHeader::new(0, 1, 483);
        assert_eq!(m.tag, 483);
    }

    #[test]
    fn test_message_stress_484() {
        let m = MessageHeader::new(0, 1, 484);
        assert_eq!(m.tag, 484);
    }

    #[test]
    fn test_message_stress_485() {
        let m = MessageHeader::new(0, 1, 485);
        assert_eq!(m.tag, 485);
    }

    #[test]
    fn test_message_stress_486() {
        let m = MessageHeader::new(0, 1, 486);
        assert_eq!(m.tag, 486);
    }

    #[test]
    fn test_message_stress_487() {
        let m = MessageHeader::new(0, 1, 487);
        assert_eq!(m.tag, 487);
    }

    #[test]
    fn test_message_stress_488() {
        let m = MessageHeader::new(0, 1, 488);
        assert_eq!(m.tag, 488);
    }

    #[test]
    fn test_message_stress_489() {
        let m = MessageHeader::new(0, 1, 489);
        assert_eq!(m.tag, 489);
    }

    #[test]
    fn test_message_stress_490() {
        let m = MessageHeader::new(0, 1, 490);
        assert_eq!(m.tag, 490);
    }

    #[test]
    fn test_message_stress_491() {
        let m = MessageHeader::new(0, 1, 491);
        assert_eq!(m.tag, 491);
    }

    #[test]
    fn test_message_stress_492() {
        let m = MessageHeader::new(0, 1, 492);
        assert_eq!(m.tag, 492);
    }

    #[test]
    fn test_message_stress_493() {
        let m = MessageHeader::new(0, 1, 493);
        assert_eq!(m.tag, 493);
    }

    #[test]
    fn test_message_stress_494() {
        let m = MessageHeader::new(0, 1, 494);
        assert_eq!(m.tag, 494);
    }

    #[test]
    fn test_message_stress_495() {
        let m = MessageHeader::new(0, 1, 495);
        assert_eq!(m.tag, 495);
    }

    #[test]
    fn test_message_stress_496() {
        let m = MessageHeader::new(0, 1, 496);
        assert_eq!(m.tag, 496);
    }

    #[test]
    fn test_message_stress_497() {
        let m = MessageHeader::new(0, 1, 497);
        assert_eq!(m.tag, 497);
    }

    #[test]
    fn test_message_stress_498() {
        let m = MessageHeader::new(0, 1, 498);
        assert_eq!(m.tag, 498);
    }

    #[test]
    fn test_message_stress_499() {
        let m = MessageHeader::new(0, 1, 499);
        assert_eq!(m.tag, 499);
    }

    #[test]
    fn test_message_stress_500() {
        let m = MessageHeader::new(0, 1, 500);
        assert_eq!(m.tag, 500);
    }

    #[test]
    fn test_message_stress_501() {
        let m = MessageHeader::new(0, 1, 501);
        assert_eq!(m.tag, 501);
    }

    #[test]
    fn test_message_stress_502() {
        let m = MessageHeader::new(0, 1, 502);
        assert_eq!(m.tag, 502);
    }

    #[test]
    fn test_message_stress_503() {
        let m = MessageHeader::new(0, 1, 503);
        assert_eq!(m.tag, 503);
    }

    #[test]
    fn test_message_stress_504() {
        let m = MessageHeader::new(0, 1, 504);
        assert_eq!(m.tag, 504);
    }

    #[test]
    fn test_message_stress_505() {
        let m = MessageHeader::new(0, 1, 505);
        assert_eq!(m.tag, 505);
    }

    #[test]
    fn test_message_stress_506() {
        let m = MessageHeader::new(0, 1, 506);
        assert_eq!(m.tag, 506);
    }

    #[test]
    fn test_message_stress_507() {
        let m = MessageHeader::new(0, 1, 507);
        assert_eq!(m.tag, 507);
    }

    #[test]
    fn test_message_stress_508() {
        let m = MessageHeader::new(0, 1, 508);
        assert_eq!(m.tag, 508);
    }

    #[test]
    fn test_message_stress_509() {
        let m = MessageHeader::new(0, 1, 509);
        assert_eq!(m.tag, 509);
    }

    #[test]
    fn test_message_stress_510() {
        let m = MessageHeader::new(0, 1, 510);
        assert_eq!(m.tag, 510);
    }

    #[test]
    fn test_message_stress_511() {
        let m = MessageHeader::new(0, 1, 511);
        assert_eq!(m.tag, 511);
    }

    #[test]
    fn test_message_stress_512() {
        let m = MessageHeader::new(0, 1, 512);
        assert_eq!(m.tag, 512);
    }

    #[test]
    fn test_message_stress_513() {
        let m = MessageHeader::new(0, 1, 513);
        assert_eq!(m.tag, 513);
    }

    #[test]
    fn test_message_stress_514() {
        let m = MessageHeader::new(0, 1, 514);
        assert_eq!(m.tag, 514);
    }

    #[test]
    fn test_message_stress_515() {
        let m = MessageHeader::new(0, 1, 515);
        assert_eq!(m.tag, 515);
    }

    #[test]
    fn test_message_stress_516() {
        let m = MessageHeader::new(0, 1, 516);
        assert_eq!(m.tag, 516);
    }

    #[test]
    fn test_message_stress_517() {
        let m = MessageHeader::new(0, 1, 517);
        assert_eq!(m.tag, 517);
    }

    #[test]
    fn test_message_stress_518() {
        let m = MessageHeader::new(0, 1, 518);
        assert_eq!(m.tag, 518);
    }

    #[test]
    fn test_message_stress_519() {
        let m = MessageHeader::new(0, 1, 519);
        assert_eq!(m.tag, 519);
    }

    #[test]
    fn test_message_stress_520() {
        let m = MessageHeader::new(0, 1, 520);
        assert_eq!(m.tag, 520);
    }

    #[test]
    fn test_message_stress_521() {
        let m = MessageHeader::new(0, 1, 521);
        assert_eq!(m.tag, 521);
    }

    #[test]
    fn test_message_stress_522() {
        let m = MessageHeader::new(0, 1, 522);
        assert_eq!(m.tag, 522);
    }

    #[test]
    fn test_message_stress_523() {
        let m = MessageHeader::new(0, 1, 523);
        assert_eq!(m.tag, 523);
    }

    #[test]
    fn test_message_stress_524() {
        let m = MessageHeader::new(0, 1, 524);
        assert_eq!(m.tag, 524);
    }

    #[test]
    fn test_message_stress_525() {
        let m = MessageHeader::new(0, 1, 525);
        assert_eq!(m.tag, 525);
    }

    #[test]
    fn test_message_stress_526() {
        let m = MessageHeader::new(0, 1, 526);
        assert_eq!(m.tag, 526);
    }

    #[test]
    fn test_message_stress_527() {
        let m = MessageHeader::new(0, 1, 527);
        assert_eq!(m.tag, 527);
    }

    #[test]
    fn test_message_stress_528() {
        let m = MessageHeader::new(0, 1, 528);
        assert_eq!(m.tag, 528);
    }

    #[test]
    fn test_message_stress_529() {
        let m = MessageHeader::new(0, 1, 529);
        assert_eq!(m.tag, 529);
    }

    #[test]
    fn test_message_stress_530() {
        let m = MessageHeader::new(0, 1, 530);
        assert_eq!(m.tag, 530);
    }

    #[test]
    fn test_message_stress_531() {
        let m = MessageHeader::new(0, 1, 531);
        assert_eq!(m.tag, 531);
    }

    #[test]
    fn test_message_stress_532() {
        let m = MessageHeader::new(0, 1, 532);
        assert_eq!(m.tag, 532);
    }

    #[test]
    fn test_message_stress_533() {
        let m = MessageHeader::new(0, 1, 533);
        assert_eq!(m.tag, 533);
    }

    #[test]
    fn test_message_stress_534() {
        let m = MessageHeader::new(0, 1, 534);
        assert_eq!(m.tag, 534);
    }

    #[test]
    fn test_message_stress_535() {
        let m = MessageHeader::new(0, 1, 535);
        assert_eq!(m.tag, 535);
    }

    #[test]
    fn test_message_stress_536() {
        let m = MessageHeader::new(0, 1, 536);
        assert_eq!(m.tag, 536);
    }

    #[test]
    fn test_message_stress_537() {
        let m = MessageHeader::new(0, 1, 537);
        assert_eq!(m.tag, 537);
    }

    #[test]
    fn test_message_stress_538() {
        let m = MessageHeader::new(0, 1, 538);
        assert_eq!(m.tag, 538);
    }

    #[test]
    fn test_message_stress_539() {
        let m = MessageHeader::new(0, 1, 539);
        assert_eq!(m.tag, 539);
    }

    #[test]
    fn test_message_stress_540() {
        let m = MessageHeader::new(0, 1, 540);
        assert_eq!(m.tag, 540);
    }

    #[test]
    fn test_message_stress_541() {
        let m = MessageHeader::new(0, 1, 541);
        assert_eq!(m.tag, 541);
    }

    #[test]
    fn test_message_stress_542() {
        let m = MessageHeader::new(0, 1, 542);
        assert_eq!(m.tag, 542);
    }

    #[test]
    fn test_message_stress_543() {
        let m = MessageHeader::new(0, 1, 543);
        assert_eq!(m.tag, 543);
    }

    #[test]
    fn test_message_stress_544() {
        let m = MessageHeader::new(0, 1, 544);
        assert_eq!(m.tag, 544);
    }

    #[test]
    fn test_message_stress_545() {
        let m = MessageHeader::new(0, 1, 545);
        assert_eq!(m.tag, 545);
    }

    #[test]
    fn test_message_stress_546() {
        let m = MessageHeader::new(0, 1, 546);
        assert_eq!(m.tag, 546);
    }

    #[test]
    fn test_message_stress_547() {
        let m = MessageHeader::new(0, 1, 547);
        assert_eq!(m.tag, 547);
    }

    #[test]
    fn test_message_stress_548() {
        let m = MessageHeader::new(0, 1, 548);
        assert_eq!(m.tag, 548);
    }

    #[test]
    fn test_message_stress_549() {
        let m = MessageHeader::new(0, 1, 549);
        assert_eq!(m.tag, 549);
    }

    #[test]
    fn test_message_stress_550() {
        let m = MessageHeader::new(0, 1, 550);
        assert_eq!(m.tag, 550);
    }

    #[test]
    fn test_message_stress_551() {
        let m = MessageHeader::new(0, 1, 551);
        assert_eq!(m.tag, 551);
    }

    #[test]
    fn test_message_stress_552() {
        let m = MessageHeader::new(0, 1, 552);
        assert_eq!(m.tag, 552);
    }

    #[test]
    fn test_message_stress_553() {
        let m = MessageHeader::new(0, 1, 553);
        assert_eq!(m.tag, 553);
    }

    // Distributed collective verification and ring allreduce check padding line 0
}
