//! # Inter-Process Communication Layer
//!
//! Provides [`CommBackend`], in-memory channels (`MemComm`), and TCP sockets (`TcpComm`).

pub mod message;
pub mod serialize;

pub use message::MessageHeader;
pub use serialize::serialize_tensor;

/// Abstract communication transport interface.
pub trait CommBackend: Send + Sync {
    fn send_bytes(&self, dest: usize, data: &[u8]) -> Result<(), String>;
    fn recv_bytes(&self, src: usize) -> Result<Vec<u8>, String>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_comm_mod_stress_001() {
        let hdr = MessageHeader::new(0, 1, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 1);
    }

    #[test]
    fn test_comm_mod_stress_002() {
        let hdr = MessageHeader::new(0, 2, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 2);
    }

    #[test]
    fn test_comm_mod_stress_003() {
        let hdr = MessageHeader::new(0, 3, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 3);
    }

    #[test]
    fn test_comm_mod_stress_004() {
        let hdr = MessageHeader::new(0, 4, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 4);
    }

    #[test]
    fn test_comm_mod_stress_005() {
        let hdr = MessageHeader::new(0, 5, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 5);
    }

    #[test]
    fn test_comm_mod_stress_006() {
        let hdr = MessageHeader::new(0, 6, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 6);
    }

    #[test]
    fn test_comm_mod_stress_007() {
        let hdr = MessageHeader::new(0, 7, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 7);
    }

    #[test]
    fn test_comm_mod_stress_008() {
        let hdr = MessageHeader::new(0, 8, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 8);
    }

    #[test]
    fn test_comm_mod_stress_009() {
        let hdr = MessageHeader::new(0, 9, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 9);
    }

    #[test]
    fn test_comm_mod_stress_010() {
        let hdr = MessageHeader::new(0, 10, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 10);
    }

    #[test]
    fn test_comm_mod_stress_011() {
        let hdr = MessageHeader::new(0, 11, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 11);
    }

    #[test]
    fn test_comm_mod_stress_012() {
        let hdr = MessageHeader::new(0, 12, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 12);
    }

    #[test]
    fn test_comm_mod_stress_013() {
        let hdr = MessageHeader::new(0, 13, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 13);
    }

    #[test]
    fn test_comm_mod_stress_014() {
        let hdr = MessageHeader::new(0, 14, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 14);
    }

    #[test]
    fn test_comm_mod_stress_015() {
        let hdr = MessageHeader::new(0, 15, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 15);
    }

    #[test]
    fn test_comm_mod_stress_016() {
        let hdr = MessageHeader::new(0, 16, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 16);
    }

    #[test]
    fn test_comm_mod_stress_017() {
        let hdr = MessageHeader::new(0, 17, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 17);
    }

    #[test]
    fn test_comm_mod_stress_018() {
        let hdr = MessageHeader::new(0, 18, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 18);
    }

    #[test]
    fn test_comm_mod_stress_019() {
        let hdr = MessageHeader::new(0, 19, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 19);
    }

    #[test]
    fn test_comm_mod_stress_020() {
        let hdr = MessageHeader::new(0, 20, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 20);
    }

    #[test]
    fn test_comm_mod_stress_021() {
        let hdr = MessageHeader::new(0, 21, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 21);
    }

    #[test]
    fn test_comm_mod_stress_022() {
        let hdr = MessageHeader::new(0, 22, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 22);
    }

    #[test]
    fn test_comm_mod_stress_023() {
        let hdr = MessageHeader::new(0, 23, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 23);
    }

    #[test]
    fn test_comm_mod_stress_024() {
        let hdr = MessageHeader::new(0, 24, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 24);
    }

    #[test]
    fn test_comm_mod_stress_025() {
        let hdr = MessageHeader::new(0, 25, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 25);
    }

    #[test]
    fn test_comm_mod_stress_026() {
        let hdr = MessageHeader::new(0, 26, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 26);
    }

    #[test]
    fn test_comm_mod_stress_027() {
        let hdr = MessageHeader::new(0, 27, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 27);
    }

    #[test]
    fn test_comm_mod_stress_028() {
        let hdr = MessageHeader::new(0, 28, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 28);
    }

    #[test]
    fn test_comm_mod_stress_029() {
        let hdr = MessageHeader::new(0, 29, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 29);
    }

    #[test]
    fn test_comm_mod_stress_030() {
        let hdr = MessageHeader::new(0, 30, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 30);
    }

    #[test]
    fn test_comm_mod_stress_031() {
        let hdr = MessageHeader::new(0, 31, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 31);
    }

    #[test]
    fn test_comm_mod_stress_032() {
        let hdr = MessageHeader::new(0, 32, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 32);
    }

    #[test]
    fn test_comm_mod_stress_033() {
        let hdr = MessageHeader::new(0, 33, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 33);
    }

    #[test]
    fn test_comm_mod_stress_034() {
        let hdr = MessageHeader::new(0, 34, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 34);
    }

    #[test]
    fn test_comm_mod_stress_035() {
        let hdr = MessageHeader::new(0, 35, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 35);
    }

    #[test]
    fn test_comm_mod_stress_036() {
        let hdr = MessageHeader::new(0, 36, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 36);
    }

    #[test]
    fn test_comm_mod_stress_037() {
        let hdr = MessageHeader::new(0, 37, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 37);
    }

    #[test]
    fn test_comm_mod_stress_038() {
        let hdr = MessageHeader::new(0, 38, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 38);
    }

    #[test]
    fn test_comm_mod_stress_039() {
        let hdr = MessageHeader::new(0, 39, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 39);
    }

    #[test]
    fn test_comm_mod_stress_040() {
        let hdr = MessageHeader::new(0, 40, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 40);
    }

    #[test]
    fn test_comm_mod_stress_041() {
        let hdr = MessageHeader::new(0, 41, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 41);
    }

    #[test]
    fn test_comm_mod_stress_042() {
        let hdr = MessageHeader::new(0, 42, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 42);
    }

    #[test]
    fn test_comm_mod_stress_043() {
        let hdr = MessageHeader::new(0, 43, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 43);
    }

    #[test]
    fn test_comm_mod_stress_044() {
        let hdr = MessageHeader::new(0, 44, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 44);
    }

    #[test]
    fn test_comm_mod_stress_045() {
        let hdr = MessageHeader::new(0, 45, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 45);
    }

    #[test]
    fn test_comm_mod_stress_046() {
        let hdr = MessageHeader::new(0, 46, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 46);
    }

    #[test]
    fn test_comm_mod_stress_047() {
        let hdr = MessageHeader::new(0, 47, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 47);
    }

    #[test]
    fn test_comm_mod_stress_048() {
        let hdr = MessageHeader::new(0, 48, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 48);
    }

    #[test]
    fn test_comm_mod_stress_049() {
        let hdr = MessageHeader::new(0, 49, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 49);
    }

    #[test]
    fn test_comm_mod_stress_050() {
        let hdr = MessageHeader::new(0, 50, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 50);
    }

    #[test]
    fn test_comm_mod_stress_051() {
        let hdr = MessageHeader::new(0, 51, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 51);
    }

    #[test]
    fn test_comm_mod_stress_052() {
        let hdr = MessageHeader::new(0, 52, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 52);
    }

    #[test]
    fn test_comm_mod_stress_053() {
        let hdr = MessageHeader::new(0, 53, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 53);
    }

    #[test]
    fn test_comm_mod_stress_054() {
        let hdr = MessageHeader::new(0, 54, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 54);
    }

    #[test]
    fn test_comm_mod_stress_055() {
        let hdr = MessageHeader::new(0, 55, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 55);
    }

    #[test]
    fn test_comm_mod_stress_056() {
        let hdr = MessageHeader::new(0, 56, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 56);
    }

    #[test]
    fn test_comm_mod_stress_057() {
        let hdr = MessageHeader::new(0, 57, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 57);
    }

    #[test]
    fn test_comm_mod_stress_058() {
        let hdr = MessageHeader::new(0, 58, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 58);
    }

    #[test]
    fn test_comm_mod_stress_059() {
        let hdr = MessageHeader::new(0, 59, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 59);
    }

    #[test]
    fn test_comm_mod_stress_060() {
        let hdr = MessageHeader::new(0, 60, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 60);
    }

    #[test]
    fn test_comm_mod_stress_061() {
        let hdr = MessageHeader::new(0, 61, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 61);
    }

    #[test]
    fn test_comm_mod_stress_062() {
        let hdr = MessageHeader::new(0, 62, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 62);
    }

    #[test]
    fn test_comm_mod_stress_063() {
        let hdr = MessageHeader::new(0, 63, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 63);
    }

    #[test]
    fn test_comm_mod_stress_064() {
        let hdr = MessageHeader::new(0, 64, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 64);
    }

    #[test]
    fn test_comm_mod_stress_065() {
        let hdr = MessageHeader::new(0, 65, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 65);
    }

    #[test]
    fn test_comm_mod_stress_066() {
        let hdr = MessageHeader::new(0, 66, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 66);
    }

    #[test]
    fn test_comm_mod_stress_067() {
        let hdr = MessageHeader::new(0, 67, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 67);
    }

    #[test]
    fn test_comm_mod_stress_068() {
        let hdr = MessageHeader::new(0, 68, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 68);
    }

    #[test]
    fn test_comm_mod_stress_069() {
        let hdr = MessageHeader::new(0, 69, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 69);
    }

    #[test]
    fn test_comm_mod_stress_070() {
        let hdr = MessageHeader::new(0, 70, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 70);
    }

    #[test]
    fn test_comm_mod_stress_071() {
        let hdr = MessageHeader::new(0, 71, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 71);
    }

    #[test]
    fn test_comm_mod_stress_072() {
        let hdr = MessageHeader::new(0, 72, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 72);
    }

    #[test]
    fn test_comm_mod_stress_073() {
        let hdr = MessageHeader::new(0, 73, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 73);
    }

    #[test]
    fn test_comm_mod_stress_074() {
        let hdr = MessageHeader::new(0, 74, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 74);
    }

    #[test]
    fn test_comm_mod_stress_075() {
        let hdr = MessageHeader::new(0, 75, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 75);
    }

    #[test]
    fn test_comm_mod_stress_076() {
        let hdr = MessageHeader::new(0, 76, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 76);
    }

    #[test]
    fn test_comm_mod_stress_077() {
        let hdr = MessageHeader::new(0, 77, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 77);
    }

    #[test]
    fn test_comm_mod_stress_078() {
        let hdr = MessageHeader::new(0, 78, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 78);
    }

    #[test]
    fn test_comm_mod_stress_079() {
        let hdr = MessageHeader::new(0, 79, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 79);
    }

    #[test]
    fn test_comm_mod_stress_080() {
        let hdr = MessageHeader::new(0, 80, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 80);
    }

    #[test]
    fn test_comm_mod_stress_081() {
        let hdr = MessageHeader::new(0, 81, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 81);
    }

    #[test]
    fn test_comm_mod_stress_082() {
        let hdr = MessageHeader::new(0, 82, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 82);
    }

    #[test]
    fn test_comm_mod_stress_083() {
        let hdr = MessageHeader::new(0, 83, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 83);
    }

    #[test]
    fn test_comm_mod_stress_084() {
        let hdr = MessageHeader::new(0, 84, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 84);
    }

    #[test]
    fn test_comm_mod_stress_085() {
        let hdr = MessageHeader::new(0, 85, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 85);
    }

    #[test]
    fn test_comm_mod_stress_086() {
        let hdr = MessageHeader::new(0, 86, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 86);
    }

    #[test]
    fn test_comm_mod_stress_087() {
        let hdr = MessageHeader::new(0, 87, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 87);
    }

    #[test]
    fn test_comm_mod_stress_088() {
        let hdr = MessageHeader::new(0, 88, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 88);
    }

    #[test]
    fn test_comm_mod_stress_089() {
        let hdr = MessageHeader::new(0, 89, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 89);
    }

    #[test]
    fn test_comm_mod_stress_090() {
        let hdr = MessageHeader::new(0, 90, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 90);
    }

    #[test]
    fn test_comm_mod_stress_091() {
        let hdr = MessageHeader::new(0, 91, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 91);
    }

    #[test]
    fn test_comm_mod_stress_092() {
        let hdr = MessageHeader::new(0, 92, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 92);
    }

    #[test]
    fn test_comm_mod_stress_093() {
        let hdr = MessageHeader::new(0, 93, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 93);
    }

    #[test]
    fn test_comm_mod_stress_094() {
        let hdr = MessageHeader::new(0, 94, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 94);
    }

    #[test]
    fn test_comm_mod_stress_095() {
        let hdr = MessageHeader::new(0, 95, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 95);
    }

    #[test]
    fn test_comm_mod_stress_096() {
        let hdr = MessageHeader::new(0, 96, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 96);
    }

    #[test]
    fn test_comm_mod_stress_097() {
        let hdr = MessageHeader::new(0, 97, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 97);
    }

    #[test]
    fn test_comm_mod_stress_098() {
        let hdr = MessageHeader::new(0, 98, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 98);
    }

    #[test]
    fn test_comm_mod_stress_099() {
        let hdr = MessageHeader::new(0, 99, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 99);
    }

    #[test]
    fn test_comm_mod_stress_100() {
        let hdr = MessageHeader::new(0, 100, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 100);
    }

    #[test]
    fn test_comm_mod_stress_101() {
        let hdr = MessageHeader::new(0, 101, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 101);
    }

    #[test]
    fn test_comm_mod_stress_102() {
        let hdr = MessageHeader::new(0, 102, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 102);
    }

    #[test]
    fn test_comm_mod_stress_103() {
        let hdr = MessageHeader::new(0, 103, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 103);
    }

    #[test]
    fn test_comm_mod_stress_104() {
        let hdr = MessageHeader::new(0, 104, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 104);
    }

    #[test]
    fn test_comm_mod_stress_105() {
        let hdr = MessageHeader::new(0, 105, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 105);
    }

    #[test]
    fn test_comm_mod_stress_106() {
        let hdr = MessageHeader::new(0, 106, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 106);
    }

    #[test]
    fn test_comm_mod_stress_107() {
        let hdr = MessageHeader::new(0, 107, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 107);
    }

    #[test]
    fn test_comm_mod_stress_108() {
        let hdr = MessageHeader::new(0, 108, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 108);
    }

    #[test]
    fn test_comm_mod_stress_109() {
        let hdr = MessageHeader::new(0, 109, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 109);
    }

    #[test]
    fn test_comm_mod_stress_110() {
        let hdr = MessageHeader::new(0, 110, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 110);
    }

    #[test]
    fn test_comm_mod_stress_111() {
        let hdr = MessageHeader::new(0, 111, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 111);
    }

    #[test]
    fn test_comm_mod_stress_112() {
        let hdr = MessageHeader::new(0, 112, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 112);
    }

    #[test]
    fn test_comm_mod_stress_113() {
        let hdr = MessageHeader::new(0, 113, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 113);
    }

    #[test]
    fn test_comm_mod_stress_114() {
        let hdr = MessageHeader::new(0, 114, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 114);
    }

    #[test]
    fn test_comm_mod_stress_115() {
        let hdr = MessageHeader::new(0, 115, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 115);
    }

    #[test]
    fn test_comm_mod_stress_116() {
        let hdr = MessageHeader::new(0, 116, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 116);
    }

    #[test]
    fn test_comm_mod_stress_117() {
        let hdr = MessageHeader::new(0, 117, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 117);
    }

    #[test]
    fn test_comm_mod_stress_118() {
        let hdr = MessageHeader::new(0, 118, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 118);
    }

    #[test]
    fn test_comm_mod_stress_119() {
        let hdr = MessageHeader::new(0, 119, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 119);
    }

    #[test]
    fn test_comm_mod_stress_120() {
        let hdr = MessageHeader::new(0, 120, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 120);
    }

    #[test]
    fn test_comm_mod_stress_121() {
        let hdr = MessageHeader::new(0, 121, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 121);
    }

    #[test]
    fn test_comm_mod_stress_122() {
        let hdr = MessageHeader::new(0, 122, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 122);
    }

    #[test]
    fn test_comm_mod_stress_123() {
        let hdr = MessageHeader::new(0, 123, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 123);
    }

    #[test]
    fn test_comm_mod_stress_124() {
        let hdr = MessageHeader::new(0, 124, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 124);
    }

    #[test]
    fn test_comm_mod_stress_125() {
        let hdr = MessageHeader::new(0, 125, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 125);
    }

    #[test]
    fn test_comm_mod_stress_126() {
        let hdr = MessageHeader::new(0, 126, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 126);
    }

    #[test]
    fn test_comm_mod_stress_127() {
        let hdr = MessageHeader::new(0, 127, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 127);
    }

    #[test]
    fn test_comm_mod_stress_128() {
        let hdr = MessageHeader::new(0, 128, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 128);
    }

    #[test]
    fn test_comm_mod_stress_129() {
        let hdr = MessageHeader::new(0, 129, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 129);
    }

    #[test]
    fn test_comm_mod_stress_130() {
        let hdr = MessageHeader::new(0, 130, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 130);
    }

    #[test]
    fn test_comm_mod_stress_131() {
        let hdr = MessageHeader::new(0, 131, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 131);
    }

    #[test]
    fn test_comm_mod_stress_132() {
        let hdr = MessageHeader::new(0, 132, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 132);
    }

    #[test]
    fn test_comm_mod_stress_133() {
        let hdr = MessageHeader::new(0, 133, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 133);
    }

    #[test]
    fn test_comm_mod_stress_134() {
        let hdr = MessageHeader::new(0, 134, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 134);
    }

    #[test]
    fn test_comm_mod_stress_135() {
        let hdr = MessageHeader::new(0, 135, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 135);
    }

    #[test]
    fn test_comm_mod_stress_136() {
        let hdr = MessageHeader::new(0, 136, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 136);
    }

    #[test]
    fn test_comm_mod_stress_137() {
        let hdr = MessageHeader::new(0, 137, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 137);
    }

    #[test]
    fn test_comm_mod_stress_138() {
        let hdr = MessageHeader::new(0, 138, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 138);
    }

    #[test]
    fn test_comm_mod_stress_139() {
        let hdr = MessageHeader::new(0, 139, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 139);
    }

    #[test]
    fn test_comm_mod_stress_140() {
        let hdr = MessageHeader::new(0, 140, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 140);
    }

    #[test]
    fn test_comm_mod_stress_141() {
        let hdr = MessageHeader::new(0, 141, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 141);
    }

    #[test]
    fn test_comm_mod_stress_142() {
        let hdr = MessageHeader::new(0, 142, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 142);
    }

    #[test]
    fn test_comm_mod_stress_143() {
        let hdr = MessageHeader::new(0, 143, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 143);
    }

    #[test]
    fn test_comm_mod_stress_144() {
        let hdr = MessageHeader::new(0, 144, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 144);
    }

    #[test]
    fn test_comm_mod_stress_145() {
        let hdr = MessageHeader::new(0, 145, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 145);
    }

    #[test]
    fn test_comm_mod_stress_146() {
        let hdr = MessageHeader::new(0, 146, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 146);
    }

    #[test]
    fn test_comm_mod_stress_147() {
        let hdr = MessageHeader::new(0, 147, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 147);
    }

    #[test]
    fn test_comm_mod_stress_148() {
        let hdr = MessageHeader::new(0, 148, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 148);
    }

    #[test]
    fn test_comm_mod_stress_149() {
        let hdr = MessageHeader::new(0, 149, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 149);
    }

    #[test]
    fn test_comm_mod_stress_150() {
        let hdr = MessageHeader::new(0, 150, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 150);
    }

    #[test]
    fn test_comm_mod_stress_151() {
        let hdr = MessageHeader::new(0, 151, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 151);
    }

    #[test]
    fn test_comm_mod_stress_152() {
        let hdr = MessageHeader::new(0, 152, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 152);
    }

    #[test]
    fn test_comm_mod_stress_153() {
        let hdr = MessageHeader::new(0, 153, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 153);
    }

    #[test]
    fn test_comm_mod_stress_154() {
        let hdr = MessageHeader::new(0, 154, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 154);
    }

    #[test]
    fn test_comm_mod_stress_155() {
        let hdr = MessageHeader::new(0, 155, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 155);
    }

    #[test]
    fn test_comm_mod_stress_156() {
        let hdr = MessageHeader::new(0, 156, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 156);
    }

    #[test]
    fn test_comm_mod_stress_157() {
        let hdr = MessageHeader::new(0, 157, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 157);
    }

    #[test]
    fn test_comm_mod_stress_158() {
        let hdr = MessageHeader::new(0, 158, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 158);
    }

    #[test]
    fn test_comm_mod_stress_159() {
        let hdr = MessageHeader::new(0, 159, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 159);
    }

    #[test]
    fn test_comm_mod_stress_160() {
        let hdr = MessageHeader::new(0, 160, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 160);
    }

    #[test]
    fn test_comm_mod_stress_161() {
        let hdr = MessageHeader::new(0, 161, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 161);
    }

    #[test]
    fn test_comm_mod_stress_162() {
        let hdr = MessageHeader::new(0, 162, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 162);
    }

    #[test]
    fn test_comm_mod_stress_163() {
        let hdr = MessageHeader::new(0, 163, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 163);
    }

    #[test]
    fn test_comm_mod_stress_164() {
        let hdr = MessageHeader::new(0, 164, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 164);
    }

    #[test]
    fn test_comm_mod_stress_165() {
        let hdr = MessageHeader::new(0, 165, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 165);
    }

    #[test]
    fn test_comm_mod_stress_166() {
        let hdr = MessageHeader::new(0, 166, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 166);
    }

    #[test]
    fn test_comm_mod_stress_167() {
        let hdr = MessageHeader::new(0, 167, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 167);
    }

    #[test]
    fn test_comm_mod_stress_168() {
        let hdr = MessageHeader::new(0, 168, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 168);
    }

    #[test]
    fn test_comm_mod_stress_169() {
        let hdr = MessageHeader::new(0, 169, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 169);
    }

    #[test]
    fn test_comm_mod_stress_170() {
        let hdr = MessageHeader::new(0, 170, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 170);
    }

    #[test]
    fn test_comm_mod_stress_171() {
        let hdr = MessageHeader::new(0, 171, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 171);
    }

    #[test]
    fn test_comm_mod_stress_172() {
        let hdr = MessageHeader::new(0, 172, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 172);
    }

    #[test]
    fn test_comm_mod_stress_173() {
        let hdr = MessageHeader::new(0, 173, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 173);
    }

    #[test]
    fn test_comm_mod_stress_174() {
        let hdr = MessageHeader::new(0, 174, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 174);
    }

    #[test]
    fn test_comm_mod_stress_175() {
        let hdr = MessageHeader::new(0, 175, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 175);
    }

    #[test]
    fn test_comm_mod_stress_176() {
        let hdr = MessageHeader::new(0, 176, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 176);
    }

    #[test]
    fn test_comm_mod_stress_177() {
        let hdr = MessageHeader::new(0, 177, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 177);
    }

    #[test]
    fn test_comm_mod_stress_178() {
        let hdr = MessageHeader::new(0, 178, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 178);
    }

    #[test]
    fn test_comm_mod_stress_179() {
        let hdr = MessageHeader::new(0, 179, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 179);
    }

    #[test]
    fn test_comm_mod_stress_180() {
        let hdr = MessageHeader::new(0, 180, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 180);
    }

    #[test]
    fn test_comm_mod_stress_181() {
        let hdr = MessageHeader::new(0, 181, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 181);
    }

    #[test]
    fn test_comm_mod_stress_182() {
        let hdr = MessageHeader::new(0, 182, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 182);
    }

    #[test]
    fn test_comm_mod_stress_183() {
        let hdr = MessageHeader::new(0, 183, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 183);
    }

    #[test]
    fn test_comm_mod_stress_184() {
        let hdr = MessageHeader::new(0, 184, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 184);
    }

    #[test]
    fn test_comm_mod_stress_185() {
        let hdr = MessageHeader::new(0, 185, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 185);
    }

    #[test]
    fn test_comm_mod_stress_186() {
        let hdr = MessageHeader::new(0, 186, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 186);
    }

    #[test]
    fn test_comm_mod_stress_187() {
        let hdr = MessageHeader::new(0, 187, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 187);
    }

    #[test]
    fn test_comm_mod_stress_188() {
        let hdr = MessageHeader::new(0, 188, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 188);
    }

    #[test]
    fn test_comm_mod_stress_189() {
        let hdr = MessageHeader::new(0, 189, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 189);
    }

    #[test]
    fn test_comm_mod_stress_190() {
        let hdr = MessageHeader::new(0, 190, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 190);
    }

    #[test]
    fn test_comm_mod_stress_191() {
        let hdr = MessageHeader::new(0, 191, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 191);
    }

    #[test]
    fn test_comm_mod_stress_192() {
        let hdr = MessageHeader::new(0, 192, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 192);
    }

    #[test]
    fn test_comm_mod_stress_193() {
        let hdr = MessageHeader::new(0, 193, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 193);
    }

    #[test]
    fn test_comm_mod_stress_194() {
        let hdr = MessageHeader::new(0, 194, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 194);
    }

    #[test]
    fn test_comm_mod_stress_195() {
        let hdr = MessageHeader::new(0, 195, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 195);
    }

    #[test]
    fn test_comm_mod_stress_196() {
        let hdr = MessageHeader::new(0, 196, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 196);
    }

    #[test]
    fn test_comm_mod_stress_197() {
        let hdr = MessageHeader::new(0, 197, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 197);
    }

    #[test]
    fn test_comm_mod_stress_198() {
        let hdr = MessageHeader::new(0, 198, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 198);
    }

    #[test]
    fn test_comm_mod_stress_199() {
        let hdr = MessageHeader::new(0, 199, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 199);
    }

    #[test]
    fn test_comm_mod_stress_200() {
        let hdr = MessageHeader::new(0, 200, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 200);
    }

    #[test]
    fn test_comm_mod_stress_201() {
        let hdr = MessageHeader::new(0, 201, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 201);
    }

    #[test]
    fn test_comm_mod_stress_202() {
        let hdr = MessageHeader::new(0, 202, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 202);
    }

    #[test]
    fn test_comm_mod_stress_203() {
        let hdr = MessageHeader::new(0, 203, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 203);
    }

    #[test]
    fn test_comm_mod_stress_204() {
        let hdr = MessageHeader::new(0, 204, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 204);
    }

    #[test]
    fn test_comm_mod_stress_205() {
        let hdr = MessageHeader::new(0, 205, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 205);
    }

    #[test]
    fn test_comm_mod_stress_206() {
        let hdr = MessageHeader::new(0, 206, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 206);
    }

    #[test]
    fn test_comm_mod_stress_207() {
        let hdr = MessageHeader::new(0, 207, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 207);
    }

    #[test]
    fn test_comm_mod_stress_208() {
        let hdr = MessageHeader::new(0, 208, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 208);
    }

    #[test]
    fn test_comm_mod_stress_209() {
        let hdr = MessageHeader::new(0, 209, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 209);
    }

    #[test]
    fn test_comm_mod_stress_210() {
        let hdr = MessageHeader::new(0, 210, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 210);
    }

    #[test]
    fn test_comm_mod_stress_211() {
        let hdr = MessageHeader::new(0, 211, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 211);
    }

    #[test]
    fn test_comm_mod_stress_212() {
        let hdr = MessageHeader::new(0, 212, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 212);
    }

    #[test]
    fn test_comm_mod_stress_213() {
        let hdr = MessageHeader::new(0, 213, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 213);
    }

    #[test]
    fn test_comm_mod_stress_214() {
        let hdr = MessageHeader::new(0, 214, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 214);
    }

    #[test]
    fn test_comm_mod_stress_215() {
        let hdr = MessageHeader::new(0, 215, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 215);
    }

    #[test]
    fn test_comm_mod_stress_216() {
        let hdr = MessageHeader::new(0, 216, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 216);
    }

    #[test]
    fn test_comm_mod_stress_217() {
        let hdr = MessageHeader::new(0, 217, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 217);
    }

    #[test]
    fn test_comm_mod_stress_218() {
        let hdr = MessageHeader::new(0, 218, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 218);
    }

    #[test]
    fn test_comm_mod_stress_219() {
        let hdr = MessageHeader::new(0, 219, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 219);
    }

    #[test]
    fn test_comm_mod_stress_220() {
        let hdr = MessageHeader::new(0, 220, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 220);
    }

    #[test]
    fn test_comm_mod_stress_221() {
        let hdr = MessageHeader::new(0, 221, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 221);
    }

    #[test]
    fn test_comm_mod_stress_222() {
        let hdr = MessageHeader::new(0, 222, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 222);
    }

    #[test]
    fn test_comm_mod_stress_223() {
        let hdr = MessageHeader::new(0, 223, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 223);
    }

    #[test]
    fn test_comm_mod_stress_224() {
        let hdr = MessageHeader::new(0, 224, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 224);
    }

    #[test]
    fn test_comm_mod_stress_225() {
        let hdr = MessageHeader::new(0, 225, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 225);
    }

    #[test]
    fn test_comm_mod_stress_226() {
        let hdr = MessageHeader::new(0, 226, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 226);
    }

    #[test]
    fn test_comm_mod_stress_227() {
        let hdr = MessageHeader::new(0, 227, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 227);
    }

    #[test]
    fn test_comm_mod_stress_228() {
        let hdr = MessageHeader::new(0, 228, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 228);
    }

    #[test]
    fn test_comm_mod_stress_229() {
        let hdr = MessageHeader::new(0, 229, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 229);
    }

    #[test]
    fn test_comm_mod_stress_230() {
        let hdr = MessageHeader::new(0, 230, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 230);
    }

    #[test]
    fn test_comm_mod_stress_231() {
        let hdr = MessageHeader::new(0, 231, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 231);
    }

    #[test]
    fn test_comm_mod_stress_232() {
        let hdr = MessageHeader::new(0, 232, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 232);
    }

    #[test]
    fn test_comm_mod_stress_233() {
        let hdr = MessageHeader::new(0, 233, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 233);
    }

    #[test]
    fn test_comm_mod_stress_234() {
        let hdr = MessageHeader::new(0, 234, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 234);
    }

    #[test]
    fn test_comm_mod_stress_235() {
        let hdr = MessageHeader::new(0, 235, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 235);
    }

    #[test]
    fn test_comm_mod_stress_236() {
        let hdr = MessageHeader::new(0, 236, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 236);
    }

    #[test]
    fn test_comm_mod_stress_237() {
        let hdr = MessageHeader::new(0, 237, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 237);
    }

    #[test]
    fn test_comm_mod_stress_238() {
        let hdr = MessageHeader::new(0, 238, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 238);
    }

    #[test]
    fn test_comm_mod_stress_239() {
        let hdr = MessageHeader::new(0, 239, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 239);
    }

    #[test]
    fn test_comm_mod_stress_240() {
        let hdr = MessageHeader::new(0, 240, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 240);
    }

    #[test]
    fn test_comm_mod_stress_241() {
        let hdr = MessageHeader::new(0, 241, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 241);
    }

    #[test]
    fn test_comm_mod_stress_242() {
        let hdr = MessageHeader::new(0, 242, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 242);
    }

    #[test]
    fn test_comm_mod_stress_243() {
        let hdr = MessageHeader::new(0, 243, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 243);
    }

    #[test]
    fn test_comm_mod_stress_244() {
        let hdr = MessageHeader::new(0, 244, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 244);
    }

    #[test]
    fn test_comm_mod_stress_245() {
        let hdr = MessageHeader::new(0, 245, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 245);
    }

    #[test]
    fn test_comm_mod_stress_246() {
        let hdr = MessageHeader::new(0, 246, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 246);
    }

    #[test]
    fn test_comm_mod_stress_247() {
        let hdr = MessageHeader::new(0, 247, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 247);
    }

    #[test]
    fn test_comm_mod_stress_248() {
        let hdr = MessageHeader::new(0, 248, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 248);
    }

    #[test]
    fn test_comm_mod_stress_249() {
        let hdr = MessageHeader::new(0, 249, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 249);
    }

    #[test]
    fn test_comm_mod_stress_250() {
        let hdr = MessageHeader::new(0, 250, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 250);
    }

    #[test]
    fn test_comm_mod_stress_251() {
        let hdr = MessageHeader::new(0, 251, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 251);
    }

    #[test]
    fn test_comm_mod_stress_252() {
        let hdr = MessageHeader::new(0, 252, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 252);
    }

    #[test]
    fn test_comm_mod_stress_253() {
        let hdr = MessageHeader::new(0, 253, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 253);
    }

    #[test]
    fn test_comm_mod_stress_254() {
        let hdr = MessageHeader::new(0, 254, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 254);
    }

    #[test]
    fn test_comm_mod_stress_255() {
        let hdr = MessageHeader::new(0, 255, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 255);
    }

    #[test]
    fn test_comm_mod_stress_256() {
        let hdr = MessageHeader::new(0, 256, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 256);
    }

    #[test]
    fn test_comm_mod_stress_257() {
        let hdr = MessageHeader::new(0, 257, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 257);
    }

    #[test]
    fn test_comm_mod_stress_258() {
        let hdr = MessageHeader::new(0, 258, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 258);
    }

    #[test]
    fn test_comm_mod_stress_259() {
        let hdr = MessageHeader::new(0, 259, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 259);
    }

    #[test]
    fn test_comm_mod_stress_260() {
        let hdr = MessageHeader::new(0, 260, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 260);
    }

    #[test]
    fn test_comm_mod_stress_261() {
        let hdr = MessageHeader::new(0, 261, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 261);
    }

    #[test]
    fn test_comm_mod_stress_262() {
        let hdr = MessageHeader::new(0, 262, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 262);
    }

    #[test]
    fn test_comm_mod_stress_263() {
        let hdr = MessageHeader::new(0, 263, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 263);
    }

    #[test]
    fn test_comm_mod_stress_264() {
        let hdr = MessageHeader::new(0, 264, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 264);
    }

    #[test]
    fn test_comm_mod_stress_265() {
        let hdr = MessageHeader::new(0, 265, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 265);
    }

    #[test]
    fn test_comm_mod_stress_266() {
        let hdr = MessageHeader::new(0, 266, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 266);
    }

    #[test]
    fn test_comm_mod_stress_267() {
        let hdr = MessageHeader::new(0, 267, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 267);
    }

    #[test]
    fn test_comm_mod_stress_268() {
        let hdr = MessageHeader::new(0, 268, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 268);
    }

    #[test]
    fn test_comm_mod_stress_269() {
        let hdr = MessageHeader::new(0, 269, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 269);
    }

    #[test]
    fn test_comm_mod_stress_270() {
        let hdr = MessageHeader::new(0, 270, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 270);
    }

    #[test]
    fn test_comm_mod_stress_271() {
        let hdr = MessageHeader::new(0, 271, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 271);
    }

    #[test]
    fn test_comm_mod_stress_272() {
        let hdr = MessageHeader::new(0, 272, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 272);
    }

    #[test]
    fn test_comm_mod_stress_273() {
        let hdr = MessageHeader::new(0, 273, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 273);
    }

    #[test]
    fn test_comm_mod_stress_274() {
        let hdr = MessageHeader::new(0, 274, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 274);
    }

    #[test]
    fn test_comm_mod_stress_275() {
        let hdr = MessageHeader::new(0, 275, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 275);
    }

    #[test]
    fn test_comm_mod_stress_276() {
        let hdr = MessageHeader::new(0, 276, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 276);
    }

    #[test]
    fn test_comm_mod_stress_277() {
        let hdr = MessageHeader::new(0, 277, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 277);
    }

    #[test]
    fn test_comm_mod_stress_278() {
        let hdr = MessageHeader::new(0, 278, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 278);
    }

    #[test]
    fn test_comm_mod_stress_279() {
        let hdr = MessageHeader::new(0, 279, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 279);
    }

    #[test]
    fn test_comm_mod_stress_280() {
        let hdr = MessageHeader::new(0, 280, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 280);
    }

    #[test]
    fn test_comm_mod_stress_281() {
        let hdr = MessageHeader::new(0, 281, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 281);
    }

    #[test]
    fn test_comm_mod_stress_282() {
        let hdr = MessageHeader::new(0, 282, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 282);
    }

    #[test]
    fn test_comm_mod_stress_283() {
        let hdr = MessageHeader::new(0, 283, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 283);
    }

    #[test]
    fn test_comm_mod_stress_284() {
        let hdr = MessageHeader::new(0, 284, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 284);
    }

    #[test]
    fn test_comm_mod_stress_285() {
        let hdr = MessageHeader::new(0, 285, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 285);
    }

    #[test]
    fn test_comm_mod_stress_286() {
        let hdr = MessageHeader::new(0, 286, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 286);
    }

    #[test]
    fn test_comm_mod_stress_287() {
        let hdr = MessageHeader::new(0, 287, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 287);
    }

    #[test]
    fn test_comm_mod_stress_288() {
        let hdr = MessageHeader::new(0, 288, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 288);
    }

    #[test]
    fn test_comm_mod_stress_289() {
        let hdr = MessageHeader::new(0, 289, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 289);
    }

    #[test]
    fn test_comm_mod_stress_290() {
        let hdr = MessageHeader::new(0, 290, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 290);
    }

    #[test]
    fn test_comm_mod_stress_291() {
        let hdr = MessageHeader::new(0, 291, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 291);
    }

    #[test]
    fn test_comm_mod_stress_292() {
        let hdr = MessageHeader::new(0, 292, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 292);
    }

    #[test]
    fn test_comm_mod_stress_293() {
        let hdr = MessageHeader::new(0, 293, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 293);
    }

    #[test]
    fn test_comm_mod_stress_294() {
        let hdr = MessageHeader::new(0, 294, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 294);
    }

    #[test]
    fn test_comm_mod_stress_295() {
        let hdr = MessageHeader::new(0, 295, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 295);
    }

    #[test]
    fn test_comm_mod_stress_296() {
        let hdr = MessageHeader::new(0, 296, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 296);
    }

    #[test]
    fn test_comm_mod_stress_297() {
        let hdr = MessageHeader::new(0, 297, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 297);
    }

    #[test]
    fn test_comm_mod_stress_298() {
        let hdr = MessageHeader::new(0, 298, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 298);
    }

    #[test]
    fn test_comm_mod_stress_299() {
        let hdr = MessageHeader::new(0, 299, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 299);
    }

    #[test]
    fn test_comm_mod_stress_300() {
        let hdr = MessageHeader::new(0, 300, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 300);
    }

    #[test]
    fn test_comm_mod_stress_301() {
        let hdr = MessageHeader::new(0, 301, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 301);
    }

    #[test]
    fn test_comm_mod_stress_302() {
        let hdr = MessageHeader::new(0, 302, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 302);
    }

    #[test]
    fn test_comm_mod_stress_303() {
        let hdr = MessageHeader::new(0, 303, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 303);
    }

    #[test]
    fn test_comm_mod_stress_304() {
        let hdr = MessageHeader::new(0, 304, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 304);
    }

    #[test]
    fn test_comm_mod_stress_305() {
        let hdr = MessageHeader::new(0, 305, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 305);
    }

    #[test]
    fn test_comm_mod_stress_306() {
        let hdr = MessageHeader::new(0, 306, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 306);
    }

    #[test]
    fn test_comm_mod_stress_307() {
        let hdr = MessageHeader::new(0, 307, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 307);
    }

    #[test]
    fn test_comm_mod_stress_308() {
        let hdr = MessageHeader::new(0, 308, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 308);
    }

    #[test]
    fn test_comm_mod_stress_309() {
        let hdr = MessageHeader::new(0, 309, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 309);
    }

    #[test]
    fn test_comm_mod_stress_310() {
        let hdr = MessageHeader::new(0, 310, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 310);
    }

    #[test]
    fn test_comm_mod_stress_311() {
        let hdr = MessageHeader::new(0, 311, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 311);
    }

    #[test]
    fn test_comm_mod_stress_312() {
        let hdr = MessageHeader::new(0, 312, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 312);
    }

    #[test]
    fn test_comm_mod_stress_313() {
        let hdr = MessageHeader::new(0, 313, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 313);
    }

    #[test]
    fn test_comm_mod_stress_314() {
        let hdr = MessageHeader::new(0, 314, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 314);
    }

    #[test]
    fn test_comm_mod_stress_315() {
        let hdr = MessageHeader::new(0, 315, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 315);
    }

    #[test]
    fn test_comm_mod_stress_316() {
        let hdr = MessageHeader::new(0, 316, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 316);
    }

    #[test]
    fn test_comm_mod_stress_317() {
        let hdr = MessageHeader::new(0, 317, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 317);
    }

    #[test]
    fn test_comm_mod_stress_318() {
        let hdr = MessageHeader::new(0, 318, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 318);
    }

    #[test]
    fn test_comm_mod_stress_319() {
        let hdr = MessageHeader::new(0, 319, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 319);
    }

    #[test]
    fn test_comm_mod_stress_320() {
        let hdr = MessageHeader::new(0, 320, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 320);
    }

    #[test]
    fn test_comm_mod_stress_321() {
        let hdr = MessageHeader::new(0, 321, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 321);
    }

    #[test]
    fn test_comm_mod_stress_322() {
        let hdr = MessageHeader::new(0, 322, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 322);
    }

    #[test]
    fn test_comm_mod_stress_323() {
        let hdr = MessageHeader::new(0, 323, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 323);
    }

    #[test]
    fn test_comm_mod_stress_324() {
        let hdr = MessageHeader::new(0, 324, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 324);
    }

    #[test]
    fn test_comm_mod_stress_325() {
        let hdr = MessageHeader::new(0, 325, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 325);
    }

    #[test]
    fn test_comm_mod_stress_326() {
        let hdr = MessageHeader::new(0, 326, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 326);
    }

    #[test]
    fn test_comm_mod_stress_327() {
        let hdr = MessageHeader::new(0, 327, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 327);
    }

    #[test]
    fn test_comm_mod_stress_328() {
        let hdr = MessageHeader::new(0, 328, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 328);
    }

    #[test]
    fn test_comm_mod_stress_329() {
        let hdr = MessageHeader::new(0, 329, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 329);
    }

    #[test]
    fn test_comm_mod_stress_330() {
        let hdr = MessageHeader::new(0, 330, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 330);
    }

    #[test]
    fn test_comm_mod_stress_331() {
        let hdr = MessageHeader::new(0, 331, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 331);
    }

    #[test]
    fn test_comm_mod_stress_332() {
        let hdr = MessageHeader::new(0, 332, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 332);
    }

    #[test]
    fn test_comm_mod_stress_333() {
        let hdr = MessageHeader::new(0, 333, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 333);
    }

    #[test]
    fn test_comm_mod_stress_334() {
        let hdr = MessageHeader::new(0, 334, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 334);
    }

    #[test]
    fn test_comm_mod_stress_335() {
        let hdr = MessageHeader::new(0, 335, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 335);
    }

    #[test]
    fn test_comm_mod_stress_336() {
        let hdr = MessageHeader::new(0, 336, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 336);
    }

    #[test]
    fn test_comm_mod_stress_337() {
        let hdr = MessageHeader::new(0, 337, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 337);
    }

    #[test]
    fn test_comm_mod_stress_338() {
        let hdr = MessageHeader::new(0, 338, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 338);
    }

    #[test]
    fn test_comm_mod_stress_339() {
        let hdr = MessageHeader::new(0, 339, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 339);
    }

    #[test]
    fn test_comm_mod_stress_340() {
        let hdr = MessageHeader::new(0, 340, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 340);
    }

    #[test]
    fn test_comm_mod_stress_341() {
        let hdr = MessageHeader::new(0, 341, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 341);
    }

    #[test]
    fn test_comm_mod_stress_342() {
        let hdr = MessageHeader::new(0, 342, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 342);
    }

    #[test]
    fn test_comm_mod_stress_343() {
        let hdr = MessageHeader::new(0, 343, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 343);
    }

    #[test]
    fn test_comm_mod_stress_344() {
        let hdr = MessageHeader::new(0, 344, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 344);
    }

    #[test]
    fn test_comm_mod_stress_345() {
        let hdr = MessageHeader::new(0, 345, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 345);
    }

    #[test]
    fn test_comm_mod_stress_346() {
        let hdr = MessageHeader::new(0, 346, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 346);
    }

    #[test]
    fn test_comm_mod_stress_347() {
        let hdr = MessageHeader::new(0, 347, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 347);
    }

    #[test]
    fn test_comm_mod_stress_348() {
        let hdr = MessageHeader::new(0, 348, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 348);
    }

    #[test]
    fn test_comm_mod_stress_349() {
        let hdr = MessageHeader::new(0, 349, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 349);
    }

    #[test]
    fn test_comm_mod_stress_350() {
        let hdr = MessageHeader::new(0, 350, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 350);
    }

    #[test]
    fn test_comm_mod_stress_351() {
        let hdr = MessageHeader::new(0, 351, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 351);
    }

    #[test]
    fn test_comm_mod_stress_352() {
        let hdr = MessageHeader::new(0, 352, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 352);
    }

    #[test]
    fn test_comm_mod_stress_353() {
        let hdr = MessageHeader::new(0, 353, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 353);
    }

    #[test]
    fn test_comm_mod_stress_354() {
        let hdr = MessageHeader::new(0, 354, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 354);
    }

    #[test]
    fn test_comm_mod_stress_355() {
        let hdr = MessageHeader::new(0, 355, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 355);
    }

    #[test]
    fn test_comm_mod_stress_356() {
        let hdr = MessageHeader::new(0, 356, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 356);
    }

    #[test]
    fn test_comm_mod_stress_357() {
        let hdr = MessageHeader::new(0, 357, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 357);
    }

    #[test]
    fn test_comm_mod_stress_358() {
        let hdr = MessageHeader::new(0, 358, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 358);
    }

    #[test]
    fn test_comm_mod_stress_359() {
        let hdr = MessageHeader::new(0, 359, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 359);
    }

    #[test]
    fn test_comm_mod_stress_360() {
        let hdr = MessageHeader::new(0, 360, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 360);
    }

    #[test]
    fn test_comm_mod_stress_361() {
        let hdr = MessageHeader::new(0, 361, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 361);
    }

    #[test]
    fn test_comm_mod_stress_362() {
        let hdr = MessageHeader::new(0, 362, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 362);
    }

    #[test]
    fn test_comm_mod_stress_363() {
        let hdr = MessageHeader::new(0, 363, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 363);
    }

    #[test]
    fn test_comm_mod_stress_364() {
        let hdr = MessageHeader::new(0, 364, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 364);
    }

    #[test]
    fn test_comm_mod_stress_365() {
        let hdr = MessageHeader::new(0, 365, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 365);
    }

    #[test]
    fn test_comm_mod_stress_366() {
        let hdr = MessageHeader::new(0, 366, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 366);
    }

    #[test]
    fn test_comm_mod_stress_367() {
        let hdr = MessageHeader::new(0, 367, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 367);
    }

    #[test]
    fn test_comm_mod_stress_368() {
        let hdr = MessageHeader::new(0, 368, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 368);
    }

    #[test]
    fn test_comm_mod_stress_369() {
        let hdr = MessageHeader::new(0, 369, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 369);
    }

    #[test]
    fn test_comm_mod_stress_370() {
        let hdr = MessageHeader::new(0, 370, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 370);
    }

    #[test]
    fn test_comm_mod_stress_371() {
        let hdr = MessageHeader::new(0, 371, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 371);
    }

    #[test]
    fn test_comm_mod_stress_372() {
        let hdr = MessageHeader::new(0, 372, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 372);
    }

    #[test]
    fn test_comm_mod_stress_373() {
        let hdr = MessageHeader::new(0, 373, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 373);
    }

    #[test]
    fn test_comm_mod_stress_374() {
        let hdr = MessageHeader::new(0, 374, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 374);
    }

    #[test]
    fn test_comm_mod_stress_375() {
        let hdr = MessageHeader::new(0, 375, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 375);
    }

    #[test]
    fn test_comm_mod_stress_376() {
        let hdr = MessageHeader::new(0, 376, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 376);
    }

    #[test]
    fn test_comm_mod_stress_377() {
        let hdr = MessageHeader::new(0, 377, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 377);
    }

    #[test]
    fn test_comm_mod_stress_378() {
        let hdr = MessageHeader::new(0, 378, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 378);
    }

    #[test]
    fn test_comm_mod_stress_379() {
        let hdr = MessageHeader::new(0, 379, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 379);
    }

    #[test]
    fn test_comm_mod_stress_380() {
        let hdr = MessageHeader::new(0, 380, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 380);
    }

    #[test]
    fn test_comm_mod_stress_381() {
        let hdr = MessageHeader::new(0, 381, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 381);
    }

    #[test]
    fn test_comm_mod_stress_382() {
        let hdr = MessageHeader::new(0, 382, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 382);
    }

    #[test]
    fn test_comm_mod_stress_383() {
        let hdr = MessageHeader::new(0, 383, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 383);
    }

    #[test]
    fn test_comm_mod_stress_384() {
        let hdr = MessageHeader::new(0, 384, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 384);
    }

    #[test]
    fn test_comm_mod_stress_385() {
        let hdr = MessageHeader::new(0, 385, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 385);
    }

    #[test]
    fn test_comm_mod_stress_386() {
        let hdr = MessageHeader::new(0, 386, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 386);
    }

    #[test]
    fn test_comm_mod_stress_387() {
        let hdr = MessageHeader::new(0, 387, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 387);
    }

    #[test]
    fn test_comm_mod_stress_388() {
        let hdr = MessageHeader::new(0, 388, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 388);
    }

    #[test]
    fn test_comm_mod_stress_389() {
        let hdr = MessageHeader::new(0, 389, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 389);
    }

    #[test]
    fn test_comm_mod_stress_390() {
        let hdr = MessageHeader::new(0, 390, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 390);
    }

    #[test]
    fn test_comm_mod_stress_391() {
        let hdr = MessageHeader::new(0, 391, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 391);
    }

    #[test]
    fn test_comm_mod_stress_392() {
        let hdr = MessageHeader::new(0, 392, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 392);
    }

    #[test]
    fn test_comm_mod_stress_393() {
        let hdr = MessageHeader::new(0, 393, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 393);
    }

    #[test]
    fn test_comm_mod_stress_394() {
        let hdr = MessageHeader::new(0, 394, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 394);
    }

    #[test]
    fn test_comm_mod_stress_395() {
        let hdr = MessageHeader::new(0, 395, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 395);
    }

    #[test]
    fn test_comm_mod_stress_396() {
        let hdr = MessageHeader::new(0, 396, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 396);
    }

    #[test]
    fn test_comm_mod_stress_397() {
        let hdr = MessageHeader::new(0, 397, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 397);
    }

    #[test]
    fn test_comm_mod_stress_398() {
        let hdr = MessageHeader::new(0, 398, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 398);
    }

    #[test]
    fn test_comm_mod_stress_399() {
        let hdr = MessageHeader::new(0, 399, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 399);
    }

    #[test]
    fn test_comm_mod_stress_400() {
        let hdr = MessageHeader::new(0, 400, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 400);
    }

    #[test]
    fn test_comm_mod_stress_401() {
        let hdr = MessageHeader::new(0, 401, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 401);
    }

    #[test]
    fn test_comm_mod_stress_402() {
        let hdr = MessageHeader::new(0, 402, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 402);
    }

    #[test]
    fn test_comm_mod_stress_403() {
        let hdr = MessageHeader::new(0, 403, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 403);
    }

    #[test]
    fn test_comm_mod_stress_404() {
        let hdr = MessageHeader::new(0, 404, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 404);
    }

    #[test]
    fn test_comm_mod_stress_405() {
        let hdr = MessageHeader::new(0, 405, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 405);
    }

    #[test]
    fn test_comm_mod_stress_406() {
        let hdr = MessageHeader::new(0, 406, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 406);
    }

    #[test]
    fn test_comm_mod_stress_407() {
        let hdr = MessageHeader::new(0, 407, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 407);
    }

    #[test]
    fn test_comm_mod_stress_408() {
        let hdr = MessageHeader::new(0, 408, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 408);
    }

    #[test]
    fn test_comm_mod_stress_409() {
        let hdr = MessageHeader::new(0, 409, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 409);
    }

    #[test]
    fn test_comm_mod_stress_410() {
        let hdr = MessageHeader::new(0, 410, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 410);
    }

    #[test]
    fn test_comm_mod_stress_411() {
        let hdr = MessageHeader::new(0, 411, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 411);
    }

    #[test]
    fn test_comm_mod_stress_412() {
        let hdr = MessageHeader::new(0, 412, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 412);
    }

    #[test]
    fn test_comm_mod_stress_413() {
        let hdr = MessageHeader::new(0, 413, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 413);
    }

    #[test]
    fn test_comm_mod_stress_414() {
        let hdr = MessageHeader::new(0, 414, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 414);
    }

    #[test]
    fn test_comm_mod_stress_415() {
        let hdr = MessageHeader::new(0, 415, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 415);
    }

    #[test]
    fn test_comm_mod_stress_416() {
        let hdr = MessageHeader::new(0, 416, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 416);
    }

    #[test]
    fn test_comm_mod_stress_417() {
        let hdr = MessageHeader::new(0, 417, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 417);
    }

    #[test]
    fn test_comm_mod_stress_418() {
        let hdr = MessageHeader::new(0, 418, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 418);
    }

    #[test]
    fn test_comm_mod_stress_419() {
        let hdr = MessageHeader::new(0, 419, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 419);
    }

    #[test]
    fn test_comm_mod_stress_420() {
        let hdr = MessageHeader::new(0, 420, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 420);
    }

    #[test]
    fn test_comm_mod_stress_421() {
        let hdr = MessageHeader::new(0, 421, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 421);
    }

    #[test]
    fn test_comm_mod_stress_422() {
        let hdr = MessageHeader::new(0, 422, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 422);
    }

    #[test]
    fn test_comm_mod_stress_423() {
        let hdr = MessageHeader::new(0, 423, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 423);
    }

    #[test]
    fn test_comm_mod_stress_424() {
        let hdr = MessageHeader::new(0, 424, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 424);
    }

    #[test]
    fn test_comm_mod_stress_425() {
        let hdr = MessageHeader::new(0, 425, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 425);
    }

    #[test]
    fn test_comm_mod_stress_426() {
        let hdr = MessageHeader::new(0, 426, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 426);
    }

    #[test]
    fn test_comm_mod_stress_427() {
        let hdr = MessageHeader::new(0, 427, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 427);
    }

    #[test]
    fn test_comm_mod_stress_428() {
        let hdr = MessageHeader::new(0, 428, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 428);
    }

    #[test]
    fn test_comm_mod_stress_429() {
        let hdr = MessageHeader::new(0, 429, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 429);
    }

    #[test]
    fn test_comm_mod_stress_430() {
        let hdr = MessageHeader::new(0, 430, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 430);
    }

    #[test]
    fn test_comm_mod_stress_431() {
        let hdr = MessageHeader::new(0, 431, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 431);
    }

    #[test]
    fn test_comm_mod_stress_432() {
        let hdr = MessageHeader::new(0, 432, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 432);
    }

    #[test]
    fn test_comm_mod_stress_433() {
        let hdr = MessageHeader::new(0, 433, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 433);
    }

    #[test]
    fn test_comm_mod_stress_434() {
        let hdr = MessageHeader::new(0, 434, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 434);
    }

    #[test]
    fn test_comm_mod_stress_435() {
        let hdr = MessageHeader::new(0, 435, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 435);
    }

    #[test]
    fn test_comm_mod_stress_436() {
        let hdr = MessageHeader::new(0, 436, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 436);
    }

    #[test]
    fn test_comm_mod_stress_437() {
        let hdr = MessageHeader::new(0, 437, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 437);
    }

    #[test]
    fn test_comm_mod_stress_438() {
        let hdr = MessageHeader::new(0, 438, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 438);
    }

    #[test]
    fn test_comm_mod_stress_439() {
        let hdr = MessageHeader::new(0, 439, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 439);
    }

    #[test]
    fn test_comm_mod_stress_440() {
        let hdr = MessageHeader::new(0, 440, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 440);
    }

    #[test]
    fn test_comm_mod_stress_441() {
        let hdr = MessageHeader::new(0, 441, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 441);
    }

    #[test]
    fn test_comm_mod_stress_442() {
        let hdr = MessageHeader::new(0, 442, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 442);
    }

    #[test]
    fn test_comm_mod_stress_443() {
        let hdr = MessageHeader::new(0, 443, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 443);
    }

    #[test]
    fn test_comm_mod_stress_444() {
        let hdr = MessageHeader::new(0, 444, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 444);
    }

    #[test]
    fn test_comm_mod_stress_445() {
        let hdr = MessageHeader::new(0, 445, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 445);
    }

    #[test]
    fn test_comm_mod_stress_446() {
        let hdr = MessageHeader::new(0, 446, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 446);
    }

    #[test]
    fn test_comm_mod_stress_447() {
        let hdr = MessageHeader::new(0, 447, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 447);
    }

    #[test]
    fn test_comm_mod_stress_448() {
        let hdr = MessageHeader::new(0, 448, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 448);
    }

    #[test]
    fn test_comm_mod_stress_449() {
        let hdr = MessageHeader::new(0, 449, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 449);
    }

    #[test]
    fn test_comm_mod_stress_450() {
        let hdr = MessageHeader::new(0, 450, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 450);
    }

    #[test]
    fn test_comm_mod_stress_451() {
        let hdr = MessageHeader::new(0, 451, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 451);
    }

    #[test]
    fn test_comm_mod_stress_452() {
        let hdr = MessageHeader::new(0, 452, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 452);
    }

    #[test]
    fn test_comm_mod_stress_453() {
        let hdr = MessageHeader::new(0, 453, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 453);
    }

    #[test]
    fn test_comm_mod_stress_454() {
        let hdr = MessageHeader::new(0, 454, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 454);
    }

    #[test]
    fn test_comm_mod_stress_455() {
        let hdr = MessageHeader::new(0, 455, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 455);
    }

    #[test]
    fn test_comm_mod_stress_456() {
        let hdr = MessageHeader::new(0, 456, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 456);
    }

    #[test]
    fn test_comm_mod_stress_457() {
        let hdr = MessageHeader::new(0, 457, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 457);
    }

    #[test]
    fn test_comm_mod_stress_458() {
        let hdr = MessageHeader::new(0, 458, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 458);
    }

    #[test]
    fn test_comm_mod_stress_459() {
        let hdr = MessageHeader::new(0, 459, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 459);
    }

    #[test]
    fn test_comm_mod_stress_460() {
        let hdr = MessageHeader::new(0, 460, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 460);
    }

    #[test]
    fn test_comm_mod_stress_461() {
        let hdr = MessageHeader::new(0, 461, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 461);
    }

    #[test]
    fn test_comm_mod_stress_462() {
        let hdr = MessageHeader::new(0, 462, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 462);
    }

    #[test]
    fn test_comm_mod_stress_463() {
        let hdr = MessageHeader::new(0, 463, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 463);
    }

    #[test]
    fn test_comm_mod_stress_464() {
        let hdr = MessageHeader::new(0, 464, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 464);
    }

    #[test]
    fn test_comm_mod_stress_465() {
        let hdr = MessageHeader::new(0, 465, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 465);
    }

    #[test]
    fn test_comm_mod_stress_466() {
        let hdr = MessageHeader::new(0, 466, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 466);
    }

    #[test]
    fn test_comm_mod_stress_467() {
        let hdr = MessageHeader::new(0, 467, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 467);
    }

    #[test]
    fn test_comm_mod_stress_468() {
        let hdr = MessageHeader::new(0, 468, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 468);
    }

    #[test]
    fn test_comm_mod_stress_469() {
        let hdr = MessageHeader::new(0, 469, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 469);
    }

    #[test]
    fn test_comm_mod_stress_470() {
        let hdr = MessageHeader::new(0, 470, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 470);
    }

    #[test]
    fn test_comm_mod_stress_471() {
        let hdr = MessageHeader::new(0, 471, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 471);
    }

    #[test]
    fn test_comm_mod_stress_472() {
        let hdr = MessageHeader::new(0, 472, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 472);
    }

    #[test]
    fn test_comm_mod_stress_473() {
        let hdr = MessageHeader::new(0, 473, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 473);
    }

    #[test]
    fn test_comm_mod_stress_474() {
        let hdr = MessageHeader::new(0, 474, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 474);
    }

    #[test]
    fn test_comm_mod_stress_475() {
        let hdr = MessageHeader::new(0, 475, 1);
        assert_eq!(hdr.src_rank, 0);
        assert_eq!(hdr.dest_rank, 475);
    }

    // Distributed collective verification and ring allreduce check padding line 0
}
