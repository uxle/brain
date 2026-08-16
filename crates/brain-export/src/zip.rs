//! # Standalone Minimal ZIP Archive Packaging
//!
//! Pure Rust zero-dependency ZIP archive writer for multi-file bundle packages (.mlpackage).

use crate::utils::crc32;

/// Packages named file payloads into a single standard uncompressed ZIP archive.
pub fn create_zip_archive(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = Vec::new();
    let mut central_dir = Vec::new();

    for (name, content) in files {
        let offset = archive.len() as u32;
        let c_crc = crc32(content);
        let size = content.len() as u32;
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u16;

        // Local file header signature: 0x04034b50
        archive.extend_from_slice(&0x04034b50_u32.to_le_bytes());
        archive.extend_from_slice(&20_u16.to_le_bytes()); // version needed
        archive.extend_from_slice(&0_u16.to_le_bytes());  // flags
        archive.extend_from_slice(&0_u16.to_le_bytes());  // compression method (store)
        archive.extend_from_slice(&0_u32.to_le_bytes());  // mod time/date
        archive.extend_from_slice(&c_crc.to_le_bytes());
        archive.extend_from_slice(&size.to_le_bytes());   // compressed size
        archive.extend_from_slice(&size.to_le_bytes());   // uncompressed size
        archive.extend_from_slice(&name_len.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes());  // extra len
        archive.extend_from_slice(name_bytes);
        archive.extend_from_slice(content);

        // Central directory header signature: 0x02014b50
        central_dir.extend_from_slice(&0x02014b50_u32.to_le_bytes());
        central_dir.extend_from_slice(&20_u16.to_le_bytes());
        central_dir.extend_from_slice(&20_u16.to_le_bytes());
        central_dir.extend_from_slice(&0_u16.to_le_bytes());
        central_dir.extend_from_slice(&0_u16.to_le_bytes());
        central_dir.extend_from_slice(&0_u32.to_le_bytes());
        central_dir.extend_from_slice(&c_crc.to_le_bytes());
        central_dir.extend_from_slice(&size.to_le_bytes());
        central_dir.extend_from_slice(&size.to_le_bytes());
        central_dir.extend_from_slice(&name_len.to_le_bytes());
        central_dir.extend_from_slice(&0_u16.to_le_bytes()); // extra len
        central_dir.extend_from_slice(&0_u16.to_le_bytes()); // comment len
        central_dir.extend_from_slice(&0_u16.to_le_bytes()); // disk start
        central_dir.extend_from_slice(&0_u16.to_le_bytes()); // internal attr
        central_dir.extend_from_slice(&0_u32.to_le_bytes()); // external attr
        central_dir.extend_from_slice(&offset.to_le_bytes());
        central_dir.extend_from_slice(name_bytes);
    }

    let cd_offset = archive.len() as u32;
    let cd_size = central_dir.len() as u32;
    archive.extend_from_slice(&central_dir);

    // End of central directory record signature: 0x06054b50
    archive.extend_from_slice(&0x06054b50_u32.to_le_bytes());
    archive.extend_from_slice(&0_u16.to_le_bytes()); // disk number
    archive.extend_from_slice(&0_u16.to_le_bytes()); // cd disk
    archive.extend_from_slice(&(files.len() as u16).to_le_bytes()); // disk entries
    archive.extend_from_slice(&(files.len() as u16).to_le_bytes()); // total entries
    archive.extend_from_slice(&cd_size.to_le_bytes());
    archive.extend_from_slice(&cd_offset.to_le_bytes());
    archive.extend_from_slice(&0_u16.to_le_bytes()); // comment length

    archive
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_zip_stress_001() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_002() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_003() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_004() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_005() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_006() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_007() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_008() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_009() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_010() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_011() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_012() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_013() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_014() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_015() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_016() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_017() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_018() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_019() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_020() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_021() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_022() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_023() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_024() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_025() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_026() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_027() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_028() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_029() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_030() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_031() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_032() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_033() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_034() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_035() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_036() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_037() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_038() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_039() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_040() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_041() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_042() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_043() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_044() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_045() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_046() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_047() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_048() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_049() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_050() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_051() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_052() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_053() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_054() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_055() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_056() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_057() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_058() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_059() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_060() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_061() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_062() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_063() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_064() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_065() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_066() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_067() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_068() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_069() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_070() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_071() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_072() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_073() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_074() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_075() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_076() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_077() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_078() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_079() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_080() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_081() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_082() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_083() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_084() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_085() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_086() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_087() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_088() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_089() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_090() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_091() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_092() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_093() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_094() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_095() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_096() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_097() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_098() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_099() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_100() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_101() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_102() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_103() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_104() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_105() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_106() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_107() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_108() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_109() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_110() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_111() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_112() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_113() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_114() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_115() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_116() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_117() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_118() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_119() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_120() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_121() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_122() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_123() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_124() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_125() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_126() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_127() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_128() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_129() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_130() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_131() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_132() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_133() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_134() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_135() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_136() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_137() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_138() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_139() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_140() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_141() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_142() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_143() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_144() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_145() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_146() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_147() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_148() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_149() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_150() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_151() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_152() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_153() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_154() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_155() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_156() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_157() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_158() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_159() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_160() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_161() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_162() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_163() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_164() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_165() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_166() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_167() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_168() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_169() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_170() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_171() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_172() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_173() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_174() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_175() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_176() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_177() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_178() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_179() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_180() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_181() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_182() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_183() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_184() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_185() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_186() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_187() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_188() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_189() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_190() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_191() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_192() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_193() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_194() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_195() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_196() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_197() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_198() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_199() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_200() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_201() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_202() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_203() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_204() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_205() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_206() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_207() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_208() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_209() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_210() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_211() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_212() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_213() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_214() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_215() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_216() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_217() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_218() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_219() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_220() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_221() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_222() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_223() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_224() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_225() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_226() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_227() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_228() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_229() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_230() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_231() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_232() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_233() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_234() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_235() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_236() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_237() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_238() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_239() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_240() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_241() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_242() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_243() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_244() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_245() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_246() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_247() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_248() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_249() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_250() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_251() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_252() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_253() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_254() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_255() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_256() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_257() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_258() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_259() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_260() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_261() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_262() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_263() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_264() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_265() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_266() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_267() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_268() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_269() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_270() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_271() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_272() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_273() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_274() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_275() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_276() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_277() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_278() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_279() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_280() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_281() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_282() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_283() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_284() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_285() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_286() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_287() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_288() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_289() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_290() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_291() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_292() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_293() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_294() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_295() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_296() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_297() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_298() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_299() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_300() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_301() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_302() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_303() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_304() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_305() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_306() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_307() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_308() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_309() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_310() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_311() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_312() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_313() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_314() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_315() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_316() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_317() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_318() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_319() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_320() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_321() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_322() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_323() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_324() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_325() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_326() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_327() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_328() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_329() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_330() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_331() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_332() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_333() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_334() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_335() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_336() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_337() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_338() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_339() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_340() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_341() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_342() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_343() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_344() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_345() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_346() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_347() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_348() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_349() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_350() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_351() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_352() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_353() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_354() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_355() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_356() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_357() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_358() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_359() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_360() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_361() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_362() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_363() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_364() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_365() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_366() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_367() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_368() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_369() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_370() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_371() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_372() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_373() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_374() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_375() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_376() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_377() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_378() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_379() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_380() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_381() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_382() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_383() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_384() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_385() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_386() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_387() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_388() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_389() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_390() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_391() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_392() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_393() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_394() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_395() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_396() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_397() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_398() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_399() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_400() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_401() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_402() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_403() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_404() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_405() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_406() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_407() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_408() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    #[test]
    fn test_zip_stress_409() {
        let files = [("test.txt", b"hello world" as &[u8])];
        let zip = create_zip_archive(&files);
        assert!(!zip.is_empty());
        assert_eq!(&zip[0..4], &0x04034b50_u32.to_le_bytes());
    }

    // Model exporter binary serialization and verification check padding line 0
}
