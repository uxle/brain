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
        archive.extend_from_slice(&0_u16.to_le_bytes()); // flags
        archive.extend_from_slice(&0_u16.to_le_bytes()); // compression method (store)
        archive.extend_from_slice(&0_u32.to_le_bytes()); // mod time/date
        archive.extend_from_slice(&c_crc.to_le_bytes());
        archive.extend_from_slice(&size.to_le_bytes()); // compressed size
        archive.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        archive.extend_from_slice(&name_len.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes()); // extra len
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
}
