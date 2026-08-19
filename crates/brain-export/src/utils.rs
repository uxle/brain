//! # Binary Serialization Utilities & Codecs
//!
//! Hand-rolled Protocol Buffers / FlatBuffers varint encoding, string encoding, and CRC32.

/// Encodes an unsigned integer as ULEB128 varint.
pub fn encode_uleb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Aligns a byte buffer to 4-byte boundaries with zero padding.
pub fn align4(buffer: &mut Vec<u8>) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
}

/// Computes standard IEEE 802.3 CRC32 checksum.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
