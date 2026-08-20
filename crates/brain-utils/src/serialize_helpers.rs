//! # Serialization Helpers
//!
//! Provides hex encoding/decoding, Base64 encoding/decoding (RFC 4648),
//! and raw binary buffer operations.

use crate::core::{UtilsError, UtilsResult};

/// Encodes byte slice into lowercase hexadecimal string.
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Decodes hexadecimal string into byte vector.
#[allow(clippy::manual_is_multiple_of)]
pub fn hex_to_bytes(hex: &str) -> UtilsResult<Vec<u8>> {
    let clean = hex.trim();
    if clean.len() % 2 != 0 {
        return Err(UtilsError::ParseError(
            "Hex string must have even length".to_string(),
        ));
    }
    let mut bytes = Vec::with_capacity(clean.len() / 2);
    let chars: Vec<char> = clean.chars().collect();
    for i in (0..chars.len()).step_by(2) {
        let hi = chars[i]
            .to_digit(16)
            .ok_or_else(|| UtilsError::ParseError(format!("Invalid hex char: {}", chars[i])))?;
        let lo = chars[i + 1]
            .to_digit(16)
            .ok_or_else(|| UtilsError::ParseError(format!("Invalid hex char: {}", chars[i + 1])))?;
        bytes.push(((hi << 4) | lo) as u8);
    }
    Ok(bytes)
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes byte slice into standard Base64 string with padding.
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut chunks = bytes.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
        out.push(BASE64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[(n & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    if rem.len() == 1 {
        let n = (rem[0] as u32) << 16;
        out.push(BASE64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem.len() == 2 {
        let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
        out.push(BASE64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Decodes standard Base64 string into byte vector.
#[allow(clippy::manual_is_multiple_of)]
pub fn base64_decode(s: &str) -> UtilsResult<Vec<u8>> {
    let clean: Vec<u8> = s.bytes().filter(|&b| !b.is_ascii_whitespace()).collect();
    if clean.len() % 4 != 0 {
        return Err(UtilsError::ParseError(
            "Base64 string length must be multiple of 4".to_string(),
        ));
    }
    let mut out = Vec::new();
    for chunk in clean.chunks_exact(4) {
        let mut buf = [0u8; 4];
        let mut pad_count = 0;
        for i in 0..4 {
            let b = chunk[i];
            if b == b'=' {
                pad_count += 1;
                buf[i] = 0;
            } else if let Some(idx) = BASE64_ALPHABET.iter().position(|&x| x == b) {
                buf[i] = idx as u8;
            } else {
                return Err(UtilsError::ParseError(format!(
                    "Invalid Base64 byte: {}",
                    b
                )));
            }
        }
        let n = ((buf[0] as u32) << 18)
            | ((buf[1] as u32) << 12)
            | ((buf[2] as u32) << 6)
            | (buf[3] as u32);
        out.push(((n >> 16) & 0xFF) as u8);
        if pad_count < 2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if pad_count < 1 {
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_hex_and_base64_1() {
        let data = b"brain deep learning";
        let hex = bytes_to_hex(data);
        assert_eq!(hex_to_bytes(&hex).unwrap(), data);

        let b64 = base64_encode(data);
        assert_eq!(base64_decode(&b64).unwrap(), data);

        assert_eq!(base64_encode(b"hello world"), "aGVsbG8gd29ybGQ=");
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
    }
}
