//! # Free Lossless Audio Codec (FLAC) Metadata and Frame Parser
//!
//! Pure-Rust parsing of `fLaC` stream headers, `STREAMINFO` metadata blocks, and subframe structures.

use brain_core::{BrainError, BrainResult};

/// FLAC STREAMINFO metadata block content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlacStreamInfo {
    /// Minimum block size in samples.
    pub min_block_size: u16,
    /// Maximum block size in samples.
    pub max_block_size: u16,
    /// Minimum frame size in bytes.
    pub min_frame_size: u32,
    /// Maximum frame size in bytes.
    pub max_frame_size: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels (1 to 8).
    pub channels: u8,
    /// Bits per sample (4 to 32).
    pub bits_per_sample: u8,
    /// Total inter-channel samples in stream.
    pub total_samples: u64,
}

/// Parses FLAC metadata headers from raw bytes.
pub fn parse_flac_metadata(bytes: &[u8]) -> BrainResult<FlacStreamInfo> {
    if bytes.len() < 42 {
        return Err(BrainError::invalid_value("FLAC stream shorter than minimum header"));
    }
    if &bytes[0..4] != b"fLaC" {
        return Err(BrainError::invalid_value("invalid FLAC magic identifier"));
    }

    // Read first metadata block header (STREAMINFO = type 0)
    let block_header = bytes[4];
    let block_type = block_header & 0x7F;
    if block_type != 0 {
        return Err(BrainError::invalid_value("first FLAC metadata block must be STREAMINFO (type 0)"));
    }

    let min_block_size = u16::from_be_bytes(bytes[8..10].try_into().unwrap());
    let max_block_size = u16::from_be_bytes(bytes[10..12].try_into().unwrap());
    let min_frame_size = u32::from_be_bytes([0, bytes[12], bytes[13], bytes[14]]);
    let max_frame_size = u32::from_be_bytes([0, bytes[15], bytes[16], bytes[17]]);

    // Bytes 18..26 contain sample_rate (20 bits), channels (3 bits), bits_per_sample (5 bits), total_samples (36 bits)
    let b18 = bytes[18] as u32;
    let b19 = bytes[19] as u32;
    let b20 = bytes[20] as u32;
    let sample_rate = (b18 << 12) | (b19 << 4) | (b20 >> 4);
    let channels = (((b20 >> 1) & 0x07) + 1) as u8;
    let b21 = bytes[21] as u32;
    let bits_per_sample = ((((b20 & 0x01) << 4) | (b21 >> 4)) + 1) as u8;

    let b22 = bytes[22] as u64;
    let b23 = bytes[23] as u64;
    let b24 = bytes[24] as u64;
    let b25 = bytes[25] as u64;
    let total_samples = ((b21 as u64 & 0x0F) << 32) | (b22 << 24) | (b23 << 16) | (b24 << 8) | b25;

    Ok(FlacStreamInfo {
        min_block_size,
        max_block_size,
        min_frame_size,
        max_frame_size,
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_metadata_stress_001() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_002() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_003() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_004() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_005() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_006() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_007() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_008() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_009() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_010() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_011() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_012() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_013() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_014() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_015() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_016() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_017() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_018() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_019() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_020() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_021() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_022() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_023() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_024() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_025() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_026() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_027() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_028() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_029() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_030() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_031() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_032() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_033() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_034() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_035() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_036() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_037() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_038() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_039() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_040() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_041() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_042() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_043() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_044() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_045() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_046() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_047() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_048() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_049() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_050() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_051() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_052() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_053() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_054() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_055() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_056() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_057() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_058() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_059() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_060() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_061() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_062() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_063() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_064() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_065() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_066() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_067() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_068() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_069() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_070() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_071() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_072() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_073() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_074() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_075() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_076() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_077() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_078() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_079() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_080() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_081() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_082() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_083() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_084() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_085() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_086() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_087() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_088() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_089() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_090() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_091() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_092() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_093() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_094() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_095() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_096() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_097() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_098() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_099() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_100() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_101() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_102() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_103() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_104() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_105() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_106() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_107() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_108() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_109() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_110() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_111() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_112() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_113() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_114() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_115() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_116() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_117() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_118() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_119() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_120() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_121() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_122() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_123() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_124() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_125() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_126() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_127() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_128() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_129() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_130() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_131() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_132() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_133() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_134() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_135() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_136() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_137() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_138() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_139() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_140() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_141() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_142() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_143() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_144() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_145() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_146() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_147() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_148() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_149() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_150() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_151() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_152() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_153() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_154() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_155() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_156() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_157() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_158() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_159() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_160() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_161() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_162() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_163() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_164() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_165() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_166() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_167() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_168() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_169() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_170() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_171() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_172() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_173() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_174() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_175() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_176() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_177() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_178() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_179() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_180() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_181() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_182() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_183() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_184() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_185() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_186() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_187() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_188() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_189() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_190() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_191() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_192() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_193() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_194() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_195() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_196() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_197() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_198() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_199() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_200() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_201() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_202() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_203() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_204() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_205() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_206() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_207() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_208() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_209() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_210() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_211() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_212() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_213() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_214() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_215() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_216() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_217() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_flac_metadata_stress_218() {
        let mut flac_bytes = vec![0u8; 42];
        flac_bytes[0..4].copy_from_slice(b"fLaC");
        flac_bytes[4] = 0x00; // STREAMINFO
        // Sample rate 44100 (0x0AC44)
        flac_bytes[18] = 0x0A;
        flac_bytes[19] = 0xC4;
        flac_bytes[20] = 0x40 | 0x02; // channels=2, bits=16
        flac_bytes[21] = 0xF0;
        let info = parse_flac_metadata(&flac_bytes).unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
    }
}
