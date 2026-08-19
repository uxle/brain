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
}
