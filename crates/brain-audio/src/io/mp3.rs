//! # Minimal MPEG-1/2 Audio Layer III (MP3) Parser and Decoder
//!
//! Pure-Rust header synchronization, bit-rate indexing, and frame decoding helpers.

use brain_core::{BrainError, BrainResult};

/// MP3 Frame Header information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp3FrameHeader {
    /// MPEG Audio Version (1 or 2).
    pub version: u8,
    /// Layer (1, 2, or 3).
    pub layer: u8,
    /// Bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Whether padding is enabled for this frame.
    pub padding: bool,
    /// Channel mode (stereo, joint stereo, dual channel, single channel).
    pub channels: u8,
    /// Frame size in bytes including header.
    pub frame_size: usize,
}

impl Mp3FrameHeader {
    /// Attempts to parse an MP3 frame header from a 4-byte slice.
    pub fn parse(header_bytes: &[u8; 4]) -> BrainResult<Self> {
        if header_bytes[0] != 0xFF || (header_bytes[1] & 0xE0) != 0xE0 {
            return Err(BrainError::invalid_value("invalid MP3 sync word"));
        }
        let version_bits = (header_bytes[1] >> 3) & 0x03;
        let layer_bits = (header_bytes[1] >> 1) & 0x03;
        let bitrate_idx = (header_bytes[2] >> 4) & 0x0F;
        let sample_rate_idx = (header_bytes[2] >> 2) & 0x03;
        let padding_bit = (header_bytes[2] >> 1) & 0x01;
        let channel_bits = (header_bytes[3] >> 6) & 0x03;

        let version = match version_bits {
            3 => 1,
            2 => 2,
            _ => return Err(BrainError::invalid_value("unsupported MPEG version")),
        };
        let layer = match layer_bits {
            1 => 3,
            2 => 2,
            3 => 1,
            _ => return Err(BrainError::invalid_value("unsupported MPEG layer")),
        };

        const BITRATES_V1_L3: [u32; 16] = [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
        ];
        let bitrate_kbps = BITRATES_V1_L3[bitrate_idx as usize];
        if bitrate_kbps == 0 {
            return Err(BrainError::invalid_value("invalid bitrate index"));
        }

        const SAMPLE_RATES_V1: [u32; 4] = [44100, 48000, 32000, 0];
        let sample_rate = SAMPLE_RATES_V1[sample_rate_idx as usize];
        if sample_rate == 0 {
            return Err(BrainError::invalid_value("invalid sample rate index"));
        }

        let padding = padding_bit == 1;
        let channels = if channel_bits == 3 { 1 } else { 2 };
        let frame_size =
            ((144 * bitrate_kbps * 1000) / sample_rate + if padding { 1 } else { 0 }) as usize;

        Ok(Mp3FrameHeader {
            version,
            layer,
            bitrate_kbps,
            sample_rate,
            padding,
            channels,
            frame_size,
        })
    }
}

/// Scans and extracts all valid MP3 frame headers from an audio stream byte buffer.
pub fn parse_mp3_frames(bytes: &[u8]) -> Vec<Mp3FrameHeader> {
    let mut frames = Vec::new();
    let mut cursor = 0;
    while cursor + 4 <= bytes.len() {
        if bytes[cursor] == 0xFF && (bytes[cursor + 1] & 0xE0) == 0xE0 {
            let header_slice: [u8; 4] = bytes[cursor..cursor + 4].try_into().unwrap();
            if let Ok(hdr) = Mp3FrameHeader::parse(&header_slice) {
                let size = hdr.frame_size;
                frames.push(hdr);
                cursor += size.max(4);
                continue;
            }
        }
        cursor += 1;
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
}
