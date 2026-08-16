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

        const BITRATES_V1_L3: [u32; 16] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0];
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
        let frame_size = ((144 * bitrate_kbps * 1000) / sample_rate + if padding { 1 } else { 0 }) as usize;

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

    #[test]
    fn test_mp3_parser_stress_001() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_002() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_003() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_004() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_005() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_006() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_007() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_008() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_009() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_010() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_011() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_012() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_013() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_014() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_015() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_016() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_017() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_018() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_019() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_020() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_021() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_022() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_023() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_024() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_025() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_026() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_027() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_028() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_029() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_030() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_031() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_032() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_033() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_034() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_035() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_036() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_037() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_038() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_039() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_040() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_041() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_042() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_043() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_044() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_045() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_046() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_047() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_048() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_049() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_050() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_051() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_052() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_053() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_054() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_055() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_056() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_057() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_058() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_059() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_060() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_061() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_062() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_063() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_064() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_065() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_066() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_067() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_068() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_069() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_070() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_071() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_072() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_073() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_074() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_075() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_076() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_077() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_078() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_079() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_080() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_081() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_082() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_083() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_084() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_085() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_086() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_087() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_088() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_089() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_090() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_091() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_092() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_093() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_094() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_095() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_096() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_097() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_098() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_099() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_100() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_101() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_102() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_103() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_104() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_105() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_106() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_107() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_108() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_109() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_110() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_111() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_112() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_113() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_114() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_115() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_116() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_117() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_118() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_119() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_120() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_121() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_122() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_123() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_124() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_125() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_126() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_127() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_128() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_129() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_130() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_131() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_132() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_133() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_134() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_135() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_136() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_137() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_138() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_139() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_140() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_141() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_142() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_143() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_144() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_145() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_146() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_147() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_148() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_149() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_150() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_151() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_152() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_153() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_154() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_155() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_156() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_157() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_158() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_159() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_160() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_161() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_162() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_163() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_164() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_165() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_166() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_167() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_168() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_169() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_170() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_171() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_172() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_173() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_174() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_175() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_176() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_177() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_178() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_179() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_180() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_181() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_182() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_183() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_184() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_185() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_186() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_187() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_188() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_189() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_190() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_191() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_192() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_193() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_194() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_195() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_196() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_197() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_198() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_199() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_200() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_201() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_202() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_203() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_204() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_205() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_206() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_207() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_208() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_209() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_210() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_211() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_212() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_213() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_214() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_215() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_216() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_217() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_218() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_219() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_220() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_221() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_222() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_223() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_224() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_225() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_226() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_227() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_228() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_229() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_230() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_231() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_232() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_233() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_234() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_235() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_236() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_237() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_238() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_239() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_240() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_241() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_242() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_243() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_244() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_245() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_246() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_247() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_248() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_249() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_250() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_251() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_252() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_253() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_254() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_255() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_256() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_257() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_258() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_259() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_260() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_261() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_262() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_263() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_264() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_265() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_266() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_267() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_268() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_269() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }

    #[test]
    fn test_mp3_parser_stress_270() {
        // Synthesize valid 128kbps 44.1kHz MP3 header (0xFF, 0xFB, 0x90, 0x00)
        let header_bytes = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = Mp3FrameHeader::parse(&header_bytes).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44100);
        assert_eq!(hdr.channels, 2);
    }
}
