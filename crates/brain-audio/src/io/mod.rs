//! # Audio I/O and Format Codecs
//!
//! Pure-Rust audio format encoders and decoders:
//! * [`wav`] - RIFF/WAV lossless PCM and IEEE float parser and writer
//! * [`mp3`] - MPEG Audio Layer III frame parsing and synchronization
//! * [`flac`] - Free Lossless Audio Codec (FLAC) stream decoding

pub mod wav;
pub mod mp3;
pub mod flac;

pub use wav::{read_wav, write_wav, WavHeader};
pub use mp3::{Mp3FrameHeader, parse_mp3_frames};
pub use flac::{FlacStreamInfo, parse_flac_metadata};

use brain_core::BrainResult;
use crate::core::AudioBuffer;

/// Reads an audio buffer from raw file bytes with auto-format detection.
pub fn read_audio_bytes(bytes: &[u8]) -> BrainResult<AudioBuffer> {
    if bytes.starts_with(b"RIFF") {
        read_wav(bytes)
    } else {
        Err(brain_core::BrainError::invalid_value("unrecognized or unsupported audio format header"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
