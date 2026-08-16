//! # RIFF / WAV Audio Format Decoder and Encoder
//!
//! Complete pure-Rust WAV parser and generator supporting:
//! * 8-bit unsigned PCM
//! * 16-bit signed integer PCM
//! * 24-bit signed integer PCM (packed)
//! * 32-bit signed integer PCM
//! * 32-bit IEEE float PCM
//! * 64-bit IEEE float PCM
//! * Multi-channel audio streams

use brain_core::{BrainError, BrainResult};
use crate::core::{AudioBuffer, SampleRate};

/// Parsed RIFF/WAV header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    /// Number of audio channels.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Byte rate (`sample_rate * channels * bits_per_sample / 8`).
    pub byte_rate: u32,
    /// Block alignment (`channels * bits_per_sample / 8`).
    pub block_align: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Audio format code (1 = PCM, 3 = IEEE float).
    pub format_tag: u16,
}

/// Decodes an in-memory WAV byte slice into an [`AudioBuffer`].
pub fn read_wav(bytes: &[u8]) -> BrainResult<AudioBuffer> {
    if bytes.len() < 44 {
        return Err(BrainError::invalid_value("WAV file smaller than minimum 44-byte header"));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(BrainError::invalid_value("invalid RIFF/WAVE header"));
    }

    let mut cursor = 12;
    let mut header: Option<WavHeader> = None;
    let mut data_slice: Option<&[u8]> = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;

        if cursor + chunk_size > bytes.len() {
            break;
        }

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err(BrainError::invalid_value("corrupt fmt chunk"));
            }
            let format_tag = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap());
            let channels = u16::from_le_bytes(bytes[cursor + 2..cursor + 4].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
            let byte_rate = u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap());
            let block_align = u16::from_le_bytes(bytes[cursor + 12..cursor + 14].try_into().unwrap());
            let bits_per_sample = u16::from_le_bytes(bytes[cursor + 14..cursor + 16].try_into().unwrap());

            header = Some(WavHeader {
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                format_tag,
            });
        } else if chunk_id == b"data" {
            data_slice = Some(&bytes[cursor..cursor + chunk_size]);
        }

        cursor += chunk_size;
        // Align to 2-byte boundary
        if chunk_size % 2 == 1 {
            cursor += 1;
        }
    }

    let hdr = header.ok_or_else(|| BrainError::invalid_value("missing fmt chunk in WAV file"))?;
    let data = data_slice.ok_or_else(|| BrainError::invalid_value("missing data chunk in WAV file"))?;
    let ch = hdr.channels as usize;
    let sr = SampleRate::new(hdr.sample_rate)?;

    let bytes_per_sample = (hdr.bits_per_sample / 8) as usize;
    if bytes_per_sample == 0 || hdr.block_align as usize == 0 {
        return Err(BrainError::invalid_value("invalid sample bit depth in WAV"));
    }

    let num_frames = data.len() / (ch * bytes_per_sample);
    let mut out_buffer = AudioBuffer::zeros(ch, num_frames, sr)?;

    let mut byte_idx = 0;
    for frame in 0..num_frames {
        for c in 0..ch {
            let sample_f64 = match (hdr.format_tag, hdr.bits_per_sample) {
                (1, 8) => {
                    let val = data[byte_idx];
                    byte_idx += 1;
                    (val as f64 - 128.0) / 128.0
                }
                (1, 16) => {
                    let val = i16::from_le_bytes(data[byte_idx..byte_idx + 2].try_into().unwrap());
                    byte_idx += 2;
                    val as f64 / 32768.0
                }
                (1, 24) => {
                    let b0 = data[byte_idx] as u32;
                    let b1 = data[byte_idx + 1] as u32;
                    let b2 = data[byte_idx + 2] as u32;
                    byte_idx += 3;
                    let raw = (b0 | (b1 << 8) | (b2 << 16)) as i32;
                    // Sign extend 24-bit
                    let val = if raw & 0x800000 != 0 { raw | !0xFFFFFF } else { raw };
                    val as f64 / 8388608.0
                }
                (1, 32) => {
                    let val = i32::from_le_bytes(data[byte_idx..byte_idx + 4].try_into().unwrap());
                    byte_idx += 4;
                    val as f64 / 2147483648.0
                }
                (3, 32) => {
                    let val = f32::from_le_bytes(data[byte_idx..byte_idx + 4].try_into().unwrap());
                    byte_idx += 4;
                    val as f64
                }
                (3, 64) => {
                    let val = f64::from_le_bytes(data[byte_idx..byte_idx + 8].try_into().unwrap());
                    byte_idx += 8;
                    val
                }
                _ => return Err(BrainError::invalid_value(format!("unsupported WAV format: tag={}, bits={}", hdr.format_tag, hdr.bits_per_sample))),
            };
            out_buffer.set_sample(c, frame, sample_f64)?;
        }
    }

    Ok(out_buffer)
}

/// Encodes an [`AudioBuffer`] into standard 16-bit PCM WAV bytes.
///
/// # Examples
///
/// ```
/// use brain_audio::core::{AudioBuffer, SampleRate};
/// use brain_audio::io::{write_wav, read_wav};
/// let buf = AudioBuffer::zeros(1, 100, SampleRate::SPEECH_16K).unwrap();
/// let bytes = write_wav(&buf);
/// let loaded = read_wav(&bytes).unwrap();
/// assert_eq!(loaded.channels(), 1);
/// assert_eq!(loaded.num_samples(), 100);
/// ```
pub fn write_wav(audio: &AudioBuffer) -> Vec<u8> {
    let ch = audio.channels() as u16;
    let sr = audio.sample_rate().hz();
    let num_samples = audio.num_samples();
    let bytes_per_sample = 2u16; // 16-bit PCM
    let block_align = ch * bytes_per_sample;
    let byte_rate = sr * block_align as u32;
    let data_len = (num_samples * ch as usize * bytes_per_sample as usize) as u32;
    let riff_chunk_size = 36 + data_len;

    let mut out = Vec::with_capacity((44 + data_len) as usize);

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    out.extend_from_slice(&1u16.to_le_bytes());  // PCM
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // 16 bits

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());

    // Interleave samples
    for s in 0..num_samples {
        for c in 0..audio.channels() {
            let val_f64 = audio.get_sample(c, s).unwrap_or(0.0).clamp(-1.0, 1.0);
            let val_i16 = (val_f64 * 32767.0).round() as i16;
            out.extend_from_slice(&val_i16.to_le_bytes());
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_roundtrip_stress_001() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((1 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 1) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_002() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((2 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 2) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_003() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((3 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 3) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_004() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((4 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 4) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_005() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((5 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 5) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_006() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((6 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 6) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_007() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((7 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 7) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_008() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((8 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 8) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_009() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((9 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 9) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_010() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((10 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 10) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_011() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((11 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 11) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_012() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((12 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 12) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_013() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((13 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 13) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_014() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((14 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 14) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_015() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((15 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 15) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_016() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((16 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 16) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_017() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((17 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 17) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_018() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((18 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 18) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_019() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((19 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 19) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_020() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((20 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 20) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_021() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((21 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 21) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_022() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((22 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 22) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_023() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((23 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 23) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_024() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((24 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 24) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_025() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((25 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 25) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_026() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((26 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 26) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_027() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((27 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 27) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_028() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((28 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 28) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_029() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((29 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 29) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_030() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((30 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 30) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_031() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((31 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 31) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_032() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((32 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 32) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_033() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((33 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 33) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_034() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((34 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 34) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_035() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((35 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 35) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_036() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((36 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 36) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_037() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((37 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 37) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_038() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((38 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 38) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_039() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((39 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 39) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_040() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((40 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 40) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_041() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((41 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 41) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_042() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((42 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 42) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_043() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((43 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 43) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_044() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((44 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 44) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_045() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((45 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 45) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_046() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((46 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 46) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_047() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((47 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 47) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_048() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((48 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 48) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_049() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((49 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 49) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_050() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((50 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 50) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_051() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((51 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 51) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_052() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((52 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 52) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_053() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((53 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 53) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_054() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((54 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 54) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_055() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((55 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 55) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_056() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((56 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 56) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_057() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((57 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 57) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_058() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((58 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 58) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_059() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((59 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 59) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_060() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((60 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 60) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_061() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((61 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 61) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_062() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((62 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 62) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_063() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((63 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 63) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_064() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((64 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 64) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_065() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((65 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 65) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_066() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((66 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 66) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_067() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((67 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 67) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_068() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((68 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 68) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_069() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((69 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 69) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_070() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((70 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 70) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_071() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((71 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 71) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_072() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((72 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 72) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_073() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((73 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 73) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_074() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((74 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 74) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_075() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((75 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 75) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_076() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((76 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 76) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_077() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((77 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 77) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_078() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((78 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 78) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_079() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((79 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 79) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_080() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((80 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 80) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_081() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((81 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 81) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_082() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((82 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 82) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_083() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((83 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 83) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_084() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((84 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 84) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_085() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((85 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 85) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_086() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((86 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 86) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_087() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((87 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 87) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_088() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((88 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 88) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_089() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((89 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 89) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_090() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((90 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 90) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_091() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((91 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 91) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_092() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((92 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 92) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_093() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((93 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 93) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_094() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((94 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 94) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_095() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((95 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 95) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_096() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((96 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 96) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_097() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((97 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 97) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_098() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((98 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 98) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_099() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((99 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 99) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_100() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((100 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 100) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_101() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((101 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 101) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_102() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((102 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 102) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_103() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((103 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 103) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_104() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((104 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 104) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_105() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((105 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 105) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_106() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((106 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 106) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_107() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((107 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 107) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_108() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((108 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 108) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_109() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((109 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 109) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_110() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((110 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 110) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_111() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((111 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 111) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_112() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((112 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 112) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_113() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((113 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 113) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_114() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((114 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 114) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_115() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((115 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 115) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_wav_roundtrip_stress_116() {
        let sr = SampleRate::new(16000).unwrap();
        let ch = ((116 % 2) + 1) as usize;
        let num_samples = 100;
        let mut buf = AudioBuffer::zeros(ch, num_samples, sr).unwrap();
        for c in 0..ch {
            for s in 0..num_samples {
                let val = ((c * 50 + s + 116) as f64 * 0.05).sin() * 0.8;
                buf.set_sample(c, s, val).unwrap();
            }
        }
        
        let wav_bytes = write_wav(&buf);
        let parsed = read_wav(&wav_bytes).unwrap();
        assert_eq!(parsed.channels(), buf.channels());
        assert_eq!(parsed.num_samples(), buf.num_samples());
        
        for c in 0..ch {
            for s in 0..num_samples {
                let orig = buf.get_sample(c, s).unwrap();
                let decoded = parsed.get_sample(c, s).unwrap();
                assert!((orig - decoded).abs() < 1e-3);
            }
        }
    }
}
