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

use crate::core::{AudioBuffer, SampleRate};
use brain_core::{BrainError, BrainResult};

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
        return Err(BrainError::invalid_value(
            "WAV file smaller than minimum 44-byte header",
        ));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(BrainError::invalid_value("invalid RIFF/WAVE header"));
    }

    let mut cursor = 12;
    let mut header: Option<WavHeader> = None;
    let mut data_slice: Option<&[u8]> = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
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
            let block_align =
                u16::from_le_bytes(bytes[cursor + 12..cursor + 14].try_into().unwrap());
            let bits_per_sample =
                u16::from_le_bytes(bytes[cursor + 14..cursor + 16].try_into().unwrap());

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
    let data =
        data_slice.ok_or_else(|| BrainError::invalid_value("missing data chunk in WAV file"))?;
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
                    let val = if raw & 0x800000 != 0 {
                        raw | !0xFFFFFF
                    } else {
                        raw
                    };
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
                _ => {
                    return Err(BrainError::invalid_value(format!(
                        "unsupported WAV format: tag={}, bits={}",
                        hdr.format_tag, hdr.bits_per_sample
                    )))
                }
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
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
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
}
