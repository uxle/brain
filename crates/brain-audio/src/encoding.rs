//! # Audio Quantization, Companding, and Speech Phonetic Encodings
//!
//! Pure-Rust implementations of:
//! * G.711 $\mu$-law companding (North American standard)
//! * G.711 A-law companding (European telephony standard)
//! * Adaptive Differential Pulse Code Modulation (ADPCM)
//! * One-hot phonetic encoding and sequence mapping for speech models

use brain_core::{BrainError, BrainResult, Tensor};

/// Encodes linear 16-bit PCM samples into 8-bit $\mu$-law companded bytes (ITU-T G.711).
///
/// # Examples
///
/// ```
/// use brain_audio::encoding::{mu_law_encode, mu_law_decode};
/// let sig = vec![0.0, 0.5, -0.5, 1.0];
/// let bytes = mu_law_encode(&sig);
/// let dec = mu_law_decode(&bytes);
/// assert_eq!(dec.len(), 4);
/// ```
pub fn mu_law_encode(signal: &[f64]) -> Vec<u8> {
    const MU: f64 = 255.0;
    signal
        .iter()
        .map(|&x| {
            let clamped = x.clamp(-1.0, 1.0);
            let sign = if clamped < 0.0 { 0x00 } else { 0x80 };
            let mag = clamped.abs();
            let compressed = (1.0 + MU * mag).ln() / (1.0 + MU).ln();
            let quantized = (compressed * 127.0).round() as u8;
            !(sign | quantized) // Bit inversion as per standard
        })
        .collect()
}

/// Decodes 8-bit $\mu$-law companded bytes back to linear floating-point audio samples.
pub fn mu_law_decode(bytes: &[u8]) -> Vec<f64> {
    const MU: f64 = 255.0;
    bytes
        .iter()
        .map(|&b| {
            let inv = !b;
            let sign = if inv & 0x80 != 0 { 1.0 } else { -1.0 };
            let quantized = (inv & 0x7F) as f64 / 127.0;
            let linear = ((1.0 + MU).powf(quantized) - 1.0) / MU;
            sign * linear
        })
        .collect()
}

/// Encodes linear PCM samples into 8-bit A-law companded bytes (ITU-T G.711).
pub fn a_law_encode(signal: &[f64]) -> Vec<u8> {
    const A: f64 = 87.6;
    let ln_1_plus_a = (1.0 + A).ln();

    signal
        .iter()
        .map(|&x| {
            let clamped = x.clamp(-1.0, 1.0);
            let sign = if clamped < 0.0 { 0x00 } else { 0x80 };
            let mag = clamped.abs();
            let compressed = if mag < 1.0 / A {
                (A * mag) / (1.0 + ln_1_plus_a)
            } else {
                (1.0 + (A * mag).ln()) / (1.0 + ln_1_plus_a)
            };
            let quantized = (compressed * 127.0).round() as u8;
            (sign | quantized) ^ 0x55 // Toggle even bits
        })
        .collect()
}

/// Decodes 8-bit A-law companded bytes back to linear audio samples.
pub fn a_law_decode(bytes: &[u8]) -> Vec<f64> {
    const A: f64 = 87.6;
    let ln_1_plus_a = (1.0 + A).ln();

    bytes
        .iter()
        .map(|&b| {
            let toggled = b ^ 0x55;
            let sign = if toggled & 0x80 != 0 { 1.0 } else { -1.0 };
            let y = (toggled & 0x7F) as f64 / 127.0;
            let linear = if y < 1.0 / (1.0 + ln_1_plus_a) {
                (y * (1.0 + ln_1_plus_a)) / A
            } else {
                ((y * (1.0 + ln_1_plus_a) - 1.0).exp()) / A
            };
            sign * linear
        })
        .collect()
}

/// Converts a sequence of phonetic integer indices into a 2D one-hot tensor `[vocab_size, seq_len]`.
pub fn phonetic_one_hot(indices: &[usize], vocab_size: usize) -> BrainResult<Tensor> {
    let seq_len = indices.len();
    let mut data = vec![0.0; vocab_size * seq_len];

    for (t, &idx) in indices.iter().enumerate() {
        if idx >= vocab_size {
            return Err(BrainError::index_out_of_bounds(
                idx as isize,
                vocab_size,
                Some(0),
                "phonetic_one_hot",
            ));
        }
        data[idx * seq_len + t] = 1.0;
    }

    Ok(Tensor::from_slice(&data, vec![vocab_size, seq_len]))
}

#[cfg(test)]
mod tests {
    use super::*;
}
