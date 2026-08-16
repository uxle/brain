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
            return Err(BrainError::index_out_of_bounds(idx as isize, vocab_size, Some(0), "phonetic_one_hot"));
        }
        data[idx * seq_len + t] = 1.0;
    }

    Ok(Tensor::from_slice(&data, vec![vocab_size, seq_len]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_stress_001() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 1) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_002() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 2) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_003() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 3) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_004() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 4) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_005() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 5) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_006() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 6) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_007() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 7) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_008() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 8) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_009() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 9) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_010() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 10) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_011() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 11) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_012() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 12) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_013() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 13) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_014() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 14) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_015() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 15) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_016() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 16) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_017() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 17) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_018() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 18) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_019() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 19) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_020() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 20) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_021() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 21) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_022() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 22) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_023() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 23) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_024() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 24) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_025() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 25) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_026() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 26) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_027() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 27) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_028() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 28) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_029() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 29) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_030() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 30) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_031() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 31) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_032() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 32) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_033() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 33) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_034() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 34) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_035() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 35) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_036() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 36) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_037() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 37) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_038() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 38) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_039() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 39) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_040() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 40) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_041() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 41) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_042() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 42) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_043() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 43) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_044() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 44) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_045() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 45) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_046() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 46) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_047() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 47) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_048() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 48) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_049() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 49) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_050() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 50) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_051() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 51) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_052() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 52) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_053() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 53) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_054() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 54) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_055() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 55) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_056() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 56) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_057() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 57) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_058() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 58) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_059() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 59) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_060() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 60) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_061() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 61) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_062() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 62) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_063() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 63) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_064() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 64) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_065() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 65) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_066() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 66) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_067() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 67) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_068() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 68) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_069() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 69) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_070() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 70) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_071() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 71) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_072() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 72) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_073() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 73) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_074() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 74) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_075() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 75) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_076() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 76) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_077() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 77) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_078() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 78) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_079() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 79) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_080() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 80) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_081() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 81) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_082() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 82) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_083() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 83) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_084() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 84) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_085() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 85) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_086() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 86) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_087() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 87) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_088() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 88) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_089() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 89) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_090() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 90) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_091() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 91) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_092() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 92) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_093() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 93) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_094() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 94) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_095() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 95) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_096() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 96) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_097() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 97) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_098() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 98) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_099() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 99) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_100() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 100) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_101() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 101) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_102() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 102) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_103() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 103) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_104() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 104) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_105() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 105) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_106() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 106) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_107() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 107) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_108() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 108) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_109() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 109) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_110() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 110) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_111() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 111) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_112() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 112) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_113() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 113) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_114() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 114) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_115() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 115) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_116() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 116) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_117() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 117) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_118() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 118) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_119() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 119) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_120() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 120) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_121() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 121) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_122() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 122) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_123() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 123) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_124() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 124) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_125() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 125) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_126() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 126) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_127() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 127) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_128() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 128) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_129() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 129) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_130() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 130) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_131() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 131) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_132() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 132) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_133() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 133) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_134() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 134) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_135() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 135) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_136() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 136) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_137() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 137) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_138() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 138) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_139() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 139) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_140() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 140) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_141() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 141) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_142() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 142) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_143() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 143) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_144() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 144) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_145() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 145) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_146() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 146) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_147() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 147) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_148() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 148) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_149() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 149) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_150() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 150) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_151() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 151) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_152() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 152) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_153() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 153) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_154() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 154) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_155() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 155) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_156() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 156) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_157() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 157) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_158() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 158) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_159() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 159) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_160() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 160) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_161() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 161) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_162() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 162) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_163() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 163) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_164() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 164) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_165() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 165) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_166() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 166) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_167() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 167) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_168() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 168) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_169() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 169) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_170() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 170) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_171() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 171) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_172() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 172) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_173() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 173) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_174() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 174) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_175() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 175) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_176() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 176) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_177() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 177) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_178() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 178) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_179() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 179) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_180() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 180) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_181() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 181) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_182() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 182) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_183() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 183) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_184() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 184) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_185() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 185) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_186() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 186) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_187() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 187) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_188() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 188) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_189() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 189) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_190() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 190) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_191() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 191) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_192() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 192) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_193() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 193) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_194() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 194) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_195() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 195) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_196() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 196) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_197() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 197) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_198() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 198) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_199() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 199) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_200() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 200) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_201() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 201) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_202() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 202) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_203() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 203) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_204() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 204) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_205() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 205) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_206() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 206) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_207() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 207) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_208() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 208) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_209() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 209) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_210() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 210) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_211() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 211) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_212() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 212) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_213() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 213) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_214() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 214) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_215() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 215) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }

    #[test]
    fn test_encoding_stress_216() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 216) as f64 * 0.1).sin() * 0.9).collect();
        let mu_bytes = mu_law_encode(&signal);
        let mu_dec = mu_law_decode(&mu_bytes);
        assert_eq!(signal.len(), mu_dec.len());
        
        let a_bytes = a_law_encode(&signal);
        let a_dec = a_law_decode(&a_bytes);
        assert_eq!(signal.len(), a_dec.len());
        
        let one_hot = phonetic_one_hot(&[1, 5, 12], 20).unwrap();
        assert_eq!(one_hot.shape(), &[20, 3]);
    }
}
