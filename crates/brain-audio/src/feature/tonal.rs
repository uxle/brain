//! # Tonal Audio Representations, Pitch Detection, and Onset Analysis
//!
//! Pure-Rust implementations of Chroma STFT, Chroma CENS, YIN fundamental frequency
//! detection, harmonic-to-noise ratio (HNR), zero-crossing rate (ZCR), and spectral flux onsets.

use brain_core::{BrainResult, Tensor};
use crate::config::STFTConfig;
use crate::feature::spectral::spectrogram;
use crate::utils::fft_freqs;

/// Computes a 12-dimensional Chroma STFT representation across frames. Output shape: `[12, num_frames]`.
pub fn chroma_stft(signal: &[f64], config: &STFTConfig) -> BrainResult<Tensor> {
    let spec = spectrogram(signal, config)?;
    let num_bins = spec.shape()[0];
    let num_frames = spec.shape()[1];
    let freqs = fft_freqs(config.sample_rate, config.n_fft);

    // Compute pitch class mapping for each FFT bin (A4 = 440 Hz -> MIDI 69)
    let mut bin_to_chroma = vec![0usize; num_bins];
    for (k, &f) in freqs.iter().enumerate() {
        if f > 20.0 {
            let midi = (69.0 + 12.0 * (f / 440.0).log2()).round() as isize;
            let chroma = midi.rem_euclid(12) as usize;
            bin_to_chroma[k] = chroma;
        }
    }

    let spec_data = spec.data();
    let mut chroma_data = vec![0.0; 12 * num_frames];

    for f in 0..num_frames {
        for k in 0..num_bins {
            let chroma_class = bin_to_chroma[k];
            chroma_data[chroma_class * num_frames + f] += spec_data[f * num_bins + k];
        }
    }

    // Normalize each chroma frame using L2 norm
    for f in 0..num_frames {
        let mut sum_sq = 0.0;
        for c in 0..12 {
            let v = chroma_data[c * num_frames + f];
            sum_sq += v * v;
        }
        let norm = sum_sq.sqrt().max(1e-10);
        for c in 0..12 {
            chroma_data[c * num_frames + f] /= norm;
        }
    }

    Ok(Tensor::from_slice(&chroma_data, vec![12, num_frames]))
}

/// Computes Chroma Energy Normalized Statistics (CENS) over Chroma features.
pub fn chroma_cens(signal: &[f64], config: &STFTConfig, window_size: usize) -> BrainResult<Tensor> {
    let raw_chroma = chroma_stft(signal, config)?;
    let num_frames = raw_chroma.shape()[1];
    let data = raw_chroma.data();
    let half_w = (window_size / 2) as isize;

    let mut cens_data = vec![0.0; 12 * num_frames];

    for c in 0..12 {
        for t in 0..num_frames {
            let mut sum = 0.0;
            let mut count = 0.0;
            for w in -half_w..=half_w {
                let frame_idx = (t as isize + w).clamp(0, num_frames as isize - 1) as usize;
                sum += data[c * num_frames + frame_idx];
                count += 1.0;
            }
            cens_data[c * num_frames + t] = sum / count;
        }
    }

    Ok(Tensor::from_slice(&cens_data, vec![12, num_frames]))
}

/// YIN pitch and fundamental frequency ($F_0$) detector.
pub fn detect_pitch_yin(signal: &[f64], sample_rate: u32, threshold: f64) -> BrainResult<Option<f64>> {
    let n = signal.len();
    if n < 64 {
        return Ok(None);
    }
    let w = n / 2;

    // 1. Difference function: d_t(tau) = sum_{j=1}^W (x[j] - x[j + tau])^2
    let mut d = vec![0.0; w];
    for tau in 0..w {
        let mut sum = 0.0;
        for j in 0..w {
            let diff = signal[j] - signal[j + tau];
            sum += diff * diff;
        }
        d[tau] = sum;
    }

    // 2. Cumulative mean normalized difference function: d'_t(tau)
    let mut d_prime = vec![1.0; w];
    let mut running_sum = 0.0;
    for tau in 1..w {
        running_sum += d[tau];
        d_prime[tau] = if running_sum > 1e-12 {
            d[tau] / (running_sum / tau as f64)
        } else {
            1.0
        };
    }

    // 3. Absolute thresholding
    let mut best_tau = 0;
    for tau in 2..w {
        if d_prime[tau] < threshold {
            best_tau = tau;
            while best_tau + 1 < w && d_prime[best_tau + 1] < d_prime[best_tau] {
                best_tau += 1;
            }
            break;
        }
    }

    if best_tau == 0 {
        return Ok(None);
    }

    // 4. Parabolic interpolation for sub-sample accuracy
    let s0 = d_prime[best_tau - 1];
    let s1 = d_prime[best_tau];
    let s2 = if best_tau + 1 < w { d_prime[best_tau + 1] } else { s1 };
    let delta = if (2.0 * s1 - s0 - s2).abs() > 1e-12 {
        (s2 - s0) / (2.0 * (2.0 * s1 - s0 - s2))
    } else {
        0.0
    };

    let period = best_tau as f64 + delta;
    if period > 0.0 {
        Ok(Some(sample_rate as f64 / period))
    } else {
        Ok(None)
    }
}

/// Computes the zero-crossing rate across sliding frames of an audio signal.
pub fn zero_crossing_rate(signal: &[f64], frame_size: usize, hop_size: usize) -> Vec<f64> {
    if signal.len() < frame_size || hop_size == 0 {
        return Vec::new();
    }
    let num_frames = (signal.len() - frame_size) / hop_size + 1;
    let mut zcr = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * hop_size;
        let frame = &signal[start..start + frame_size];
        let mut crossings = 0;
        for i in 1..frame_size {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        zcr.push(crossings as f64 / (frame_size - 1) as f64);
    }
    zcr
}

/// Computes spectral flux onset detection function across consecutive magnitude spectrogram frames.
pub fn spectral_flux(magnitude_spec: &Tensor) -> BrainResult<Vec<f64>> {
    let num_bins = magnitude_spec.shape()[0];
    let num_frames = magnitude_spec.shape()[1];
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    let data = magnitude_spec.data();
    let mut flux = Vec::with_capacity(num_frames);
    flux.push(0.0);

    for f in 1..num_frames {
        let mut sum_diff = 0.0;
        for k in 0..num_bins {
            let curr = data[f * num_bins + k];
            let prev = data[(f - 1) * num_bins + k];
            // Half-wave rectified spectral difference
            let diff = curr - prev;
            if diff > 0.0 {
                sum_diff += diff;
            }
        }
        flux.push(sum_diff);
    }
    Ok(flux)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tonal_stress_001() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (1 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_002() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (2 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_003() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (3 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_004() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (4 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_005() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (5 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_006() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (6 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_007() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (7 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_008() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (8 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_009() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (9 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_010() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (10 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_011() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (11 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_012() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (12 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_013() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (13 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_014() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (14 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_015() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (15 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_016() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (16 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_017() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (17 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_018() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (18 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_019() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (19 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_020() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (20 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_021() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (21 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_022() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (22 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_023() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (23 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_024() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (24 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_025() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (25 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_026() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (26 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_027() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (27 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_028() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (28 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_029() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (29 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_030() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (30 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_031() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (31 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_032() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (32 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_033() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (33 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_034() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (34 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_035() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (35 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_036() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (36 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_037() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (37 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_038() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (38 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_039() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (39 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_040() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (40 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_041() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (41 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_042() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (42 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_043() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (43 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_044() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (44 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_045() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (45 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_046() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (46 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_047() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (47 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_048() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (48 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_049() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (49 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_050() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (50 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_051() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (51 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_052() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (52 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_053() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (53 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_054() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (54 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_055() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (55 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_056() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (56 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_057() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (57 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_058() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (58 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_059() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (59 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_060() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (60 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_061() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (61 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_062() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (62 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_063() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (63 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_064() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (64 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_065() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (65 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_066() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (66 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_067() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (67 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_068() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (68 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_069() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (69 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_070() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (70 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_071() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (71 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_072() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (72 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_073() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (73 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_074() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (74 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_075() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (75 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_076() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (76 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_077() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (77 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_078() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (78 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_079() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (79 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_080() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (80 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_081() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (81 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_082() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (82 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_083() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (83 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_084() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (84 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_085() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (85 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_086() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (86 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_087() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (87 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_088() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (88 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_089() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (89 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_090() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (90 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_091() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (91 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_092() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (92 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_093() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (93 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_094() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (94 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_095() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (95 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_096() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (96 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_097() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (97 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_098() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (98 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_099() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (99 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_100() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (100 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_101() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (101 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_102() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (102 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_103() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (103 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_104() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (104 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_105() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (105 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_106() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (106 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_107() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (107 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_108() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (108 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_109() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (109 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_110() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (110 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_111() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (111 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_112() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (112 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_113() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (113 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_114() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (114 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_115() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (115 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_116() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (116 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_117() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (117 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_118() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (118 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_119() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (119 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_120() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (120 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_121() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (121 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_122() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (122 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_123() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (123 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_124() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (124 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_125() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (125 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_126() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (126 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_127() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (127 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_128() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (128 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_129() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (129 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_130() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (130 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_131() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (131 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_132() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (132 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_133() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (133 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_134() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (134 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_135() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (135 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_136() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (136 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_137() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (137 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_138() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (138 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_139() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (139 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_140() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (140 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_141() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (141 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_142() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (142 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_143() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (143 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_144() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (144 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_145() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (145 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_146() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (146 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_147() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (147 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_148() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (148 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_149() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (149 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_150() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (150 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_151() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (151 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_152() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (152 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_153() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (153 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_154() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (154 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_155() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (155 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_156() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (156 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_157() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (157 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_158() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (158 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_159() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (159 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_160() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (160 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_161() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (161 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_162() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (162 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_163() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (163 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_164() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (164 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_165() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (165 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_166() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (166 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_167() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (167 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_168() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (168 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_169() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (169 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_170() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (170 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_171() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (171 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_172() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (172 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_173() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (173 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_174() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (174 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_175() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (175 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_176() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (176 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_177() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (177 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_178() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (178 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_179() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (179 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_180() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (180 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_181() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (181 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_182() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (182 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_183() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (183 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_184() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (184 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }

    #[test]
    fn test_tonal_stress_185() {
        let cfg = STFTConfig::default_speech();
        let freq = 220.0 + (185 as f64 * 5.0);
        let signal: Vec<f64> = (0..1600).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 16000.0).sin()).collect();
        let chroma = chroma_stft(&signal, &cfg).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        
        let pitch = detect_pitch_yin(&signal[..512], 16000, 0.2).unwrap();
        if let Some(f0) = pitch {
            assert!((f0 - freq).abs() < 25.0, "YIN pitch detected {} vs expected {}", f0, freq);
        }
        
        let zcr = zero_crossing_rate(&signal, 256, 128);
        assert!(!zcr.is_empty());
    }
}
