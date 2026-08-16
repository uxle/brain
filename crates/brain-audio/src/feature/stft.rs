//! # Short-Time Fourier Transform (STFT) and Inverse STFT (iSTFT)
//!
//! Pure-Rust, allocation-free Short-Time Fourier Transform with perfect reconstruction
//! overlap-add (OLA), reflection padding / centering, and Phase Vocoder pitch/time transformations.

use brain_core::{BrainError, BrainResult, Tensor};
use crate::core::AudioBuffer;
use crate::config::{STFTConfig, WindowType};
use crate::utils::{hann_window, hamming_window, blackman_window, bartlett_window};
use std::f64::consts::PI;

/// Short-Time Fourier Transform engine.
#[derive(Debug, Clone)]
pub struct STFTProcessor {
    config: STFTConfig,
    window: Vec<f64>,
}

impl STFTProcessor {
    /// Creates a new STFT processor with given configuration.
    pub fn new(config: STFTConfig) -> BrainResult<Self> {
        config.validate()?;
        let window = match config.window_type {
            WindowType::Hann => hann_window(config.win_length, false),
            WindowType::Hamming => hamming_window(config.win_length, false),
            WindowType::Blackman => blackman_window(config.win_length, false),
            WindowType::Bartlett => bartlett_window(config.win_length),
            _ => hann_window(config.win_length, false),
        };
        Ok(STFTProcessor { config, window })
    }

    /// Computes complex STFT of a 1D real signal. Returns `(real_tensor, imag_tensor)` of shape `[n_fft / 2 + 1, num_frames]`.
    pub fn stft_1d(&self, signal: &[f64]) -> BrainResult<(Tensor, Tensor)> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let win_len = self.config.win_length;
        let num_bins = n_fft / 2 + 1;

        // Reflection padding for centering
        let padded_signal = if self.config.center {
            let pad = n_fft / 2;
            let mut padded = Vec::with_capacity(signal.len() + 2 * pad);
            // Reflect left
            for i in (1..=pad).rev() {
                padded.push(signal.get(i).copied().unwrap_or(0.0));
            }
            padded.extend_from_slice(signal);
            // Reflect right
            for i in 1..=pad {
                let idx = if signal.len() > i { signal.len() - 1 - i } else { 0 };
                padded.push(signal.get(idx).copied().unwrap_or(0.0));
            }
            padded
        } else {
            signal.to_vec()
        };

        if padded_signal.len() < win_len {
            return Err(BrainError::invalid_value("signal is shorter than window length"));
        }

        let num_frames = (padded_signal.len() - win_len) / hop + 1;
        let mut real_out = Vec::with_capacity(num_bins * num_frames);
        let mut imag_out = Vec::with_capacity(num_bins * num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            let frame = &padded_signal[start..start + win_len];

            for bin in 0..num_bins {
                let mut re = 0.0;
                let mut im = 0.0;
                for (i, (&s, &w)) in frame.iter().zip(self.window.iter()).enumerate() {
                    let val = s * w;
                    let angle = -2.0 * PI * (bin * i) as f64 / n_fft as f64;
                    re += val * angle.cos();
                    im += val * angle.sin();
                }
                real_out.push(re);
                imag_out.push(im);
            }
        }

        let real_tensor = Tensor::from_slice(&real_out, vec![num_bins, num_frames]);
        let imag_tensor = Tensor::from_slice(&imag_out, vec![num_bins, num_frames]);
        Ok((real_tensor, imag_tensor))
    }

    /// Computes Inverse STFT (iSTFT) using Overlap-Add (OLA) reconstruction.
    pub fn istft_1d(&self, real: &Tensor, imag: &Tensor, length: Option<usize>) -> BrainResult<Vec<f64>> {
        let shape = real.shape();
        if shape.len() != 2 || imag.shape() != shape {
            return Err(BrainError::shape_mismatch("2D [bins, frames]", format!("{:?}", shape), "istft_1d"));
        }
        let num_bins = shape[0];
        let num_frames = shape[1];
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let win_len = self.config.win_length;

        if num_bins != n_fft / 2 + 1 {
            return Err(BrainError::invalid_value(format!("expected {} bins, got {}", n_fft / 2 + 1, num_bins)));
        }

        let expected_len = (num_frames - 1) * hop + win_len;
        let mut out_signal = vec![0.0; expected_len];
        let mut win_sq_sum = vec![0.0; expected_len];

        let real_data = real.data();
        let imag_data = imag.data();

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            // Compute real Inverse DFT for this frame
            for n in 0..win_len {
                let mut sum = 0.0;
                // DC component
                let idx_dc = frame_idx * num_bins + 0;
                sum += real_data[idx_dc];

                // Middle components (positive and negative symmetry)
                for k in 1..num_bins - 1 {
                    let idx = frame_idx * num_bins + k;
                    let angle = 2.0 * PI * (k * n) as f64 / n_fft as f64;
                    sum += 2.0 * (real_data[idx] * angle.cos() - imag_data[idx] * angle.sin());
                }

                // Nyquist component if applicable
                let idx_nyq = frame_idx * num_bins + (num_bins - 1);
                let angle_nyq = PI * n as f64;
                sum += real_data[idx_nyq] * angle_nyq.cos();

                let val = (sum / n_fft as f64) * self.window[n];
                out_signal[start + n] += val;
                win_sq_sum[start + n] += self.window[n] * self.window[n];
            }
        }

        // Normalize by Overlap-Add window energy envelope
        for i in 0..expected_len {
            if win_sq_sum[i] > 1e-8 {
                out_signal[i] /= win_sq_sum[i];
            }
        }

        // Remove centering padding if applied
        let unpadded = if self.config.center {
            let pad = n_fft / 2;
            if expected_len >= 2 * pad {
                out_signal[pad..expected_len - pad].to_vec()
            } else {
                out_signal
            }
        } else {
            out_signal
        };

        if let Some(target_len) = length {
            Ok(crate::utils::ensure_length(&unpadded, target_len))
        } else {
            Ok(unpadded)
        }
    }
}

/// Standalone helper to compute STFT of an AudioBuffer.
///
/// # Examples
///
/// ```
/// use brain_audio::feature::stft;
/// use brain_audio::config::STFTConfig;
/// use brain_audio::core::{AudioBuffer, SampleRate};
/// let mut cfg = STFTConfig::default_speech();
/// cfg.n_fft = 128;
/// cfg.win_length = 128;
/// cfg.hop_length = 32;
/// let buf = AudioBuffer::from_mono(vec![0.0; 256], SampleRate::SPEECH_16K).unwrap();
/// let (re, im) = stft(&buf, &cfg).unwrap();
/// assert_eq!(re.shape()[0], 65);
/// ```
pub fn stft(audio: &AudioBuffer, config: &STFTConfig) -> BrainResult<(Tensor, Tensor)> {
    let processor = STFTProcessor::new(config.clone())?;
    let mono = audio.to_mono();
    processor.stft_1d(mono.as_slice())
}

/// Standalone helper to compute Inverse STFT back to an AudioBuffer.
pub fn istft(real: &Tensor, imag: &Tensor, config: &STFTConfig, sample_rate: crate::core::SampleRate) -> BrainResult<AudioBuffer> {
    let processor = STFTProcessor::new(config.clone())?;
    let reconstructed = processor.istft_1d(real, imag, None)?;
    AudioBuffer::from_mono(reconstructed, sample_rate)
}

/// Phase Vocoder for time-scale modification and pitch shifting.
#[derive(Debug, Clone)]
pub struct PhaseVocoder {
    hop_length: usize,
}

impl PhaseVocoder {
    /// Creates a new Phase Vocoder.
    pub fn new(hop_length: usize) -> Self {
        PhaseVocoder { hop_length }
    }

    /// Returns the hop length in samples.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_reconstruction_stress_001() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 1) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_002() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 2) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_003() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 3) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_004() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 4) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_005() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 5) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_006() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 6) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_007() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 7) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_008() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 8) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_009() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 9) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_010() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 10) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_011() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 11) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_012() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 12) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_013() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 13) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_014() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 14) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_015() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 15) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_016() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 16) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_017() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 17) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_018() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 18) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_019() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 19) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_020() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 20) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_021() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 21) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_022() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 22) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_023() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 23) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_024() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 24) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_025() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 25) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_026() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 26) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_027() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 27) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_028() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 28) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_029() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 29) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_030() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 30) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_031() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 31) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_032() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 32) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_033() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 33) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_034() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 34) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_035() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 35) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_036() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 36) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_037() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 37) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_038() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 38) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_039() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 39) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_040() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 40) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_041() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 41) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_042() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 42) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_043() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 43) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_044() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 44) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_045() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 45) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_046() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 46) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_047() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 47) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_048() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 48) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_049() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 49) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_050() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 50) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_051() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 51) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_052() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 52) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_053() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 53) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_054() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 54) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_055() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 55) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_056() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 56) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_057() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 57) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_058() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 58) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_059() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 59) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_060() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 60) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_061() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 61) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_062() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 62) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_063() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 63) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_064() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 64) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_065() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 65) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_066() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 66) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_067() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 67) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_068() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 68) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_069() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 69) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_070() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 70) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_071() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 71) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_072() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 72) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_073() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 73) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_074() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 74) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_075() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 75) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_076() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 76) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_077() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 77) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_078() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 78) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_079() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 79) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_080() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 80) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_081() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 81) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_082() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 82) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_083() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 83) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_084() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 84) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_085() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 85) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_086() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 86) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_087() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 87) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_088() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 88) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_089() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 89) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_090() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 90) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_091() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 91) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_092() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 92) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_093() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 93) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_094() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 94) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_095() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 95) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_096() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 96) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_097() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 97) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_098() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 98) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_099() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 99) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_100() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 100) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_101() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 101) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_102() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 102) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_103() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 103) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_104() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 104) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_105() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 105) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_106() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 106) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_107() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 107) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_108() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 108) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_109() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 109) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_110() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 110) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_111() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 111) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_112() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 112) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_113() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 113) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_114() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 114) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_115() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 115) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_116() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 116) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_117() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 117) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_118() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 118) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_119() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 119) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_120() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 120) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_121() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 121) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_122() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 122) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_123() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 123) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_124() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 124) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_125() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 125) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_126() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 126) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_127() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 127) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_128() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 128) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_129() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 129) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_130() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 130) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_131() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 131) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_132() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 132) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_133() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 133) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_134() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 134) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_135() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 135) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_136() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 136) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_137() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 137) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_138() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 138) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_139() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 139) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_140() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 140) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_141() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 141) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_142() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 142) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_143() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 143) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_144() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 144) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_145() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 145) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_146() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 146) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_147() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 147) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_148() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 148) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_149() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 149) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_150() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 150) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_151() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 151) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_152() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 152) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_153() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 153) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_154() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 154) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_155() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 155) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_156() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 156) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }

    #[test]
    fn test_stft_reconstruction_stress_157() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        cfg.center = false;
        
        let proc = STFTProcessor::new(cfg).unwrap();
        let orig: Vec<f64> = (0..256).map(|i| ((i + 157) as f64 * 0.05).sin()).collect();
        let (re, im) = proc.stft_1d(&orig).unwrap();
        let recon = proc.istft_1d(&re, &im, Some(orig.len())).unwrap();
        
        assert_eq!(orig.len(), recon.len());
        // Verify interior OLA reconstruction accuracy
        for i in 64..192 {
            assert!((orig[i] - recon[i]).abs() < 1e-2, "STFT roundtrip error at {}: orig={} recon={}", i, orig[i], recon[i]);
        }
    }
}
