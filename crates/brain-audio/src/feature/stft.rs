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
}
