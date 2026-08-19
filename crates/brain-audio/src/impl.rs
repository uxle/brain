//! # Composite Audio Pipelines and Transform Implementations
//!
//! High-level feature extraction pipelines combining STFT, Mel filter banks,
//! normalization, and batch collation.

use crate::config::MelConfig;
use crate::core::AudioBuffer;
use crate::utils::hann_window;
use brain_core::{BrainError, BrainResult, Tensor};
use std::f64::consts::PI;

/// Composite Audio Feature Extractor for STFT, Mel, and MFCC representations.
#[derive(Debug, Clone)]
pub struct AudioFeatureExtractor {
    config: MelConfig,
    window: Vec<f64>,
}

impl AudioFeatureExtractor {
    /// Creates a new feature extractor with given Mel configuration.
    pub fn new(config: MelConfig) -> BrainResult<Self> {
        config.validate()?;
        let window = hann_window(config.stft.win_length, false);
        Ok(AudioFeatureExtractor { config, window })
    }

    /// Extracts power spectrogram from an AudioBuffer. Output shape: `[channels, n_fft / 2 + 1, num_frames]`.
    pub fn extract_spectrogram(&self, audio: &AudioBuffer) -> BrainResult<Tensor> {
        let channels = audio.channels();
        let num_samples = audio.num_samples();
        let n_fft = self.config.stft.n_fft;
        let hop = self.config.stft.hop_length;
        let win_len = self.config.stft.win_length;
        let num_bins = n_fft / 2 + 1;

        if num_samples < win_len {
            return Err(BrainError::invalid_value("audio length shorter than window size"));
        }
        let num_frames = (num_samples - win_len) / hop + 1;
        let mut out_data = Vec::with_capacity(channels * num_bins * num_frames);

        for ch in 0..channels {
            let ch_data = audio.channel(ch)?;
            for frame_idx in 0..num_frames {
                let start = frame_idx * hop;
                let frame = &ch_data[start..start + win_len];
                // Compute real DFT for this windowed frame
                for bin in 0..num_bins {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for (i, (&s, &w)) in frame.iter().zip(self.window.iter()).enumerate() {
                        let sample = s * w;
                        let angle = -2.0 * PI * (bin * i) as f64 / n_fft as f64;
                        re += sample * angle.cos();
                        im += sample * angle.sin();
                    }
                    let power = re * re + im * im;
                    out_data.push(power);
                }
            }
        }

        Ok(Tensor::from_slice(&out_data, vec![channels, num_bins, num_frames]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
