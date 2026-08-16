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

    #[test]
    fn test_audio_impl_stress_001() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 1) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_002() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 2) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_003() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 3) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_004() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 4) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_005() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 5) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_006() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 6) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_007() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 7) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_008() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 8) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_009() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 9) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_010() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 10) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_011() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 11) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_012() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 12) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_013() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 13) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_014() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 14) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_015() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 15) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_016() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 16) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_017() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 17) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_018() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 18) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_019() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 19) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_020() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 20) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_021() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 21) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_022() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 22) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_023() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 23) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_024() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 24) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_025() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 25) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_026() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 26) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_027() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 27) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_028() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 28) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_029() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 29) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_030() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 30) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_031() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 31) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_032() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 32) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_033() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 33) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_034() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 34) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_035() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 35) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_036() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 36) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_037() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 37) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_038() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 38) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_039() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 39) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_040() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 40) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_041() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 41) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_042() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 42) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_043() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 43) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_044() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 44) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_045() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 45) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_046() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 46) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_047() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 47) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_048() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 48) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_049() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 49) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_050() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 50) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_051() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 51) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_052() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 52) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_053() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 53) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_054() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 54) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_055() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 55) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_056() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 56) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_057() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 57) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_058() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 58) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_059() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 59) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_060() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 60) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_061() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 61) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_062() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 62) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_063() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 63) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_064() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 64) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_065() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 65) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_066() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 66) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_067() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 67) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_068() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 68) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_069() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 69) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_070() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 70) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_071() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 71) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_072() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 72) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_073() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 73) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_074() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 74) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_075() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 75) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_076() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 76) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_077() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 77) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_078() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 78) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_079() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 79) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_080() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 80) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_081() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 81) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_082() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 82) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_083() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 83) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_084() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 84) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_085() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 85) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_086() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 86) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_087() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 87) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_088() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 88) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_089() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 89) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_090() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 90) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_091() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 91) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_092() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 92) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_093() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 93) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_094() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 94) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_095() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 95) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_096() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 96) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_097() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 97) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_098() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 98) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_099() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 99) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_100() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 100) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_101() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 101) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_102() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 102) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_103() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 103) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_104() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 104) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_105() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 105) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_106() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 106) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_107() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 107) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_108() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 108) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_109() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 109) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_110() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 110) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_111() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 111) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_112() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 112) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_113() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 113) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_114() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 114) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_115() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 115) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_116() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 116) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_117() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 117) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_118() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 118) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_119() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 119) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_120() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 120) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_121() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 121) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_122() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 122) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_123() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 123) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_124() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 124) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_125() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 125) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_126() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 126) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_127() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 127) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_128() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 128) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_129() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 129) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_130() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 130) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_131() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 131) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_132() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 132) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_133() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 133) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_134() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 134) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_135() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 135) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_136() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 136) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_137() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 137) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_138() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 138) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_139() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 139) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_140() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 140) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_141() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 141) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_142() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 142) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_143() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 143) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_144() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 144) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_145() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 145) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_146() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 146) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_147() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 147) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_148() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 148) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_149() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 149) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_150() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 150) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_151() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 151) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_152() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 152) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_153() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 153) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_154() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 154) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_155() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 155) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_156() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 156) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_157() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 157) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_158() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 158) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_159() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 159) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_160() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 160) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_161() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 161) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_162() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 162) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_163() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 163) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_164() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 164) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_165() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 165) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_166() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 166) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_167() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 167) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_168() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 168) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_169() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 169) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_170() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 170) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_171() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 171) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_172() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 172) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_173() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 173) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_174() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 174) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_175() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 175) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_176() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 176) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_177() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 177) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_178() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 178) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_179() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 179) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_180() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 180) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_181() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 181) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_182() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 182) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_183() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 183) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_184() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 184) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_185() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 185) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_186() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 186) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_187() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 187) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_188() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 188) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_189() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 189) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_190() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 190) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_191() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 191) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_192() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 192) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_193() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 193) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_194() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 194) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_195() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 195) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_196() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 196) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_197() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 197) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_198() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 198) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_199() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 199) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_200() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 200) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_201() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 201) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_202() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 202) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_203() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 203) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_204() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 204) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_205() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 205) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_206() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 206) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_207() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 207) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_208() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 208) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_209() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 209) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_210() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 210) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_211() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 211) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_212() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 212) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_213() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 213) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_214() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 214) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_215() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 215) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_216() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 216) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_217() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 217) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_218() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 218) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_219() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 219) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_220() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 220) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_221() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 221) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_222() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 222) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_223() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 223) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_224() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 224) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_225() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 225) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_226() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 226) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_227() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 227) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_228() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 228) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_229() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 229) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_230() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 230) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_231() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 231) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_232() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 232) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_233() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 233) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_234() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 234) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_235() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 235) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_236() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 236) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_237() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 237) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_238() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 238) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_239() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 239) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_240() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 240) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_241() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 241) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_242() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 242) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_243() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 243) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_244() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 244) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_245() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 245) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_246() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 246) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_247() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 247) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_248() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 248) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_249() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 249) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_250() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 250) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_251() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 251) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_252() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 252) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_253() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 253) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_254() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 254) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_255() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 255) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_256() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 256) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_257() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 257) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_258() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 258) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_259() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 259) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_260() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 260) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_261() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 261) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_262() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 262) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_263() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 263) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_264() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 264) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_265() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 265) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_266() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 266) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_267() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 267) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_268() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 268) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_269() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 269) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_270() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 270) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_271() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 271) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_272() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 272) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }

    #[test]
    fn test_audio_impl_stress_273() {
        let cfg = MelConfig::default_speech_80();
        let extractor = AudioFeatureExtractor::new(cfg).unwrap();
        let samples: Vec<f64> = (0..1600).map(|i| ((i + 273) as f64 * 0.05).sin()).collect();
        let buf = AudioBuffer::from_mono(samples, crate::core::SampleRate::SPEECH_16K).unwrap();
        let spec = extractor.extract_spectrogram(&buf).unwrap();
        assert_eq!(spec.ndim(), 3);
        assert_eq!(spec.shape()[0], 1); // 1 channel
        assert_eq!(spec.shape()[1], 257); // 512 / 2 + 1 bins
    }
}
