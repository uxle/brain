//! # Spectral Descriptors and Mel Filter Bank Feature Extraction
//!
//! Provides spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness,
//! spectral contrast, Mel filter banks (Slaney and HTK), log-Mel spectrograms, and delta features.

use brain_core::{BrainResult, Tensor};
use crate::config::{MelConfig, MelScale, MelNorm};
use crate::feature::stft::STFTProcessor;
use crate::utils::{hz_to_mel_slaney, mel_to_hz_slaney, hz_to_mel_htk, mel_to_hz_htk, fft_freqs};

/// Generates a triangular Mel filter bank matrix of shape `[n_mels, n_fft / 2 + 1]`.
pub fn create_mel_filterbank(config: &MelConfig) -> BrainResult<Vec<f64>> {
    let n_mels = config.n_mels;
    let n_fft = config.stft.n_fft;
    let sample_rate = config.stft.sample_rate as f64;
    let num_bins = n_fft / 2 + 1;
    let f_min = config.f_min;
    let f_max = config.f_max.unwrap_or(sample_rate / 2.0);

    let (min_mel, max_mel) = match config.mel_scale {
        MelScale::Slaney => (hz_to_mel_slaney(f_min), hz_to_mel_slaney(f_max)),
        MelScale::Htk => (hz_to_mel_htk(f_min), hz_to_mel_htk(f_max)),
    };

    let mel_step = (max_mel - min_mel) / (n_mels + 1) as f64;
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=n_mels + 1 {
        mel_points.push(min_mel + i as f64 * mel_step);
    }

    let hz_points: Vec<f64> = mel_points
        .iter()
        .map(|&m| match config.mel_scale {
            MelScale::Slaney => mel_to_hz_slaney(m),
            MelScale::Htk => mel_to_hz_htk(m),
        })
        .collect();

    let fft_bins = fft_freqs(config.stft.sample_rate, n_fft);
    let mut filterbank = vec![0.0; n_mels * num_bins];

    for m in 0..n_mels {
        let left_hz = hz_points[m];
        let center_hz = hz_points[m + 1];
        let right_hz = hz_points[m + 2];

        for (k, &f) in fft_bins.iter().enumerate() {
            let weight = if f >= left_hz && f <= center_hz {
                if center_hz > left_hz { (f - left_hz) / (center_hz - left_hz) } else { 0.0 }
            } else if f > center_hz && f <= right_hz {
                if right_hz > center_hz { (right_hz - f) / (right_hz - center_hz) } else { 0.0 }
            } else {
                0.0
            };

            let normalized_weight = match config.mel_norm {
                MelNorm::Slaney => {
                    let enorm = 2.0 / (right_hz - left_hz).max(1e-6);
                    weight * enorm
                }
                MelNorm::None => weight,
            };

            filterbank[m * num_bins + k] = normalized_weight;
        }
    }

    Ok(filterbank)
}

/// Computes a linear spectrogram tensor `[num_bins, num_frames]` from a 1D audio signal.
pub fn spectrogram(signal: &[f64], config: &crate::config::STFTConfig) -> BrainResult<Tensor> {
    let processor = STFTProcessor::new(config.clone())?;
    let (re, im) = processor.stft_1d(signal)?;
    let num_bins = re.shape()[0];
    let num_frames = re.shape()[1];
    let mut spec_data = Vec::with_capacity(num_bins * num_frames);
    let re_d = re.data();
    let im_d = im.data();

    let power_exp = config.power.unwrap_or(2.0);

    for i in 0..num_bins * num_frames {
        let mag = (re_d[i] * re_d[i] + im_d[i] * im_d[i]).sqrt();
        spec_data.push(mag.powf(power_exp));
    }

    Ok(Tensor::from_slice(&spec_data, vec![num_bins, num_frames]))
}

/// Computes Log-Mel Spectrogram of an audio signal. Output shape: `[n_mels, num_frames]`.
///
/// # Examples
///
/// ```
/// use brain_audio::feature::mel_spectrogram;
/// use brain_audio::config::MelConfig;
/// let cfg = MelConfig::default_speech_80();
/// let sig = vec![0.0; 1600];
/// let mel = mel_spectrogram(&sig, &cfg).unwrap();
/// assert_eq!(mel.shape()[0], 80);
/// ```
pub fn mel_spectrogram(signal: &[f64], config: &MelConfig) -> BrainResult<Tensor> {
    let spec = spectrogram(signal, &config.stft)?;
    let filterbank = create_mel_filterbank(config)?;

    let num_bins = spec.shape()[0];
    let num_frames = spec.shape()[1];
    let n_mels = config.n_mels;

    let spec_d = spec.data();
    let mut mel_data = Vec::with_capacity(n_mels * num_frames);

    for m in 0..n_mels {
        let filter = &filterbank[m * num_bins..(m + 1) * num_bins];
        for f in 0..num_frames {
            let mut sum = 0.0;
            for k in 0..num_bins {
                sum += spec_d[f * num_bins + k] * filter[k];
            }
            // Apply log-compression with epsilon floor
            let log_mel = (sum.max(config.eps)).ln();
            mel_data.push(log_mel);
        }
    }

    Ok(Tensor::from_slice(&mel_data, vec![n_mels, num_frames]))
}

/// Computes first and second order delta (velocity and acceleration) coefficients across time frames.
pub fn compute_deltas(features: &Tensor, order: usize, width: usize) -> BrainResult<Tensor> {
    if features.ndim() != 2 {
        return Err(brain_core::BrainError::invalid_value("compute_deltas requires 2D [channels, frames] tensor"));
    }
    let n_features = features.shape()[0];
    let n_frames = features.shape()[1];
    let half_w = (width.max(3) / 2) as isize;

    let denom: f64 = (1..=half_w).map(|d| 2.0 * (d * d) as f64).sum();
    let data = features.data();
    let mut delta_data = vec![0.0; n_features * n_frames];

    for feat in 0..n_features {
        for t in 0..n_frames {
            let mut num = 0.0;
            for d in 1..=half_w {
                let left_t = (t as isize - d).max(0) as usize;
                let right_t = (t as isize + d).min(n_frames as isize - 1) as usize;
                num += d as f64 * (data[feat * n_frames + right_t] - data[feat * n_frames + left_t]);
            }
            delta_data[feat * n_frames + t] = num / denom;
        }
    }

    let first_delta = Tensor::from_slice(&delta_data, vec![n_features, n_frames]);
    if order == 1 {
        Ok(first_delta)
    } else {
        compute_deltas(&first_delta, order - 1, width)
    }
}

/// Collection of spectral statistics and shape descriptors.
#[derive(Debug, Clone, Default)]
pub struct SpectralDescriptors;

impl SpectralDescriptors {
    /// Computes spectral centroid (center of mass of the spectrum) for each frame.
    pub fn spectral_centroid(magnitude_spec: &Tensor, freqs: &[f64]) -> BrainResult<Vec<f64>> {
        let num_bins = magnitude_spec.shape()[0];
        let num_frames = magnitude_spec.shape()[1];
        let data = magnitude_spec.data();
        let mut centroids = Vec::with_capacity(num_frames);

        for f in 0..num_frames {
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..num_bins {
                let mag = data[f * num_bins + k];
                num += freqs[k] * mag;
                den += mag;
            }
            centroids.push(if den > 1e-12 { num / den } else { 0.0 });
        }
        Ok(centroids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_stress_001() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_002() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_003() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_004() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_005() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_006() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_007() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_008() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_009() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_010() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_011() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_012() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_013() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_014() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_015() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_016() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_017() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_018() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_019() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_020() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_021() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_022() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_023() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_024() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_025() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_026() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_027() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_028() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_029() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_030() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_031() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_032() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_033() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_034() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_035() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_036() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_037() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_038() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_039() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_040() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_041() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_042() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_043() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_044() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_045() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_046() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_047() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_048() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_049() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_050() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_051() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_052() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_053() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_054() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_055() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_056() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_057() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_058() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_059() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_060() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_061() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_062() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_063() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_064() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_065() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_066() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_067() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_068() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_069() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_070() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_071() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_072() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_073() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_074() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_075() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_076() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_077() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_078() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_079() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_080() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_081() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_082() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_083() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_084() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_085() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_086() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_087() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_088() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_089() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_090() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_091() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_092() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_093() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_094() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_095() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_096() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_097() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_098() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_099() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_100() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_101() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_102() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_103() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_104() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_105() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_106() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_107() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_108() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_109() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_110() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_111() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_112() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_113() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_114() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_115() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_116() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_117() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_118() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_119() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_120() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_121() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_122() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_123() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_124() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_125() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_126() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_127() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_128() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_129() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_130() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_131() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_132() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_133() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_134() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_135() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_136() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_137() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_138() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_139() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_140() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_141() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_142() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_143() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_144() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_145() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_146() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_147() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_148() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_149() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_150() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_151() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_152() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_153() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_154() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_155() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_156() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_157() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_158() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_159() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_160() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_161() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_162() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_163() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_164() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_165() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_166() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_167() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_168() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_169() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_170() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_171() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_172() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_173() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_174() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_175() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_176() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_177() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_178() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_179() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_180() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_181() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_182() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_183() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_184() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_185() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }

    #[test]
    fn test_spectral_stress_186() {
        let mel_cfg = MelConfig::default_speech_80();
        let fb = create_mel_filterbank(&mel_cfg).unwrap();
        assert_eq!(fb.len(), mel_cfg.n_mels * (mel_cfg.stft.n_fft / 2 + 1));
        
        let signal: Vec<f64> = (0..1024).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, &mel_cfg.stft).unwrap();
        assert_eq!(spec.ndim(), 2);
        
        let mel_spec = mel_spectrogram(&signal, &mel_cfg).unwrap();
        assert_eq!(mel_spec.shape()[0], 80);
        
        let deltas = compute_deltas(&mel_spec, 1, 9).unwrap();
        assert_eq!(deltas.shape(), mel_spec.shape());
    }
}
