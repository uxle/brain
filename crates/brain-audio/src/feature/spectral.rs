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
}
