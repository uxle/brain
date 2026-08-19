//! # Mel-Frequency Cepstral Coefficients (MFCC) Extraction
//!
//! Production-grade MFCC extraction via Discrete Cosine Transform Type-II (DCT-II),
//! sine liftering, and cepstral mean and variance normalization (CMVN).

use brain_core::{BrainError, BrainResult, Tensor};
use crate::config::MFCCConfig;
use crate::feature::spectral::mel_spectrogram;
use std::f64::consts::PI;

/// MFCC computation engine.
#[derive(Debug, Clone)]
pub struct MFCCProcessor {
    config: MFCCConfig,
    dct_matrix: Vec<f64>,
    lifter_weights: Vec<f64>,
}

impl MFCCProcessor {
    /// Creates a new MFCC processor.
    pub fn new(config: MFCCConfig) -> BrainResult<Self> {
        config.validate()?;
        let n_mfcc = config.n_mfcc;
        let n_mels = config.mel.n_mels;

        // Ortho-normalized DCT-II matrix [n_mfcc, n_mels]
        let mut dct_matrix = vec![0.0; n_mfcc * n_mels];
        for i in 0..n_mfcc {
            let factor = if i == 0 { (1.0 / n_mels as f64).sqrt() } else { (2.0 / n_mels as f64).sqrt() };
            for j in 0..n_mels {
                let angle = PI * i as f64 * (j as f64 + 0.5) / n_mels as f64;
                dct_matrix[i * n_mels + j] = factor * angle.cos();
            }
        }

        // Sine lifter weights: 1 + (L / 2) * sin(pi * n / L)
        let mut lifter_weights = Vec::with_capacity(n_mfcc);
        let l = config.lifter;
        for n in 0..n_mfcc {
            if l > 0.0 {
                lifter_weights.push(1.0 + (l / 2.0) * (PI * n as f64 / l).sin());
            } else {
                lifter_weights.push(1.0);
            }
        }

        Ok(MFCCProcessor {
            config,
            dct_matrix,
            lifter_weights,
        })
    }

    /// Computes MFCC features for a 1D audio signal. Output shape: `[n_mfcc, num_frames]`.
    pub fn compute_mfcc(&self, signal: &[f64]) -> BrainResult<Tensor> {
        let log_mel = mel_spectrogram(signal, &self.config.mel)?;
        let n_mels = self.config.mel.n_mels;
        let n_mfcc = self.config.n_mfcc;
        let num_frames = log_mel.shape()[1];

        let mel_data = log_mel.data();
        let mut mfcc_data = Vec::with_capacity(n_mfcc * num_frames);

        for i in 0..n_mfcc {
            let dct_row = &self.dct_matrix[i * n_mels..(i + 1) * n_mels];
            let lifter = self.lifter_weights[i];
            for f in 0..num_frames {
                let mut sum = 0.0;
                for j in 0..n_mels {
                    sum += mel_data[j * num_frames + f] * dct_row[j];
                }
                mfcc_data.push(sum * lifter);
            }
        }

        let mut out_tensor = Tensor::from_slice(&mfcc_data, vec![n_mfcc, num_frames]);

        // Apply Cepstral Mean & Variance Normalization if requested
        if self.config.cepstral_mean_norm || self.config.cepstral_var_norm {
            apply_cmvn(&mut out_tensor, self.config.cepstral_mean_norm, self.config.cepstral_var_norm)?;
        }

        Ok(out_tensor)
    }
}

/// Applies Cepstral Mean and Variance Normalization (CMVN) in-place across the frame axis.
pub fn apply_cmvn(features: &mut Tensor, norm_mean: bool, norm_var: bool) -> BrainResult<()> {
    if features.ndim() != 2 {
        return Err(BrainError::invalid_value("apply_cmvn requires 2D [features, frames] tensor"));
    }
    let n_features = features.shape()[0];
    let n_frames = features.shape()[1];
    if n_frames == 0 {
        return Ok(());
    }

    let data = features.data_mut();

    for feat in 0..n_features {
        let row_start = feat * n_frames;
        let row_end = row_start + n_frames;
        let row = &mut data[row_start..row_end];

        let mean = if norm_mean {
            let sum: f64 = row.iter().sum();
            sum / n_frames as f64
        } else {
            0.0
        };

        if norm_mean {
            for v in row.iter_mut() {
                *v -= mean;
            }
        }

        if norm_var {
            let sum_sq: f64 = row.iter().map(|&v| v * v).sum();
            let std = (sum_sq / n_frames as f64).sqrt().max(1e-10);
            for v in row.iter_mut() {
                *v /= std;
            }
        }
    }
    Ok(())
}

/// Standalone helper to compute MFCC features directly from a 1D audio slice.
///
/// # Examples
///
/// ```
/// use brain_audio::feature::mfcc;
/// use brain_audio::config::MFCCConfig;
/// let cfg = MFCCConfig::default();
/// let sig = vec![0.0; 1600];
/// let feat = mfcc(&sig, &cfg).unwrap();
/// assert_eq!(feat.shape()[0], 13);
/// ```
pub fn mfcc(signal: &[f64], config: &MFCCConfig) -> BrainResult<Tensor> {
    let processor = MFCCProcessor::new(config.clone())?;
    processor.compute_mfcc(signal)
}

/// Standalone helper to compute MFCC features from an AudioBuffer.
pub fn compute_mfcc(audio: &crate::core::AudioBuffer, config: &MFCCConfig) -> BrainResult<Tensor> {
    let mono = audio.to_mono();
    mfcc(mono.as_slice(), config)
}

#[cfg(test)]
mod tests {
    use super::*;
}
