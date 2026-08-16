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

    #[test]
    fn test_mfcc_stress_001() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 1) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_002() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 2) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_003() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 3) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_004() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 4) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_005() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 5) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_006() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 6) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_007() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 7) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_008() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 8) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_009() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 9) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_010() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 10) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_011() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 11) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_012() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 12) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_013() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 13) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_014() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 14) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_015() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 15) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_016() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 16) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_017() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 17) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_018() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 18) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_019() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 19) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_020() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 20) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_021() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 21) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_022() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 22) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_023() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 23) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_024() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 24) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_025() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 25) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_026() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 26) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_027() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 27) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_028() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 28) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_029() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 29) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_030() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 30) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_031() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 31) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_032() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 32) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_033() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 33) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_034() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 34) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_035() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 35) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_036() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 36) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_037() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 37) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_038() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 38) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_039() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 39) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_040() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 40) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_041() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 41) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_042() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 42) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_043() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 43) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_044() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 44) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_045() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 45) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_046() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 46) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_047() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 47) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_048() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 48) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_049() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 49) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_050() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 50) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_051() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 51) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_052() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 52) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_053() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 53) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_054() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 54) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_055() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 55) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_056() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 56) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_057() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 57) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_058() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 58) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_059() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 59) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_060() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 60) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_061() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 61) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_062() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 62) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_063() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 63) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_064() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 64) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_065() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 65) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_066() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 66) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_067() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 67) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_068() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 68) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_069() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 69) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_070() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 70) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_071() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 71) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_072() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 72) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_073() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 73) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_074() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 74) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_075() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 75) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_076() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 76) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_077() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 77) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_078() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 78) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_079() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 79) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_080() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 80) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_081() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 81) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_082() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 82) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_083() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 83) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_084() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 84) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_085() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 85) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_086() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 86) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_087() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 87) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_088() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 88) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_089() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 89) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_090() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 90) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_091() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 91) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_092() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 92) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_093() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 93) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_094() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 94) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_095() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 95) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_096() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 96) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_097() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 97) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_098() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 98) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_099() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 99) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_100() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 100) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_101() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 101) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_102() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 102) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_103() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 103) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_104() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 104) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_105() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 105) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_106() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 106) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_107() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 107) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_108() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 108) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_109() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 109) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_110() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 110) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_111() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 111) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_112() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 112) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_113() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 113) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_114() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 114) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_115() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 115) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_116() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 116) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_117() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 117) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_118() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 118) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_119() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 119) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_120() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 120) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_121() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 121) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_122() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 122) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_123() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 123) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_124() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 124) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_125() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 125) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_126() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 126) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_127() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 127) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_128() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 128) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_129() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 129) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_130() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 130) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_131() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 131) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_132() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 132) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_133() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 133) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_134() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 134) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_135() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 135) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_136() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 136) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_137() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 137) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_138() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 138) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_139() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 139) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_140() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 140) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_141() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 141) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_142() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 142) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_143() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 143) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_144() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 144) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_145() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 145) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_146() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 146) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_147() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 147) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_148() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 148) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_149() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 149) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_150() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 150) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_151() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 151) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_152() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 152) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_153() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 153) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_154() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 154) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_155() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 155) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_156() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 156) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_157() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 157) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_158() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 158) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_159() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 159) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_160() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 160) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_161() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 161) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_162() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 162) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_163() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 163) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_164() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 164) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_165() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 165) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_166() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 166) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_167() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 167) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_168() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 168) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_169() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 169) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_170() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 170) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_171() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 171) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_172() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 172) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_173() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 173) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_174() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 174) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_175() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 175) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_176() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 176) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_177() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 177) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_178() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 178) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_179() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 179) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_180() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 180) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_181() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 181) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_182() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 182) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_183() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 183) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_184() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 184) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_185() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 185) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_186() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 186) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_187() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 187) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_188() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 188) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_189() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 189) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_190() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 190) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_191() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 191) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_192() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 192) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_193() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 193) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_194() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 194) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_195() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 195) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_196() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 196) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_197() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 197) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_198() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 198) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_199() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 199) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_200() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 200) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_201() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 201) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_202() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 202) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_203() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 203) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_204() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 204) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_205() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 205) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_206() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 206) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_207() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 207) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_208() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 208) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_209() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 209) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_210() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 210) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_211() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 211) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_212() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 212) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_213() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 213) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_214() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 214) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_215() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 215) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_216() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 216) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_217() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 217) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_218() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 218) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_219() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 219) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_220() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 220) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_221() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 221) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_222() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 222) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_223() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 223) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_224() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 224) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_225() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 225) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_226() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 226) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_227() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 227) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_228() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 228) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }

    #[test]
    fn test_mfcc_stress_229() {
        let cfg = MFCCConfig::default();
        let proc = MFCCProcessor::new(cfg.clone()).unwrap();
        let signal: Vec<f64> = (0..1600).map(|i| ((i + 229) as f64 * 0.05).sin()).collect();
        let res = proc.compute_mfcc(&signal).unwrap();
        assert_eq!(res.shape()[0], cfg.n_mfcc);
        assert!(res.shape()[1] > 0);
        
        let mut res_cmvn = res.clone();
        apply_cmvn(&mut res_cmvn, true, true).unwrap();
        assert_eq!(res_cmvn.shape(), res.shape());
    }
}
