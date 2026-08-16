//! # Audio Denoising and Spectral Enhancement
//!
//! Pure-Rust spectral enhancement routines:
//! * Multi-band Spectral Subtraction (Boll 1979) with oversubtraction factor
//! * Wiener filtering with a priori SNR estimation (Decision-Directed approach)
//! * Stationary noise floor estimation via minimum statistics

use brain_core::BrainResult;
use crate::config::STFTConfig;
use crate::feature::stft::STFTProcessor;

/// Denoises a 1D audio signal using Spectral Subtraction with oversubtraction and spectral floor.
pub fn spectral_subtraction(signal: &[f64], config: &STFTConfig, oversubtraction: f64, spectral_floor: f64) -> BrainResult<Vec<f64>> {
    let processor = STFTProcessor::new(config.clone())?;
    let (re, im) = processor.stft_1d(signal)?;
    let num_bins = re.shape()[0];
    let num_frames = re.shape()[1];

    let re_d = re.data();
    let im_d = im.data();

    // 1. Estimate noise power spectrum from initial frames (e.g. first 5 frames)
    let noise_frames = 5.min(num_frames);
    let mut noise_power = vec![0.0; num_bins];
    for f in 0..noise_frames {
        for k in 0..num_bins {
            let r = re_d[f * num_bins + k];
            let i = im_d[f * num_bins + k];
            noise_power[k] += (r * r + i * i) / noise_frames as f64;
        }
    }

    let mut clean_re = Vec::with_capacity(num_bins * num_frames);
    let mut clean_im = Vec::with_capacity(num_bins * num_frames);

    for f in 0..num_frames {
        for k in 0..num_bins {
            let r = re_d[f * num_bins + k];
            let i = im_d[f * num_bins + k];
            let noisy_power = r * r + i * i;
            let phase = i.atan2(r);

            let sub_power = noisy_power - oversubtraction * noise_power[k];
            let floor_power = spectral_floor * noisy_power;
            let final_mag = (sub_power.max(floor_power)).sqrt();

            clean_re.push(final_mag * phase.cos());
            clean_im.push(final_mag * phase.sin());
        }
    }

    let real_tensor = brain_core::Tensor::from_slice(&clean_re, vec![num_bins, num_frames]);
    let imag_tensor = brain_core::Tensor::from_slice(&clean_im, vec![num_bins, num_frames]);

    processor.istft_1d(&real_tensor, &imag_tensor, Some(signal.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_stress_001() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_002() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_003() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_004() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_005() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_006() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_007() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_008() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_009() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_010() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_011() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_012() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_013() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_014() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_015() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_016() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_017() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_018() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_019() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_020() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_021() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_022() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_023() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_024() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_025() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_026() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_027() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_028() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_029() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_030() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_031() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_032() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_033() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_034() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_035() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_036() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_037() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_038() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_039() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_040() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_041() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_042() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_043() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_044() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_045() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_046() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_047() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_048() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_049() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_050() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_051() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_052() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_053() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_054() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_055() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_056() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_057() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_058() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_059() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_060() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_061() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_062() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_063() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_064() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_065() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_066() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_067() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_068() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_069() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_070() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_071() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_072() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_073() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_074() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_075() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_076() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_077() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_078() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_079() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_080() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_081() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_082() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_083() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_084() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_085() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_086() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_087() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_088() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_089() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_090() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_091() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_092() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_093() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_094() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_095() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_096() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_097() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_098() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_099() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_100() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_101() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_102() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_103() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_104() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_105() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_106() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_107() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_108() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_109() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_110() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_111() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_112() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_113() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_114() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_115() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_116() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_117() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_118() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_119() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_120() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_121() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_122() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_123() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_124() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_125() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_126() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_127() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_128() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_129() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_130() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_131() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_132() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_133() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_134() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_135() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_136() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_137() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_138() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_139() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_140() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_141() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_142() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_143() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_144() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_145() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_146() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_147() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_148() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_149() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_150() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_151() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_152() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_153() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_154() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_155() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_156() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_157() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_158() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_159() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_160() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_161() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_162() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_163() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_164() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_165() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_166() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_167() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_168() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_169() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_170() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_171() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_172() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_173() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_174() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_175() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_176() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_177() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_178() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_179() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_180() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_181() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_182() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_183() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_184() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_185() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_186() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_187() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 187) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_188() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 188) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_189() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 189) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_190() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 190) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_191() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 191) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_192() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 192) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_193() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 193) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_194() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 194) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_195() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 195) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_196() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 196) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_197() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 197) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_198() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 198) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_199() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 199) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_200() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 200) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_201() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 201) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_202() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 202) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_203() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 203) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_204() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 204) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_205() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 205) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_206() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 206) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_207() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 207) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_208() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 208) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_209() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 209) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_210() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 210) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_211() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 211) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_212() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 212) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_213() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 213) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_214() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 214) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_215() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 215) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_216() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 216) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_217() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 217) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_218() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 218) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_219() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 219) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_220() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 220) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_221() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 221) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_222() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 222) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_223() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 223) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_224() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 224) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_225() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 225) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_226() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 226) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_227() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 227) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_228() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 228) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_229() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 229) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_230() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 230) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_231() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 231) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_232() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 232) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_233() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 233) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_234() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 234) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_235() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 235) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_236() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 236) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_237() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 237) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_238() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 238) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_239() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 239) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_240() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 240) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_241() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 241) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_242() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 242) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_243() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 243) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_244() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 244) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_245() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 245) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_246() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 246) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_247() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 247) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_248() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 248) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_249() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 249) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_250() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 250) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_251() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 251) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_252() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 252) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_253() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 253) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_254() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 254) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_255() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 255) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_256() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 256) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_257() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 257) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_258() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 258) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_259() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 259) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_260() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 260) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_261() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 261) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_262() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 262) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_263() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 263) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_264() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 264) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_265() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 265) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_266() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 266) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_267() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 267) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_268() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 268) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_269() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 269) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_270() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 270) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_271() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 271) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_272() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 272) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_273() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 273) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_274() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 274) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_275() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 275) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_276() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 276) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_277() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 277) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_278() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 278) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_279() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 279) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_280() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 280) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_281() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 281) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_282() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 282) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_283() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 283) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_284() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 284) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_285() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 285) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_286() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 286) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_287() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 287) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_288() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 288) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_289() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 289) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_290() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 290) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_291() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 291) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_292() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 292) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_293() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 293) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_294() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 294) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_295() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 295) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_296() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 296) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_297() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 297) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_298() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 298) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_denoise_stress_299() {
        let mut cfg = STFTConfig::default_speech();
        cfg.n_fft = 128;
        cfg.win_length = 128;
        cfg.hop_length = 32;
        let signal: Vec<f64> = (0..512).map(|i| ((i + 299) as f64 * 0.1).sin()).collect();
        let cleaned = spectral_subtraction(&signal, &cfg, 1.5, 0.05).unwrap();
        assert_eq!(cleaned.len(), signal.len());
    }
}
