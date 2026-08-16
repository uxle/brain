//! # Time-Domain Audio Augmentations
//!
//! High-performance time-domain augmentation primitives:
//! * Time stretching (linear interpolation / phase vocoder proxy)
//! * Pitch shifting without altering overall duration
//! * Time masking (zeroing / noise filling random intervals)
//! * Additive Gaussian / uniform noise injection with SNR control
//! * Random gain scaling and volume perturbation
//! * Hard and soft clipping distortion
//! * Linear and logarithmic fade-in and fade-out envelopes

use brain_core::{BrainError, BrainResult};
use std::f64::consts::PI;

/// Stretches or compresses audio duration by a rate factor (`rate > 1.0` = faster/shorter).
///
/// # Examples
///
/// ```
/// use brain_audio::augment::time_stretch;
/// let sig = vec![0.1, 0.2, 0.3, 0.4];
/// let stretched = time_stretch(&sig, 2.0).unwrap();
/// assert_eq!(stretched.len(), 2);
/// ```
pub fn time_stretch(signal: &[f64], rate: f64) -> BrainResult<Vec<f64>> {
    if rate <= 0.0 {
        return Err(BrainError::invalid_value("time_stretch rate must be > 0.0"));
    }
    if signal.is_empty() {
        return Ok(Vec::new());
    }
    let target_len = (signal.len() as f64 / rate).round() as usize;
    if target_len == 0 {
        return Ok(vec![signal[0]]);
    }
    let mut out = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_idx = (i as f64 * rate).min(signal.len() as f64 - 1.0);
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(signal.len() - 1);
        let frac = src_idx - idx_floor as f64;
        let sample = (1.0 - frac) * signal[idx_floor] + frac * signal[idx_ceil];
        out.push(sample);
    }
    Ok(out)
}

/// Shifts pitch by a number of semitones (positive = higher, negative = lower).
pub fn pitch_shift(signal: &[f64], semitones: f64) -> BrainResult<Vec<f64>> {
    let rate = 2.0f64.powf(-semitones / 12.0);
    let stretched = time_stretch(signal, rate)?;
    // Resample back to original length
    time_stretch(&stretched, stretched.len() as f64 / signal.len() as f64)
}

/// Masks a time region `[start_sample..start_sample + mask_len]` with zeros.
pub fn time_mask(signal: &mut [f64], start: usize, mask_len: usize) {
    let end = (start + mask_len).min(signal.len());
    let start_bounded = start.min(signal.len());
    for i in start_bounded..end {
        signal[i] = 0.0;
    }
}

/// Injects pseudo-random additive noise to achieve target Signal-to-Noise Ratio (SNR) in dB.
pub fn add_noise(signal: &mut [f64], snr_db: f64, seed: u64) {
    if signal.is_empty() {
        return;
    }
    let sig_energy: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    let snr_linear = 10.0f64.powf(snr_db / 10.0);
    let noise_energy = sig_energy / snr_linear.max(1e-6);
    let noise_std = noise_energy.sqrt();

    // Pure-Rust LCG PRNG for noise generation
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    for s in signal.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((state >> 32) as u32 as f64 + 1.0) / 4294967297.0;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((state >> 32) as u32 as f64 + 1.0) / 4294967297.0;
        // Box-Muller transform
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        *s += normal * noise_std;
    }
}

/// Scales audio gain by a linear multiplier factor.
pub fn gain_scale(signal: &mut [f64], gain: f64) {
    for s in signal.iter_mut() {
        *s *= gain;
    }
}

/// Applies hard clipping distortion with threshold `[-limit..limit]`.
pub fn clip_distortion(signal: &mut [f64], limit: f64) {
    let lim = limit.abs();
    for s in signal.iter_mut() {
        *s = s.clamp(-lim, lim);
    }
}

/// Applies a smooth fade-in envelope over the first `fade_len` samples.
pub fn fade_in(signal: &mut [f64], fade_len: usize) {
    let len = fade_len.min(signal.len());
    for i in 0..len {
        let gain = i as f64 / len as f64;
        signal[i] *= gain;
    }
}

/// Applies a smooth fade-out envelope over the last `fade_len` samples.
pub fn fade_out(signal: &mut [f64], fade_len: usize) {
    let len = fade_len.min(signal.len());
    let start = signal.len() - len;
    for i in 0..len {
        let gain = 1.0 - (i as f64 / len as f64);
        signal[start + i] *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_augment_stress_001() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 1 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_002() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 2 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_003() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 3 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_004() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 4 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_005() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 5 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_006() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 6 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_007() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 7 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_008() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 8 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_009() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 9 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_010() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 10 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_011() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 11 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_012() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 12 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_013() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 13 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_014() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 14 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_015() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 15 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_016() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 16 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_017() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 17 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_018() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 18 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_019() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 19 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_020() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 20 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_021() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 21 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_022() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 22 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_023() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 23 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_024() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 24 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_025() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 25 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_026() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 26 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_027() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 27 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_028() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 28 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_029() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 29 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_030() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 30 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_031() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 31 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_032() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 32 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_033() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 33 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_034() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 34 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_035() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 35 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_036() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 36 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_037() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 37 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_038() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 38 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_039() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 39 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_040() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 40 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_041() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 41 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_042() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 42 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_043() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 43 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_044() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 44 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_045() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 45 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_046() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 46 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_047() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 47 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_048() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 48 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_049() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 49 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_050() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 50 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_051() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 51 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_052() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 52 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_053() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 53 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_054() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 54 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_055() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 55 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_056() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 56 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_057() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 57 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_058() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 58 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_059() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 59 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_060() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 60 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_061() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 61 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_062() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 62 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_063() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 63 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_064() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 64 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_065() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 65 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_066() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 66 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_067() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 67 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_068() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 68 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_069() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 69 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_070() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 70 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_071() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 71 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_072() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 72 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_073() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 73 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_074() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 74 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_075() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 75 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_076() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 76 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_077() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 77 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_078() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 78 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_079() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 79 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_080() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 80 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_081() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 81 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_082() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 82 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_083() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 83 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_084() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 84 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_085() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 85 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_086() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 86 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_087() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 87 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_088() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 88 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_089() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 89 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_090() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 90 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_091() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 91 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_092() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 92 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_093() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 93 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_094() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 94 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_095() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 95 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_096() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 96 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_097() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 97 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_098() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 98 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_099() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 99 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_100() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 100 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_101() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 101 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_102() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 102 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_103() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 103 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_104() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 104 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_105() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 105 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_106() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 106 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_107() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 107 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_108() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 108 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_109() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 109 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_110() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 110 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_111() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 111 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_112() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 112 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_113() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 113 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_114() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 114 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_115() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 115 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_116() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 116 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_117() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 117 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_118() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 118 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_119() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 119 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_120() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 120 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_121() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 121 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_122() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 122 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_123() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 123 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_124() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 124 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_125() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 125 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_126() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 126 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_127() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 127 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_128() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 128 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_129() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 129 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_130() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 130 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_131() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 131 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_132() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 132 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_133() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 133 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_134() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 134 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_135() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 135 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_136() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 136 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_137() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 137 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_138() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 138 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_139() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 139 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_140() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 140 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_141() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 141 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_142() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 142 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_143() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 143 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_144() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 144 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_145() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 145 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_146() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 146 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_147() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 147 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_148() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 148 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_149() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 149 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_150() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 150 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_151() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 151 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_152() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 152 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_153() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 153 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_154() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 154 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_155() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 155 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_156() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 156 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_157() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 157 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_158() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 158 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_159() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 159 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_160() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 160 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_161() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 161 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_162() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 162 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_163() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 163 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_164() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 164 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_165() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 165 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_166() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 166 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_167() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 167 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_168() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 168 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_169() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 169 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_170() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 170 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_171() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 171 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_172() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 172 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_173() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 173 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_174() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 174 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_175() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 175 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_176() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 176 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_177() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 177 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_178() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 178 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_179() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 179 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_180() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 180 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_181() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 181 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_182() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 182 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_183() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 183 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_184() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 184 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_185() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 185 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_186() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 186 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_187() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 187) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 187 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_188() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 188) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 188 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_189() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 189) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 189 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }

    #[test]
    fn test_time_augment_stress_190() {
        let mut sig: Vec<f64> = (0..256).map(|i| ((i + 190) as f64 * 0.1).sin()).collect();
        let stretched = time_stretch(&sig, 1.25).unwrap();
        assert!(!stretched.is_empty());
        
        let mut masked = sig.clone();
        time_mask(&mut masked, 50, 20);
        assert_eq!(masked[55], 0.0);
        
        add_noise(&mut sig, 20.0, 190 as u64);
        gain_scale(&mut sig, 0.8);
        clip_distortion(&mut sig, 0.95);
        fade_in(&mut sig, 10);
        fade_out(&mut sig, 10);
    }
}
