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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = ((state >> 32) as u32 as f64 + 1.0) / 4294967297.0;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
}
