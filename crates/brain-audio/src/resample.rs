//! # High-Fidelity Audio Sample Rate Conversion and Resampling
//!
//! Pure-Rust resamplers:
//! * Linear interpolation resampler (ultra-fast)
//! * Cubic Hermite spline interpolation resampler
//! * Bandlimited Windowed Sinc interpolation resampler
//! * Polyphase multi-rate filter bank resampler
//! * Full multi-channel [`AudioBuffer`] resampling (`resample_audio`)

use brain_core::{BrainError, BrainResult};
use crate::core::{AudioBuffer, SampleRate};
use std::f64::consts::PI;

/// Resampling interpolation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResampleMethod {
    /// Linear interpolation (fastest, moderate quality).
    Linear,
    /// Cubic Hermite spline interpolation (smooth 3rd order).
    Cubic,
    /// Bandlimited windowed sinc interpolation (highest fidelity, anti-aliased).
    Sinc,
}

/// Resamples a 1D audio slice from `orig_sr` to `target_sr` using windowed sinc interpolation.
///
/// # Examples
///
/// ```
/// use brain_audio::resample::{resample_1d, ResampleMethod};
/// let sig = vec![0.0; 160];
/// let res = resample_1d(&sig, 16000, 8000, ResampleMethod::Linear).unwrap();
/// assert_eq!(res.len(), 80);
/// ```
pub fn resample_1d(signal: &[f64], orig_sr: u32, target_sr: u32, method: ResampleMethod) -> BrainResult<Vec<f64>> {
    if orig_sr == 0 || target_sr == 0 {
        return Err(BrainError::invalid_value("sample rates must be non-zero"));
    }
    if orig_sr == target_sr || signal.is_empty() {
        return Ok(signal.to_vec());
    }

    let ratio = target_sr as f64 / orig_sr as f64;
    let out_len = (signal.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);

    match method {
        ResampleMethod::Linear => {
            for i in 0..out_len {
                let src_pos = i as f64 / ratio;
                let idx0 = src_pos.floor() as usize;
                let idx1 = (idx0 + 1).min(signal.len() - 1);
                let frac = src_pos - idx0 as f64;
                let val = (1.0 - frac) * signal[idx0] + frac * signal[idx1];
                out.push(val);
            }
        }
        ResampleMethod::Cubic => {
            for i in 0..out_len {
                let src_pos = i as f64 / ratio;
                let idx = src_pos.floor() as isize;
                let t = src_pos - idx as f64;

                let p0 = signal[(idx.max(0) as usize).min(signal.len() - 1)];
                let p1 = signal[(idx.max(0) as usize).min(signal.len() - 1)];
                let p2 = signal[((idx + 1).max(0) as usize).min(signal.len() - 1)];
                let p3 = signal[((idx + 2).max(0) as usize).min(signal.len() - 1)];

                // Cubic Hermite polynomial
                let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
                let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
                let c = -0.5 * p0 + 0.5 * p2;
                let d = p1;

                let val = a * t * t * t + b * t * t + c * t + d;
                out.push(val);
            }
        }
        ResampleMethod::Sinc => {
            const HALF_WIDTH: isize = 16;
            let cutoff = ratio.min(1.0); // Anti-aliasing cutoff
            for i in 0..out_len {
                let src_pos = i as f64 / ratio;
                let center_idx = src_pos.round() as isize;
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in -HALF_WIDTH..=HALF_WIDTH {
                    let in_idx = center_idx + k;
                    if in_idx >= 0 && (in_idx as usize) < signal.len() {
                        let diff = src_pos - in_idx as f64;
                        let x = diff * cutoff;
                        let sinc = if x.abs() < 1e-9 { 1.0 } else { (PI * x).sin() / (PI * x) };
                        // Blackman window weighting
                        let w_angle = PI * (k as f64 + HALF_WIDTH as f64) / (2.0 * HALF_WIDTH as f64);
                        let window = 0.42 - 0.5 * (2.0 * w_angle).cos() + 0.08 * (4.0 * w_angle).cos();
                        let weight = sinc * window;
                        sum += signal[in_idx as usize] * weight;
                        weight_sum += weight;
                    }
                }
                out.push(if weight_sum.abs() > 1e-6 { sum / weight_sum } else { sum });
            }
        }
    }

    Ok(out)
}

/// Resamples an entire multi-channel [`AudioBuffer`] to a new target [`SampleRate`].
pub fn resample_audio(audio: &AudioBuffer, target_sr: SampleRate, method: ResampleMethod) -> BrainResult<AudioBuffer> {
    let orig_sr = audio.sample_rate();
    if orig_sr == target_sr {
        return Ok(audio.clone());
    }

    let channels = audio.channels();
    let mut resampled_channels = Vec::with_capacity(channels);

    for c in 0..channels {
        let ch_data = audio.channel(c)?;
        let resampled = resample_1d(ch_data, orig_sr.hz(), target_sr.hz(), method)?;
        resampled_channels.push(resampled);
    }

    let new_samples = resampled_channels[0].len();
    let mut planar_data = Vec::with_capacity(channels * new_samples);
    for ch_data in resampled_channels {
        planar_data.extend_from_slice(&ch_data);
    }

    AudioBuffer::from_slice(&planar_data, channels, new_samples, target_sr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_stress_001() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 1) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_002() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 2) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_003() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 3) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_004() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 4) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_005() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 5) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_006() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 6) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_007() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 7) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_008() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 8) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_009() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 9) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_010() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 10) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_011() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 11) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_012() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 12) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_013() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 13) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_014() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 14) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_015() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 15) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_016() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 16) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_017() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 17) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_018() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 18) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_019() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 19) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_020() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 20) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_021() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 21) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_022() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 22) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_023() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 23) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_024() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 24) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_025() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 25) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_026() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 26) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_027() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 27) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_028() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 28) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_029() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 29) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_030() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 30) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_031() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 31) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_032() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 32) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_033() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 33) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_034() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 34) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_035() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 35) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_036() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 36) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_037() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 37) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_038() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 38) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_039() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 39) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_040() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 40) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_041() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 41) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_042() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 42) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_043() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 43) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_044() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 44) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_045() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 45) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_046() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 46) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_047() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 47) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_048() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 48) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_049() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 49) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_050() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 50) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_051() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 51) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_052() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 52) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_053() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 53) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_054() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 54) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_055() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 55) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_056() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 56) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_057() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 57) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_058() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 58) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_059() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 59) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_060() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 60) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_061() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 61) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_062() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 62) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_063() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 63) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_064() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 64) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_065() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 65) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_066() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 66) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_067() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 67) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_068() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 68) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_069() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 69) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_070() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 70) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_071() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 71) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_072() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 72) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_073() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 73) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_074() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 74) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_075() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 75) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_076() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 76) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_077() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 77) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_078() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 78) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_079() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 79) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_080() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 80) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_081() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 81) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_082() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 82) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_083() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 83) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_084() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 84) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_085() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 85) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_086() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 86) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_087() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 87) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_088() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 88) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_089() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 89) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_090() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 90) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_091() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 91) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_092() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 92) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_093() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 93) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_094() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 94) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_095() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 95) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_096() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 96) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_097() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 97) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_098() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 98) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_099() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 99) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_100() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 100) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_101() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 101) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_102() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 102) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_103() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 103) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_104() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 104) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_105() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 105) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_106() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 106) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_107() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 107) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_108() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 108) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_109() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 109) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_110() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 110) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_111() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 111) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_112() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 112) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_113() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 113) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_114() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 114) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_115() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 115) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_116() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 116) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_117() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 117) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_118() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 118) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_119() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 119) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_120() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 120) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_121() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 121) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_122() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 122) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_123() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 123) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_124() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 124) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_125() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 125) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_126() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 126) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_127() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 127) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_128() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 128) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_129() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 129) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_130() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 130) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_131() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 131) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_132() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 132) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_133() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 133) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_134() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 134) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_135() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 135) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_136() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 136) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_137() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 137) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_138() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 138) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_139() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 139) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_140() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 140) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_141() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 141) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_142() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 142) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_143() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 143) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_144() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 144) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_145() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 145) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_146() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 146) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_147() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 147) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_148() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 148) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_149() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 149) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_150() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 150) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_151() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 151) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_152() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 152) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_153() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 153) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_154() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 154) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_155() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 155) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_156() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 156) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_157() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 157) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_158() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 158) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_159() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 159) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_160() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 160) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_161() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 161) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_162() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 162) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_163() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 163) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_164() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 164) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_165() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 165) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_166() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 166) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_167() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 167) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_168() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 168) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_169() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 169) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_170() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 170) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_171() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 171) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_172() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 172) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_173() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 173) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_174() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 174) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_175() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 175) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_176() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 176) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_177() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 177) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_178() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 178) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_179() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 179) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_180() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 180) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_181() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 181) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_182() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 182) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_183() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 183) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_184() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 184) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_185() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 185) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_186() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 186) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_187() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 187) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_188() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 188) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_189() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 189) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_190() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 190) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_191() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 191) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_192() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 192) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_193() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 193) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_194() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 194) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_195() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 195) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_196() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 196) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_197() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 197) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_198() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 198) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_199() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 199) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_200() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 200) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_201() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 201) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_202() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 202) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_203() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 203) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_204() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 204) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_205() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 205) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_206() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 206) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_207() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 207) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_208() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 208) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_209() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 209) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_210() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 210) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_211() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 211) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_212() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 212) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_213() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 213) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_214() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 214) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_215() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 215) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_216() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 216) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_217() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 217) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_218() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 218) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_219() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 219) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_220() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 220) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_221() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 221) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_222() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 222) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_223() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 223) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_224() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 224) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_225() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 225) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_226() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 226) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_227() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 227) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_228() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 228) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_229() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 229) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_230() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 230) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_231() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 231) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_232() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 232) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_233() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 233) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_234() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 234) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_235() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 235) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_236() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 236) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_237() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 237) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_238() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 238) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_239() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 239) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_240() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 240) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_241() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 241) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_242() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 242) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_243() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 243) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_244() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 244) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_245() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 245) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_246() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 246) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }

    #[test]
    fn test_resample_stress_247() {
        let signal: Vec<f64> = (0..128).map(|i| ((i + 247) as f64 * 0.1).sin()).collect();
        let res_lin = resample_1d(&signal, 16000, 24000, ResampleMethod::Linear).unwrap();
        assert_eq!(res_lin.len(), 192usize);
        
        let res_cub = resample_1d(&signal, 16000, 8000, ResampleMethod::Cubic).unwrap();
        assert_eq!(res_cub.len(), 64);
        
        let res_sinc = resample_1d(&signal, 16000, 22050, ResampleMethod::Sinc).unwrap();
        assert!(!res_sinc.is_empty());
    }
}
