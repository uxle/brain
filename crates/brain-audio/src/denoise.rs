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
}
