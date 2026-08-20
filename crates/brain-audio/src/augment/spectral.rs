//! # Spectral-Domain Audio Augmentation (SpecAugment)
//!
//! Implementations of SpecAugment (Park et al. 2019):
//! * Frequency masking (zeroing horizontal bands across all time frames)
//! * Time masking (zeroing vertical bands across all frequency channels)
//! * Spectrogram Cutout and rectangular patch masking
//! * Spectrogram Mixup and linear feature interpolation

use brain_core::{BrainError, BrainResult, Tensor};

/// Applies SpecAugment (time and frequency masking) to a 2D spectrogram tensor `[channels, frames]`.
pub fn spec_augment(
    spectrogram: &mut Tensor,
    freq_mask_param: usize,
    num_freq_masks: usize,
    time_mask_param: usize,
    num_time_masks: usize,
    seed: u64,
) -> BrainResult<()> {
    if spectrogram.ndim() != 2 {
        return Err(BrainError::invalid_value(
            "spec_augment requires 2D [channels, frames] tensor",
        ));
    }
    let n_freq = spectrogram.shape()[0];
    let n_time = spectrogram.shape()[1];

    let mut state = seed.wrapping_add(0xdeadbeef);

    // Apply frequency masks
    for _ in 0..num_freq_masks {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f_len = (state >> 32) as usize % freq_mask_param.min(n_freq);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f_start = (state >> 32) as usize % (n_freq - f_len).max(1);
        frequency_mask(spectrogram, f_start, f_len)?;
    }

    // Apply time masks
    for _ in 0..num_time_masks {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t_len = (state >> 32) as usize % time_mask_param.min(n_time);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t_start = (state >> 32) as usize % (n_time - t_len).max(1);
        time_mask_spec(spectrogram, t_start, t_len)?;
    }

    Ok(())
}

/// Zeroes out a contiguous frequency band `[f_start..f_start + f_len]` across all time frames.
pub fn frequency_mask(spectrogram: &mut Tensor, f_start: usize, f_len: usize) -> BrainResult<()> {
    let n_freq = spectrogram.shape()[0];
    let n_time = spectrogram.shape()[1];
    let f_end = (f_start + f_len).min(n_freq);
    let data = spectrogram.data_mut();

    for f in f_start..f_end {
        for t in 0..n_time {
            data[f * n_time + t] = 0.0;
        }
    }
    Ok(())
}

/// Zeroes out a contiguous time band `[t_start..t_start + t_len]` across all frequency bins.
pub fn time_mask_spec(spectrogram: &mut Tensor, t_start: usize, t_len: usize) -> BrainResult<()> {
    let n_freq = spectrogram.shape()[0];
    let n_time = spectrogram.shape()[1];
    let t_end = (t_start + t_len).min(n_time);
    let data = spectrogram.data_mut();

    for f in 0..n_freq {
        for t in t_start..t_end {
            data[f * n_time + t] = 0.0;
        }
    }
    Ok(())
}

/// Masks a rectangular cutout region `[f_start..f_end, t_start..t_end]` with zeros.
pub fn spec_cutout(
    spectrogram: &mut Tensor,
    f_start: usize,
    f_len: usize,
    t_start: usize,
    t_len: usize,
) -> BrainResult<()> {
    let n_freq = spectrogram.shape()[0];
    let n_time = spectrogram.shape()[1];
    let f_end = (f_start + f_len).min(n_freq);
    let t_end = (t_start + t_len).min(n_time);
    let data = spectrogram.data_mut();

    for f in f_start..f_end {
        for t in t_start..t_end {
            data[f * n_time + t] = 0.0;
        }
    }
    Ok(())
}

/// Computes linear Mixup interpolation of two spectrograms: `alpha * spec1 + (1 - alpha) * spec2`.
pub fn spec_mixup(spec1: &Tensor, spec2: &Tensor, alpha: f64) -> BrainResult<Tensor> {
    if spec1.shape() != spec2.shape() {
        return Err(BrainError::shape_mismatch(
            format!("{:?}", spec1.shape()),
            format!("{:?}", spec2.shape()),
            "spec_mixup",
        ));
    }
    let a = alpha.clamp(0.0, 1.0);
    let b = 1.0 - a;
    let d1 = spec1.data();
    let d2 = spec2.data();
    let mixed: Vec<f64> = d1
        .iter()
        .zip(d2.iter())
        .map(|(&x, &y)| a * x + b * y)
        .collect();
    Ok(Tensor::from_slice(&mixed, spec1.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
}
